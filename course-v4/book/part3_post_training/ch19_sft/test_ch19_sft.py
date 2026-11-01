"""CPU 快速机制测试；真实 tokenizer 用例在没有本地权重时 skip。"""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from miniocr.data.sft import encode_sft, collate_sft
from miniocr.models.qwen3 import find_local_qwen3
from miniocr.train.sft import causal_loss, train_sft, mean_loss
from .tour import ByteTokenizer, TinyLM, final_manifest, main


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_answer_boundary_and_eos():
    tok = ByteTokenizer()
    ex = encode_sft(tok, 'x_2', 'x^2')
    p = ex.prompt_length
    assert ex.labels[:p] == (-100,) * p
    assert ex.labels[p:] == tuple(tok.encode('x^2') + [tok.eos_token_id])
    assert ex.input_ids[p:] == ex.labels[p:]
    assert encode_sft(tok, 'x_2', 'x^2', max_length=len(ex.input_ids)) == ex
    assert encode_sft(tok, 'x_2', 'x^2', max_length=len(ex.input_ids)-1) is None
    with pytest.raises(ValueError, match='非空'):
        encode_sft(tok, 'x', '')


def test_padding_when_pad_equals_eos_and_alignment():
    tok = ByteTokenizer()
    examples = [encode_sft(tok, 'x', 'a'), encode_sft(tok, 'long', 'abcd')]
    batch = collate_sft(examples, tok.eos_token_id, multiple=8)
    assert batch['input_ids'].shape[1] % 8 == 0
    for i, e in enumerate(examples):
        n = len(e.input_ids)
        assert batch['labels'][i, n-1] == tok.eos_token_id
        assert batch['labels'][i, n:].eq(-100).all()
        assert batch['attention_mask'][i, :n].eq(1).all()
        assert batch['attention_mask'][i, n:].eq(0).all()
    with pytest.raises(ValueError):
        collate_sft([replace(examples[0], labels=(-100,)*len(examples[0].input_ids))], 0)


def test_faults_have_precise_semantics():
    tok = ByteTokenizer()
    examples = {v: encode_sft(tok, 'x_2', 'x^2', variant=v) for v in ('correct', 'shift', 'prompt', 'no_eos')}
    correct = examples['correct']
    assert examples['shift'].labels == correct.labels[1:] + (-100,)
    assert examples['prompt'].labels == correct.input_ids
    assert examples['no_eos'].input_ids == correct.input_ids[:-1]
    assert examples['no_eos'].labels == correct.labels[:-1]


def test_causal_loss_independent_oracle_and_gradient_mask():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7, requires_grad=True)
    labels = torch.tensor([[-100, -100, 3, 2, 1], [-100, 4, 1, -100, -100]])
    actual = causal_loss(logits, labels)
    expected = torch.stack([-F.log_softmax(logits[b, t-1], -1)[labels[b, t]]
                            for b in range(2) for t in range(1, 5) if labels[b, t] != -100]).mean()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert logits.grad[0, 0].eq(0).all()
    assert logits.grad[:, -1].eq(0).all()
    assert logits.grad[0, 1].abs().sum() > 0
    with pytest.raises(ValueError):
        causal_loss(logits, torch.full_like(labels, -100))


def test_hf_causal_lm_loss_uses_same_single_shift():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                      num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                      head_dim=8, max_position_embeddings=32, attention_dropout=0.)
    model = Qwen3ForCausalLM(cfg).eval()
    ids = torch.tensor([[3, 4, 5, 6, 1], [7, 8, 1, 0, 0]])
    labels = torch.tensor([[-100, -100, 5, 6, 1], [-100, 8, 1, -100, -100]])
    output = model(input_ids=ids, attention_mask=ids.ne(0), labels=labels, use_cache=False)
    torch.testing.assert_close(causal_loss(output.logits, labels), output.loss)


def test_token_weighted_accumulation_matches_one_batch():
    tok = ByteTokenizer()
    data = [encode_sft(tok, 'x', 'a'), encode_sft(tok, 'y', 'long')]
    torch.manual_seed(1)
    model = TinyLM()
    other = TinyLM()
    other.load_state_dict(model.state_dict())
    a = train_sft(model, data, 0, steps=1, batch_size=2, lr=.001)
    b = train_sft(other, data, 0, steps=1, batch_size=1, accumulation=2, lr=.001)
    assert a[0]['supervised_tokens'] == b[0]['supervised_tokens']
    assert a[0]['loss'] == pytest.approx(b[0]['loss'], abs=1e-5)
    for x, y in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(x, y, atol=3e-6, rtol=1e-4)


def test_tiny_training_reduces_correct_answer_nll():
    tok = ByteTokenizer()
    data = [encode_sft(tok, 'x_2', 'x^2'), encode_sft(tok, 'y_2', 'y^2')]
    torch.manual_seed(1)
    model = TinyLM()
    before = mean_loss(model, data, 0)['loss']
    train_sft(model, data, 0, steps=12, batch_size=2, lr=.01)
    after = mean_loss(model, data, 0)['loss']
    assert after < before * .5


def test_final_test_guard_before_claim_and_no_repeat(tmp_path):
    checkpoint = tmp_path/'training'/'best'
    checkpoint.mkdir(parents=True)
    (checkpoint.parent/'selected.json').write_text('{"step": 10}')
    args = SimpleNamespace(final=False, out=tmp_path/'eval', checkpoint=checkpoint)
    with pytest.raises(ValueError, match='冻结'):
        final_manifest(args)
    assert not args.out.exists()
    args.final = True
    final_manifest(args)
    with pytest.raises(FileExistsError):
        final_manifest(args)


@pytest.mark.skipif(find_local_qwen3() is None, reason='缺本地 Qwen3 权重')
def test_real_qwen_template_boundary_and_no_duplicate_specials():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(find_local_qwen3()), local_files_only=True)
    ex = encode_sft(tok, r'\frac{a}{b', r'\frac{a}{b}')
    assert ex is not None
    assert ex.input_ids[-1] == tok.eos_token_id
    assert tok.decode(ex.input_ids[ex.prompt_length:-1]) == r'\frac{a}{b}'
    assert '<think>\n\n</think>\n\n' in tok.decode(ex.input_ids[:ex.prompt_length])


def test_cli_defaults_are_small_and_parts_validate(capsys):
    main(['3', '--device', 'cpu'])
    assert '逐token手算' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(['9'])
