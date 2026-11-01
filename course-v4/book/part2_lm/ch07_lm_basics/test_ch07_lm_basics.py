"""第 7 章：无需缓存，手算对照、边界协议和模型行为测试。"""
import math

import pytest
import torch
import torch.nn.functional as F

from miniocr.models.ngram import (
    BigramLM, MLPLM, NGramLM, bits_per_byte_to_nats, bits_to_nats,
    make_examples, make_window_examples, nats_to_bits, nats_to_bits_per_byte, perplexity,
)
from miniocr.train.lm import bits_per_byte, get_batch


def test_shifted_examples():
    x, y = make_examples([9, 2, 3, 4, 8], 3)
    assert x.tolist() == [[9, 2, 3], [2, 3, 4]]
    assert y.tolist() == [[2, 3, 4], [3, 4, 8]]
    assert x.dtype == y.dtype == torch.long


def test_window_examples():
    x, y = make_window_examples([9, 9, 9, 2, 3, 8], 3)
    assert x.tolist() == [[9, 9, 9], [9, 9, 2], [9, 2, 3]]
    assert y.tolist() == [2, 3, 8]


def test_matches_existing_get_batch():
    ids = torch.arange(12)
    x, y = make_examples(ids, 4)
    bx, by = get_batch(ids, 20, 4, torch.Generator().manual_seed(0))
    assert torch.equal(x[bx[:, 0]], bx)
    assert torch.equal(y[bx[:, 0]], by)


@pytest.mark.parametrize("ids", [[], [1], [1, 2]])
def test_short_sequences(ids):
    x, y = make_examples(ids, 2)
    wx, wy = make_window_examples(ids, 2)
    assert x.shape == y.shape == wx.shape == (0, 2)
    assert wy.shape == (0,)


@pytest.mark.parametrize("function", [make_examples, make_window_examples])
def test_invalid_context(function):
    with pytest.raises(ValueError):
        function([1, 2], 0)


def test_bigram_hand_count_and_smoothing():
    # 普通 token 为 0、1，BOS=2、EOS=3。预测类别数为 3。
    model = BigramLM(4, 2, 3, k=1).fit([[0, 1], [0]])
    assert model.counts == {(2,): {0: 2}, (0,): {1: 1, 3: 1}, (1,): {3: 1}}
    assert model.totals == {(2,): 2, (0,): 2, (1,): 1}
    assert model.prob((2,), 0) == pytest.approx(3/5)
    assert model.prob((0,), 1) == pytest.approx(2/5)
    assert model.prob((1,), 3) == pytest.approx(2/4)
    for context in ((0,), (1,), (2,), (3,)):
        assert sum(model.prob(context, t) for t in range(4)) == pytest.approx(1)
        assert model.prob(context, 2) == 0
    expected = math.log(3/5) + math.log(2/5) + math.log(2/4)
    assert model.logprob([0, 1]) == pytest.approx(expected)
    # 两条长度不同的公式必须按 5 个目标加权，而非公式等权。
    assert model.nll_per_token([[0, 1], [0]]) == pytest.approx(
        -(expected + math.log(3/5) + math.log(2/5))/5)


def test_unseen_pair_finite_only_with_smoothing():
    model = BigramLM(4, 2, 3, k=0).fit([[0, 1]])
    assert model.prob((0,), 0) == 0
    assert model.logprob([0, 0]) == -math.inf
    assert model.nll_per_token([[0, 0]]) == math.inf
    model.k = 0.1
    assert model.prob((0,), 0) == pytest.approx(0.1/1.3)
    assert math.isfinite(model.nll_per_token([[0, 0]]))


def test_empty_formula_predicts_eos_and_fit_resets():
    model = BigramLM(4, 2, 3, k=0).fit([[]])
    assert model.logprob([]) == 0
    assert model.sample(5, seed=8) == [[]]*5
    model.fit([[0]])
    assert model.counts[(2,)] == {0: 1}
    assert (3,) not in model.counts


def test_sampling_reproducible_and_boundary_free():
    model = BigramLM(4, 2, 3, k=1).fit([[0, 1], [1, 0]])
    first = model.sample(20, seed=42, max_tokens=16)
    assert first == model.sample(20, seed=42, max_tokens=16)
    assert all(len(s) <= 16 and set(s) <= {0, 1} for s in first)


def test_no_eos_reaches_cap():
    model = BigramLM(4, 2, 3, k=0).fit([[0, 1, 0, 1]], add_eos=False)
    assert model.sample(3, seed=0, max_tokens=10) == [[0, 1]*5]*3


def test_perplexity():
    assert perplexity(0) == 1
    assert perplexity(math.log(4)) == pytest.approx(4)
    assert perplexity(math.inf) == math.inf


def test_information_units():
    assert nats_to_bits(math.log(2)) == pytest.approx(1)
    assert bits_to_nats(1) == pytest.approx(math.log(2))
    expected = 2/(3*math.log(2))
    assert nats_to_bits_per_byte(2, 3) == pytest.approx(expected)
    assert bits_per_byte(2, 100, 300) == pytest.approx(expected)
    assert bits_per_byte_to_nats(expected, 3) == pytest.approx(2)


def test_mlp_shape_and_initial_loss():
    torch.manual_seed(7)
    model = MLPLM(101, context=3, embed_dim=16, hidden_dim=32)
    x = torch.randint(101, (1024, 3))
    y = torch.randint(101, (1024,))
    output = model(x)
    assert output.shape == (1024, 101)
    loss = F.cross_entropy(output, y)
    assert abs(loss.item() - math.log(101)) < 0.15
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_mlp_forward_matches_manual_computation():
    torch.manual_seed(7)
    model = MLPLM(5, context=2, embed_dim=3, hidden_dim=4)
    x = torch.tensor([[0, 1], [1, 0], [2, 4]])
    with torch.no_grad():
        # 按位置分别取嵌入并拼接，独立核对展平顺序、激活函数和两个偏置。
        embedded = torch.cat([
            model.embedding.weight[x[:, 0]],
            model.embedding.weight[x[:, 1]],
        ], dim=1)
        hidden = embedded @ model.hidden.weight.T + model.hidden.bias
        expected = torch.tanh(hidden) @ model.output.weight.T + model.output.bias
        actual = model(x)
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(actual[0], actual[1])


def test_unigram_equivalence():
    model = NGramLM(4, 2, 3, n=1, k=0.5).fit([[0, 1], [0]])
    assert model.counts == {(): {0: 2, 1: 1, 3: 2}}
    for t, count in ((0, 2), (1, 1), (3, 2)):
        assert model.prob((), t) == pytest.approx((count+0.5)/(5+1.5))
    expected = -(2*math.log(2.5/6.5) + math.log(1.5/6.5) + 2*math.log(2.5/6.5))/5
    assert model.nll_per_token([[0, 1], [0]]) == pytest.approx(expected)


def test_generic_bigram_matches_exercise_model():
    seqs = [[0, 1], [], [1, 0, 1]]
    generic = NGramLM(4, 2, 3, n=2, k=0.1).fit(seqs)
    model = BigramLM(4, 2, 3, k=0.1).fit(seqs)
    assert model.counts == generic.counts
    for a in range(4):
        for b in range(4):
            assert model.prob((a,), b) == generic.prob((a,), b)
    assert model.sample(10, 4) == generic.sample(10, 4)


def test_ngram_padding_and_unseen_context():
    model = NGramLM(5, 3, 4, n=3, k=0.1).fit([[0, 1], [2]])
    assert list(model.events([0, 1])) == [((3, 3), 0), ((3, 0), 1), ((0, 1), 4)]
    assert (4, 3) not in model.counts
    assert model.prob((2, 2), 1) == pytest.approx(1/4)
    assert model.nll_per_token([[2, 2]]) == pytest.approx(-model.logprob([2, 2])/3)


def test_formula_window_boundaries_and_eos():
    from book.part2_lm.ch07_lm_basics.tour import window_dataset
    x, y = window_dataset([[0, 1], [1]], 3, 2, 3)
    assert x.tolist() == [[2, 2, 2], [2, 2, 0], [2, 0, 1], [2, 2, 2], [2, 2, 1]]
    assert y.tolist() == [0, 1, 3, 1, 3]


def test_diagnostic_nested_matching_and_byte_offsets():
    from book.part2_lm.ch07_lm_basics.tour import right_positions
    from miniocr.tokenizer import BPETokenizer
    # 独立手写合并，不依赖第 6 章的训练练习。
    tok = BPETokenizer([(b'r', b'i'), (b'ri', b'g'), (b'rig', b'h'), (b'righ', b't')], specials=())
    text = r"模\left(a+\left[b\right]+c\right)"
    matches = right_positions(tok, text)
    assert len(matches) == 1
    distance, target, source, ids = matches[0]
    assert tok.vocab[ids[target]] == b'right'
    assert b'(' in tok.vocab[ids[source]]
    assert distance == target-source-1


@pytest.mark.parametrize("command", [r"\leftarrow", r"\rightarrow", r"\leftrightarrow",
                                    r"\leftA", r"\rightZ"])
def test_diagnostic_command_boundaries(command):
    from book.part2_lm.ch07_lm_basics.tour import right_positions
    from miniocr.tokenizer import BPETokenizer

    tok = BPETokenizer([(b'r', b'i'), (b'ri', b'g'), (b'rig', b'h'), (b'righ', b't')], specials=())
    assert right_positions(tok, command) == []
    text = r"\left(" + command + r" x\right)"
    matches = right_positions(tok, text)
    assert len(matches) == 1
    distance, target, source, ids = matches[0]
    assert b''.join(tok.vocab[i] for i in ids[:source]) == b'\\left'
    assert tok.vocab[ids[source]] == b'('
    assert b''.join(tok.vocab[i] for i in ids[:target]) == text[:text.rindex(r"\right") + 1].encode()
    assert tok.vocab[ids[target]] == b'right'
    assert distance == target - source - 1


def test_braces_ignore_escaped_delimiters():
    from book.part2_lm.ch07_lm_basics.tour import braces_balanced
    assert braces_balanced(r"\{x\}")
    assert braces_balanced(r"{a{b}c}")
    assert not braces_balanced("}{")
    assert not braces_balanced(r"\\{x")
