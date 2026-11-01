"""第 9 章的数值、结构与共享评测检查；不读取真实语料。"""
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from miniocr.data.latex_lm import (
    FormulaWindows, encode_formulas, evaluate_latex, evaluate_lm, evaluate_nll,
    train_tokenizer,
)
from miniocr.models.gpt import Block, GPTConfig, RMSNorm, SwiGLU
from miniocr.tokenizer import BPETokenizer


def test_rmsnorm_hand_calculation():
    x = torch.tensor([1., 2., 3., 4.])
    assert torch.allclose(RMSNorm(4)(x), x / math.sqrt(7.5+1e-6))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rmsnorm_reference_and_dtype(dtype):
    x = torch.tensor([[10000., 10064., 10128., 9984.], [-2., 0., 1., 5.]], dtype=dtype)
    norm = RMSNorm(4)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([0.5, 1., -2., 3.]))
    ref = (x.float() / (x.float().square().mean(-1, keepdim=True)+norm.eps).sqrt() * norm.weight).to(dtype)
    result = norm(x)
    assert result.dtype == dtype
    torch.testing.assert_close(result, ref, atol=1e-6 if dtype == torch.float32 else 0, rtol=1e-6 if dtype == torch.float32 else 0)
    if dtype == torch.bfloat16:
        bad = (x * torch.rsqrt(x.square().mean(-1, keepdim=True)+norm.eps) * norm.weight).to(dtype)
        assert not torch.equal(bad, ref), "此输入必须能区分低精度中间计算"


def test_swiglu_fixed_weights_distinguish_gate_and_activation():
    m = SwiGLU(GPTConfig(d_model=2, d_ff=2))
    gate_w = torch.tensor([[1., -2.], [-1., 1.]])
    up_w = torch.tensor([[2., 1.], [1., -1.]])
    down_w = torch.tensor([[1., 2.], [-1., 1.]])
    with torch.no_grad():
        m.gate.weight.copy_(gate_w)
        m.up.weight.copy_(up_w)
        m.down.weight.copy_(down_w)
    x = torch.tensor([[1., 2.], [-2., 0.5]])
    gate, up = x @ gate_w.T, x @ up_w.T
    # 独立写出 sigmoid 公式，不调用被测实现或 F.silu。
    expected = ((gate / (1+torch.exp(-gate))) * up) @ down_w.T
    torch.testing.assert_close(m(x), expected)
    assert not torch.allclose(expected, (F.relu(gate)*up) @ down_w.T)
    assert not torch.allclose(expected, (F.silu(up)*gate) @ down_w.T)


def tiny_block():
    torch.manual_seed(0)
    return Block(GPTConfig(d_model=8, n_head=2, d_ff=16, dropout=0.0))


def tables(T=5):
    return torch.ones(T, 4), torch.zeros(T, 4)


def test_block_zero_branches_are_identity():
    block = tiny_block()
    with torch.no_grad():
        for module in (block.attn, block.mlp):
            for p in module.parameters():
                p.zero_()
    x = torch.randn(2, 5, 8)
    assert torch.equal(block(x, *tables()), x)


def test_block_pre_norm_order_and_updated_second_residual():
    block = tiny_block()
    x = torch.arange(80).reshape(2, 5, 8).float()/13+0.2
    cos, sin = tables()
    # 独立计算两次归一化；hook 检查子层实际读取的张量和 cache 透传。
    norm = lambda z, weight: z / (z.square().mean(-1, keepdim=True)+1e-6).sqrt() * weight
    first = norm(x, block.n1.weight)
    h = x + block.attn(first, cos, sin)
    second = norm(h, block.n2.weight)
    expected = h + block.mlp(second)
    seen = {}
    handles = [block.attn.register_forward_pre_hook(lambda m, a: seen.update(attn=a)),
               block.mlp.register_forward_pre_hook(lambda m, a: seen.update(mlp=a))]
    cache = {}
    try:
        result = block(x, cos, sin, cache)
    finally:
        for handle in handles:
            handle.remove()
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(seen['attn'][0], first)
    torch.testing.assert_close(seen['mlp'][0], second)
    assert seen['attn'][3] is cache and 'k' in cache
    assert not torch.allclose(first, second)


@pytest.mark.parametrize("impl", ["manual", "sdpa"])
def test_block_causality(impl):
    block = tiny_block().eval()
    block.attn.impl = impl
    x = torch.randn(2, 5, 8)
    changed = x.clone()
    changed[:, 3:] += 10
    assert torch.equal(block(x, *tables())[:, :3], block(changed, *tables())[:, :3])


@pytest.mark.parametrize("d, expected", [(1, 64), (24, 64), (25, 128), (64, 192), (192, 512), (256, 704)])
def test_ff_dim_rounds_up(d, expected):
    assert GPTConfig(d_model=d).ff_dim() == expected
    assert expected % 64 == 0 and expected >= 8*d/3
    assert expected - 64 < 8*d/3
    assert GPTConfig(d_model=d, d_ff=17).ff_dim() == 17


def test_encoding_boundaries_stream_and_utf8_bytes():
    tok = BPETokenizer([], specials=("<|bos|>", "<|eos|>"))
    texts = ["x", "模", "<|eos|>"]
    seqs, n_bytes = encode_formulas(tok, texts)
    stream, n2 = encode_formulas(tok, texts, stream=True)
    assert n_bytes == n2 == 11
    assert stream.tolist() == sum(seqs, [])
    assert all(s[0] == 256 and s[-1] == 257 for s in seqs)
    assert seqs[-1][1:-1] == list(b"<|eos|>")


class ConstantLM(nn.Module):
    def forward(self, x):
        # BOS logit 很大：若忘了屏蔽，指标会严重错误。
        return torch.tensor([100., math.log(2), math.log(3), math.log(5)]).expand(*x.shape, 4)


def test_evaluation_token_weighted_eos_and_bits_per_byte():
    # 两条内容序列长度 1 和 3；目标分别 [1,EOS] 与 [2,1,2,EOS]。
    data = FormulaWindows([[1], [2, 1, 2]], bos=0, eos=3, context=2)
    model = ConstantLM().train()
    result = evaluate_lm(model, data, n_bytes=7, batch=4)
    total = -2*math.log(0.2)-2*math.log(0.3)-2*math.log(0.5)
    assert result['nat_per_token'] == pytest.approx(total/6, abs=1e-6)
    assert result['bits_per_byte'] == pytest.approx(total/(7*math.log(2)), abs=1e-6)
    assert result['n_tokens'] == 6 and model.training
    assert evaluate_nll(model, data, batch=1) == pytest.approx(total/6, abs=1e-6)
    x, lengths, y = data.batch(torch.arange(len(data)))
    assert y.tolist() == [1, 3, 2, 1, 2, 3]
    assert x[-1].tolist() == [1, 2] and lengths[-1] == 2
    # 首窗口不能读取目标；右补齐仅重复 BOS。
    assert x[0].tolist() == [0, 0] and lengths[0] == 1


def test_evaluation_restores_mode_on_error():
    model = ConstantLM().train()
    data = FormulaWindows([[1]], bos=0, eos=3, context=2)
    def fail(*args):
        assert not model.training
        raise RuntimeError("fixture")
    with pytest.raises(RuntimeError, match="fixture"):
        evaluate_nll(model, data, window_logits=fail)
    assert model.training


def test_evaluate_latex_prefix_and_generic_tuple_logits():
    tok = BPETokenizer([], specials=("<|bos|>", "<|eos|>"))
    class Uniform(nn.Module):
        def forward(self, x):
            return torch.zeros(*x.shape, len(tok)), None
    texts = ["x"] * 2000 + ["not included"]
    result = evaluate_latex(Uniform(), tok, texts, context=1)
    assert result['n_tokens'] == 4000 and result['n_bytes'] == 2000
    assert result['nat_per_token'] == pytest.approx(math.log(257), abs=2e-6)
    assert result['bits_per_byte'] == pytest.approx(2*math.log2(257), abs=5e-6)


def test_shared_tokenizer_uses_only_training_prefix(monkeypatch):
    seen = []
    def fake_train(texts, n):
        seen.append((texts, n))
        return []
    monkeypatch.setattr('miniocr.data.latex_lm.train_bpe', fake_train)
    tok = train_tokenizer(["x"]*20000+["excluded"])
    assert seen == [(["x"]*20000, 2000)]
    assert tok.specials == ["<|bos|>", "<|eos|>"]


def test_tour_unigram_smoothing_counts_eos_excludes_bos():
    from book.part2_lm.ch09_block.tour import unigram_baseline

    train = FormulaWindows([[1], [1, 2]], bos=0, eos=3, context=2)
    val = FormulaWindows([[2, 4], []], bos=0, eos=3, context=2)
    loss, k = unigram_baseline(train, val, vocab=5)
    # 训练目标计数为 [0,2,1,2,0]；验证目标为 [2,4,EOS,EOS]。
    # 未见 token 4 使这组样例选中 k=1；分母是 5+1*4，不能包括 BOS。
    assert k == 1.0
    assert loss == pytest.approx(-(math.log(2/9) + math.log(1/9) + 2*math.log(3/9))/4)
