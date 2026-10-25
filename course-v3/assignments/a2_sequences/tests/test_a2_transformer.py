"""A2 第三部分：RMSNorm、RoPE、注意力、GPT（快照，已与第 16 章的实现交叉核对）、KV cache、采样。"""
import math
import os

import pytest
import torch
import torch.nn.functional as F

from ._impl import FIXTURES
from ._impl import adapters as A

GPT_FX = None


def _gpt():
    global GPT_FX
    if GPT_FX is None:
        GPT_FX = torch.load(os.path.join(FIXTURES, "gpt.pt"), weights_only=False)
    return GPT_FX


def test_rmsnorm():
    g = torch.Generator().manual_seed(0)
    x, w = torch.randn(2, 5, 8, generator=g) * 3, torch.randn(8, generator=g)
    want = x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + 1e-6) * w
    torch.testing.assert_close(A.run_rmsnorm(8, 1e-6, w, x), want)


def _rope_complex(x, base, pos0):
    """独立的写法：把第 i 维和第 i + dh/2 维看成复数 a + bi，乘以 e^{i·t·θ_i}。"""
    dh = x.shape[-1]
    half = dh // 2
    z = torch.complex(x[..., :half].double(), x[..., half:].double())
    theta = base ** (-torch.arange(half, dtype=torch.float64) * 2 / dh)
    t = torch.arange(pos0, pos0 + x.shape[-2], dtype=torch.float64)
    z = z * torch.polar(torch.ones(len(t), half, dtype=torch.float64), t[:, None] * theta[None])
    return torch.cat([z.real, z.imag], -1).to(x.dtype)


@pytest.mark.parametrize("pos0", [0, 7])
def test_rope(pos0):
    x = torch.randn(2, 3, 6, 8, generator=torch.Generator().manual_seed(1))
    y = A.run_rope(x, 10000.0, pos0)
    torch.testing.assert_close(y, _rope_complex(x, 10000.0, pos0), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1))           # 旋转不改变长度


def test_rope_relative():
    q = torch.randn(1, 1, 1, 16, generator=torch.Generator().manual_seed(2))
    k = torch.randn(1, 1, 1, 16, generator=torch.Generator().manual_seed(3))
    dots = [(A.run_rope(q, 100.0, m) * A.run_rope(k, 100.0, m - 3)).sum().item() for m in (3, 10, 25)]
    assert max(dots) - min(dots) < 1e-4                                 # 只取决于相对位置 m − n


def test_attention():
    g = torch.Generator().manual_seed(4)
    q, k, v = (torch.randn(2, 3, 5, 8, generator=g, dtype=torch.float64) for _ in range(3))
    k, v = k.repeat(1, 1, 2, 1)[:, :, :7], v.repeat(1, 1, 2, 1)[:, :, :7]
    masks = [None, torch.ones(5, 7, dtype=torch.bool).tril(2), torch.rand(5, 7, generator=g) > 0.5]
    masks[2][:, 0] = True                                               # 每一行至少能看一个位置
    for m in masks:
        torch.testing.assert_close(A.run_attention(q, k, v, m), F.scaled_dot_product_attention(q, k, v, attn_mask=m))


def test_gpt_matches_snapshot():
    fx = _gpt()
    torch.testing.assert_close(A.run_gpt(fx["cfg"], fx["state_dict"], fx["idx"]), fx["logits"], atol=1e-4, rtol=1e-4)


def test_gpt_structure():
    fx = _gpt()
    m = A.get_gpt(fx["cfg"])
    assert set(m.state_dict()) == set(fx["state_dict"])
    assert sum(p.numel() for p in m.parameters()) == 34976              # 共享的嵌入只算一次
    m.load_state_dict(fx["state_dict"])
    m.eval()
    idx = fx["idx"].clone()
    logits, loss = m(idx, idx)
    torch.testing.assert_close(loss, F.cross_entropy(logits.reshape(-1, 64), idx.reshape(-1)))
    idx2 = idx.clone()
    idx2[:, 10:] = (idx2[:, 10:] + 1) % 64                             # 改变后面的 token，前面的 logits 不能变
    torch.testing.assert_close(m(idx2)[0][:, :10], logits[:, :10])


@pytest.mark.parametrize("n_prefill", [1, 5, 16])
def test_kv_cache_matches_full(n_prefill):
    fx = _gpt()
    out = A.run_gpt_incremental(fx["cfg"], fx["state_dict"], fx["idx"], n_prefill)
    torch.testing.assert_close(out, fx["logits"], atol=1e-4, rtol=1e-4)


def _finite(x):
    return torch.isfinite(x)


def test_process_logits():
    g = torch.Generator().manual_seed(5)
    logits = torch.randn(4, 20, generator=g, dtype=torch.float64) * 2
    torch.testing.assert_close(A.run_process_logits(logits, 0.5, 0, 1.0), logits / 0.5)
    out = A.run_process_logits(logits, 1.0, 3, 1.0)
    top = logits.topk(3, -1).indices
    want = torch.zeros_like(logits, dtype=torch.bool).scatter(1, top, True)
    assert torch.equal(_finite(out), want)
    torch.testing.assert_close(out[want], logits[want])
    for p in (0.3, 0.8, 0.95):
        out = A.run_process_logits(logits, 1.0, 0, p)
        probs = logits.softmax(-1)
        for b in range(4):
            order = probs[b].argsort(descending=True)
            cum = probs[b][order].cumsum(0)
            n_keep = int((cum < p).sum()) + 1                          # 累计概率第一次达到 p 为止
            assert set(torch.nonzero(_finite(out[b])).flatten().tolist()) == set(order[:n_keep].tolist()), (p, b)
    out = A.run_process_logits(logits, 1.0, 0, 1e-9)
    assert (_finite(out).sum(-1) == 1).all()                           # 至少保留 1 个
