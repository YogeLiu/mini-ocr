"""第 15 章的自动检查。"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from miniocr.nn.attention import MultiHeadAttention, attention, causal_mask


def test_attention_hand_example():
    q = torch.tensor([[1.0, 0.0]])
    k = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    v = torch.tensor([[10.0], [20.0], [30.0]])
    out, w = attention(q, k, v)
    s = torch.tensor([1.0, 0.0, 1.0]) / math.sqrt(2)
    p = s.exp() / s.exp().sum()
    assert torch.allclose(w[0], p, atol=1e-6)
    assert math.isclose(out.item(), (p * torch.tensor([10.0, 20.0, 30.0])).sum().item(), rel_tol=1e-6)
    assert torch.allclose(w.sum(-1), torch.ones(1))


def test_attention_matches_torch_and_mask():
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 3, 5, 8), torch.randn(2, 3, 7, 8), torch.randn(2, 3, 7, 4)
    out, _ = attention(q, k, v)
    assert torch.allclose(out, F.scaled_dot_product_attention(q, k, v), atol=1e-5)
    x = torch.randn(2, 3, 6, 8)
    out, w = attention(x, x, x, causal_mask(6))
    assert torch.allclose(out, F.scaled_dot_product_attention(x, x, x, is_causal=True), atol=1e-5)
    assert (w[..., 0, 1:] == 0).all() and torch.allclose(w[..., 0, 0], torch.ones(1))


def test_causal_mask():
    m = causal_mask(4)
    assert m.dtype == torch.bool and m.shape == (4, 4)
    assert m.tolist() == [[True, False, False, False], [True, True, False, False],
                          [True, True, True, False], [True, True, True, True]]


def test_mha_matches_nn_multihead_attention():
    torch.manual_seed(0)
    d, h = 16, 4
    mine = MultiHeadAttention(d, h)
    ref = nn.MultiheadAttention(d, h, batch_first=True)
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat([mine.q.weight, mine.k.weight, mine.v.weight]))
        ref.in_proj_bias.copy_(torch.cat([mine.q.bias, mine.k.bias, mine.v.bias]))
        ref.out_proj.weight.copy_(mine.o.weight); ref.out_proj.bias.copy_(mine.o.bias)
    x, mem = torch.randn(2, 5, d), torch.randn(2, 9, d)
    assert torch.allclose(mine(x), ref(x, x, x)[0], atol=1e-5)
    assert torch.allclose(mine(x, kv=mem), ref(x, mem, mem)[0], atol=1e-5)          # 交叉注意力
    assert mine.last_weights.shape == (2, h, 5, 9)
    cm = MultiHeadAttention(d, h, causal=True)
    cm.load_state_dict(mine.state_dict())
    y = cm(x)
    x2 = x.clone(); x2[:, 3:] = torch.randn(2, 2, d)                                # 改动"未来"的 token
    assert torch.allclose(cm(x2)[:, :3], y[:, :3], atol=1e-6)                       # 前 3 个位置的输出不变
    assert not torch.allclose(cm(x2)[:, 3:], y[:, 3:])


def test_self_attention_is_permutation_equivariant():
    torch.manual_seed(0)
    m = MultiHeadAttention(8, 2)
    x = torch.randn(1, 6, 8)
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    assert torch.allclose(m(x)[:, perm], m(x[:, perm]), atol=1e-6)
