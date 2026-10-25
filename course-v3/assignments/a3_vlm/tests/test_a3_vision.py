"""A3 第一部分：切块、位置编码、插值、二维 RoPE、ViT 块。对照 PyTorch 自带的实现或独立写的公式。"""
import math

import pytest
import torch
import torch.nn.functional as F

from ._impl import adapters as A


def _rot(x, cos, sin):
    x1, x2 = x.chunk(2, -1)
    return x * cos + torch.cat([-x2, x1], -1) * sin


def test_patchify_matches_conv():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 24, 40)
    w, b = torch.randn(16, 3, 8, 8), torch.randn(16)
    t = A.run_patchify(x, 8)
    assert t.shape == (2, 15, 192)
    ref = F.conv2d(x, w, b, stride=8).flatten(2).transpose(1, 2)        # 核大小 = 步长 = p 的卷积
    assert torch.allclose(t @ w.reshape(16, -1).T + b, ref, atol=1e-4)


def test_sincos_2d():
    h, w, d = 3, 5, 16
    pe = A.run_sincos_2d(h, w, d)
    assert pe.shape == (h * w, d)
    i, j, k = 2, 3, 1
    om = 10000 ** (-2 * k / (d // 2))
    row = pe[i * w + j]
    assert row[2 * k].item() == pytest.approx(math.sin(i * om), abs=1e-5)
    assert row[2 * k + 1].item() == pytest.approx(math.cos(i * om), abs=1e-5)
    assert row[d // 2 + 2 * k].item() == pytest.approx(math.sin(j * om), abs=1e-5)
    assert row[d // 2 + 2 * k + 1].item() == pytest.approx(math.cos(j * om), abs=1e-5)


@pytest.mark.parametrize("old,new", [((4, 4), (6, 6)), ((4, 6), (3, 9)), ((5, 5), (5, 5)), ((4, 4), (2, 2))])
def test_interpolate_pos(old, new):
    torch.manual_seed(0)
    pos = torch.randn(old[0] * old[1], 8)
    out = A.run_interpolate_pos(pos, old, new)
    ref = F.interpolate(pos.reshape(*old, 8).permute(2, 0, 1)[None], size=new, mode="bilinear", align_corners=False)
    assert torch.allclose(out, ref[0].permute(1, 2, 0).reshape(-1, 8), atol=1e-5)


def test_rope_2d_matches_complex():
    h, w, dh, base = 3, 4, 16, 100.0
    cos, sin = A.run_rope_2d(h, w, dh, base)
    assert cos.shape == sin.shape == (h * w, dh)
    torch.manual_seed(0)
    x = torch.randn(h * w, dh, dtype=torch.float64)
    q = dh // 4
    theta = base ** (-torch.arange(q, dtype=torch.float64) / q)
    ii, jj = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    ang = torch.cat([ii.reshape(-1, 1) * theta, jj.reshape(-1, 1) * theta], -1)       # (h·w, dh/2)
    z = torch.complex(x[:, :dh // 2], x[:, dh // 2:]) * torch.polar(torch.ones_like(ang), ang)
    ref = torch.cat([z.real, z.imag], -1)
    assert torch.allclose(_rot(x, cos.double(), sin.double()), ref, atol=1e-5)


def test_rope_2d_relative():
    # 旋转之后的内积只取决于两个 patch 的行差、列差
    h, w, dh = 5, 5, 16
    cos, sin = A.run_rope_2d(h, w, dh, 100.0)
    torch.manual_seed(1)
    q, k = torch.randn(dh), torch.randn(dh)

    def dot(a, b):
        return (_rot(q, cos[a], sin[a]) * _rot(k, cos[b], sin[b])).sum().item()

    assert dot(0 * w + 1, 2 * w + 3) == pytest.approx(dot(1 * w + 0, 3 * w + 2), abs=1e-4)      # 位移 (2, 2)
    assert dot(0, 1) != pytest.approx(dot(0, w), abs=1e-3)                                          # 横移 ≠ 竖移


def test_gelu_and_layer_norm():
    torch.manual_seed(0)
    x = torch.randn(4, 7, 12) * 3
    assert torch.allclose(A.run_gelu(x), F.gelu(x), atol=1e-6)
    wts = {"weight": torch.randn(12), "bias": torch.randn(12)}
    assert torch.allclose(A.run_layer_norm(12, wts, x), F.layer_norm(x, (12,), wts["weight"], wts["bias"], 1e-5), atol=1e-5)


def _block_weights(d, gen):
    shapes = {"n1.weight": (d,), "n1.bias": (d,), "n2.weight": (d,), "n2.bias": (d,),
              "attn.q.weight": (d, d), "attn.q.bias": (d,), "attn.kv.weight": (2 * d, d), "attn.kv.bias": (2 * d,),
              "attn.o.weight": (d, d), "attn.o.bias": (d,),
              "mlp.fc1.weight": (4 * d, d), "mlp.fc1.bias": (4 * d,), "mlp.fc2.weight": (d, 4 * d), "mlp.fc2.bias": (d,)}
    return {k: torch.randn(s, generator=gen) * (0.3 if len(s) == 2 else 0.1) + (1.0 if k.endswith("n1.weight") or k.endswith("n2.weight") else 0)
            for k, s in shapes.items()}


def _ref_block(W, x, n_head, key_mask=None, cos=None, sin=None):
    B, T, d = x.shape
    dh = d // n_head
    h = F.layer_norm(x, (d,), W["n1.weight"], W["n1.bias"], 1e-5)
    q = F.linear(h, W["attn.q.weight"], W["attn.q.bias"])
    k, v = F.linear(h, W["attn.kv.weight"], W["attn.kv.bias"]).chunk(2, -1)
    q, k, v = (t.view(B, T, n_head, dh).transpose(1, 2) for t in (q, k, v))
    if cos is not None:
        q, k = _rot(q, cos, sin), _rot(k, cos, sin)
    mask = None if key_mask is None else key_mask[:, None, None, :]
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(B, T, d)
    x = x + F.linear(y, W["attn.o.weight"], W["attn.o.bias"])
    h = F.layer_norm(x, (d,), W["n2.weight"], W["n2.bias"], 1e-5)
    return x + F.linear(F.gelu(F.linear(h, W["mlp.fc1.weight"], W["mlp.fc1.bias"])), W["mlp.fc2.weight"], W["mlp.fc2.bias"])


def test_vit_block():
    g = torch.Generator().manual_seed(0)
    d, n_head = 24, 3
    W = _block_weights(d, g)
    x = torch.randn(2, 6, d, generator=g)
    assert torch.allclose(A.run_vit_block(d, n_head, W, x), _ref_block(W, x, n_head), atol=1e-4)


def test_vit_block_mask_and_rope():
    g = torch.Generator().manual_seed(1)
    d, n_head = 32, 2
    W = _block_weights(d, g)
    x = torch.randn(2, 6, d, generator=g)
    key_mask = torch.tensor([[True] * 6, [True] * 4 + [False] * 2])
    cos, sin = A.run_rope_2d(2, 3, d // n_head, 100.0)
    out = A.run_vit_block(d, n_head, W, x, key_mask=key_mask, cos=cos, sin=sin)
    assert torch.allclose(out, _ref_block(W, x, n_head, key_mask, cos, sin), atol=1e-4)


@pytest.mark.parametrize("pos", ["learned", "sincos", "rope2d"])
def test_vit_shapes_and_resolution(pos):
    torch.manual_seed(0)
    vit = A.get_vit(32, 8, 32, 2, 2, pos)
    t, hw = vit.patch_tokens(torch.randn(2, 3, 32, 32))
    assert t.shape == (2, 16, 32) and tuple(hw) == (4, 4)
    t, hw = vit.patch_tokens(torch.randn(1, 3, 48, 64))                 # 换分辨率也能跑
    assert t.shape == (1, 48, 32) and tuple(hw) == (6, 8)
    assert torch.isfinite(t).all()

