"""第 19 章的自动检查。"""
import math

import torch

from miniocr.models.gpt import apply_rope
from miniocr.models.vit import PatchEmbed, ViT, ViTConfig, interpolate_pos, patchify, rope_2d, sincos_2d


def test_patchify_equals_conv():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 16, 24)
    pt = patchify(x, 8)
    assert pt.shape == (2, 6, 192)
    assert torch.equal(pt[1, 4], x[1, :, 8:16, 8:16].reshape(-1))         # 第 1 行第 1 列（4 = 1·3 + 1）
    pe = PatchEmbed(3, 32, 8)
    y, hw = pe(x)
    assert hw == (2, 3) and y.shape == (2, 6, 32)
    lin = pt @ pe.proj.weight.reshape(32, -1).T + pe.proj.bias
    assert torch.allclose(y, lin, atol=1e-5)                              # 切块 + 线性层 = 卷积


def test_sincos_and_interpolation():
    pe = sincos_2d(3, 4, 16)
    assert pe.shape == (12, 16)
    assert torch.equal(pe[1 * 4 + 2, :8], pe[1 * 4 + 0, :8])            # 同一行：前一半相同
    assert torch.equal(pe[0 * 4 + 2, 8:], pe[2 * 4 + 2, 8:])            # 同一列：后一半相同
    assert not torch.equal(pe[0], pe[1])
    pos = torch.randn(16, 8)
    assert interpolate_pos(pos, (4, 4), (4, 4)) is pos
    big = interpolate_pos(pos, (4, 4), (6, 8))
    assert big.shape == (48, 8)
    const = interpolate_pos(torch.ones(16, 8) * 3, (4, 4), (7, 5))       # 常数插值后还是常数
    assert torch.allclose(const, torch.full((35, 8), 3.0), atol=1e-5)


def test_rope_2d_relative():
    h, w, dh = 5, 6, 16
    cos, sin = rope_2d(h, w, dh)
    assert cos.shape == (30, 16) and torch.allclose(cos[0], torch.ones(16))
    torch.manual_seed(0)
    q, k = torch.randn(dh), torch.randn(dh)
    rq, rk = apply_rope(q.expand(30, dh), cos, sin), apply_rope(k.expand(30, dh), cos, sin)
    at = lambda i, j: i * w + j
    s1 = rq[at(3, 4)] @ rk[at(1, 2)]                                     # 行差 2、列差 2
    s2 = rq[at(4, 3)] @ rk[at(2, 1)]                                     # 同样的行差、列差
    assert math.isclose(s1.item(), s2.item(), rel_tol=1e-4, abs_tol=1e-5)
    s3 = rq[at(3, 4)] @ rk[at(1, 4)]                                     # 行差 2、列差 0
    s4 = rq[at(3, 4)] @ rk[at(3, 2)]                                     # 行差 0、列差 2：和上面不同
    assert not math.isclose(s3.item(), s4.item(), rel_tol=1e-3)
    assert torch.equal(cos[at(2, 0), :4], cos[at(2, 5), :4])            # 前 dh/4 维只看行号


def test_vit_shapes_all_variants():
    x = torch.randn(2, 3, 32, 32)
    for pos in ("learned", "sincos", "rope2d", "none"):
        for pool in ("cls", "mean"):
            torch.manual_seed(0)
            m = ViT(ViTConfig(img_size=32, patch=8, d_model=32, n_layer=2, n_head=4, pos=pos, pool=pool, n_classes=5))
            assert m(x).shape == (2, 5)
            t, hw = m.encode(x)
            assert hw == (4, 4) and t.shape == (2, 16 + (pool == "cls"), 32)
            pt, _ = m.patch_tokens(torch.randn(1, 3, 48, 64))            # 换分辨率也能跑
            assert pt.shape == (1, 48, 32)


def test_vit_position_matters():
    """不加位置信息时，打乱 patch 的顺序不改变 CLS 的输出（置换等变，第 15 章）；加了位置信息就会改变。"""
    torch.manual_seed(0)
    x = torch.randn(1, 3, 32, 32)
    perm = torch.randperm(16)
    def shuffled(img):
        pt = patchify(img, 8)[:, perm]                                   # 把 16 个 patch 打乱后拼回图
        return pt.reshape(1, 4, 4, 3, 8, 8).permute(0, 3, 1, 4, 2, 5).reshape(1, 3, 32, 32)
    assert torch.equal(patchify(shuffled(x), 8), patchify(x, 8)[:, perm])
    for pos, same in (("none", True), ("learned", False), ("rope2d", False)):
        torch.manual_seed(0)
        m = ViT(ViTConfig(img_size=32, patch=8, d_model=32, n_layer=2, n_head=4, pos=pos)).eval()
        if pos == "learned":
            m.pos.data.normal_(0, 1)
        a, b = m(x), m(shuffled(x))
        assert torch.allclose(a, b, atol=1e-5) == same, pos
