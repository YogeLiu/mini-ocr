"""第 1 章的自动检查：python -m pytest book/part1_foundations/ch01_tensors"""
import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from miniocr.tensor_utils import (batch_to_image, broadcast_shape, image_to_batch, one_hot,
                                  pairwise_sq_dists, pairwise_sq_dists_mm)


@pytest.mark.parametrize("a, b", [((3, 1), (4,)), ((2, 3), (3,)), ((2, 3), (2, 1)), ((8, 4, 16, 16), (8, 1, 1, 16)),
                                  ((8, 3, 64, 256), (3, 1, 1)), ((), (2, 3)), ((1,), (5, 1, 7)), ((5, 1, 7), (1, 6, 1))])
def test_broadcast_shape_matches_torch(a, b):
    assert broadcast_shape(a, b) == tuple(torch.broadcast_shapes(a, b))
    assert broadcast_shape(torch.Size(a), torch.Size(b)) == tuple(torch.broadcast_shapes(a, b))


@pytest.mark.parametrize("a, b", [((2, 3), (2,)), ((8, 3, 64, 256), (3,)), ((4, 2), (3, 2))])
def test_broadcast_shape_rejects(a, b):
    with pytest.raises(ValueError):
        broadcast_shape(a, b)


def test_pairwise_hand_example():
    """1.3 节的例子：3 个点对 2 个点。"""
    from book.part1_foundations.ch01_tensors.tour import A3, B3
    expect = torch.tensor([[2.0, 9.0], [1.0, 4.0], [2.0, 13.0]])
    assert torch.equal(pairwise_sq_dists(A3, B3), expect)
    assert torch.allclose(pairwise_sq_dists_mm(A3, B3), expect)


def test_pairwise_random_matches_loop_and_cdist():
    torch.manual_seed(0)
    A, B = torch.randn(7, 5, dtype=torch.float64), torch.randn(4, 5, dtype=torch.float64)
    loop = torch.tensor([[((a - b) ** 2).sum().item() for b in B] for a in A], dtype=torch.float64)
    for fn in (pairwise_sq_dists, pairwise_sq_dists_mm):
        D = fn(A, B)
        assert D.shape == (7, 4)                       # N ≠ M，形状写反会在这里暴露
        assert torch.allclose(D, loop)
        assert torch.allclose(D, torch.cdist(A, B) ** 2)


def test_pairwise_mm_nonnegative():
    """展开写法在 float32 下可能出现负数，必须截掉。"""
    torch.manual_seed(0)
    X = torch.randn(200, 64) + 100
    D = pairwise_sq_dists_mm(X, X)
    assert (D >= 0).all()
    assert D.shape == (200, 200)


def test_one_hot_matches_torch():
    idx = torch.tensor([[0, 2, 1], [3, 3, 0]])
    oh = one_hot(idx, 4)
    assert oh.shape == (2, 3, 4) and oh.dtype == torch.float32
    assert torch.equal(oh, F.one_hot(idx, 4).float())
    assert one_hot(torch.tensor(2), 5).tolist() == [0, 0, 1, 0, 0]          # 0 维输入 → (V,)
    assert one_hot(idx, 4, dtype=torch.bool).dtype == torch.bool
    torch.manual_seed(0)
    big = torch.randint(0, 10, (2, 3, 4))
    assert torch.equal(one_hot(big, 10), F.one_hot(big, 10).float())


def test_one_hot_rejects_out_of_range():
    with pytest.raises(ValueError):
        one_hot(torch.tensor([0, 4]), 4)
    with pytest.raises(ValueError):
        one_hot(torch.tensor([-1, 0]), 4)


def test_image_to_batch_layout():
    torch.manual_seed(0)
    img = torch.randint(0, 256, (5, 7, 3), dtype=torch.uint8)               # (H, W, C)
    x = image_to_batch(img)
    assert x.shape == (1, 3, 5, 7) and x.dtype == torch.float32
    assert x.is_contiguous()                                                # 题目要求内存连续
    assert 0.0 <= x.min() and x.max() <= 1.0
    assert x[0, 2, 4, 1] == img[4, 1, 2].float() / 255                      # 像素 (行 4, 列 1) 的蓝色通道
    assert torch.equal(x, rearrange(img, "h w c -> 1 c h w").float() / 255)
    assert torch.equal(batch_to_image(x), img)
    with pytest.raises(ValueError):
        image_to_batch(img.float())
    with pytest.raises(ValueError):
        image_to_batch(img[..., 0])                                         # 二维 uint8
    with pytest.raises(ValueError):
        image_to_batch(img[None])                                           # 四维 uint8


def test_batch_to_image_picks_sample():
    x = torch.zeros(2, 3, 4, 6)
    x[1, 0] = 1.0                                                           # 第 2 张图的红色通道全满
    out = batch_to_image(x, 1)
    assert out.shape == (4, 6, 3) and out.dtype == torch.uint8
    assert out[..., 0].eq(255).all() and out[..., 1:].eq(0).all()
    assert batch_to_image(x, 0).eq(0).all()


def test_batch_to_image_rounds():
    """非整数像素要四舍五入：0.999 × 255 = 254.7 → 255，截断会得到 254。"""
    x = torch.tensor([0.999, 0.5, 0.001, 1.5, -0.2]).view(1, 1, 1, 5)
    assert batch_to_image(x)[0, :, 0].tolist() == [255, 128, 0, 255, 0]
