"""A3 第五部分：Qwen 系列的图像预处理。装了 transformers 时额外与它的 PIL 版 processor 逐位对照。"""
import numpy as np
import pytest
import torch
from PIL import Image

from ._impl import adapters as A

KW = dict(factor=32, min_pixels=256 * 256, max_pixels=4096 * 4096)


def test_smart_resize():
    assert A.run_smart_resize(256, 256, **KW) == (256, 256)
    assert A.run_smart_resize(192, 256, **KW) == (224, 320)                 # 太小：放大到 ≥ 65536 像素
    assert A.run_smart_resize(300, 500, **KW) == (288, 512)                 # 只取整
    h, w = A.run_smart_resize(3000, 4000, 32, 256 * 256, 1024 * 1024)
    assert (h, w) == (864, 1152)
    h, w = A.run_smart_resize(32, 2000, **KW)                               # 文字行：很扁
    assert h % 32 == 0 and w % 32 == 0 and h * w >= 256 * 256
    with pytest.raises(ValueError):
        A.run_smart_resize(10, 3000, **KW)


def test_qwen_patchify_order():
    x = torch.arange(16.0).reshape(1, 4, 4)                                 # 4×4 个 patch（patch=1），值 = 行·4 + 列
    pv, grid = A.run_qwen_patchify(x, 1, 2, 2)
    assert grid.tolist() == [[1, 4, 4]]
    assert pv.shape == (16, 2)
    assert pv[:, 0].tolist() == [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
    assert torch.equal(pv[:, 0], pv[:, 1])                                  # 时间维是复制的
    pv, grid = A.run_qwen_patchify(torch.randn(3, 64, 96), 16, 2, 2)
    assert pv.shape == (24, 3 * 2 * 16 * 16) and grid.tolist() == [[1, 4, 6]]


def test_qwen_patchify_conv3d():
    torch.manual_seed(0)
    x = torch.randn(3, 64, 64)
    pv, _ = A.run_qwen_patchify(x, 16, 2, 2)
    conv = torch.nn.Conv3d(3, 8, kernel_size=(2, 16, 16), stride=(2, 16, 16), bias=False)
    y1 = pv @ conv.weight.reshape(8, -1).T
    video = x.unsqueeze(1).expand(-1, 2, -1, -1).unsqueeze(0)               # (1, C, T=2, H, W)
    y2 = conv(video)[0].permute(1, 2, 3, 0).reshape(4, 4, 8)                # (行, 列, 8)
    # 第 2 行（窗口 1）是 patch (0, 2)；第 5 行是 patch (0, 3)... 按窗口顺序重排后逐个对照
    order = [(r, c) for wr in range(2) for wc in range(2) for r in (2 * wr, 2 * wr + 1) for c in (2 * wc, 2 * wc + 1)]
    ref = torch.stack([y2[r, c] for r, c in order])
    assert torch.allclose(y1, ref, atol=1e-4)


def test_expand_image_tokens():
    ids = [1, 9, 2, 9, 3]
    out = A.run_expand_image_tokens(ids, [[1, 4, 6], [1, 2, 2]], 9, 2)
    assert out == [1] + [9] * 6 + [2] + [9] * 1 + [3]
    with pytest.raises(ValueError):
        A.run_expand_image_tokens(ids, [[1, 4, 6]], 9, 2)
    with pytest.raises(ValueError):
        A.run_expand_image_tokens([1, 9], [[1, 4, 6], [1, 2, 2]], 9, 2)


def test_mrope_positions():
    ids = [5, 5] + [9] * 6 + [5, 5]                                         # 2 个文字 + 合并后 2×3 的图 + 2 个文字
    p = A.run_mrope_positions(ids, [[1, 4, 6]], 9, 2)
    assert p.shape == (3, 10)
    assert p[:, :2].tolist() == [[0, 1]] * 3
    assert p[0, 2:8].tolist() == [2] * 6
    assert p[1, 2:8].tolist() == [2, 2, 2, 3, 3, 3]
    assert p[2, 2:8].tolist() == [2, 3, 4, 2, 3, 4]
    assert p[:, 8:].tolist() == [[5, 6]] * 3
    assert A.run_mrope_positions([5, 5, 5], [], 9, 2).tolist() == [[0, 1, 2]] * 3


def test_mrope_two_images():
    ids = [5] + [9] * 1 + [5] + [9] * 4 + [5]                              # 图 1：合并后 1×1；图 2：2×2
    p = A.run_mrope_positions(ids, [[1, 2, 2], [1, 4, 4]], 9, 2)
    assert p[:, 0].tolist() == [0, 0, 0]
    assert p[:, 1].tolist() == [1, 1, 1]
    assert p[:, 2].tolist() == [2, 2, 2]                                    # 1 + max(1, 1)
    assert p[1, 3:7].tolist() == [3, 3, 4, 4] and p[2, 3:7].tolist() == [3, 4, 3, 4]
    assert p[:, 7].tolist() == [5, 5, 5]


def test_against_transformers():
    pil = pytest.importorskip("transformers.models.qwen2_vl.image_processing_pil_qwen2_vl")
    ip = pil.Qwen2VLImageProcessorPil(patch_size=16, merge_size=2, temporal_patch_size=2,
                                      image_mean=[0.5] * 3, image_std=[0.5] * 3,
                                      size={"shortest_edge": 256 * 256, "longest_edge": 4096 * 4096})
    rng = np.random.default_rng(0)
    for hw in [(150, 230), (300, 300), (40, 900)]:
        img = Image.fromarray(rng.integers(0, 256, (*hw, 3), dtype=np.uint8))
        ref = ip(images=[img], return_tensors="pt")
        pv, grid = A.run_qwen_preprocess(img)
        assert grid.tolist() == ref["image_grid_thw"].tolist()
        assert torch.allclose(pv, ref["pixel_values"].float(), atol=1e-4)
