"""第 24 章的自动检查。不需要下载模型；装了 transformers 时额外与它的 PIL 版 processor 对照。"""
import math

import numpy as np
import pytest
import torch
from PIL import Image

from miniocr.hf_vlm import (expand_image_tokens, idefics3_layout, mrope_axes, mrope_positions, qwen_patchify,
                            qwen_preprocess, smart_resize)


def test_smart_resize():
    assert smart_resize(256, 256) == (256, 256)
    assert smart_resize(192, 256) == (224, 320)                         # 太小：放大到 ≥ 65536 像素
    h, w = smart_resize(3000, 4000, max_pixels=1024 * 1024)
    assert h % 32 == 0 and w % 32 == 0 and h * w <= 1024 * 1024
    assert abs(w / h - 4 / 3) < 0.05
    h, w = smart_resize(32, 2000)                                        # 文字行：很扁
    assert h % 32 == 0 and w % 32 == 0 and h * w >= 256 * 256


def test_qwen_patchify_order():
    # 4×4 个 patch（patch=1），值 = 行·4 + 列；merge=2 时顺序按 2×2 窗口分组
    x = torch.arange(16.0).reshape(1, 4, 4)
    pv, grid = qwen_patchify(x, patch=1, merge=2, temporal=2)
    assert grid.tolist() == [[1, 4, 4]]
    assert pv.shape == (16, 2)
    assert pv[:, 0].tolist() == [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
    assert torch.equal(pv[:, 0], pv[:, 1])                               # 时间维是复制的
    pv, _ = qwen_patchify(torch.randn(3, 64, 96), patch=16, merge=2, temporal=2)
    assert pv.shape == (4 * 6, 3 * 2 * 16 * 16)


def test_qwen_patchify_conv3d():
    # 每一行按 (C, T, p, p) 拉平：乘 Conv3d 的权重 = 对那个 patch 做 Conv3d
    torch.manual_seed(0)
    x = torch.randn(3, 64, 64)
    pv, _ = qwen_patchify(x, patch=16, merge=2, temporal=2)
    conv = torch.nn.Conv3d(3, 8, kernel_size=(2, 16, 16), stride=(2, 16, 16), bias=False)
    y1 = pv @ conv.weight.reshape(8, -1).T
    first = x[:, :16, :16]                                               # 第 0 个 patch = 左上角
    y2 = conv(first.unsqueeze(1).expand(-1, 2, -1, -1).unsqueeze(0)).flatten()
    assert torch.allclose(y1[0], y2, atol=1e-4)


def test_expand_image_tokens():
    ids = [1, 9, 2, 9, 3]
    out = expand_image_tokens(ids, [[1, 4, 6], [1, 2, 2]], image_id=9, merge=2)
    assert out == [1] + [9] * 6 + [2] + [9] * 1 + [3]
    with pytest.raises(ValueError):
        expand_image_tokens(ids, [[1, 4, 6]], image_id=9)


def test_mrope_positions():
    # 2 个文字 + 一张合并后 2×3 的图 + 2 个文字
    ids = [5, 5] + [9] * 6 + [5, 5]
    p = mrope_positions(ids, [[1, 4, 6]], image_id=9, merge=2)
    assert p.shape == (3, 10)
    assert p[:, :2].tolist() == [[0, 1]] * 3
    assert p[0, 2:8].tolist() == [2] * 6                                 # 时间：都是起点
    assert p[1, 2:8].tolist() == [2, 2, 2, 3, 3, 3]                      # 行
    assert p[2, 2:8].tolist() == [2, 3, 4, 2, 3, 4]                      # 列
    assert p[:, 8:].tolist() == [[5, 6]] * 3                             # 2 + max(2, 3) = 5
    q = mrope_positions([5, 5, 5], [], image_id=9)                       # 纯文字 = 普通的一维位置
    assert q.tolist() == [[0, 1, 2]] * 3


def test_mrope_axes():
    a = mrope_axes(32, (11, 11, 10), interleaved=True)
    assert a[:6] == ["T", "H", "W", "T", "H", "W"]
    assert (a.count("T"), a.count("H"), a.count("W")) == (11, 11, 10)
    b = mrope_axes(64, (16, 24, 24), interleaved=False)
    assert b[15] == "T" and b[16] == "H" and b[40] == "W"


def test_idefics3_layout():
    assert idefics3_layout(192, 256) == (3, 4, 13 * 64)                  # 小图被放大到长边 2048
    assert idefics3_layout(512, 512, longest=512) == (0, 0, 64)          # 正好一块：不切
    r, c, n = idefics3_layout(32, 2000)
    assert r == 1 and c == 4 and n == 5 * 64


def test_against_transformers():
    pil = pytest.importorskip("transformers.models.qwen2_vl.image_processing_pil_qwen2_vl")
    ip = pil.Qwen2VLImageProcessorPil(patch_size=16, merge_size=2, temporal_patch_size=2,
                                      image_mean=[0.5] * 3, image_std=[0.5] * 3,
                                      size={"shortest_edge": 256 * 256, "longest_edge": 4096 * 4096})
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (150, 230, 3), dtype=np.uint8))
    ref = ip(images=[img], return_tensors="pt")
    pv, grid = qwen_preprocess(img)
    assert grid.tolist() == ref["image_grid_thw"].tolist()
    assert torch.allclose(pv, ref["pixel_values"].float(), atol=1e-4)
