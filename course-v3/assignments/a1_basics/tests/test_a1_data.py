"""A1 问题 data：生成、去重、取批。"""
import numpy as np
import torch

from miniocr.data.tasks import glyph_arrays

from ._impl import adapters as A


def test_build_split_matches_renderer():
    X, y = A.run_build_split(7, 2, 1.0, 0.05)
    assert X.shape == (124, 1, 32, 32) and X.dtype == np.float32
    assert y.shape == (124,) and y.dtype == np.int64 and set(y.tolist()) == set(range(62))
    ref, ry = glyph_arrays(np.random.default_rng(7), n_per_class=2, noise=0.05, distort=1.0, flat=False)
    assert np.array_equal(X[:, 0], ref) and np.array_equal(y, ry)
    assert 0 <= X.min() and X.max() <= 1 and X.mean() < 0.5               # 墨迹 = 1，背景占多数


def test_build_split_seeds():
    a, _ = A.run_build_split(3, 1, 1.0, 0.0)
    b, _ = A.run_build_split(3, 1, 1.0, 0.0)
    c, _ = A.run_build_split(4, 1, 1.0, 0.0)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_remove_overlap():
    rng = np.random.default_rng(0)
    X = rng.random((6, 1, 4, 4)).astype(np.float32)
    y = np.arange(6)
    ref = np.concatenate([X[[1, 4]], rng.random((3, 1, 4, 4)).astype(np.float32)])
    X2, y2, n = A.run_remove_overlap(X, y, ref)
    assert n == 2 and y2.tolist() == [0, 2, 3, 5] and np.array_equal(X2, X[[0, 2, 3, 5]])
    X3 = X.copy()
    X3[1, 0, 0, 0] += 1e-3                                                  # 改一个像素就不算重复
    assert A.run_remove_overlap(X3, y, ref)[2] == 1


def test_get_batch():
    X = np.arange(20 * 4, dtype=np.float32).reshape(20, 1, 2, 2)
    y = np.arange(20, dtype=np.int64)
    xb, yb = A.run_get_batch(X, y, 64, torch.Generator().manual_seed(0))
    assert isinstance(xb, torch.Tensor) and xb.shape == (64, 1, 2, 2) and xb.dtype == torch.float32
    assert yb.shape == (64,) and yb.dtype == torch.int64
    assert torch.equal(xb[:, 0, 0, 0], yb.float() * 4)                     # 图和标签来自同一行
    assert len(set(yb.tolist())) > 10                                       # 不是总取同几个
    xb2, yb2 = A.run_get_batch(X, y, 64, torch.Generator().manual_seed(0))
    assert torch.equal(yb, yb2)
    _, yb3 = A.run_get_batch(X, y, 64, torch.Generator().manual_seed(1))
    assert not torch.equal(yb, yb3)
