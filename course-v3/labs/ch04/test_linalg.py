"""第 4 章的自动检查。"""
import numpy as np
import pytest

from miniocr.linalg import (cosine_similarity, linear, matmul_flops, matmul_loops, mlp_param_count,
                            pairwise_sq_dists_mm, rotation)
from miniocr.utils import pairwise_sq_dists


def test_matmul_loops_hand_example():
    A = np.array([[1, 2], [3, 4], [5, 6]])
    B = np.array([[7, 8, 9], [10, 11, 12]])
    assert matmul_loops(A, B).tolist() == [[27, 30, 33], [61, 68, 75], [95, 106, 117]]


def test_matmul_loops_matches_numpy_and_checks_shapes():
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(4, 5)), rng.normal(size=(5, 3))
    assert np.allclose(matmul_loops(A, B), A @ B)
    with pytest.raises(ValueError):
        matmul_loops(A, A)


def test_cosine_similarity():
    A = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    B = np.array([[2.0, 0.0], [0.0, -3.0]])
    S = cosine_similarity(A, B)
    assert S.shape == (3, 2)
    assert np.allclose(S[0], [1.0, 0.0])
    assert np.isclose(S[1, 0], 1 / np.sqrt(2))
    assert np.allclose(S[2], 0.0)                                 # 零向量不产生 nan
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6, 8))
    assert np.allclose(cosine_similarity(X, 5 * X).diagonal(), 1.0)   # 与长度无关


def test_pairwise_mm_matches_broadcast():
    rng = np.random.default_rng(2)
    A, B = rng.normal(size=(30, 16)), rng.normal(size=(7, 16))
    assert np.allclose(pairwise_sq_dists_mm(A, B), pairwise_sq_dists(A, B))
    assert (pairwise_sq_dists_mm(A, A).diagonal() >= 0).all()


def test_linear_and_counts():
    x = np.ones((2, 3, 4))
    W, b = np.ones((4, 5)), np.arange(5.0)
    assert linear(x, W, b).shape == (2, 3, 5) and linear(x, W, b)[0, 0].tolist() == [4, 5, 6, 7, 8]
    assert mlp_param_count([1024, 256, 62]) == 278334
    assert mlp_param_count([3, 1]) == 4
    assert matmul_flops(2, 3, 4) == 48


def test_rotation():
    R = rotation(np.pi / 2)
    assert np.allclose(R @ [1, 0], [0, 1])
    assert np.allclose(R.T @ R, np.eye(2))
