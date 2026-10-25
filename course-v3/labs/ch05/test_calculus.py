"""第 5 章的自动检查。"""
import numpy as np

from miniocr.calculus import numerical_derivative, numerical_gradient, rel_error, sigmoid


def test_numerical_derivative_known_functions():
    assert abs(numerical_derivative(lambda x: x ** 2, 3.0) - 6.0) < 1e-6
    assert abs(numerical_derivative(np.sin, 0.5) - np.cos(0.5)) < 1e-8
    assert abs(numerical_derivative(np.exp, 1.0) - np.e) < 1e-6
    assert abs(numerical_derivative(np.log, 2.0) - 0.5) < 1e-8
    s = sigmoid(0.3)
    assert abs(numerical_derivative(sigmoid, 0.3) - s * (1 - s)) < 1e-9


def test_central_beats_forward():
    f, x, h = np.exp, 1.0, 1e-4
    central = abs(numerical_derivative(f, x, h) - np.e)
    forward = abs((f(x + h) - f(x)) / h - np.e)
    assert central < forward / 100


def test_numerical_gradient_matches_analytic():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 3))
    x = rng.normal(size=3)
    f = lambda v: v @ A @ v                       # f(x) = xᵀAx，梯度 (A + Aᵀ)x
    g = numerical_gradient(f, x)
    assert g.shape == x.shape
    assert rel_error(g, (A + A.T) @ x) < 1e-8
    W = rng.normal(size=(2, 4))
    gW = numerical_gradient(lambda M: (M ** 2).sum(), W)     # 梯度 2W，形状与 W 相同
    assert gW.shape == (2, 4) and rel_error(gW, 2 * W) < 1e-8


def test_numerical_gradient_does_not_modify_input():
    x = np.array([1.0, 2.0])
    numerical_gradient(lambda v: (v ** 3).sum(), x)
    assert x.tolist() == [1.0, 2.0]


def test_rel_error():
    assert rel_error([1.0, 2.0], [1.0, 2.0]) == 0.0
    assert rel_error([0.0], [0.0]) == 0.0
    assert abs(rel_error([1.0], [1.1]) - 0.1 / 2.1) < 1e-12
