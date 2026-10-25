"""第 7 章的自动检查：三个模型的梯度都用数值梯度验证。"""
import numpy as np

from miniocr.calculus import numerical_gradient, rel_error
from miniocr.data.tasks import CHARSET, glyph_arrays, shape_sizes
from miniocr.models.linear_np import logistic_loss_grad, mse_loss_grad, softmax_loss_grad, standardize


def _data(rng, N=20, D=4):
    return rng.normal(size=(N, D)), rng.normal(size=D), float(rng.normal())


def test_mse_grad():
    rng = np.random.default_rng(0)
    X, w, b = _data(rng)
    y = rng.normal(size=len(X))
    L, gw, gb = mse_loss_grad(X, y, w, b)
    assert np.isclose(L, ((X @ w + b - y) ** 2).mean())
    assert rel_error(gw, numerical_gradient(lambda v: mse_loss_grad(X, y, v, b)[0], w)) < 1e-7
    assert rel_error(gb, numerical_gradient(lambda v: mse_loss_grad(X, y, w, v[0])[0], [b])) < 1e-7


def test_logistic_grad_and_stability():
    rng = np.random.default_rng(1)
    X, w, b = _data(rng)
    y = (rng.random(len(X)) < 0.5).astype(float)
    L, gw, gb = logistic_loss_grad(X, y, w, b)
    p = 1 / (1 + np.exp(-(X @ w + b)))
    assert np.isclose(L, -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    assert rel_error(gw, numerical_gradient(lambda v: logistic_loss_grad(X, y, v, b)[0], w)) < 1e-7
    assert rel_error(gb, numerical_gradient(lambda v: logistic_loss_grad(X, y, w, v[0])[0], [b])) < 1e-7
    L_big, _, _ = logistic_loss_grad(X * 1000, y, w, b)                   # z 非常大时也不出现 inf/nan
    assert np.isfinite(L_big)


def test_softmax_grad():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(15, 4))
    W, b = rng.normal(size=(4, 3)), rng.normal(size=3)
    y = rng.integers(0, 3, 15)
    L, gW, gb = softmax_loss_grad(X, y, W, b)
    assert gW.shape == W.shape and gb.shape == b.shape
    assert rel_error(gW, numerical_gradient(lambda M: softmax_loss_grad(X, y, M, b)[0], W)) < 1e-7
    assert rel_error(gb, numerical_gradient(lambda v: softmax_loss_grad(X, y, W, v)[0], b)) < 1e-7
    assert np.isclose(softmax_loss_grad(X, y, W * 0, b * 0)[0], np.log(3))      # 全 0 参数：损失 ln K


def test_standardize_uses_train_stats():
    a, b = np.array([[0.0], [2.0]]), np.array([[4.0]])
    sa, sb = standardize(a, b)
    assert np.allclose(sa.ravel(), [-1, 1]) and np.allclose(sb.ravel(), [3])


def test_task_data():
    X, y = glyph_arrays(np.random.default_rng(0), chars="ab", n_per_class=3, size=16, align=True)
    assert X.shape == (6, 256) and X.dtype == np.float32 and y.tolist() == [0, 1, 0, 1, 0, 1]
    assert X.min() >= 0 and X.max() <= 1 and len(CHARSET) == 62
    area, side, kind = shape_sizes(np.random.default_rng(0), n=30)
    sq = kind == 1
    assert np.allclose(area[sq], side[sq] ** 2)                             # 正方形的面积就是边长的平方
