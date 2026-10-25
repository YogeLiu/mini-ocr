"""第 6 章的自动检查。"""
import numpy as np

from miniocr.calculus import numerical_gradient, rel_error
from miniocr.prob import accuracy, cross_entropy, entropy, kl_divergence, log_softmax, softmax
from miniocr.utils import one_hot


def test_softmax_basic_and_stable():
    p = softmax([1.0, 2.0, 3.0])
    assert np.isclose(p.sum(), 1) and (np.diff(p) > 0).all()
    assert np.allclose(p, np.exp([1, 2, 3]) / np.exp([1, 2, 3]).sum())
    assert np.allclose(softmax([1000.0, 1001.0]), softmax([0.0, 1.0]))      # 不溢出、平移不变
    P = softmax(np.random.default_rng(0).normal(size=(4, 5, 6)))
    assert P.shape == (4, 5, 6) and np.allclose(P.sum(-1), 1)


def test_log_softmax():
    z = np.array([[1.0, 2.0, 3.0], [-1000.0, 0.0, 1000.0]])
    lp = log_softmax(z)
    assert np.allclose(lp[0], np.log(softmax(z[0])))
    assert np.isfinite(lp).all() and np.isclose(lp[1, 0], -2000.0)           # np.log(softmax) 会得到 -inf


def test_cross_entropy_values():
    K = 62
    assert np.isclose(cross_entropy(np.zeros((3, K)), [0, 5, 61]), np.log(K))   # 均匀猜测：ln K
    z = np.log(np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]]))
    assert np.isclose(cross_entropy(z, [0, 2]), -(np.log(0.7) + np.log(0.8)) / 2)


def test_cross_entropy_gradient_is_p_minus_y():
    rng = np.random.default_rng(1)
    z, y = rng.normal(size=(4, 5)), np.array([0, 3, 1, 4])
    g = numerical_gradient(lambda t: cross_entropy(t, y), z)
    assert rel_error(g, (softmax(z) - one_hot(y, 5)) / 4) < 1e-7


def test_entropy_and_kl():
    assert np.isclose(entropy([0.5, 0.5]), np.log(2))
    assert np.isclose(entropy([1.0, 0.0]), 0.0)
    assert np.isclose(entropy(np.full(62, 1 / 62)), np.log(62))
    p, q = np.array([0.7, 0.2, 0.1]), np.array([0.4, 0.4, 0.2])
    assert kl_divergence(p, p) == 0 and kl_divergence(p, q) > 0
    assert not np.isclose(kl_divergence(p, q), kl_divergence(q, p))           # 不对称
    assert np.isclose(-(p * np.log(q)).sum(), entropy(p) + kl_divergence(p, q))  # 交叉熵 = 熵 + KL
    assert np.isinf(kl_divergence([0.5, 0.5], [1.0, 0.0]))


def test_accuracy():
    assert accuracy([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]], [1, 0, 0]) == 2 / 3
