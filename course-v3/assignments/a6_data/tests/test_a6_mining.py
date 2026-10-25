"""A6 第四部分：自洽度、公式特征、k-means、按簇配额、簇内抽样权重、难例候选。"""
import numpy as np
import pytest

from ._impl import adapters as A


def test_self_consistency():
    assert A.run_self_consistency(["a", "a", "b"]) == pytest.approx(1 / 3)
    assert A.run_self_consistency(["ab", "ac"]) == pytest.approx(0.5)
    assert A.run_self_consistency(["only"]) == 1.0


@pytest.mark.parametrize("s,f", [
    ("\\frac{y^{2}}{k}+3x", [14, 1, 0, 1, 2, 1, 1]),
    ("\\sqrt{\\frac{a}{b_{c}}}", [14, 1, 1, 1, 3, 0, 1]),
    ("x^{2}+y_{n}=0", [13, 0, 0, 2, 1, 2, 1]),
    ("\\frac{a", [3, 1, 0, 0, 1, 0, 0]),
    ("}}a{", [4, 0, 0, 0, 1, 0, 0])])
def test_formula_features(s, f):
    assert np.asarray(A.run_formula_features(s), dtype=float).tolist() == f


def test_kmeans_blobs():
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [10, 0], [0, 10]], dtype=float)
    X = np.concatenate([c + rng.normal(0, 0.5, (40, 2)) for c in centers])
    labels, C = A.run_kmeans(X, 3, np.random.default_rng(1))
    labels = np.asarray(labels)
    assert len(set(labels[:40])) == len(set(labels[40:80])) == len(set(labels[80:])) == 1
    assert len(set(labels.tolist())) == 3
    d = ((X[:, None] - np.asarray(C)[None]) ** 2).sum(-1)
    assert np.array_equal(d.argmin(1), labels)                                            # 每个点都分在最近的中心
    for j in range(3):                                                                    # 收敛：中心就是簇内的平均
        assert np.allclose(np.asarray(C)[j], X[labels == j].mean(0), atol=1e-6)
    l2, _ = A.run_kmeans(X, 3, np.random.default_rng(1))
    assert np.array_equal(labels, np.asarray(l2))


def test_quota_sample():
    labels = np.array([0] * 5 + [1] * 2 + [2] * 10)
    pick = np.asarray(A.run_quota_sample(labels, np.ones(17), 9, np.random.default_rng(0)))
    assert len(set(pick.tolist())) == 9
    assert [int((labels[pick] == c).sum()) for c in range(3)] == [4, 2, 3]                # 3/3/3 → 簇 1 只有 2 个 → 多的 1 个给簇 0
    assert list(labels[pick]) == sorted(labels[pick])                                      # 按簇编号排
    w = np.zeros(17)
    w[[0, 3, 5, 6, 12, 15]] = 1.0
    pick = np.asarray(A.run_quota_sample(labels, w, 6, np.random.default_rng(0)))
    assert set(pick.tolist()) == {0, 3, 5, 6, 12, 15}                                     # 每簇 2 个名额，正权重的正好 2 个
    pick = np.asarray(A.run_quota_sample(labels, np.zeros(17), 17, np.random.default_rng(0)))
    assert sorted(pick.tolist()) == list(range(17))


def test_quota_sample_follows_weights():
    labels = np.zeros(4, dtype=int)
    w = np.array([0.7, 0.1, 0.1, 0.1])
    first = [int(np.asarray(A.run_quota_sample(labels, w, 1, np.random.default_rng(s)))[0]) for s in range(2000)]
    assert np.mean(np.array(first) == 0) == pytest.approx(0.7, abs=0.04)


def test_sampling_weights():
    a = [0.9, 0.9, 0.5, 1.0, 0.2]
    u = [0.2, 0.8, 0.2, 0.5, 0.5]
    lab = [0, 0, 0, 1, 1]
    p = np.asarray(A.run_sampling_weights(a, u, lab, 1.0, 1.0, 1.0, 0.0))
    w = np.array([0.9 * 0.8, 0.9 * 0.2, 0.5 * 0.8])
    assert np.allclose(p[:3], w / w.sum()) and p[3:].sum() == pytest.approx(1.0)
    sharp = np.asarray(A.run_sampling_weights(a, u, lab, 1.0, 1.0, 0.25, 0.0))
    assert sharp[0] > p[0]
    only_a = np.asarray(A.run_sampling_weights(a, u, lab, 1.0, 0.0, 1.0, 0.0))
    assert np.allclose(only_a[:3], np.array([0.9, 0.9, 0.5]) / 2.3)


def test_hard_candidates():
    got = np.asarray(A.run_hard_candidates([0.95, 0.95, 0.5, 0.9], [0.5, 0.8, 0.1, 0.69], 0.9, 0.7))
    assert got.tolist() == [True, False, False, True]
