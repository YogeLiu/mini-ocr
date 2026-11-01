"""第 5 章的自动检查：python -m pytest book/part1_foundations/ch05_generalization"""
import numpy as np
import pytest

from miniocr.data.symbols import (SYMBOLS, augment, average_hash, count_shared_groups, cross_duplicates,
                                  group_split, hamming_matrix, make_dataset, random_split, render_symbol)


@pytest.fixture(scope="module")
def small():
    return make_dataset(sources_per_class=5, per_source=3, classes=range(6), seed=0)


# ---------------------------------------------------------------- 数据本身（参考实现，不是练习）
def test_render_and_augment_shapes():
    m = render_symbol(r"\alpha", "cm")
    assert m.shape == (96, 96) and m.dtype == np.float32
    assert m.max() > 0.9 and m.min() >= 0.0
    img = augment(m, scale=1.1, angle=10, shift=(2, -1), thickness=1, contrast=0.8, background=0.1)
    assert img.shape == (32, 32) and 0.0 <= img.min() and img.max() <= 1.0
    # 平移 +4 像素以后，墨迹的重心应该向右移约 4 像素
    a, b = augment(m), augment(m, shift=(4, 0))
    cx = lambda x: (x.sum(0) * np.arange(32)).sum() / x.sum()
    assert 3.0 < cx(b) - cx(a) < 5.0


def test_make_dataset_groups(small):
    X, y, g = small["images"], small["labels"], small["groups"]
    assert X.shape == (6 * 5 * 3, 32, 32) and len(y) == len(g) == len(X)
    assert len(np.unique(g)) == 30
    for gid in np.unique(g):                       # 一个来源只属于一个类别，样本数等于 per_source
        assert len(np.unique(y[g == gid])) == 1 and (g == gid).sum() == 3
    again = make_dataset(sources_per_class=5, per_source=3, classes=range(6), seed=0)
    assert np.array_equal(again["images"], X)      # 固定种子 → 完全可复现


def test_random_split_leaks(small):
    tr, va, te = random_split(len(small["groups"]), (0.6, 0.2, 0.2), seed=0)
    assert len(tr) + len(va) + len(te) == len(small["groups"])
    g = small["groups"]
    assert np.isin(g[va], g[tr]).any()             # 随机划分会把同一来源拆到两边


def test_count_shared_groups_counts_each_group_once():
    g = np.array([0, 0, 0, 1, 1, 2, 3])
    tr, va, te = np.array([0, 3, 6]), np.array([1, 4]), np.array([2, 5])
    # 组 0 同时在三个集合（只算一次），组 1 在 train 和 val；组 2、3 各只在一个集合
    assert count_shared_groups(g, tr, va, te) == 2
    assert count_shared_groups(g, np.array([0, 3]), np.array([5, 6])) == 0


# ---------------------------------------------------------------- 练习 5.1：分组划分
def test_group_split_disjoint_and_complete(small):
    g = small["groups"]
    parts = group_split(g, (0.6, 0.2, 0.2), seed=0)
    assert len(parts) == 3
    allidx = np.concatenate(parts)
    assert sorted(allidx.tolist()) == list(range(len(g)))       # 不重不漏
    for p in parts:
        assert np.all(np.diff(p) > 0)                            # 升序
    sets = [set(g[p]) for p in parts]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert [len(s) for s in sets] == [18, 6, 6]                  # 按"组数"切：30 组 → 18 / 6 / 6


def test_group_split_seed_and_fractions():
    g = np.repeat(np.arange(10), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])   # 组的大小不同
    a = group_split(g, (8, 2), seed=1)                               # 比例会先归一化
    b = group_split(g, (0.8, 0.2), seed=1)
    assert all(np.array_equal(x, z) for x, z in zip(a, b))
    assert len(np.unique(g[a[0]])) == 8 and len(np.unique(g[a[1]])) == 2
    seen = {tuple(np.unique(g[group_split(g, (0.8, 0.2), seed=s)[1]])) for s in range(10)}
    assert len(seen) > 1                                             # 换种子会换划分
    s_str = np.array(["p1", "p1", "p2", "p3", "p3", "p3"])           # 组编号也可以是字符串
    tr, va = group_split(s_str, (2, 1), seed=0)
    assert not set(s_str[tr]) & set(s_str[va])


# ---------------------------------------------------------------- 练习 5.2：平均哈希
def test_average_hash_hand_example():
    img = np.zeros((4, 4), np.float32)
    img[:2, :2] = 1.0                                                # 左上角亮
    h = average_hash(img, hash_size=2)
    assert h.shape == (1, 4) and h.dtype == bool
    assert h[0].tolist() == [True, False, False, False]
    h4 = average_hash(np.stack([img, img.T[::-1]]), hash_size=4)
    assert h4.shape == (2, 16)
    hf = average_hash(img[:, ::-1], hash_size=2)                     # 直接传入负步长的翻转视图
    assert hf[0].tolist() == [False, True, False, False]


def test_average_hash_invariances():
    rng = np.random.default_rng(0)
    x = rng.random((3, 32, 32)).astype(np.float32)
    h = average_hash(x)
    assert np.array_equal(h, average_hash(0.5 * x + 0.2))           # 整体变亮、变淡，哈希不变
    assert (h != average_hash(1 - x)).all()                          # 反色，每一位都翻转（没有恰好等于平均值的格子）


# ---------------------------------------------------------------- 练习 5.3：跨集合近重复
def test_cross_duplicates_hand_example():
    A = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 0, 0]], bool)
    B = np.array([[1, 1, 1, 0], [0, 1, 0, 1]], bool)
    assert hamming_matrix(A, B).tolist() == [[3, 2], [1, 2], [1, 2]]
    nearest, dist, dup = cross_duplicates(A, B, max_dist=1)
    assert nearest.tolist() == [1, 0] and dist.tolist() == [1, 2] and dup.tolist() == [True, False]


def test_cross_duplicates_finds_same_source(small):
    g = small["groups"]
    H = average_hash(small["images"])
    tr, va = random_split(len(g), (0.7, 0.3), seed=0)
    nearest, dist, dup = cross_duplicates(H[tr], H[va], max_dist=2)
    leak = np.isin(g[va], g[tr])
    assert dup[leak].mean() > 0.5                                    # 同源泄漏的样本，大多能被找出来
    assert (g[tr][nearest][dup] == g[va][dup]).mean() > 0.5          # 找出来的大多确实同源
