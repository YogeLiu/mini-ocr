"""A4 第二部分：bootstrap、配对 bootstrap、McNemar、校准分段、ECE、AUROC。"""
import math

import numpy as np
import pytest

from ._impl import adapters as A


def test_bootstrap_ci_exact_procedure():
    rng = np.random.default_rng(3)
    num = rng.integers(0, 5, 40).astype(float)
    den = rng.integers(5, 20, 40).astype(float)
    est, lo, hi = A.run_bootstrap_ci(num, den, 500, 0.1, 7)
    idx = np.random.default_rng(7).integers(0, 40, size=(500, 40))                  # 说明书规定的抽样方式
    vals = num[idx].sum(1) / den[idx].sum(1)
    assert est == pytest.approx(num.sum() / den.sum())
    assert (lo, hi) == pytest.approx(tuple(np.quantile(vals, [0.05, 0.95])))


def test_bootstrap_ci_properties():
    small = np.random.default_rng(0).random(50) < 0.7
    big = np.random.default_rng(0).random(2000) < 0.7
    e1, lo1, hi1 = A.run_bootstrap_ci(small, None, 2000, 0.05, 0)
    e2, lo2, hi2 = A.run_bootstrap_ci(big, None, 2000, 0.05, 0)
    assert lo1 <= e1 <= hi1 and lo2 <= e2 <= hi2
    assert (hi2 - lo2) < (hi1 - lo1) / 3                                               # 样本多 40 倍，区间约窄 √40 倍
    w = 2 * 1.96 * math.sqrt(e2 * (1 - e2) / 2000)                                     # 正态近似
    assert hi2 - lo2 == pytest.approx(w, rel=0.15)


def test_paired_bootstrap():
    rng = np.random.default_rng(0)
    hard = rng.random(300) < 0.5                                                       # 样本难度差别很大
    a = (~hard) | (rng.random(300) < 0.3)
    b = a.copy()
    flip = np.flatnonzero(a)[:15]
    b[flip] = False                                                                    # b 比 a 少对 15 题，其余完全一样
    d, lo, hi, p_le = A.run_paired_bootstrap(a, b, None, 2000, 0.05, 0)
    assert d == pytest.approx(15 / 300)
    assert lo > 0 and p_le == 0.0                                                      # 配对：差别方向稳定
    _, lo_a, hi_a = A.run_bootstrap_ci(a, None, 2000, 0.05, 0)
    _, lo_b, hi_b = A.run_bootstrap_ci(b, None, 2000, 0.05, 0)
    assert lo_a < hi_b                                                                 # 不配对：两个区间重叠
    idx = np.random.default_rng(0).integers(0, 300, size=(2000, 300))
    diffs = (a[idx].sum(1) - b[idx].sum(1)) / 300.0
    assert (lo, hi) == pytest.approx(tuple(np.quantile(diffs, [0.025, 0.975])))


def test_mcnemar():
    a = [1] * 3 + [0] * 12 + [1] * 50 + [0] * 35
    b = [0] * 3 + [1] * 12 + [1] * 50 + [0] * 35
    b10, b01, p = A.run_mcnemar(a, b)
    assert (b10, b01) == (3, 12)
    assert p == pytest.approx(2 * sum(math.comb(15, i) for i in range(4)) / 2 ** 15)
    assert A.run_mcnemar([1, 0, 1], [1, 0, 1]) == (0, 0, 1.0)
    assert A.run_mcnemar([1] * 5 + [0] * 5, [0] * 5 + [1] * 5)[2] == pytest.approx(1.0)


def test_reliability_and_ece():
    conf = [0.05, 0.15, 0.95, 0.95, 1.0, 0.5]
    ok = [0, 0, 1, 0, 1, 1]
    bins = A.run_reliability(conf, ok, 10)
    assert [(round(b[0], 1), b[4]) for b in bins] == [(0.0, 1), (0.1, 1), (0.5, 1), (0.9, 3)]   # 1.0 落在最后一段
    last = bins[-1]
    assert last[2] == pytest.approx((0.95 + 0.95 + 1.0) / 3) and last[3] == pytest.approx(2 / 3)
    expect = (1 * 0.05 + 1 * 0.15 + 1 * 0.5 + 3 * abs(2 / 3 - 2.9 / 3)) / 6
    assert A.run_ece(conf, ok, 10) == pytest.approx(expect)
    assert A.run_ece([1.0] * 10, [1, 0] * 5, 10) == pytest.approx(0.5)


def test_auroc_bruteforce():
    rng = np.random.default_rng(0)
    s = rng.integers(0, 5, 60).astype(float)                                           # 大量并列
    y = rng.random(60) < 0.4
    pos, neg = s[y], s[~y]
    brute = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert A.run_auroc(s, y) == pytest.approx(brute)
    assert A.run_auroc([1, 2, 3], [0, 1, 1]) == pytest.approx(1.0)
    assert math.isnan(A.run_auroc([1, 2], [1, 1]))
