"""A5 第五部分：帕累托前沿、重放、RFT 的挑选、探测 KL。"""
import math

import numpy as np
import pytest
import torch

from ._impl import adapters as A


def test_pareto_front():
    pts = [(0.5, 0.5), (0.6, 0.4), (0.4, 0.6), (0.5, 0.4), (0.6, 0.4), (0.3, 0.3)]
    assert A.run_pareto_front(pts) == [0, 1, 2, 4]                                     # 重复的点互不支配


def test_mixed_batch():
    rng = np.random.default_rng(0)
    items, old = A.run_mixed_batch(rng, 10, 0.25, lambda r: ("new", int(r.integers(100))), lambda r: ("old", int(r.integers(100))))
    assert sum(old) == 2 and len(items) == 10                                            # round(2.5) = 2（银行家舍入）
    assert all((it[0] == "old") == o for it, o in zip(items, old))
    rng2 = np.random.default_rng(0)
    exp_old = [("old", int(rng2.integers(100))) for _ in range(2)]
    exp_new = [("new", int(rng2.integers(100))) for _ in range(8)]
    order = rng2.permutation(10)
    assert items == [(exp_old + exp_new)[i] for i in order]


def test_rft_select():
    r = torch.tensor([1.0, 0.8, 1.0, 0.5, 0.6, 0.9, 0.7, 0.4])
    assert A.run_rft_select(r, 4, 1.0, 1) == [0]
    assert A.run_rft_select(r, 4, 0.8, 2) == [0, 2, 5]


def test_probe_kl():
    pb = torch.tensor([0.7, 0.2, 0.1]).log()
    pn = torch.tensor([0.5, 0.3, 0.2]).log()
    lb = torch.stack([pb, pn])[None]                                                    # 第 2 个位置两者相同
    ln = torch.stack([pn, pn])[None]
    assert A.run_probe_kl(ln, lb, torch.tensor([[1.0, 0.0]])) == pytest.approx(0.0851, abs=1e-4)
    assert A.run_probe_kl(ln, lb, torch.tensor([[1.0, 1.0]])) == pytest.approx(0.0851 / 2, abs=1e-4)
    kl_rev = float((pn.exp() * (pn - pb)).sum())
    assert kl_rev == pytest.approx(0.0920, abs=1e-4) and not math.isclose(kl_rev, 0.0851, abs_tol=1e-3)
