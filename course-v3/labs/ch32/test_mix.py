"""第 32 章的自动检查。"""
import numpy as np
import torch

from miniocr.train.mix import mixed_batch, pareto_front, probe_kl, rft_select


def test_pareto_front():
    pts = [(0.5, 0.2), (0.6, 0.1), (0.4, 0.3), (0.3, 0.1), (0.6, 0.2), (0.6, 0.2)]
    # (0.5, 0.2) 被 (0.6, 0.2) 支配；(0.6, 0.1) 被 (0.6, 0.2) 支配；(0.3, 0.1) 被很多点支配；两个相同的 (0.6, 0.2) 互不支配
    assert pareto_front(pts) == [2, 4, 5]
    assert pareto_front([(1, 1)]) == [0]


def test_mixed_batch():
    rng = np.random.default_rng(0)
    items, old = mixed_batch(rng, 10, 0.3, lambda r: "new", lambda r: "old")
    assert len(items) == 10 and sum(old) == 3 and all((it == "old") == o for it, o in zip(items, old))
    items, old = mixed_batch(rng, 8, 0.0, lambda r: "new", lambda r: "old")
    assert sum(old) == 0
    a = mixed_batch(np.random.default_rng(1), 16, 0.5, lambda r: r.integers(100), lambda r: -1)
    b = mixed_batch(np.random.default_rng(1), 16, 0.5, lambda r: r.integers(100), lambda r: -1)
    assert [int(x) for x in a[0]] == [int(x) for x in b[0]]


def test_rft_select():
    r = torch.tensor([1.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.9, 0.95, 0.2, 0.95])
    assert rft_select(r, 4) == [0]                               # 每组最多 1 个满分
    assert rft_select(r, 4, k=2) == [0, 2]
    assert rft_select(r, 4, threshold=0.9, k=2) == [0, 2, 9, 11]
    assert rft_select(r, 4, threshold=0.9, k=3) == [0, 2, 9, 11, 8]


def test_probe_kl():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(2, 5, 7, generator=g)
    m = torch.ones(2, 5)
    assert abs(probe_kl(a, a, m)) < 1e-7
    b = a + 0.5 * torch.randn(2, 5, 7, generator=g)
    k = probe_kl(b, a, m)
    pa, pb = a.softmax(-1), b.softmax(-1)
    assert k > 0 and abs(k - (pa * (pa.log() - pb.log())).sum(-1).mean().item()) < 1e-5
    m2 = torch.tensor([[1, 1, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.float)
    assert abs(probe_kl(b, a, m2) - (pa * (pa.log() - pb.log())).sum(-1)[0, :2].mean().item()) < 1e-5
