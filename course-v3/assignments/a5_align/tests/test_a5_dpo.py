"""A5 第二部分：最优策略、DPO 损失与它的梯度系数、成对数据、制造幻觉。"""
import numpy as np
import pytest
import torch

from miniocr.data.shapes import COLORS, SHAPES, SIZES, describe, sample_scene

from ._impl import adapters as A


def test_optimal_policy_maximizes_objective():
    g = torch.Generator().manual_seed(0)
    ref = torch.softmax(torch.randn(6, generator=g, dtype=torch.float64), -1)
    R = torch.rand(6, generator=g, dtype=torch.float64)
    beta = 0.3
    pi = A.run_optimal_policy(ref.log(), R, beta)
    assert pi.sum().item() == pytest.approx(1.0)

    def J(p):
        return float((p * R).sum() - beta * (p * (p / ref).log()).sum())

    best = J(pi)
    for _ in range(300):
        q = torch.softmax(pi.log() + 0.3 * torch.randn(6, generator=g, dtype=torch.float64), -1)
        assert J(q) <= best + 1e-12
    assert J(ref) < best and J(torch.nn.functional.one_hot(R.argmax(), 6).double() * 0.999 + 0.001 / 6) < best


def test_dpo_loss_value_and_stats():
    pw, pl = torch.tensor([-10.0, -5.0]), torch.tensor([-12.0, -4.0])
    rw, rl = torch.tensor([-11.0, -5.0]), torch.tensor([-11.0, -5.0])
    loss, st = A.run_dpo_loss(pw, pl, rw, rl, 0.5, 0.0, None)
    h = 0.5 * ((pw - rw) - (pl - rl))
    assert loss.item() == pytest.approx(float(-torch.nn.functional.logsigmoid(h).mean()))
    assert st["acc"] == pytest.approx(0.5) and st["margin"] == pytest.approx(float(h.mean()))
    assert st["reward_w"] == pytest.approx(0.5 * float((pw - rw).mean()))
    loss2, _ = A.run_dpo_loss(pw, pl, rw, rl, 0.5, 0.2, torch.tensor([4.0, 2.0]))
    assert loss2.item() == pytest.approx(loss.item() + 0.2 * float((-pw / torch.tensor([4.0, 2.0])).mean()))


def test_dpo_gradient_coefficient():
    beta = 0.1
    pw = torch.tensor([0.5, 3.0], requires_grad=True)                                   # ρ_w（ref 为 0）
    pl = torch.tensor([-0.2, -3.0], requires_grad=True)
    loss, _ = A.run_dpo_loss(pw, pl, torch.zeros(2), torch.zeros(2), beta, 0.0, None)
    loss.backward()
    gc = beta * torch.sigmoid(beta * (pl.detach() - pw.detach()))
    assert torch.allclose(-pw.grad * 2, gc, atol=1e-6)                                  # 平均：每条乘 1/B
    assert torch.allclose(pl.grad * 2, gc, atol=1e-6)
    assert gc[0].item() == pytest.approx(0.04825, abs=1e-4)                             # 第 32 章的手算


def test_dpo_loss_stable():
    big = torch.tensor([1e4])
    loss, _ = A.run_dpo_loss(big, -big, torch.zeros(1), torch.zeros(1), 1.0, 0.0, None)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    loss, _ = A.run_dpo_loss(-big, big, torch.zeros(1), torch.zeros(1), 1.0, 0.0, None)
    assert loss.item() == pytest.approx(2e4, rel=1e-6)


def test_pairs_from_groups():
    r = torch.tensor([0.2, 1.0, 1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.3, 0.1, 0.25, 0.9])
    assert A.run_pairs_from_groups(r, 4, 0.0) == [(1, 3), (11, 9)]                     # 并列取第一个；全相同的组没有
    assert A.run_pairs_from_groups(r, 4, 0.85) == [(1, 3)]


def _ref_hallucination(objs, rng, kind):
    present = {(o["color"], o["shape"]) for o in objs}
    absent = [(c, s) for c in COLORS for s in SHAPES if (c, s) not in present]
    c, s = absent[int(rng.integers(len(absent)))]
    if kind == "substitute":
        k = int(rng.integers(len(objs)))
        return describe([dict(o, color=c, shape=s) if i == k else o for i, o in enumerate(objs)]), (objs[k]["size"], c, s)
    z = list(SIZES)[int(rng.integers(len(SIZES)))]
    return describe(objs) + f", a {z} {c} {s}", (z, c, s)


@pytest.mark.parametrize("kind", ["substitute", "add"])
def test_hallucinated_caption(kind):
    for seed in range(20):
        objs = sample_scene(np.random.default_rng(seed), 64, 64, 1, 3)
        got = A.run_hallucinated_caption(objs, np.random.default_rng(100 + seed), kind)
        exp = _ref_hallucination(objs, np.random.default_rng(100 + seed), kind)
        assert got == exp
        _, (size, c, s) = got
        assert (c, s) not in {(o["color"], o["shape"]) for o in objs}
        assert got[0] != describe(objs)
