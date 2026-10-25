"""第 30 章的自动检查。"""
import math

import numpy as np
import torch

from miniocr.data.prefs import hallucinated_caption
from miniocr.data.shapes import describe, sample_scene
from miniocr.rl.dpo import dpo_loss, optimal_policy, pairs_from_groups


def objective(pi, ref, R, beta):
    return (pi * R).sum() - beta * (pi * (pi.log() - ref.log())).sum()


def test_optimal_policy_maximizes_objective():
    g = torch.Generator().manual_seed(0)
    ref = torch.softmax(torch.randn(6, generator=g), 0)
    R = torch.rand(6, generator=g)
    for beta in (0.05, 0.5, 5.0):
        pi = optimal_policy(ref.log(), R, beta)
        assert abs(pi.sum() - 1) < 1e-6
        best = objective(pi, ref, R, beta)
        for _ in range(200):                                   # 随机扰动都不会更好
            other = torch.softmax(pi.log() + 0.3 * torch.randn(6, generator=g), 0)
            assert objective(other, ref, R, beta) <= best + 1e-6
        # 隐含奖励：β (log π* − log π_ref) = R − β log Z，任意两个回答的差等于奖励的差
        imp = beta * (pi.log() - ref.log())
        assert torch.allclose(imp - imp[0], R - R[0], atol=1e-4)
    assert torch.allclose(optimal_policy(ref.log(), R, 1e4), ref, atol=1e-3)      # β 很大：不动
    assert optimal_policy(ref.log(), R, 1e-3).argmax() == R.argmax()               # β 很小：全给奖励最高的


def test_dpo_loss_values_and_gradient():
    pw, pl = torch.tensor([-3.0, -5.0]), torch.tensor([-4.0, -2.0])
    loss, st = dpo_loss(pw, pl, pw.clone(), pl.clone(), beta=0.1)
    assert abs(loss.item() - math.log(2)) < 1e-6              # 策略 = 参考模型：每对的损失都是 log 2
    assert st["margin"] == 0 and st["acc"] == 0
    rw, rl, beta = torch.tensor([-2.0]), torch.tensor([-6.0]), 0.5
    pw = torch.tensor([-1.0], requires_grad=True)
    pl = torch.tensor([-7.0], requires_grad=True)
    loss, st = dpo_loss(pw, pl, rw, rl, beta)
    h = beta * ((-1 + 2) - (-7 + 6))                           # = 0.5 × (1 − (−1)) = 1
    assert abs(loss.item() - math.log(1 + math.exp(-h))) < 1e-6
    loss.backward()
    w = 1 / (1 + math.exp(h))                                  # σ(−h)：已经分对的对，梯度变小
    assert abs(pw.grad.item() + beta * w) < 1e-6 and abs(pl.grad.item() - beta * w) < 1e-6
    big, _ = dpo_loss(torch.tensor([-1000.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), 1.0)
    assert torch.isfinite(big) and abs(big.item() - 1000) < 1e-3
    l2, _ = dpo_loss(torch.tensor([-6.0]), torch.tensor([-7.0]), torch.tensor([-6.0]), torch.tensor([-7.0]), 0.1,
                     sft_weight=0.5, n_w=torch.tensor([3.0]))
    assert abs(l2.item() - (math.log(2) + 0.5 * 2.0)) < 1e-6


def test_pairs_from_groups():
    r = torch.tensor([0.2, 0.9, 0.5, 0.9, 1.0, 1.0, 1.0, 1.0, 0.0, 0.3, 0.1, 0.25])
    assert pairs_from_groups(r, 4) == [(1, 0), (9, 8)]        # 第 2 组全相同：没有成对数据；并列取第一个
    assert pairs_from_groups(r, 4, min_gap=0.5) == [(1, 0)]


def test_hallucinated_caption():
    rng = np.random.default_rng(0)
    for _ in range(50):
        objs = sample_scene(rng, 64, 64, 1, 4)
        good = describe(objs)
        present = {(o["color"], o["shape"]) for o in objs}
        bad, (z, c, s) = hallucinated_caption(objs, rng, "substitute")
        assert (c, s) not in present
        gw, bw = good.split(", "), bad.split(", ")
        assert len(gw) == len(bw) and sum(a != b for a, b in zip(gw, bw)) == 1
        assert f"a {z} {c} {s}" in bw
        bad2, (z2, c2, s2) = hallucinated_caption(objs, rng, "add")
        assert bad2 == good + f", a {z2} {c2} {s2}" and (c2, s2) not in present
    a = hallucinated_caption(objs, np.random.default_rng(5))
    b = hallucinated_caption(objs, np.random.default_rng(5))
    assert a == b
