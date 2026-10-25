"""A3 第二部分：CLIP / SigLIP 损失、召回率、零样本分类。对照直接按定义写的循环。"""
import math

import pytest
import torch

from ._impl import adapters as A


def _unit(x):
    return x / x.norm(dim=-1, keepdim=True)


def _data(B=6, D=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return _unit(torch.randn(B, D, generator=g, dtype=torch.float64)), _unit(torch.randn(B, D, generator=g, dtype=torch.float64))


def test_clip_loss_loop():
    zi, zt = _data()
    s = torch.tensor(7.0, dtype=torch.float64)
    B = len(zi)
    rows = cols = 0.0
    for i in range(B):
        li = [s.item() * float(zi[i] @ zt[j]) for j in range(B)]
        rows += -(li[i] - math.log(sum(math.exp(v) for v in li)))
        lt = [s.item() * float(zi[j] @ zt[i]) for j in range(B)]
        cols += -(lt[i] - math.log(sum(math.exp(v) for v in lt)))
    assert A.run_clip_loss(zi, zt, s).item() == pytest.approx((rows + cols) / (2 * B), rel=1e-6)


def test_siglip_loss_loop():
    zi, zt = _data()
    s, b = torch.tensor(10.0, dtype=torch.float64), torch.tensor(-3.0, dtype=torch.float64)
    B = len(zi)
    tot = 0.0
    for i in range(B):
        for j in range(B):
            y = 1 if i == j else -1
            z = y * (s.item() * float(zi[i] @ zt[j]) + b.item())
            tot += math.log1p(math.exp(-z))
    assert A.run_siglip_loss(zi, zt, s, b).item() == pytest.approx(tot / B, rel=1e-6)


def test_siglip_stable_and_grads():
    zi, zt = _data()
    s = torch.tensor(1000.0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(-50.0, dtype=torch.float64, requires_grad=True)
    loss = A.run_siglip_loss(zi, zt, s, b)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(s.grad) and torch.isfinite(b.grad)
    zi2, zt2 = _data()
    s2, b2 = s.detach().clone().requires_grad_(), b.detach().clone().requires_grad_()
    assert torch.autograd.gradcheck(lambda a, c: A.run_siglip_loss(zi2, zt2, a * 0.01, c * 0.1), (s2, b2))


def test_recall_at_k():
    sim = torch.tensor([[0.9, 0.1, 0.0], [0.8, 0.5, 0.1], [0.1, 0.2, 0.3]])
    assert A.run_recall_at_k(sim, 1) == pytest.approx(2 / 3)
    assert A.run_recall_at_k(sim, 2) == pytest.approx(1.0)
    same = torch.tensor([[True, False, False], [True, True, False], [False, False, True]])   # 第 1 行：第 0 列也算对
    assert A.run_recall_at_k(sim, 1, same) == pytest.approx(1.0)


def test_zero_shot():
    cls = _unit(torch.eye(4)[:3])
    zi = _unit(torch.tensor([[0.1, 0.9, 0.0, 0.2], [1.0, 0.0, 0.1, 0.0], [0.0, 0.2, 0.3, 0.9]]))
    assert A.run_zero_shot(zi, cls).tolist() == [1, 0, 2]
