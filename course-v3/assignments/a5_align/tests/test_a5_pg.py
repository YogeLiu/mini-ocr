"""A5 第一部分：组内标准化与留一的优势、没有梯度的组、KL 的三个估计量、裁剪的目标。"""
import math

import pytest
import torch

from ._impl import adapters as A


def test_group_advantages():
    r = torch.tensor([1.0, 0, 0, 1, 0.5, 0.5, 0.5, 0.5])
    a = A.run_group_advantages(r, 4, True)
    s = math.sqrt(4 * 0.25 / 3)                                                          # 无偏标准差
    assert a[:4].tolist() == pytest.approx([0.5 / s, -0.5 / s, -0.5 / s, 0.5 / s], rel=1e-4)
    assert a[4:].tolist() == [0.0] * 4                                                   # 全相同：没有梯度
    assert A.run_group_advantages(r, 4, False)[:4].tolist() == [0.5, -0.5, -0.5, 0.5]


def test_loo_and_zero_var():
    r = torch.tensor([1.0, 0, 0, 1, 2, 2, 2, 2, 0, 0, 0, 3])
    a = A.run_loo_advantages(r, 4)
    assert a[:4].tolist() == pytest.approx([1 - 1 / 3, -2 / 3, -2 / 3, 1 - 1 / 3])
    assert a[8:].tolist() == pytest.approx([-1, -1, -1, 3])
    assert A.run_zero_variance_fraction(r, 4) == pytest.approx(1 / 3)


def test_kl_estimators_formulas():
    logp = torch.tensor([-1.0, -2.0, -0.5])
    ref = torch.tensor([-1.5, -1.0, -0.5])
    lr = ref - logp
    assert torch.allclose(A.run_kl_estimate(logp, ref, "k1"), -lr)
    assert torch.allclose(A.run_kl_estimate(logp, ref, "k2"), lr ** 2 / 2)
    assert torch.allclose(A.run_kl_estimate(logp, ref, "k3"), lr.exp() - 1 - lr)


def test_kl_estimators_unbiased():
    torch.manual_seed(0)
    p = torch.tensor([0.5, 0.3, 0.15, 0.05])
    q = torch.tensor([0.25, 0.25, 0.25, 0.25])
    kl = float((p * (p / q).log()).sum())
    x = torch.multinomial(p, 200000, replacement=True)
    logp, ref = p.log()[x], q.log()[x]
    k1, k3 = A.run_kl_estimate(logp, ref, "k1"), A.run_kl_estimate(logp, ref, "k3")
    assert k1.mean().item() == pytest.approx(kl, abs=0.01)
    assert k3.mean().item() == pytest.approx(kl, abs=0.01)
    assert (k3 >= 0).all() and (k1 < 0).any()                                            # k3 每个都 ≥ 0，k1 不是
    assert k3.var() < k1.var()


def test_clipped_objective_cases():
    old = torch.zeros(4)
    logp = torch.log(torch.tensor([1.5, 0.5, 1.5, 0.5]))
    adv = torch.tensor([1.0, 1.0, -1.0, -1.0])
    obj, clipped = A.run_clipped_objective(logp, old, adv, 0.2, None)
    assert obj.tolist() == pytest.approx([1.2, 0.5, -1.5, -0.8])
    assert clipped.tolist() == [True, False, False, True]
    obj, clipped = A.run_clipped_objective(logp, old, adv, 0.2, 0.28)                   # clip-higher
    assert obj[0].item() == pytest.approx(1.28)


def test_clipped_tokens_have_no_gradient():
    old = torch.zeros(4)
    logp = torch.log(torch.tensor([1.5, 0.5, 1.5, 0.5])).requires_grad_()
    adv = torch.tensor([1.0, 1.0, -1.0, -1.0])
    obj, clipped = A.run_clipped_objective(logp, old, adv, 0.2, None)
    obj.sum().backward()
    assert (logp.grad[clipped] == 0).all() and (logp.grad[~clipped] != 0).all()
    adv2 = torch.tensor([[1.0], [-1.0]])                                                # (B, 1) 广播到每个 token
    obj, _ = A.run_clipped_objective(torch.zeros(2, 3), torch.zeros(2, 3), adv2, 0.2, None)
    assert obj.tolist() == [[1.0] * 3, [-1.0] * 3]
