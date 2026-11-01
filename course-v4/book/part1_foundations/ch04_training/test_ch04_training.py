"""第 4 章的自动检查：python -m pytest book/part1_foundations/ch04_training

练习 4.1–4.4 在 miniocr/train/optim.py，练习 4.5–4.6 在 miniocr/train/checkpoint.py。
"""
import math
import os
import random

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from miniocr.train.checkpoint import load_checkpoint, rng_state, save_checkpoint, set_rng_state
from miniocr.train.optim import SGD, Adam, AdamW, clip_grad_norm, warmup_cosine


def _pair(seed=0):
    torch.manual_seed(seed)
    a = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 3))
    b = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 3))
    b.load_state_dict(a.state_dict())
    return a, b


def _lockstep(mine_fn, ref_fn, steps=8):
    """两个一模一样的模型，分别用两个优化器在同样的数据上训练，返回每一步参数的最大差。"""
    a, b = _pair()
    oa, ob = mine_fn(a.parameters()), ref_fn(b.parameters())
    g = torch.Generator().manual_seed(1)
    worst = 0.0
    for _ in range(steps):
        x, y = torch.randn(16, 4, generator=g), torch.randint(0, 3, (16,), generator=g)
        for m, o in ((a, oa), (b, ob)):
            o.zero_grad()
            F.cross_entropy(m(x), y).backward()
            o.step()
        worst = max(worst, max((p - q).abs().max().item() for p, q in zip(a.parameters(), b.parameters())))
    return worst


# ---------------------------------------------------------------- 练习 4.1
def test_sgd_plain_hand_example():
    p = torch.tensor([-4.0, 1.0], dtype=torch.float64, requires_grad=True)
    opt = SGD([p], lr=0.015)
    for _ in range(2):
        opt.zero_grad()
        (0.5 * (p[0] ** 2 + 100 * p[1] ** 2)).backward()
        opt.step()
    assert torch.allclose(p.detach(), torch.tensor([-3.8809, 0.25], dtype=torch.float64))


def test_sgd_momentum_matches_torch():
    assert _lockstep(lambda ps: SGD(ps, lr=0.1, momentum=0.9),
                     lambda ps: torch.optim.SGD(ps, lr=0.1, momentum=0.9)) < 1e-6
    assert _lockstep(lambda ps: SGD(ps, lr=0.1, momentum=0.9, weight_decay=0.01),
                     lambda ps: torch.optim.SGD(ps, lr=0.1, momentum=0.9, weight_decay=0.01)) < 1e-6


def test_sgd_momentum_velocity_limit():
    p = torch.zeros(1, requires_grad=True)
    opt = SGD([p], lr=1.0, momentum=0.9)
    for _ in range(200):
        p.grad = torch.ones(1)
        opt.step()
    assert abs(opt.state[0].item() - 10.0) < 1e-3          # v → g / (1 − μ)


# ---------------------------------------------------------------- 练习 4.2
def test_adam_first_step_is_lr_times_sign():
    p = torch.tensor([-4.0, 1.0], requires_grad=True)
    opt = Adam([p], lr=0.3)
    p.grad = torch.tensor([-4.0, 100.0])
    opt.step()
    assert torch.allclose(p.detach(), torch.tensor([-3.7, 0.7]), atol=1e-6)


def test_adam_matches_torch():
    assert _lockstep(lambda ps: Adam(ps, lr=1e-2), lambda ps: torch.optim.Adam(ps, lr=1e-2)) < 1e-6
    assert _lockstep(lambda ps: Adam(ps, lr=1e-2, weight_decay=0.1),
                     lambda ps: torch.optim.Adam(ps, lr=1e-2, weight_decay=0.1)) < 1e-6


def test_adamw_matches_torch():
    assert _lockstep(lambda ps: AdamW(ps, lr=1e-2, weight_decay=0.1),
                     lambda ps: torch.optim.AdamW(ps, lr=1e-2, weight_decay=0.1), steps=30) < 1e-5


def test_adam_counts_steps_per_parameter():
    """两个参数轮流没有梯度：没有梯度的那一步，参数、m、v 和步数 t 都不动，也不做权重衰减。
    用一个全局计数器代替每个参数自己的 t，这里的偏差修正就会算错。"""
    for mine, ref in ((lambda ps: Adam(ps, lr=0.1), lambda ps: torch.optim.Adam(ps, lr=0.1)),
                      (lambda ps: AdamW(ps, lr=0.1, weight_decay=0.5),
                       lambda ps: torch.optim.AdamW(ps, lr=0.1, weight_decay=0.5))):
        a = [torch.tensor([1.0, -2.0], requires_grad=True), torch.tensor([0.5], requires_grad=True)]
        b = [x.detach().clone().requires_grad_() for x in a]
        oa, ob = mine(a), ref(b)
        g = torch.Generator().manual_seed(0)
        for t in range(10):
            grads = [torch.randn(2, generator=g), torch.randn(1, generator=g)]
            k = t % 2                                                       # 这一步第 k 个参数没有梯度
            before = a[k].detach().clone()
            for ps in (a, b):
                for j, p in enumerate(ps):
                    p.grad = None if j == k else grads[j].clone()
            oa.step(); ob.step()
            assert torch.equal(a[k].detach(), before)
            for p, q in zip(a, b):
                assert torch.allclose(p, q, atol=1e-6)
        assert oa.state[0][0] == 5 and oa.state[1][0] == 5                # 每个参数各自只走了 5 步


def test_adamw_decay_independent_of_gradient():
    """数据梯度为 0 的参数：AdamW 每步乘以 (1 − lr·λ)；Adam + L2 头几步每步大约走 lr，和 λ 无关。"""
    p = torch.ones(1, dtype=torch.float64, requires_grad=True)
    opt = AdamW([p], lr=0.01, weight_decay=0.1)
    for _ in range(100):
        p.grad = torch.zeros(1, dtype=torch.float64)
        opt.step()
    assert math.isclose(p.item(), 0.999 ** 100, rel_tol=1e-9)
    q = torch.ones(1, dtype=torch.float64, requires_grad=True)
    opt = Adam([q], lr=0.01, weight_decay=0.001)
    for _ in range(10):
        q.grad = torch.zeros(1, dtype=torch.float64)
        opt.step()
    assert abs(q.item() - (1 - 10 * 0.01)) < 1e-3


# ---------------------------------------------------------------- 练习 4.3
def test_warmup_cosine_values():
    T, W, base, lo = 1000, 100, 1e-3, 1e-4
    assert math.isclose(warmup_cosine(0, T, W, base, lo), 1e-5)
    assert math.isclose(warmup_cosine(99, T, W, base, lo), 1e-3)
    assert math.isclose(warmup_cosine(100, T, W, base, lo), 1e-3)
    assert math.isclose(warmup_cosine(550, T, W, base, lo), lo + (base - lo) * 0.5)
    assert math.isclose(warmup_cosine(1000, T, W, base, lo), lo)
    assert math.isclose(warmup_cosine(5000, T, W, base, lo), lo)          # 超过总步数后停在 min_lr
    lrs = [warmup_cosine(s, T, W, base, lo) for s in range(W, T)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))                       # 预热之后单调不增
    assert math.isclose(warmup_cosine(0, 10, 0, 0.5), 0.5)                 # warmup = 0 时第 0 步就是 base_lr
    assert [warmup_cosine(s, 10, 10, 1.0, 0.0) for s in (9, 10, 11)] == [1.0, 0.0, 0.0]   # 全程预热：用完计划后停在 min_lr


# ---------------------------------------------------------------- 练习 4.4
def test_clip_grad_norm_hand_example_and_torch():
    a, b = torch.zeros(2, requires_grad=True), torch.zeros(1, requires_grad=True)
    a.grad, b.grad = torch.tensor([3.0, 4.0]), torch.tensor([12.0])
    assert math.isclose(clip_grad_norm([a, b], 1.0), 13.0, rel_tol=1e-6)
    assert math.isclose(torch.cat([a.grad, b.grad]).norm().item(), 1.0, rel_tol=1e-5)
    a.grad, b.grad = torch.tensor([3.0, 4.0]), torch.tensor([12.0])
    clip_grad_norm([a, b], 20.0)
    assert a.grad.tolist() == [3.0, 4.0] and b.grad.tolist() == [12.0]   # 没超过就不动
    m1, m2 = _pair()
    x, y = torch.randn(32, 4), torch.randint(0, 3, (32,))
    for m in (m1, m2):
        F.cross_entropy(m(x), y).backward()
    m1[0].bias.grad = None                                                 # 没有梯度的参数要跳过
    m2[0].bias.grad = None
    n1 = clip_grad_norm(m1.parameters(), 0.01)
    n2 = torch.nn.utils.clip_grad_norm_([p for p in m2.parameters() if p.grad is not None], 0.01).item()
    assert math.isclose(n1, n2, rel_tol=1e-5)
    for p, q in zip(m1.parameters(), m2.parameters()):
        if p.grad is not None:
            assert torch.allclose(p.grad, q.grad, atol=1e-7)


def test_clip_grad_norm_without_any_grad():
    assert clip_grad_norm([], 1.0) == 0.0
    a, b = torch.zeros(2, requires_grad=True), torch.zeros(1, requires_grad=True)
    assert clip_grad_norm([a, b], 1.0) == 0.0                              # 全部 .grad is None
    assert a.grad is None and b.grad is None


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="需要 Apple 的 MPS 设备")
def test_clip_grad_norm_on_mps():
    """MPS 不支持 float64，裁剪要能在 MPS 上运行，结果和 CPU 上一致。"""
    a = torch.zeros(2, device="mps", requires_grad=True)
    a.grad = torch.tensor([3.0, 4.0], device="mps")
    assert math.isclose(clip_grad_norm([a], 1.0), 5.0, rel_tol=1e-6)
    assert math.isclose(a.grad.norm().item(), 1.0, rel_tol=1e-5)


# ---------------------------------------------------------------- 练习 4.5
def test_rng_state_roundtrip():
    g = torch.Generator().manual_seed(5)
    st = rng_state({"data": g})
    a = (random.random(), np.random.rand(), torch.rand(3), torch.rand(2, generator=g))
    for _ in range(3):                                                      # 把状态搅乱
        random.random(); np.random.rand(); torch.rand(5); torch.rand(5, generator=g)
    set_rng_state(st, {"data": g})
    b = (random.random(), np.random.rand(), torch.rand(3), torch.rand(2, generator=g))
    assert a[0] == b[0] and a[1] == b[1]
    assert torch.equal(a[2], b[2]) and torch.equal(a[3], b[3])


def test_rng_state_is_a_snapshot():
    st = rng_state()
    x = torch.rand(4)
    torch.rand(100)                                                         # 之后再抽，记录不能跟着变
    set_rng_state(st)
    assert torch.equal(torch.rand(4), x)


# ---------------------------------------------------------------- 练习 4.6
def _train(model, opt, g, a, b, total):
    for step in range(a, b):
        model.train()
        x = torch.randn(16, 4, generator=g)
        y = torch.randint(0, 3, (16,), generator=g)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.lr = warmup_cosine(step, total, 5, 1e-2)
        opt.step()


def _net(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Dropout(0.3), nn.Linear(16, 3))


def test_checkpoint_resume_is_bitwise_identical(tmp_path):
    total, cut = 40, 17
    ref = _net(0)
    opt = AdamW(ref.parameters(), lr=1e-2)
    g = torch.Generator().manual_seed(3)
    torch.manual_seed(11)
    _train(ref, opt, g, 0, total, total)

    m = _net(0)
    opt = AdamW(m.parameters(), lr=1e-2)
    g = torch.Generator().manual_seed(3)
    torch.manual_seed(11)
    _train(m, opt, g, 0, cut, total)
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, m, opt, step=cut, generators={"data": g}, extra={"note": "hi"})
    assert os.path.exists(path) and not os.path.exists(path + ".tmp")

    m2 = _net(123)                                                          # 不同的初始参数
    opt2 = AdamW(m2.parameters(), lr=1e-2)
    g2 = torch.Generator().manual_seed(999)
    torch.manual_seed(2024)
    step, extra = load_checkpoint(path, m2, opt2, generators={"data": g2})
    assert step == cut and extra == {"note": "hi"}
    _train(m2, opt2, g2, step, total, total)
    for p, q in zip(ref.parameters(), m2.parameters()):
        assert torch.equal(p, q)


def test_checkpoint_restores_optimizer_config(tmp_path):
    """用非默认配置训练、保存；加载到一个默认配置的优化器里，续训结果仍要逐位一致。"""
    total, cut = 20, 7
    mk = lambda ps: Adam(ps, lr=1e-2, betas=(0.8, 0.99), eps=1e-6, weight_decay=0.05, decoupled=True)
    for make, make_default in ((mk, lambda ps: Adam(ps)),
                               (lambda ps: SGD(ps, lr=1e-2, momentum=0.8, weight_decay=0.05),
                                lambda ps: SGD(ps, lr=1.0))):
        ref = _net(0)
        opt = make(ref.parameters())
        g = torch.Generator().manual_seed(3)
        torch.manual_seed(11)
        _train(ref, opt, g, 0, total, total)

        m = _net(0)
        opt = make(m.parameters())
        g = torch.Generator().manual_seed(3)
        torch.manual_seed(11)
        _train(m, opt, g, 0, cut, total)
        path = str(tmp_path / "cfg.pt")
        save_checkpoint(path, m, opt, step=cut, generators={"data": g})
        m2 = _net(1)
        opt2 = make_default(m2.parameters())
        g2 = torch.Generator()
        step, _ = load_checkpoint(path, m2, opt2, generators={"data": g2})
        _train(m2, opt2, g2, step, total, total)
        for p, q in zip(ref.parameters(), m2.parameters()):
            assert torch.equal(p, q)


def test_checkpoint_with_torch_optimizer(tmp_path):
    m = _net(0)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    x, y = torch.randn(8, 4), torch.randint(0, 3, (8,))
    F.cross_entropy(m(x), y).backward()
    opt.step()
    path = str(tmp_path / "c.pt")
    save_checkpoint(path, m, opt, step=1)
    m2 = _net(5)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-2)
    step, _ = load_checkpoint(path, m2, opt2)
    assert step == 1
    s1, s2 = opt.state_dict()["state"][0], opt2.state_dict()["state"][0]
    assert torch.equal(s1["exp_avg"], s2["exp_avg"]) and torch.equal(s1["exp_avg_sq"], s2["exp_avg_sq"])
    assert all(torch.equal(p, q) for p, q in zip(m.parameters(), m2.parameters()))
