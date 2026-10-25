"""A1 问题 adamw、schedule、clip。"""
import copy
import math

import torch
import torch.nn as nn

from ._impl import adapters as A


def _pair():
    torch.manual_seed(0)
    a = nn.Sequential(nn.Linear(5, 7), nn.Tanh(), nn.Linear(7, 3))
    return a, copy.deepcopy(a)


def _run(model, opt, steps=12):
    g = torch.Generator().manual_seed(1)
    for s in range(steps):
        for group in opt.param_groups:
            group["lr"] = 0.02 if s < 6 else 0.005                    # 中途改学习率：训练时调度就是这么做的
        x, y = torch.randn(8, 5, generator=g), torch.randint(0, 3, (8,), generator=g)
        opt.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        opt.step()


def test_adamw_matches_torch():
    AdamW = A.get_adamw_cls()
    assert issubclass(AdamW, torch.optim.Optimizer)
    for kw in (dict(weight_decay=0.1), dict(weight_decay=0.0, betas=(0.8, 0.95), eps=1e-6)):
        a, b = _pair()
        _run(a, AdamW(a.parameters(), lr=0.02, **kw))
        _run(b, torch.optim.AdamW(b.parameters(), lr=0.02, **kw))
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-5)


def test_adamw_state_roundtrip():
    AdamW = A.get_adamw_cls()
    a, b = _pair()
    oa = AdamW(a.parameters(), lr=0.02, weight_decay=0.1)
    _run(a, oa, steps=3)
    b.load_state_dict(a.state_dict())
    ob = AdamW(b.parameters(), lr=0.02, weight_decay=0.1)
    ob.load_state_dict(copy.deepcopy(oa.state_dict()))               # 优化器状态要能存、能恢复（深拷贝：模拟存盘）
    _run(a, oa, steps=3)
    _run(b, ob, steps=3)
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p, q)


def test_lr_cosine_schedule():
    f = lambda it: A.run_lr_cosine_schedule(it, 1.0, 0.1, 10, 30)
    assert f(0) == 0 and math.isclose(f(5), 0.5) and math.isclose(f(10), 1.0)
    assert math.isclose(f(20), 0.55)                                  # 余弦段的正中间：(max + min) / 2
    assert math.isclose(f(30), 0.1) and math.isclose(f(100), 0.1)
    vals = [f(i) for i in range(10, 31)]
    assert all(x >= y for x, y in zip(vals, vals[1:]))


def test_clip_grad_norm():
    a, b = _pair()
    for m in (a, b):
        m(torch.ones(2, 5)).sum().backward()
        m[0].bias.grad = None                                         # 没有梯度的参数要跳过
    pa, pb = list(a.parameters()), list(b.parameters())
    want = torch.nn.utils.clip_grad_norm_(pb, 0.01).item()
    got = A.run_clip_grad_norm(pa, 0.01)
    assert math.isclose(got, want, rel_tol=1e-5)
    for p, q in zip(pa, pb):
        if q.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-4)
    before = [p.grad.clone() for p in pa if p.grad is not None]
    A.run_clip_grad_norm(pa, 1e9)                                     # 没超过上限：不改
    for g0, p in zip(before, [p for p in pa if p.grad is not None]):
        assert torch.equal(g0, p.grad)
