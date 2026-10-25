"""第 11 章的自动检查：自己写的优化器、调度、裁剪、归一化层，与 PyTorch 逐步对拍。"""
import copy
import math

import torch
import torch.nn as nn

from miniocr.nn.layers import MyBatchNorm1d, MyLayerNorm
from miniocr.train.optim import SGD, Adam, AdamW, clip_grad_norm, warmup_cosine


def _pair(seed=0):
    torch.manual_seed(seed)
    a = nn.Sequential(nn.Linear(5, 7), nn.Tanh(), nn.Linear(7, 3))
    return a, copy.deepcopy(a)


def _run(model, opt, steps=6):
    g = torch.Generator().manual_seed(1)
    for _ in range(steps):
        x, y = torch.randn(8, 5, generator=g), torch.randint(0, 3, (8,), generator=g)
        opt.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        opt.step()


def _same(a, b):
    return all(torch.allclose(p, q, atol=1e-6) for p, q in zip(a.parameters(), b.parameters()))


def test_sgd_matches_torch():
    for kw in (dict(), dict(momentum=0.9), dict(momentum=0.9, weight_decay=0.01)):
        a, b = _pair()
        _run(a, SGD(a.parameters(), lr=0.1, **kw))
        _run(b, torch.optim.SGD(b.parameters(), lr=0.1, **kw))
        assert _same(a, b), kw


def test_adam_and_adamw_match_torch():
    a, b = _pair()
    _run(a, Adam(a.parameters(), lr=0.01))
    _run(b, torch.optim.Adam(b.parameters(), lr=0.01))
    assert _same(a, b)
    a, b = _pair()
    _run(a, Adam(a.parameters(), lr=0.01, weight_decay=0.1))
    _run(b, torch.optim.Adam(b.parameters(), lr=0.01, weight_decay=0.1))
    assert _same(a, b)
    a, b = _pair()
    _run(a, AdamW(a.parameters(), lr=0.01, weight_decay=0.1))
    _run(b, torch.optim.AdamW(b.parameters(), lr=0.01, weight_decay=0.1))
    assert _same(a, b)


def test_adam_first_step_is_lr_times_sign():
    """偏差修正后，第一步的更新量是 lr · g/(|g| + eps) ≈ lr · sign(g)，与梯度大小无关。"""
    p = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
    p.grad = torch.tensor([1e-3, -5.0, 200.0])
    Adam([p], lr=0.1).step()
    assert torch.allclose(p.detach(), torch.tensor([0.9, 1.1, 0.9]), atol=1e-4)


def test_warmup_cosine():
    lrs = [warmup_cosine(s, 100, 10, 1.0, 0.1) for s in range(100)]
    assert math.isclose(lrs[0], 0.1) and math.isclose(lrs[9], 1.0)
    assert math.isclose(lrs[10], 1.0) and math.isclose(lrs[55], 0.55, abs_tol=1e-9)
    assert all(x >= y for x, y in zip(lrs[10:], lrs[11:])) and lrs[-1] > 0.1
    assert math.isclose(warmup_cosine(100, 100, 10, 1.0, 0.1), 0.1)


def test_clip_grad_norm_matches_torch():
    a, b = _pair()
    for m in (a, b):
        m.zero_grad()
        (m(torch.ones(4, 5)) * 100).sum().backward()
    n1 = clip_grad_norm(list(a.parameters()), 1.0)
    n2 = torch.nn.utils.clip_grad_norm_(b.parameters(), 1.0).item()
    assert math.isclose(n1, n2, rel_tol=1e-5)
    assert all(torch.allclose(p.grad, q.grad, atol=1e-6) for p, q in zip(a.parameters(), b.parameters()))
    total = math.sqrt(sum((p.grad ** 2).sum().item() for p in a.parameters()))
    assert math.isclose(total, 1.0, rel_tol=1e-4)
    assert clip_grad_norm(list(a.parameters()), 10.0) < 10.0          # 不超过就不变


def test_layernorm_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 6) * 3 + 1
    mine, ref = MyLayerNorm(6), nn.LayerNorm(6)
    with torch.no_grad():
        for m in (mine, ref):
            m.weight.uniform_(0.5, 1.5)
            m.bias.uniform_(-1, 1)
        ref.weight.copy_(mine.weight)
        ref.bias.copy_(mine.bias)
    assert torch.allclose(mine(x), ref(x), atol=1e-5)


def test_batchnorm_matches_torch_train_and_eval():
    torch.manual_seed(0)
    mine, ref = MyBatchNorm1d(4), nn.BatchNorm1d(4)
    for _ in range(3):
        x = torch.randn(16, 4) * 2 + 3
        assert torch.allclose(mine(x), ref(x), atol=1e-5)
    assert torch.allclose(mine.running_mean, ref.running_mean, atol=1e-5)
    assert torch.allclose(mine.running_var, ref.running_var, atol=1e-5)
    assert set(mine.state_dict()) >= {"weight", "bias", "running_mean", "running_var"}
    assert len(list(mine.parameters())) == 2
    mine.eval(); ref.eval()
    x = torch.randn(5, 4)
    assert torch.allclose(mine(x), ref(x), atol=1e-5)
    assert torch.allclose(mine(x[:1]), mine(x)[:1], atol=1e-6)       # 评测时与批里其他样本无关
