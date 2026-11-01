"""A2 第三部分：取批、变长样本的梯度累积、按窗口求总负对数似然。"""
import math
import os

import torch
import torch.nn.functional as F

from ._impl import FIXTURES
from ._impl import adapters as A


def _model():
    fx = torch.load(os.path.join(FIXTURES, "gpt.pt"), weights_only=False)
    m = A.get_gpt(fx["cfg"])
    m.load_state_dict(fx["state_dict"])
    return m


def test_get_batch():
    ids = torch.arange(100)
    x, y = A.run_get_batch(ids, 8, 16, torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (8, 16) and x.dtype == torch.long
    assert torch.equal(y, x + 1)                                        # 错开一位
    assert torch.equal(x[:, 1:], x[:, :-1] + 1)                         # 连续的一段
    x2, _ = A.run_get_batch(ids, 8, 16, torch.Generator().manual_seed(0))
    assert torch.equal(x, x2)
    starts = torch.cat([A.run_get_batch(ids, 64, 16, torch.Generator().manual_seed(s))[0][:, 0] for s in range(10)])
    assert starts.max() == 83 and starts.min() == 0                     # 最后一个合法起点也能取到


def test_accumulate_grads_variable_length_matches_full_batch():
    g = torch.Generator().manual_seed(1)
    x, y = torch.randint(0, 64, (4, 16), generator=g), torch.randint(0, 64, (4, 16), generator=g)
    for i, n in enumerate([2, 16, 16, 10]):                             # 有效 token 数不同
        y[i, n:] = -100
    m = _model()
    m.zero_grad()
    logits = m(x)[0]
    full_loss = F.cross_entropy(logits.reshape(-1, 64), y.reshape(-1), ignore_index=-100)
    full_loss.backward()
    want = {n: p.grad.clone() for n, p in m.named_parameters()}
    loss = A.run_accumulate_grads(m, [(x[:2], y[:2]), (x[2:], y[2:])])   # 必须先清零，否则会和上面的梯度叠加
    assert math.isclose(loss, full_loss.item(), rel_tol=1e-5)
    for n, p in m.named_parameters():
        torch.testing.assert_close(p.grad, want[n], atol=1e-6, rtol=1e-4, msg=n)


def _manual_nll(m, ids, context):
    total = 0.0
    with torch.no_grad():
        for s in range(0, len(ids) - 1, context):
            y = ids[s + 1:s + context + 1]
            x = ids[s:s + len(y)]
            total += F.cross_entropy(m(x[None])[0][0], y, reduction="sum").item()
    return total


def test_nll_sum():
    m = _model().eval()
    ids = torch.randint(0, 64, (40,), generator=torch.Generator().manual_seed(0))
    for context in (8, 16, 32):
        assert math.isclose(A.run_nll_sum(m, ids, context), _manual_nll(m, ids, context), rel_tol=1e-5)
    short = ids[:33]                                                     # 只有一个窗口：就是一次普通的前向
    with torch.no_grad():
        full = F.cross_entropy(m(short[:-1][None])[0][0], short[1:], reduction="sum").item()
    assert math.isclose(A.run_nll_sum(m, short, 32), full, rel_tol=1e-5)
