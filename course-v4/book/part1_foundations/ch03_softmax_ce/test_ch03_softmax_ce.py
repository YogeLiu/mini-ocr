"""第 3 章的自动检查：python -m pytest book/part1_foundations/ch03_softmax_ce"""
import math

import torch
import torch.nn.functional as F

from miniocr.nn.losses import (cross_entropy, log_softmax, numerical_gradient, rel_error,
                               sequence_mean_cross_entropy, softmax, softmax_ce_grad, softmax_loss_vectorized)


# ---------------------------------------------------------------- 练习 3.1、3.2
def test_softmax_hand_example():
    """3.2 节的天气例子：z = [2, 1, 0.1] → p = [0.6590, 0.2424, 0.0986]。"""
    p = softmax(torch.tensor([2.0, 1.0, 0.1]))
    assert torch.allclose(p, torch.tensor([0.6590, 0.2424, 0.0986]), atol=1e-4)
    assert math.isclose(p.sum().item(), 1.0, rel_tol=1e-6)


def test_softmax_matches_torch_any_dim():
    torch.manual_seed(0)
    z = torch.randn(3, 4, 5) * 5
    for dim in (0, 1, 2, -1):
        assert torch.allclose(softmax(z, dim), torch.softmax(z, dim), atol=1e-6)
        assert torch.allclose(log_softmax(z, dim), torch.log_softmax(z, dim), atol=1e-5)


def test_softmax_is_stable_fp32_and_fp16():
    """不减最大值时这些输入会得到 nan（3.3 节）。"""
    z = torch.tensor([1000.0, 999.0, 998.0])
    p = softmax(z)
    assert torch.isfinite(p).all() and torch.allclose(p, torch.tensor([0.6652, 0.2447, 0.0900]), atol=1e-4)
    assert torch.isfinite(softmax(-z)).all()                                    # 全部很负：不能 0/0
    z16 = torch.tensor([12.0, 10.0, 8.0], dtype=torch.float16)
    p16 = softmax(z16)
    assert p16.dtype == torch.float16 and torch.isfinite(p16).all()
    assert torch.allclose(p16.float(), torch.tensor([0.8668, 0.1173, 0.0159]), atol=2e-3)


def test_softmax_subtracts_max_per_row():
    """每一行（沿归一化的那一维）各减各的最大值；用全局 z.max() 时第二行会全部下溢成 0/0（D1）。"""
    z = torch.tensor([[1000.0, 999.0], [0.0, -1.0]])
    ref = torch.tensor([1.0, 0.0]).softmax(0).expand(2, 2)                  # 每行都是 softmax([0, −1]) = [0.7311, 0.2689]
    for dim, zz, rr in ((-1, z, ref), (1, z, ref), (0, z.T, ref.T)):
        p = softmax(zz, dim)
        assert torch.isfinite(p).all() and torch.allclose(p, rr, atol=1e-4)
        assert torch.allclose(p.sum(dim), torch.ones(2), atol=1e-6)
        lp = log_softmax(zz, dim)
        assert torch.isfinite(lp).all() and torch.allclose(lp, rr.log(), atol=1e-4)


def test_fp16_large_vocab_sum_does_not_overflow():
    """fp16 上限 65504：65536 个类别、全 0 logits 时，Σ e^{z−m} 在 fp16 里会溢出（3.3.4 节）。"""
    z = torch.zeros(1, 65536, dtype=torch.float16)
    p, lp = softmax(z), log_softmax(z)
    assert p.dtype == torch.float16 and lp.dtype == torch.float16
    assert torch.allclose(p.float(), torch.full((1, 65536), 1 / 65536), rtol=1e-3)
    assert torch.allclose(lp.float(), torch.full((1, 65536), -math.log(65536)), atol=1e-2)


def test_log_softmax_no_underflow():
    """softmax 下溢成 0 时，log(softmax) 是 −inf；log_softmax 必须给出有限值。"""
    lp = log_softmax(torch.tensor([0.0, -200.0]))
    assert torch.isfinite(lp).all()
    assert math.isclose(lp[1].item(), -200.0, rel_tol=1e-6)
    lp16 = log_softmax(torch.tensor([12.0, 10.0, 8.0], dtype=torch.float16))
    assert torch.isfinite(lp16).all()


# ---------------------------------------------------------------- 练习 3.3
def test_cross_entropy_2d_matches_torch():
    torch.manual_seed(0)
    z, y = torch.randn(7, 5), torch.randint(0, 5, (7,))
    assert torch.allclose(cross_entropy(z, y), F.cross_entropy(z, y), atol=1e-6)
    assert torch.allclose(cross_entropy(z, y, reduction="sum"), F.cross_entropy(z, y, reduction="sum"), atol=1e-5)
    assert torch.allclose(cross_entropy(z, y, reduction="none"), F.cross_entropy(z, y, reduction="none"), atol=1e-6)


def test_cross_entropy_btv_with_ignore_index():
    torch.manual_seed(0)
    B, T, V = 3, 6, 11
    logits = torch.randn(B, T, V) * 3
    y = torch.randint(0, V, (B, T))
    y[0, 4:] = -100
    y[2, 1:] = -100
    ref = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), ignore_index=-100)
    assert torch.allclose(cross_entropy(logits, y), ref, atol=1e-6)
    none = cross_entropy(logits, y, reduction="none")
    assert none.shape == (B, T) and (none[y == -100] == 0).all()
    # 自定义的 ignore_index（例如用 pad 的 id 0）
    y0 = torch.where(y == -100, torch.zeros_like(y), y)
    assert torch.allclose(cross_entropy(logits, y0, ignore_index=0),
                          F.cross_entropy(logits.reshape(-1, V), y0.reshape(-1), ignore_index=0), atol=1e-6)


def test_cross_entropy_ignored_positions_get_zero_grad():
    """改动补齐位置的 logits，loss 不变；补齐位置的梯度精确为 0。"""
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 5, requires_grad=True)
    y = torch.tensor([[1, 4, 2, 0], [3, 3, -100, -100]])
    cross_entropy(logits, y).backward()
    assert (logits.grad[1, 2:] == 0).all()
    l2 = logits.detach().clone(); l2[1, 2:] = 1e4
    assert torch.allclose(cross_entropy(l2, y), cross_entropy(logits.detach(), y))


def test_cross_entropy_large_logits_finite():
    z = torch.tensor([[1000.0, 0.0, -1000.0]])
    assert torch.isfinite(cross_entropy(z, torch.tensor([2])))
    assert math.isclose(cross_entropy(z, torch.tensor([2])).item(), 2000.0, rel_tol=1e-6)


def test_cross_entropy_all_ignored():
    """全部位置被忽略：mean → nan（0/0），sum → 0，none → 全 0，都和 PyTorch 一致；梯度全为 0。"""
    z = torch.randn(3, 4, requires_grad=True)
    y = torch.full((3,), -100)
    loss = cross_entropy(z, y)
    assert torch.isnan(loss) and torch.isnan(F.cross_entropy(z, y))
    assert cross_entropy(z, y, reduction="sum").item() == 0.0
    assert torch.equal(cross_entropy(z, y, reduction="none"), torch.zeros(3))
    loss.backward()
    assert (z.grad == 0).all()


def test_initial_loss_is_ln_c():
    torch.manual_seed(0)
    for C in (3, 10, 100):
        logits = torch.zeros(64, C)
        assert math.isclose(cross_entropy(logits, torch.randint(0, C, (64,))).item(), math.log(C), rel_tol=1e-6)


# ---------------------------------------------------------------- 练习 3.4
def test_sequence_mean_vs_token_mean():
    """3.7 节的例子：甲 8 个 token、每个 loss 0.5；乙 2 个 token、每个 loss 3.0。"""
    V = 4
    def logits_for(loss):                       # 正确类别 0 的 −ln p 恰好等于 loss
        p0 = math.exp(-loss)
        rest = (1 - p0) / (V - 1)
        return torch.tensor([math.log(p0)] + [math.log(rest)] * (V - 1))
    logits = torch.zeros(2, 8, V)
    y = torch.full((2, 8), -100)
    logits[0, :8] = logits_for(0.5); y[0, :8] = 0
    logits[1, :2] = logits_for(3.0); y[1, :2] = 0
    assert math.isclose(cross_entropy(logits, y).item(), 1.0, rel_tol=1e-5)
    assert math.isclose(sequence_mean_cross_entropy(logits, y).item(), 1.75, rel_tol=1e-5)


def test_sequence_mean_skips_empty_sequences():
    torch.manual_seed(0)
    logits = torch.randn(3, 4, 5)
    y = torch.randint(0, 5, (3, 4))
    y[1] = -100
    per = [F.cross_entropy(logits[b], y[b]) for b in (0, 2)]
    assert torch.allclose(sequence_mean_cross_entropy(logits, y), torch.stack(per).mean(), atol=1e-6)


# ---------------------------------------------------------------- 练习 3.5
def test_softmax_ce_grad_hand_example():
    """3.6 节：z = [2, 1, 0.1]，y = 0 → p − y = [−0.3410, 0.2424, 0.0986]。"""
    g = softmax_ce_grad(torch.tensor([[2.0, 1.0, 0.1]]), torch.tensor([0]))
    assert torch.allclose(g, torch.tensor([[-0.3410, 0.2424, 0.0986]]), atol=1e-4)


def test_softmax_ce_grad_matches_autograd():
    torch.manual_seed(0)
    z, y = torch.randn(6, 4), torch.randint(0, 4, (6,))
    zt = z.clone().requires_grad_()
    F.cross_entropy(zt, y).backward()
    before = z.clone()
    g = softmax_ce_grad(z, y)
    assert torch.allclose(g, zt.grad, atol=1e-6)
    assert torch.allclose(g.sum(1), torch.zeros(6), atol=1e-6)                  # 每一行和为 0
    assert torch.equal(z, before)                                              # 不能原地改 logits


# ---------------------------------------------------------------- 练习 3.6
def test_softmax_loss_vectorized_matches_autograd():
    torch.manual_seed(0)
    N, D, C, reg = 20, 6, 4, 0.05
    W = torch.randn(D, C, dtype=torch.float64) * 0.1
    X = torch.randn(N, D, dtype=torch.float64)
    y = torch.randint(0, C, (N,))
    loss, dW = softmax_loss_vectorized(W, X, y, reg)
    Wt = W.clone().requires_grad_()
    ref = F.cross_entropy(X @ Wt, y) + reg * (Wt ** 2).sum()
    ref.backward()
    assert math.isclose(loss.item(), ref.item(), rel_tol=1e-10)
    assert dW.shape == (D, C) and torch.allclose(dW, Wt.grad, atol=1e-10)


def test_softmax_loss_initial_value():
    torch.manual_seed(0)
    W = torch.randn(10, 7) * 1e-4
    loss, _ = softmax_loss_vectorized(W, torch.randn(500, 10), torch.randint(0, 7, (500,)), 0.0)
    assert abs(loss.item() - math.log(7)) < 1e-2


# ---------------------------------------------------------------- 练习 3.7
def test_numerical_gradient_simple():
    x = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)
    g = numerical_gradient(lambda t: (t ** 3).sum(), x)
    assert torch.allclose(g, 3 * x ** 2, atol=1e-6)
    assert torch.equal(x, torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64))   # 用完要恢复原值


def test_gradient_check_softmax_loss():
    torch.manual_seed(0)
    W = torch.randn(5, 3, dtype=torch.float64) * 0.01
    X, y = torch.randn(30, 5, dtype=torch.float64), torch.randint(0, 3, (30,))
    _, dW = softmax_loss_vectorized(W, X, y, 0.1)
    num = numerical_gradient(lambda w: softmax_loss_vectorized(w, X, y, 0.1)[0], W.clone())
    assert rel_error(dW, num) < 1e-6


# ---------------------------------------------------------------- 正文的数字
def test_chapter_tour_numbers():
    """tour 第 6 部分的例子：按 token 平均 1.9538，按序列平均 1.7404。"""
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 4, 5, generator=g)
    y = torch.tensor([[1, 4, 2, 0], [3, 3, -100, -100]])
    assert math.isclose(cross_entropy(logits, y).item(), 1.9538, abs_tol=1e-4)
    assert math.isclose(sequence_mean_cross_entropy(logits, y).item(), 1.7404, abs_tol=1e-4)
