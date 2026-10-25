"""A2 第二部分：CTC 损失（与 F.ctc_loss 对拍）、解码（与穷举对照）、LSTM（与 nn.LSTM 对拍）、CRNN、CER。"""
import itertools
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._impl import adapters as A


def _case(seed, T=12, N=4, C=6, S=5):
    g = torch.Generator().manual_seed(seed)
    lp = torch.randn(T, N, C, generator=g, dtype=torch.float64).log_softmax(-1)
    tl = torch.randint(1, S + 1, (N,), generator=g)
    tg = torch.randint(1, C, (N, S), generator=g)
    tg[0, :3] = torch.tensor([2, 2, 3])                                 # 一定包含相邻重复的标签
    tl[0] = max(int(tl[0]), 3)
    il = torch.randint(2 * S + 1, T + 1, (N,), generator=g)
    il[0] = T
    return lp, tg, il, tl


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_ctc_loss_matches_torch(reduction):
    for seed in range(3):
        lp, tg, il, tl = _case(seed)
        # 梯度要对 log_softmax 之前的 logits 比较：PyTorch 的 CTC 反向假定输入来自 log_softmax，
        # 直接返回对 logits 的梯度，单独看它对 log_probs 的"梯度"和真实的偏导数不同（说明书第 4.1 节）
        a = lp.clone().requires_grad_(True)
        b = lp.clone().requires_grad_(True)
        la = A.run_ctc_loss(a.log_softmax(-1), tg, il, tl, reduction)
        lb = F.ctc_loss(b.log_softmax(-1), tg, il, tl, blank=0, reduction=reduction)
        torch.testing.assert_close(la, lb)
        la.sum().backward()
        lb.sum().backward()
        torch.testing.assert_close(a.grad, b.grad)


def test_ctc_loss_impossible_is_inf():
    lp = torch.randn(3, 1, 4, dtype=torch.float64).log_softmax(-1)
    tg, il, tl = torch.tensor([[1, 1, 1]]), torch.tensor([3]), torch.tensor([3])   # 三个相同字符至少要 5 步
    assert math.isinf(A.run_ctc_loss(lp, tg, il, tl, "none")[0].item())
    assert math.isinf(F.ctc_loss(lp, tg, il, tl, reduction="none")[0].item())


def test_greedy():
    C = 4
    path = [0, 1, 1, 0, 1, 2, 2, 0, 0, 3]
    lp = torch.full((len(path), C), -5.0)
    lp[torch.arange(len(path)), torch.tensor(path)] = -0.1
    assert A.run_ctc_greedy(lp) == [1, 1, 2, 3]


def _brute_best(lp):
    """穷举所有路径，按折叠后的标签累加概率，返回概率最大的标签。"""
    T, C = lp.shape
    score = {}
    for path in itertools.product(range(C), repeat=T):
        lab, prev = [], None
        for p in path:
            if p != prev and p != 0:
                lab.append(p)
            prev = p
        score[tuple(lab)] = score.get(tuple(lab), 0.0) + math.exp(sum(lp[t, p].item() for t, p in enumerate(path)))
    return list(max(score, key=score.get))


def test_beam_search_is_exact_with_wide_beam():
    for seed in range(6):
        lp = (torch.randn(6, 3, generator=torch.Generator().manual_seed(seed), dtype=torch.float64) * 1.5).log_softmax(-1)
        assert A.run_ctc_beam(lp, 1000) == _brute_best(lp), seed


def test_beam_beats_greedy_on_classic_case():
    # 两步，空白概率 0.6、"a" 0.4：贪心得到空串（0.36），但"a"的总概率是 0.4·0.6·2 + 0.4·0.4 = 0.64
    lp = torch.tensor([[0.6, 0.4], [0.6, 0.4]], dtype=torch.float64).log()
    assert A.run_ctc_greedy(lp) == []
    assert A.run_ctc_beam(lp, 4) == [1]


def test_lstm_matches_torch():
    torch.manual_seed(0)
    ref = nn.LSTM(5, 7)
    w = {k[:-3]: v.detach().clone() for k, v in ref.state_dict().items()}   # weight_ih_l0 → weight_ih
    g = torch.Generator().manual_seed(1)
    x, h0, c0 = torch.randn(9, 3, 5, generator=g), torch.randn(3, 7, generator=g), torch.randn(3, 7, generator=g)
    out, (h, c) = A.run_lstm(5, 7, w, x, h0, c0)
    out2, (h2, c2) = ref(x, (h0[None], c0[None]))
    torch.testing.assert_close(out, out2)
    torch.testing.assert_close(h, h2[0])
    torch.testing.assert_close(c, c2[0])


def test_crnn_shapes_and_gradients():
    torch.manual_seed(0)
    m = A.get_crnn(75)
    for W in (256, 128):
        lp = m(torch.rand(2, 1, 32, W))
        assert lp.shape == (W // 4, 2, 75)
        torch.testing.assert_close(lp.exp().sum(-1), torch.ones(W // 4, 2), atol=1e-4, rtol=0)
    loss = A.run_ctc_loss(lp, torch.tensor([[1, 2, 3], [4, 5, 0]]), torch.tensor([32, 32]), torch.tensor([3, 2]), "mean")
    loss.backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None or not p.grad.abs().sum() > 0]
    assert not missing, f"这些参数没有得到梯度：{missing}"


def test_cer():
    assert math.isclose(A.run_cer(["kitten", "", "abc"], ["sitting", "ab", "abc"]), 5 / 12)
    assert A.run_cer(["same"], ["same"]) == 0
