"""第 13 章的自动检查。"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from miniocr.ctc import collapse, ctc_brute_force, ctc_forward, greedy_decode, prefix_beam_search
from miniocr.metrics import cer, edit_distance
from miniocr.models.crnn import CRNN, CTCCollate, MyRNN, line_vocab


def test_collapse():
    assert collapse([1, 1, 0, 1, 2, 2]) == [1, 1, 2]
    assert collapse([0, 0, 0]) == [] and collapse([3]) == [3]
    assert collapse([1, 1, 1, 0, 0, 2, 0, 2]) == [1, 2, 2]


def test_ctc_forward_matches_brute_force_and_torch():
    rng = np.random.default_rng(0)
    for T, K, target in ((3, 3, [1, 2]), (4, 3, [1, 1]), (5, 4, [2, 3, 2]), (4, 3, [])):
        logits = rng.normal(size=(T, K))
        lp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
        p, _ = ctc_brute_force(np.exp(lp), target)
        assert math.isclose(ctc_forward(lp, target), -math.log(p), rel_tol=1e-9)
        if target:
            ref = F.ctc_loss(torch.from_numpy(lp)[:, None], torch.tensor([target]), torch.tensor([T]),
                             torch.tensor([len(target)]), reduction="sum")
            assert math.isclose(ctc_forward(lp, target), ref.item(), rel_tol=1e-6)
    lp = np.log(np.full((2, 3), 1 / 3))
    assert ctc_forward(lp, [1, 1]) == math.inf                     # "aa" 至少需要 3 步（a 空白 a）


def test_greedy_and_beam():
    lp = np.log(np.array([[0.1, 0.8, 0.1], [0.6, 0.3, 0.1], [0.1, 0.8, 0.1]]))
    assert greedy_decode(lp) == [1, 1]
    # 经典例子：最可能的单条路径是全空白（0.6·0.6 = 0.36），但输出 "a" 的总概率 0.64 更大
    lp = np.log(np.array([[0.6, 0.4], [0.6, 0.4]]))
    assert greedy_decode(lp) == []
    assert prefix_beam_search(lp, beam=4) == [1]
    rng = np.random.default_rng(1)
    for _ in range(5):                                             # 束足够宽时，束搜索的结果概率不低于贪心
        logits = rng.normal(size=(5, 3)) * 2
        lp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
        b, g = prefix_beam_search(lp, beam=16), greedy_decode(lp)
        pb, _ = ctc_brute_force(np.exp(lp), b)
        pg, _ = ctc_brute_force(np.exp(lp), g)
        assert pb >= pg - 1e-12


def test_edit_distance_and_cer():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3 and edit_distance("abc", "abc") == 0
    assert edit_distance([1, 2, 3], [1, 3]) == 1
    assert math.isclose(cer(["abd", "x"], ["abc", "xy"]), 2 / 5)


def test_my_rnn_matches_torch():
    torch.manual_seed(0)
    mine, ref = MyRNN(5, 7), torch.nn.RNN(5, 7)
    with torch.no_grad():
        for n, p in ref.named_parameters():
            p.copy_(getattr(mine, n))
    x, h0 = torch.randn(4, 3, 5), torch.randn(3, 7)
    o1, h1 = mine(x, h0)
    o2, h2 = ref(x, h0[None])
    assert torch.allclose(o1, o2, atol=1e-6) and torch.allclose(h1, h2[0], atol=1e-6)


def test_crnn_shapes_and_collate():
    v = line_vocab()
    assert v.itos[0] == "<blank>" and len(v) == 1 + 72
    m = CRNN(len(v))
    out = m(torch.zeros(2, 1, 32, 256))
    assert out.shape == (64, 2, len(v)) and torch.allclose(out.exp().sum(-1), torch.ones(64, 2))
    xs, tg, ln, texts = CTCCollate(v)([(torch.zeros(1, 32, 256), "ab"), (torch.zeros(1, 32, 256), "c 1")])
    assert xs.shape == (2, 1, 32, 256) and ln.tolist() == [2, 3] and tg.tolist() == v.encode("ab") + v.encode("c 1")
    assert CRNN(len(v), rnn="none")(torch.zeros(1, 1, 32, 128)).shape == (32, 1, len(v))
