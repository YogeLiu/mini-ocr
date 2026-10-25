"""第 1 章的自动检查。"""
import numpy as np
import pytest

from labs.ch01.drills import char_histogram, most_common
from miniocr.utils import Vocab, minibatches, one_hot, pad_sequences, pairwise_sq_dists


def test_char_histogram():
    assert char_histogram(["ab", "b"]) == {"a": 1, "b": 2}
    assert char_histogram([]) == {}
    h = char_histogram(["x^{2}", "x_{1}"])
    assert h["x"] == 2 and h["{"] == 2 and h["^"] == 1
    assert most_common(h, 2) == [("x", 2), ("{", 2)]             # 次数相同按码位排序：x(120) < {(123) < }(125)


def test_vocab_roundtrip_and_order():
    v = Vocab("hello world", specials=("<pad>", "<eos>"))
    assert v.itos[:2] == ["<pad>", "<eos>"]
    assert v.itos[2:] == sorted(set("hello world"))          # 按码位排序：空格最小
    assert len(v) == 2 + len(set("hello world"))
    ids = v.encode("hold")
    assert v.decode(ids) == "hold"
    assert v.decode([0] + ids + [1]) == "hold"                # 特殊符号被跳过
    assert Vocab("dlrow olleh", ("<pad>", "<eos>")).itos == v.itos   # 与字符出现顺序无关
    with pytest.raises(KeyError):
        v.encode("z")


def test_one_hot():
    y = one_hot([2, 0, 1], 4)
    assert y.shape == (3, 4) and y.dtype == np.float32
    assert (y.sum(1) == 1).all() and y[0, 2] == 1 and y[1, 0] == 1 and y[2, 1] == 1


def test_pad_sequences():
    arr, mask = pad_sequences([[5, 6, 7], [8], []], pad=-1)
    assert arr.tolist() == [[5, 6, 7], [8, -1, -1], [-1, -1, -1]]
    assert mask.tolist() == [[True] * 3, [True, False, False], [False] * 3]
    assert mask.sum() == 4


def test_pairwise_sq_dists_matches_loops():
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(5, 3)), rng.normal(size=(4, 3))
    D = pairwise_sq_dists(A, B)
    assert D.shape == (5, 4)
    for i in range(5):
        for j in range(4):
            assert abs(D[i, j] - ((A[i] - B[j]) ** 2).sum()) < 1e-12


def test_minibatches():
    batches = list(minibatches(10, 4))
    assert [b.tolist() for b in batches] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert len(list(minibatches(10, 4, drop_last=True))) == 2
    shuffled = np.concatenate(list(minibatches(10, 3, np.random.default_rng(0))))
    assert sorted(shuffled.tolist()) == list(range(10)) and shuffled.tolist() != list(range(10))
