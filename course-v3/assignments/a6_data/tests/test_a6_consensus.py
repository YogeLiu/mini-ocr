"""A6 第二部分：NED、两两一致度、三元组共识、分档、多数票、加权投票、单参数 EM。"""
import numpy as np
import pytest

from ._impl import adapters as A


def test_ned():
    assert A.run_ned("kitten", "sitting") == pytest.approx(3 / 7)
    assert A.run_ned("", "") == 0.0 and A.run_ned("", "ab") == 1.0


def test_pairwise_and_triplet():
    A_ = A.run_pairwise_agreement(["abc", "abd", "abc"])
    assert np.allclose(A_, [[1, 2 / 3, 1], [2 / 3, 1, 2 / 3], [1, 2 / 3, 1]])
    med, score, group = A.run_triplet_consensus(["abcde", "abcdx", "abcxx", "zzzzz"], 3)
    assert group == (0, 1, 2) and med == 1 and score == pytest.approx(2.2 / 3)           # 0.8、0.6、0.8：1 号和另两个都最像
    med, _, _ = A.run_triplet_consensus(["abcde", "abcdx", "abcde", "zzzzz"], 3)
    assert med == 0                                                                       # 0 和 2 平局，取下标小的
    assert A.run_tier(score, 0.9, 0.6) == "medium" and A.run_tier(0.9, 0.9, 0.6) == "high" and A.run_tier(0.59, 0.9, 0.6) == "low"
    med, score, group = A.run_triplet_consensus(["ab", "cd"], 3)
    assert group == (0, 1) and score == 0.0


def test_triplet_beats_correlated_pair():
    # 两个专家犯同一个错，三个专家写对了：三元组选写对的那三个
    preds = ["x_{2}", "x^{2}", "x^{2}", "x_{2}", "x^{2}"]
    med, score, group = A.run_triplet_consensus(preds, 3)
    assert group == (1, 2, 4) and preds[med] == "x^{2}" and score == 1.0
    # 但只要写对的不到三个，三元组就会被"同一个错"带走
    preds = ["x_{2}", "x^{2}", "x^{3}", "x_{2}", "y^{2}"]
    med, score, group = A.run_triplet_consensus(preds, 3)
    assert preds[med] == "x_{2}"


def test_votes():
    assert A.run_majority_vote(["b", "a", "a", "b", "c"]) == "b"
    assert A.run_weighted_vote(["b", "a", "a"], [2.5, 1.0, 1.0]) == "b"
    assert A.run_weighted_vote(["b", "a", "a"], [2.0, 1.0, 1.0]) == "b"                  # 平局取先出现的
    assert A.run_weighted_vote(["b", "a", "a"], [1.0, 1.0, 1.0]) == "a"
    assert A.run_weighted_vote(["b", "a", "c"], [-1.0, 0.5, 0.2]) == "a"


def test_one_coin_em_hand():
    out = [["a", "a", "b"], ["c", "c", "c"], ["d", "e", "e"], ["f", "f", "g"]]
    labels, acc = A.run_one_coin_em(out, 1)
    assert labels == ["a", "c", "e", "f"]
    assert np.allclose(acc, [(3 + 1) / 6, (4 + 1) / 6, (2 + 1) / 6])


def test_one_coin_em_recovers_reliability():
    rng = np.random.default_rng(0)
    p = [0.9, 0.6, 0.6, 0.6, 0.85]
    truth = [f"t{i}" for i in range(400)]
    out = []
    for t in truth:
        row = [t if rng.random() < q else f"w{int(rng.integers(3))}" for q in p]
        row[2] = row[1] if rng.random() < 0.8 else row[2]                                   # 1、2 号专家高度相关
        out.append(row)
    labels, acc = A.run_one_coin_em(out, 10)
    assert np.argmax(acc) == 0 and acc[4] > acc[3]
    assert np.mean([l == t for l, t in zip(labels, truth)]) >= np.mean([A.run_majority_vote(r) == t for r, t in zip(out, truth)])
