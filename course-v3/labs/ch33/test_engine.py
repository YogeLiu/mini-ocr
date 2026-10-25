"""第 33 章的自动检查。"""
import numpy as np

from miniocr.data.latex import to_latex
from miniocr.data.render import sample_formula
from miniocr.engine.clean import group_split, jaccard, latex_key, lsh_pairs, minhash, shingles
from miniocr.engine.consensus import pairwise_agreement, tier, triplet_consensus
from miniocr.engine.mining import quota_sample, self_consistency
from miniocr.engine.verify import render_score
from miniocr.eval.page import match_elements, page_score


def test_pairwise_and_triplet():
    A = pairwise_agreement(["abc", "abd", "abc"])
    assert np.allclose(A, [[1, 2 / 3, 1], [2 / 3, 1, 2 / 3], [1, 2 / 3, 1]])
    med, score, group = triplet_consensus(["hello", "hello", "hallo", "xyz"])
    assert group == (0, 1, 2) and med == 0                     # (1 + 0.8 + 0.8) / 3；0 和 1 平局，取下标小的
    assert abs(score - 2.6 / 3) < 1e-9 and tier(score) == "medium"
    med, score, group = triplet_consensus(["ab", "cd"])        # 少于 3 个专家：用全部
    assert group == (0, 1) and score == 0.0


def test_render_score():
    rng = np.random.default_rng(0)
    tree, img, _ = sample_formula(rng, level=3, noise=0.05)
    assert render_score(to_latex(tree), img) > 0.99
    assert render_score(to_latex(tree, np.random.default_rng(5)), img) > 0.99   # 等价写法画出来一样
    assert render_score("\\frac{", img) == 0.0                                   # 解析失败
    assert render_score("x+y+z+a+b+c+1+2+3+4+5", img) == 0.0                     # 大小差得太多


def test_self_consistency():
    assert abs(self_consistency(["a", "a", "b"]) - 1 / 3) < 1e-9
    assert self_consistency(["only"]) == 1.0


def test_quota_sample():
    labels = np.array([0] * 5 + [1] * 2 + [2] * 10)
    w = np.ones(len(labels))
    pick = quota_sample(labels, w, 9, np.random.default_rng(0))
    assert len(set(pick.tolist())) == 9
    assert [int((labels[pick] == c).sum()) for c in range(3)] == [4, 2, 3]   # 3/3/3 → 簇 1 只有 2 个 → 多的 1 个给簇 0
    w = np.zeros(len(labels))
    w[[7, 8, 9]] = 1.0
    pick = quota_sample(labels, w, 6, np.random.default_rng(0))
    in2 = pick[labels[pick] == 2].tolist()
    assert len(in2) == 2 and set(in2) <= {7, 8, 9}                           # 每簇 2 个名额；簇内只抽正权重的样本


def test_minhash_lsh_and_keys():
    assert latex_key("\\begin{align*} x ^{2}\\, + 1\\end{align*}") == latex_key("x^{2}+1")
    assert latex_key("\\alpha x") == "\\alpha x" != latex_key("\\alphax")           # 命令名后的空格不能删
    assert latex_key("$$x^2+y_i$$") == latex_key("\\[ x^{2} + y_{i} \\]") == "x^{2}+y_{i}"     # 外壳、单字符上下标
    a, b = shingles("abcdefghijklmnop", 3), shingles("abcdefghijXYZmnop", 3)
    est = (minhash(a, 512) == minhash(b, 512)).mean()
    assert abs(est - jaccard(a, b)) < 0.1
    s = [minhash(shingles(t, 3)) for t in ("the same formula x", "the same formula x", "completely different")]
    assert (0, 1) in lsh_pairs(s) and (0, 2) not in lsh_pairs(s)


def test_group_split():
    groups = ["p1"] * 4 + ["p2"] * 3 + ["p3"] * 2 + ["p4"] * 1 + ["p5"] * 5
    tr, va = group_split(groups, 0.3, seed=0)
    assert sorted(tr + va) == list(range(len(groups)))
    assert not ({groups[i] for i in tr} & {groups[i] for i in va})
    assert len(va) >= 0.3 * len(groups)


def test_page_score():
    ref = [("text", "hello world"), ("formula", "x^{2}"), ("text", "abc")]
    pred = [("text", "abc"), ("text", "hello world"), ("text", "extra")]
    assert [(i, j) for i, j, _ in match_elements(pred, ref)] == [(0, 2), (1, 0)]
    score, per = page_score(pred, ref)
    assert abs(per["text"] - 14 / 19) < 1e-9 and per["formula"] == 0.0     # 多写的 "extra" 算进分母
    assert abs(score - 14 / 19 * 14 / 19) < 1e-9
    assert page_score(ref, ref)[0] == 1.0
