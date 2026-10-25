"""第 25 章的自动检查。"""
import math

import numpy as np
import pytest

from miniocr.eval.cdm import cdm, match_count, symbols
from miniocr.eval.stats import bootstrap_ci, mcnemar, paired_bootstrap
from miniocr.eval.teds import Node, teds, tree_edit_distance
from miniocr.metrics import anls, bleu, ned, vqa_accuracy, wer


def test_wer_and_ned():
    assert wer(["the cat sat"], ["the cat sat"]) == 0
    assert wer(["the bat sat down"], ["the cat sat"]) == pytest.approx(2 / 3)     # 1 替换 + 1 插入
    assert ned("abc", "abd") == pytest.approx(1 / 3)
    assert ned("", "") == 0 and ned("", "ab") == 1


def test_bleu():
    ref = "the cat is on the mat".split()
    assert bleu([ref], [ref]) == pytest.approx(1.0)
    assert bleu([["the", "the", "the"]], [ref], max_n=1) == pytest.approx(2 / 3 * math.exp(1 - 6 / 3))  # 截断 + 简短惩罚
    assert bleu([["mat", "the", "on"]], [ref], max_n=2) == 0.0                      # 没有任何二元组对得上


def test_anls_and_vqa():
    assert anls(["Paris"], [["paris"]]) == 1.0
    assert anls(["Pariss"], [["paris"]]) == pytest.approx(1 - 1 / 6)
    assert anls(["London"], [["paris"]]) == 0.0                                      # NED ≥ 0.5 记 0
    assert anls(["12"], [["twelve", "12"]]) == 1.0                                    # 取多个答案里最好的
    assert vqa_accuracy("red", ["red"] * 2 + ["blue"] * 8) == pytest.approx(2 / 3)
    assert vqa_accuracy("red", ["red"] * 5 + ["blue"] * 5) == 1.0


def test_cdm():
    assert cdm("x^2+1", "x^{2}+1") == 1.0                                              # 写法不同、渲染相同
    assert cdm("{a \\over b}", "\\frac{a}{b}") == 1.0
    assert cdm("x_{2}", "x^{2}") == pytest.approx(0.5)                                 # 2 的位置不同：x 对上、2 对不上
    assert cdm("x+", "x^{2}+1") < 1
    assert cdm("\\frac{a}{", "\\frac{a}{b}") == 0.0                                    # 无法渲染
    s = symbols("\\frac{1}{x}")
    assert [t[0] for t in s].count("frac") == 1
    a, b = symbols("2x+1"), symbols("x+1")
    assert match_count(b, a) == 3                                                      # 整体平移也能对上


def test_tree_edit_distance():
    def T(label, *kids):
        n = Node(label)
        n.children = list(kids)
        return n
    # Zhang–Shasha 论文的例子：f(d(a c(b)) e) 与 f(c(d(a b)) e)，距离 2
    t1 = T("f", T("d", T("a"), T("c", T("b"))), T("e"))
    t2 = T("f", T("c", T("d", T("a"), T("b"))), T("e"))
    ren = lambda x, y: 0.0 if x.tag == y.tag else 1.0                                    # noqa: E731
    assert tree_edit_distance(t1, t2, ren) == 2
    assert tree_edit_distance(t1, t1, ren) == 0
    assert tree_edit_distance(T("a"), T("a", T("b"), T("c")), ren) == 2


def test_teds():
    ref = "<table><tr><td>Name</td><td>Age</td></tr><tr><td>bob</td><td>12</td></tr></table>"
    assert teds(ref, ref) == 1.0
    typo = ref.replace("bob", "bab")
    assert teds(typo, ref) == pytest.approx(1 - (1 / 3) / 7)                           # 7 个节点，一格错 1/3
    assert teds(typo, ref, structure_only=True) == 1.0
    merged = "<table><tr><td>Name</td><td>Age</td></tr><tr><td colspan=\"2\">bob 12</td></tr></table>"
    assert teds(merged, ref) < teds(typo, ref)
    assert teds("<table><tr><td>Name", ref) < 0.5                                     # 残缺的 HTML 也能算


def test_bootstrap_and_mcnemar():
    rng = np.random.default_rng(0)
    errs = rng.integers(0, 3, 500)
    lens = rng.integers(5, 20, 500)
    p, lo, hi = bootstrap_ci(errs, lens, n_boot=500)
    assert lo < p < hi and p == pytest.approx(errs.sum() / lens.sum())
    acc, lo2, hi2 = bootstrap_ci(rng.random(400) < 0.8, n_boot=500)
    assert lo2 < acc < hi2 and hi2 - lo2 < 0.12                                        # 约 ±2 × √(0.16 / 400) = ±0.04
    d, dlo, dhi, frac = paired_bootstrap(errs + 1, errs, lens, n_boot=200)
    assert d > 0 and dlo > 0 and frac == 0.0
    b10, b01, pv = mcnemar([1, 1, 1, 1, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1])
    assert (b10, b01) == (6, 0) and pv == pytest.approx(2 / 64)
