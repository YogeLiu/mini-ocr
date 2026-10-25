"""A4 第一部分：编辑距离与回溯、CER/WER/NED、BLEU、ANLS、表格解析、树编辑距离、TEDS。对照独立写的递归版本和手算。"""
import math
import random
from functools import lru_cache

import pytest

from ._impl import adapters as A


def _ed_rec(a, b):
    @lru_cache(None)
    def d(i, j):
        if i == 0 or j == 0:
            return i + j
        return min(d(i - 1, j) + 1, d(i, j - 1) + 1, d(i - 1, j - 1) + (a[i - 1] != b[j - 1]))
    return d(len(a), len(b))


def test_edit_distance_random():
    rng = random.Random(0)
    for _ in range(200):
        a = "".join(rng.choice("abc") for _ in range(rng.randint(0, 7)))
        b = "".join(rng.choice("abc") for _ in range(rng.randint(0, 7)))
        assert A.run_edit_distance(a, b) == _ed_rec(a, b)
    assert A.run_edit_distance(["the", "cat"], ["a", "cat"]) == 1


def test_align_ops():
    rng = random.Random(1)
    for _ in range(100):
        p = "".join(rng.choice("ab ") for _ in range(rng.randint(0, 6)))
        r = "".join(rng.choice("ab ") for _ in range(rng.randint(0, 6)))
        ops = A.run_align_ops(p, r)
        assert "".join(x for op, x, _ in ops if op != "I") == p                 # 按顺序拼回 pred
        assert "".join(y for op, _, y in ops if op != "D") == r                 # 按顺序拼回 ref
        assert sum(op != "=" for op, _, _ in ops) == _ed_rec(p, r)              # 操作数最少
        assert all((op == "=") == (x == y) for op, x, y in ops if op in "=S")
    assert A.run_align_ops("kitten", "sitting") == [("S", "k", "s"), ("=", "i", "i"), ("=", "t", "t"), ("=", "t", "t"),
                                                    ("S", "e", "i"), ("=", "n", "n"), ("I", "", "g")]
    assert A.run_align_ops("aab", "ab") == [("D", "a", ""), ("=", "a", "a"), ("=", "b", "b")]   # 对角线优先：从末尾回溯


def test_cer_wer_ned():
    assert A.run_cer(["abc", "x"], ["abd", "xyzw"]) == pytest.approx(4 / 7)                # 语料级，不是逐行平均（那样是 0.54）
    assert A.run_wer(["the cat sat", "hi"], ["the dog sat", "hi there"]) == pytest.approx(2 / 5)
    assert A.run_ned("abc", "abd") == pytest.approx(1 / 3)
    assert A.run_ned("", "") == 0.0
    assert A.run_ned("", "abcd") == 1.0


def test_bleu():
    toks = "the cat sat on the mat".split()
    assert A.run_bleu([toks], [toks], 4) == pytest.approx(1.0)
    assert A.run_bleu([["a", "b"]], [["c", "d"]], 2) == 0.0
    # 预测 the the the the（4 个），参考 the cat：1 元截断精确率 1/4（the 在参考里只出现 1 次）；2 元 0 → 0
    assert A.run_bleu([["the"] * 4], [["the", "cat"]], 1) == pytest.approx(0.25)
    # 简短惩罚：预测 3 个词全对，参考 6 个词；p1 = 1、p2 = 1 → BP = e^(1 − 6/3)
    pred = "the cat sat".split()
    assert A.run_bleu([pred], [toks], 2) == pytest.approx(math.exp(1 - 2))
    # 语料级：分子分母分别累加
    p = [["a", "b", "c"], ["x", "y"]]
    r = [["a", "b", "d"], ["x", "y"]]
    p1, p2 = 4 / 5, 2 / 3
    assert A.run_bleu(p, r, 2) == pytest.approx(math.sqrt(p1 * p2))


def test_anls():
    assert A.run_anls(["Hello"], [["hello"]], 0.5) == pytest.approx(1.0)                   # 小写后完全相同
    assert A.run_anls(["helo"], [["hello"]], 0.5) == pytest.approx(0.8)
    assert A.run_anls(["xyz"], [["hello"]], 0.5) == 0.0
    assert A.run_anls(["helo", "abc"], [["nope", "hello"], ["abd"]], 0.5) == pytest.approx((0.8 + 2 / 3) / 2)


def test_parse_table():
    html = ('<table><thead><tr><th>a</th><th colspan="2"> b </th></tr></thead>'
            '<tbody><tr><td rowspan="2">c</td><td>d</td><td>e</td></tr><tr><td>f</td></tr></tbody></table>')
    t = A.run_parse_table(html)
    assert t[0] == "table" and [c[0] for c in t[4]] == ["tr", "tr", "tr"]           # thead / tbody 被展开
    r0, r1, r2 = t[4]
    assert [(c[0], c[1], c[2], c[3]) for c in r0[4]] == [("td", 1, 1, "a"), ("td", 2, 1, "b")]   # th 当作 td，文字去空白
    assert [(c[1], c[2], c[3]) for c in r1[4]] == [(1, 2, "c"), (1, 1, "d"), (1, 1, "e")]
    assert len(r2[4]) == 1
    broken = A.run_parse_table("<table><tr><td>x<td>y</tr></table>")                # 没闭合的 td
    assert broken[0] == "table" and len(broken[4]) == 1
    assert A.run_parse_table("no table here")[0] == "table"


# 树编辑距离的独立实现：森林上的递归（指数级，但树很小）
def _ted(a, b, rename):
    def size(t):
        return 1 + sum(size(c) for c in t[1])

    def freeze(t):
        return (t[0], tuple(freeze(c) for c in t[1]))

    @lru_cache(None)
    def fd(F, G):
        if not F and not G:
            return 0.0
        if not F:
            return float(sum(size(t) for t in G))
        if not G:
            return float(sum(size(t) for t in F))
        v, w = F[-1], G[-1]
        return min(fd(F[:-1] + v[1], G) + 1, fd(F, G[:-1] + w[1]) + 1,
                   fd(v[1], w[1]) + fd(F[:-1], G[:-1]) + rename(v[0], w[0]))

    return fd((freeze(a),), (freeze(b),))


def _rand_tree(rng, n):
    nodes = [("x", [])]
    root = nodes[0]
    for _ in range(n - 1):
        par = rng.choice(nodes)
        c = (rng.choice("abc"), [])
        par[1].append(c)
        nodes.append(c)
    return (rng.choice("abc"), root[1])


def test_tree_edit_distance_random():
    rng = random.Random(0)
    unit = lambda x, y: float(x != y)                                                 # noqa: E731
    half = lambda x, y: 0.0 if x == y else 0.5                                        # noqa: E731
    for _ in range(60):
        a, b = _rand_tree(rng, rng.randint(1, 7)), _rand_tree(rng, rng.randint(1, 7))
        for ren in (unit, half):
            assert A.run_tree_edit_distance(a, b, ren) == pytest.approx(_ted(a, b, ren))


def test_teds_cases():
    ref = "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>"
    assert A.run_teds(ref, ref, False) == pytest.approx(1.0)
    merged = '<table><tr><td colspan="2">ab</td></tr><tr><td>c</td><td>d</td></tr></table>'
    assert A.run_teds(merged, ref, False) == pytest.approx(1 - 2 / 7)                 # 删一个 td + 改 colspan
    typo = "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>dd</td></tr></table>"
    assert A.run_teds(typo, ref, False) == pytest.approx(1 - 0.5 / 7)                 # ned("dd", "d") = 0.5
    assert A.run_teds(typo, ref, True) == pytest.approx(1.0)                          # 只比结构
    assert A.run_teds("garbage", ref, False) == pytest.approx(1 - 6 / 7)
