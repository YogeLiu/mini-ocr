"""A6 第一部分：规范化、精确去重、MinHash、LSH、近似重复的簇、按组划分、去污染。"""
import hashlib
from collections import defaultdict
from itertools import combinations

import numpy as np
import pytest

from ._impl import adapters as A


def test_latex_key():
    assert A.run_latex_key("\\begin{align*} x ^{2}\\, + 1\\end{align*}") == A.run_latex_key("x^{2}+1") == "x^{2}+1"
    assert A.run_latex_key("\\alpha x") == "\\alpha x" != A.run_latex_key("\\alphax")       # 命令名后的空格不能删
    assert A.run_latex_key("$$x^2+y_i$$") == A.run_latex_key("\\[ x^{2} + y_{i} \\]") == "x^{2}+y_{i}"
    assert A.run_latex_key("\\displaystyle a\\quad b") == "ab"
    assert A.run_latex_key("x\\^2") == "x\\^2"                                               # 转义的 ^ 不是上标


@pytest.mark.parametrize("s,c", [
    ("x^2", "x^{2}"), ("{a \\over b}", "\\frac{a}{b}"), ("x^{2}_{n}", "x_{n}^{2}"), ("$$ 3x + y $$", "3x+y"),
    ("\\frac{a", "\\frac{a"), ("a^b + \\frac", "a^{b}+\\frac"), ("x^{2}_{i}", "x^{2}_{i}"), ("{a+b}^2", "{a+b}^{2}")])
def test_canon(s, c):
    assert A.run_canon(s) == c


def test_dedup_shingles_jaccard():
    assert A.run_dedup(["b", "a", "B", "a", "c"], str.lower) == [0, 1, 4]
    assert A.run_shingles("abcab", 3) == {"abc", "bca", "cab"}
    assert A.run_shingles("ab", 3) == {"ab"}
    assert A.run_jaccard({1, 2, 3}, {2, 3, 4}) == 0.5 and A.run_jaccard(set(), set()) == 1.0


def _ref_minhash(sh, n_hash):
    P = (1 << 31) - 1
    ab = np.random.default_rng(12345).integers(1, P, size=(2, 1024), dtype=np.int64)
    x = [int.from_bytes(hashlib.blake2b(e.encode(), digest_size=4).digest(), "little") for e in sh]
    return np.array([min((int(ab[0, i]) * v + int(ab[1, i])) % P for v in x) for i in range(n_hash)])


def test_minhash_exact():
    for s in ["\\frac{a}{b}+x^{2}", "k", "3x+y=7"]:
        sh = A.run_shingles(s, 3)
        assert np.array_equal(np.asarray(A.run_minhash(sh, 32)), _ref_minhash(sh, 32))


def test_minhash_estimates_jaccard():
    rng = np.random.default_rng(0)
    for _ in range(5):
        base = {f"t{i}" for i in rng.choice(400, 60, replace=False)}
        other = set(list(base)[:int(rng.integers(10, 60))]) | {f"u{i}" for i in range(int(rng.integers(0, 40)))}
        est = np.mean(np.asarray(A.run_minhash(base, 512)) == np.asarray(A.run_minhash(other, 512)))
        assert est == pytest.approx(A.run_jaccard(base, other), abs=0.07)


def test_lsh_pairs_matches_definition():
    rng = np.random.default_rng(1)
    sigs = rng.integers(0, 3, size=(30, 12))
    got = A.run_lsh_pairs(sigs, 4)
    exp = {(i, j) for i, j in combinations(range(30), 2)
           if any((sigs[i, b * 3:(b + 1) * 3] == sigs[j, b * 3:(b + 1) * 3]).all() for b in range(4))}
    assert got == exp and len(exp) > 0


def test_near_dup_clusters():
    sets = [{1, 2, 3}, {1, 2, 3, 4}, {9}, {1, 2, 3, 4, 5}, {9, 8}]
    pairs = {(0, 1), (1, 3), (2, 4), (0, 2)}
    assert A.run_near_dup_clusters(sets, pairs, 0.75) == [0, 0, 2, 0, 4]                  # 0–1 0.75、1–3 0.8 连起来；2–4 只有 0.5
    assert A.run_near_dup_clusters(sets, pairs, 0.8) == [0, 1, 2, 1, 4]
    assert A.run_near_dup_clusters(sets, set(), 0.0) == [0, 1, 2, 3, 4]                    # 只看候选对


def test_group_split():
    groups = ["d3", "d1", "d1", "d2", "d3", "d3", "d4", "d2", "d5", "d1"]
    tr, va = A.run_group_split(groups, 0.3, 0)
    uniq = sorted(set(groups), key=str)
    order = np.random.default_rng(0).permutation(len(uniq))
    val, n = set(), 0
    for k in order:
        if n >= 3:
            break
        val.add(uniq[k])
        n += groups.count(uniq[k])
    assert va == [i for i, g in enumerate(groups) if g in val] and tr == [i for i, g in enumerate(groups) if g not in val]
    for seed in range(10):
        tr, va = A.run_group_split(groups, 0.45, seed)
        assert sorted(tr + va) == list(range(10)) and not ({groups[i] for i in tr} & {groups[i] for i in va})
        assert len(va) >= 4.5


def test_decontaminate():
    test = ["\\frac{a}{b}+x^{2}=7", "\\sqrt{3k-n}+y_{c}", "2a+3b-c"]
    pool = ["\\frac{a}{b}+x^{2}=7", "z", "\\sqrt{3k-n}+y_{c}+1", "\\frac{8}{9}-y", "2a+3b-z", "x^{2}"]
    ps, ts = [A.run_shingles(s, 3) for s in pool], [A.run_shingles(s, 3) for s in test]
    exp = sorted(i for i in range(len(pool)) if max(A.run_jaccard(ps[i], t) for t in ts) >= 0.6)
    assert A.run_decontaminate(ps, ts, 0.6, 64, 64) == exp == [0, 2, 4]
    assert A.run_decontaminate(ps, ts, 0.99, 64, 16) == [0]
    assert A.run_decontaminate(ps, ps[:1], 0.5, 64, 16) == [0]                            # 只返回数据池的下标
