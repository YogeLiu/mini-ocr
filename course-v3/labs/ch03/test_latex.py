"""第 3 章的自动检查：LaTeX 的解析、规范写法与归一化。"""
import numpy as np
import pytest

from miniocr.data.latex import (LatexError, gen_formula, leaves, normalize, parse, structure_of, to_latex,
                                tokenize)


def test_tokenize():
    assert tokenize("\\frac{a}{b} + x^2") == ["\\frac", "{", "a", "}", "{", "b", "}", "+", "x", "^", "2"]


@pytest.mark.parametrize("raw, canon", [
    ("x^2", "x^{2}"),
    ("x^{2}", "x^{2}"),
    ("x^2_n", "x_{n}^{2}"),                     # 上下标的先后不影响含义；规范写法是先下标
    ("{a \\over b}+1", "\\frac{a}{b}+1"),
    (" 3x + y = 7 ", "3x+y=7"),                 # 空格不影响含义
    ("{{a \\over b}}", "\\frac{a}{b}"),        # 多余的花括号不影响含义
    ("{x}^2", "x^{2}"),
    ("\\sqrt{x_1+1}", "\\sqrt{x_{1}+1}"),
])
def test_known_cases(raw, canon):
    assert normalize(raw) == canon


@pytest.mark.parametrize("bad", ["x^", "\\frac{a}", "a \\over b", "x^2^3", "{a \\over b \\over c}",
                                 "\\alpha", "}", "x_{", "x^+"])
def test_invalid_raises(bad):
    with pytest.raises(LatexError):
        normalize(bad)


def test_roundtrip_and_all_styles_normalize_to_canonical():
    """对每个等级随机生成 500 个公式：规范写法解析回来是同一棵树；任何一种随机写法归一化后等于规范写法；归一化幂等。"""
    rng = np.random.default_rng(0)
    for level in (1, 2, 3, 4):
        for _ in range(500):
            tree = gen_formula(rng, level)
            canon = to_latex(tree)
            assert parse(canon) == tree
            assert normalize(canon) == canon                  # 幂等：规范写法再归一化不变
            for _ in range(3):
                assert normalize(to_latex(tree, rng)) == canon


def test_random_styles_really_differ():
    """随机写法确实会产生不同的字符串（否则"写法不唯一"的问题就不存在了）。"""
    rng = np.random.default_rng(1)
    differ = sum(to_latex(t, rng) != to_latex(t) for t in (gen_formula(rng, 3) for _ in range(300)))
    assert differ > 100


def test_levels_have_required_structure():
    rng = np.random.default_rng(2)
    for _ in range(200):
        assert structure_of(gen_formula(rng, 1)) == set()
        assert "script" in structure_of(gen_formula(rng, 2))
        assert "frac" in structure_of(gen_formula(rng, 3))
        assert structure_of(gen_formula(rng, 4)) & {"sqrt", "paren", "nested"}


def test_leaves_order():
    assert leaves(parse("\\frac{a+1}{b}=x_{n}^{2}")) == ["a", "+", "1", "b", "=", "x", "n", "2"]
