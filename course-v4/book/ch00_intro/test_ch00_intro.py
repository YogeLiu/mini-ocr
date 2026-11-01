"""第 0 章的自动检查：python -m pytest book/ch00_intro"""
import numpy as np
import pytest

from miniocr.eval.latex_norm import (LatexSyntaxError, apply_token_rules, is_well_formed, normalize_latex,
                                     parse_latex, serialize, tokenize_latex)
from miniocr.eval.metrics import (edit_distance, error_rate, exact_match, sample_error_rate, score_pair, summarize,
                                  units)
from miniocr.eval.render import can_parse, compare_images, compare_render, render


# ---------------------------------------------------------------- 练习 0.1
def test_tokenize_basic():
    assert tokenize_latex(r"\frac{a}{b} + x^2") == ["\\frac", "{", "a", "}", "{", "b", "}", "+", "x", "^", "2"]
    assert tokenize_latex("S_{面积}") == ["S", "_", "{", "面", "积", "}"]
    assert tokenize_latex(r"\alpha\beta") == ["\\alpha", "\\beta"]
    assert tokenize_latex("  x \n ") == ["x"]


def test_tokenize_symbols_and_envs():
    assert tokenize_latex(r"a\,b\\c\{") == ["a", "\\,", "b", "\\\\", "c", "\\{"]
    assert tokenize_latex(r"\begin{ pmatrix }a&b\end{pmatrix}") == ["\\begin{pmatrix}", "a", "&", "b", "\\end{pmatrix}"]
    assert tokenize_latex("a\\") == ["a", "\\"]
    assert tokenize_latex(r"\alpha2") == ["\\alpha", "2"]


# ---------------------------------------------------------------- 练习 0.2
def test_token_rules():
    assert apply_token_rules(["\\left", "(", "a", "\\,", "b", "\\right", "."]) == ["(", "a", "b"]
    assert apply_token_rules(["\\dfrac", "\\le", "\\Big", "|", "\\quad", "~"]) == ["\\frac", "\\leq", "|"]
    toks = ["\\tfrac", "x"]
    apply_token_rules(toks)
    assert toks == ["\\tfrac", "x"], "不能修改输入列表"


# ---------------------------------------------------------------- 练习 0.3
def test_parse_tree():
    assert parse_latex(["x", "^", "2"]) == [("script", ("tok", "x"), None, ("tok", "2"))]
    assert parse_latex(["x", "^", "2", "_", "i"]) == [("script", ("tok", "x"), ("tok", "i"), ("tok", "2"))]
    assert parse_latex(["\\frac", "1", "2"]) == [("cmd", "\\frac", None, [("tok", "1"), ("tok", "2")])]
    assert parse_latex(["{", "a", "}"]) == [("group", [("tok", "a")])]
    assert parse_latex(["\\sqrt", "[", "3", "]", "x"]) == [("cmd", "\\sqrt", [("tok", "3")], [("tok", "x")])]
    assert parse_latex(["^", "2"]) == [("script", ("group", []), None, ("tok", "2"))]


@pytest.mark.parametrize("toks", [["x", "^", "{", "2"], ["x", "}"], ["\\frac", "a"], ["x", "^"], ["x", "^", "1", "^", "2"]])
def test_parse_errors(toks):
    with pytest.raises(LatexSyntaxError):
        parse_latex(toks)


# ---------------------------------------------------------------- 练习 0.4
def test_serialize():
    assert serialize(parse_latex(["x", "^", "2", "_", "i"])) == ["x", "_", "{", "i", "}", "^", "{", "2", "}"]
    assert serialize(parse_latex(tokenize_latex("{{a+b}}^{2}"))) == ["{", "a", "+", "b", "}", "^", "{", "2", "}"]
    assert serialize(parse_latex(tokenize_latex(r"\frac{{a}}{b}"))) == ["\\frac", "{", "a", "}", "{", "b", "}"]
    assert serialize(parse_latex(tokenize_latex(r"\sqrt[3]x"))) == ["\\sqrt", "[", "3", "]", "{", "x", "}"]
    # 参数里多元素的内层分组也只保留一层
    assert serialize(parse_latex(tokenize_latex(r"\frac{{a+b}}{c}"))) == ["\\frac", "{", "a", "+", "b", "}", "{", "c", "}"]


def test_serialize_keeps_braces_around_scripted_base():
    # {x^2}^3 的底自己带上标，花括号不能去掉，否则会写成不合法的 x^{2}^{3}
    assert normalize_latex("{x^2}^3") == "{ x ^ { 2 } } ^ { 3 }"
    assert normalize_latex("{{x^2}}^3") == "{ x ^ { 2 } } ^ { 3 }"
    assert not score_pair("{x^2}^3", "x^{2}^{3}")["em_norm"], "非法预测不能和合法参考匹配"


@pytest.mark.parametrize("s", ["{x^2}^3", r"\frac{{a+b}}{c}", "x^{y^2}_i", r"\sqrt[3]{{x}^2}", "{{a+b}}^{2}",
                               r"\left(\frac12\right)^2", r"\sum^{n}_{i=1}i^2"])
def test_normalize_keeps_legal_and_is_idempotent(s):
    n = normalize_latex(s)
    assert is_well_formed(n), "合法输入规范化以后必须仍然合法"
    assert normalize_latex(n) == n, "规范化两次应该和一次一样"


@pytest.mark.parametrize("a,b", [
    ("x^2 + y^2", "x^{2}+y^{2}"), (r"\frac12", r"\dfrac{1}{2}"), (r"\sum^{n}_{i=1}", r"\sum_{i=1}^{n}"),
    (r"\left( a \right)", "(a)"), (r"a\le b", r"a\leq b"), (r"f(x)\,dx", "f(x)dx"), ("{a}^{2}", "a^2"),
    (r"\frac{{a+b}}{c}", r"\frac{a+b}{c}"),
])
def test_normalize_equal(a, b):
    assert normalize_latex(a) == normalize_latex(b)


@pytest.mark.parametrize("a,b", [("x^{ab}", "x^ab"), ("x^{2}", "x_{2}"), (r"\frac{a}{b}", r"\frac{b}{a}"),
                                 ("a+b", "b+a"), (r"\sqrt[3]{x}", r"\sqrt{x}")])
def test_normalize_different(a, b):
    assert normalize_latex(a) != normalize_latex(b)


def test_normalize_falls_back_on_bad_input():
    assert normalize_latex("x^{2") == "x ^ { 2"


# ---------------------------------------------------------------- 练习 0.5
def test_edit_distance():
    assert edit_distance("x^{2}", "x_2") == 3
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3 and edit_distance("abc", "") == 3 and edit_distance("", "") == 0
    assert edit_distance(["\\alpha", "+"], ["a", "+"]) == 1
    assert edit_distance("面积", "面枳") == 1


def test_edit_distance_symmetric_random():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = "".join(rng.choice(list("ab{}"), rng.integers(0, 8)))
        b = "".join(rng.choice(list("ab{}"), rng.integers(0, 8)))
        d = edit_distance(a, b)
        assert d == edit_distance(b, a)
        assert abs(len(a) - len(b)) <= d <= max(len(a), len(b))


# ---------------------------------------------------------------- 练习 0.6
def test_error_rate_corpus_level():
    assert error_rate(["ab", "cd"], ["ab", "xd"], normalize=False) == pytest.approx(1 / 4)
    refs, preds = [r"\pi", r"\sum_{k=1}^{n}k=\frac{n(n+1)}{2}"], ["n", r"\sum_{k=1}^{n}k=\frac{n(n-1)}{2}"]
    assert error_rate(refs, preds, "token") == pytest.approx(2 / 26)
    assert error_rate(["x^2"], ["x^{2}"], "char") == 0.0
    # 零分母约定：参考为空时，预测也为空算 0，预测不空算 1；三个入口要一致
    assert error_rate([""], [""]) == 0.0
    assert error_rate([""], ["xx"]) == 1.0
    assert sample_error_rate("", "xx") == 1.0
    assert summarize([score_pair("", "xx")])["cer_char"] == 1.0
    assert summarize([score_pair("", "xx")])["cer_token"] == 1.0


def test_error_rate_matches_summarize():
    refs, preds = ["x^{2}", r"\frac{a}{b}", r"\alpha+\beta"], ["x_2", r"\frac{b}{a}", r"a+\beta"]
    s = summarize([score_pair(r, p) for r, p in zip(refs, preds)])
    assert s["cer_char"] == pytest.approx(error_rate(refs, preds, "char"))
    assert s["cer_token"] == pytest.approx(error_rate(refs, preds, "token"))


def test_error_rate_rejects_length_mismatch():
    with pytest.raises(ValueError):
        error_rate(["a", "b"], ["a"])
    with pytest.raises(ValueError):
        error_rate(["a"], ["a", "b"])
    with pytest.raises(ValueError):
        exact_match(["a", "b"], ["a"])


def test_error_rate_units():
    assert units(r"\alpha+1", "token") == ["\\alpha", "+", "1"]
    assert len(units(r"\alpha+1", "char")) == 8
    assert error_rate([r"\alpha+\beta"], [r"a+\beta"], "char") == pytest.approx(5 / 12)
    assert error_rate([r"\alpha+\beta"], [r"a+\beta"], "token") == pytest.approx(1 / 3)
    assert sample_error_rate("x^{2}", "x^{2}x^{2}x^{2}x^{2}", "token") == pytest.approx(3.0)


def test_exact_match():
    refs, preds = ["x^{2}", "a"], ["x^2", "b"]
    assert exact_match(refs, preds, normalize=False) == 0.0
    assert exact_match(refs, preds) == 0.5


# ---------------------------------------------------------------- 练习 0.7
def test_compare_images_hand_examples():
    A = np.array([[0, 255, 255, 0], [0, 255, 0, 0], [0, 255, 0, 0]], dtype=np.uint8)
    B = np.array([[0, 255, 255, 0], [0, 255, 255, 0], [0, 255, 0, 0]], dtype=np.uint8)
    assert compare_images(A, B) == pytest.approx(0.2)
    C = np.array([[255, 0, 255, 0]], dtype=np.uint8)
    E = np.array([[255, 0, 0, 255]], dtype=np.uint8)
    assert compare_images(C, E) == pytest.approx(2 / 3)
    shifted = np.zeros((5, 6), dtype=np.uint8)
    shifted[2:5, 2:6] = A
    assert compare_images(A, shifted) == 0.0, "整体平移不应该算不一致（要先裁掉空白）"
    blank = np.zeros((3, 3), dtype=np.uint8)
    assert compare_images(blank, blank) == 0.0
    assert compare_images(blank, A) == 1.0


def test_render_and_compare():
    assert can_parse("x^{2}") and not can_parse("x^{2") and not can_parse("")
    img = render("x^{2}")
    assert img.dtype == np.uint8 and img.ndim == 2 and img.max() > 128
    assert compare_render("x^2", "x^{2}")["image_match"] is True
    assert compare_render("x^{2}", "x_{2}")["image_match"] is False
    r = compare_render("x^{2}", "x^{2")
    assert r["compiles"] is False and r["image_match"] is None


def test_summarize_image_match_undefined():
    # 没有任何能判断渲染的样本：一致率是"无法判断"（None），不是 0
    row = score_pair("x^{2}", "x^{2")
    row.update(compare_render("x^{2}", "x^{2"))
    s = summarize([row])
    assert s["image_judged"] == 0 and s["image_match"] is None
