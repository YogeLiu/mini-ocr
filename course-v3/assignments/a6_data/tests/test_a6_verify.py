"""A6 第三部分：墨迹外接框、模糊与归一化互相关、渲染校验、用渲染分在候选里重选。"""
import numpy as np
import pytest

from miniocr.data.latex import parse, to_latex
from miniocr.data.render import render_formula

from ._impl import adapters as A


def _img(s, noise=0.0, seed=1):
    return render_formula(parse(s), np.random.default_rng(seed), H=48, W=192, size=16, noise=noise)[0]


def test_ink_crop():
    img = np.full((10, 12), 255, np.uint8)
    img[2:5, 3:9] = 0
    img[6, 4] = 100
    assert A.run_ink_crop(img, 128, 0).shape == (5, 6)
    assert A.run_ink_crop(img, 128, 2).shape == (3, 6)                                     # (6, 4) 是孤立的暗点
    assert A.run_ink_crop(np.full((4, 4), 200, np.uint8), 128, 0) is None
    line = np.full((5, 9), 255, np.uint8)
    line[2, 1:8] = 0
    assert A.run_ink_crop(line, 128, 2).shape == (1, 5)                                    # 1 像素的线：两端只有 1 个邻居
    assert A.run_ink_crop(line, 128, 1).shape == (1, 7)


def test_blur_and_ncc():
    m = np.arange(12, dtype=float).reshape(3, 4)
    b = A.run_blur3(m)
    p = np.pad(m, 1, mode="edge")
    assert b.shape == (3, 4) and b[0, 0] == pytest.approx(p[0:3, 0:3].mean()) and b[1, 2] == pytest.approx(m[0:3, 1:4].mean())
    x = np.random.default_rng(0).random((5, 6))
    assert A.run_ncc(x, 3 * x + 2) == pytest.approx(1.0) and A.run_ncc(x, -x) == pytest.approx(-1.0)
    assert A.run_ncc(x, np.ones((5, 6))) == 0.0


def test_render_score_cases():
    s = "\\frac{y^{2}}{k}+3x"
    img = _img(s)
    assert A.run_render_score(s, img, 16) > 0.99
    assert A.run_render_score("{y^2 \\over k}+3x", img, 16) == pytest.approx(A.run_render_score(s, img, 16))  # 等价写法
    assert A.run_render_score("\\frac{y_{2}}{k}+3x", img, 16) < 0.95                       # 上标写成下标
    assert A.run_render_score("\\frac{", img, 16) == 0.0                                   # 解析失败
    assert A.run_render_score("x+y+z+a+b+c+1+2+3+4+5", img, 16) == 0.0                     # 大小差得太多
    assert A.run_render_score(s, np.full((48, 192), 255, np.uint8), 16) == 0.0              # 原图没有墨迹


def test_render_score_noise_and_shift():
    s = "2^{n}=\\sqrt{7}"
    clean, noisy = A.run_render_score(s, _img(s, 0.0, 3), 16), A.run_render_score(s, _img(s, 0.2, 3), 16)
    assert clean > 0.99 and 0.9 < noisy < clean                                            # 噪声点不能把外接框撑大
    img = _img(s, 0.0, 4)
    shifted = np.full_like(img, 255)
    shifted[1:, 1:] = img[:-1, :-1]
    assert A.run_render_score(s, shifted, 16) > 0.99


def test_render_select():
    img = _img("a^{2}+\\frac{1}{b}")
    cands = ["a_{2}+\\frac{1}{b}", "a^{2}+\\frac{1}{b}", "a^2+{1 \\over b}", "a^{2}+\\frac{b}{1}", "\\frac{"]
    k, best, scores = A.run_render_select(cands, img, 16)
    assert k == 1 and best == pytest.approx(scores[1]) and scores[2] == pytest.approx(scores[1])
    assert scores[4] == 0.0 and scores[0] < best and scores[3] < best and len(scores) == 5
