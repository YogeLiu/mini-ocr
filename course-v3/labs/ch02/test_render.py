"""第 2 章的自动检查：渲染器的形状、确定性、边界框。"""
import numpy as np

from miniocr.data.latex import gen_formula, leaves, to_latex
from miniocr.data.render import ink_box, render_formula, render_glyph, render_line, sample_formula, to_ascii
from miniocr.data.shapes import COLORS, describe, draw_scene, sample_scene


def test_glyph_shape_dtype_and_box():
    img, box = render_glyph("A", np.random.default_rng(0))
    assert img.shape == (32, 32) and img.dtype == np.uint8
    assert img.max() == 255 and img.min() < 50                  # 白底，有黑色笔画
    x0, y0, x1, y1 = box
    assert 0 <= x0 < x1 <= 32 and 0 <= y0 < y1 <= 32
    assert (img[y0:y1, x0:x1] < 128).sum() == (img < 128).sum()   # 所有墨迹都在框里


def test_same_seed_same_image():
    a, _ = render_glyph("7", np.random.default_rng(3))
    b, _ = render_glyph("7", np.random.default_rng(3))
    c, _ = render_glyph("7", np.random.default_rng(4))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_line_boxes_left_to_right():
    img, boxes = render_line("ab 12", np.random.default_rng(0))
    assert img.shape == (32, 256)
    assert boxes[2] is None                                     # 空格没有框
    xs = [b[0] for b in boxes if b is not None]
    assert xs == sorted(xs)


def test_formula_boxes_follow_leaves_and_stay_inside():
    rng = np.random.default_rng(0)
    for level in (1, 2, 3, 4):
        for _ in range(100):
            tree, img, boxes = sample_formula(rng, level)
            assert img.shape == (48, 192)
            assert [c for c, _ in boxes] == leaves(tree)
            for _, (x0, y0, x1, y1) in boxes:
                assert 0 <= x0 < x1 <= 192 and 0 <= y0 < y1 <= 48


def test_superscript_is_higher_and_smaller():
    tree = ("row", [("script", ("sym", "x"), None, ("row", [("sym", "2")]))])   # x^{2}；直接写树，不依赖第 3 章的解析器
    img, boxes = render_formula(tree, np.random.default_rng(0))
    (_, bx), (_, bs) = boxes
    assert bs[1] < bx[1]                                        # 上标的顶部更靠上（y 更小）
    assert (bs[3] - bs[1]) < (bx[3] - bx[1]) + 4                # 上标的字更小


def test_ascii_view():
    img = np.full((2, 3), 255, dtype=np.uint8)
    img[0, 0] = 0
    assert to_ascii(img) == "@  \n   "


def test_shapes_scene():
    rng = np.random.default_rng(0)
    objs = sample_scene(rng)
    img = draw_scene(objs, rng)
    assert img.shape == (64, 64, 3) and img.dtype == np.uint8
    for o in objs:                                              # 每个物体中心的像素就是它的颜色
        x0, y0, x1, y1 = o["box"]
        cx, cy = (x0 + x1) // 2, (y0 + y1 - 1) // 2 + 1
        assert tuple(img[cy, cx]) == COLORS[o["color"]]
    assert describe(objs).count(", ") == len(objs) - 1
