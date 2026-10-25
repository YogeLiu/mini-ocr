"""第 2 章：图像操作的自动检查（与 Pillow 对拍）。"""
import numpy as np
from PIL import Image

from miniocr.data.imgops import crop, hwc_to_chw, letterbox, resize_bilinear, resize_nearest, to_float, to_gray
from miniocr.data.render import render_glyph, render_formula
from miniocr.data.shapes import draw_scene, sample_scene


def _ties(in_n, out_n):
    """输出像素中心恰好落在两个输入像素交界处的位置（这里 Pillow 因为浮点误差会取左边的像素）。"""
    c = (np.arange(out_n) + 0.5) * in_n
    return set(np.nonzero(c % out_n == 0)[0])


def test_gray_matches_pillow():
    rng = np.random.default_rng(0)
    img = draw_scene(sample_scene(rng), rng)
    assert np.array_equal(to_gray(img), np.array(Image.fromarray(img).convert("L")))


def test_nearest_matches_pillow_except_ties():
    g, _ = render_glyph("B", np.random.default_rng(0))
    for h, w in [(16, 16), (8, 8), (64, 64), (20, 45), (45, 20)]:
        mine = resize_nearest(g, h, w)
        ref = np.array(Image.fromarray(g).resize((w, h), Image.NEAREST))
        th, tw = _ties(32, h), _ties(32, w)
        for r, c in zip(*np.nonzero(mine != ref)):
            assert r in th or c in tw                                    # 每个不同的像素都在"平局"的行或列上


def test_bilinear_upscale_matches_pillow():
    g, _ = render_glyph("B", np.random.default_rng(0))
    mine = resize_bilinear(g, 64, 80)
    ref = np.array(Image.fromarray(g).resize((80, 64), Image.BILINEAR))
    assert np.abs(mine.astype(int) - ref).max() <= 1          # 只差舍入


def test_letterbox_keeps_aspect_and_maps_boxes():
    from miniocr.data.latex import parse
    img, boxes = render_formula(parse("\\frac{a}{b}"), np.random.default_rng(0))   # 48×192
    out, scale, (px, py) = letterbox(img, 96, 96)
    assert out.shape == (96, 96)
    assert abs(scale - 96 / 192) < 1e-9 and px == 0 and py == 36                 # 宽度占满，上下各补 36 行白边
    assert (out[:py] == 255).all() and (out[-py:] == 255).all()
    _, (x0, y0, x1, y1) = boxes[0]                                               # 分子 a 的框换算到新图
    nx0, ny0 = int(x0 * scale + px), int(y0 * scale + py)
    nx1, ny1 = int(np.ceil(x1 * scale + px)), int(np.ceil(y1 * scale + py))
    assert out[ny0:ny1, nx0:nx1].min() < 200                                    # 新框里仍然有墨迹


def test_to_float_ranges():
    img = np.array([[0, 255]], dtype=np.uint8)
    assert to_float(img).tolist() == [[0.0, 1.0]]
    assert to_float(img, invert=True).tolist() == [[1.0, 0.0]]
    assert to_float(img, mean=0.5, std=0.5).tolist() == [[-1.0, 1.0]]
    assert to_float(img).dtype == np.float32


def test_crop_and_chw():
    a = np.arange(5 * 6 * 3).reshape(5, 6, 3)
    assert crop(a, (1, 2, 4, 5)).shape == (3, 3, 3)                              # (x0,y0,x1,y1)：3 行 3 列
    assert hwc_to_chw(a).shape == (3, 5, 6)
    assert hwc_to_chw(a)[2, 1, 4] == a[1, 4, 2]


def test_area_resize_properties():
    from miniocr.data.imgops import _area_matrix, resize_area
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, size=(32, 32)).astype(np.uint8)
    blocks = a.reshape(16, 2, 16, 2).astype(float).mean(axis=(1, 3))         # 整数倍：就是 2×2 块平均
    assert np.abs(resize_area(a, 16, 16).astype(float) - blocks).max() <= 0.5
    assert (resize_area(np.full((32, 32), 77, np.uint8), 10, 7) == 77).all()  # 常数图缩放后不变
    M = _area_matrix(32, 7)
    assert np.allclose(M.sum(1), 1) and np.allclose(M.sum(0), 7 / 32)         # 每个输入像素的"总贡献"相同


def test_normalize_glyph_removes_position():
    from miniocr.data.imgops import normalize_glyph
    img = np.full((32, 32), 255, np.uint8)
    img[3:9, 4:7] = 0                                                       # 左上角一个小竖条
    img2 = np.full((32, 32), 255, np.uint8)
    img2[20:26, 25:28] = 0                                                  # 右下角同样的竖条
    assert np.array_equal(normalize_glyph(img, 16), normalize_glyph(img2, 16))
