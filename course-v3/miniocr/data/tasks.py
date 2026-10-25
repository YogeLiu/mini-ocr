"""把渲染器产生的图像整理成"任务数据"：输入数组 + 标签。

    glyph_arrays   单字符识别（第 7、10、11、12 章）
    shape_sizes    ShapesWorld 单物体的边长回归（第 7 章）

所有函数都接受一个 rng；训练、验证、测试用不同的种子生成，保证互不重叠（第 10、25 章）。
"""
import string

import numpy as np

from .imgops import normalize_glyph, to_float
from .render import render_glyph
from .shapes import COLORS, SHAPES, SIZES, draw_scene

CHARSET = string.digits + string.ascii_letters          # 62 个字符：0-9 a-z A-Z


def glyph_arrays(rng, chars=CHARSET, n_per_class=100, size=32, align=False, noise=0.0, flat=True, distort=0.0):
    """渲染单个字符，返回 (X, y)。

    X：float32，墨迹 = 1、背景 = 0（反色）；flat=True 时形状 (N, size*size)，否则 (N, size, size)。
    y：int64，字符在 chars 中的下标。
    align=True 时先做字符归一化（第 2 章），否则把 32×32 的原图缩放到 size×size（size=32 时不变）。
    distort > 0 时每个字符先做随机仿射形变（render_glyph 的 distort 参数，第 10 章）。
    样本按类别交错排列（第 0 类、第 1 类、……、第 0 类……），不需要打乱也不会出现"一整批都是同一类"。
    """
    from .imgops import resize_area
    X, y = [], []
    for _ in range(n_per_class):
        for k, ch in enumerate(chars):
            img, _ = render_glyph(ch, rng, noise=noise, distort=distort)
            img = normalize_glyph(img, size) if align else (img if size == 32 else resize_area(img, size, size))
            X.append(to_float(img, invert=True))
            y.append(k)
    X = np.stack(X).astype(np.float32)
    return (X.reshape(len(X), -1) if flat else X), np.array(y, dtype=np.int64)


def shape_sizes(rng, n=500):
    """ShapesWorld 里只放一个物体，返回 (area, side, shape_id)：
    area 是非背景像素的个数，side 是物体的真实边长（像素），shape_id 是形状在 SHAPES 中的下标。
    用于第 7 章的线性回归：能不能从"面积"猜出"边长"？"""
    areas, sides, kinds = [], [], []
    for _ in range(n):
        s = int(rng.integers(SIZES["small"][0], SIZES["large"][1] + 1))
        x0, y0 = int(rng.integers(1, 64 - s - 1)), int(rng.integers(1, 64 - s - 1))
        k = int(rng.integers(len(SHAPES)))
        obj = {"shape": SHAPES[k], "color": str(rng.choice(list(COLORS))), "size": "large", "box": (x0, y0, x0 + s, y0 + s)}
        img = draw_scene([obj], background=(255, 255, 255))
        areas.append(int((img != 255).any(-1).sum()))
        sides.append(s)
        kinds.append(k)
    return np.array(areas, dtype=np.float64), np.array(sides, dtype=np.float64), np.array(kinds)
