"""第 33 章：渲染校验。把候选答案重新"画"出来，和原图比：画出来一样，答案就（在显示上）对。

    ink_crop       裁到墨迹的外接框
    render_score   公式：解析 → 渲染 → 裁到墨迹、对齐 → 和原图的墨迹比归一化互相关（解析失败或放不下得 0）

和文本指标的区别：两种写法不同、画出来一样的 LaTeX（x^2 与 x^{2}），NED 会扣分，渲染校验不扣。
它也有盲点：画出来几乎一样、含义不同的写法（比如 1 和 l 在某些字体里），以及原图本身有噪声时的误差。
"""
import numpy as np

from ..data.imgops import resize_nearest
from ..data.latex import LatexError, parse
from ..data.render import render_formula


def ink_crop(img, thresh=128):
    """img (H, W) uint8，白底黑字。裁到墨迹（比 thresh 暗的像素）的外接框；没有墨迹时 None。"""
    ys, xs = np.nonzero(img < thresh)
    if len(ys) == 0:
        return None
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _blur(m):
    """3×3 的均值模糊（边缘复制）：让比较对噪声和抗锯齿不那么敏感。"""
    p = np.pad(m, 1, mode="edge")
    return sum(p[dy:dy + m.shape[0], dx:dx + m.shape[1]] for dy in range(3) for dx in range(3)) / 9


def _ncc(a, b):
    """归一化互相关：两张图减去各自的均值之后的余弦相似度，在 [−1, 1] 之间。"""
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


# >>> 练习 33.3：渲染校验
def render_score(latex, img, size=16, thresh=128, shift=2):
    """把 latex 解析、在 3 倍大的白底上渲染（不加噪声，rng 固定为 0），两张图都裁到墨迹（ink_crop）。
    1. 外接框的高或宽相差超过原图的 25%：结构明显不同，得 0；
    2. 相差超过 shift 个像素：把渲染图缩放（resize_nearest）到原图墨迹的大小；
    3. 渲染图四周补 shift 个像素的白边，在所有 (dy, dx) 的平移下取一个和原图墨迹一样大的窗口，
       墨迹强度（255 − 像素）模糊之后和原图的（同样处理）算归一化互相关，取最大值。
    第 3 步的平移搜索是必要的：原图的噪声会让边缘一个抗锯齿的像素越过阈值，外接框偏一个像素，不对齐就比不准。
    解析失败、放不下、任一边没有墨迹：0。"""
    try:
        tree = parse(latex)
        H, W = img.shape
        pred, _ = render_formula(tree, np.random.default_rng(0), H=3 * H, W=3 * W, size=size)
    except (LatexError, ValueError):
        return 0.0
    a, b = ink_crop(pred, thresh), ink_crop(img, thresh)
    if a is None or b is None:
        return 0.0
    h, w = b.shape
    if abs(a.shape[0] - h) > 0.25 * h or abs(a.shape[1] - w) > 0.25 * w:
        return 0.0
    if abs(a.shape[0] - h) > shift or abs(a.shape[1] - w) > shift:
        a = resize_nearest(a, h, w)
    ap = np.pad(a, shift, constant_values=255).astype(float)
    B = _blur(255.0 - b)
    best = -1.0
    for dy in range(ap.shape[0] - h + 1):
        for dx in range(ap.shape[1] - w + 1):
            best = max(best, _ncc(_blur(255.0 - ap[dy:dy + h, dx:dx + w]), B))
    return best
# <<<
