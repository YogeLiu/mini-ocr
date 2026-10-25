"""A6 参考实现：渲染校验。把候选的 LaTeX 重新画出来，和原图的墨迹比。"""
import numpy as np

from miniocr.data.imgops import resize_nearest
from miniocr.data.latex import LatexError, parse
from miniocr.data.render import render_formula


def ink_mask(img, thresh=128, min_neighbors=0):
    """比 thresh 暗、而且 8 个邻居里至少有 min_neighbors 个也暗的像素（边界外当作不暗）。"""
    dark = img < thresh
    if min_neighbors <= 0:
        return dark
    p = np.pad(dark, 1).astype(int)
    H, W = dark.shape
    n = sum(p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W] for dy in (-1, 0, 1) for dx in (-1, 0, 1)) - dark
    return dark & (n >= min_neighbors)


def ink_crop(img, thresh=128, min_neighbors=0):
    """裁到 ink_mask 的外接框；没有墨迹时 None。"""
    ys, xs = np.nonzero(ink_mask(img, thresh, min_neighbors))
    if len(ys) == 0:
        return None
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def blur3(m):
    p = np.pad(m, 1, mode="edge")
    return sum(p[dy:dy + m.shape[0], dx:dx + m.shape[1]] for dy in range(3) for dx in range(3)) / 9


def ncc(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def render_score(latex, img, size=16, thresh=128, shift=2, min_neighbors=2):
    try:
        tree = parse(latex)
        H, W = img.shape
        pred, _ = render_formula(tree, np.random.default_rng(0), H=3 * H, W=3 * W, size=size)
    except (LatexError, ValueError):
        return 0.0
    a, b = ink_crop(pred, thresh), ink_crop(img, thresh, min_neighbors)      # 原图有噪声：孤立的暗点不算墨迹
    if a is None or b is None:
        return 0.0
    h, w = b.shape
    if abs(a.shape[0] - h) > 0.25 * h or abs(a.shape[1] - w) > 0.25 * w:
        return 0.0
    if abs(a.shape[0] - h) > shift or abs(a.shape[1] - w) > shift:
        a = resize_nearest(a, h, w)
    ap = np.pad(a, shift, constant_values=255).astype(float)
    B = blur3(255.0 - b)
    best = -1.0
    for dy in range(ap.shape[0] - h + 1):
        for dx in range(ap.shape[1] - w + 1):
            best = max(best, ncc(blur3(255.0 - ap[dy:dy + h, dx:dx + w]), B))
    return best


def render_select(cands, img, size=16):
    """每个候选的渲染分，返回 (最高分的下标（平局取小的）, 最高分, 全部分数)。"""
    scores = [render_score(c, img, size) for c in cands]
    k = int(np.argmax(scores))
    return k, scores[k], scores
