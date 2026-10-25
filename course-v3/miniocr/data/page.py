"""MiniDoc 的页面（第 21 章起）：标题、正文段落、公式组成的一页，单栏或双栏。

每一页都带完整的标注：
    blocks   版面区块，按**正确的阅读顺序**排列：标题 → 第一栏从上到下 → 第二栏从上到下
             每个区块 {"type": "title" | "text" | "formula", "box": (x0, y0, x1, y1), "lines": [...]}
    lines    区块里的每一行 {"box": 墨迹的外接框, "text": 文字（公式是 LaTeX）}

正文行用第 13 章的 render_line 画（32 像素高、不超过 256 像素宽、同样的字号范围和形变），
所以按框把一行裁出来，就是第 13 章 CRNN 见过的那种文字行图——第 21 章的流水线 OCR 直接用它识别。
"""
import numpy as np

from .latex import to_latex
from .render import add_noise, ink_box, render_line, sample_formula
from .text import WORDS, sample_line_text

BLOCK_TYPES = ("title", "text", "formula")
COL_W = 256                                         # 每一栏的宽度 = 第 13 章文字行的宽度


def _paste(canvas, img, x, y):
    """把 img（白底）"压"到 canvas 上：取两者中更暗的像素。返回 img 墨迹在 canvas 上的框（没有墨迹时 None）。"""
    h, w = img.shape
    region = canvas[y:y + h, x:x + w]
    np.minimum(region, img, out=region)
    b = ink_box(img)
    return None if b is None else (x + b[0], y + b[1], x + b[2], y + b[3])


def _union(boxes, pad=0):
    return (min(b[0] for b in boxes) - pad, min(b[1] for b in boxes) - pad,
            max(b[2] for b in boxes) + pad, max(b[3] for b in boxes) + pad)


def _title_text(rng):
    n = int(rng.integers(2, 5))
    return " ".join(WORDS[int(rng.integers(len(WORDS)))].capitalize() for _ in range(n))


def _text_block(canvas, rng, x, y, y_max, distort):
    """从 (x, y) 开始画一个 2–5 行的段落，不超过 y_max。返回 (区块, 新的 y)；放不下一行时返回 (None, y)。"""
    lines = []
    for _ in range(int(rng.integers(2, 6))):
        if y + 32 > y_max:
            break
        for _ in range(20):
            t = sample_line_text(rng, min_words=2, max_words=6)
            try:
                img, _ = render_line(t, rng, H=32, W=COL_W, distort=distort)
                break
            except ValueError:
                continue
        box = _paste(canvas, img, x, y)
        if box is not None:
            lines.append({"box": box, "text": t})
        y += 32
    if not lines:
        return None, y
    return {"type": "text", "box": _union([l["box"] for l in lines], 3), "lines": lines}, y


def _formula_block(canvas, rng, x, y, y_max):
    if y + 48 > y_max:
        return None, y
    tree, img, _ = sample_formula(rng, level=int(rng.integers(1, 4)), H=48, W=192)
    box = _paste(canvas, img, x + int(rng.integers(0, COL_W - 192 + 1)), y)
    if box is None:
        return None, y + 48
    return {"type": "formula", "box": _union([box], 3), "lines": [{"box": box, "text": to_latex(tree)}]}, y + 48


def sample_page(rng, H=384, W=576, columns=None, noise=0.05, distort=0.5):
    """画一页。columns 为 None 时随机选单栏或双栏（各一半）。返回 {"img": (H, W) uint8, "blocks": [...], "columns": 1 或 2}。
    双栏：两栏各宽 256，左右边距 16，栏间距 32；单栏：一栏宽 256，水平位置随机。"""
    columns = columns or int(rng.integers(1, 3))
    canvas = np.full((H, W), 255, np.uint8)
    blocks = []
    y0 = 16
    xs = [16, 16 + COL_W + 32] if columns == 2 else [int(rng.integers(16, W - COL_W - 16 + 1))]
    if rng.random() < 0.7:                                          # 70% 的页有标题：双栏时在整页宽度内，单栏时在这一栏内
        tw = W - 32 if columns == 2 else COL_W
        for _ in range(20):
            t = _title_text(rng)
            try:
                img, _ = render_line(t, rng, H=40, W=tw, size=int(rng.integers(26, 31)))
                break
            except ValueError:
                continue
        box = _paste(canvas, img, xs[0], y0)
        blocks.append({"type": "title", "box": _union([box], 3), "lines": [{"box": box, "text": t}]})
        y0 += 48
    for x in xs:
        y = y0
        while True:
            y += int(rng.integers(4, 16))                           # 区块之间的空白
            if rng.random() < 0.25:
                b, y = _formula_block(canvas, rng, x, y, H - 16)
            else:
                b, y = _text_block(canvas, rng, x, y, H - 16, distort)
            if b is None:
                break
            blocks.append(b)
    return {"img": add_noise(canvas, rng, noise), "blocks": blocks, "columns": columns}


def all_lines(page, types=("title", "text", "formula")):
    """按阅读顺序列出一页的所有行：[(区块类型, 行框, 文字), ...]。"""
    return [(b["type"], l["box"], l["text"]) for b in page["blocks"] if b["type"] in types for l in b["lines"]]
