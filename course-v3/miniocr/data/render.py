"""第 2 章：把字符、文字行和公式画成灰度图，同时记录每个符号的边界框。

约定（全书通用）：
    图像是 uint8 的二维数组，形状 (H, W)；255 是白色背景，0 是黑色墨迹。
    坐标 (x, y)：x 向右增大，y 向下增大，原点在左上角。
    边界框 (x0, y0, x1, y1)：左上角 (x0, y0)，右下角 (x1, y1)，右、下边界不含（和 Python 切片一样）。
        所以 img[y0:y1, x0:x1] 就是框里的像素。

字体：用 Pillow 自带的字体（ImageFont.load_default），任何电脑上画出来都一样。
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

_FONTS = {}


def font(size):
    """取一个指定字号的字体（缓存起来，避免重复加载）。"""
    size = int(size)
    if size not in _FONTS:
        _FONTS[size] = ImageFont.load_default(size=size)
    return _FONTS[size]


def ink_box(img, thresh=128):
    """图里所有"墨迹"像素（比 thresh 暗）的外接框；没有墨迹时返回 None。"""
    ys, xs = np.nonzero(img < thresh)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def add_noise(arr, rng, noise):
    """加高斯噪声：每个像素加上均值 0、标准差 noise×255 的随机数，再截断到 [0, 255]。"""
    if noise <= 0:
        return arr
    out = arr.astype(np.float32) + rng.normal(0, noise * 255, arr.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


# ====================================================================== 单个字符
def distort_affine(im, rng, amount):
    """以图像中心为原点做一次随机仿射形变（第 10 章）：旋转 ±12°、错切 ±0.25、横纵各自缩放 1±0.15，都乘以 amount。

    正向变换 A = 旋转 · 错切 · 缩放，把原图上的点 p 送到 c + A(p − c)。
    Pillow 的 transform 要的是反方向：对输出的每个像素 q，告诉它去原图的哪里取值，即 p = c + A⁻¹(q − c)。
    用双线性插值取值（第 2 章），所以形变后的灰度是连续的，几乎不会画出两张完全相同的图。"""
    W, H = im.size
    th = math.radians(rng.uniform(-12, 12) * amount)
    sh = rng.uniform(-0.25, 0.25) * amount
    sx, sy = 1 + rng.uniform(-0.15, 0.15) * amount, 1 + rng.uniform(-0.15, 0.15) * amount
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    A = R @ np.array([[1.0, sh], [0.0, 1.0]]) @ np.diag([sx, sy])
    (a, b), (d, e) = np.linalg.inv(A)
    cx, cy = W / 2, H / 2
    coeffs = (a, b, cx - a * cx - b * cy, d, e, cy - d * cx - e * cy)
    return im.transform((W, H), Image.AFFINE, coeffs, resample=Image.BILINEAR, fillcolor=255)


def render_glyph(ch, rng, size=32, noise=0.0, distort=0.0):
    """在 size×size 的白底上画一个字符。字号、位置、粗细都随机，模拟同一个字符的不同"写法"。

    distort > 0 时（第 10 章起），先在 2 倍大的画布中央画字、做 distort_affine 形变，
    再裁出墨迹、随机放进 size×size 的格子（放不下时等比缩小）。distort=0 时与第 2 章的行为完全相同。
    返回 (img, box)：img 形状 (size, size)；box 是字符墨迹的外接框。
    """
    fs = int(rng.integers(int(size * 0.55), int(size * 0.8) + 1))
    f = font(fs)
    if distort > 0:
        return _render_glyph_distorted(ch, rng, size, noise, distort, f)
    im = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(im)
    x0, y0, x1, y1 = draw.textbbox((0, 0), ch, font=f)            # 先量出字符的大小
    w, h = x1 - x0, y1 - y0
    dx = int(rng.integers(0, max(1, size - w))) - x0               # 随机平移，但保证不出界
    dy = int(rng.integers(0, max(1, size - h))) - y0
    stroke = int(rng.random() < 0.3)                               # 30% 的概率加粗
    draw.text((dx, dy), ch, font=f, fill=0, stroke_width=stroke, stroke_fill=0)
    arr = add_noise(np.array(im, dtype=np.uint8), rng, noise)
    return arr, ink_box(np.array(im, dtype=np.uint8))


def _render_glyph_distorted(ch, rng, size, noise, distort, f):
    big = Image.new("L", (2 * size, 2 * size), 255)
    draw = ImageDraw.Draw(big)
    x0, y0, x1, y1 = draw.textbbox((0, 0), ch, font=f)
    stroke = int(rng.random() < 0.3)
    draw.text((size - (x0 + x1) / 2, size - (y0 + y1) / 2), ch, font=f, fill=0, stroke_width=stroke, stroke_fill=0)
    big = distort_affine(big, rng, distort)
    bx0, by0, bx1, by1 = ink_box(np.array(big)) or (0, 0, 1, 1)
    piece = big.crop((bx0, by0, bx1, by1))
    w, h = piece.size
    if max(w, h) > size:                                           # 形变后放不下：等比缩小
        k = size / max(w, h)
        piece = piece.resize((max(1, round(w * k)), max(1, round(h * k))), Image.BILINEAR)
        w, h = piece.size
    im = Image.new("L", (size, size), 255)
    im.paste(piece, (int(rng.integers(0, size - w + 1)), int(rng.integers(0, size - h + 1))))
    clean = np.array(im, dtype=np.uint8)
    return add_noise(clean, rng, noise), ink_box(clean)


def render_line(text, rng, H=32, W=256, size=None, noise=0.0, distort=0.0):
    """把一行文字画成 H×W 的图。返回 (img, boxes)：boxes[i] 是第 i 个字符的框（空格为 None）。

    如果文字太长放不下，抛出 ValueError（调用者应该换一段更短的文字）。
    distort > 0 时（第 13 章）再做 distort_line 形变；boxes 仍是形变前的框，只能作参考。
    distort=0 时与以前的行为完全相同。
    """
    size = size or int(rng.integers(int(H * 0.5), int(H * 0.7) + 1))
    f = font(size)
    width = f.getlength(text)
    if width > W - 4:
        raise ValueError(f"文字太长：宽 {width:.0f} 像素，图只有 {W} 像素")
    im = Image.new("L", (W, H), 255)
    draw = ImageDraw.Draw(im)
    x = int(rng.integers(2, int(W - width) - 1))
    base = int(H * 0.72) + int(rng.integers(-1, 2))                 # 基线：字母"站"在这条线上
    boxes = []
    for ch in text:
        if ch == " ":
            boxes.append(None)
        else:
            b = draw.textbbox((x, base), ch, font=f, anchor="ls")
            draw.text((x, base), ch, font=f, fill=0, anchor="ls")
            boxes.append((int(b[0]), int(b[1]), int(b[2]), int(b[3])))
        x += f.getlength(ch)
    if distort > 0:
        im = distort_line(im, rng, distort)
    return add_noise(np.array(im, dtype=np.uint8), rng, noise), boxes


def distort_line(im, rng, amount):
    """一行文字的随机"拍摄"形变（第 13 章），各项都乘以 amount：
    旋转 ±2°、错切 ±0.35（像斜体）、纵向缩放 1±0.15、高斯模糊半径 0–1.2 像素、墨迹浓度降到 60%–100%。
    行图很扁（256×32），旋转角稍大一点两端就会出界，所以角度比单个字符（±12°）小得多。"""
    W, H = im.size
    th = math.radians(rng.uniform(-2, 2) * amount)
    sh = rng.uniform(-0.35, 0.35) * amount
    sy = 1 + rng.uniform(-0.15, 0.15) * amount
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    A = R @ np.array([[1.0, sh], [0.0, 1.0]]) @ np.diag([1.0, sy])
    (a, b), (d, e) = np.linalg.inv(A)
    cx, cy = W / 2, H / 2
    im = im.transform((W, H), Image.AFFINE, (a, b, cx - a * cx - b * cy, d, e, cy - d * cx - e * cy),
                      resample=Image.BILINEAR, fillcolor=255)
    im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0, 1.2) * amount))
    ink = 1 - rng.uniform(0, 0.4) * amount                          # 墨迹浓度：1 = 纯黑
    arr = 255 - (255 - np.asarray(im, dtype=np.float32)) * ink
    return Image.fromarray(arr.astype(np.uint8))


# ====================================================================== 公式：盒子模型排版
class Box:
    """排版的基本单位：一个矩形"盒子"。

        w     宽度
        asc   基线以上的高度（ascent）
        desc  基线以下的深度（descent）
        cmds  要画的东西，坐标相对于（盒子左端, 基线）：
              ("t", x, y, 字符, 字号)    在 (x, y) 处画字符，y 是该字符的基线
              ("l", x0, x1, y)           从 x0 到 x1 画一条水平线（分数线、根号顶上的横线）
              ("p", [(x, y), ...])       画一条折线（根号的勾）
    """

    def __init__(self, w, asc, desc, cmds):
        self.w, self.asc, self.desc, self.cmds = w, asc, desc, cmds

    def shifted(self, dx, dy):
        """把盒子里所有绘图命令平移 (dx, dy)，用于把子盒子放进父盒子。"""
        out = []
        for c in self.cmds:
            if c[0] == "t":
                out.append(("t", c[1] + dx, c[2] + dy, c[3], c[4]))
            elif c[0] == "l":
                out.append(("l", c[1] + dx, c[2] + dx, c[3] + dy))
            else:
                out.append(("p", [(x + dx, y + dy) for x, y in c[1]]))
        return out


def _glyph_box(ch, size, pad):
    f = font(size)
    l, t, r, b = f.getbbox(ch, anchor="ls")                        # 相对于基线起点的墨迹范围
    return Box(f.getlength(ch) + pad, -t, max(0, b), [("t", pad / 2, 0, ch, size)])


def layout(node, size):
    """把语法树排成一个盒子。字号 size 决定了整体大小；上下标、分子分母用更小的字号。"""
    kind = node[0]
    if kind == "sym":
        return _glyph_box(node[1], size, 1)
    if kind == "op":
        return _glyph_box(node[1], size, max(4, size // 3))           # 运算符两边留白多一些
    if kind == "row":
        x, cmds, asc, desc = 0.0, [], 0.0, 0.0
        for c in node[1]:
            b = layout(c, size)
            cmds += b.shifted(x, 0)
            x += b.w
            asc, desc = max(asc, b.asc), max(desc, b.desc)
        return Box(x, asc, desc, cmds)
    if kind == "script":
        _, base, sub, sup = node
        bb = layout(base, size)
        small = max(7, round(size * 0.7))
        cmds, w, asc, desc = list(bb.cmds), bb.w, bb.asc, bb.desc
        sw = 0.0
        if sub is not None:                                          # 先下标后上标，与 leaves() 的顺序一致
            sb = layout(sub, small)
            dy = round(size * 0.28)                                   # 下标的基线压低
            cmds += sb.shifted(bb.w, dy)
            sw, desc = max(sw, sb.w), max(desc, sb.desc + dy)
        if sup is not None:
            sb = layout(sup, small)
            dy = -round(size * 0.42)                                  # 上标的基线抬高
            cmds += sb.shifted(bb.w, dy)
            sw, asc = max(sw, sb.w), max(asc, sb.asc - dy)
        return Box(w + sw, asc, desc, cmds)
    if kind == "frac":
        s = max(8, round(size * 0.8))
        num, den = layout(node[1], s), layout(node[2], s)
        w = max(num.w, den.w) + 4
        axis = round(size * 0.3)                                      # 分数线在基线以上 axis 像素
        gap = 2
        cmds = [("l", 1, w - 1, -axis)]
        cmds += num.shifted((w - num.w) / 2, -axis - gap - num.desc)
        cmds += den.shifted((w - den.w) / 2, -axis + gap + den.asc)
        return Box(w, axis + gap + num.desc + num.asc, -axis + gap + den.asc + den.desc, cmds)
    if kind == "sqrt":
        inner = layout(node[1], size)
        hook = max(5, size // 3)                                      # 根号左边那个"勾"的宽度
        top = -inner.asc - 2
        bottom = inner.desc + 1
        cmds = [("p", [(0, (top + bottom) / 2 + 1), (hook * 0.4, bottom), (hook, top)]),
                ("l", hook, hook + inner.w + 1, top)]
        cmds += inner.shifted(hook + 1, 0)
        return Box(hook + inner.w + 2, -top + 1, bottom, cmds)
    raise ValueError(f"未知的节点类型: {kind}")


def formula_size(node, size=16):
    """公式排版后的 (宽, 高)，用来判断放不放得进图里。"""
    b = layout(node, size)
    return b.w, b.asc + b.desc


def render_formula(node, rng, H=48, W=192, size=16, noise=0.0):
    """把公式的语法树画成 H×W 的图。

    返回 (img, boxes)。boxes 按 latex.leaves(node) 的顺序列出每个符号的 (字符, 框)，
    所以第 i 个框就是第 i 个叶子符号所在的位置（第 22、25、34 章会用到）。
    放不下时抛出 ValueError。
    """
    box = layout(node, size)
    if box.w > W - 4 or box.asc + box.desc > H - 2:
        raise ValueError(f"公式放不下：{box.w:.0f}×{box.asc + box.desc:.0f}，图是 {W}×{H}")
    im = Image.new("L", (W, H), 255)
    draw = ImageDraw.Draw(im)
    x0 = int(rng.integers(2, max(3, int(W - box.w) - 1)))
    lo = int(np.ceil(box.asc)) + 1                                    # 基线最高能放在哪
    hi = int(H - np.ceil(box.desc)) - 1                               # 基线最低能放在哪
    y0 = int(rng.integers(lo, max(lo + 1, hi + 1)))
    boxes = []
    for c in box.cmds:
        if c[0] == "t":
            x, y = x0 + c[1], y0 + c[2]
            b = draw.textbbox((x, y), c[3], font=font(c[4]), anchor="ls")
            draw.text((x, y), c[3], font=font(c[4]), fill=0, anchor="ls")
            boxes.append((c[3], (int(b[0]), int(b[1]), int(b[2]), int(b[3]))))
        elif c[0] == "l":
            y = round(y0 + c[3])
            draw.line((x0 + c[1], y, x0 + c[2], y), fill=0, width=1)
        else:
            draw.line([(x0 + x, y0 + y) for x, y in c[1]], fill=0, width=1)
    return add_noise(np.array(im, dtype=np.uint8), rng, noise), boxes


# ====================================================================== 查看工具
def to_ascii(img, chars=" .:-=+*#%@"):
    """把灰度图变成字符画，在终端里直接"看"图（越黑的像素用越密的字符）。"""
    levels = len(chars) - 1
    rows = []
    for row in img:
        rows.append("".join(chars[(255 - int(v)) * levels // 255] for v in row))
    return "\n".join(rows)


def save_grid(imgs, path, scale=3, cols=4, pad=2):
    """把若干张同样大小的图拼成一张大图并放大保存，便于肉眼检查。支持灰度 (H,W) 和彩色 (H,W,3)。"""
    h, w = imgs[0].shape[:2]
    color = imgs[0].ndim == 3
    rows = (len(imgs) + cols - 1) // cols
    shape = (rows * (h + pad), cols * (w + pad)) + ((3,) if color else ())
    canvas = np.full(shape, 160, dtype=np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas[r * (h + pad):r * (h + pad) + h, c * (w + pad):c * (w + pad) + w] = im
    big = Image.fromarray(canvas).resize((canvas.shape[1] * scale, canvas.shape[0] * scale), Image.NEAREST)
    big.save(path)
    return path


def sample_formula(rng, level=3, H=48, W=192, size=16, noise=0.0, max_tries=100):
    """随机生成一个**放得进图里**的公式，返回 (tree, img, boxes)。放不下就重新生成。"""
    from miniocr.data.latex import gen_formula
    for _ in range(max_tries):
        tree = gen_formula(rng, level)
        try:
            img, boxes = render_formula(tree, rng, H=H, W=W, size=size, noise=noise)
            return tree, img, boxes
        except ValueError:
            continue
    raise RuntimeError(f"连续 {max_tries} 次都放不下：图太小或 level 太高")
