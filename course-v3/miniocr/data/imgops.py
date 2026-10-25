"""第 2 章：最基本的图像操作，全部用 NumPy 亲手实现。

真实的 VLM processor（第 24 章）做的事情，本质上就是这几步的组合：
    转成 RGB 或灰度 → 缩放到合适的大小（保持长宽比）→ 转成浮点数并归一化 → 切成 patch。

约定：灰度图 (H, W)，彩色图 (H, W, 3)，dtype 都是 uint8，取值 0–255。
"""
import numpy as np


# >>> 练习 2.3：彩色转灰度
def to_gray(rgb):
    """RGB (H, W, 3) -> 灰度 (H, W)，uint8。

    人眼对绿色最敏感、对蓝色最不敏感，所以不是简单平均，而是加权（ITU-R BT.601 标准）：
        gray = 0.299·R + 0.587·G + 0.114·B
    结果四舍五入到整数。
    """
    rgb = rgb.astype(np.float64)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return np.clip(np.round(gray), 0, 255).astype(np.uint8)
# <<<


def crop(img, box):
    """按边界框 (x0, y0, x1, y1) 裁剪。注意数组下标是 [行, 列] = [y, x]。"""
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1]


# >>> 练习 2.4：最近邻缩放
def resize_nearest(img, out_h, out_w):
    """最近邻插值：输出的每个像素，直接取输入里"对应位置"最近的那个像素。

    输出像素 i 的中心在输出坐标里是 i + 0.5；乘以 in/out 换算到输入坐标；
    这个位置落在哪个输入像素里（向下取整），就取哪个像素。这种"按像素中心对齐"的写法和 Pillow 一致。
    """
    in_h, in_w = img.shape[:2]
    ys = np.floor((np.arange(out_h) + 0.5) * in_h / out_h).astype(int)
    xs = np.floor((np.arange(out_w) + 0.5) * in_w / out_w).astype(int)
    ys = np.clip(ys, 0, in_h - 1)
    xs = np.clip(xs, 0, in_w - 1)
    return img[ys[:, None], xs[None, :]]
# <<<


# >>> 练习 2.5：双线性缩放
def resize_bilinear(img, out_h, out_w):
    """双线性插值：输出像素对应到输入里的一个"小数坐标"，用它周围 4 个像素按距离加权平均。

    一维的情形：位置 2.3 处的值 = 0.7 × 第 2 个像素 + 0.3 × 第 3 个像素。
    二维：先在 x 方向插值两次，再在 y 方向插值一次。
    （只适合放大或小幅缩小；大幅缩小时 Pillow 会先做平均，结果会和这里不同。）
    """
    in_h, in_w = img.shape[:2]
    src = img.astype(np.float64)
    y = (np.arange(out_h) + 0.5) * in_h / out_h - 0.5            # 输出像素中心在输入坐标中的位置
    x = (np.arange(out_w) + 0.5) * in_w / out_w - 0.5
    y = np.clip(y, 0, in_h - 1)
    x = np.clip(x, 0, in_w - 1)
    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x1 = np.minimum(x0 + 1, in_w - 1)
    wy = (y - y0)[:, None]                                        # 到上面那一行的距离，越大越靠近下面
    wx = (x - x0)[None, :]
    if img.ndim == 3:
        wy, wx = wy[..., None], wx[..., None]
    top = src[y0[:, None], x0[None, :]] * (1 - wx) + src[y0[:, None], x1[None, :]] * wx
    bot = src[y1[:, None], x0[None, :]] * (1 - wx) + src[y1[:, None], x1[None, :]] * wx
    out = top * (1 - wy) + bot * wy
    return np.clip(np.round(out), 0, 255).astype(np.uint8)
# <<<


def _area_matrix(in_n, out_n):
    """(out_n, in_n) 的权重矩阵：第 i 行表示输出像素 i 覆盖了哪些输入像素、各覆盖多少（每行加起来为 1）。"""
    M = np.zeros((out_n, in_n))
    step = in_n / out_n                                       # 每个输出像素覆盖 step 个输入像素
    for i in range(out_n):
        a, b = i * step, (i + 1) * step                        # 输出像素 i 在输入坐标里的范围 [a, b)
        for j in range(int(np.floor(a)), min(in_n, int(np.ceil(b)))):
            M[i, j] = min(b, j + 1) - max(a, j)                # 区间 [a,b) 与 [j,j+1) 重叠的长度
    return M / step


# >>> 练习 2.6：区域平均缩放
def resize_area(img, out_h, out_w):
    """区域平均（box filter）：输出像素 = 它覆盖的那一块输入像素的平均值。大幅缩小时用它，细笔画不会"消失"。

    它可以写成两个矩阵乘法：out = Ay · img · Axᵀ，其中 Ay、Ax 是 _area_matrix 给出的权重矩阵
    （先对每一列做竖直方向的平均，再对每一行做水平方向的平均）。
    """
    Ay = _area_matrix(img.shape[0], out_h)
    Ax = _area_matrix(img.shape[1], out_w)
    src = img.astype(np.float64)
    if img.ndim == 3:
        out = np.einsum("ih,hwc,jw->ijc", Ay, src, Ax)
    else:
        out = Ay @ src @ Ax.T
    return np.clip(np.round(out), 0, 255).astype(np.uint8)
# <<<


def normalize_glyph(img, size, thresh=128):
    """经典 OCR 的"字符归一化"：裁出墨迹的外接框，再保持比例缩放、居中放进 size×size 的格子。
    这样不管字符原来在图里的哪个位置、写得多大，归一化后都差不多——位置和大小的差异被消掉了。"""
    ys, xs = np.nonzero(img < thresh)
    if len(xs) == 0:
        return np.full((size, size), 255, dtype=np.uint8)
    c = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    out, _, _ = letterbox(c, size, size, resize=resize_area)
    return out


def letterbox(img, out_h, out_w, fill=255, resize=None):
    """保持长宽比缩放，再用 fill 色把空白补齐到 out_h × out_w（像信封里装照片）。

    返回 (新图, scale, (pad_x, pad_y))。原图里的坐标 (x, y) 在新图里变成 (x·scale + pad_x, y·scale + pad_y)，
    边界框也要按这个公式换算——否则框就"对不上"字了。
    """
    in_h, in_w = img.shape[:2]
    scale = min(out_h / in_h, out_w / in_w)
    nh, nw = max(1, round(in_h * scale)), max(1, round(in_w * scale))
    small = (resize or (resize_area if scale < 1 else resize_bilinear))(img, nh, nw)
    canvas = np.full((out_h, out_w) + img.shape[2:], fill, dtype=np.uint8)
    py, px = (out_h - nh) // 2, (out_w - nw) // 2
    canvas[py:py + nh, px:px + nw] = small
    return canvas, scale, (px, py)


# >>> 练习 2.7：归一化
def to_float(img, mean=None, std=None, invert=False):
    """uint8 -> float32。

    1. 除以 255，变成 [0, 1]；
    2. invert=True 时取 1 - x：让"墨迹"变成大数、背景变成 0（文字图常这样做，第 10 章）；
    3. 给了 mean 和 std 时再做 (x - mean) / std。
       例如 Qwen3.5 的 processor 用 mean = std = 0.5，把 [0, 1] 变成 [-1, 1]（第 24 章）。
    """
    x = img.astype(np.float32) / 255.0
    if invert:
        x = 1.0 - x
    if mean is not None:
        x = (x - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return x
# <<<


def hwc_to_chw(img):
    """(H, W, C) -> (C, H, W)。PyTorch 的卷积层要求通道在前（第 12 章）。"""
    return np.transpose(img, (2, 0, 1))


def draw_boxes(img, boxes, value=0):
    """在灰度图的副本上画出若干个矩形框的边（只画边线），用来检查框画得对不对。"""
    out = img.copy()
    for b in boxes:
        if b is None:
            continue
        x0, y0, x1, y1 = b
        out[y0, x0:x1] = value
        out[y1 - 1, x0:x1] = value
        out[y0:y1, x0] = value
        out[y0:y1, x1 - 1] = value
    return out
