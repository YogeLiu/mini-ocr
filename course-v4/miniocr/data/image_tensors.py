"""第 27 章的 RGB 张量、精确面积缩放与可逆坐标 letterbox。

图片值为白=1、黑=0；渲染器的墨迹数组必须先反转。面积缩放按像素
矩形的真实重叠长度加权，不把非整数比例的 adaptive average pooling
称为精确面积积分。仅处理静态图片，不改任何既有数据或评测接口。
"""
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


# >>> 练习 27.1：轴顺序与 uint8 转浮点，不能用 reshape 代替 permute。
def image_to_chw(image, device="cpu"):
    """PIL 图片合成白背景后返回连续 float32 RGB CHW，范围 [0,1]。"""
    rgba = image.convert("RGBA")
    paper = Image.new("RGBA", rgba.size, "white")
    paper.alpha_composite(rgba)
    hwc = torch.from_numpy(np.asarray(paper.convert("RGB")).copy()).to(device)
    return hwc.permute(2, 0, 1).contiguous().float() / 255
# <<<


# >>> 练习 27.2：逐通道归一化和逆变换。
def normalize(chw, mean=(.5, .5, .5), std=(.5, .5, .5), inverse=False):
    """归一化 (x-mean)/std；inverse=True 时回到原像素口径。"""
    if chw.ndim != 3 or len(mean) != chw.shape[0] or len(std) != chw.shape[0]:
        raise ValueError("expected CHW and one mean/std per channel")
    m, s = [torch.as_tensor(v, dtype=chw.dtype, device=chw.device)[:, None, None] for v in (mean, std)]
    if not torch.all(s > 0):
        raise ValueError("std must be positive")
    return chw*s+m if inverse else (chw-m)/s
# <<<


def chw_to_image(chw):
    """[0,1] RGB CHW 量化到 PIL；在这里显式承担 uint8 舍入误差。"""
    array = chw.detach().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


# >>> 练习 27.3：从源像素覆盖长度构造面积权重；每行的和为 1。
def area_weights(source, target, device="cpu"):
    """返回 [target, source] 的面积积分权重，支持非整数缩放。"""
    if min(source, target) <= 0:
        raise ValueError("sizes must be positive")
    starts = torch.arange(target, device=device, dtype=torch.float32)[:, None] * source/target
    pixels = torch.arange(source, device=device, dtype=torch.float32)[None, :]
    overlap = (torch.minimum(starts+source/target, pixels+1)-torch.maximum(starts, pixels)).clamp(min=0)
    return overlap/(source/target)
# <<<


def resize_chw(chw, hw, method="bilinear"):
    """最近邻采用像素中心 nearest-exact；双线性 align_corners=False，无额外抗锯齿。"""
    h, w = hw
    if chw.ndim != 3 or min(h, w) < 1:
        raise ValueError("expected CHW and positive output size")
    if method == "area":
        wy, wx = area_weights(chw.shape[1], h, chw.device), area_weights(chw.shape[2], w, chw.device)
        return torch.einsum("yh,chw,xw->cyx", wy, chw, wx)
    if method not in ("nearest", "bilinear"):
        raise ValueError("method must be nearest, bilinear or area")
    kwargs = dict(align_corners=False) if method == "bilinear" else {}
    return F.interpolate(chw[None], size=(h, w), mode="nearest-exact" if method == "nearest" else method, **kwargs)[0]


# >>> 练习 27.4：保持长宽比、奇数 padding 和从画布返回原坐标。
def transform_image(image, side, method="bilinear", geometry="letterbox", device="cpu"):
    """返回处理后 RGB PIL 与几何元数据；square 拉伸，letterbox 补白。"""
    if side < 1 or geometry not in ("square", "letterbox"):
        raise ValueError("positive side and square/letterbox required")
    x = image_to_chw(image, device)
    h, w = x.shape[1:]
    scale = side/max(h, w)
    rh, rw = (max(1, round(h*scale)), max(1, round(w*scale))) if geometry == "letterbox" else (side, side)
    small = resize_chw(x, (rh, rw), method)
    top, left = (side-rh)//2, (side-rw)//2
    canvas = x.new_ones((3, side, side))
    canvas[:, top:top+rh, left:left+rw] = small
    return chw_to_image(canvas), dict(original_hw=[h, w], resized_hw=[rh, rw], top=top, left=left,
                                     side=side, method=method, geometry=geometry)
# <<<


def restore_coordinates(image, meta, device="cpu"):
    """裁掉已知 padding 后双线性回到原始坐标；仅用于度量，不声称恢复信息。"""
    x = image_to_chw(image, device)
    h, w = meta["resized_hw"]
    y, z = meta["top"], meta["left"]
    return resize_chw(x[:, y:y+h, z:z+w], meta["original_hw"], "bilinear")


def pixel_metrics(original, processed, meta, line_mask=None, device="cpu"):
    """固定原坐标比较：MAE、阈值 0.5 墨迹 IoU/召回；可选已标注分式线召回。

    fraction_line_recall 是原始线像素中，往返变换后仍为深色的比例。
    fraction_line_soft 是这些像素的平均连续墨迹量；二者不会凭空标注真实图片。
    """
    ref = image_to_chw(original, device).mean(0)
    back = restore_coordinates(processed, meta, device).mean(0)
    a, b = ref < .5, back < .5
    union = (a | b).sum().item()
    out = dict(mae=float((ref-back).abs().mean()), ink_iou=float((a & b).sum())/max(1, union),
               ink_recall=float((a & b).sum())/max(1, int(a.sum())))
    if line_mask is not None:
        mask = torch.as_tensor(line_mask, device=device, dtype=torch.bool)
        if mask.shape != ref.shape or not mask.any():
            raise ValueError("line mask must match original and contain pixels")
        out.update(fraction_line_pixels=int(mask.sum()), fraction_line_recall=float(b[mask].float().mean()),
                   fraction_line_soft=float((1-back[mask]).mean()))
    return out


def fraction_bar_mask(image):
    """只用于已知单分式受控图：在中央 60% 高度找最长连续深色水平段。

    返回该行的线像素掩码，要求线段至少占图宽 1/3；不用于自动标注开源图。
    """
    ink = np.asarray(image.convert("L")) < 128
    h, w = ink.shape
    candidates = []
    for y in range(int(h*.2), max(int(h*.2)+1, int(h*.8))):
        padded = np.pad(ink[y].astype(int), (1, 1))
        starts, ends = np.where(np.diff(padded) == 1)[0], np.where(np.diff(padded) == -1)[0]
        candidates.extend((b-a, y, a, b) for a, b in zip(starts, ends))
    length, y, a, b = max(candidates)
    if length < w/3:
        raise ValueError("no sufficiently long controlled fraction bar")
    mask = np.zeros_like(ink)
    mask[y, a:b] = True
    return mask
