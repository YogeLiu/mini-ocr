"""A3 参考实现：Qwen3.5 / Qwen2.5-VL 的图像预处理（smart_resize、切块、占位符展开、M-RoPE 位置编号）。"""
import math

import numpy as np
import torch
from PIL import Image


def smart_resize(h, w, factor=32, min_pixels=256 * 256, max_pixels=4096 * 4096):
    """变成 factor 的整数倍，总像素落在 [min_pixels, max_pixels]，尽量保持宽高比（与 Qwen2-VL 的 smart_resize 一致）。"""
    if max(h, w) / min(h, w) > 200:
        raise ValueError("宽高比不能超过 200")
    h_bar, w_bar = round(h / factor) * factor, round(w / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    return h_bar, w_bar


def to_tensor(img, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    x = torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (x - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)


def qwen_patchify(x, patch=16, merge=2, temporal=2):
    """(C, H, W) → pixel_values (N, C·temporal·patch·patch)、grid (1, 3) = [[1, H/patch, W/patch]]。
    patch 先按 merge×merge 的窗口分组（窗口逐行、窗口内逐行），时间维复制 temporal 份，每行按 (C, T, p, p) 拉平。"""
    C, H, W = x.shape
    gh, gw = H // patch, W // patch
    t = x.reshape(C, gh // merge, merge, patch, gw // merge, merge, patch).permute(1, 4, 2, 5, 0, 3, 6)
    t = t.unsqueeze(5).expand(-1, -1, -1, -1, -1, temporal, -1, -1)
    return t.reshape(gh * gw, C * temporal * patch * patch), torch.tensor([[1, gh, gw]])


def qwen_preprocess(img, patch=16, merge=2, temporal=2, min_pixels=256 * 256, max_pixels=4096 * 4096):
    """PIL 图 → (pixel_values, image_grid_thw)。缩放用 PIL 的双三次插值（与 transformers 的 PIL 后端一致）。"""
    h, w = smart_resize(img.height, img.width, patch * merge, min_pixels, max_pixels)
    img = img.convert("RGB").resize((w, h), Image.BICUBIC)
    return qwen_patchify(to_tensor(img), patch, merge, temporal)


def expand_image_tokens(ids, grids, image_id, merge=2):
    """第 k 个 image_id 展开成 t·h·w / merge² 个。占位符和图的个数不一致时抛出 ValueError。"""
    out, k = [], 0
    for t in ids:
        if t != image_id:
            out.append(t)
            continue
        if k == len(grids):
            raise ValueError("占位符比图多")
        out += [image_id] * (int(torch.as_tensor(grids[k]).prod()) // (merge * merge))
        k += 1
    if k != len(grids):
        raise ValueError("占位符比图少")
    return out


def mrope_positions(ids, grids, image_id, merge=2):
    """(3, L)：时间 / 行 / 列。文字三行相同、逐个加 1；一张图的 token 是 pos + (ti, 行号, 列号)，图结束后 pos += max(h, w)。"""
    L = len(ids)
    out = torch.zeros(3, L, dtype=torch.long)
    pos, i, k = 0, 0, 0
    while i < L:
        if ids[i] != image_id:
            out[:, i] = pos
            pos += 1
            i += 1
            continue
        t, gh, gw = (int(v) for v in torch.as_tensor(grids[k]).flatten())
        h, w = gh // merge, gw // merge
        tt, hh, ww = torch.meshgrid(torch.arange(t), torch.arange(h), torch.arange(w), indexing="ij")
        n = t * h * w
        out[:, i:i + n] = torch.stack([tt.flatten(), hh.flatten(), ww.flatten()]) + pos
        pos += max(h, w)
        i += n
        k += 1
    return out
