"""第 24 章：真实 VLM 的输入管线。

自己实现 Qwen3.5 / Qwen2.5-VL 和 SmolVLM（Idefics3）processor 的关键步骤，
再在 labs/ch24 里与 transformers 的实现逐项对照。
本模块只依赖 numpy / torch / PIL，不需要下载任何模型就能测试。
"""
import math

import numpy as np
import torch
from PIL import Image

QWEN = "Qwen/Qwen3.5-0.8B"
SMOL = "HuggingFaceTB/SmolVLM-256M-Instruct"


# ---------------------------------------------------------------- Qwen：缩放

# >>> 练习 24.1：smart_resize
def smart_resize(h, w, factor=32, min_pixels=256 * 256, max_pixels=4096 * 4096):
    """把 (h, w) 变成 factor 的整数倍，总像素落在 [min_pixels, max_pixels]，尽量保持宽高比。

    factor = patch_size × merge_size（Qwen3.5 是 16 × 2 = 32），这样合并后的网格正好是整数。
    规则（与 Qwen2-VL 的 smart_resize 一致）：
      1. 先各自四舍五入到 factor 的倍数；
      2. 太大：按 beta = sqrt(h·w / max_pixels) 缩小，再向下取整到 factor 的倍数（至少 factor）；
      3. 太小：按 beta = sqrt(min_pixels / (h·w)) 放大，再向上取整到 factor 的倍数。
    返回 (new_h, new_w)。
    """
    if max(h, w) / min(h, w) > 200:
        raise ValueError("宽高比不能超过 200")
    h_bar = round(h / factor) * factor
    w_bar = round(w / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    return h_bar, w_bar
# <<<


def to_tensor(img, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """PIL RGB → (3, H, W) float32：先除以 255，再减均值、除以标准差。"""
    x = torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (x - m) / s


# ---------------------------------------------------------------- Qwen：切块

# >>> 练习 24.2：Qwen 的切块
def qwen_patchify(x, patch=16, merge=2, temporal=2):
    """(C, H, W) → pixel_values (N, C·temporal·patch·patch) 和 grid (1, H/patch, W/patch)。

    两个细节和第 19 章的 patchify 不同：
      - 顺序：不是逐行扫描所有 patch，而是先按 merge×merge 的窗口分组（窗口逐行扫描，
        窗口内部再逐行），这样后面的 merger 只要把相邻的 merge² 行拼起来就完成了 2×2 合并；
      - 时间维：静态图复制 temporal 份，假装是 temporal 帧的视频，好和视频共用一个 Conv3d。
    每一行的元素顺序是 (C, temporal, patch, patch)。
    """
    C, H, W = x.shape
    gh, gw = H // patch, W // patch
    t = x.reshape(C, gh // merge, merge, patch, gw // merge, merge, patch)
    t = t.permute(1, 4, 2, 5, 0, 3, 6)                    # (gh/m, gw/m, m, m, C, p, p)
    t = t.unsqueeze(5).expand(-1, -1, -1, -1, -1, temporal, -1, -1)
    pv = t.reshape(gh * gw, C * temporal * patch * patch)
    return pv, torch.tensor([[1, gh, gw]])
# <<<


def qwen_preprocess(img, patch=16, merge=2, temporal=2, min_pixels=256 * 256, max_pixels=4096 * 4096,
                    mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """一张 PIL 图 → (pixel_values, image_grid_thw)。缩放用 PIL 的双三次插值，和 transformers 的 PIL 后端一致。"""
    h, w = smart_resize(img.height, img.width, patch * merge, min_pixels, max_pixels)
    img = img.convert("RGB").resize((w, h), Image.BICUBIC)
    return qwen_patchify(to_tensor(img, mean, std), patch, merge, temporal)


def n_qwen_tokens(h, w, patch=16, merge=2, **kw):
    """原图 h×w 在 LLM 里占多少个视觉 token。"""
    rh, rw = smart_resize(h, w, patch * merge, **kw)
    return (rh // patch) * (rw // patch) // (merge * merge)


# ---------------------------------------------------------------- Qwen：占位符

# >>> 练习 24.3：展开占位符
def expand_image_tokens(ids, grids, image_id, merge=2):
    """把 ids 里第 k 个 image_id 展开成 grids[k] 对应的 t·h·w / merge² 个 image_id。

    聊天模板只给每张图放一个 <|image_pad|>；processor 在分词前（或后）把它复制成正确的个数，
    模型里再把这些位置的嵌入替换成视觉特征（第 22 章的 VLM.embed）。
    """
    out, k = [], 0
    for t in ids:
        if t == image_id:
            if k == len(grids):
                raise ValueError(f"占位符比图像（{len(grids)} 张）多")
            n = int(torch.as_tensor(grids[k]).prod()) // (merge * merge)
            out += [image_id] * n
            k += 1
        else:
            out.append(t)
    if k != len(grids):
        raise ValueError(f"占位符 {k} 个，图像 {len(grids)} 张")
    return out
# <<<


# ---------------------------------------------------------------- Qwen：M-RoPE

# >>> 练习 24.4：M-RoPE 位置编号
def mrope_positions(ids, grids, image_id, merge=2):
    """Qwen2-VL 系列的 3D 位置编号，返回 (3, L)，三行分别是 时间 / 行 / 列。

    - 文字 token：三行相同，都是 pos，然后 pos += 1；
    - 一张图（t, gh, gw，合并后 h = gh/merge，w = gw/merge）的 t·h·w 个 token：
        时间 = pos + ti，行 = pos + 行号，列 = pos + 列号（按 t, 行, 列 的顺序展开）；
      整张图结束后 pos += max(h, w)，而不是 += h·w。
    结果：图后面文字的编号只比图前面多 max(h, w)，一张 1000 个 token 的图不会把文字"推远" 1000 个位置。
    ids 里的 image_id 必须已经展开过。
    """
    L = len(ids)
    pos_ids = torch.zeros(3, L, dtype=torch.long)
    pos, i, k = 0, 0, 0
    while i < L:
        if ids[i] != image_id:
            pos_ids[:, i] = pos
            pos += 1
            i += 1
            continue
        t, gh, gw = (int(v) for v in torch.as_tensor(grids[k]).flatten())
        h, w = gh // merge, gw // merge
        tt, hh, ww = torch.meshgrid(torch.arange(t), torch.arange(h), torch.arange(w), indexing="ij")
        n = t * h * w
        pos_ids[:, i:i + n] = torch.stack([tt.flatten(), hh.flatten(), ww.flatten()]) + pos
        pos += max(h, w)
        i += n
        k += 1
    return pos_ids
# <<<


def mrope_axes(n_freq, section, interleaved):
    """每个频率分给哪个轴（'T' / 'H' / 'W'）。n_freq = 旋转维度 / 2，section 的和应等于 n_freq。

    - 分段式（Qwen2-VL、Qwen2.5-VL）：前 section[0] 个频率给时间，接着 section[1] 个给行，再 section[2] 个给列；
    - 交错式（Qwen3-VL、Qwen3.5）：T H W T H W …，每个轴都分到从高频到低频的整个频段。
    """
    assert sum(section) == n_freq
    if not interleaved:
        return ["T"] * section[0] + ["H"] * section[1] + ["W"] * section[2]
    axes = ["T"] * n_freq
    for i in range(1, 3 * section[1], 3):
        axes[i] = "H"
    for i in range(2, 3 * section[2], 3):
        axes[i] = "W"
    return axes


# ---------------------------------------------------------------- SmolVLM（Idefics3）

def idefics3_layout(h, w, longest=2048, tile=512, seq=64):
    """SmolVLM 的切图规则，返回 (行数, 列数, 视觉 token 数)。

    1. 长边缩放到 longest（小图会被放大），短边按比例、取偶数；
    2. 两边各自向上取整到 tile 的倍数；
    3. 如果比一块大，就切成 行×列 块 tile×tile 的子图，再加一张缩到 tile×tile 的全局图；
    4. 每块经过 SigLIP（patch 16 → 32×32）和 pixel shuffle（4×4 合并）后是 seq = 64 个 token。
    """
    if w >= h:
        nw = longest
        nh = int(nw / (w / h))
        nh += nh % 2
    else:
        nh = longest
        nw = int(nh * (w / h))
        nw += nw % 2
    if nw >= nh:
        W = math.ceil(nw / tile) * tile
        H = math.ceil(int(W / (nw / nh)) / tile) * tile
    else:
        H = math.ceil(nh / tile) * tile
        W = math.ceil(int(H * (nw / nh)) / tile) * tile
    if H > tile or W > tile:
        rows, cols = H // tile, W // tile
        return rows, cols, (rows * cols + 1) * seq
    return 0, 0, seq


# ---------------------------------------------------------------- 配置

def count_params(cfg):
    """在 meta 设备上按配置建模型（不分配内存、不需要权重），统计各部分参数量。"""
    from transformers import AutoModelForImageTextToText

    with torch.device("meta"):
        m = AutoModelForImageTextToText.from_config(cfg)
    parts = {"vision": 0, "connector": 0, "lm": 0}
    for n, p in m.named_parameters():
        if any(k in n for k in ("merger", "multi_modal_projector", "connector")):
            parts["connector"] += p.numel()
        elif any(k in n for k in ("visual", "vision_tower", "vision_model")):
            parts["vision"] += p.numel()
        else:
            parts["lm"] += p.numel()
    return parts
