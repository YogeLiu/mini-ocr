"""第 32 章：静态图片 processor 复刻与连接器结构探针。

只依赖 NumPy、Pillow、torch；不下载权重。processor 对照目标是
transformers 5.19 的 Qwen2VLImageProcessorPil（Qwen3.5 也使用它）。
这是静态 RGB 图片路径，不声称支持视频。教学连接器是结构缩小版，
不是 LLaVA/BLIP-2 的预训练权重或完整 Q-Former 实现。
"""
import math
from pathlib import Path
import os
import numpy as np
from PIL import Image
import torch
from torch import nn

MODELS = {
    "qwen35": "Qwen/Qwen3.5-0.8B",
    "smol": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "qwen3vl": "Qwen/Qwen3-VL-2B-Instruct",
}


def local_snapshot(key, cache=None):
    """定位完整本地 safetensors 快照；绝不联网，缺权重返回 None。"""
    root = Path(cache or os.environ.get("HF_HUB_CACHE", Path(os.environ.get(
        "HF_HOME", Path.home()/".cache/huggingface"))/"hub"))
    repo = root / ("models--" + MODELS[key].replace("/", "--"))
    ref = repo / "refs/main"
    candidates = [repo/"snapshots"/ref.read_text().strip()] if ref.exists() else []
    candidates += sorted((repo/"snapshots").glob("*"))
    for path in candidates:
        if (path/"config.json").exists() and list(path.glob("*.safetensors")):
            import json
            index = path/"model.safetensors.index.json"
            if index.exists() and not all((path/f).is_file() for f in set(json.loads(index.read_text())["weight_map"].values())):
                continue
            return path
    return None


# >>> 练习 32.1：动态尺寸，注意 round 的半偶规则与长宽比限制。
def smart_resize(height, width, factor=32, min_pixels=65536, max_pixels=262144):
    """复刻 Qwen smart_resize；像素上下限是面积预算，非边长。"""
    if min(height, width, factor, min_pixels) <= 0 or max_pixels < min_pixels:
        raise ValueError("尺寸与预算必须为正，max_pixels >= min_pixels")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be <= 200")
    h, w = round(height/factor)*factor, round(width/factor)*factor
    if h*w > max_pixels:
        scale = math.sqrt(height*width/max_pixels)
        h, w = [max(factor, math.floor(x/scale/factor)*factor) for x in (height, width)]
    elif h*w < min_pixels:
        scale = math.sqrt(min_pixels/(height*width))
        h, w = [math.ceil(x*scale/factor)*factor for x in (height, width)]
    return h, w
# <<<


# >>> 练习 32.2：归一化、静态时间复制、merge 邻域优先的切块排列。
def replica_processor(image, *, patch_size=16, temporal_patch_size=2, merge_size=2,
                      min_pixels=65536, max_pixels=262144, mean=(.5, .5, .5), std=(.5, .5, .5)):
    """返回 pixel_values[N, C*T*P*P]、image_grid_thw[1,3] 与 resized_hw。

    共享底层 PIL bicubic 插值原语，但尺寸计算、归一化、切块排列独立实现。
    float64 rescale 后转换 float32 对齐 HF PIL 后端的数值顺序。
    """
    image = image.convert("RGB")
    h, w = smart_resize(image.height, image.width, patch_size*merge_size, min_pixels, max_pixels)
    image = image.resize((w, h), resample=Image.Resampling.BICUBIC)
    # HF multiplies by 1/255; retain exactly that arithmetic ordering.
    pixels = (np.asarray(image).astype(np.float64)*(1/255)).astype(np.float32)
    pixels = ((pixels-np.asarray(mean, dtype=np.float32))/np.asarray(std, dtype=np.float32)).transpose(2, 0, 1)
    p, m, t = patch_size, merge_size, temporal_patch_size
    gh, gw = h//p, w//p
    blocks = pixels.reshape(3, gh//m, m, p, gw//m, m, p).transpose(1, 4, 2, 5, 0, 3, 6)
    blocks = np.repeat(blocks[..., None, :, :], t, axis=-3)
    return {"pixel_values": blocks.reshape(gh*gw, 3*t*p*p),
            "image_grid_thw": np.array([[1, gh, gw]], dtype=np.int64), "resized_hw": [h, w]}
# <<<


# >>> 练习 32.3：连接器改变宽度，合并改变序列长度。
class MLPConnector(nn.Module):
    """LLaVA 风格两层 MLP：保持 token 数，将视觉宽度映射到语言宽度。"""
    def __init__(self, vision_dim=32, language_dim=48):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(vision_dim, language_dim), nn.GELU(), nn.Linear(language_dim, language_dim))

    def forward(self, x):
        return self.net(x)


class QueryConnector(nn.Module):
    """Q-Former 结构示意：可学习 query + cross attention + FFN；无完整 BERT 堆叠。"""
    def __init__(self, vision_dim=32, language_dim=48, queries=8, heads=4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, queries, language_dim)*.02)
        self.kv = nn.Linear(vision_dim, language_dim)
        self.attn = nn.MultiheadAttention(language_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(language_dim)
        self.ffn = nn.Sequential(nn.Linear(language_dim, language_dim*2), nn.GELU(), nn.Linear(language_dim*2, language_dim))

    def forward(self, x):
        q = self.queries.expand(x.shape[0], -1, -1)
        kv = self.kv(x)
        z = self.norm(q + self.attn(q, kv, kv, need_weights=False)[0])
        return z + self.ffn(z)
# <<<


def load_image_only_processor(path, key):
    """离线 PIL processor；Qwen 静态图适配器不构造未使用的视频后端。

    transformers 5.19 AutoProcessor 的 backend='pil' 会同时传给视频组件，
    但 Qwen 的视频组件没有 PIL 实现。缩窄构造签名使 get_attributes 只加载
    image_processor/tokenizer；仍复用原版图像 token 展开、模板与校验。
    不 monkeypatch，不更改系统包，不支持视频输入。
    """
    from transformers import AutoProcessor
    if key == "smol":
        return AutoProcessor.from_pretrained(path, local_files_only=True, backend="pil")
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

    class QwenImageOnlyProcessor(Qwen3VLProcessor):
        def __init__(self, image_processor=None, tokenizer=None, chat_template=None):
            super().__init__(image_processor=image_processor, tokenizer=tokenizer,
                             video_processor=None, chat_template=chat_template)

    return QwenImageOnlyProcessor.from_pretrained(path, local_files_only=True, backend="pil")


# >>> 练习 32.4：视觉占位符展开，不能把原始 patch 数当作语言模型 token 数。
def expand_image_placeholder(prompt, image_token, grid, merge_size=2):
    """单图 prompt 中的占位符替换；tokenizer/BPE 仍使用模型官方词表。"""
    if prompt.count(image_token) != 1:
        raise ValueError("静态单图实验要求恰好一个 image placeholder")
    count = int(np.prod(grid))
    if count <= 0 or count % (merge_size**2):
        raise ValueError("grid 的 patch 数必须能按 merge_size 平方合并")
    return prompt.replace(image_token, image_token*(count//merge_size**2))
# <<<
