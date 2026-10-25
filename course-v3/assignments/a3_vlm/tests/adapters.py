"""A3 的适配器：把你在 assignments/a3_vlm/a3/ 里写的代码接到测试上。这里只做"接线"，不放实质逻辑。"""
from __future__ import annotations

from typing import Callable

import torch
from PIL import Image


# ---------- 第一部分：视觉编码器 ----------
def run_patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    """(B, C, H, W) → (B, h·w, C·p·p)。patch 按行排列，每个 patch 内部按 (C, p, p) 拉平。"""
    raise NotImplementedError


def run_sincos_2d(h: int, w: int, d: int) -> torch.Tensor:
    """(h·w, d)：前 d/2 维编码行号、后 d/2 维编码列号；每一半里第 2i 维 sin(t·ω_i)、第 2i+1 维 cos(t·ω_i)，
    ω_i = 10000^(−2i/(d/2))。"""
    raise NotImplementedError


def run_interpolate_pos(pos: torch.Tensor, old_hw: tuple[int, int], new_hw: tuple[int, int]) -> torch.Tensor:
    """pos (old_h·old_w, d) → (new_h·new_w, d)，与 F.interpolate(mode="bilinear", align_corners=False) 相同。"""
    raise NotImplementedError


def run_rope_2d(h: int, w: int, dh: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (cos, sin)，形状 (h·w, dh)。约定见说明书 §2.3。"""
    raise NotImplementedError


def run_gelu(x: torch.Tensor) -> torch.Tensor:
    """精确版（erf）GELU。"""
    raise NotImplementedError


def run_layer_norm(d: int, weights: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """weights：weight、bias。eps = 1e-5。"""
    raise NotImplementedError


def run_vit_block(d: int, n_head: int, weights: dict[str, torch.Tensor], x: torch.Tensor,
                  key_mask: torch.Tensor | None = None, cos: torch.Tensor | None = None,
                  sin: torch.Tensor | None = None) -> torch.Tensor:
    """一个 Pre-LN ViT 块（mlp_ratio = 4）。weights 的键见说明书 §2.4（n1.*、n2.*、attn.q.*、attn.kv.*、attn.o.*、mlp.fc1.*、mlp.fc2.*）。
    key_mask (B, T)：True = 真实 token。cos/sin 给出时对 q、k 加 RoPE（前后两半的约定）。"""
    raise NotImplementedError


def get_vit(image_size: int, patch: int, d: int, n_layer: int, n_head: int, pos: str) -> torch.nn.Module:
    """新初始化的 ViT（in_ch = 3）。返回的模块要有 patch_tokens(x) → ((B, h·w, d), (h, w))，
    pos ∈ {"learned", "sincos", "rope2d"}，输入图的大小可以与 image_size 不同（但要是 patch 的倍数）。"""
    raise NotImplementedError


# ---------- 第二部分：对比学习 ----------
def run_clip_loss(zi: torch.Tensor, zt: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def run_siglip_loss(zi: torch.Tensor, zt: torch.Tensor, logit_scale: torch.Tensor, logit_bias: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def run_recall_at_k(sim: torch.Tensor, k: int, same: torch.Tensor | None = None) -> float:
    """sim (N, M)，第 i 行是第 i 个查询对所有候选的相似度；same (N, M) bool，默认单位矩阵。"""
    raise NotImplementedError


def run_zero_shot(zi: torch.Tensor, class_zt: torch.Tensor) -> torch.Tensor:
    """(N, D)、(K, D) → 每张图的类别 (N,)。"""
    raise NotImplementedError


# ---------- 第三部分：框 ----------
def run_box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(N, 4)、(M, 4) 的 (x0, y0, x1, y1) → (N, M)。"""
    raise NotImplementedError


def run_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float) -> torch.Tensor:
    """保留的下标（long），按分数从高到低。"""
    raise NotImplementedError


def run_average_precision(preds: list[tuple[torch.Tensor, torch.Tensor]], gts: list[torch.Tensor],
                          iou_thr: float) -> float:
    raise NotImplementedError


# ---------- 第四部分：VLM ----------
def run_merge_2x2(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    raise NotImplementedError


def get_connector(kind: str, d_v: int, d: int) -> torch.nn.Module:
    """kind ∈ {"mlp", "merge", "qformer"}。返回的模块 forward(x (B, N, d_v), hw) → (B, N', d)。qformer 用 16 个查询。"""
    raise NotImplementedError


def run_chat_template(messages: list[dict], n_img: int, encode: Callable[[str], list[int]], bos: int, eos: int,
                      img: int) -> tuple[list[int], list[int]]:
    raise NotImplementedError


def run_scatter_visual(emb: torch.Tensor, visual: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def get_vlm(vocab_size: int, img_id: int, connector: str) -> torch.nn.Module:
    """新初始化的小 VLM：ViT(image_size=32, patch=8, d=32, n_layer=1, n_head=2, pos="rope2d") + 连接器 +
    GPT(vocab_size, block_size=128, n_layer=2, n_head=2, d_model=32)。"""
    raise NotImplementedError


def run_vlm_logits(model: torch.nn.Module, images: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """完整前向（不用 KV cache），返回 (B, T, vocab) 的 logits。"""
    raise NotImplementedError


def run_vlm_generate(model: torch.nn.Module, images: torch.Tensor, prompt: torch.Tensor, max_new: int) -> torch.Tensor:
    """贪心生成 max_new 个 token（不提前停），返回 (B, max_new)。"""
    raise NotImplementedError


# ---------- 第五部分：Qwen 的图像预处理 ----------
def run_smart_resize(h: int, w: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    raise NotImplementedError


def run_qwen_patchify(x: torch.Tensor, patch: int, merge: int, temporal: int) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError


def run_qwen_preprocess(img: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
    """patch=16、merge=2、temporal=2、mean=std=0.5、min_pixels=256²、max_pixels=4096²。"""
    raise NotImplementedError


def run_expand_image_tokens(ids: list[int], grids: list, image_id: int, merge: int) -> list[int]:
    raise NotImplementedError


def run_mrope_positions(ids: list[int], grids: list, image_id: int, merge: int) -> torch.Tensor:
    raise NotImplementedError
