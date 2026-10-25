"""A4 参考实现：ViT（从 A3 复制）。"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .nn import LayerNorm, Linear, MultiHeadAttention, gelu


def patchify(x, p):
    """(B, C, H, W) → (B, h·w, C·p·p)。patch 按行排列（第 i 行第 j 列是第 i·w + j 个），每个 patch 内部按 (C, p, p) 拉平。"""
    B, C, H, W = x.shape
    h, w = H // p, W // p
    x = x[:, :, :h * p, :w * p].reshape(B, C, h, p, w, p).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(B, h * w, C * p * p)


class PatchEmbed(nn.Module):
    """切块 + 线性层。weight (d, C·p·p) 与核大小 = 步长 = p 的 Conv2d 权重 reshape 后相同。"""

    def __init__(self, in_ch, d, p):
        super().__init__()
        self.p = p
        self.proj = Linear(in_ch * p * p, d, std=math.sqrt(1 / (in_ch * p * p)))

    def forward(self, x):
        return self.proj(patchify(x, self.p)), (x.shape[2] // self.p, x.shape[3] // self.p)


def sincos_1d(n, d, base=10000.0):
    """第 2i 维 sin(t·ω_i)，第 2i+1 维 cos(t·ω_i)，ω_i = base^(−2i/d)。"""
    t = torch.arange(n, dtype=torch.float32)[:, None]
    omega = base ** (-torch.arange(0, d, 2, dtype=torch.float32) / d)
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(t * omega)
    pe[:, 1::2] = torch.cos(t * omega)
    return pe


def sincos_2d(h, w, d):
    """(h·w, d)：前 d/2 维是行号的一维编码，后 d/2 维是列号的一维编码。"""
    row, col = sincos_1d(h, d // 2), sincos_1d(w, d // 2)
    return torch.cat([row[:, None].expand(h, w, d // 2), col[None].expand(h, w, d // 2)], -1).reshape(h * w, d)


def _interp_axis(x, n_out, dim):
    """沿 dim 做一维线性插值，坐标约定与 F.interpolate(mode="bilinear", align_corners=False) 相同。"""
    n_in = x.shape[dim]
    src = ((torch.arange(n_out, dtype=torch.float64) + 0.5) * (n_in / n_out) - 0.5).clamp(min=0)
    i0 = src.floor().long().clamp(max=n_in - 1)
    i1 = (i0 + 1).clamp(max=n_in - 1)
    t = (src - i0).to(x.dtype)
    shape = [1] * x.dim()
    shape[dim] = n_out
    t = t.view(shape)
    return x.index_select(dim, i0) * (1 - t) + x.index_select(dim, i1) * t


def interpolate_pos(pos, old_hw, new_hw):
    """pos (old_h·old_w, d) 看成 d 通道的图，双线性缩放到 new_hw，再拉平回 (new_h·new_w, d)。"""
    if tuple(old_hw) == tuple(new_hw):
        return pos
    g = pos.reshape(old_hw[0], old_hw[1], -1)
    g = _interp_axis(_interp_axis(g, new_hw[0], 0), new_hw[1], 1)
    return g.reshape(new_hw[0] * new_hw[1], -1)


def rope_2d(h, w, dh, base=100.0, device=None):
    """cos、sin 形状 (h·w, dh)。dh/4 个频率 θ_k = base^(−k/(dh/4))；第 i·w + j 个 patch 的角度表
    ang = cat([i·θ, j·θ])（长 dh/2），再按"前后两半"的约定重复一次：cat([ang, ang])。"""
    q = dh // 4
    theta = base ** (-torch.arange(q, dtype=torch.float32, device=device) / q)
    ii, jj = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    ang = torch.cat([ii.reshape(-1, 1).float() * theta, jj.reshape(-1, 1).float() * theta], -1)
    ang = torch.cat([ang, ang], -1)
    return ang.cos(), ang.sin()


class MLP(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.fc1, self.fc2 = Linear(d, hidden), Linear(hidden, d)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))


class ViTBlock(nn.Module):
    """x ← x + Attn(LN₁(x))；x ← x + MLP(LN₂(x))。注意力非因果，可选 RoPE 和填充 mask。"""

    def __init__(self, d, n_head, mlp_ratio=4.0):
        super().__init__()
        self.n1, self.n2 = LayerNorm(d), LayerNorm(d)
        self.attn = MultiHeadAttention(d, n_head)
        self.mlp = MLP(d, int(d * mlp_ratio))

    def forward(self, x, key_mask=None, cos=None, sin=None):
        x = x + self.attn(self.n1(x), key_mask=key_mask, cos=cos, sin=sin)
        return x + self.mlp(self.n2(x))


@dataclass
class ViTConfig:
    image_size: int = 64
    patch: int = 8
    in_ch: int = 3
    d: int = 192
    n_layer: int = 4
    n_head: int = 4
    pos: str = "rope2d"              # "learned"、"sincos" 或 "rope2d"
    n_classes: int = 0               # > 0 时带一个分类头（对 patch 求平均后分类）


class ViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = PatchEmbed(cfg.in_ch, cfg.d, cfg.patch)
        g = cfg.image_size // cfg.patch
        self.grid = (g, g)
        if cfg.pos == "learned":
            self.pos = nn.Parameter(torch.randn(g * g, cfg.d) * 0.02)
        self.blocks = nn.ModuleList(ViTBlock(cfg.d, cfg.n_head) for _ in range(cfg.n_layer))
        self.norm = LayerNorm(cfg.d)
        self.head = Linear(cfg.d, cfg.n_classes) if cfg.n_classes else None

    def patch_tokens(self, x):
        """(B, C, H, W) → ((B, h·w, d), (h, w))。图的大小可以与训练时不同。"""
        t, hw = self.embed(x)
        cos = sin = None
        if self.cfg.pos == "learned":
            t = t + interpolate_pos(self.pos, self.grid, hw)
        elif self.cfg.pos == "sincos":
            t = t + sincos_2d(*hw, self.cfg.d).to(t.device, t.dtype)
        else:
            cos, sin = rope_2d(*hw, self.cfg.d // self.cfg.n_head, device=t.device)
        for b in self.blocks:
            t = b(t, cos=cos, sin=sin)
        return self.norm(t), hw

    def forward(self, x):
        t, _ = self.patch_tokens(x)
        pooled = t.mean(1)
        return self.head(pooled) if self.head is not None else pooled
