"""第 19 章：视觉 Transformer（ViT）。第 20 章在它上面做对比学习，第 22 章把它的输出接到第 16 章的 GPT 前面。

    patchify        把图切成 p×p 的小块，每块拉平成一个向量
    PatchEmbed      "切块 + 线性层"，用核大小 = 步长 = p 的卷积实现
    sincos_2d       固定的二维正弦位置编码（前一半维度编码行，后一半编码列）
    interpolate_pos 把学到的位置编码插值到新的网格大小（换分辨率时用）
    rope_2d         二维 RoPE：头维度的一半按行号旋转，另一半按列号旋转（与 Qwen2-VL 视觉塔的做法相同）
    ViTBlock        Pre-LN 块：LayerNorm → 非因果多头自注意力 → 残差；LayerNorm → GELU MLP → 残差
    ViT             PatchEmbed → 位置信息 →（可选的 CLS token）→ n_layer 个块 → LayerNorm → 池化 → 分类头

图像约定：输入 (B, C, H, W) 的 float 张量；patch 按"从左到右、从上到下"排成序列，第 i 行第 j 列的 patch 是第 i·w + j 个。
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gpt import apply_rope


@dataclass
class ViTConfig:
    img_size: int = 64             # 训练时的边长（学到的位置编码按它建表）
    patch: int = 8
    in_ch: int = 3
    d_model: int = 128
    n_layer: int = 4
    n_head: int = 4
    mlp_ratio: float = 4.0
    pos: str = "learned"           # "learned" | "sincos" | "rope2d" | "none"
    pool: str = "cls"              # "cls"：用 CLS token 的输出；"mean"：所有 patch 输出的平均
    n_classes: int = 0             # 0：没有分类头，forward 返回池化后的向量
    rope_base: float = 100.0       # 网格只有几十格，比文字的 10000 小得多

    def grid(self):
        return self.img_size // self.patch


# >>> 练习 19.1：切块
def patchify(x, p):
    """x (B, C, H, W)，H、W 都是 p 的倍数。返回 (B, N, C·p·p)，N = (H/p)·(W/p)。
    patch 的顺序：先第 0 行从左到右，再第 1 行……；每个 patch 内部按 (通道, 行, 列) 的顺序拉平——
    这与 nn.Conv2d 的权重 (d, C, p, p) 拉平的顺序一致，所以 patchify 之后乘一个线性层 = 核大小和步长都是 p 的卷积。"""
    B, C, H, W = x.shape
    h, w = H // p, W // p
    x = x.reshape(B, C, h, p, w, p)                 # 把 H 拆成 (h, p)、W 拆成 (w, p)
    x = x.permute(0, 2, 4, 1, 3, 5)                 # (B, h, w, C, p, p)
    return x.reshape(B, h * w, C * p * p)
# <<<


class PatchEmbed(nn.Module):
    """(B, C, H, W) → (B, N, d) 和网格大小 (h, w)。"""

    def __init__(self, in_ch, d, p):
        super().__init__()
        self.p = p
        self.proj = nn.Conv2d(in_ch, d, kernel_size=p, stride=p)

    def forward(self, x):
        y = self.proj(x)                            # (B, d, h, w)
        h, w = y.shape[-2:]
        return y.flatten(2).transpose(1, 2), (h, w)


# >>> 练习 19.2：二维位置编码
def sincos_1d(n, d, base=10000.0):
    """(n, d) 的一维正弦位置编码（第 15 章）：第 2i 维 sin(t·ω_i)，第 2i+1 维 cos(t·ω_i)，ω_i = base^(−2i/d)。"""
    t = torch.arange(n, dtype=torch.float32)[:, None]
    omega = base ** (-torch.arange(0, d, 2, dtype=torch.float32) / d)
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(t * omega)
    pe[:, 1::2] = torch.cos(t * omega)
    return pe


def sincos_2d(h, w, d):
    """(h·w, d)。前 d/2 维是行号的一维编码，后 d/2 维是列号的一维编码；第 i·w + j 个位置 = cat(行 i 的编码, 列 j 的编码)。"""
    row = sincos_1d(h, d // 2)                      # (h, d/2)
    col = sincos_1d(w, d // 2)                      # (w, d/2)
    return torch.cat([row[:, None, :].expand(h, w, d // 2), col[None, :, :].expand(h, w, d // 2)], -1).reshape(h * w, d)


def interpolate_pos(pos, old_hw, new_hw):
    """pos (old_h·old_w, d) 是按 old_hw 网格学到的位置编码。把它看成一张 d 通道、old_h × old_w 的"图"，
    用双三次插值缩放到 new_hw，再拉平回 (new_h·new_w, d)。网格相同时原样返回。"""
    if tuple(old_hw) == tuple(new_hw):
        return pos
    d = pos.shape[-1]
    g = pos.reshape(1, old_hw[0], old_hw[1], d).permute(0, 3, 1, 2)
    g = F.interpolate(g, size=tuple(new_hw), mode="bicubic", align_corners=False)
    return g.permute(0, 2, 3, 1).reshape(new_hw[0] * new_hw[1], d)
# <<<


# >>> 练习 19.3：二维 RoPE
def rope_2d(h, w, dh, base=100.0, device=None):
    """返回 cos, sin，形状都是 (h·w, dh)，可以直接交给第 16 章的 apply_rope。
    做法：dh/4 个频率 θ_k = base^(−k/(dh/4))，k = 0 … dh/4 − 1；第 i·w + j 个 patch 的角度表是
        ang = cat([i·θ, j·θ])            （长 dh/2：前 dh/4 个按行号转，后 dh/4 个按列号转）
    再按 apply_rope 的"前后两半"约定重复一次：cat([ang, ang])。
    这样第 m 维和第 m + dh/2 维组成一个平面：m < dh/4 时按行号转，dh/4 ≤ m < dh/2 时按列号转，
    两个 patch 的注意力分数只取决于它们的行差和列差。"""
    q = dh // 4
    theta = base ** (-torch.arange(q, dtype=torch.float32, device=device) / q)
    ii, jj = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    ang = torch.cat([ii.reshape(-1, 1).float() * theta, jj.reshape(-1, 1).float() * theta], -1)   # (h·w, dh/2)
    ang = torch.cat([ang, ang], -1)
    return ang.cos(), ang.sin()
# <<<


class ViTAttention(nn.Module):
    """非因果多头自注意力：每个 patch 都能看所有 patch。qkv 带偏置（ViT、SigLIP 的习惯）。
    keep_attn = True 时把注意力权重 (B, h, N, N) 存到 last_weights，供可视化。
    mask：可选的 (B, N) 布尔张量，True 表示"这个位置是真实的 token"，False 是填充（第 20 章的文字编码器用）；
    填充位置不会被任何 token 看到。"""

    def __init__(self, d, n_head):
        super().__init__()
        self.h, self.dh = n_head, d // n_head
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.keep_attn = False
        self.last_weights = None

    def forward(self, x, cos=None, sin=None, mask=None):
        B, N, d = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)   # 各 (B, h, N, dh)
        if cos is not None:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        am = None if mask is None else mask[:, None, None, :]           # (B, 1, 1, N)：对每个查询都一样
        if self.keep_attn:
            s = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
            if am is not None:
                s = s.masked_fill(~am, float("-inf"))
            w = torch.softmax(s, -1)
            self.last_weights = w.detach()
            y = w @ v
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
        return self.o(y.transpose(1, 2).reshape(B, N, d))


class ViTBlock(nn.Module):
    def __init__(self, d, n_head, mlp_ratio=4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = ViTAttention(d, n_head)
        self.ln2 = nn.LayerNorm(d)
        f = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, f), nn.GELU(), nn.Linear(f, d))

    def forward(self, x, cos=None, sin=None, mask=None):
        x = x + self.attn(self.ln1(x), cos, sin, mask)
        return x + self.mlp(self.ln2(x))


class ViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, g = cfg.d_model, cfg.grid()
        self.patch = PatchEmbed(cfg.in_ch, d, cfg.patch)
        if cfg.pos == "learned":
            self.pos = nn.Parameter(torch.randn(g * g, d) * 0.02)       # 与 patch 嵌入同一量级（第 15 章的教训）
        if cfg.pool == "cls":
            self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.blocks = nn.ModuleList([ViTBlock(d, cfg.n_head, cfg.mlp_ratio) for _ in range(cfg.n_layer)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.n_classes) if cfg.n_classes else None

    # >>> 练习 19.4：ViT 的前向
    def encode(self, x):
        """返回 (tokens, (h, w))。tokens 形状 (B, n_cls + N, d)，已经过最后的 LayerNorm；pool="cls" 时第 0 个是 CLS。
        步骤：
          1. self.patch(x) 得到 (B, N, d) 和网格 (h, w)；
          2. 位置信息：learned → 加 interpolate_pos(self.pos, (g, g), (h, w))（g = cfg.grid()）；
                       sincos → 加 sincos_2d(h, w, d)（放到 x 的设备上）；
                       rope2d → 算 rope_2d(h, w, d_model // n_head, cfg.rope_base)，在每个块的注意力里旋转 q、k；
                       none   → 什么都不加；
          3. pool="cls" 时在序列最前面拼上 self.cls（扩展到 B 份）。rope2d 时 CLS 不旋转：
             在 cos 前面拼一行 1、sin 前面拼一行 0（转 0 度）；
          4. 依次经过 self.blocks（把 cos、sin 传进去，没有就传 None），最后 self.ln。"""
        cfg = self.cfg
        t, (h, w) = self.patch(x)
        B, N, d = t.shape
        cos = sin = None
        if cfg.pos == "learned":
            t = t + interpolate_pos(self.pos, (cfg.grid(), cfg.grid()), (h, w))
        elif cfg.pos == "sincos":
            t = t + sincos_2d(h, w, d).to(t.device)
        elif cfg.pos == "rope2d":
            cos, sin = rope_2d(h, w, d // cfg.n_head, cfg.rope_base, device=t.device)
        if cfg.pool == "cls":
            t = torch.cat([self.cls.expand(B, 1, d), t], 1)
            if cos is not None:
                cos = torch.cat([torch.ones_like(cos[:1]), cos], 0)
                sin = torch.cat([torch.zeros_like(sin[:1]), sin], 0)
        for blk in self.blocks:
            t = blk(t, cos, sin)
        return self.ln(t), (h, w)
    # <<<

    def patch_tokens(self, x):
        """只要 patch 的输出 (B, N, d)（去掉 CLS），第 22 章把它们送进 VLM 的连接器。"""
        t, hw = self.encode(x)
        return (t[:, 1:] if self.cfg.pool == "cls" else t), hw

    def forward(self, x):
        t, _ = self.encode(x)
        z = t[:, 0] if self.cfg.pool == "cls" else t.mean(1)
        return self.head(z) if self.head is not None else z

    def keep_attn(self, on=True):
        for b in self.blocks:
            b.attn.keep_attn = on
