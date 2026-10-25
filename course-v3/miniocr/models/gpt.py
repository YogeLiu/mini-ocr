"""第 16 章：从零写一个 GPT（解码器式 Transformer），结构与 Qwen / Llama 一致，只是小得多。

    RMSNorm              只除以均方根、不减均值的归一化（Qwen、Llama 用它代替 LayerNorm）
    rope_cache / apply_rope   旋转位置编码 RoPE：把位置信息"转"进 q 和 k
    CausalSelfAttention  因果多头自注意力 + RoPE；可以带 KV cache（第 17 章）
    SwiGLU               门控 MLP：W2 · (silu(W1 x) ⊙ W3 x)
    GPT                  token 嵌入 → n_layer 个 Pre-Norm 块 → RMSNorm → 输出层（与嵌入共享权重）

第 22 章会把视觉特征拼在文字 token 前面送进这个 GPT：forward 接受 inputs_embeds 就是为此准备的。
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 512
    block_size: int = 256          # 最长上下文
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 256
    d_ff: int = 0                  # SwiGLU 中间维；0 表示取 ≈ 8/3 · d 并取整到 64 的倍数
    rope_base: float = 10000.0
    rope: bool = True              # False：不加任何位置信息（第 16 章的对照实验）
    dropout: float = 0.0
    tie_weights: bool = True

    def ff_dim(self):
        return self.d_ff or 64 * math.ceil(8 * self.d_model / 3 / 64)


# >>> 练习 16.1：RMSNorm
class RMSNorm(nn.Module):
    """y = x / √(mean(x²) + eps) · γ，对最后一维；γ 初始化为 1。与 LayerNorm 相比：不减均值、没有 β。"""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
# <<<


# >>> 练习 16.2：RoPE
def rope_cache(T, dh, base=10000.0, device=None):
    """返回 cos, sin，形状都是 (T, dh)。
    频率 θ_i = base^(−2i/dh)，i = 0 … dh/2 − 1；位置 t 的角度 t·θ_i。
    按 Qwen / Llama 的"前后两半"约定，角度表重复两次：ang = cat([t·θ, t·θ])，cos = cos(ang)，sin = sin(ang)。"""
    inv = base ** (-torch.arange(0, dh, 2, dtype=torch.float32, device=device) / dh)    # (dh/2,)
    ang = torch.arange(T, dtype=torch.float32, device=device)[:, None] * inv[None]       # (T, dh/2)
    ang = torch.cat([ang, ang], -1)
    return ang.cos(), ang.sin()


def rotate_half(x):
    """[x1, x2] → [−x2, x1]，x1、x2 是最后一维的前一半和后一半。"""
    x1, x2 = x.chunk(2, -1)
    return torch.cat([-x2, x1], -1)


def apply_rope(x, cos, sin):
    """x 形状 (..., T, dh)，cos/sin 形状 (T, dh)。把第 i 维和第 i + dh/2 维看成一个二维平面上的点，
    转过角度 t·θ_i：x · cos + rotate_half(x) · sin。"""
    return x * cos + rotate_half(x) * sin
# <<<


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h, self.dh = cfg.n_head, cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    # >>> 练习 16.3：因果自注意力 + RoPE
    def forward(self, x, cos, sin, cache=None):
        """x (B, T, d)；cos/sin 是这 T 个位置的 RoPE 表 (T, dh)。
        1. qkv = self.qkv(x)，拆成 q、k、v，各变形为 (B, h, T, dh)；
        2. q、k 加 RoPE（v 不加）；
        3. cache 不为 None 时（第 17 章）：cache 是一个 dict，可能已有 "k"、"v"（之前所有位置的 k、v，
           形状 (B, h, T_past, dh)）。把新的 k、v 接在后面，并写回 cache；
        4. F.scaled_dot_product_attention(q, k, v, is_causal=...)：
           没有历史时用 is_causal=True；有历史时 q 只有新位置，新位置可以看全部历史和自己之前的新位置，
           只有一个新位置（逐个生成）时不需要 mask，多个新位置时需要一个 (T, T_past+T) 的 mask；
        5. 合并头 → (B, T, d)，乘 self.o。"""
        B, T, d = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        past = 0
        if cache is not None:
            if "k" in cache:
                past = cache["k"].shape[2]
                k = torch.cat([cache["k"], k], 2)
                v = torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
        drop = self.dropout if self.training else 0.0
        if past == 0:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop)
        elif T == 1:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=drop)
        else:
            mask = torch.ones(T, past + T, dtype=torch.bool, device=x.device).tril(past)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop)
        return self.o(y.transpose(1, 2).reshape(B, T, d))
    # <<<


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        f = cfg.ff_dim()
        self.gate = nn.Linear(cfg.d_model, f, bias=False)
        self.up = nn.Linear(cfg.d_model, f, bias=False)
        self.down = nn.Linear(f, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1, self.n2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.drop(self.attn(self.n1(x), cos, sin, cache))
        return x + self.drop(self.mlp(self.n2(x)))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok.weight             # 输出层与嵌入共享同一个矩阵
        cos, sin = rope_cache(cfg.block_size, cfg.d_model // cfg.n_head, cfg.rope_base)
        if not cfg.rope:                                   # cos = 1、sin = 0：旋转角为 0，apply_rope 什么也不做
            cos, sin = torch.ones_like(cos), torch.zeros_like(sin)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for n, p in self.named_parameters():               # 残差支路的输出层再缩小 √(2L)（GPT-2 的做法）
            if n.endswith("o.weight") or n.endswith("down.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def n_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())     # 共享的权重只算一次
        return n - self.tok.weight.numel() if non_embedding else n

    # >>> 练习 16.4：GPT 的前向
    def forward(self, idx=None, targets=None, inputs_embeds=None, caches=None, pos0=0, loss_mask=None):
        """idx (B, T) 的 token 编号，或直接给 inputs_embeds (B, T, d)（第 22 章拼接视觉特征时用）。
        pos0：第一个位置的编号（用 KV cache 逐个生成时，新 token 的位置接在历史后面）。
        caches：每层一个 dict 的列表（第 17 章），None 表示不用缓存。
        返回 (logits, loss)。targets (B, T) 给出时 loss = 交叉熵；targets 中 −100 的位置不计入损失（与 PyTorch 约定一致）；
        loss_mask 给出时（bool，形状同 targets）只在 True 的位置计算损失。"""
        x = self.tok(idx) if inputs_embeds is None else inputs_embeds
        T = x.shape[1]
        assert pos0 + T <= self.cfg.block_size, f"序列长 {pos0 + T} 超过 block_size {self.cfg.block_size}"
        cos, sin = self.cos[pos0:pos0 + T], self.sin[pos0:pos0 + T]
        for i, b in enumerate(self.blocks):
            x = b(x, cos, sin, None if caches is None else caches[i])
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            if loss_mask is not None:
                targets = targets.masked_fill(~loss_mask, -100)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1), ignore_index=-100)
        return logits, loss
    # <<<
