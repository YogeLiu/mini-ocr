"""第 9–11 章：从零写一个 GPT（解码器式 Transformer）。结构与 Qwen3 / Llama 相同，只是小得多。

    RMSNorm                    只除以均方根、不减均值的归一化
    rope_cache / apply_rope    旋转位置编码 RoPE：把位置"转"进 q 和 k
    CausalSelfAttention        因果多头自注意力 + RoPE；可选 GQA（几个 q 头共用一组 k、v）、Q/K norm、KV cache
    SwiGLU                     门控 MLP：down(silu(gate(x)) ⊙ up(x))
    GPT                        token 嵌入 → n_layer 个 Pre-Norm 块 → RMSNorm → 输出层（可与嵌入共享权重）

两个可选项是给第 17 章准备的：Qwen3 用 GQA（n_kv_head < n_head）和 Q/K norm（对每个头的 q、k 先做 RMSNorm 再加 RoPE）。
默认关闭，第 11 章的小 GPT 就是最朴素的多头注意力。
attn_impl="manual" 用第 8 章自己写的 attention；"sdpa" 用 PyTorch 的融合实现（快，结果在浮点误差内相同）。
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nn.attention import attention


@dataclass
class GPTConfig:
    vocab_size: int = 2048
    block_size: int = 256          # 最长上下文
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int = 0             # 0 表示等于 n_head（普通多头）；小于 n_head 即 GQA，必须能整除 n_head
    d_model: int = 256
    d_ff: int = 0                  # SwiGLU 中间维；0 表示取 ≈ 8/3 · d 并取整到 64 的倍数
    rope_base: float = 10000.0
    rope: bool = True              # False：不加任何位置信息（第 10 章的对照实验）
    qk_norm: bool = False          # Qwen3 的做法
    dropout: float = 0.0
    tie_weights: bool = True
    attn_impl: str = "sdpa"        # "manual" 或 "sdpa"
    pos_emb: str = "rope"          # "rope" / "none" / "learned"；旧 rope=False 优先，强制无位置
    norm: str = "rmsnorm"          # "layernorm" 含 beta；默认不新增 state_dict 键
    resid_scale: bool = True       # 残差输出初始化 std = 0.02 / sqrt(2L)

    def kv_heads(self):
        return self.n_kv_head or self.n_head

    def ff_dim(self):
        return self.d_ff or 64 * math.ceil(8 * self.d_model / 3 / 64)


# >>> 练习 9.1：RMSNorm
class RMSNorm(nn.Module):
    """y = x / √(mean(x²) + eps) · γ，对最后一维；γ 初始化为 1。与 LayerNorm 相比：不减均值、没有 β。
    在 float32 里算均方根再转回输入的 dtype（bf16 下直接算平方和容易损失精度）。"""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight).to(x.dtype)
# <<<


# >>> 练习 10.1：RoPE
def rope_cache(T, dh, base=10000.0, device=None):
    """返回 cos, sin，形状都是 (T, dh)。
    频率 θ_i = base^(−2i/dh)，i = 0 … dh/2 − 1；位置 t 的角度 t·θ_i。
    按 Qwen / Llama 的"前后两半"约定，角度表重复两次：ang = cat([t·θ, t·θ])，cos = cos(ang)，sin = sin(ang)。"""
    if dh <= 0 or dh % 2:
        raise ValueError("RoPE 的 dh 必须是正偶数")
    inv = base ** (-torch.arange(0, dh, 2, dtype=torch.float32, device=device) / dh)    # (dh/2,)
    ang = torch.arange(T, dtype=torch.float32, device=device)[:, None] * inv[None]       # (T, dh/2)
    ang = torch.cat([ang, ang], -1)
    return ang.cos(), ang.sin()


def rotate_half(x):
    """[x1, x2] → [−x2, x1]，x1、x2 是最后一维的前一半和后一半。"""
    if x.shape[-1] <= 0 or x.shape[-1] % 2:
        raise ValueError("RoPE 的 dh 必须是正偶数")
    x1, x2 = x.chunk(2, -1)
    return torch.cat([-x2, x1], -1)


def apply_rope(x, cos, sin):
    """x 形状 (..., T, dh)，cos/sin 形状 (T, dh)。把第 i 维和第 i + dh/2 维看成一个二维平面上的点，
    转过角度 t·θ_i：x · cos + rotate_half(x) · sin。"""
    return x * cos.to(x.dtype) + rotate_half(x) * sin.to(x.dtype)
# <<<


def repeat_kv(x, n_rep):
    """GQA：(B, h_kv, T, dh) → (B, h_kv · n_rep, T, dh)，每个 k/v 头复制 n_rep 份，给 n_rep 个 q 头共用。"""
    if n_rep == 1:
        return x
    B, h, T, dh = x.shape
    return x[:, :, None].expand(B, h, n_rep, T, dh).reshape(B, h * n_rep, T, dh)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h, self.hk, self.dh = cfg.n_head, cfg.kv_heads(), cfg.d_model // cfg.n_head
        assert cfg.d_model % cfg.n_head == 0 and self.h % self.hk == 0
        self.q = nn.Linear(cfg.d_model, self.h * self.dh, bias=False)
        self.k = nn.Linear(cfg.d_model, self.hk * self.dh, bias=False)
        self.v = nn.Linear(cfg.d_model, self.hk * self.dh, bias=False)
        self.o = nn.Linear(self.h * self.dh, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(self.dh) if cfg.qk_norm else None
        self.k_norm = RMSNorm(self.dh) if cfg.qk_norm else None
        self.dropout, self.impl = cfg.dropout, cfg.attn_impl

    # >>> 练习 11.1：因果自注意力 + RoPE + KV cache
    def forward(self, x, cos, sin, cache=None):
        """x (B, T, d)；cos/sin 是这 T 个新位置的 RoPE 表 (T, dh)。
        1. q = self.q(x) 变形为 (B, h, T, dh)；k、v 变形为 (B, h_kv, T, dh)；
        2. 有 q_norm/k_norm 时先对每个头做 RMSNorm；然后 q、k 加 RoPE（v 不加）；
        3. cache 不为 None 时（第 15 章）：cache 是 dict，可能已有 "k"、"v"（之前所有位置，形状 (B, h_kv, T_past, dh)）。
           把新的 k、v 接在后面，写回 cache。缓存的是 repeat_kv 之前的 k、v——GQA 省显存就省在这里；
        4. repeat_kv 把 k、v 扩到 h 个头；
        5. 注意力：新位置可以看全部历史和自己之前的新位置。mask 形状 (T, T_past + T)，第 i 行允许 j ≤ T_past + i。
           impl == "manual" 时调用 attention(q, k, v, mask)；"sdpa" 时调用 F.scaled_dot_product_attention；
           训练且 dropout > 0 时两条路径都对注意力权重做 dropout（随机数不同，所以对拍时要关掉 dropout）；
        6. 合并头 → (B, T, h·dh)，乘 self.o。"""
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.k(x).view(B, T, self.hk, self.dh).transpose(1, 2)
        v = self.v(x).view(B, T, self.hk, self.dh).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        past = 0
        if cache is not None:
            if "k" in cache:
                past = cache["k"].shape[2]
                k = torch.cat([cache["k"], k], 2)
                v = torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
        k, v = repeat_kv(k, self.h // self.hk), repeat_kv(v, self.h // self.hk)
        mask = torch.ones(T, past + T, dtype=torch.bool, device=x.device).tril(past)
        drop = self.dropout if self.training else 0.0
        if self.impl == "manual":
            y, w = attention(q, k, v, mask)
            if drop > 0:                                   # 与 SDPA 相同：对注意力权重做 dropout
                y = F.dropout(w, drop) @ v
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop)
        return self.o(y.transpose(1, 2).reshape(B, T, self.h * self.dh))
    # <<<


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        f = cfg.ff_dim()
        self.gate = nn.Linear(cfg.d_model, f, bias=False)
        self.up = nn.Linear(cfg.d_model, f, bias=False)
        self.down = nn.Linear(f, cfg.d_model, bias=False)

    # >>> 练习 9.2：SwiGLU 的前向
    def forward(self, x):
        """返回 down(silu(gate(x)) * up(x))，只在 gate 支路使用 SiLU。"""
        return self.down(F.silu(self.gate(x)) * self.up(x))
    # <<<


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        norm = RMSNorm if cfg.norm == "rmsnorm" else nn.LayerNorm
        self.n1, self.n2 = norm(cfg.d_model), norm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    # >>> 练习 9.3：Decoder 块的前向
    def forward(self, x, cos, sin, cache=None):
        """两次 Pre-Norm 残差：先注意力，再前馈；第二次归一化读取更新后的 x。"""
        x = x + self.drop(self.attn(self.n1(x), cos, sin, cache))
        return x + self.drop(self.mlp(self.n2(x)))
    # <<<


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.pos_emb not in ("rope", "none", "learned") or cfg.norm not in ("rmsnorm", "layernorm"):
            raise ValueError("pos_emb 须为 rope/none/learned；norm 须为 rmsnorm/layernorm")
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.rope and cfg.pos_emb == "learned":
            self.pos = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = (RMSNorm if cfg.norm == "rmsnorm" else nn.LayerNorm)(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok.weight             # 输出层与嵌入共享同一个矩阵
        cos, sin = rope_cache(cfg.block_size, cfg.d_model // cfg.n_head, cfg.rope_base)
        if not cfg.rope or cfg.pos_emb != "rope":           # cos = 1、sin = 0：旋转角为 0，apply_rope 什么也不做
            cos, sin = torch.ones_like(cos), torch.zeros_like(sin)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for n, p in self.named_parameters():               # 残差支路的输出层再缩小 √(2L)（GPT-2 的做法）
            if n.endswith("attn.o.weight") or n.endswith("mlp.down.weight"):
                nn.init.normal_(p, 0.0, 0.02 / (math.sqrt(2 * cfg.n_layer) if cfg.resid_scale else 1))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def n_params(self, non_embedding=False):
        """唯一参数量；Kaplan 非嵌入口径扣除 token 与学习式位置嵌入。

        独立输出层保留；共享输出层随 token 嵌入扣除。默认 RoPE 的旧口径不变。
        """
        n = sum(p.numel() for p in self.parameters())     # 共享的权重只算一次
        embeddings = self.tok.weight.numel() + (self.pos.weight.numel() if hasattr(self, "pos") else 0)
        return n - embeddings if non_embedding else n

    def new_caches(self):
        return [dict() for _ in self.blocks]

    # >>> 练习 11.2：GPT 的前向
    def forward(self, idx=None, targets=None, inputs_embeds=None, caches=None, pos0=0, loss_mask=None):
        """idx (B, T) 的 token 编号，或直接给 inputs_embeds (B, T, d)（第 31 章拼接视觉特征时用）。
        pos0：第一个位置的编号（用 KV cache 逐个生成时，新 token 的位置接在历史后面）。
        caches：每层一个 dict 的列表（第 15 章），None 表示不用缓存。
        返回 (logits, loss)。targets (B, T) 给出时 loss = 按 token 平均的交叉熵；targets 中 −100 的位置不计入；
        loss_mask 给出时（bool，形状同 targets）只在 True 的位置计算损失。交叉熵在 float32 里算。"""
        x = self.tok(idx) if inputs_embeds is None else inputs_embeds
        T = x.shape[1]
        assert pos0 + T <= self.cfg.block_size, f"序列长 {pos0 + T} 超过 block_size {self.cfg.block_size}"
        if hasattr(self, "pos"):
            x = x + self.pos(torch.arange(pos0, pos0 + T, device=x.device))
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
