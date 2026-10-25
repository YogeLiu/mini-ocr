"""A2 参考实现：GPT（Pre-Norm、RMSNorm、RoPE、SwiGLU、共享嵌入），带 KV cache。"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .nn import Embedding, Linear, RMSNorm, apply_rope, cross_entropy, rope_tables, scaled_dot_product_attention, silu


@dataclass
class GPTConfig:
    vocab_size: int = 2048
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 256
    d_ff: int = 0                  # 0：取 64 · ⌈(8/3)·d_model / 64⌉
    rope_base: float = 10000.0
    tie_weights: bool = True

    def ff_dim(self):
        return self.d_ff or 64 * math.ceil(8 * self.d_model / 3 / 64)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h, self.dh = cfg.n_head, cfg.d_model // cfg.n_head
        self.qkv = Linear(cfg.d_model, 3 * cfg.d_model, bias=False, std=0.02)
        self.o = Linear(cfg.d_model, cfg.d_model, bias=False, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def forward(self, x, cos, sin, cache=None):
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
        mask = torch.ones(T, past + T, dtype=torch.bool, device=x.device).tril(past)   # 新位置看全部历史和自己之前的位置
        y = scaled_dot_product_attention(q, k, v, mask)
        return self.o(y.transpose(1, 2).reshape(B, T, d))


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        f = cfg.ff_dim()
        self.gate = Linear(cfg.d_model, f, bias=False, std=0.02)
        self.up = Linear(cfg.d_model, f, bias=False, std=0.02)
        self.down = Linear(f, cfg.d_model, bias=False, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def forward(self, x):
        return self.down(silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1, self.n2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.n1(x), cos, sin, cache)
        return x + self.mlp(self.n2(x))


class GPT(nn.Module):
    """state_dict 的键：tok.weight、blocks.{i}.{n1,n2}.weight、blocks.{i}.attn.{qkv,o}.weight、
    blocks.{i}.mlp.{gate,up,down}.weight、norm.weight、head.weight。"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model)
        self.head = Linear(cfg.d_model, cfg.vocab_size, bias=False, std=0.02)
        if cfg.tie_weights:
            self.head.weight = self.tok.weight
        cos, sin = rope_tables(cfg.block_size, cfg.d_model // cfg.n_head, cfg.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, caches=None, pos0=0):
        """返回 (logits, loss)。caches：每层一个 dict 的列表；pos0：第一个 token 的位置编号。"""
        T = idx.shape[1]
        assert pos0 + T <= self.cfg.block_size, "超过 block_size"
        x = self.tok(idx)
        cos, sin = self.cos[pos0:pos0 + T], self.sin[pos0:pos0 + T]
        for i, b in enumerate(self.blocks):
            x = b(x, cos, sin, None if caches is None else caches[i])
        logits = self.head(self.norm(x))
        loss = None if targets is None else cross_entropy(logits, targets)
        return logits, loss
