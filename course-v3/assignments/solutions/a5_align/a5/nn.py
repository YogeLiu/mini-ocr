"""A5 参考实现：基础层（从 A3 复制）。"""
import math

import torch
import torch.nn as nn


def softmax(x, dim=-1):
    x = x - x.amax(dim=dim, keepdim=True)
    e = torch.exp(x)
    return e / e.sum(dim=dim, keepdim=True)


def log_softmax(x, dim=-1):
    x = x - x.amax(dim=dim, keepdim=True)
    return x - torch.log(torch.exp(x).sum(dim=dim, keepdim=True))


def log_sigmoid(x):
    """log σ(x) = −log(1 + e^{−x})，按 x 的正负分开算，大的 |x| 也不会溢出。"""
    return -(torch.clamp(-x, min=0) + torch.log1p(torch.exp(-x.abs())))


def silu(x):
    return x * torch.sigmoid(x)


def gelu(x):
    """精确版 GELU：x · Φ(x) = x · ½(1 + erf(x/√2))。"""
    return 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))


def cross_entropy(logits, targets, ignore_index=-100):
    logits = logits.reshape(-1, logits.shape[-1])
    targets = targets.reshape(-1)
    keep = targets != ignore_index
    logp = log_softmax(logits[keep].float(), dim=-1)
    return -logp.gather(1, targets[keep][:, None]).mean()


class Linear(nn.Module):
    def __init__(self, d_in, d_out, bias=True, std=0.02):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_out, d_in) * std)
        self.bias = nn.Parameter(torch.zeros(d_out)) if bias else None

    def forward(self, x):
        y = x @ self.weight.T
        return y if self.bias is None else y + self.bias


class Embedding(nn.Module):
    def __init__(self, n, d, std=0.02):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n, d) * std)

    def forward(self, idx):
        return self.weight[idx]


class LayerNorm(nn.Module):
    """y = (x − mean) / √(var + eps) · weight + bias，对最后一维，方差用有偏估计。"""

    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(var + self.eps) * self.weight + self.bias


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).to(dt) * self.weight


def scaled_dot_product_attention(q, k, v, mask=None):
    s = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf"))
    return softmax(s, dim=-1) @ v


def rope_tables(T, dh, base=10000.0, device=None):
    inv = base ** (-torch.arange(0, dh, 2, dtype=torch.float32, device=device) / dh)
    ang = torch.arange(T, dtype=torch.float32, device=device)[:, None] * inv[None]
    ang = torch.cat([ang, ang], -1)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x.chunk(2, -1)
    return x * cos + torch.cat([-x2, x1], -1) * sin


class MultiHeadAttention(nn.Module):
    """非因果多头注意力（带偏置）。kv 给出时是交叉注意力：q 来自 x，k、v 来自 kv。
    key_mask (B, Tk) 为 True 的位置是真实的 token，False 是填充（不许看）。cos/sin 给出时对 q、k 加 RoPE（只用于自注意力）。"""

    def __init__(self, d, n_head):
        super().__init__()
        self.h, self.dh = n_head, d // n_head
        self.q = Linear(d, d)
        self.kv = Linear(d, 2 * d)
        self.o = Linear(d, d)

    def forward(self, x, kv=None, key_mask=None, cos=None, sin=None):
        B, T, d = x.shape
        src = x if kv is None else kv
        q = self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k, v = self.kv(src).view(B, src.shape[1], 2, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if cos is not None:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        mask = None if key_mask is None else key_mask[:, None, None, :]
        y = scaled_dot_product_attention(q, k, v, mask)
        return self.o(y.transpose(1, 2).reshape(B, T, d))
