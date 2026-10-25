"""A2 参考实现：基础层。卷积、池化、BN 从 A1 复制并扩展（池化窗口可以不是正方形）；新增 Embedding、RMSNorm、注意力等。"""
import math

import torch
import torch.nn as nn


def relu(x):
    return x.clamp(min=0)


def silu(x):
    return x * torch.sigmoid(x)


def softmax(x, dim=-1):
    x = x - x.amax(dim=dim, keepdim=True)
    e = torch.exp(x)
    return e / e.sum(dim=dim, keepdim=True)


def log_softmax(x, dim=-1):
    x = x - x.amax(dim=dim, keepdim=True)
    return x - torch.log(torch.exp(x).sum(dim=dim, keepdim=True))


def cross_entropy(logits, targets, ignore_index=-100):
    """logits (..., V)，targets (...)；targets 等于 ignore_index 的位置不计入，其余位置求平均。"""
    logits = logits.reshape(-1, logits.shape[-1])
    targets = targets.reshape(-1)
    keep = targets != ignore_index
    logp = log_softmax(logits[keep].float(), dim=-1)
    return -logp.gather(1, targets[keep][:, None]).mean()


class Linear(nn.Module):
    def __init__(self, d_in, d_out, bias=True, std=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_out, d_in) * (std if std is not None else math.sqrt(2 / d_in)))
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


class RMSNorm(nn.Module):
    """y = x / √(mean(x²) + eps) · weight，对最后一维。"""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).to(dt) * self.weight


def scaled_dot_product_attention(q, k, v, mask=None):
    """q (..., Tq, d)，k、v (..., Tk, d)；mask 是 bool (Tq, Tk) 或能广播的形状，True 表示可以看。"""
    s = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf"))
    return softmax(s, dim=-1) @ v


def rope_tables(T, dh, base=10000.0, device=None):
    """cos、sin 形状 (T, dh)。θ_i = base^(−2i/dh)；按"前后两半"的约定，角度表 cat([t·θ, t·θ])。"""
    inv = base ** (-torch.arange(0, dh, 2, dtype=torch.float32, device=device) / dh)
    ang = torch.arange(T, dtype=torch.float32, device=device)[:, None] * inv[None]
    ang = torch.cat([ang, ang], -1)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """x (..., T, dh)：第 i 维和第 i + dh/2 维组成一个平面，转过位置对应的角度。"""
    x1, x2 = x.chunk(2, -1)
    return x * cos + torch.cat([-x2, x1], -1) * sin


# ---------- 以下从 A1 复制：卷积、池化、BN ----------
def unfold(x, kh, kw, stride=1, padding=0):
    N, C, H, W = x.shape
    if padding:
        xp = x.new_zeros(N, C, H + 2 * padding, W + 2 * padding)
        xp[:, :, padding:padding + H, padding:padding + W] = x
        x = xp
    Ho = (x.shape[2] - kh) // stride + 1
    Wo = (x.shape[3] - kw) // stride + 1
    cols = [x[:, :, i:i + stride * (Ho - 1) + 1:stride, j:j + stride * (Wo - 1) + 1:stride]
            for i in range(kh) for j in range(kw)]
    return torch.stack(cols, 2).reshape(N, C * kh * kw, Ho * Wo)


def conv2d(x, w, b=None, stride=1, padding=0):
    N, _, H, W = x.shape
    O, _, kh, kw = w.shape
    Ho = (H + 2 * padding - kh) // stride + 1
    Wo = (W + 2 * padding - kw) // stride + 1
    y = w.reshape(O, -1) @ unfold(x, kh, kw, stride, padding)
    if b is not None:
        y = y + b[:, None]
    return y.reshape(N, O, Ho, Wo)


class Conv2d(nn.Module):
    def __init__(self, c_in, c_out, k, stride=1, padding=0, bias=True):
        super().__init__()
        self.stride, self.padding = stride, padding
        self.weight = nn.Parameter(torch.randn(c_out, c_in, k, k) * math.sqrt(2 / (c_in * k * k)))
        self.bias = nn.Parameter(torch.zeros(c_out)) if bias else None

    def forward(self, x):
        return conv2d(x, self.weight, self.bias, self.stride, self.padding)


def max_pool2d(x, kh, kw=None):
    """窗口 kh×kw、步长等于窗口（kw 省略时等于 kh）。"""
    kw = kw or kh
    N, C, H, W = x.shape
    Ho, Wo = H // kh, W // kw
    return x[:, :, :Ho * kh, :Wo * kw].reshape(N, C, Ho, kh, Wo, kw).amax(dim=(3, 5))


def batch_norm2d(x, weight, bias, running_mean, running_var, training, momentum=0.1, eps=1e-5):
    if training:
        n = x.numel() / x.shape[1]
        mean = x.mean(dim=(0, 2, 3))
        var = x.var(dim=(0, 2, 3), unbiased=False)
        with torch.no_grad():
            running_mean.mul_(1 - momentum).add_(momentum * mean)
            running_var.mul_(1 - momentum).add_(momentum * var * n / max(n - 1, 1))
    else:
        mean, var = running_mean, running_var
    xh = (x - mean[:, None, None]) / torch.sqrt(var[:, None, None] + eps)
    return xh * weight[:, None, None] + bias[:, None, None]


class BatchNorm2d(nn.Module):
    def __init__(self, c, momentum=0.1, eps=1e-5):
        super().__init__()
        self.momentum, self.eps = momentum, eps
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.register_buffer("running_mean", torch.zeros(c))
        self.register_buffer("running_var", torch.ones(c))

    def forward(self, x):
        return batch_norm2d(x, self.weight, self.bias, self.running_mean, self.running_var,
                            self.training, self.momentum, self.eps)
