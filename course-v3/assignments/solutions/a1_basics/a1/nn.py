"""A1 参考实现：层与损失。只用张量运算和 autograd，不用 torch.nn 的现成层和 torch.nn.functional。"""
import math

import torch
import torch.nn as nn


def relu(x):
    return x.clamp(min=0)


class Linear(nn.Module):
    """y = x Wᵀ + b。weight (d_out, d_in)，Kaiming 正态初始化 std = √(2 / d_in)；bias 初始为 0。"""

    def __init__(self, d_in, d_out, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_out, d_in) * math.sqrt(2 / d_in))
        self.bias = nn.Parameter(torch.zeros(d_out)) if bias else None

    def forward(self, x):
        y = x @ self.weight.T
        return y if self.bias is None else y + self.bias


def unfold(x, kh, kw, stride=1, padding=0):
    """im2col：(N, C, H, W) → (N, C·kh·kw, L)，排列顺序与 F.unfold 相同（通道在外，窗口内先行后列）。"""
    N, C, H, W = x.shape
    if padding:
        xp = x.new_zeros(N, C, H + 2 * padding, W + 2 * padding)
        xp[:, :, padding:padding + H, padding:padding + W] = x
        x = xp
    Ho = (x.shape[2] - kh) // stride + 1
    Wo = (x.shape[3] - kw) // stride + 1
    cols = [x[:, :, i:i + stride * (Ho - 1) + 1:stride, j:j + stride * (Wo - 1) + 1:stride]
            for i in range(kh) for j in range(kw)]                  # kh·kw 个 (N, C, Ho, Wo)
    return torch.stack(cols, 2).reshape(N, C * kh * kw, Ho * Wo)


def conv2d(x, w, b=None, stride=1, padding=0):
    """二维卷积（互相关）。x (N, C, H, W)，w (O, C, kh, kw)，b (O,) 或 None。"""
    N, _, H, W = x.shape
    O, _, kh, kw = w.shape
    Ho = (H + 2 * padding - kh) // stride + 1
    Wo = (W + 2 * padding - kw) // stride + 1
    y = w.reshape(O, -1) @ unfold(x, kh, kw, stride, padding)      # (N, O, L)
    if b is not None:
        y = y + b[:, None]
    return y.reshape(N, O, Ho, Wo)


class Conv2d(nn.Module):
    """weight (c_out, c_in, k, k)，Kaiming 正态初始化 std = √(2 / (c_in·k·k))；bias 初始为 0。"""

    def __init__(self, c_in, c_out, k, stride=1, padding=0, bias=True):
        super().__init__()
        self.stride, self.padding = stride, padding
        self.weight = nn.Parameter(torch.randn(c_out, c_in, k, k) * math.sqrt(2 / (c_in * k * k)))
        self.bias = nn.Parameter(torch.zeros(c_out)) if bias else None

    def forward(self, x):
        return conv2d(x, self.weight, self.bias, self.stride, self.padding)


def max_pool2d(x, k):
    """k×k 窗口、步长 k 的最大池化；放不下一整个窗口的边缘丢掉（与 F.max_pool2d(x, k) 相同）。"""
    N, C, H, W = x.shape
    Ho, Wo = H // k, W // k
    x = x[:, :, :Ho * k, :Wo * k].reshape(N, C, Ho, k, Wo, k)
    return x.amax(dim=(3, 5))


def batch_norm2d(x, weight, bias, running_mean, running_var, training, momentum=0.1, eps=1e-5):
    """与 F.batch_norm 相同。training=True：用本批的均值和有偏方差归一化，
    并就地更新 running ← (1 − momentum)·running + momentum·本批统计量（方差用无偏估计）；
    training=False：用 running_mean、running_var 归一化。"""
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
    """参数 weight（初始 1）、bias（初始 0）；缓冲区 running_mean（0）、running_var（1）。"""

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


def log_softmax(x, dim=-1):
    """数值稳定：先减去最大值。"""
    x = x - x.amax(dim=dim, keepdim=True)
    return x - torch.log(torch.exp(x).sum(dim=dim, keepdim=True))


def cross_entropy(logits, targets, label_smoothing=0.0):
    """logits (N, K)，targets (N,)。返回批内平均的损失，与 F.cross_entropy(..., label_smoothing=ε) 相同：
    (1 − ε)·(−log p_y) + ε·平均_k(−log p_k)。"""
    logp = log_softmax(logits, dim=-1)
    nll = -logp.gather(1, targets[:, None]).squeeze(1)
    if label_smoothing:
        nll = (1 - label_smoothing) * nll - label_smoothing * logp.mean(dim=-1)
    return nll.mean()
