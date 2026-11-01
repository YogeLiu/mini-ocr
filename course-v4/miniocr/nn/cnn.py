"""第 28 章/A5：透明的卷积机制、几何工具和 BatchNorm。

手写卷积使用 NumPy NCHW、OIHW，做互相关（与 PyTorch Conv2d 一致），
不调用卷积/unfold/autograd；训练实验使用单独的优化实现以节省教学等待时间。
"""
import numpy as np
import torch
from torch import nn


# >>> 练习 28.1：输出尺寸与感受野
def output_size(size, kernel, stride=1, padding=0, dilation=1):
    """一维输出尺寸 floor((size+2p-d(k-1)-1)/s)+1；二维分别调用。"""
    if any(not isinstance(v, int) for v in (size, kernel, stride, padding, dilation)):
        raise ValueError("geometry arguments must be integers")
    if min(size, kernel, stride, dilation) < 1 or padding < 0:
        raise ValueError("invalid convolution geometry")
    result = (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
    if result < 1:
        raise ValueError("kernel exceeds padded input")
    return result


def receptive_field(layers):
    """各层 (kernel, stride, padding, dilation) → 理论感受野、跳距、首中心。

    输入像素中心从 0.5 开始；池化也看作一个局部核。不是学习后的有效感受野。
    """
    field, jump, center = 1, 1, .5
    rows = []
    for kernel, stride, padding, dilation in layers:
        output_size(max(1, dilation * (kernel - 1) + 1), kernel, stride, padding, dilation)
        effective = dilation * (kernel - 1) + 1
        center += ((effective - 1) / 2 - padding) * jump
        field += (effective - 1) * jump
        jump *= stride
        rows.append(dict(field=field, jump=jump, center=center))
    return rows
# <<<


# >>> 练习 28.2：卷积前向
def conv_forward(x, weight, bias, stride=1, padding=0):
    """NCHW 前向；返回 (输出, backward 所需缓存)，支持矩形核、整型步长/padding。"""
    if x.ndim != 4 or weight.ndim != 4 or x.shape[1] != weight.shape[1]:
        raise ValueError("expected NCHW and OIHW with matching channels")
    n, _, h, w = x.shape
    filters, _, kh, kw = weight.shape
    if bias.shape != (filters,):
        raise ValueError("bias must have shape (out_channels,)")
    oh = output_size(h, kh, stride, padding)
    ow = output_size(w, kw, stride, padding)
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    out = np.empty((n, filters, oh, ow), dtype=np.result_type(x, weight, bias))
    for i in range(oh):
        for j in range(ow):
            patch = xp[:, :, i*stride:i*stride+kh, j*stride:j*stride+kw]
            out[:, :, i, j] = np.einsum("nchw,fchw->nf", patch, weight) + bias
    return out, (x.copy(), weight.copy(), stride, padding)
# <<<


# >>> 练习 28.3：卷积反向
def conv_backward(dout, cache):
    """显式链式法则，重叠窗口的 dx 必须累加；返回 dx, dw, db。"""
    x, weight, stride, padding = cache
    kh, kw = weight.shape[-2:]
    expected = (x.shape[0], weight.shape[0], output_size(x.shape[2], kh, stride, padding),
                output_size(x.shape[3], kw, stride, padding))
    if dout.shape != expected:
        raise ValueError("dout shape does not match forward output")
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    dxp, dw = np.zeros_like(xp), np.zeros_like(weight)
    for i in range(dout.shape[2]):
        for j in range(dout.shape[3]):
            a, b = i * stride, j * stride
            g = dout[:, :, i, j]
            dw += np.einsum("nf,nchw->fchw", g, xp[:, :, a:a+kh, b:b+kw])
            dxp[:, :, a:a+kh, b:b+kw] += np.einsum("nf,fchw->nchw", g, weight)
    return dxp[:, :, padding:padding+x.shape[2], padding:padding+x.shape[3]], dw, dout.sum((0, 2, 3))
# <<<


def numerical_gradient(function, x, eps=1e-5):
    """float64 中心差分；function 返回标量，逐坐标扰动后恢复输入。"""
    if x.dtype != np.float64:
        raise ValueError("gradient check requires float64")
    grad = np.empty_like(x)
    for idx in np.ndindex(x.shape):
        old = x[idx]
        try:
            x[idx] = old + eps
            plus = function(x)
            x[idx] = old - eps
            minus = function(x)
            grad[idx] = (plus - minus) / (2 * eps)
        finally:
            x[idx] = old
    return grad


# >>> 练习 28.4：BatchNorm 的统计量、训练/推理状态
class ManualBatchNorm2d(nn.Module):
    """按 (N,H,W) 归一化各通道；前向有偏方差、running 无偏方差。

    momentum 是新观测的权重。仅反向链式法则交给 torch autograd；
    不调用现成的 batch_norm。与 nn.BatchNorm2d 默认语义一致。
    """
    def __init__(self, channels, eps=1e-5, momentum=.1):
        super().__init__()
        if channels < 1 or eps <= 0 or not 0 <= momentum <= 1:
            raise ValueError("invalid BatchNorm arguments")
        self.eps, self.momentum = eps, momentum
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != len(self.weight):
            raise ValueError("expected NCHW with matching channels")
        if self.training:
            count = x.numel() // x.shape[1]
            if count <= 1:
                raise ValueError("BatchNorm training needs more than one value per channel")
            mean = x.mean((0, 2, 3))
            var = ((x - mean[None, :, None, None]) ** 2).mean((0, 2, 3))
            with torch.no_grad():
                self.running_mean.lerp_(mean.detach(), self.momentum)
                self.running_var.lerp_(var.detach() * count / (count - 1), self.momentum)
                self.num_batches_tracked.add_(1)
        else:
            mean, var = self.running_mean, self.running_var
        normalized = (x - mean[None, :, None, None]) * torch.rsqrt(var[None, :, None, None] + self.eps)
        return normalized * self.weight[None, :, None, None] + self.bias[None, :, None, None]
# <<<


class SymbolClassifier(nn.Module):
    """第 5 章同任务的 CNN/MLP；输入 N1×32×32，输出 24 类 logits。

    CNN 两层局部卷积后池化；MLP 两层隐藏层，参数量由调用者报告。
    训练层使用 torch 卷积，不混入用于梯度检查的慢速机制实现。
    """
    def __init__(self, kind="cnn", width=24, classes=24):
        super().__init__()
        if kind == "cnn":
            self.net = nn.Sequential(nn.Conv2d(1, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                     nn.Conv2d(width, 2*width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                     nn.Flatten(), nn.Linear(2*width*8*8, classes))
        elif kind == "mlp":
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(1024, 3*width), nn.ReLU(),
                                     nn.Linear(3*width, 3*width), nn.ReLU(), nn.Linear(3*width, classes))
        else:
            raise ValueError("kind must be cnn or mlp")

    def forward(self, x):
        return self.net(x)
