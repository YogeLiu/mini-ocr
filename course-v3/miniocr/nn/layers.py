"""第 11 章：自己写两种归一化层，再和 PyTorch 自带的对照。

    MyLayerNorm     对每个样本的特征维做归一化（Transformer 用它，第 16 章）
    MyBatchNorm1d   对一批样本的每个特征做归一化；训练和评测时行为不同（CNN 常用，第 12 章）
"""
import torch
import torch.nn as nn


# >>> 练习 11.5：LayerNorm
class MyLayerNorm(nn.Module):
    """输入 x 形状 (..., d)。对最后一维：
        μ = mean(x)，σ² = mean((x − μ)²)（除以 d，不是 d−1），x̂ = (x − μ) / √(σ² + eps)，y = γ · x̂ + β。
    γ（weight）初始化为 1，β（bias）初始化为 0，都是长度 d 的可训练参数。与 nn.LayerNorm(d) 相同。
    """

    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        var = ((x - mu) ** 2).mean(-1, keepdim=True)
        return (x - mu) / torch.sqrt(var + self.eps) * self.weight + self.bias
# <<<


# >>> 练习 11.6：BatchNorm
class MyBatchNorm1d(nn.Module):
    """输入 x 形状 (B, d)。与 nn.BatchNorm1d(d) 相同：

    训练模式（self.training 为 True）：
        对每个特征（沿第 0 维）求这一批的 μ 和 σ²（除以 B，有偏），用它们归一化：x̂ = (x − μ)/√(σ² + eps)；
        同时更新滑动平均：running_mean ← (1 − m)·running_mean + m·μ，
                          running_var  ← (1 − m)·running_var  + m·σ²_无偏（除以 B−1）。
    评测模式：不看这一批的统计量，用 running_mean、running_var 归一化（于是每个样本的输出与批里其他样本无关）。
    最后 y = γ · x̂ + β。running_mean/var 用 register_buffer 登记：随 state_dict 保存、随 .to() 搬移，但不是参数。
    """

    def __init__(self, d, momentum=0.1, eps=1e-5):
        super().__init__()
        self.momentum, self.eps = momentum, eps
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))
        self.register_buffer("running_mean", torch.zeros(d))
        self.register_buffer("running_var", torch.ones(d))

    def forward(self, x):
        if self.training:
            mu = x.mean(0)
            var = x.var(0, unbiased=False)
            with torch.no_grad():
                n = x.shape[0]
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mu)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * var * n / max(1, n - 1))
        else:
            mu, var = self.running_mean, self.running_var
        return (x - mu) / torch.sqrt(var + self.eps) * self.weight + self.bias
# <<<
