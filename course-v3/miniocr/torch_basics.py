"""第 9 章：PyTorch 的几个基本部件，先自己写一遍，再和 PyTorch 自带的对照。

    MyLinear        自己写的全连接层（对照 nn.Linear）
    GlyphDataset    把渲染出的字符图包装成 PyTorch 的 Dataset（第 10–12 章都用它）
    softmax_regression_grads   用 autograd 求第 7 章 softmax 回归的梯度，与手推结果对照
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .data.tasks import CHARSET, glyph_arrays


# >>> 练习 9.1：自己写一个全连接层
class MyLinear(nn.Module):
    """y = x Wᵀ + b，与 nn.Linear 的约定相同：weight 形状 (out_features, in_features)，bias 形状 (out_features,)。

    参数要用 nn.Parameter 包起来，nn.Module 才会把它们登记为"可训练参数"
    （出现在 .parameters() 里、被优化器更新、被 .to(device) 搬走、被 state_dict 保存）。
    初始化与 nn.Linear 相同：U(-1/√in, 1/√in)。
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        bound = 1.0 / math.sqrt(in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(out_features).uniform_(-bound, bound)) if bias else None

    def forward(self, x):
        y = x @ self.weight.T
        return y + self.bias if self.bias is not None else y
# <<<


# >>> 练习 9.2：Dataset
class GlyphDataset(Dataset):
    """单字符识别数据集。构造时一次性渲染好全部图像（数据很小，放得进内存）。

    __len__ 返回样本数；__getitem__(i) 返回 (图像张量, 标签)：
        图像 float32，形状 (1, size, size)——通道维在前（第 2 章的 CHW），卷积网络（第 12 章）需要；
        标签 int64 的标量张量。
    """

    def __init__(self, seed, chars=CHARSET, n_per_class=100, size=32, align=False, noise=0.0, distort=0.0):
        X, y = glyph_arrays(np.random.default_rng(seed), chars, n_per_class, size, align, noise, flat=False,
                            distort=distort)
        self.x = torch.from_numpy(X).unsqueeze(1)          # (N, 1, size, size)
        self.y = torch.from_numpy(y)
        self.chars = chars

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]
# <<<


# >>> 练习 9.3：用 autograd 求梯度
def softmax_regression_grads(X, y, W, b):
    """X (N, D)、y (N,)、W (D, K)、b (K,) 都是 NumPy 数组。用 PyTorch 计算交叉熵损失及其对 W、b 的梯度，
    返回 (loss, gW, gb)，都转回 NumPy（float64）。应与第 7 章手推的 softmax_loss_grad 完全一致。

    步骤：把 W、b 变成 requires_grad=True 的 float64 张量 → 前向 → F.cross_entropy（输入是 logits）→ backward → 读 .grad。
    """
    Xt = torch.as_tensor(X, dtype=torch.float64)
    yt = torch.as_tensor(y, dtype=torch.int64)
    Wt = torch.tensor(W, dtype=torch.float64, requires_grad=True)
    bt = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    loss = F.cross_entropy(Xt @ Wt + bt, yt)
    loss.backward()
    return loss.item(), Wt.grad.numpy(), bt.grad.numpy()
# <<<


def pick_device():
    """有 NVIDIA GPU 用 cuda，有 Apple 芯片用 mps，否则用 cpu。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
