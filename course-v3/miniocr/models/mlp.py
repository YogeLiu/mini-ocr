"""第 10 章：多层感知机（MLP），单字符识别的第一个"真正的"神经网络。

    输入 (B, 1, H, W) → 拉平 (B, H·W) → [Linear → ReLU → (Dropout)] × 若干 → Linear → logits (B, K)

最后一层不加激活函数：输出 logits，交给 F.cross_entropy（第 6、9 章）。
"""
import torch.nn as nn


# >>> 练习 10.1：MLP
class MLP(nn.Module):
    """sizes = [d_in, h1, h2, ..., d_out]，例如 [1024, 512, 62]。
    相邻两个数之间是一个 nn.Linear；除最后一层外，每个 Linear 后面接 ReLU，dropout > 0 时再接 nn.Dropout(dropout)。
    forward 先把输入拉平（保留第 0 维），再依次通过各层。参数量与第 4 章 mlp_param_count(sizes) 相同。
    """

    def __init__(self, sizes, dropout=0.0):
        super().__init__()
        layers = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(1))
# <<<
