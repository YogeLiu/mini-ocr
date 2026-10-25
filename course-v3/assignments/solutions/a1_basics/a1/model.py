"""A1 参考实现：CharCNN 与检查点。"""
import torch
import torch.nn as nn

from .nn import BatchNorm2d, Conv2d, Linear, max_pool2d, relu


class CharCNN(nn.Module):
    """三个"卷积 3×3（无偏置）→ BN → ReLU → 2×2 最大池化"块，再全局平均池化、线性分类。
    32×32 的输入经过三次池化变成 4×4。state_dict 的键：convs.{i}.weight、bns.{i}.{weight,bias,running_mean,running_var}、
    head.weight、head.bias。"""

    def __init__(self, num_classes=62, channels=(32, 64, 128)):
        super().__init__()
        cins = (1,) + tuple(channels[:-1])
        self.convs = nn.ModuleList(Conv2d(a, b, 3, padding=1, bias=False) for a, b in zip(cins, channels))
        self.bns = nn.ModuleList(BatchNorm2d(c) for c in channels)
        self.head = Linear(channels[-1], num_classes)

    def forward(self, x):
        for conv, bn in zip(self.convs, self.bns):
            x = max_pool2d(relu(bn(conv(x))), 2)
        return self.head(x.mean(dim=(2, 3)))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(model, optimizer, iteration, out, extra=None):
    """把模型、优化器、当前步数（以及 extra，例如随机数生成器的状态）存到 out（路径或文件对象）。"""
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "iteration": iteration, "extra": extra}, out)


def load_checkpoint(src, model, optimizer=None):
    """从 src 恢复模型（和优化器），返回 (iteration, extra)。"""
    ck = torch.load(src, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ck["optimizer"])
    return ck["iteration"], ck["extra"]
