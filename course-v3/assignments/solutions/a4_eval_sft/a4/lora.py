"""A4 参考实现：LoRA。冻结 W，只训练 ΔW = (α/r)·B·A。包的是自己写的 Linear（weight (out, in)）。"""
import math

import torch
import torch.nn as nn

from .nn import Linear


class LoRALinear(nn.Module):
    """y = base(x) + (α/r)·(x Aᵀ) Bᵀ。A (r, in) ~ U(−1/√in, 1/√in)（与 kaiming_uniform(a=√5) 的范围相同），B (out, r) = 0。"""

    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base, self.r, self.alpha = base, r, alpha
        for p in base.parameters():
            p.requires_grad_(False)
        d_out, d_in = base.weight.shape
        bound = 1 / math.sqrt(d_in)
        self.A = nn.Parameter((torch.rand(r, d_in, device=base.weight.device) * 2 - 1) * bound)
        self.B = nn.Parameter(torch.zeros(d_out, r, device=base.weight.device))

    @property
    def scale(self):
        return self.alpha / self.r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    @torch.no_grad()
    def merged(self):
        d_out, d_in = self.base.weight.shape
        lin = Linear(d_in, d_out, bias=self.base.bias is not None)
        lin.weight.copy_(self.base.weight + self.scale * (self.B @ self.A))
        if self.base.bias is not None:
            lin.bias.copy_(self.base.bias)
        return lin.to(self.base.weight.device)


def apply_lora(model, targets, r=8, alpha=16):
    """冻结全部参数；把名字（子模块自己的名字，如 "qkv"）在 targets 里的 Linear 换成 LoRALinear；
    所有 LoRALinear 的 A、B 可训练。返回被替换的层的完整名字（如 "blocks.0.attn.qkv"）。"""
    for p in model.parameters():
        p.requires_grad_(False)
    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            continue
        for child_name, child in list(module.named_children()):
            if isinstance(child, Linear) and child_name in targets:
                setattr(module, child_name, LoRALinear(child, r, alpha))
                replaced.append(f"{name}.{child_name}" if name else child_name)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.A.requires_grad_(True)
            m.B.requires_grad_(True)
    return replaced


def merge_lora(model):
    n = 0
    for _, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, child_name, child.merged())
                n += 1
    return n


def count_params(model):
    """(总参数, 可训练参数)；共享的参数只算一次（model.parameters() 本来就去重）。"""
    ps = list(model.parameters())
    return sum(p.numel() for p in ps), sum(p.numel() for p in ps if p.requires_grad)
