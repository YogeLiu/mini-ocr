"""第 28 章：LoRA。冻结原来的权重 W，只训练一个低秩的增量 ΔW = (α/r)·B·A。

    LoRALinear    包住一个 nn.Linear：y = x Wᵀ + b + (α/r) · (x Aᵀ) Bᵀ；A (r, in) 随机初始化，B (out, r) 初始化为 0
    apply_lora    把模型里名字在 targets 里的 nn.Linear 换成 LoRALinear，并冻结其余所有参数
    merge_lora    把增量并进原权重、换回普通的 nn.Linear（推理时没有额外开销）
    count_params  (总参数, 可训练参数)

形状约定与 nn.Linear 相同：weight 是 (out, in)。所以 A 是 (r, in)、B 是 (out, r)，ΔW = B @ A 也是 (out, in)。
（peft 库的 lora_A、lora_B 就是这两个矩阵，形状相同。）
"""
import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)                       # 冻结原权重（和偏置）
        self.r, self.alpha = r, alpha
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))   # 与 nn.Linear 的默认初始化相同
        self.drop = nn.Dropout(dropout)

    @property
    def scale(self):
        return self.alpha / self.r

    # >>> 练习 28.1：LoRA 的前向
    def forward(self, x):
        """base(x) + scale · (drop(x) @ Aᵀ) @ Bᵀ。先乘 A 再乘 B：中间结果只有 r 维，计算量 ∝ r·(in + out)，
        而不是先算出 B @ A（out × in 的大矩阵）。B 初始为 0，所以刚包上时输出与原来的层完全相同。"""
        return self.base(x) + (self.drop(x) @ self.A.T @ self.B.T) * self.scale
    # <<<

    @torch.no_grad()
    def merged(self):
        """返回一个新的 nn.Linear，权重 = W + scale · B @ A。"""
        lin = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None,
                        device=self.base.weight.device, dtype=self.base.weight.dtype)
        lin.weight.copy_(self.base.weight + self.scale * (self.B @ self.A))
        if self.base.bias is not None:
            lin.bias.copy_(self.base.bias)
        return lin


# >>> 练习 28.2：把 LoRA 装进模型
def apply_lora(model, targets, r=8, alpha=16, dropout=0.0):
    """先冻结 model 的所有参数；再遍历所有子模块，把"自己的名字在 targets 里"的 nn.Linear 子模块
    换成 LoRALinear（用 setattr(父模块, 名字, 新模块)）。返回被替换的层的完整名字列表（如 "blocks.0.attn.qkv"）。
    注意：遍历时先 list(...) 一份，不要边遍历边修改；已经是 LoRALinear 的层不要再包一次；
    最后让模型里所有 LoRALinear 的 A、B 都可训练（包括之前已经装好的）。"""
    for p in model.parameters():
        p.requires_grad_(False)
    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            continue
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                setattr(module, child_name, LoRALinear(child, r, alpha, dropout))
                replaced.append(f"{name}.{child_name}" if name else child_name)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.A.requires_grad_(True)
            module.B.requires_grad_(True)
    return replaced
# <<<


def merge_lora(model):
    """把所有 LoRALinear 换回合并后的 nn.Linear（原地修改），返回合并的层数。"""
    n = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, child_name, child.merged())
                n += 1
    return n


def count_params(model):
    """共享的参数（比如 GPT 的输出层与嵌入共享）只算一次。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
