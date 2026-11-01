"""教学用 LoRA：冻结 Linear 基座，训练 A/B；无 dropout、无 bias 更新。

权重布局沿用 torch Linear：[out,in]，因此增量为 alpha/r * B @ A。
只匹配明确的叶模块名，不替换 embedding、Conv1D 或量化 Linear。
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """保留原 Linear；B 零初始化使初始输出严格等于基座。"""

    # >>> 练习 20.1：参数形状、初始化和低秩前向
    def __init__(self, base, rank=4, alpha=8):
        super().__init__()
        if type(base) is not nn.Linear or not isinstance(rank, int) or rank < 1:
            raise ValueError("需要普通 nn.Linear 与正整数 rank")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha 必须为正有限数")
        self.base = base.requires_grad_(False)
        self.rank, self.alpha = rank, alpha
        self.A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    @property
    def scale(self):
        """alpha 只改变增量缩放，不改变参数数目。"""
        return self.alpha / self.rank

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scale
    # <<<

    # >>> 练习 20.2：合并到独立的普通 Linear，不能原地重复加增量
    @torch.no_grad()
    def merged(self):
        """返回独立冻结 Linear，不修改基座或 adapter；低精度合并有舍入误差。"""
        result = nn.Linear(self.base.in_features, self.base.out_features,
                           bias=self.base.bias is not None,
                           device=self.base.weight.device, dtype=self.base.weight.dtype)
        result.weight.copy_(self.base.weight + (self.B @ self.A) * self.scale)
        if result.bias is not None:
            result.bias.copy_(self.base.bias)
        return result.requires_grad_(False).train(self.training)
    # <<<


def inject_lora(model, targets=("q_proj", "v_proj"), rank=4, alpha=8):
    """冻结全模型并按叶名挂载；先校验全部目标，失败不留下部分修改。返回路径列表。"""
    targets = set(targets)
    modules = [(name, m) for name, m in model.named_modules()
               if name and name.rsplit(".", 1)[-1] in targets]
    found = {name.rsplit(".", 1)[-1] for name, _ in modules}
    if not targets or found != targets or any(type(m) is not nn.Linear for _, m in modules):
        raise ValueError("目标必须全部存在且为未挂载的普通 Linear")
    if not isinstance(rank, int) or rank < 1 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("rank/alpha 非法")
    model.requires_grad_(False)
    for name, base in modules:
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent), leaf, LoRALinear(base, rank, alpha))
    return [name for name, _ in modules]


def merge_lora(model):
    """递归替换全部 LoRALinear；第二次调用没有目标，因此不会重复加增量。"""
    if isinstance(model, LoRALinear):
        return model.merged()
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            parent, _, leaf = name.rpartition(".")
            setattr(model.get_submodule(parent), leaf, module.merged())
    return model


def parameter_counts(model):
    """返回公式与实际可训练参数统计；冻结 bias 不计入 LoRA 参数。"""
    expected = sum(m.rank * (m.base.in_features + m.base.out_features)
                   for m in model.modules() if isinstance(m, LoRALinear))
    return dict(formula=expected, trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                total=sum(p.numel() for p in model.parameters()))


def adapter_state(model):
    """只保存 A/B 的 CPU 副本；调用方另存基座快照与 rank/alpha/targets。"""
    return {f"{name}.{key}": getattr(m, key).detach().cpu().clone()
            for name, m in model.named_modules() if isinstance(m, LoRALinear) for key in ("A", "B")}


def memory_estimate(model, batch, length, *, activation_bytes=2):
    """分项估算字节，不冒充峰值：fp32 AdamW 的梯度+m+v 为 12P。

激活仅用每层 12 个 hidden 张量作教学代理；未含 attention workspace、
CUDA context、碎片或完整反向图。logits 单独列出，实际还可能有 fp32 CE 副本。
"""
    cfg = model.config
    weights = sum(p.numel()*p.element_size() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(weights_bytes=weights, gradients_bytes=trainable*4,
                adam_m_v_bytes=trainable*8,
                activation_proxy_bytes=batch*length*cfg.hidden_size*cfg.num_hidden_layers*12*activation_bytes,
                logits_bytes=batch*length*cfg.vocab_size*activation_bytes,
                caveat="估算而非峰值上界；未含CE副本、workspace、CUDA context与碎片")
