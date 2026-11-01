"""维护者接线；与 B2 一样参考实现放在受限 mechanism 外。"""
import math
import torch
from miniocr.data.sft import encode_sft, collate_sft
from miniocr.train.sft import causal_loss
from miniocr.nn.lora import LoRALinear

run_encode_sft = encode_sft
run_collate = collate_sft


def run_causal_loss(logits, labels, *, reduction="mean"):
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("需要 (B,T,V) 与 (B,T)")
    valid = labels[labels != -100]
    if ((valid < 0) | (valid >= logits.shape[-1])).any():
        raise ValueError("标签越界")
    return causal_loss(logits, labels, reduction=reduction)


def get_lora(weight, bias=None, *, rank=4, alpha=8):
    if weight.ndim != 2 or min(weight.shape) < 1 or not weight.is_floating_point():
        raise ValueError("weight 必须为非空浮点矩阵")
    if bias is not None and (bias.shape != (weight.shape[0],) or bias.dtype != weight.dtype or bias.device != weight.device):
        raise ValueError("bias 形状/dtype/device 不匹配")
    if type(rank) is not int or rank < 1 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("rank/alpha 非法")
    base = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None,
                           dtype=weight.dtype, device=weight.device)
    with torch.no_grad():
        base.weight.copy_(weight)
        if bias is not None:
            base.bias.copy_(bias)
    return LoRALinear(base, rank, alpha)
