"""第 26 章的最小视觉/文字连接机制；随机张量演示，不是训练好的 OCR。

约定输入分别为 [B,V,Dv] 和 [B,T,Dt]。视觉前缀使用双向视觉区和
因果文字区；交叉注意力的 query 来自文字、key/value 来自视觉。
这里用单头与显式矩阵乘法，让学习者能检查每个轴的含义。
"""
import math

import torch
from torch import nn


def _check_inputs(vision, text):
    if vision.ndim != 3 or text.ndim != 3 or vision.shape[0] != text.shape[0]:
        raise ValueError("需要批大小相同的 [B,V,Dv]、[B,T,Dt]")
    if min(*vision.shape, *text.shape) < 1:
        raise ValueError("所有轴必须非空")


# >>> 练习 26.1：视觉前缀的连接和注意力可见范围。
def prefix_mask(visual_tokens, text_tokens, device=None):
    """返回 [V+T,V+T] bool；True 可见。视觉不能读取文字，文字不能偷看未来。"""
    if visual_tokens < 1 or text_tokens < 1:
        raise ValueError("视觉、文字 token 数必须为正")
    mask = torch.zeros(visual_tokens+text_tokens, visual_tokens+text_tokens,
                       dtype=torch.bool, device=device)
    mask[:, :visual_tokens] = True
    mask[visual_tokens:, visual_tokens:] = torch.ones(
        text_tokens, text_tokens, dtype=torch.bool, device=device).tril()
    return mask
# <<<


class VisualPrefix(nn.Module):
    """线性连接器把 Dv 映射到 Dt，再沿 token 轴拼接；不混淆长度和宽度。"""
    def __init__(self, vision_dim=8, text_dim=12):
        super().__init__()
        self.projection = nn.Linear(vision_dim, text_dim)

    # >>> 练习 26.1：视觉前缀的连接。
    def forward(self, vision, text):
        _check_inputs(vision, text)
        return torch.cat((self.projection(vision), text), dim=1)
    # <<<


class CrossAttention(nn.Module):
    """单头交叉注意力；返回 [B,T,Dt] 及 [B,T,V] 权重，无自注意力/FFN。"""
    def __init__(self, vision_dim=8, text_dim=12):
        super().__init__()
        self.q = nn.Linear(text_dim, text_dim, bias=False)
        self.k = nn.Linear(vision_dim, text_dim, bias=False)
        self.v = nn.Linear(vision_dim, text_dim, bias=False)

    # >>> 练习 26.2：显式实现 QK^T / sqrt(d)、softmax 和加权求和。
    def forward(self, vision, text):
        _check_inputs(vision, text)
        query, key, value = self.q(text), self.k(vision), self.v(vision)
        logits = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
        # 低精度输入也在 fp32 中算 softmax；输出恢复 value 的精度。
        weights = torch.softmax(logits.float(), dim=-1).to(value.dtype)
        return weights @ value, weights
    # <<<
