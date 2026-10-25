"""A3 参考实现：对比学习（CLIP / SigLIP）、零样本分类、检索召回率、字符级文字编码器、双塔模型。"""
import math

import torch
import torch.nn as nn

from .nn import Embedding, LayerNorm, Linear, log_sigmoid, log_softmax
from .vision import ViTBlock


def _ce_rows(logits):
    """每一行的正确答案是对角线上那一列，返回平均交叉熵。"""
    lp = log_softmax(logits, -1)
    return -lp.diagonal().mean()


def clip_loss(zi, zt, logit_scale):
    """zi、zt (B, D) 已归一化，第 i 张图配第 i 句话。logits = s·zi ztᵀ，返回 (按行的交叉熵 + 按列的交叉熵) / 2。"""
    logits = logit_scale * zi @ zt.T
    return (_ce_rows(logits) + _ce_rows(logits.T)) / 2


def siglip_loss(zi, zt, logit_scale, logit_bias):
    """每一对 (i, j) 是一个二分类：logit = s·zi_i·zt_j + b，y = +1（i = j）或 −1。损失 = −Σ_ij log σ(y·logit) / B。"""
    logits = logit_scale * zi @ zt.T + logit_bias
    y = 2 * torch.eye(len(zi), device=zi.device, dtype=logits.dtype) - 1
    return -log_sigmoid(y * logits).sum() / len(zi)


def zero_shot(zi, class_zt):
    return (zi @ class_zt.T).argmax(-1)


def recall_at_k(sim, k, same=None):
    """前 k 名里至少有一个"算对"的候选的查询所占的比例。same 默认是单位矩阵（只有 j = i 算对）。"""
    if same is None:
        same = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    top = sim.topk(k, dim=1).indices
    return same.gather(1, top).any(1).float().mean().item()


def normalize(x, eps=1e-8):
    return x / (x.pow(2).sum(-1, keepdim=True).sqrt() + eps)


class TextEncoder(nn.Module):
    """字符级的小 Transformer：嵌入 + 可学习位置表 → 非因果块（填充位置被 mask 掉）→ LayerNorm → 对真实字符求平均。"""

    def __init__(self, vocab_size, max_len=96, d=128, n_layer=3, n_head=4, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.tok = Embedding(vocab_size, d)
        self.pos = nn.Parameter(torch.randn(max_len, d) * 0.02)
        self.blocks = nn.ModuleList(ViTBlock(d, n_head) for _ in range(n_layer))
        self.norm = LayerNorm(d)

    def forward(self, ids):
        real = ids != self.pad_id
        x = self.tok(ids) + self.pos[:ids.shape[1]]
        for b in self.blocks:
            x = b(x, key_mask=real)
        x = self.norm(x)
        return (x * real[..., None]).sum(1) / real.sum(1, keepdim=True).clamp(min=1)


class DualEncoder(nn.Module):
    def __init__(self, vision, text, d_img, d_txt, d_emb=128, loss="siglip"):
        super().__init__()
        self.vision, self.text, self.loss_kind = vision, text, loss
        self.pi, self.pt = Linear(d_img, d_emb, bias=False), Linear(d_txt, d_emb, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.bias = nn.Parameter(torch.tensor(-10.0 if loss == "siglip" else 0.0))

    def encode_image(self, x):
        return normalize(self.pi(self.vision(x)))

    def encode_text(self, ids):
        return normalize(self.pt(self.text(ids)))

    def forward(self, x, ids):
        zi, zt = self.encode_image(x), self.encode_text(ids)
        s = self.log_scale.exp().clamp(max=100)
        return clip_loss(zi, zt, s) if self.loss_kind == "clip" else siglip_loss(zi, zt, s, self.bias)
