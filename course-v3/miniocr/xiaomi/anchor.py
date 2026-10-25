"""第 34 章：Q-Mask 式的文字锚定（报告 3.1 节，式 12）。

    L_SSA = λ_txt · L_NTP + λ_seg · (L_Dice + L_CE)

模型照常写答案（NTP），同时用一个"按查询条件化的掩码解码器"预测答案对应的内容在图里的位置（每个视觉 token 一个概率），
用 Dice 和逐 token 的交叉熵监督。这一阶段结束后掩码解码器被扔掉，只留下学到了"文字在哪里"的视觉—语言主干。

我们的简化版：
    box_to_grid_mask  把像素框变成视觉 token 网格上的 0/1 掩码（我们的 VLM：64×64 的图、8×8 = 64 个视觉 token）
    MaskHead          查询向量 = 语言模型在"答案开始前最后一个位置"的隐藏状态；每个视觉 token 的分数 = 查询与它的点积
    dice_loss         软 Dice 损失：对前景很小的掩码也敏感
    ssa_loss          式 12；没有掩码标注的样本只算 NTP
    lm_hidden         第 16 章 GPT 的前向，同时返回最后一层（归一化后）的隐藏状态
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# >>> 练习 34.1：框 → 视觉 token 掩码
def box_to_grid_mask(box, img_hw=(64, 64), grid_hw=(8, 8), min_cover=0.3):
    """box = (x0, y0, x1, y1)，像素坐标（x1、y1 不含）。图 img_hw = (H, W) 被均匀切成 grid_hw = (gh, gw) 个格子，按行排列。
    每个格子被框覆盖的面积比例 cover ∈ [0, 1]；cover ≥ min_cover 的格子记 1。
    框很小、没有任何格子达到 min_cover 时，把 cover 最大的那个格子（并列取下标小的）记 1，保证掩码不为空。
    返回 float32 张量 (gh·gw,)。"""
    H, W = img_hw
    gh, gw = grid_hw
    ch, cw = H / gh, W / gw
    x0, y0, x1, y1 = box
    cover = torch.zeros(gh, gw)
    for r in range(gh):
        oy = max(0.0, min(y1, (r + 1) * ch) - max(y0, r * ch))
        for c in range(gw):
            ox = max(0.0, min(x1, (c + 1) * cw) - max(x0, c * cw))
            cover[r, c] = oy * ox / (ch * cw)
    m = (cover >= min_cover).float().flatten()
    if m.sum() == 0:
        m[int(cover.flatten().argmax())] = 1.0
    return m
# <<<


# >>> 练习 34.2：Dice 损失
def dice_loss(logits, target, eps=1.0):
    """logits、target：(B, N)。p = sigmoid(logits)。每条样本 dice = (2 Σ p·t + eps) / (Σ p + Σ t + eps)，
    损失 = 1 − dice，再对 B 条样本取平均。eps 让全空的掩码也有定义（预测也全空时损失接近 0）。"""
    p = torch.sigmoid(logits.float())
    t = target.float()
    dice = (2 * (p * t).sum(-1) + eps) / (p.sum(-1) + t.sum(-1) + eps)
    return (1 - dice).mean()
# <<<


class MaskHead(nn.Module):
    """按查询条件化的掩码头：score_j = (W_q q) · (W_k v_j) / √d_k + b。参数只有两个 d×d_k 的矩阵和一个偏置。"""

    def __init__(self, d, d_k=64):
        super().__init__()
        self.q = nn.Linear(d, d_k, bias=False)
        self.k = nn.Linear(d, d_k, bias=False)
        self.b = nn.Parameter(torch.tensor(-2.0))           # 初始时大多数 token 的概率约 0.12：前景本来就少
        self.d_k = d_k

    # >>> 练习 34.3：掩码头的前向
    def forward(self, query, vis):
        """query (B, d)，vis (B, N, d) → logits (B, N)。"""
        q = self.q(query)                                    # (B, d_k)
        k = self.k(vis)                                      # (B, N, d_k)
        return torch.einsum("bd,bnd->bn", q, k) / math.sqrt(self.d_k) + self.b
    # <<<


# >>> 练习 34.4：SSA 损失（式 12）
def ssa_loss(ntp, mask_logits, target, has_mask, lam_txt=1.0, lam_seg=0.5):
    """ntp：已经算好的 NTP 损失（标量）。mask_logits、target：(B, N)；has_mask：(B,) bool，哪些样本有掩码标注。
    seg = dice_loss + 逐 token 的二元交叉熵（binary_cross_entropy_with_logits 的平均），都只在 has_mask 的样本上算。
    返回 (总损失 = lam_txt·ntp + lam_seg·seg, {"ntp", "dice", "ce"} 的浮点数)。没有任何样本有掩码时 seg = 0。"""
    if has_mask.any():
        ml, tg = mask_logits[has_mask], target[has_mask]
        dice = dice_loss(ml, tg)
        ce = F.binary_cross_entropy_with_logits(ml.float(), tg.float())
    else:
        dice = ce = ntp.new_zeros(())
    total = lam_txt * ntp + lam_seg * (dice + ce)
    return total, {"ntp": ntp.item(), "dice": dice.item(), "ce": ce.item()}
# <<<


def lm_hidden(lm, emb):
    """第 16 章 GPT 的前向（不用 KV cache），返回 (logits, 最后一层归一化后的隐藏状态 (B, T, d))。"""
    T = emb.shape[1]
    x = emb
    for b in lm.blocks:
        x = b(x, lm.cos[:T], lm.sin[:T])
    h = lm.norm(x)
    return lm.head(h), h
