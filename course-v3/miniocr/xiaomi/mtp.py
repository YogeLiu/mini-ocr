"""第 34 章：多 token 预测（MTP，报告 3.2 节，式 13）。

    L_CPT = L_NTP + λ_MTP · L_MTP

主干在位置 t 的隐藏状态 h_t 本来只用来预测 y_{t+1}。MTP 头（DeepSeek-V3 的做法，Qwen3.5 和 GLM-OCR 的配置里都带一层）
再把 h_t 和"下一个 token 的嵌入" e(y_{t+1}) 拼起来，过一个 Transformer 块，用**共享的输出层**预测 y_{t+2}：

    h'_t = Block( W · [RMSNorm(h_t); RMSNorm(e(y_{t+1}))] )        logits_{t+2} = head(RMSNorm(h'_t))

训练时它是一个额外的监督信号（每个位置多学"往后看两步"）；推理时可以扔掉，也可以拿来做投机解码的草稿。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.gpt import Block, RMSNorm


class MTPHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.nh, self.ne, self.norm = RMSNorm(d), RMSNorm(d), RMSNorm(d)
        self.proj = nn.Linear(2 * d, d, bias=False)
        self.block = Block(cfg)

    # >>> 练习 34.5：MTP 头的前向
    def forward(self, lm, hidden, emb):
        """hidden (B, T, d)：主干最后一层的隐藏状态；emb (B, T, d)：主干的输入嵌入（VLM 里图像位置已经是视觉 token）。
        位置 t（t = 0 … T−2）用 hidden[:, t] 和 emb[:, t+1]，返回 logits (B, T−1, V)，第 t 个预测的是第 t+2 个 token。
        块里的 RoPE 用 lm.cos[:T−1]、lm.sin[:T−1]；输出层用 lm.head（与主干共享）。"""
        x = self.proj(torch.cat([self.nh(hidden[:, :-1]), self.ne(emb[:, 1:])], -1))
        T1 = x.shape[1]
        x = self.block(x, lm.cos[:T1], lm.sin[:T1])
        return lm.head(self.norm(x))
    # <<<


# >>> 练习 34.6：MTP 损失
def mtp_loss(mtp_logits, targets):
    """mtp_logits (B, T−1, V)；targets (B, T)：主干的标签，targets[:, t] 是位置 t 要预测的 y_{t+1}（−100 表示不计算）。
    MTP 的第 t 个预测对应 y_{t+2} = targets[:, t+1]。返回这些位置交叉熵的平均（全是 −100 时返回 0）。"""
    tg = targets[:, 1:]
    if (tg != -100).sum() == 0:
        return mtp_logits.sum() * 0.0
    return F.cross_entropy(mtp_logits.reshape(-1, mtp_logits.shape[-1]).float(), tg.reshape(-1), ignore_index=-100)
# <<<
