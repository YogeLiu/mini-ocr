"""第 8 章的最小因果语言模型；辅助演示代码，不是挖空练习。

token 嵌入 + 可学习的位置嵌入 + 一层因果注意力 + 输出层。
不含残差、归一化或前馈块；位置嵌入是第 10 章的预告。
"""
import torch
from torch import nn

from miniocr.nn.attention import MultiHeadAttention


class AttentionLM(nn.Module):
    """右侧补齐的输入 (B, T)，输出 (B, T, V)，BOS 的 logit 恒为 -inf。

    lengths 指明每行真实长度；省略时所有位置有效。补齐只在右侧，
    每行至少一个真实 token，因果 mask 与 key mask 相与不会产生空行。
    last_only=True 只投影最后一个真实位置，节省逐目标滑窗的输出层计算。
    每个窗口的位置从零开始；上下文包含 BOS（若尚未滑出窗口）。
    """

    def __init__(self, vocab_size, bos_id, context=64, d=64, n_heads=1):
        super().__init__()
        if min(vocab_size, context, d, n_heads) < 1 or d % n_heads:
            raise ValueError("维度必须为正，d 必须能被头数整除")
        if not 0 <= bos_id < vocab_size:
            raise ValueError("BOS 必须在词表内")
        self.context, self.bos_id = context, bos_id
        self.token = nn.Embedding(vocab_size, d)
        self.position = nn.Embedding(context, d)
        nn.init.normal_(self.token.weight, std=0.1)
        nn.init.normal_(self.position.weight, std=0.02)
        self.attn = MultiHeadAttention(d, n_heads, causal=True)
        self.output = nn.Linear(d, vocab_size)

    def forward(self, idx, lengths=None, last_only=False):
        if idx.ndim != 2 or not 1 <= idx.shape[1] <= self.context:
            raise ValueError("输入须为 (B, T)，且 1 ≤ T ≤ context")
        batch, width = idx.shape
        if lengths is None:
            lengths = torch.full((batch,), width, device=idx.device, dtype=torch.long)
        if lengths.shape != (batch,) or ((lengths < 1) | (lengths > width)).any():
            raise ValueError("lengths 须为每个样本给出 1 至 T 的真实长度")
        positions = torch.arange(width, device=idx.device)
        keep = positions[None, :] < lengths[:, None]
        x = self.token(idx) + self.position(positions)
        x = self.attn(x, mask=keep[:, None, None, :])
        if last_only:
            x = x[torch.arange(batch, device=idx.device), lengths - 1]
        logits = self.output(x)
        logits[..., self.bos_id] = -torch.inf
        return logits
