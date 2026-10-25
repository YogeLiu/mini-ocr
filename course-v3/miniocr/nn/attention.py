"""第 15 章：注意力。

    attention              缩放点积注意力：softmax(Q Kᵀ / √d) V，可选 mask
    causal_mask            因果 mask：位置 t 只能看 ≤ t 的位置（语言模型用，第 16 章）
    MultiHeadAttention     多头注意力；kv 不为 None 时是交叉注意力（Q 来自一个序列，K、V 来自另一个）
    Block                  注意力 + MLP，各带残差和 Pre-LN（第 16 章 GPT 的一层就是它）
    LookupNet              第 15 章演示用的小模型：token 嵌入 + 位置嵌入 + 若干 Block
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# >>> 练习 15.1：缩放点积注意力
def attention(q, k, v, mask=None):
    """q 形状 (..., Tq, d)，k 形状 (..., Tk, d)，v 形状 (..., Tk, dv)。
        scores = q @ kᵀ / √d            形状 (..., Tq, Tk)：第 i 个查询与第 j 个键的相似度
        mask 为 bool 张量，可广播到 scores 的形状；False 的位置不许看：把 score 设成 −∞（softmax 后权重为 0）
        weights = softmax(scores, 沿最后一维)   每一行和为 1
        out = weights @ v              形状 (..., Tq, dv)
    返回 (out, weights)。"""
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = scores.softmax(-1)
    return weights @ v, weights
# <<<


# >>> 练习 15.2：因果 mask
def causal_mask(T, device=None):
    """形状 (T, T) 的 bool 张量，第 i 行第 j 列为 True 当且仅当 j ≤ i（下三角，含对角线）。"""
    return torch.ones(T, T, dtype=torch.bool, device=device).tril()
# <<<


# >>> 练习 15.3：多头注意力
class MultiHeadAttention(nn.Module):
    """d 维输入，n_heads 个头，每个头 dh = d / n_heads 维。

        q = x Wq，k = kv Wk，v = kv Wv（kv 为 None 时 kv = x，即自注意力）   各 (B, T, d)
        拆成头：(B, T, d) → (B, T, h, dh) → (B, h, T, dh)
        每个头各自做 attention（mask 广播到所有头）→ (B, h, Tq, dh)
        合并：→ (B, Tq, h, dh) → (B, Tq, d)，再乘输出矩阵 Wo
    Wq、Wk、Wv、Wo 是 nn.Linear(d, d)，名字分别为 q、k、v、o。
    forward 返回输出；最近一次的注意力权重 (B, h, Tq, Tk) 存在 self.last_weights，方便画图。
    """

    def __init__(self, d, n_heads, causal=False, bias=True):
        super().__init__()
        assert d % n_heads == 0, "d 必须能被头数整除"
        self.d, self.h, self.dh, self.causal = d, n_heads, d // n_heads, causal
        self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=bias) for _ in range(4))
        self.last_weights = None

    def _split(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.dh).transpose(1, 2)

    def forward(self, x, kv=None, mask=None):
        kv = x if kv is None else kv
        q, k, v = self._split(self.q(x)), self._split(self.k(kv)), self._split(self.v(kv))
        if self.causal:
            cm = causal_mask(x.shape[1], x.device)
            mask = cm if mask is None else mask & cm
        out, w = attention(q, k, v, mask)
        self.last_weights = w.detach()
        B, _, T, _ = out.shape
        return self.o(out.transpose(1, 2).reshape(B, T, self.d))
# <<<


class Block(nn.Module):
    """x ← x + Attn(LN(x))；x ← x + MLP(LN(x))。MLP 是 d → 4d → d，中间用 GELU。"""

    def __init__(self, d, n_heads, causal=False):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = MultiHeadAttention(d, n_heads, causal)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class LookupNet(nn.Module):
    """token 嵌入 + 可学习的位置嵌入 + n_layers 个 Block，最后一个位置的输出经 LN 和线性层得到 n_out 类的 logits。
    pos=False 时不加位置嵌入（演示"注意力本身不知道顺序"）。"""

    def __init__(self, vocab, T, n_out, d=64, n_heads=4, n_layers=2, pos=True, causal=False):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        nn.init.normal_(self.tok.weight, 0.0, 0.02)       # 与位置嵌入同一尺度；nn.Embedding 默认是 N(0, 1)，
        self.pos = nn.Parameter(torch.randn(T, d) * 0.02) if pos else None   # 那样位置信号只有 token 的 1/50，模型几乎看不到顺序
        self.blocks = nn.ModuleList(Block(d, n_heads, causal) for _ in range(n_layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_out)

    def forward(self, idx):
        x = self.tok(idx)
        if self.pos is not None:
            x = x + self.pos[: idx.shape[1]]
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x[:, -1]))
