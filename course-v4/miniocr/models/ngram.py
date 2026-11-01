"""第 7 章：从计数语言模型到固定窗口 MLP。

计数模型的输入 seqs 不含边界符；模型逐条补 BOS 上文和一个 EOS 目标。
BOS 只作条件，不在预测词表中；EOS 参与 loss，特殊符号不贡献文本字节。
辅助验证、采样和通用 n-gram 是参考代码，练习集中在 7.1–7.4。
"""
from collections import Counter
import math

import numpy as np
import torch
from torch import nn

from miniocr.train.lm import bits_per_byte


# >>> 练习 7.1：错开一位与固定窗口样本
def make_examples(ids, context):
    """滑动步长为一，返回 GPT 式 X、Y，形状均为 (样本数, context)。

    X[i]=ids[i:i+context]，Y[i]=ids[i+1:i+context+1]。
    不自动补边界；不足 context+1 个 token 时返回空的二维 LongTensor。
    """
    if context < 1:
        raise ValueError("context 必须为正")
    ids = torch.as_tensor(ids, dtype=torch.long)
    if ids.ndim != 1:
        raise ValueError("ids 必须是一维")
    if len(ids) <= context:
        return ids.new_empty((0, context)), ids.new_empty((0, context))
    windows = ids.unfold(0, context + 1, 1)
    return windows[:, :-1].clone(), windows[:, 1:].clone()


def make_window_examples(ids, context):
    """固定窗口：X[i] 为前 context 个 token，Y[i] 为紧接的一个 token。

    返回形状 (样本数, context)、(样本数,)，不自动补边界。
    要预测公式首 token，调用方应在每条公式前补 context 个 BOS。
    """
    x, y = make_examples(ids, context)
    return x, y[:, -1].clone()
# <<<


# >>> 练习 7.3：困惑度与信息单位换算
def perplexity(nll):
    """nll 是以自然对数计算的平均 nat/token；困惑度为 exp(nll)。"""
    try:
        return math.exp(nll)
    except OverflowError:
        return math.inf


def nats_to_bits(nll):
    """nat/token → bits/token。"""
    return nll / math.log(2)


def bits_to_nats(bits):
    """bits/token → nat/token。"""
    return bits * math.log(2)


def nats_to_bits_per_byte(nll, bytes_per_token):
    """nat/token → bits/byte；bytes_per_token 为原文字节数 / 预测目标数。"""
    if bytes_per_token <= 0:
        raise ValueError("每 token 字节数必须为正")
    return bits_per_byte(nll, 1, bytes_per_token)


def bits_per_byte_to_nats(bpb, bytes_per_token):
    """bits/byte → nat/token，使用相同的目标和字节计数口径。"""
    if bytes_per_token <= 0:
        raise ValueError("每 token 字节数必须为正")
    return bits_to_nats(bpb * bytes_per_token)
# <<<


class NGramLM:
    """稀疏计数 n-gram，无回退、无插值，方便观察长上下文的稀疏。

    vocab_size 包含 BOS/EOS；可预测类别数为 vocab_size-1（排除 BOS）。
    未见上下文在 k>0 时为均匀分布；k=0 时约定所有概率为零，loss=inf，
    此时不是可采样分布。sample 返回 n 条内容 ID 列表，不包含边界符。
    """
    def __init__(self, vocab_size, bos_id, eos_id, n=2, k=1.0):
        if n < 1 or vocab_size < 2 or not math.isfinite(k) or k < 0:
            raise ValueError("n、词表大小或平滑参数不合法")
        if bos_id == eos_id or not (0 <= bos_id < vocab_size and 0 <= eos_id < vocab_size):
            raise ValueError("BOS/EOS 必须是词表内不同的编号")
        self.vocab_size, self.bos_id, self.eos_id = vocab_size, bos_id, eos_id
        self.n, self.k = n, k
        self.counts, self.totals = {}, Counter()

    def events(self, seq, add_eos=True):
        """逐个生成 (上文元组, 目标)，公式间重置，不预测 BOS。"""
        seq = list(seq)
        if any(t < 0 or t >= self.vocab_size or t in (self.bos_id, self.eos_id) for t in seq):
            raise ValueError("seq 只能含词表内的普通 token，不要自行添加 BOS/EOS")
        padded = [self.bos_id] * (self.n - 1) + seq + ([self.eos_id] if add_eos else [])
        for i in range(self.n - 1, len(padded)):
            yield tuple(padded[i - self.n + 1:i]), padded[i]

    def fit(self, seqs, add_eos=True):
        """重置后计数；add_eos=False 只用于忘记终止符的故障演示。"""
        self.counts, self.totals = {}, Counter()
        for seq in seqs:
            for context, target in self.events(seq, add_eos):
                self.counts.setdefault(context, Counter())[target] += 1
                self.totals[context] += 1
        return self

    def prob(self, context, target):
        """加 k 平滑的条件概率；context 长度须恰好为 n-1。"""
        context = tuple(context)
        if len(context) != self.n - 1:
            raise ValueError("上文长度应等于 n-1")
        if not 0 <= target < self.vocab_size:
            raise ValueError("目标不在词表中")
        if target == self.bos_id:
            return 0.0
        denominator = self.totals.get(context, 0) + self.k * (self.vocab_size - 1)
        count = self.counts.get(context, {}).get(target, 0)
        return (count + self.k) / denominator if denominator else 0.0

    def logprob(self, seq):
        """整条公式（含 EOS）的对数概率；遇到零概率返回 -inf。"""
        total = 0.0
        for context, target in self.events(seq):
            p = self.prob(context, target)
            if p == 0:
                return -math.inf
            total += math.log(p)
        return total

    def score_counts(self, seqs):
        """聚合评测事件，返回 (相邻计数, 上文计数, 频次) 数组供多组 k 重用。"""
        pairs = Counter(event for seq in seqs for event in self.events(seq))
        rows = [(self.counts.get(c, {}).get(t, 0), self.totals.get(c, 0), f)
                for (c, t), f in pairs.items()]
        if not rows:
            raise ValueError("评测语料不能为空")
        return np.asarray(rows, dtype=np.float64).T

    def nll_from_counts(self, stats):
        """按目标 token 加权，不能先按公式等权平均。"""
        count, total, frequency = stats
        with np.errstate(divide="ignore", invalid="ignore"):
            p = (count + self.k) / (total + self.k * (self.vocab_size - 1))
            losses = -np.log(p)
        losses = np.where(np.isnan(losses), np.inf, losses)
        return float(np.dot(losses, frequency) / frequency.sum())

    def nll_per_token(self, seqs):
        """所有内容目标及 EOS 的总 NLL / 总目标数。"""
        return self.nll_from_counts(self.score_counts(seqs))

    def sample(self, n, seed=0, max_tokens=128):
        """独立从 BOS 起采样，遇 EOS 停止；到上限就截断。固定 seed 可复现。"""
        if n < 0 or max_tokens < 1:
            raise ValueError("采样条数须非负，上限须为正")
        rng = np.random.default_rng(seed)
        cache, out = {}, []
        for _ in range(n):
            history, seq = [self.bos_id] * (self.n - 1), []
            for _ in range(max_tokens):
                context = tuple(history[-(self.n - 1):]) if self.n > 1 else ()
                if context not in cache:
                    p = np.array([self.prob(context, t) for t in range(self.vocab_size)])
                    if not p.sum():
                        raise ValueError("无平滑的未见上下文没有可采样分布")
                    cache[context] = p / p.sum()
                t = int(rng.choice(self.vocab_size, p=cache[context]))
                if t == self.eos_id:
                    break
                seq.append(t)
                history.append(t)
            out.append(seq)
        return out


class BigramLM(NGramLM):
    """二元模型：每个 token 只依赖前一个，边界协议与 NGramLM 相同。"""
    def __init__(self, vocab_size, bos_id, eos_id, k=1.0):
        super().__init__(vocab_size, bos_id, eos_id, n=2, k=k)

    # >>> 练习 7.2：bigram 计数与加 k 平滑
    def fit(self, seqs, add_eos=True):
        """逐条补 BOS/EOS，统计 counts[(前一个,)][当前] 和 totals[(前一个,)]。"""
        self.counts, self.totals = {}, Counter()
        for seq in seqs:
            for context, target in self.events(seq, add_eos):
                self.counts.setdefault(context, Counter())[target] += 1
                self.totals[context] += 1
        return self

    def prob(self, context, target):
        """P(t|c)=(C(c,t)+k)/(C(c)+k*(V-1))；BOS 不是预测目标。

        k=0 且没有上文计数时约定概率为零。context 为长度一的元组。
        """
        context = tuple(context)
        if len(context) != 1 or not 0 <= target < self.vocab_size:
            raise ValueError("上文长度应为一，目标应在词表内")
        if target == self.bos_id:
            return 0.0
        total = self.totals.get(context, 0)
        denominator = total + self.k * (self.vocab_size - 1)
        count = self.counts.get(context, {}).get(target, 0)
        return (count + self.k) / denominator if denominator else 0.0
    # <<<


class MLPLM(nn.Module):
    """固定窗口：嵌入 → 拼接 → tanh 隐藏层 → 词表 logits。

    forward 只返回 logits；采样策略、BOS 输出掩码、交叉熵由调用方处理。
    """
    def __init__(self, vocab_size, context=3, embed_dim=64, hidden_dim=256):
        super().__init__()
        if min(vocab_size, context, embed_dim, hidden_dim) < 1:
            raise ValueError("模型各维度必须为正")
        self.context = context
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.hidden = nn.Linear(context * embed_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    # >>> 练习 7.4：固定窗口 MLP 的前向传播
    def forward(self, x):
        """x 形状 (B, context)，返回 (B, vocab_size) 的未归一化 logits。"""
        if x.ndim != 2 or x.shape[1] != self.context:
            raise ValueError("输入形状应为 (B, context)")
        embedded = self.embedding(x).flatten(1)
        return self.output(torch.tanh(self.hidden(embedded)))
    # <<<
