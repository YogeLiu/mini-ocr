"""第 14 章：分词与 n 元语法语言模型。

    pretokenize    预切分：先把文本切成"块"（单词连同前面的空格、单个数字、LaTeX 命令、单个标点……），BPE 只在块内合并
    bpe_train      在语料上学习 BPE 合并规则
    BPETokenizer   用学到的规则编码 / 解码（字符串 ↔ token 编号）
    NGramLM        加 k 平滑的 n 元语法模型，计算交叉熵（第 6 章）
"""
import math
import re
from collections import Counter, defaultdict

# 依次尝试：[空格]字母串、单个数字、[空格]LaTeX 命令（\frac）、换行、连续空格、[空格]其他单个符号
_PAT = re.compile(r" ?[A-Za-z]+|[0-9]| ?\\[A-Za-z]+|\n| +(?=\S)| +| ?[^\sA-Za-z0-9]")


# >>> 练习 14.1：预切分
def pretokenize(text):
    """用正则 _PAT 把文本切成块，返回块的列表。所有块首尾相接必须恰好等于原文（不丢任何字符）。
    例如 "The cat 42\\frac{a}" → ["The", " cat", " ", "4", "2", "\\frac", "{", "a", "}"]。"""
    chunks = _PAT.findall(text)
    assert "".join(chunks) == text, "预切分丢了字符"
    return chunks
# <<<


def _merge(sym, pair, new):
    """把元组 sym 里所有相邻的 pair 换成 new（从左到右，不重叠）。"""
    out, i = [], 0
    while i < len(sym):
        if i + 1 < len(sym) and sym[i] == pair[0] and sym[i + 1] == pair[1]:
            out.append(new)
            i += 2
        else:
            out.append(sym[i])
            i += 1
    return tuple(out)


# >>> 练习 14.2：训练 BPE
def bpe_train(text, n_merges, verbose=False):
    """返回合并规则列表 [(左, 右), ...]，按学到的先后排列（先学到的优先级高）。

    1. 预切分，统计每个块出现的次数；每个块初始表示为字符元组，例如 " cat" → (" ", "c", "a", "t")。
    2. 重复 n_merges 次：统计所有块里相邻符号对的出现次数（乘以块的次数），取次数最多的一对
       （次数相同时取字典序最小的一对，保证结果确定）；把所有块里的这一对合并成一个新符号；记录这条规则。
       没有可以合并的对时提前停止。
    """
    words = Counter(pretokenize(text))
    syms = {w: tuple(w) for w in words}
    merges = []
    for step in range(n_merges):
        pairs = Counter()
        for w, c in words.items():
            s = syms[w]
            for a, b in zip(s, s[1:]):
                pairs[(a, b)] += c
        if not pairs:
            break
        best = min(pairs, key=lambda p: (-pairs[p], p))
        merges.append(best)
        new = best[0] + best[1]
        for w in words:
            if len(syms[w]) > 1:
                syms[w] = _merge(syms[w], best, new)
        if verbose:
            print(f"    第 {step + 1:3d} 次合并：{best[0]!r} + {best[1]!r} → {new!r}（出现 {pairs[best]} 次）")
    return merges
# <<<


class BPETokenizer:
    """词表 = 语料里出现过的所有单个字符（排序）+ 每条合并规则产生的新符号（按顺序）。"""

    def __init__(self, merges, chars):
        self.merges = list(merges)
        self.rank = {p: i for i, p in enumerate(self.merges)}
        self.itos = sorted(set(chars)) + [a + b for a, b in self.merges]
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self._cache = {}

    @classmethod
    def train(cls, text, n_merges, verbose=False):
        return cls(bpe_train(text, n_merges, verbose), set(text))

    def __len__(self):
        return len(self.itos)

    # >>> 练习 14.3：用 BPE 编码
    def encode_chunk(self, chunk):
        """一个块 → token 字符串的列表。从字符开始，反复找出"优先级最高"（rank 最小）的相邻对并合并，
        直到没有任何相邻对在规则里。注意：要按训练时的顺序应用规则，不能简单地"能合并就合并"。"""
        if chunk in self._cache:
            return self._cache[chunk]
        s = tuple(chunk)
        while len(s) > 1:
            pairs = [(self.rank.get(p, math.inf), p) for p in zip(s, s[1:])]
            r, best = min(pairs)
            if r == math.inf:
                break
            s = _merge(s, best, best[0] + best[1])
        self._cache[chunk] = list(s)
        return self._cache[chunk]
    # <<<

    def tokens(self, text):
        return [t for c in pretokenize(text) for t in self.encode_chunk(c)]

    def encode(self, text):
        return [self.stoi[t] for t in self.tokens(text)]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


# >>> 练习 14.4：n 元语法语言模型
class NGramLM:
    """p(x_t | 前 n−1 个 token)，用加 k 平滑估计：
        p(w | h) = (count(h, w) + k) / (count(h) + k · V)
    其中 h 是前 n−1 个 token（不足时在开头补 n−1 个特殊的 <s>，编号 −1），V 是词表大小。
    n = 1 时 h 为空，就是第 6 章的 unigram 模型。
    """

    def __init__(self, n, vocab_size, k=0.1):
        self.n, self.V, self.k = n, vocab_size, k
        self.ctx = defaultdict(int)          # count(h)
        self.joint = defaultdict(int)        # count(h, w)

    def _contexts(self, ids):
        pad = [-1] * (self.n - 1) + list(ids)
        for t in range(len(ids)):
            yield tuple(pad[t:t + self.n - 1]), ids[t]

    def fit(self, ids):
        for h, w in self._contexts(ids):
            self.ctx[h] += 1
            self.joint[(h, w)] += 1
        return self

    def logprob(self, h, w):
        """k = 0 且 (h, w) 从未出现时，概率为 0，返回 −∞（交叉熵变成无穷大：这就是需要平滑的原因）。"""
        num = self.joint.get((h, w), 0) + self.k
        return math.log(num / (self.ctx.get(h, 0) + self.k * self.V)) if num > 0 else -math.inf

    def cross_entropy(self, ids):
        """平均每个 token 的负对数似然（nat）。"""
        return -sum(self.logprob(h, w) for h, w in self._contexts(ids)) / len(ids)
# <<<


def bits_per_char(nll_per_token, n_tokens, n_chars):
    """把"每个 token 的交叉熵（nat）"换算成"每个字符多少比特"：总信息量相同，换个分母。
    不同分词方式的困惑度不能直接比较，但 bits per char 可以。"""
    return nll_per_token * n_tokens / n_chars / math.log(2)
