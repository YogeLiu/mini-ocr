"""第 6 章：字节级 BPE 分词器（GPT-2、Qwen 系列用的就是这一类）。

    PAT                 预分词的正则：先把文本切成"词块"，BPE 只在词块内部合并
    pretokenize         文本 → 词块列表
    train_bpe_naive     朴素的 BPE 训练：每合并一次，就把所有词块的相邻对重新数一遍
    train_bpe           增量版：只更新受这次合并影响的词块，结果与朴素版逐条相同，快得多
    BPETokenizer        编码（按合并顺序应用合并规则）、解码、特殊符号、保存与读取
    show_tokens         把每个 token 的字节显示成人能读的样子（半个汉字显示成 <e6><a8>）

字节级的意思：初始词表就是 256 个字节，任何文本（中文、emoji、乱码）都能编码，永远不会遇到"词表里没有的字"。
常用汉字在 UTF-8 里占 3 个字节，所以没训练过的分词器会把一个常用汉字切成 3 个 token；训练让常见的字和词合并成一个 token。
"""
import json
from collections import Counter, defaultdict

import regex

# 与 GPT-2 的写法相近；数字一位一位地切（Qwen 也这样做），英文缩写单独成块，空格粘在后面那个词的前面。
# 中文没有空格，"\p{L}+" 会把一串汉字整个当作一个词块，BPE 再在里面合并出字和词。
PAT = r"""'(?:s|t|re|ve|m|ll|d)| ?\p{L}+| ?\p{N}| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_PAT = regex.compile(PAT)

COMMAND_PAT = r"\\[A-Za-z]+|" + PAT.replace(r"[^\s\p{L}\p{N}]", r"[^\s\p{L}\p{N}\\]") + r"|\\"
r"""命令完整词块的预分词规则（不是练习）。

普通标点分支必须排除反斜杠，否则它会从前面的花括号开始匹配，抢走命令开头：
例如 {\frac{a}{b}} 的 {\，以及 \frac{\alpha}{b} 中的 {\。
最后的反斜杠分支保留孤立反斜杠，保证无损。完整词块只允许 BPE 合并，不保证已学到整条命令。
"""


# >>> 练习 6.1：预分词
def pretokenize(text, pattern=None):
    """用 pattern（默认 PAT）把 text 切成词块，返回字符串列表。把它们拼起来必须正好等于 text（一个字符都不能丢）。
    例：pretokenize("I'm 12 岁。") == ["I", "'m", " 1", "2", " 岁", "。"]"""
    return (_PAT if pattern is None else regex.compile(pattern)).findall(text)
# <<<


def _word_counts(texts, pattern=None):
    """{词块的字节元组: 出现次数}。每个字节先是一个长度为 1 的 bytes。"""
    if isinstance(texts, str):
        texts = [texts]
    c = Counter()
    for t in texts:
        c.update(pretokenize(t, pattern))
    return {tuple(bytes([b]) for b in w.encode("utf-8")): n for w, n in c.items()}


def _merge_word(word, pair, new):
    """把 word（bytes 元组）里所有相邻的 pair 换成 new，从左往右、不重叠。"""
    out, i = [], 0
    while i < len(word):
        if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
            out.append(new)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def _best(pairs):
    """出现次数最多的对；次数相同时取字节序更大的那一对（规则固定下来，结果才可复现）。"""
    return max(pairs.items(), key=lambda kv: (kv[1], kv[0]))[0]


# >>> 练习 6.2：朴素 BPE 训练
def train_bpe_naive(texts, n_merges, pattern=None):
    """返回合并规则列表 merges：[(b"a", b"b"), ...]，第 i 条是第 i 次合并。
    每一轮：
      1. 对每个词块（字节元组，带出现次数 n），数出所有相邻对，每对加 n；
      2. 取次数最多的一对（平局用 _best 的规则），次数 < 2 时提前停止（再合并已经没有意义）；
      3. 把所有词块里的这一对合并成一个新 token（两个 bytes 拼起来），记下这条规则。
    用 _word_counts(texts, pattern) 得到初始词块，用 _merge_word 做合并。
    pattern 默认用 PAT；自定义时编码也须使用同一规则。"""
    words = _word_counts(texts, pattern)
    merges = []
    for _ in range(n_merges):
        pairs = Counter()
        for w, n in words.items():
            for a, b in zip(w, w[1:]):
                pairs[(a, b)] += n
        if not pairs:
            break
        best = _best(pairs)
        if pairs[best] < 2:
            break
        merges.append(best)
        new = best[0] + best[1]
        words = {_merge_word(w, best, new): n for w, n in words.items()}
    return merges
# <<<


# >>> 练习 6.3：增量 BPE 训练
def train_bpe(texts, n_merges, pattern=None):
    """结果必须与 train_bpe_naive 逐条相同，但每轮只处理包含被合并的那一对的词块：
      - pairs：{对: 总次数}；where：{对: 包含这一对的词块集合}；
      - 合并 best 时，对 where[best] 里的每个词块 w：先把 w 的所有相邻对从 pairs 里减掉（乘以 w 的次数），
        合并得到 w2，再把 w2 的相邻对加回去，并更新 where；
      - 次数降到 0 的对从 pairs 里删掉。
    词块用列表下标标识（同一个词块合并后内容会变，不能拿内容当键）。"""
    counts = _word_counts(texts, pattern)
    words = list(counts.keys())
    freq = list(counts.values())
    pairs, where = Counter(), defaultdict(set)
    for i, w in enumerate(words):
        for p in zip(w, w[1:]):
            pairs[p] += freq[i]
            where[p].add(i)
    merges = []
    for _ in range(n_merges):
        if not pairs:
            break
        best = _best(pairs)
        if pairs[best] < 2:
            break
        merges.append(best)
        new = best[0] + best[1]
        for i in list(where.pop(best, ())):
            w, n = words[i], freq[i]
            for p in zip(w, w[1:]):
                pairs[p] -= n
                if pairs[p] <= 0:
                    del pairs[p]
                if p != best:
                    where[p].discard(i)
            w2 = _merge_word(w, best, new)
            words[i] = w2
            for p in zip(w2, w2[1:]):
                pairs[p] += n
                where[p].add(i)
    return merges
# <<<


class BPETokenizer:
    """编号约定：0–255 是单个字节；256 起按合并顺序编号；特殊符号排在最后。

    pattern 默认用 PAT，绑定到实例并随合并表保存；须与训练时的规则一致。
    """

    def __init__(self, merges, specials=("<|endoftext|>",), pattern=None):
        self.pattern = PAT if pattern is None else pattern
        self.merges = [tuple(m) for m in merges]
        self.rank = {m: i for i, m in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = a + b
        self.specials = list(specials)
        for s in self.specials:
            self.vocab[len(self.vocab)] = s.encode("utf-8")
        self.special_ids = {s: 256 + len(self.merges) + i for i, s in enumerate(self.specials)}
        self.bytes_to_id = {v: k for k, v in self.vocab.items() if k < 256 + len(self.merges)}
        self._cache = {}

    def __len__(self):
        return len(self.vocab)

    # >>> 练习 6.4：编码一个词块
    def _encode_word(self, word):
        """word 是一个词块（字符串）。先拆成单字节，然后反复：在相邻对里找 rank 最小（最早学到）的那一对，
        把它的所有出现合并；直到没有任何相邻对在 self.rank 里。返回 token 编号列表。
        注意：要按合并规则学到的先后顺序合并，而不是"能合并就合并"——两种做法的结果可能不同（见第 6 章正文 6.6）。"""
        parts = tuple(bytes([b]) for b in word.encode("utf-8"))
        while len(parts) > 1:
            cand = [(self.rank[p], p) for p in zip(parts, parts[1:]) if p in self.rank]
            if not cand:
                break
            _, best = min(cand)
            parts = _merge_word(parts, best, best[0] + best[1])
        return [self.bytes_to_id[p] for p in parts]
    # <<<

    def encode(self, text, allow_special=True):
        """先把特殊符号整段切出来（它们不参与 BPE），其余部分预分词后逐块编码。"""
        ids = []
        for piece, is_special in self._split_specials(text, allow_special):
            if is_special:
                ids.append(self.special_ids[piece])
                continue
            for w in pretokenize(piece, self.pattern):
                if w not in self._cache:
                    self._cache[w] = self._encode_word(w)
                ids.extend(self._cache[w])
        return ids

    def _split_specials(self, text, allow_special):
        if not allow_special or not self.specials:
            return [(text, False)]
        pat = regex.compile("(" + "|".join(regex.escape(s) for s in sorted(self.specials, key=len, reverse=True)) + ")")
        out = []
        for i, piece in enumerate(pat.split(text)):
            if piece:
                out.append((piece, i % 2 == 1))
        return out

    def decode(self, ids):
        """字节拼起来再按 UTF-8 解码。中间截断出半个汉字时用 � 代替，不抛异常。"""
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        data = {"merges": [[a.hex(), b.hex()] for a, b in self.merges],
                "specials": self.specials, "pattern": self.pattern}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls([(bytes.fromhex(a), bytes.fromhex(b)) for a, b in data["merges"]],
                   data["specials"], pattern=data.get("pattern"))


def show_tokens(pieces):
    """pieces：每个 token 的 bytes。能按 UTF-8 完整解码的显示原文，否则显示成 <e6><a8> 这样的十六进制。"""
    out = []
    for b in pieces:
        try:
            out.append(b.decode("utf-8"))
        except UnicodeDecodeError:
            out.append("".join(f"<{x:02x}>" for x in b))
    return out
