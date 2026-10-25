"""A4 参考实现：从 A2 复制的 BPE 分词器。A3 用它的字符级版本（vocab_size = 特殊符号数 + 96，不做合并）。"""
import heapq
import re
from collections import Counter, defaultdict

PAT = re.compile(r" ?[A-Za-z]+|[0-9]| ?\\[A-Za-z]+|\n| +(?=\S)| +| ?[^\sA-Za-z0-9]")
BASE_CHARS = "\n" + "".join(chr(c) for c in range(32, 127))


def pretokenize(text):
    return PAT.findall(text)


def _split_specials(text, specials):
    """按特殊符号切开，返回 [(片段, 是否特殊符号), ...]。重叠时优先匹配更长的特殊符号。"""
    if not specials:
        return [(text, False)] if text else []
    pat = re.compile("(" + "|".join(re.escape(s) for s in sorted(specials, key=len, reverse=True)) + ")")
    out = []
    for i, part in enumerate(pat.split(text)):
        if part:
            out.append((part, i % 2 == 1))
    return out


def _check_chars(text):
    bad = set(text) - set(BASE_CHARS)
    if bad:
        raise ValueError(f"字符不在基础字符表里：{sorted(bad)[:5]}")


def train_bpe(text, vocab_size, special_tokens=()):
    """返回 (vocab, merges)。vocab：{id: 字符串}，顺序是 特殊符号 → 96 个基础字符 → 每条合并产生的新符号。
    合并次数 = vocab_size − len(special_tokens) − 96。特殊符号先从文本里切掉，不参与统计。
    每次合并出现次数最多的相邻对；次数相同时取 (左, 右) 字典序最小的一对。"""
    n_merges = vocab_size - len(special_tokens) - len(BASE_CHARS)
    if n_merges < 0:
        raise ValueError("vocab_size 太小")
    counts = Counter()
    for part, special in _split_specials(text, special_tokens):
        if not special:
            _check_chars(part)
            counts.update(pretokenize(part))
    words = [list(w) for w in counts]
    freq = list(counts.values())

    pair_count = Counter()
    where = defaultdict(set)                          # 相邻对 → 出现在哪些块里
    for i, (w, c) in enumerate(zip(words, freq)):
        for p in zip(w, w[1:]):
            pair_count[p] += c
            where[p].add(i)
    heap = [(-c, p) for p, c in pair_count.items()]
    heapq.heapify(heap)

    merges = []
    while len(merges) < n_merges and heap:
        negc, p = heapq.heappop(heap)
        if pair_count.get(p, 0) != -negc or negc == 0:
            continue                                  # 过期的条目（次数已经变了）
        merges.append(p)
        new = p[0] + p[1]
        changed = set()
        for i in list(where[p]):
            w, c = words[i], freq[i]
            for q in zip(w, w[1:]):                   # 先把这个块的旧相邻对减掉
                pair_count[q] -= c
                where[q].discard(i)
                changed.add(q)
            out, j = [], 0
            while j < len(w):
                if j + 1 < len(w) and w[j] == p[0] and w[j + 1] == p[1]:
                    out.append(new)
                    j += 2
                else:
                    out.append(w[j])
                    j += 1
            words[i] = out
            for q in zip(out, out[1:]):               # 再把新的相邻对加回来
                pair_count[q] += c
                where[q].add(i)
                changed.add(q)
        for q in changed:
            if pair_count[q] > 0:
                heapq.heappush(heap, (-pair_count[q], q))
            else:
                pair_count.pop(q, None)
    vocab = {}
    for s in list(special_tokens) + list(BASE_CHARS) + [a + b for a, b in merges]:
        vocab[len(vocab)] = s
    return vocab, merges


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=()):
        self.vocab = dict(vocab)
        self.stoi = {s: i for i, s in self.vocab.items()}
        self.rank = {tuple(p): i for i, p in enumerate(merges)}
        self.specials = list(special_tokens)
        self._cache = {}

    def _encode_chunk(self, chunk):
        if chunk in self._cache:
            return self._cache[chunk]
        s = list(chunk)
        while len(s) > 1:
            best, bi = None, -1
            for i, p in enumerate(zip(s, s[1:])):
                r = self.rank.get(p)
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if best is None:
                break
            p = (s[bi], s[bi + 1])
            out, j = [], 0
            while j < len(s):
                if j + 1 < len(s) and (s[j], s[j + 1]) == p:
                    out.append(p[0] + p[1])
                    j += 2
                else:
                    out.append(s[j])
                    j += 1
            s = out
        ids = [self.stoi[t] for t in s]
        self._cache[chunk] = ids
        return ids

    def encode(self, text):
        ids = []
        for part, special in _split_specials(text, self.specials):
            if special:
                ids.append(self.stoi[part])
            else:
                _check_chars(part)
                for c in pretokenize(part):
                    ids.extend(self._encode_chunk(c))
        return ids

    def encode_iterable(self, iterable):
        """逐段读入，结果与 encode("".join(iterable)) 相同。只在换行符之后切开：换行总是单独一块，特殊符号里也没有换行。"""
        buf = ""
        for piece in iterable:
            buf += piece
            cut = buf.rfind("\n")
            if cut >= 0:
                yield from self.encode(buf[:cut + 1])
                buf = buf[cut + 1:]
        if buf:
            yield from self.encode(buf)

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)
