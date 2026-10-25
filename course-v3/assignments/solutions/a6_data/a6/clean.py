"""A6 参考实现：规范化、精确去重、MinHash 与 LSH 的近似去重、近似重复的簇、按组划分、去污染。"""
import hashlib
import re
from collections import defaultdict

import numpy as np

from miniocr.data.latex import LatexError, normalize

_MATH = re.compile(r"^(\$\$|\\\[|\$)(.*?)(\$\$|\\\]|\$)$", re.S)
_ENV = re.compile(r"^\\begin\{(align\*?|equation\*?|gather\*?|displaymath)\}(.*)\\end\{\1\}$", re.S)


def latex_key(s):
    """不解析的规范化（任意 LaTeX 都能用）：去外壳、去外层环境、单字符上下标加花括号、去 \\displaystyle、
    间距命令变空格、去掉所有空格（但"命令名 + 空格 + 字母"里的空格换成一个保留的空格）。"""
    s = s.strip()
    m = _MATH.match(s)
    if m:
        s = m.group(2).strip()
    m = _ENV.match(s)
    if m:
        s = m.group(2)
    s = re.sub(r"(?<!\\)([_^])([A-Za-z0-9])", r"\1{\2}", s)
    s = s.replace("\\displaystyle", "")
    s = re.sub(r"\\[,;:! ]|~|\\quad|\\qquad", " ", s)
    s = re.sub(r"(\\[A-Za-z]+)\s+(?=[A-Za-z])", "\\1\x00", s)
    return re.sub(r"\s+", "", s).replace("\x00", " ")


def canon(s):
    """能按本书的文法解析就写成规范写法（normalize），否则退回 latex_key。"""
    try:
        return normalize(latex_key(s))
    except LatexError:
        return latex_key(s)


def dedup(items, key=lambda x: x):
    seen, keep = set(), []
    for i, x in enumerate(items):
        k = key(x)
        if k not in seen:
            seen.add(k)
            keep.append(i)
    return keep


def shingles(s, n=3):
    return {s[i:i + n] for i in range(len(s) - n + 1)} or {s}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def str_hash(x):
    return int.from_bytes(hashlib.blake2b(x.encode(), digest_size=4).digest(), "little")


P = (1 << 31) - 1
AB = np.random.default_rng(12345).integers(1, P, size=(2, 1024), dtype=np.int64)


def minhash(sh, n_hash=64):
    x = np.array([str_hash(e) for e in sh], dtype=np.int64)
    a, b = AB[0, :n_hash, None], AB[1, :n_hash, None]
    return ((a * x[None] + b) % P).min(1)


def lsh_pairs(sigs, bands=16):
    sigs = np.asarray(sigs)
    r = sigs.shape[1] // bands
    out = set()
    for b in range(bands):
        buckets = defaultdict(list)
        for i, row in enumerate(sigs[:, b * r:(b + 1) * r]):
            buckets[row.tobytes()].append(i)
        for ids in buckets.values():
            for x in range(len(ids)):
                for y in range(x + 1, len(ids)):
                    out.add((ids[x], ids[y]))
    return out


def near_dup_clusters(sets, pairs, tau):
    """候选对里 Jaccard ≥ tau 的连起来（并查集），返回每个样本所在簇的编号 = 簇里最小的下标。"""
    parent = list(range(len(sets)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in sorted(pairs):
        if jaccard(sets[i], sets[j]) >= tau:
            a, b = find(i), find(j)
            if a != b:
                parent[max(a, b)] = min(a, b)
    return [find(i) for i in range(len(sets))]


def group_split(groups, val_frac, seed=0):
    uniq = sorted(set(groups), key=str)
    order = np.random.default_rng(seed).permutation(len(uniq))
    count = defaultdict(int)
    for g in groups:
        count[g] += 1
    target, n_val, val_groups = val_frac * len(groups), 0, set()
    for k in order:
        if n_val >= target:
            break
        val_groups.add(uniq[k])
        n_val += count[uniq[k]]
    return ([i for i, g in enumerate(groups) if g not in val_groups], [i for i, g in enumerate(groups) if g in val_groups])


def decontaminate(pool_sets, test_sets, tau=0.8, n_hash=64, bands=16):
    """数据池里和任何一条测试样本 Jaccard ≥ tau 的样本下标（升序）。候选来自 LSH（数据池在前、测试集在后拼在一起），再用精确的 Jaccard 核对。"""
    n = len(pool_sets)
    sigs = np.stack([minhash(s, n_hash) for s in list(pool_sets) + list(test_sets)])
    out = set()
    for i, j in lsh_pairs(sigs, bands):
        if i < n <= j and jaccard(pool_sets[i], test_sets[j - n]) >= tau:
            out.add(i)
    return sorted(out)
