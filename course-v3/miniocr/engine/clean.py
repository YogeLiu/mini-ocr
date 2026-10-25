"""第 33 章：清洗、去重与划分。真实数据进训练之前的最后几道闸门。

    latex_key          去重用的规范化：去掉 $$…$$、\\[…\\] 外壳和外层的 align/equation 环境、\\displaystyle、多余的空格
                       （命令名后面紧跟字母的空格要留下），单个字符的上下标加花括号
    dedup              精确去重：按 key 保留第一次出现的样本
    shingles           字符 n-gram 的集合（近似去重的基本单位）
    minhash            MinHash 签名：两个集合的签名有多少位相同，近似它们的 Jaccard 相似度
    lsh_pairs          局部敏感哈希：把签名切成若干段，任一段完全相同的两条成为候选对
    group_split        按组划分：同一个组（同一篇论文、同一个文档、同一个模板）的样本只进训练或只进验证
"""
import hashlib
import re
from collections import defaultdict

import numpy as np

_ENV = re.compile(r"^\\begin\{(align\*?|equation\*?|gather\*?|displaymath)\}(.*)\\end\{\1\}$", re.S)


_MATH = re.compile(r"^(\$\$|\\\[|\$)(.*?)(\$\$|\\\]|\$)$", re.S)


def latex_key(s):
    s = s.strip()
    m = _MATH.match(s)                                              # 模型的输出常带 $$…$$、\[…\]、$…$ 外壳
    if m:
        s = m.group(2).strip()
    m = _ENV.match(s)
    if m:
        s = m.group(2)
    s = re.sub(r"(?<!\\)([_^])([A-Za-z0-9])", r"\1{\2}", s)         # x^2 与 x^{2} 等价：单个字符的上下标统一加花括号
    s = s.replace("\\displaystyle", "")
    s = re.sub(r"\\[,;:! ]|~|\\quad|\\qquad", " ", s)              # 只影响间距的命令
    s = re.sub(r"(\\[A-Za-z]+)\s+(?=[A-Za-z])", "\\1\x00", s)          # "\\alpha x" 里的空格不能删：删了就成了 \\alphax
    return re.sub(r"\s+", "", s).replace("\x00", " ")


def dedup(items, key=lambda x: x):
    """返回保留下来的下标（每个 key 第一次出现的那条）。"""
    seen, keep = set(), []
    for i, x in enumerate(items):
        k = key(x)
        if k not in seen:
            seen.add(k)
            keep.append(i)
    return keep


def shingles(s, n=5):
    """字符串的所有长度为 n 的子串组成的集合；比 n 短的字符串返回 {s}。"""
    return {s[i:i + n] for i in range(len(s) - n + 1)} or {s}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def _h(x):
    """字符串 -> [0, 2³²) 的整数（确定性的哈希，和 Python 的 hash() 不同，每次运行都一样）。"""
    return int.from_bytes(hashlib.blake2b(x.encode(), digest_size=4).digest(), "little")


_P = (1 << 31) - 1                                              # 一个素数
_AB = np.random.default_rng(12345).integers(1, _P, size=(2, 1024), dtype=np.int64)


# >>> 练习 33.6：MinHash
def minhash(sh, n_hash=64):
    """sh：一个字符串集合。先把每个元素哈希成整数 x = _h(元素)，第 i 个哈希函数是 h_i(x) = (a_i · x + b_i) mod P
    （a_i = _AB[0, i]，b_i = _AB[1, i]，P = _P，用 int64 计算不会溢出）。签名的第 i 位 = 集合里所有元素 h_i 的最小值。
    返回长度为 n_hash 的 numpy int64 数组。性质：两个集合签名的某一位相同的概率 ≈ 它们的 Jaccard 相似度。"""
    x = np.array([_h(e) for e in sh], dtype=np.int64)
    a, b = _AB[0, :n_hash, None], _AB[1, :n_hash, None]
    return ((a * x[None] + b) % _P).min(1)
# <<<


def lsh_pairs(sigs, bands=16):
    """sigs (N, n_hash)。把每条签名切成 bands 段（每段 r = n_hash // bands 位），任一段完全相同的两条成为候选对。
    两个 Jaccard 为 J 的集合成为候选的概率 = 1 − (1 − J^r)^bands。返回候选对 (i, j)（i < j）的集合。"""
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


# >>> 练习 33.7：按组划分
def group_split(groups, val_frac, seed=0):
    """groups：每个样本所属的组（任意可哈希的值）。把所有不同的组按 seed 随机打乱（np.random.default_rng(seed).permutation），
    依次放进验证集，直到验证集的**样本数**第一次 ≥ val_frac × 总数为止；其余的组进训练集。
    返回 (train_idx, val_idx) 两个下标列表（升序）。同一组的样本永远在同一边。"""
    uniq = sorted(set(groups), key=str)
    order = np.random.default_rng(seed).permutation(len(uniq))
    target, n_val, val_groups = val_frac * len(groups), 0, set()
    count = defaultdict(int)
    for g in groups:
        count[g] += 1
    for k in order:
        if n_val >= target:
            break
        val_groups.add(uniq[k])
        n_val += count[uniq[k]]
    val = [i for i, g in enumerate(groups) if g in val_groups]
    train = [i for i, g in enumerate(groups) if g not in val_groups]
    return train, val
# <<<
