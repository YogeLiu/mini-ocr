"""第 12 章：规范键去重、MinHash 与 LSH；只依赖标准库和 numpy。"""
from collections import defaultdict
import hashlib
from itertools import combinations
import json
import random
import time

import numpy as np

from miniocr.data.latex_corpus import normalization_key


def exact_key(text):
    """规范化键的 SHA-256 十六进制摘要。"""
    return hashlib.sha256(normalization_key(text).encode("utf-8")).hexdigest()


def exact_dedup(texts):
    """返回（保留的原文列表，删除条数）；每组保留首次出现者，顺序稳定。"""
    seen, kept, removed = set(), [], 0
    for text in texts:
        key = exact_key(text)
        if key in seen:
            removed += 1
        else:
            seen.add(key)
            kept.append(text)
    return kept, removed


def shingles(text, n=3):
    """取规范键空格切分后的 token n-gram 集合，元素为元组。

    token 保留完整命令，避免字符切片把不同命令的共同前缀当作相似结构。
    少于 n 个 token 时整串为一个 shingle；空串为空集。解析失败时沿用
    normalization_key 的原串回退，因此坏串的空格切分不保证是 LaTeX token。
    """
    if n < 1:
        raise ValueError("n 必须为正")
    tokens = normalization_key(text).split()
    if not tokens:
        return set()
    return {tuple(tokens[i:i+n]) for i in range(max(1, len(tokens)-n+1))}


def jaccard(a, b):
    """精确集合 Jaccard；两个空集约定为 1.0。"""
    intersection = len(a & b)
    union = len(a) + len(b) - intersection
    return intersection / union if union else 1.0


def _stable_hash(shingle):
    """字符串或 token 元组 → 稳定的 64 位整数；编码保留 token 边界。"""
    payload = json.dumps(shingle, ensure_ascii=False, separators=(",", ":"))
    return int.from_bytes(hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest(), "little")


class MinHasher:
    """k 个通用哈希；空集签名全部为 uint64 最大值，不与非空签名相撞。

    h_i(x)=(a_i*x+b_i) mod p，选梅森素数 p=2**31-1。稳定 blake2b
    64 位哈希先 mod p 得到 x；1 <= a < p、0 <= b < p，故 a*x+b < 2**62，
    可直接用 uint64 乘加而不溢出。31 位值域足够本章规模的近似候选筛选；
    理想均匀假设下，单对不同 shingle 碰撞概率约为 1/p，m 个不同 shingle
    的期望碰撞对数约为 m*(m-1)/(2*p)，并非保证全语料无碰撞。
    系数用局部随机数发生器，完全不依赖 Python 的 hash 随机盐。
    通用哈希是教学近似，并非严格的随机排列族。
    """
    P = (1 << 31) - 1

    def __init__(self, num_perm=128, seed=0):
        if num_perm < 1:
            raise ValueError("num_perm 必须为正")
        rng = random.Random(seed)
        self.a = np.array([rng.randrange(1, self.P) for _ in range(num_perm)], dtype=np.uint64)
        self.b = np.array([rng.randrange(self.P) for _ in range(num_perm)], dtype=np.uint64)
        self.num_perm = num_perm

    def _hashes(self, x):
        """x 是 [0, p) 内的 uint64 数组；广播得到 (k, m) 哈希矩阵。"""
        return (self.a[:, None] * x[None, :] + self.b[:, None]) % self.P

    def signature(self, shingle_set):
        """返回长度 k 的 uint64 数组；m 个 shingle 的临时矩阵占 O(k*m) 内存。"""
        # >>> 练习 12.1：MinHash 签名
        if not shingle_set:
            return np.full(self.num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
        x = np.array([_stable_hash(s) % self.P for s in shingle_set], dtype=np.uint64)
        hashes = self._hashes(x)
        return hashes.min(axis=1)
        # <<<


def estimate_jaccard(sig1, sig2):
    """签名逐位相等的比例；要求非空、同长的一维签名。"""
    a, b = np.asarray(sig1), np.asarray(sig2)
    if a.ndim != 1 or a.shape != b.shape or not a.size:
        raise ValueError("签名必须为非空、同长的一维数组")
    return float(np.mean(a == b))


def _check_bands(bands, rows):
    if bands < 1 or rows < 1:
        raise ValueError("bands 和 rows 必须为正")


def lsh_candidates(signatures, bands, rows):
    """返回候选下标对集合；只使用前 bands*rows 行，不截断大桶。"""
    _check_bands(bands, rows)
    signatures = np.asarray(signatures, dtype=np.uint64)
    if signatures.ndim != 2 or bands*rows > signatures.shape[1]:
        raise ValueError("签名须为二维数组且 bands*rows 不得超过签名长度")
    candidates = set()
    # >>> 练习 12.2：LSH 分段分桶
    for band in range(bands):
        buckets = defaultdict(list)
        for i, sig in enumerate(signatures):
            key = sig[band*rows:(band+1)*rows].astype("<u8", copy=False).tobytes()
            buckets[key].append(i)
        for indices in buckets.values():
            candidates.update(combinations(indices, 2))
    # <<<
    return candidates


def lsh_probability(s, bands, rows):
    """独立理想哈希假设下，至少一段相同的概率。"""
    _check_bands(bands, rows)
    if not 0 <= s <= 1:
        raise ValueError("相似度须在 [0, 1] 内")
    return 1 - (1 - s**rows)**bands


def lsh_threshold(bands, rows):
    """S 曲线的近似转折点，不是硬阈值或精确的半概率点。"""
    _check_bands(bands, rows)
    return (1/bands)**(1/rows)


def near_dedup(texts, threshold=.8, num_perm=128, bands=16, rows=8, n=3, seed=0):
    """返回（保留下标，信息字典）；簇含全部成员下标，含单例簇。

    候选经精确 Jaccard 核验，再用并查集合成连通分量；保留最早下标。
    连通不等于簇内任意两条都超过阈值。LSH 可能漏召回，不能保证全局无近重复。
    信息保留候选对及通过对，供打印误报与召回；极端大桶仍可能二次增长。
    timings 记录实际计算秒数：signature_seconds 含 shingle 构造与签名，
    lsh_verify_seconds 含 LSH、精确核验及合簇，total_seconds 覆盖完整流程。
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold 须在 [0, 1] 内")
    if n < 1:
        raise ValueError("n 必须为正")
    begun = time.perf_counter()
    sets = [shingles(t, n) for t in texts]
    hasher = MinHasher(num_perm, seed)
    signatures = np.array([hasher.signature(s) for s in sets], dtype=np.uint64).reshape(-1, num_perm)
    signed = time.perf_counter()
    candidates = lsh_candidates(signatures, bands, rows)
    parent = list(range(len(sets)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    accepted = set()
    for i, j in sorted(candidates):
        if jaccard(sets[i], sets[j]) >= threshold:
            accepted.add((i, j))
            a, b = find(i), find(j)
            parent[max(a, b)] = min(a, b)
    groups = defaultdict(list)
    for i in range(len(sets)):
        groups[find(i)].append(i)
    clusters = list(groups.values())
    keep = [c[0] for c in clusters]
    finished = time.perf_counter()
    timings = dict(signature_seconds=signed-begun, lsh_verify_seconds=finished-signed,
                   total_seconds=finished-begun)
    return keep, dict(clusters=clusters, candidates=candidates, accepted=accepted,
                     signatures=signatures, timings=timings)
