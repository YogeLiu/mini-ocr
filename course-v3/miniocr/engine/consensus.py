"""第 33 章：多模型共识。几个"专家"模型各自给同一个样本写一个答案，从它们的一致程度里挑出伪标签。

    pairwise_agreement   n 个答案两两之间的相似度矩阵
    triplet_consensus    选出彼此最一致的 k 个专家（默认 3 个，"三元组"），取其中的中心点（medoid）当伪标签
    tier                 按三元组的一致度分档：high / medium / low
    majority_vote        对照组：出现次数最多的答案

相似度函数 sim(a, b) 在 [0, 1] 之间、对称：文字用 1 − NED，公式用 CDM，表格用 TEDS（第 25 章）。
"""
from collections import Counter
from itertools import combinations

import numpy as np

from ..metrics import ned


def text_sim(a, b):
    return 1.0 - ned(a, b)


# >>> 练习 33.1：两两一致度
def pairwise_agreement(preds, sim=text_sim):
    """preds：n 个答案。返回 (n, n) 的 numpy 数组 A，A[i, j] = sim(preds[i], preds[j])，对角线为 1。
    sim 对称，只算 i < j 的一半。"""
    n = len(preds)
    A = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            A[i, j] = A[j, i] = sim(preds[i], preds[j])
    return A
# <<<


# >>> 练习 33.2：三元组共识
def triplet_consensus(preds, sim=text_sim, k=3):
    """在所有 C(n, k) 个专家组合里，选组内两两一致度的平均值最高的一组（平局时取组合顺序靠前的）；
    组内的中心点 = 和组内其他成员一致度之和最大的那个（平局时取下标小的）。
    返回 (中心点在 preds 里的下标, 组内平均一致度, 组内专家的下标元组)。n < k 时用全部 n 个。"""
    A = pairwise_agreement(preds, sim)
    n = len(preds)
    k = min(k, n)
    best, best_score = tuple(range(k)), -1.0
    for group in combinations(range(n), k):
        pairs = list(combinations(group, 2))
        score = float(np.mean([A[i, j] for i, j in pairs])) if pairs else 1.0
        if score > best_score + 1e-12:
            best, best_score = group, score
    medoid = max(best, key=lambda i: (sum(A[i, j] for j in best if j != i), -i))
    return medoid, best_score, best
# <<<


def tier(score, hi=0.9, lo=0.6):
    """Xiaomi-OCR-0 报告里的三档：≥ 0.9 高（直接收下）、0.6–0.9 中（送去校验）、< 0.6 低。"""
    return "high" if score >= hi else ("medium" if score >= lo else "low")


def majority_vote(preds):
    """出现次数最多的答案（平局时取最先出现的）。只适合答案很短、容易完全一样的任务。"""
    c = Counter(preds)
    top = max(c.values())
    return next(p for p in preds if c[p] == top)
