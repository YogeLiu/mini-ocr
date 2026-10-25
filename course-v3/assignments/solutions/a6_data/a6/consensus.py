"""A6 参考实现：多模型共识。两两一致度、三元组共识、分档、多数票、加权投票、单参数的 EM（专家可靠度）。"""
from collections import Counter
from itertools import combinations

import numpy as np

from .text import text_sim


def pairwise_agreement(preds, sim=text_sim):
    n = len(preds)
    A = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            A[i, j] = A[j, i] = sim(preds[i], preds[j])
    return A


def triplet_consensus(preds, sim=text_sim, k=3):
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


def tier(score, hi=0.9, lo=0.6):
    return "high" if score >= hi else ("medium" if score >= lo else "low")


def majority_vote(preds):
    c = Counter(preds)
    top = max(c.values())
    return next(p for p in preds if c[p] == top)


def weighted_vote(preds, weights):
    """每个不同的答案得分 = 写它的专家的权重之和；最高分的答案（平局时取最先出现的）。"""
    score = {}
    for p, w in zip(preds, weights):
        score[p] = score.get(p, 0.0) + float(w)
    top = max(score.values())
    return next(p for p in preds if score[p] >= top - 1e-12)


def one_coin_em(outputs, iters=10, eps=1e-3):
    """outputs：N 行，每行 K 个专家的答案。权重从全 1 开始；每轮：加权投票得到标签，
    acc_k = (和标签相同的条数 + 1) / (N + 2)，裁到 [eps, 1 − eps]，w_k = log(acc_k / (1 − acc_k))。
    返回 (标签列表, acc (K,))。"""
    K = len(outputs[0])
    w = np.ones(K)
    acc = np.full(K, 0.5)
    for _ in range(iters):
        labels = [weighted_vote(row, w) for row in outputs]
        hits = np.array([sum(row[k] == l for row, l in zip(outputs, labels)) for k in range(K)], dtype=float)
        acc = np.clip((hits + 1) / (len(outputs) + 2), eps, 1 - eps)
        w = np.log(acc / (1 - acc))
    return [weighted_vote(row, w) for row in outputs], acc
