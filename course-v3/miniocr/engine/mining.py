"""第 33 章：难例挖掘。从一大堆样本里挑出"值得训练"的：模型自己没把握、而标签又可信的样本。

    self_consistency   同一个样本采样 G 次，回答两两之间的平均相似度：越低说明模型越没把握
    kmeans             最简单的 k-means（numpy），给样本的特征聚类
    quota_sample       按簇分配名额：每个簇的名额相同（簇不够大时把多出的名额分给其他簇），簇内按权重无放回抽
    hard_candidates    难例候选：专家一致度高（标签可信）且模型自洽度低（模型不会）
"""
from itertools import combinations

import numpy as np

from .consensus import text_sim


# >>> 练习 33.4：自洽度
def self_consistency(samples, sim=text_sim):
    """samples：同一个输入的 G 个采样回答。返回两两相似度的平均值；G < 2 时返回 1.0。"""
    pairs = list(combinations(range(len(samples)), 2))
    if not pairs:
        return 1.0
    return float(np.mean([sim(samples[i], samples[j]) for i, j in pairs]))
# <<<


def kmeans(X, k, rng, iters=50):
    """X (N, D) 的 numpy 数组。k-means++ 初始化，再交替"分配到最近的中心 / 中心取平均"。返回 (labels (N,), centers (k, D))。"""
    X = np.asarray(X, dtype=float)
    centers = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d = np.min([((X - c) ** 2).sum(1) for c in centers], axis=0)
        centers.append(X[rng.choice(len(X), p=d / d.sum())])
    C = np.array(centers)
    for _ in range(iters):
        labels = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        new = np.array([X[labels == j].mean(0) if (labels == j).any() else C[j] for j in range(k)])
        if np.allclose(new, C):
            break
        C = new
    return labels, C


# >>> 练习 33.5：按簇分配名额
def quota_sample(labels, weights, n, rng):
    """labels (N,)：每个样本的簇编号；weights (N,)：簇内抽样的权重（≥ 0，比如"难度"）；一共抽 n 个（n ≤ N）。
    名额：先给每个簇平均分 n // 簇数 个（余数给编号小的簇，一簇一个），簇的样本数不够名额时只取它全部的样本，
    剩下的名额再在"还有剩余样本的簇"之间重新平均分配，重复直到分完。
    簇内：按 weights 成比例、无放回地抽（rng.choice(..., replace=False, p=...)；权重全为 0 时均匀抽）。
    返回选中的下标（numpy 数组，按簇编号、簇内抽取的顺序）。"""
    labels = np.asarray(labels)
    weights = np.asarray(weights, dtype=float)
    clusters = sorted(set(labels.tolist()))
    size = {c: int((labels == c).sum()) for c in clusters}
    quota = {c: 0 for c in clusters}
    left = n
    while left > 0:
        open_ = [c for c in clusters if quota[c] < size[c]]
        if not open_:
            break
        share, extra = divmod(left, len(open_))
        for i, c in enumerate(open_):
            add = min(share + (1 if i < extra else 0), size[c] - quota[c])
            quota[c] += add
            left -= add
    out = []
    for c in clusters:
        idx = np.nonzero(labels == c)[0]
        w = weights[idx]
        p = w / w.sum() if w.sum() > 0 else None
        if p is not None and (p > 0).sum() < quota[c]:              # 正权重的样本不够名额：退回均匀
            p = None
        out.extend(rng.choice(idx, size=quota[c], replace=False, p=p).tolist())
    return np.array(out, dtype=int)
# <<<


def hard_candidates(agreement, consistency, tau_a=0.9, tau_u=0.7):
    """标签可信（专家一致度 ≥ tau_a）而模型没把握（自洽度 < tau_u）的样本。返回布尔数组。"""
    return (np.asarray(agreement) >= tau_a) & (np.asarray(consistency) < tau_u)
