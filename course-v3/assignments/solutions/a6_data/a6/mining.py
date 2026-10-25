"""A6 参考实现：难例挖掘与配比。自洽度、公式的特征、k-means、按簇配额、簇内抽样权重、难例候选。"""
import re
from itertools import combinations

import numpy as np

from miniocr.data.latex import LatexError, parse

from .text import text_sim

_TOK = re.compile(r"\\[A-Za-z]+|\S")


def self_consistency(samples, sim=text_sim):
    pairs = list(combinations(range(len(samples)), 2))
    if not pairs:
        return 1.0
    return float(np.mean([sim(samples[i], samples[j]) for i, j in pairs]))


def formula_features(s):
    """[记号数, \\frac 数, \\sqrt 数, ^ 和 _ 的个数, 花括号的最大嵌套深度, + − = 的个数, 能否解析(1/0)]，float 数组。"""
    toks = _TOK.findall(s)
    depth = best = 0
    for t in toks:
        if t == "{":
            depth += 1
            best = max(best, depth)
        elif t == "}":
            depth = max(0, depth - 1)
    try:
        parse(s)
        ok = 1.0
    except LatexError:
        ok = 0.0
    return np.array([len(toks), toks.count("\\frac"), toks.count("\\sqrt"), sum(t in "^_" for t in toks), best,
                     sum(t in "+-=" for t in toks), ok], dtype=float)


def kmeans(X, k, rng, iters=50):
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
    labels = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
    return labels, C


def quota_sample(labels, weights, n, rng):
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
        if p is not None and (p > 0).sum() < quota[c]:
            p = None
        out.extend(rng.choice(idx, size=quota[c], replace=False, p=p).tolist())
    return np.array(out, dtype=int)


def sampling_weights(a, u, labels, f_a=1.0, f_u=1.0, tau=1.0, eps=1e-3):
    a, u, labels = np.asarray(a, float), np.asarray(u, float), np.asarray(labels)
    w = (a + eps) ** (f_a / tau) * (1 - u + eps) ** (f_u / tau)
    out = np.zeros_like(w)
    for k in np.unique(labels):
        m = labels == k
        out[m] = w[m] / w[m].sum()
    return out


def hard_candidates(agreement, consistency, tau_a=0.9, tau_u=0.7):
    return (np.asarray(agreement) >= tau_a) & (np.asarray(consistency) < tau_u)
