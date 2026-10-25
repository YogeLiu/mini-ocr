"""A4 参考实现：一个数字有多可信。bootstrap、配对 bootstrap、McNemar；置信度的校准（分段、ECE）与区分能力（AUROC）。"""
import math

import numpy as np


def bootstrap_ci(num, den=None, n_boot=2000, alpha=0.05, seed=0):
    """比值型指标 Σnum / Σden 的百分位 bootstrap。下标用 default_rng(seed).integers(0, n, (n_boot, n)) 一次抽完，
    分位数用 np.quantile 的默认（线性插值）。返回 (点估计, 下界, 上界)。"""
    num = np.asarray(num, dtype=np.float64)
    den = np.ones_like(num) if den is None else np.asarray(den, dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, len(num), size=(n_boot, len(num)))
    vals = num[idx].sum(1) / den[idx].sum(1)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(num.sum() / den.sum()), float(lo), float(hi)


def paired_bootstrap(num_a, num_b, den=None, n_boot=2000, alpha=0.05, seed=0):
    """同一组下标同时抽两个系统，差 = (Σa − Σb) / Σden。返回 (差, 下界, 上界, 差 ≤ 0 的比例)。"""
    num_a, num_b = np.asarray(num_a, np.float64), np.asarray(num_b, np.float64)
    den = np.ones_like(num_a) if den is None else np.asarray(den, np.float64)
    idx = np.random.default_rng(seed).integers(0, len(num_a), size=(n_boot, len(num_a)))
    d = (num_a[idx].sum(1) - num_b[idx].sum(1)) / den[idx].sum(1)
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    return float((num_a.sum() - num_b.sum()) / den.sum()), float(lo), float(hi), float((d <= 0).mean())


def mcnemar(correct_a, correct_b):
    """精确检验。b10 = a 对 b 错，b01 = a 错 b 对；p = min(1, 2·P(Binomial(b10 + b01, ½) ≤ min(b10, b01)))。
    返回 (b10, b01, p)。"""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    b10, b01 = int((a & ~b).sum()), int((~a & b).sum())
    n, k = b10 + b01, min(b10, b01)
    if n == 0:
        return b10, b01, 1.0
    return b10, b01, min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def reliability(conf, correct, n_bins=10):
    """等宽分段 [0, 1/n), …, [(n−1)/n, 1]（1.0 落在最后一段）。返回有样本的段 [(下界, 上界, 平均置信度, 准确率, 样本数)]。"""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, n_bins - 1)
    out = []
    for k in range(n_bins):
        sel = idx == k
        if sel.any():
            out.append((float(edges[k]), float(edges[k + 1]), float(conf[sel].mean()), float(correct[sel].mean()), int(sel.sum())))
    return out


def ece(conf, correct, n_bins=10):
    n = len(conf)
    return float(sum(c / n * abs(acc - m) for _, _, m, acc, c in reliability(conf, correct, n_bins)))


def auroc(scores, labels):
    """Mann–Whitney：并列取平均名次，(R₊ − n₊(n₊+1)/2) / (n₊·n₋)。只有一类时返回 nan。"""
    s, y = np.asarray(scores, float), np.asarray(labels).astype(bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))
