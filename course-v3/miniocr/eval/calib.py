"""第 27 章：置信度有多可信。模型说"我有 90% 的把握"时，它真的有 90% 的时候是对的吗？

    reliability   把样本按置信度分成 n_bins 段，每段的平均置信度、准确率、样本数
    ece           期望校准误差（expected calibration error）：各段 |准确率 − 置信度| 按样本数加权平均
    auroc         用一个分数区分两类样本的能力：随机抽一个正例、一个负例，正例分数更高的概率（并列算一半）

用途：OCR 里用置信度挑出"可能读错的行"交给人看，或者在第 33 章的数据工程里挑难例。
"""
import numpy as np


def reliability(conf, correct, n_bins=10):
    """conf：每个样本的置信度（0–1），correct：每个样本是否正确（0/1）。等宽分段 [0, 0.1), [0.1, 0.2), …, [0.9, 1.0]。
    返回 [(段的下界, 段的上界, 平均置信度, 准确率, 样本数), …]，只包括有样本的段。"""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, n_bins - 1)
    out = []
    for k in range(n_bins):
        sel = idx == k
        if sel.any():
            out.append((edges[k], edges[k + 1], conf[sel].mean(), correct[sel].mean(), int(sel.sum())))
    return out


# >>> 练习 27.5：期望校准误差
def ece(conf, correct, n_bins=10):
    """ECE = Σ_段 (段内样本数 / 总样本数) · |段内准确率 − 段内平均置信度|。用 reliability 分段。
    完全校准的模型 ECE = 0；总说"100% 有把握"、实际只对一半的模型 ECE = 0.5。"""
    n = len(conf)
    return float(sum(cnt / n * abs(acc - c) for _, _, c, acc, cnt in reliability(conf, correct, n_bins)))
# <<<


# >>> 练习 27.6：AUROC
def auroc(scores, labels):
    """scores：每个样本的分数（越大越像正例），labels：1 = 正例，0 = 负例。
    AUROC = P(随机正例的分数 > 随机负例的分数) + 0.5·P(两者相等)。用排名计算（Mann–Whitney U）：
    把所有分数排序，并列的取平均名次（从 1 开始），R = 正例名次之和，
        AUROC = (R − n₊(n₊ + 1)/2) / (n₊ · n₋)。
    只有一类样本时返回 nan。"""
    s, y = np.asarray(scores, float), np.asarray(labels).astype(bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):                                   # 并列的一组取平均名次
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))
# <<<
