"""评测结果的不确定性：bootstrap 置信区间、配对 bootstrap、McNemar 检验。

一个测试集上的 CER = 2.28% 只是"这 2000 行"上的结果。换一批同分布的 2000 行，数字会变。
bootstrap（Efron，1979）的做法：把测试集当成总体的样本，有放回地重新抽 n 行、重新算指标，重复几千次，
看指标的分布有多宽。
"""
import math

import numpy as np


# >>> 练习 25.6：bootstrap 置信区间
def bootstrap_ci(num, den=None, n_boot=2000, alpha=0.05, seed=0):
    """比值型指标（如 CER = Σ编辑距离 / Σ参考长度）的 bootstrap 置信区间。

    num[i]、den[i] 是第 i 个样本的分子和分母；den 为 None 时每个样本的分母是 1（即普通的平均值，如准确率）。
    每次有放回地抽 n 个下标，算 Σnum / Σden；返回 (点估计, 下界, 上界)，下界、上界是
    n_boot 个结果的 alpha/2 和 1 − alpha/2 分位数（"百分位法"）。
    """
    num = np.asarray(num, dtype=np.float64)
    den = np.ones_like(num) if den is None else np.asarray(den, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    vals = num[idx].sum(1) / den[idx].sum(1)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return num.sum() / den.sum(), float(lo), float(hi)
# <<<


def paired_bootstrap(num_a, num_b, den=None, n_boot=2000, alpha=0.05, seed=0):
    """两个系统在**同一批**样本上的差（a − b）的置信区间。每次抽同一组下标，两个系统一起算——
    这样"这批样本恰好难/容易"的波动被抵消，只剩两个系统之间真正的差别。
    返回 (差的点估计, 下界, 上界, 差 ≤ 0 的比例)。最后一项接近 0 或 1 时，说明差别的方向很稳定。"""
    num_a, num_b = np.asarray(num_a, np.float64), np.asarray(num_b, np.float64)
    den = np.ones_like(num_a) if den is None else np.asarray(den, np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num_a), size=(n_boot, len(num_a)))
    d = (num_a[idx].sum(1) - num_b[idx].sum(1)) / den[idx].sum(1)
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    return (num_a.sum() - num_b.sum()) / den.sum(), float(lo), float(hi), float((d <= 0).mean())


def mcnemar(correct_a, correct_b):
    """McNemar 精确检验：两个系统在同一批样本上的对错（布尔数组）。
    只有"一个对、一个错"的样本有信息：b01 = a 错 b 对，b10 = a 对 b 错。
    原假设"两个系统一样好"下，这 b01 + b10 个样本里 a 赢的次数服从 Binomial(b01 + b10, 0.5)。
    返回 (b10, b01, 双侧 p 值)。"""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    b10, b01 = int((a & ~b).sum()), int((~a & b).sum())
    n, k = b10 + b01, min(b10, b01)
    if n == 0:
        return b10, b01, 1.0
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return b10, b01, min(1.0, 2 * p)
