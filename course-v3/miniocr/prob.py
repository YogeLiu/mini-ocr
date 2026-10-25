"""第 6 章：概率分布、softmax、交叉熵、熵与 KL 散度（NumPy 版，第 9 章起改用 PyTorch 的同名函数）。

约定：logits 是模型输出的"未归一化的分数"，形状 (..., K)，最后一维是 K 个类别。
所有函数都沿最后一维计算。
"""
import numpy as np


# >>> 练习 6.1：数值稳定的 softmax
def softmax(logits):
    """softmax(z)_k = e^{z_k} / Σ_j e^{z_j}，把任意实数变成和为 1 的正数（一个概率分布）。

    直接算 e^{z} 时，z = 1000 会溢出成 inf，inf/inf = nan。
    技巧：先减去最大值。分子分母同乘 e^{-max}，结果不变，但最大的指数变成 e^0 = 1，不会溢出。
    """
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
# <<<


# >>> 练习 6.2：log-softmax
def log_softmax(logits):
    """log softmax(z)_k = z_k - log Σ_j e^{z_j}。

    不要写成 np.log(softmax(z))：概率极小时 softmax 会下溢成 0，log(0) = -inf。
    用 logsumexp 技巧：log Σ e^{z_j} = m + log Σ e^{z_j - m}，m = max z。
    """
    z = np.asarray(logits, dtype=np.float64)
    m = z.max(axis=-1, keepdims=True)
    return z - (m + np.log(np.exp(z - m).sum(axis=-1, keepdims=True)))
# <<<


# >>> 练习 6.3：交叉熵损失
def cross_entropy(logits, labels):
    """分类的交叉熵损失：每个样本取 -log(模型给正确类别的概率)，再对样本求平均。

    logits (N, K)，labels (N,) 是 0..K-1 的整数。返回一个数。
    用 log_softmax 再取出正确类别那一列（花式下标），不要先算 softmax 再取 log。
    """
    lp = log_softmax(logits)
    labels = np.asarray(labels)
    return float(-lp[np.arange(len(labels)), labels].mean())
# <<<


# >>> 练习 6.4：熵与 KL 散度
def entropy(p):
    """H(p) = -Σ p_k log p_k（自然对数，单位"奈特"nat）。约定 0·log 0 = 0。沿最后一维计算。"""
    p = np.asarray(p, dtype=np.float64)
    safe = np.where(p > 0, p, 1.0)
    return -(p * np.log(safe)).sum(axis=-1)


def kl_divergence(p, q):
    """KL(p || q) = Σ p_k log(p_k / q_k)。p_k = 0 的项为 0；p_k > 0 而 q_k = 0 时为无穷大。"""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    with np.errstate(divide="ignore"):
        terms = np.where(p > 0, p * (np.log(np.where(p > 0, p, 1.0)) - np.log(q)), 0.0)
    return terms.sum(axis=-1)
# <<<


def accuracy(logits, labels):
    """准确率：预测（分数最高的类）等于正确答案的比例。"""
    return float((np.asarray(logits).argmax(-1) == np.asarray(labels)).mean())
