"""第 7 章：用 NumPy 和手推梯度实现三个最简单的模型。

    线性回归         ŷ = Xw + b                       损失：均方误差
    逻辑回归         p = σ(Xw + b)                    损失：二元交叉熵
    softmax 回归     p = softmax(XW + b)              损失：交叉熵（多分类）

每个 *_loss_grad 函数返回 (损失, 各参数的梯度)，梯度与参数形状相同。
损失都是对 N 个样本的平均，所以梯度里都有 1/N。
"""
import numpy as np

from ..calculus import sigmoid
from ..prob import log_softmax, softmax


# >>> 练习 7.1：线性回归的损失与梯度
def mse_loss_grad(X, y, w, b):
    """X (N, D)，y (N,)，w (D,)，b 是一个数。
    损失 L = (1/N) Σ (ŷ_i - y_i)²，ŷ = Xw + b。
    梯度：令 r = ŷ - y（残差），则 ∂L/∂w = (2/N) Xᵀ r，∂L/∂b = (2/N) Σ r。
    """
    r = X @ w + b - y
    N = len(y)
    return float((r ** 2).mean()), 2.0 / N * (X.T @ r), 2.0 / N * float(r.sum())
# <<<


# >>> 练习 7.2：逻辑回归的损失与梯度
def logistic_loss_grad(X, y, w, b):
    """X (N, D)，y (N,) 取 0 或 1，w (D,)，b 是一个数。
    p = σ(z)，z = Xw + b；损失 L = -(1/N) Σ [y log p + (1-y) log(1-p)]。
    梯度（第 5 章自己动手第 4 题）：∂L/∂z_i = (p_i - y_i)/N，于是 ∂L/∂w = Xᵀ(p - y)/N，∂L/∂b = Σ(p - y)/N。
    为了数值稳定，log p 用 -log(1 + e^{-z})，log(1-p) 用 -log(1 + e^{z})，都用 np.logaddexp(0, ·) 计算。
    """
    z = X @ w + b
    N = len(y)
    loss = float((y * np.logaddexp(0, -z) + (1 - y) * np.logaddexp(0, z)).mean())
    g = (sigmoid(z) - y) / N
    return loss, X.T @ g, float(g.sum())
# <<<


# >>> 练习 7.3：softmax 回归的损失与梯度
def softmax_loss_grad(X, y, W, b):
    """X (N, D)，y (N,) 是 0..K-1 的整数，W (D, K)，b (K,)。
    Z = XW + b，损失是交叉熵的平均。由第 6 章，∂L/∂Z = (softmax(Z) - Y)/N（Y 是独热矩阵），于是
        ∂L/∂W = Xᵀ ∂L/∂Z      (D, N) @ (N, K) = (D, K)
        ∂L/∂b = Σ_i ∂L/∂Z[i]  (K,)
    """
    Z = X @ W + b
    N = len(y)
    loss = float(-log_softmax(Z)[np.arange(N), y].mean())
    G = softmax(Z)
    G[np.arange(N), y] -= 1.0
    G /= N
    return loss, X.T @ G, G.sum(0)
# <<<


def standardize(X_train, *others, eps=1e-8):
    """用训练集的均值和标准差把每个特征变成"均值 0、标准差 1"，并对其他数据集做同样的变换。
    注意只能用训练集的统计量——用测试集的统计量就是让模型"偷看"了测试集（第 25 章的数据泄漏）。"""
    mu, sd = X_train.mean(0), X_train.std(0) + eps
    return [(X - mu) / sd for X in (X_train,) + others]
