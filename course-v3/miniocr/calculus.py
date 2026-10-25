"""第 5 章：用"数值方法"求导数和梯度。

数值导数又慢又不太准，训练时不用它；但它几乎不可能写错，所以是检查手推梯度（第 7 章）、
自动求导（第 8 章）和自定义算子（第 13 章 CTC）的标准工具，叫"梯度检查"（gradient check）。
"""
import numpy as np


# >>> 练习 5.1：数值导数
def numerical_derivative(f, x, h=1e-5):
    """一元函数 f 在 x 处的导数，用中心差分：
        f'(x) ≈ (f(x + h) - f(x - h)) / (2h)
    比前向差分 (f(x+h) - f(x)) / h 准得多：误差是 h² 量级而不是 h 量级（见第 5 章"深入一步"）。
    """
    return (f(x + h) - f(x - h)) / (2 * h)
# <<<


# >>> 练习 5.2：数值梯度
def numerical_gradient(f, x, h=1e-5):
    """多元函数 f（输入一个任意形状的数组，输出一个数）在 x 处的梯度，形状与 x 相同。

    对 x 的每一个元素分别做中心差分：只改这一个元素，其他不动。
    注意：要在 x 的副本上修改，并且改完一定要恢复原值，否则后面的元素会在错误的位置求导。
    """
    x = np.array(x, dtype=np.float64)                 # 副本，不修改调用者的数组
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):                   # 遍历每一个下标，例如 (0, 0), (0, 1), ...
        old = x[idx]
        x[idx] = old + h
        fp = f(x)
        x[idx] = old - h
        fm = f(x)
        x[idx] = old                                  # 恢复
        grad[idx] = (fp - fm) / (2 * h)
    return grad
# <<<


def rel_error(a, b, eps=1e-12):
    """两个数组的最大相对误差 max |a - b| / max(|a| + |b|, eps)。梯度检查时，< 1e-6 一般说明正确。"""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(a) + np.abs(b), eps)))


def sigmoid(z):
    """σ(z) = 1 / (1 + e^{-z})，把任意实数压到 (0, 1)。导数 σ'(z) = σ(z)(1 - σ(z))。

    z 是很大的负数时 e^{-z} 会溢出，所以写成 exp(-log(1 + e^{-z}))，log(1 + e^{-z}) 用 np.logaddexp(0, -z) 稳定地计算。
    """
    return np.exp(-np.logaddexp(0.0, -np.asarray(z, dtype=np.float64)))
