"""第 4 章：向量与矩阵的几个基本运算，亲手实现一遍，再和 NumPy 对照。"""
import numpy as np


# >>> 练习 4.1：用循环实现矩阵乘法
def matmul_loops(A, B):
    """C = A @ B。A 形状 (n, k)，B 形状 (k, m)，C 形状 (n, m)。

    C[i, j] = Σ_t A[i, t] · B[t, j]：A 的第 i 行与 B 的第 j 列逐项相乘再相加（点积）。
    内维 k 必须相等，否则抛出 ValueError。只用 Python 循环，不用 @ 或 np.dot。
    """
    n, k = A.shape
    k2, m = B.shape
    if k != k2:
        raise ValueError(f"内维不一致：A 是 {A.shape}，B 是 {B.shape}")
    C = np.zeros((n, m), dtype=np.result_type(A, B, np.float64))
    for i in range(n):
        for j in range(m):
            s = 0.0
            for t in range(k):
                s += A[i, t] * B[t, j]
            C[i, j] = s
    return C
# <<<


# >>> 练习 4.2：余弦相似度
def cosine_similarity(A, B, eps=1e-12):
    """A (N, D)、B (M, D) -> (N, M)，第 [i, j] 个元素是 A[i] 与 B[j] 夹角的余弦：
        cos = (a · b) / (|a| |b|)
    先把每一行除以它的长度（变成单位向量），再做一次矩阵乘法。eps 防止除以 0。
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + eps)
    return An @ Bn.T
# <<<


# >>> 练习 4.3：只用矩阵乘法的两两距离
def pairwise_sq_dists_mm(A, B):
    """与 miniocr.utils.pairwise_sq_dists 结果相同，但不创建 (N, M, D) 的中间数组：
        ||a - b||² = ||a||² - 2 a·b + ||b||²
    ||a||² 形状 (N, 1)，||b||² 形状 (1, M)，a·b 是 A @ B.T，形状 (N, M)。
    浮点误差可能让结果出现极小的负数，用 np.maximum(…, 0) 截掉。
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    aa = (A ** 2).sum(1, keepdims=True)
    bb = (B ** 2).sum(1, keepdims=True).T
    return np.maximum(aa - 2 * A @ B.T + bb, 0.0)
# <<<


def linear(x, W, b):
    """全连接层（线性层）：y = x W + b。x (..., d_in)，W (d_in, d_out)，b (d_out,)，y (..., d_out)。"""
    return x @ W + b


# >>> 练习 4.4：数参数
def mlp_param_count(sizes):
    """多层感知机（第 10 章）的参数个数。sizes = [d_in, h1, h2, ..., d_out]，
    每相邻两层之间是一个线性层：权重 d_a × d_b 个，偏置 d_b 个。
    例如 [1024, 256, 62] -> 1024·256 + 256 + 256·62 + 62 = 278,334。
    """
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))
# <<<


def matmul_flops(n, k, m):
    """(n, k) @ (k, m) 需要的浮点运算次数：每个输出元素 k 次乘法、k 次加法（约），共 2·n·k·m。"""
    return 2 * n * k * m


def rotation(theta):
    """把平面上的点逆时针旋转 theta（弧度）的 2×2 矩阵。注意图像坐标 y 向下，看起来是顺时针。"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])
