"""第 12 章：卷积和池化，先用循环写出定义，再用 im2col 变成一次矩阵乘法，最后与 F.conv2d 对拍。

约定与 PyTorch 相同：
    输入 x (N, C, H, W)；卷积核 w (O, C, kh, kw)；偏置 b (O,)；输出 (N, O, H', W')。
    H' = (H + 2·padding − kh) // stride + 1（W' 同理）。
    深度学习里的"卷积"其实是互相关（cross-correlation）：卷积核不翻转。
"""
import numpy as np


# >>> 练习 12.1：输出大小与感受野
def conv_out_size(n, k, stride=1, padding=0):
    """边长为 n 的输入，经过核大小 k、步长 stride、四周补 padding 个 0 的卷积（或池化）后的边长。"""
    return (n + 2 * padding - k) // stride + 1


def receptive_field(layers):
    """layers：[(k, stride), ...]，从输入到输出依次排列。返回最后一层的一个输出像素"看得到"的输入范围边长。

    递推：r ← r + (k − 1) · j，j ← j · stride，初始 r = 1、j = 1。
    j 是当前层相邻两个像素在输入图上相距多少个像素（"跳距"）。
    """
    r, j = 1, 1
    for k, s in layers:
        r += (k - 1) * j
        j *= s
    return r
# <<<


# >>> 练习 12.2：用循环实现卷积
def conv2d_loops(x, w, b=None, stride=1, padding=0):
    """按定义计算：out[n, o, i, j] = b[o] + Σ_c Σ_u Σ_v x_pad[n, c, i·s + u, j·s + v] · w[o, c, u, v]。
    x_pad 是四周补 padding 个 0 的输入（np.pad）。可以对 (n, o, i, j) 四重循环，内部用切片与乘积求和。"""
    N, C, H, W = x.shape
    O, C2, kh, kw = w.shape
    if C != C2:
        raise ValueError(f"输入通道 {C} 与卷积核通道 {C2} 不一致")
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    Ho, Wo = conv_out_size(H, kh, stride, padding), conv_out_size(W, kw, stride, padding)
    out = np.zeros((N, O, Ho, Wo), dtype=np.result_type(x, w))
    for n in range(N):
        for o in range(O):
            for i in range(Ho):
                for j in range(Wo):
                    patch = xp[n, :, i * stride:i * stride + kh, j * stride:j * stride + kw]
                    out[n, o, i, j] = (patch * w[o]).sum()
    if b is not None:
        out += b[None, :, None, None]
    return out
# <<<


# >>> 练习 12.3：im2col
def im2col(x, kh, kw, stride=1, padding=0):
    """把每个卷积窗口拉平成一行：返回 (N·H'·W', C·kh·kw) 的矩阵，行按 (n, i, j) 的顺序排列，
    每行内部按 (c, u, v) 的顺序排列——与 w.reshape(O, -1) 的列顺序一致。"""
    N, C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    Ho, Wo = conv_out_size(H, kh, stride, padding), conv_out_size(W, kw, stride, padding)
    cols = np.empty((N, Ho, Wo, C, kh, kw), dtype=x.dtype)
    for u in range(kh):
        for v in range(kw):
            cols[:, :, :, :, u, v] = xp[:, :, u:u + stride * Ho:stride, v:v + stride * Wo:stride].transpose(0, 2, 3, 1)
    return cols.reshape(N * Ho * Wo, C * kh * kw)


def conv2d_im2col(x, w, b=None, stride=1, padding=0):
    """卷积 = im2col 得到的矩阵 @ 卷积核矩阵的转置，再把结果排回 (N, O, H', W')。"""
    N, C, H, W = x.shape
    O, _, kh, kw = w.shape
    Ho, Wo = conv_out_size(H, kh, stride, padding), conv_out_size(W, kw, stride, padding)
    out = im2col(x, kh, kw, stride, padding) @ w.reshape(O, -1).T          # (N·Ho·Wo, O)
    if b is not None:
        out = out + b
    return out.reshape(N, Ho, Wo, O).transpose(0, 3, 1, 2)
# <<<


# >>> 练习 12.4：最大池化
def maxpool2d(x, k):
    """k×k 不重叠的最大池化（步长 = k），H、W 不是 k 的倍数时丢掉多出的行列。x (N, C, H, W)。
    提示：先截成 (N, C, H//k·k, W//k·k)，reshape 成 (N, C, H//k, k, W//k, k)，在第 3、5 维取最大值。"""
    N, C, H, W = x.shape
    Hk, Wk = H // k, W // k
    return x[:, :, :Hk * k, :Wk * k].reshape(N, C, Hk, k, Wk, k).max(axis=(3, 5))
# <<<
