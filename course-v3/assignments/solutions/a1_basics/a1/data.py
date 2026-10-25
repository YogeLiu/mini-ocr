"""A1 参考实现：数据。只用 miniocr.data 的渲染器生成图像，其余自己写。"""
import numpy as np
import torch

from miniocr.data.tasks import CHARSET, glyph_arrays


def build_split(seed, n_per_class, distort=1.0, noise=0.05, chars=CHARSET):
    """用种子 seed 渲染一份数据。返回 X：float32 (N, 1, 32, 32)，墨迹 = 1；y：int64 (N,)。"""
    X, y = glyph_arrays(np.random.default_rng(seed), chars=chars, n_per_class=n_per_class,
                        noise=noise, distort=distort, flat=False)
    return X[:, None], y


def remove_overlap(X, y, X_ref):
    """去掉 X 里与 X_ref 中某张图逐像素完全相同的样本。返回 (X, y, 去掉的个数)。"""
    seen = {r.tobytes() for r in np.ascontiguousarray(X_ref)}
    keep = np.array([r.tobytes() not in seen for r in np.ascontiguousarray(X)], dtype=bool)
    return X[keep], y[keep], int((~keep).sum())


def get_batch(X, y, batch_size, generator, device="cpu"):
    """有放回地随机取 batch_size 个样本，下标由 generator（torch.Generator）决定。返回 (xb, yb) 两个张量。"""
    idx = torch.randint(0, len(X), (batch_size,), generator=generator).numpy()
    return torch.from_numpy(X[idx]).to(device), torch.from_numpy(y[idx]).to(device)
