"""第 5 章的数据：单个公式符号的小灰度图，带"来源组"标记，用来演示划分、泄漏和近重复检测。

    SYMBOLS          符号表：[(类别名, mathtext 写法), ...]，共 24 类
    FONTSETS         matplotlib mathtext 自带的 5 套数学字体
    render_symbol    用 mathtext 把一个符号渲染成 96×96 的"母版"（墨迹 = 1，纸 = 0）
    augment          对母版做增广：缩放、旋转、平移、笔画粗细、对比度、噪声，输出 32×32
    make_dataset     生成数据集：每个"来源"有自己的字体、姿态和扫描污点，同一来源的样本是近重复
    random_split     按样本随机划分（会泄漏）
    group_split      按来源分组划分（练习 5.1）
    count_shared_groups  有多少个组出现在两个及以上集合里
    average_hash     平均哈希（练习 5.2）
    hamming_matrix   两组哈希之间的汉明距离矩阵
    cross_duplicates 对集合 B 的每个样本，找它在集合 A 里最像的样本（练习 5.3）
    nearest_pixel_distance  另一种近重复度量：像素空间的最近邻距离

"来源"是什么？真实数据里，同一页论文截出来的几张公式图、同一个人手写的几个符号、同一次扫描的几张图片，
彼此长得很像：字体一样、倾斜一样、纸上的污点也一样。我们把这样一批图片叫作一个"来源组"。
make_dataset 用一组固定参数模拟一个来源，组内样本只在这组参数附近做很小的抖动，所以组内样本是近重复。

只依赖 numpy、torch 和 matplotlib（不需要安装 LaTeX）。
"""
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

# (类别名, mathtext)。分式横线用 \frac{\qquad}{\qquad} 渲染：只剩一条比减号长的横线。
SYMBOLS = [
    ("0", "0"), ("1", "1"), ("2", "2"), ("3", "3"), ("4", "4"),
    ("5", "5"), ("6", "6"), ("7", "7"), ("8", "8"), ("9", "9"),
    ("a", "a"), ("b", "b"), ("x", "x"), ("y", "y"),
    ("+", "+"), ("-", "-"), ("=", "="),
    ("alpha", r"\alpha"), ("beta", r"\beta"), ("pi", r"\pi"),
    ("sum", r"\sum"), ("int", r"\int"), ("infty", r"\infty"), ("frac_bar", r"\frac{\qquad}{\qquad}"),
]
CLASS_NAMES = [n for n, _ in SYMBOLS]
FONTSETS = ("dejavusans", "dejavuserif", "cm", "stix", "stixsans")
MASTER = 96     # 母版边长（像素）；输出时缩小 3 倍
SIZE = 32       # 输出图片边长（像素）


@lru_cache(maxsize=None)
def _render_cached(tex, fontset, fontsize, canvas):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    def draw(fs):                                  # 画在两倍大的画布上，避免高个子符号（如 \int）被切掉
        fig = Figure(figsize=(2, 2), dpi=canvas)
        cv = FigureCanvasAgg(fig)
        fig.text(0.5, 0.5, f"${tex}$", ha="center", va="center", fontsize=fs, math_fontfamily=fontset)
        cv.draw()
        return 1.0 - np.asarray(cv.buffer_rgba())[..., 0].astype(np.float32) / 255.0

    ink = draw(fontsize)
    ys, xs = np.nonzero(ink > 0.2)
    if len(ys) == 0:
        return np.zeros((canvas, canvas), np.float32)
    side = max(ys.max() - ys.min(), xs.max() - xs.min()) + 1
    if side > 0.85 * canvas:                       # 太大的符号按比例缩小字号重画（只影响少数字体里的 \int、\sum）
        ink = draw(fontsize * 0.85 * canvas / side)
        ys, xs = np.nonzero(ink > 0.2)
    cy, cx = (ys.min() + ys.max() + 1) // 2, (xs.min() + xs.max() + 1) // 2
    h = canvas // 2                                # 按墨迹的外接框把符号移到正中（除上面的防裁切缩小外，保留相对大小）
    return np.ascontiguousarray(ink[cy - h:cy + h, cx - h:cx + h])


def render_symbol(tex, fontset="dejavusans", fontsize=40, canvas=MASTER):
    """返回 (canvas, canvas) 的 float32 数组，墨迹为 1、纸为 0，符号的外接框居中。结果有缓存。
    通常保留符号之间的相对大小；外接框超过画布 85% 的符号会缩小字号重画，以免被裁切。"""
    return _render_cached(tex, fontset, fontsize, canvas).copy()


def augment(master, scale=1.0, angle=0.0, shift=(0.0, 0.0), thickness=0, contrast=1.0,
            background=0.0, noise=0.0, out=SIZE, rng=None):
    """对 (H, W) 母版做一次增广，返回 (out, out) 的 float32 数组，取值裁剪到 [0, 1]。

    scale      >1 放大符号            angle     逆时针旋转的角度（度）
    shift      (dx, dy)，输出像素      thickness 笔画粗细：+1 加粗，-1 变细，0 不变
    contrast   墨迹的深浅（乘在墨迹上） background 纸的灰度（加在整张图上）
    noise      高斯噪声的标准差，需要 rng（np.random.Generator）
    """
    x = torch.as_tensor(master, dtype=torch.float32)[None, None]          # (1, 1, H, W)
    if thickness > 0:                                                       # 加粗：最大值滤波 = 膨胀
        x = F.max_pool2d(x, 2 * thickness + 1, stride=1, padding=thickness)
    elif thickness < 0:                                                     # 变细：最小值滤波 = 腐蚀
        k = -thickness
        x = -F.max_pool2d(-x, 2 * k + 1, stride=1, padding=k)
    H = x.shape[-1]
    a = np.deg2rad(angle)                                                  # 正角度 = 画面上逆时针
    c, s = np.cos(a) / scale, np.sin(a) / scale
    # affine_grid 的 theta 把"输出坐标"映射到"输入坐标"，坐标归一化到 [-1, 1]，所以平移要除以半边长
    tx, ty = -2.0 * shift[0] / out, -2.0 * shift[1] / out
    theta = torch.tensor([[c, -s, c * tx - s * ty], [s, c, s * tx + c * ty]], dtype=torch.float32)[None]
    grid = F.affine_grid(theta, (1, 1, H, H), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    x = F.adaptive_avg_pool2d(x, out)[0, 0].numpy()                         # 先在大图上变换，再平均池化缩小
    x = background + contrast * x
    if noise > 0:
        rng = rng if rng is not None else np.random.default_rng()
        x = x + rng.normal(0.0, noise, x.shape)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _specks(rng, n, out=SIZE):
    """一个来源固有的扫描污点：n 个小斑点或短划痕，位置和深浅随机。返回 (out, out)。"""
    m = np.zeros((out, out), np.float32)
    for _ in range(n):
        y, x = rng.integers(0, out, size=2)
        v = rng.uniform(0.3, 0.8)
        if rng.random() < 0.5:                                  # 斑点：2×2
            m[y:y + 2, x:x + 2] = np.maximum(m[y:y + 2, x:x + 2], v)
        else:                                                   # 划痕：横向或纵向 3–6 像素
            L = rng.integers(3, 7)
            if rng.random() < 0.5:
                m[y, x:x + L] = np.maximum(m[y, x:x + L], v)
            else:
                m[y:y + L, x] = np.maximum(m[y:y + L, x], v)
    return m


def source_params(rng):
    """随机抽一个来源的固定参数：字体、姿态、粗细、对比度、纸色和污点。"""
    return dict(
        fontset=FONTSETS[rng.integers(len(FONTSETS))],
        scale=rng.uniform(0.8, 1.15), angle=rng.uniform(-12, 12),
        shift=(rng.uniform(-3, 3), rng.uniform(-3, 3)),
        thickness=int(rng.choice([0, 1, 2])),
        contrast=rng.uniform(0.6, 1.0), background=rng.uniform(0.0, 0.15),
        specks=_specks(rng, int(rng.integers(1, 5))),
    )


def sample_from_source(cls, p, rng, jitter=1.0, noise=0.04):
    """从一个来源里取一张图：在来源参数附近做很小的抖动，再叠加来源的污点和新的噪声。
    jitter=0 时同一来源的图只差噪声；jitter 越大，组内样本越不像。"""
    master = render_symbol(SYMBOLS[cls][1], p["fontset"])
    img = augment(master,
                  scale=p["scale"] * (1 + 0.03 * jitter * rng.uniform(-1, 1)),
                  angle=p["angle"] + 2.0 * jitter * rng.uniform(-1, 1),
                  shift=(p["shift"][0] + 0.7 * jitter * rng.uniform(-1, 1),
                         p["shift"][1] + 0.7 * jitter * rng.uniform(-1, 1)),
                  thickness=p["thickness"], contrast=p["contrast"], background=p["background"])
    img = np.maximum(img, p["specks"])
    img = img + rng.normal(0.0, noise, img.shape)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def make_dataset(sources_per_class=20, per_source=8, classes=None, seed=0, jitter=1.0, noise=0.04,
                 group_offset=0):
    """生成带来源组标记的数据集。

    每个类别有 sources_per_class 个来源，每个来源出 per_source 张近重复的图。
    返回 dict：
        images (N, 32, 32) float32，labels (N,) int64，groups (N,) int64（来源编号，全局唯一），
        fontsets (N,) 字体名，class_names。
    group_offset 让不同调用生成的来源编号不重叠（例如另外生成一批"全新来源"当外部测试集）。
    """
    rng = np.random.default_rng(seed)
    classes = range(len(SYMBOLS)) if classes is None else classes
    imgs, labels, groups, fonts = [], [], [], []
    g = group_offset
    for c in classes:
        for _ in range(sources_per_class):
            p = source_params(rng)
            for _ in range(per_source):
                imgs.append(sample_from_source(c, p, rng, jitter=jitter, noise=noise))
                labels.append(c)
                groups.append(g)
                fonts.append(p["fontset"])
            g += 1
    return dict(images=np.stack(imgs), labels=np.array(labels, np.int64), groups=np.array(groups, np.int64),
                fontsets=np.array(fonts), class_names=CLASS_NAMES)


def _cut(n, fractions):
    """把 n 个东西按比例切成几段，返回切点。最后一段拿走余数。"""
    fr = np.asarray(fractions, dtype=float)
    fr = fr / fr.sum()
    return np.round(np.cumsum(fr)[:-1] * n).astype(int)


def random_split(n, fractions=(0.8, 0.1, 0.1), seed=0):
    """按样本随机划分：打乱 0..n-1，按比例切开。返回各集合的下标数组（升序）。
    同一来源的近重复样本会被拆到不同集合里——这就是泄漏。"""
    perm = np.random.default_rng(seed).permutation(n)
    return tuple(np.sort(part) for part in np.split(perm, _cut(n, fractions)))


def count_shared_groups(groups, *parts):
    """parts 是几组样本下标（例如 train / val / test）。返回出现在两个及以上集合里的组的个数。
    同时出现在三个集合里的组只算一次（先对两两交集取并集，再计数）。"""
    groups = np.asarray(groups)
    sets = [set(groups[p].tolist()) for p in parts]
    shared = set()
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            shared |= sets[i] & sets[j]
    return len(shared)


# >>> 练习 5.1：分组划分
def group_split(groups, fractions=(0.8, 0.1, 0.1), seed=0):
    """按组划分：同一个组的样本一定落在同一个集合里。

    groups    (N,) 每个样本的组编号（例如来源编号）
    fractions 各集合分到的"组数"比例（不是样本数）；会先归一化
    返回各集合的样本下标数组（升序），元组长度等于 len(fractions)。

    做法：取出所有不同的组编号并排序，用 seed 打乱，按比例把组切成几段，
    再把每段组里的所有样本下标收集起来。
    """
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(uniq)
    parts = np.split(perm, _cut(len(uniq), fractions))
    return tuple(np.sort(np.nonzero(np.isin(groups, part))[0]) for part in parts)
# <<<


# >>> 练习 5.2：平均哈希
def average_hash(images, hash_size=8):
    """平均哈希（aHash）：把每张图缩成 hash_size×hash_size 的小图（分块求平均），
    每个格子比这张小图的平均值亮就记 True，否则记 False。

    images  (N, H, W) 或 (H, W) 的数组或张量
    返回    (N, hash_size*hash_size) 的 bool numpy 数组（输入是单张图时 N = 1）
    """
    x = torch.as_tensor(np.ascontiguousarray(images), dtype=torch.float32)   # 翻转过的数组步长为负，先复制
    if x.dim() == 2:
        x = x[None]
    small = F.adaptive_avg_pool2d(x[:, None], hash_size)[:, 0].reshape(len(x), -1)    # (N, hash_size²)
    return (small > small.mean(dim=1, keepdim=True)).numpy()
# <<<


def hamming_matrix(ha, hb):
    """(Na, B) 和 (Nb, B) 的 bool 哈希 → (Na, Nb) 的汉明距离（有几位不同）。"""
    a = torch.as_tensor(ha, dtype=torch.float32)
    b = torch.as_tensor(hb, dtype=torch.float32)
    # 不同的位数 = a 为 1 且 b 为 0 的位数 + a 为 0 且 b 为 1 的位数
    return (a @ (1 - b).T + (1 - a) @ b.T).round().long().numpy()


# >>> 练习 5.3：跨集合近重复
def cross_duplicates(hash_a, hash_b, max_dist=2):
    """对集合 B 的每个样本，在集合 A 里找汉明距离最小的样本。

    返回 (nearest, dist, is_dup)，都是长度 Nb 的数组：
        nearest[j]  A 中离 B[j] 最近的样本下标
        dist[j]     对应的汉明距离
        is_dup[j]   dist[j] <= max_dist，即 B[j] 在 A 里有近重复
    """
    d = hamming_matrix(hash_a, hash_b)               # (Na, Nb)
    nearest = d.argmin(axis=0)
    dist = d[nearest, np.arange(d.shape[1])]
    return nearest, dist, dist <= max_dist
# <<<


def nearest_pixel_distance(a, b):
    """另一种近重复度量：对 b 的每张图，求它到 a 中最近一张图的像素距离（均方根，单位是灰度）。
    返回 (nearest, dist)。比哈希更精细，但对平移、亮度变化更敏感，计算量也是 Na×Nb×像素数。"""
    A = torch.as_tensor(np.asarray(a), dtype=torch.float32).reshape(len(a), -1)
    B = torch.as_tensor(np.asarray(b), dtype=torch.float32).reshape(len(b), -1)
    d = torch.cdist(B, A) / A.shape[1] ** 0.5            # (Nb, Na)
    dist, nearest = d.min(dim=1)
    return nearest.numpy(), dist.numpy()
