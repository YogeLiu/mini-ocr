"""第 5 章　泛化与实验方法：配套实验脚本。正文里引用的每一个数字都来自这里。

    python -m book.part1_foundations.ch05_generalization.tour            # 全部 7 个部分（CPU 约 2 分钟，随机器负载波动）
    python -m book.part1_foundations.ch05_generalization.tour 3 4        # 只运行第 3、4 部分
    python -m book.part1_foundations.ch05_generalization.tour 4 --device mps   # 训练放到 Mac 上

部分与正文小节的对应：
    1  符号数据：渲染与类别（5.2）            5  学习曲线：欠拟合与过拟合（5.6）
    2  增广与来源组：组内有多像（5.2、5.4）    6  多个种子：一次结果为什么不可信（5.7）
    3  划分与近重复检测（5.4、5.5）           7  评测协议：测试集只报告、不挑选（5.1、5.8）
    4  泄漏实验：随机划分 vs 分组划分（5.3）

第 4–7 部分要训练小 CNN，每次几秒钟。数据由 miniocr.data.symbols 现场生成，不需要下载。
"""
import os
import sys
import time

# 除非明确要求 --device cuda，否则对 PyTorch 隐藏 GPU（本机的 GPU 正被别的服务占用）。
if "cuda" not in sys.argv:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from miniocr.data.symbols import (CLASS_NAMES, FONTSETS, SYMBOLS, augment, average_hash, count_shared_groups,
                                  cross_duplicates, group_split, hamming_matrix, make_dataset,
                                  nearest_pixel_distance, random_split, render_symbol, sample_from_source,
                                  source_params)
from miniocr.device import device_from_argv

OUT = "outputs/book/ch05"
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(min(8, os.cpu_count() or 1))
DEVICE = device_from_argv(default="cpu")          # 只有训练用它

# 全章共用的数据配置：24 类 × 每类 100 个来源 × 每个来源 4 张 = 9600 张
SOURCES, PER_SOURCE, DATA_SEED = 100, 4, 0
EXT_SOURCES, EXT_PER_SOURCE, EXT_SEED = 10, 8, 123      # "全新来源"：模拟上线后遇到的数据
HASH_MAX_DIST = 2                                        # 近重复的汉明距离门槛（第 2、3 部分据此选定）
PIX_MAX_DIST = 0.09                                      # 近重复的像素距离门槛（均方根，灰度单位）


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


_CACHE = {}


def data():
    """主数据集（带来源组）和外部数据集（全新来源，编号不和主数据集重叠）。结果缓存。"""
    if "d" not in _CACHE:
        _CACHE["d"] = make_dataset(SOURCES, PER_SOURCE, seed=DATA_SEED)
        _CACHE["ext"] = make_dataset(EXT_SOURCES, EXT_PER_SOURCE, seed=EXT_SEED, group_offset=10 ** 6)
    return _CACHE["d"], _CACHE["ext"]


# ---------------------------------------------------------------------- 模型与训练循环（第 4–7 部分共用）
class SymbolCNN(nn.Module):
    """两层卷积 + 一层全连接。输入 (B, 32, 32)，输出 (B, 24) 的 logits。
    width=16 时参数量 = (1·9+1)·16 + (16·9+1)·32 + (32·8·8+1)·24 = 160 + 4640 + 49176 = 53976。"""

    def __init__(self, width=16, n_classes=len(SYMBOLS)):
        super().__init__()
        self.conv1 = nn.Conv2d(1, width, 3, padding=1)            # (B,1,32,32) → (B,w,32,32)
        self.conv2 = nn.Conv2d(width, 2 * width, 3, padding=1)    # (B,w,16,16) → (B,2w,16,16)
        self.fc = nn.Linear(2 * width * 8 * 8, n_classes)          # (B,2w·8·8) → (B,24)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x[:, None])), 2)       # (B,w,16,16)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)                # (B,2w,8,8)
        return self.fc(x.flatten(1))


class Linear(nn.Module):
    """softmax 回归（第 3 章）：把 32×32 的像素直接线性映射到 24 个 logits。参数量 1024·24 + 24 = 24600。"""

    def __init__(self, n_classes=len(SYMBOLS)):
        super().__init__()
        self.fc = nn.Linear(32 * 32, n_classes)

    def forward(self, x):
        return self.fc(x.flatten(1))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def jitter_batch(x, gen, max_shift=2.0, max_angle=8.0):
    """训练时的在线增广：每张图随机平移 ±max_shift 像素、旋转 ±max_angle 度。x: (B, 32, 32)。"""
    B, H, _ = x.shape
    a = (torch.rand(B, generator=gen) * 2 - 1) * np.deg2rad(max_angle)
    t = (torch.rand(B, 2, generator=gen) * 2 - 1) * (2 * max_shift / H)
    theta = torch.stack([torch.stack([a.cos(), -a.sin(), t[:, 0]], 1),
                         torch.stack([a.sin(), a.cos(), t[:, 1]], 1)], 1).to(x.device)
    grid = F.affine_grid(theta, (B, 1, H, H), align_corners=False)
    return F.grid_sample(x[:, None], grid, align_corners=False, padding_mode="border")[:, 0]


@torch.no_grad()
def evaluate(model, X, y, bs=2048):
    """返回 (平均交叉熵, 准确率)。会切到 eval 模式（本章的模型没有 Dropout，但养成习惯）。"""
    model.eval()
    loss, correct = 0.0, 0
    for i in range(0, len(X), bs):
        xb, yb = X[i:i + bs].to(DEVICE), y[i:i + bs].to(DEVICE)
        logits = model(xb)
        loss += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(1) == yb).sum().item()
    return loss / len(X), correct / len(X)


def fit(Xtr, ytr, Xva=None, yva=None, *, model="cnn", width=16, steps=2000, lr=2e-3, weight_decay=0.0,
        bs=64, seed=0, aug=False, eval_every=0, idx_log=None):
    """本章的训练循环：取一批 → 前向 → 交叉熵 → 反向 → AdamW 更新。

    seed 同时决定模型初始化、取批顺序和增广。取批和增广各用一个生成器，所以开不开增广，取到的批次都一样。
    eval_every > 0 时每隔这么多步在整个训练集和验证集上各评一次，记进 history（没给验证集时验证指标记 nan）。
    idx_log 是一个列表时，把每一步取批的下标追加进去（用来检查取批顺序）。返回 (model, history)。
    """
    torch.manual_seed(seed)                                           # ① 固定初始化
    gen = torch.Generator().manual_seed(seed)                         # ② 固定取批顺序
    aug_gen = torch.Generator().manual_seed(seed + 10_000)            # ③ 增广单独一个生成器，不影响取批
    net = (SymbolCNN(width) if model == "cnn" else Linear()).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    hist = []
    for step in range(1, steps + 1):
        net.train()
        idx = torch.randint(len(Xtr), (bs,), generator=gen)
        if idx_log is not None:
            idx_log.append(idx)
        xb, yb = Xtr[idx], ytr[idx]
        if aug:
            xb = jitter_batch(xb, aug_gen)
        loss = F.cross_entropy(net(xb.to(DEVICE)), yb.to(DEVICE))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if eval_every and (step % eval_every == 0 or step == 1):
            tr = evaluate(net, Xtr, ytr)                              # 整个训练集，覆盖全部类别
            va = evaluate(net, Xva, yva) if Xva is not None else (float("nan"), float("nan"))
            hist.append(dict(step=step, train_loss=tr[0], train_acc=tr[1], val_loss=va[0], val_acc=va[1]))
    return net, hist


def tensors(d, idx=None):
    X, y = torch.tensor(d["images"]), torch.tensor(d["labels"])
    return (X, y) if idx is None else (X[idx], y[idx])


# ---------------------------------------------------------------------- 1
def part1_symbols():
    section("1. 符号数据（5.2）：用 matplotlib mathtext 渲染单个公式符号")
    print(f"  类别数 {len(SYMBOLS)}：" + " ".join(CLASS_NAMES))
    print(f"  字体 {len(FONTSETS)} 套：" + "、".join(FONTSETS))
    m = render_symbol(r"\alpha", "cm")
    print(f"  母版：render_symbol(r'\\alpha', 'cm') 形状 {m.shape}，取值 [{m.min():.2f}, {m.max():.2f}]，墨迹像素占比 {(m > 0.5).mean():.3f}")
    img = augment(m)
    print(f"  不做增广直接缩小：augment(母版) 形状 {img.shape}，dtype {img.dtype}")
    fig, axes = plt.subplots(len(FONTSETS), len(SYMBOLS), figsize=(len(SYMBOLS) * 0.6, len(FONTSETS) * 0.7))
    for r, fs in enumerate(FONTSETS):
        for c, (name, tex) in enumerate(SYMBOLS):
            ax = axes[r, c]
            ax.imshow(augment(render_symbol(tex, fs)), cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=6)
            if c == 0:
                ax.set_ylabel(fs, fontsize=6)
    fig.tight_layout()
    fig.savefig(f"{OUT}/symbols_gallery.png", dpi=150)
    plt.close(fig)
    print(f"  图：{OUT}/symbols_gallery.png（5 套字体 × 24 个符号，未增广）")
    for a, b in (("-", "frac_bar"), ("0", "pi")):
        ia = augment(render_symbol(dict(SYMBOLS)[a]))
        ib = augment(render_symbol(dict(SYMBOLS)[b]))
        wa = (ia.max(0) > 0.5).sum()
        wb = (ib.max(0) > 0.5).sum()
        print(f"  {a!r} 的墨迹宽 {wa} 像素，{b!r} 宽 {wb} 像素（dejavusans，32×32）")


# ---------------------------------------------------------------------- 2
def part2_sources():
    section("2. 增广与来源组（5.2、5.4）：同一来源的图有多像")
    rng = np.random.default_rng(0)
    m = render_symbol("7")
    steps = [("original", {}), ("scale 1.15", dict(scale=1.15)), ("angle 12", dict(angle=12)),
             ("shift (3,-2)", dict(shift=(3, -2))), ("thickness +2", dict(thickness=2)),
             ("contrast 0.6", dict(contrast=0.6)), ("background 0.15", dict(background=0.15)),
             ("noise 0.04", dict(noise=0.04, rng=rng))]
    fig, axes = plt.subplots(1, len(steps), figsize=(len(steps) * 1.3, 1.6))
    for ax, (t, kw) in zip(axes, steps):
        ax.imshow(augment(m, **kw), cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(t, fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{OUT}/augment_steps.png", dpi=150)
    plt.close(fig)
    print(f"  图：{OUT}/augment_steps.png（每种增广单独作用在符号 7 上）")

    # 来源组示意：符号 alpha 的 5 个来源，每个来源 6 张
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(5, 6, figsize=(6, 5))
    for r in range(5):
        p = source_params(rng)
        for c in range(6):
            axes[r, c].imshow(sample_from_source(17, p, rng), cmap="gray_r", vmin=0, vmax=1)
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(f"{OUT}/source_groups.png", dpi=150)
    plt.close(fig)
    print(f"  图：{OUT}/source_groups.png（alpha 的 5 个来源，每行一个来源、6 张图）")

    d, _ = data()
    X, y, g = d["images"], d["labels"], d["groups"]
    print(f"  主数据集：{len(SYMBOLS)} 类 × {SOURCES} 个来源 × {PER_SOURCE} 张 = {len(X)} 张，形状 {X.shape}，"
          f"来源组 {len(np.unique(g))} 个")
    # 三种样本对的距离：同一来源 / 同类不同来源 / 不同类
    rs = np.random.default_rng(0)
    i = rs.integers(0, len(X), 4000)
    j_same = np.array([rs.choice(np.nonzero(g == g[k])[0]) for k in i])
    j_same = np.where(j_same == i, (i // PER_SOURCE) * PER_SOURCE + (i + 1) % PER_SOURCE, j_same)
    j_cls = np.array([rs.choice(np.nonzero((y == y[k]) & (g != g[k]))[0]) for k in i])
    j_other = np.array([rs.choice(np.nonzero(y != y[k])[0]) for k in i])
    H = average_hash(X)
    print("  4000 对随机样本对，像素距离（均方根，灰度单位）和平均哈希（64 位）的汉明距离：")
    print(f"    {'样本对':14s}{'像素距离中位数':>12s}{'哈希距离中位数':>12s}{'像素 ≤ ' + str(PIX_MAX_DIST):>14s}{'哈希 ≤ ' + str(HASH_MAX_DIST):>12s}")
    for name, j in (("同一来源", j_same), ("同类、不同来源", j_cls), ("不同类", j_other)):
        pix = np.sqrt(((X[i] - X[j]) ** 2).mean((1, 2)))
        ham = (H[i] != H[j]).sum(1)
        print(f"    {name:14s}{np.median(pix):12.3f}{np.median(ham):12.0f}{(pix <= PIX_MAX_DIST).mean():14.1%}"
              f"{(ham <= HASH_MAX_DIST).mean():12.1%}")
    for th in (1, 2, 4, 6):
        same = ((H[i] != H[j_same]).sum(1) <= th).mean()
        cls = ((H[i] != H[j_cls]).sum(1) <= th).mean()
        print(f"    哈希门槛 {th}：同一来源判为近重复 {same:6.1%}；同类不同来源被误判 {cls:5.1%}")
    print("  注意：这是\"一对一\"比较的误判率。第 3 部分在几千张图里找最近邻，误判会多得多")


# ---------------------------------------------------------------------- 3
def part3_splits():
    section("3. 划分与近重复检测（5.4、5.5）：随机划分会把同一来源拆开")
    d, _ = data()
    g = d["groups"]
    H = average_hash(d["images"])
    X = d["images"]
    for name, (tr, va, te) in (("随机划分", random_split(len(g), (0.8, 0.1, 0.1), seed=0)),
                               ("分组划分", group_split(g, (0.8, 0.1, 0.1), seed=0))):
        shared = count_shared_groups(g, tr, va, te)
        triple = len(set(g[tr]) & set(g[va]) & set(g[te]))
        leak = np.isin(g[va], g[tr])
        print(f"  {name}：train {len(tr)} / val {len(va)} / test {len(te)} 张；出现在两个及以上集合的来源 {shared} 个"
              f"（其中同时在三个集合的 {triple} 个）；验证集里来源也在训练集的样本 {leak.mean():.1%}")
        nearest, dist, dup = cross_duplicates(H[tr], H[va], max_dist=HASH_MAX_DIST)
        pn, pdist = nearest_pixel_distance(X[tr], X[va])
        pdup = pdist <= PIX_MAX_DIST
        print(f"    最近邻距离中位数：哈希 {np.median(dist):.0f} 位，像素 {np.median(pdist):.3f}")
        for det, flag, nn_idx in (("哈希 ≤ " + str(HASH_MAX_DIST), dup, nearest), ("像素 ≤ " + str(PIX_MAX_DIST), pdup, pn)):
            same = (g[tr][nn_idx] == g[va])
            line = f"    {det:10s}：验证集被标为近重复 {flag.mean():6.1%}"
            if flag.any():
                line += f"；标出来的里面最近邻确实同源 {same[flag].mean():6.1%}"
            if leak.any():
                line += f"；同源泄漏样本被抓到 {flag[leak].mean():6.1%}"
            print(line)
    tr, va, te = group_split(g, (0.8, 0.1, 0.1), seed=0)
    print(f"  分组划分按\"组数\"切：{len(np.unique(g))} 个来源 → {len(np.unique(g[tr]))} / {len(np.unique(g[va]))} / "
          f"{len(np.unique(g[te]))} 个")
    cls_counts = np.bincount(d["labels"][va], minlength=len(SYMBOLS))
    print(f"  分组划分后验证集每类样本数：最少 {cls_counts.min()}，最多 {cls_counts.max()}（没有按类分层，所以不完全均衡）")


# ---------------------------------------------------------------------- 4
LEAK_STEPS, LEAK_SEEDS = 3000, (0, 1, 2)


def part4_leakage():
    section("4. 泄漏实验（5.3）：随机划分 vs 分组划分，验证集分数虚高多少")
    d, ext = data()
    X, y = tensors(d)
    Xe, ye = tensors(ext)
    print(f"  训练集/验证集按 8:2 切；另有 {len(Xe)} 张\"全新来源\"的图（{EXT_SOURCES} 个来源 × 24 类 × {EXT_PER_SOURCE} 张），"
          f"模拟上线后遇到的数据")
    print(f"  模型 SymbolCNN(width=16)，参数 {n_params(SymbolCNN(16)):,}；AdamW lr=2e-3，batch 64，{LEAK_STEPS} 步")
    rows = []
    t0 = time.time()
    for seed in LEAK_SEEDS:
        for name, (tr, va) in (("随机划分", random_split(len(X), (0.8, 0.2), seed=seed)),
                               ("分组划分", group_split(d["groups"], (0.8, 0.2), seed=seed))):
            net, _ = fit(X[tr], y[tr], steps=LEAK_STEPS, seed=seed)
            _, acc_tr = evaluate(net, X[tr], y[tr])
            _, acc_va = evaluate(net, X[va], y[va])
            _, acc_ext = evaluate(net, Xe, ye)
            rows.append((name, seed, acc_tr, acc_va, acc_ext))
            print(f"    种子 {seed} {name}：训练 {acc_tr:.3f}  验证 {acc_va:.3f}  全新来源 {acc_ext:.3f}  "
                  f"验证 − 全新 = {acc_va - acc_ext:+.3f}")
    print(f"  （{len(rows)} 次训练用时 {time.time() - t0:.0f} 秒）")
    for name in ("随机划分", "分组划分"):
        r = np.array([x[2:] for x in rows if x[0] == name])
        gap = r[:, 1] - r[:, 2]
        print(f"  {name} 3 个种子平均：验证 {r[:, 1].mean():.3f}，全新来源 {r[:, 2].mean():.3f}，"
              f"差 {gap.mean():+.3f}（范围 {gap.min():+.3f} 到 {gap.max():+.3f}）")


# ---------------------------------------------------------------------- 5
def part5_curves():
    section("5. 学习曲线（5.6）：欠拟合、过拟合和各自的处理")
    d, _ = data()
    X, y = tensors(d)
    tr, va, _ = group_split(d["groups"], (0.8, 0.1, 0.1), seed=0)
    small = tr[np.isin(d["groups"][tr], np.unique(d["groups"][tr])[::8])]     # 只留八分之一的来源
    print(f"  完整训练集 {len(tr)} 张；小训练集 {len(small)} 张（只取 1/8 的来源）；验证集 {len(va)} 张；全部分组划分")
    runs = [
        ("A", "underfit", "A 欠拟合：线性模型，lr=1e-4", dict(model="linear", lr=1e-4), tr),
        ("A2", "linear, lr 2e-3", "A2 处理欠拟合（训练得更充分）：A 把 lr 调到 2e-3", dict(model="linear"), tr),
        ("B", "overfit", "B 过拟合：CNN，小训练集", dict(), small),
        ("C", "B + augmentation", "C 处理过拟合（增广）：B + 在线增广", dict(aug=True), small),
        ("D", "B + 8x data", "D 处理过拟合（更多数据）：B 换成完整训练集", dict(), tr),
    ]
    curves = {}
    for key, title, name, kw, idx in runs:
        net, h = fit(X[idx], y[idx], X[va], y[va], seed=0, steps=3000, eval_every=100, **kw)
        curves[key] = (title, h)
        best = min(h, key=lambda r: r["val_loss"])
        last = h[-1]
        print(f"  {name}（参数 {n_params(net):,}）")
        print(f"    第 3000 步：训练 loss {last['train_loss']:.3f} 准确率 {last['train_acc']:.3f} | "
              f"验证 loss {last['val_loss']:.3f} 准确率 {last['val_acc']:.3f} | 差距（验证 − 训练 loss）{last['val_loss'] - last['train_loss']:.3f}")
        print(f"    验证 loss 最低在第 {best['step']} 步：{best['val_loss']:.3f}（这一步的验证准确率 {best['val_acc']:.3f}）；"
              f"验证准确率最高 {max(r['val_acc'] for r in h):.3f}")
    fig, axes = plt.subplots(1, len(curves), figsize=(3.2 * len(curves), 3.2), sharey=True)
    for ax, (key, (title, h)) in zip(axes, curves.items()):
        s = [r["step"] for r in h]
        ax.plot(s, [r["train_loss"] for r in h], label="train")
        ax.plot(s, [r["val_loss"] for r in h], label="val")
        ax.set_title(f"{key}: {title}", fontsize=9)
        ax.set_xlabel("step")
        ax.set_ylim(0, 4)
    axes[0].set_ylabel("cross-entropy")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/learning_curves.png", dpi=150)
    plt.close(fig)
    print(f"  图：{OUT}/learning_curves.png（A、A2、B、C、D 五条学习曲线；纵轴截在 4，B 的验证 loss 后段超出画面）")


# ---------------------------------------------------------------------- 6
SEED_STEPS, SEEDS = 1500, (0, 1, 2, 3, 4, 5, 6, 7)


def seed_runs():
    """配置 A（width=16）和配置 B（width=24）各跑 8 个种子，只改宽度这一个变量。结果缓存，第 7 部分复用。"""
    if "runs" in _CACHE:
        return _CACHE["runs"]
    d, ext = data()
    X, y = tensors(d)
    Xe, ye = tensors(ext)
    tr, va, te = group_split(d["groups"], (0.8, 0.1, 0.1), seed=0)
    runs = []
    for width in (16, 24):
        for seed in SEEDS:
            net, _ = fit(X[tr], y[tr], steps=SEED_STEPS, width=width, seed=seed)
            r = dict(width=width, seed=seed)
            for name, (xs, ys) in (("val", (X[va], y[va])), ("test", (X[te], y[te])), ("ext", (Xe, ye))):
                net.eval()
                with torch.no_grad():
                    r[name + "_ok"] = (net(xs.to(DEVICE)).argmax(1).cpu() == ys).numpy()   # 每张图对不对
                r[name] = r[name + "_ok"].mean()
            runs.append(r)
    _CACHE["runs"] = runs
    return runs


def part6_seeds():
    section("6. 多个种子（5.7）：一次实验的结果为什么不可信")
    d, _ = data()
    X, y = tensors(d)
    tr, va, _ = group_split(d["groups"], (0.8, 0.1, 0.1), seed=0)
    a, _ = fit(X[tr], y[tr], steps=200, seed=7)
    b, _ = fit(X[tr], y[tr], steps=200, seed=7)
    same = all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    print(f"  可复现检查：同一个种子训练两次（200 步），参数逐个完全相同：{same}")
    c, _ = fit(X[tr], y[tr], steps=200, seed=8)
    print(f"  换一个种子：验证准确率 {evaluate(a, X[va], y[va])[1]:.3f} → {evaluate(c, X[va], y[va])[1]:.3f}")
    log0, log1 = [], []
    fit(X[tr], y[tr], steps=50, seed=7, aug=False, idx_log=log0)
    fit(X[tr], y[tr], steps=50, seed=7, aug=True, idx_log=log1)
    print(f"  取批检查：同一个种子，开和不开增广，50 步取到的批次下标完全相同："
          f"{all(torch.equal(p, q) for p, q in zip(log0, log1))}")
    t0 = time.time()
    runs = seed_runs()
    print(f"  对照实验：只改卷积宽度（A: width=16，参数 {n_params(SymbolCNN(16)):,}；B: width=24，参数 "
          f"{n_params(SymbolCNN(24)):,}），其余全部相同，{SEED_STEPS} 步，各 {len(SEEDS)} 个种子"
          f"（用时 {time.time() - t0:.0f} 秒）")
    va_a = np.array([r["val"] for r in runs if r["width"] == 16])
    va_b = np.array([r["val"] for r in runs if r["width"] == 24])
    print("    种子      " + "".join(f"{s:>7d}" for s in SEEDS))
    print("    A 验证    " + "".join(f"{v:7.3f}" for v in va_a))
    print("    B 验证    " + "".join(f"{v:7.3f}" for v in va_b))
    print("    B − A     " + "".join(f"{v:+7.3f}" for v in va_b - va_a))
    for name, v in (("A", va_a), ("B", va_b)):
        print(f"    {name}：平均 {v.mean():.3f}，标准差 {v.std(ddof=1):.3f}，最低 {v.min():.3f}，最高 {v.max():.3f}，"
              f"极差 {v.max() - v.min():.3f}")
    diff = va_b - va_a
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    print(f"    配对差 B − A：平均 {diff.mean():+.3f}，标准误 {se:.3f}，约 95% 区间 "
          f"[{diff.mean() - 2.365 * se:+.3f}, {diff.mean() + 2.365 * se:+.3f}]（t 分布，自由度 {len(diff) - 1}）")
    r = np.corrcoef(va_a, va_b)[0, 1]
    se_unpaired = np.sqrt((va_a.var(ddof=1) + va_b.var(ddof=1)) / len(diff))
    print(f"    A、B 跨种子的相关系数 {r:+.3f}；忽略配对（当成两组独立样本）的标准误 {se_unpaired:.4f}，"
          f"配对差的标准误 {se:.4f}")
    print(f"    {len(SEEDS)} 个种子里 B 赢 {int((diff > 0).sum())} 次、输 {int((diff < 0).sum())} 次、平 {int((diff == 0).sum())} 次")
    worst = (va_b.min() - va_a.max())
    best = (va_b.max() - va_a.min())
    print(f"    如果各只跑一次、而且运气不同：B − A 可能小到 {worst:+.3f}，也可能大到 {best:+.3f}")
    n_va = len(va)
    p = va_a.mean()
    print(f"    验证集 {n_va} 张时，准确率 {p:.3f} 的抽样标准误 √(p(1−p)/n) = {np.sqrt(p * (1 - p) / n_va):.4f}"
          f"（公式假设每张图独立，忽略了同一来源的相关性）")
    # 按来源做 cluster bootstrap：每次有放回地抽 240 个来源，同一来源的图一起抽
    g_va = d["groups"][va]
    uniq, inv = np.unique(g_va, return_inverse=True)
    rb = np.random.default_rng(0)
    for r0 in [r for r in runs if r["width"] == 16 and r["seed"] == 0]:
        ok_g = np.bincount(inv, weights=r0["val_ok"].astype(float))           # 每个来源答对几张
        n_g = np.bincount(inv)                                                 # 每个来源几张
        boots = []
        for _ in range(2000):
            pick = rb.integers(0, len(uniq), len(uniq))
            boots.append(ok_g[pick].sum() / n_g[pick].sum())
        print(f"    按来源 cluster bootstrap（A 种子 0，{len(uniq)} 个来源，2000 次）：准确率 {r0['val']:.3f}，"
              f"标准误 {np.std(boots, ddof=1):.4f}")


# ---------------------------------------------------------------------- 7
def part7_protocol():
    section("7. 评测协议（5.1、5.8）：拿测试集做决定，分数就会偏乐观")
    runs = seed_runs()
    val = np.array([r["val"] for r in runs])
    test = np.array([r["test"] for r in runs])
    ext = np.array([r["ext"] for r in runs])
    print(f"  第 6 部分的 {len(runs)} 个模型（2 种宽度 × {len(SEEDS)} 个种子），每个都在三份数据上评过：")
    print(f"    验证集（分组划分的 10%）、测试集（另外 10%）、全新来源（另一批来源，当作\"新补的留出测试集\"）")
    print(f"    全部模型的平均：验证 {val.mean():.3f}，测试 {test.mean():.3f}，全新来源 {ext.mean():.3f}")
    i = int(val.argmax())
    print(f"  按验证集选模型 → width={runs[i]['width']}，种子 {runs[i]['seed']}：验证 {val[i]:.3f}，测试 {test[i]:.3f}，"
          f"全新来源 {ext[i]:.3f}")
    print(f"  同一批模型，验证集和测试集的平均分差了 {val.mean() - test.mean():+.3f}：两份数据各只有 240 个来源，"
          f"抽到的来源难度不同（划分本身也有随机性）")

    # 赢家诅咒：用一份"选择集"挑出最好的模型，再看它在没参与挑选的数据上的分数
    ok = np.stack([np.concatenate([r["test_ok"], r["ext_ok"]]) for r in runs])      # (16, 2880)
    rng = np.random.default_rng(0)
    print(f"  模拟\"拿测试集做决定\"：把测试集和全新来源合成 {ok.shape[1]} 张的池子，反复 500 次：")
    print("    随机抽 n 张当\"测试集\"，按它的分数从 16 个模型里挑最好的，再用池子里其余的图给这个模型打分")
    print(f"    {'n':>6s}{'挑出的模型在选择集上':>14s}{'在其余图上':>10s}{'虚高':>10s}{'随便挑一个的虚高':>14s}")
    for n in (100, 300, 960):
        sel, rest, rnd = [], [], []
        for _ in range(500):
            perm = rng.permutation(ok.shape[1])
            a, b = perm[:n], perm[n:]
            sa, sb = ok[:, a].mean(1), ok[:, b].mean(1)
            k = int(sa.argmax())
            sel.append(sa[k]); rest.append(sb[k])
            k2 = rng.integers(len(runs))
            rnd.append(sa[k2] - sb[k2])
        sel, rest = np.array(sel), np.array(rest)
        print(f"    {n:6d}{sel.mean():20.3f}{rest.mean():14.3f}{(sel - rest).mean():+12.3f}{np.mean(rnd):+16.3f}")
    print("  选择集越小、候选越多，挑出来的分数越偏乐观；这就是测试集只能报告、不能用来挑的原因")

PARTS = {1: part1_symbols, 2: part2_sources, 3: part3_splits, 4: part4_leakage,
         5: part5_curves, 6: part6_seeds, 7: part7_protocol}

if __name__ == "__main__":
    t_all = time.time()
    for k in [int(a) for a in sys.argv[1:] if a.isdigit()] or list(PARTS):
        t_part = time.time()
        PARTS[k]()
        print(f"  （第 {k} 部分用时 {time.time() - t_part:.0f} 秒）")
    print(f"\n第 5 章的实验全部完成（用时 {time.time() - t_all:.0f} 秒）。")
