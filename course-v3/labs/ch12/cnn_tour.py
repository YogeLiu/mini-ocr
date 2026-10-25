"""第 12 章：卷积网络。逐段运行，对照正文阅读。

    python -m labs.ch12.cnn_tour             # 全部 7 个部分，GPU 上约 2–3 分钟
    python -m labs.ch12.cnn_tour 3 4         # 只运行第 3、4 部分
    python -m labs.ch12.cnn_tour --cpu
"""
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from miniocr.data.render import render_glyph, save_grid
from miniocr.data.tasks import CHARSET
from miniocr.models.cnn import DeepCNN, SmallCNN, random_translate
from miniocr.models.mlp import MLP
from miniocr.nn.conv import conv2d_im2col, conv2d_loops, conv_out_size, receptive_field
from miniocr.torch_basics import GlyphDataset, pick_device
from miniocr.train.loop import confusion_matrix, evaluate, fit, predict, top_confusions

DEV = torch.device("cpu") if "--cpu" in sys.argv else pick_device()
torch.backends.cudnn.deterministic = True             # 让 GPU 上的卷积每次结果相同（略慢）
torch.backends.cudnn.benchmark = False
OUT = "outputs/ch12"
os.makedirs(OUT, exist_ok=True)
_CACHE = {}
_MODELS = {}


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def data(align, n_train=100):
    key = (align, n_train)
    if key not in _CACHE:
        _CACHE[key] = tuple(GlyphDataset(s, n_per_class=n, align=align, distort=1.0) for s, n in ((0, n_train), (1, 20), (2, 50)))
    return _CACHE[key]


def loaders(align, shift=0, n_train=100):
    tr, va, te = data(align, n_train)
    g = torch.Generator().manual_seed(0)
    collate = None
    if shift:
        def collate(batch):
            x = torch.stack([b[0] for b in batch])
            return random_translate(x, shift, g), torch.stack([b[1] for b in batch])
    return (DataLoader(tr, batch_size=64, shuffle=True, generator=g, collate_fn=collate),
            DataLoader(va, batch_size=128), DataLoader(te, batch_size=128))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def train(name, make, align, epochs=15, shift=0, n_train=100):
    """训练并缓存模型（第 4、5、6 部分会复用第 3 部分训练好的模型）。"""
    key = (name, align, shift, n_train, epochs)
    for m, *_ in _MODELS.values():                       # GPU 显存有限：缓存的模型都放回内存
        m.cpu()
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    if key not in _MODELS:
        trl, val, tel = loaders(align, shift, n_train)
        torch.manual_seed(0)
        model = make().to(DEV)
        t = time.perf_counter()
        hist = fit(model, trl, val, torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01), epochs, DEV, log=None)
        _MODELS[key] = (model, hist, evaluate(model, tel, DEV)[1], time.perf_counter() - t)
    _MODELS[key][0].to(DEV)
    return _MODELS[key]


# ---------------------------------------------------------------------- 1
def part1_conv():
    section("1. 卷积：手算、循环、im2col、PyTorch")
    x = np.arange(16, dtype=np.float64).reshape(1, 1, 4, 4)
    w = np.array([[[[1, 0], [0, -1]]]], dtype=np.float64)
    print("  输入 4×4：\n" + "\n".join("    " + " ".join(f"{v:3.0f}" for v in row) for row in x[0, 0]))
    print("  卷积核 [[1, 0], [0, -1]]（左上 − 右下）。输出 3×3：")
    print("\n".join("    " + " ".join(f"{v:3.0f}" for v in row) for row in conv2d_loops(x, w)[0, 0]))
    img, _ = render_glyph("A", np.random.default_rng(0))
    a = (255 - img.astype(np.float64)) / 255
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    kernels = {"vertical edge (Sobel x)": sobel_x, "horizontal edge (Sobel y)": sobel_x.T,
               "blur 3x3": np.full((3, 3), 1 / 9), "diagonal": np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], float)}
    W = np.stack([k[None] for k in kernels.values()])                     # (4, 1, 3, 3)
    out = conv2d_loops(a[None, None], W, padding=1)[0]
    fig, ax = plt.subplots(1, 5, figsize=(12, 2.6))
    ax[0].imshow(a, cmap="gray_r"); ax[0].set_title("input 'A'")
    for i, name in enumerate(kernels):
        ax[i + 1].imshow(out[i], cmap="RdBu_r", vmin=-np.abs(out[i]).max(), vmax=np.abs(out[i]).max())
        ax[i + 1].set_title(name, fontsize=9)
    for a_ in ax:
        a_.axis("off")
    fig.tight_layout(); fig.savefig(f"{OUT}/kernels.png", dpi=100); plt.close(fig)
    print(f"  四个手工设计的卷积核作用在字母 A 上，结果存到 {OUT}/kernels.png")
    for i, name in enumerate(kernels):
        print(f"    {name:26s} 输出范围 [{out[i].min():6.2f}, {out[i].max():6.2f}]")
    rng = np.random.default_rng(0)
    xb, wb, bb = rng.normal(size=(8, 16, 16, 16)), rng.normal(size=(32, 16, 3, 3)), rng.normal(size=32)
    t = time.perf_counter(); y1 = conv2d_loops(xb, wb, bb, padding=1); t1 = time.perf_counter() - t
    t = time.perf_counter(); y2 = conv2d_im2col(xb, wb, bb, padding=1); t2 = time.perf_counter() - t
    t = time.perf_counter()
    y3 = F.conv2d(torch.from_numpy(xb), torch.from_numpy(wb), torch.from_numpy(bb), padding=1).numpy()
    t3 = time.perf_counter() - t
    print(f"  批 8、16→32 通道、16×16、3×3：循环 {t1 * 1000:8.1f} ms   im2col {t2 * 1000:6.2f} ms   F.conv2d（CPU）{t3 * 1000:6.2f} ms")
    print(f"  三者最大差异：循环 vs PyTorch {np.abs(y1 - y3).max():.1e}，im2col vs PyTorch {np.abs(y2 - y3).max():.1e}")
    print(f"  im2col 矩阵的形状：({8 * 16 * 16}, {16 * 9})，乘以卷积核矩阵 ({16 * 9}, 32)；"
          f"它比输入大 {8 * 16 * 16 * 16 * 9 / xb.size:.0f} 倍（每个像素被复制进 9 个窗口）")


# ---------------------------------------------------------------------- 2
def part2_shapes():
    section("2. SmallCNN 的形状、参数量与感受野")
    m = SmallCNN(62, 32)
    x = torch.zeros(1, 1, 32, 32)
    print(f"  输入 {tuple(x.shape)}")
    h = x
    for layer in m.features:
        h = layer(h)
        kind = type(layer).__name__ if not isinstance(layer, torch.nn.Sequential) else "conv3×3+BN+ReLU"
        print(f"    {kind:16s} → {tuple(h.shape)}  参数 {n_params(layer):6d}")
    print(f"    全局平均池化     → (1, {h.shape[1]})")
    print(f"    Linear           → (1, 62)  参数 {n_params(m.head):6d}")
    print(f"  合计 {n_params(m):,} 个参数；第 10 章的 MLP [1024, 512, 62]：{n_params(MLP([1024, 512, 62])):,} 个")
    layers = [(3, 1), (2, 2), (3, 1), (2, 2), (3, 1), (2, 2)]
    rf = [receptive_field(layers[:i + 1]) for i in range(len(layers))]
    print(f"  感受野（每层之后，一个像素看得到的输入范围）：{rf}  → 最后一层每个位置看到 {rf[-1]}×{rf[-1]} 的输入")
    print(f"  验证 conv_out_size：32 → 3×3 卷积(padding=1) → {conv_out_size(32, 3, 1, 1)} → 2×2 池化 → {conv_out_size(32, 2, 2)}")
    conv = torch.nn.Conv2d(1, 32, 3, padding=1)
    lin = torch.nn.Linear(1024, 32 * 1024)
    print(f"  第一层：卷积 1→32 通道只有 {n_params(conv)} 个参数；若用全连接产生同样大小的输出 (32×32×32)，"
          f"需要 {lin.weight.numel() + lin.bias.numel():,} 个")


# ---------------------------------------------------------------------- 3
def part3_train():
    section("3. CNN 对 MLP：AdamW lr=1e-3，15 轮")
    rows = [("MLP", lambda: MLP([1024, 512, 62])), ("SmallCNN", lambda: SmallCNN(62, 32)),
            ("SmallCNN（拉平，不做全局池化）", lambda: SmallCNN(62, 32, pool="flatten"))]
    for align in (False, True):
        print(f"  {'归一化字符' if align else '原图（位置、大小随机）'}：")
        for name, make in rows:
            model, hist, te, dt = train(name, make, align)
            print(f"    {name:30s} 参数 {n_params(model):7,}  训练 {hist[-1]['train_acc']:6.1%}  验证（最好）{max(h['val_acc'] for h in hist):6.1%}"
                  f"  测试 {te:6.1%}  {dt:4.1f} 秒")
    for align in (False, True):
        model = train("SmallCNN", rows[1][1], align)[0]
        pred, y = predict(model, loaders(align)[2], DEV)
        C = confusion_matrix(pred, y, 62)
        print(f"  SmallCNN（{'归一化' if align else '原图'}）最常见错误：" + "  ".join(f"{a}→{b}×{n}" for a, b, n in top_confusions(C, CHARSET, 8)))
        pairs = "cosvwxz"
        wrong = sum(C[CHARSET.index(c), CHARSET.index(c.upper())] + C[CHARSET.index(c.upper()), CHARSET.index(c)] for c in pairs)
        print(f"    7 对大小写同形字母之间互相认错的总次数：{wrong}（共 {7 * 2 * 50} 张）")


# ---------------------------------------------------------------------- 4
def shifted(ds, dx, dy=0):
    """把整个数据集的图平移 (dy, dx) 像素，空出的地方补 0。"""
    x = F.pad(ds.x, (abs(dx), abs(dx), abs(dy), abs(dy)))
    H, W = ds.x.shape[2:]
    return x[:, :, abs(dy) - dy:abs(dy) - dy + H, abs(dx) - dx:abs(dx) - dx + W]


def acc_on(model, x, y):
    model.to(DEV).eval()
    with torch.no_grad():
        return (torch.cat([model(x[i:i + 128].to(DEV)).argmax(1).cpu() for i in range(0, len(x), 128)]) == y).float().mean().item()


def part4_shift():
    section("4. 平移测试：在归一化字符上训练，测试时把图向右平移 k 个像素")
    te = data(True)[2]
    models = {"MLP": train("MLP", lambda: MLP([1024, 512, 62]), True)[0],
              "SmallCNN": train("SmallCNN", lambda: SmallCNN(62, 32), True)[0],
              "SmallCNN（拉平）": train("SmallCNN（拉平，不做全局池化）", lambda: SmallCNN(62, 32, pool="flatten"), True)[0],
              "MLP + 平移增强 ±4": train("MLP", lambda: MLP([1024, 512, 62]), True, shift=4)[0],
              "SmallCNN + 平移增强 ±4": train("SmallCNN", lambda: SmallCNN(62, 32), True, shift=4)[0]}
    shifts = (0, 1, 2, 3, 4, 6, 8)
    print("  平移像素数          " + "".join(f"{k:7d}" for k in shifts))
    res = {}
    for name, m in models.items():
        res[name] = [acc_on(m, shifted(te, k), te.y) for k in shifts]
        print(f"  {name:20s}" + "".join(f"{a:7.1%}" for a in res[name]))
        m.cpu()
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for name, r in res.items():
        ax.plot(shifts, r, "o-", label=name.replace("（拉平）", " (flatten)").replace(" + 平移增强 ±4", " + shift aug"))
    ax.set_xlabel("shift (pixels)"); ax.set_ylabel("test accuracy"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/shift.png", dpi=100); plt.close(fig)
    print(f"  曲线存到 {OUT}/shift.png")
    m = models["SmallCNN"].to(DEV)
    m.eval()
    with torch.no_grad():
        f0 = m.features(te.x[:64].to(DEV))
        f1 = m.features(shifted(te, 8)[:64].to(DEV))
    print(f"  SmallCNN 最后一层特征图 (128, 4, 4)：平移 8 像素（= 1 个特征图格子）后，"
          f"把特征图移回一格与原来的平均差异 {(f1[..., :, 1:] - f0[..., :, :-1]).abs().mean().item():.4f}，"
          f"不移回的差异 {(f1 - f0).abs().mean().item():.4f}（特征的平均大小 {f0.abs().mean().item():.4f}）")


# ---------------------------------------------------------------------- 5
def part5_filters():
    section("5. 网络学到了什么：第一层卷积核与特征图")
    m = train("SmallCNN", lambda: SmallCNN(62, 32), False)[0]
    w = m.features[0][0].weight.detach().cpu().numpy()[:, 0]               # (32, 3, 3)
    fig, ax = plt.subplots(4, 8, figsize=(8, 4))
    for i in range(32):
        a = ax[i // 8, i % 8]
        a.imshow(w[i], cmap="RdBu_r", vmin=-np.abs(w[i]).max(), vmax=np.abs(w[i]).max()); a.axis("off")
    fig.tight_layout(); fig.savefig(f"{OUT}/filters.png", dpi=100); plt.close(fig)
    te = data(False)[2]
    x = te.x[CHARSET.index("A"):CHARSET.index("A") + 1].to(DEV)
    m.eval()
    with torch.no_grad():
        maps = [x]
        h = x
        for layer in m.features:
            h = layer(h)
            if isinstance(layer, torch.nn.MaxPool2d):
                maps.append(h)
    fig, ax = plt.subplots(4, 6, figsize=(9, 6))
    for r, fm in enumerate(maps):
        for c in range(6):
            a = ax[r, c]
            a.axis("off")
            if c >= fm.shape[1]:
                continue
            a.imshow(fm[0, c].cpu().numpy(), cmap="viridis")
            if c == 0:
                a.set_title(f"{tuple(fm.shape[1:])}", fontsize=8, loc="left")
    fig.tight_layout(); fig.savefig(f"{OUT}/feature_maps.png", dpi=100); plt.close(fig)
    norms = np.sqrt((w ** 2).sum((1, 2)))
    sums = w.sum((1, 2))
    print(f"  32 个 3×3 卷积核存到 {OUT}/filters.png；字母 A 在各层的前 6 个特征图存到 {OUT}/feature_maps.png")
    print(f"  卷积核的权重之和（≈0 表示对均匀区域不响应，像边缘检测器）：|和|/范数 的中位数 {np.median(np.abs(sums) / norms):.2f}")
    with torch.no_grad():
        act = [(fm[0] > 0).float().mean().item() for fm in maps[1:]]
    print(f"  三个池化层之后，特征图中大于 0 的比例：" + "  ".join(f"{a:.0%}" for a in act))


# ---------------------------------------------------------------------- 6
def part6_data():
    section("6. 数据量与增强：原图，每类 5 / 20 / 100 张，SmallCNN")
    for n in (5, 20, 100):
        for shift in (0, 3):
            epochs = {5: 150, 20: 60, 100: 15}[n]
            model, hist, te, dt = train("SmallCNN", lambda: SmallCNN(62, 32), False, epochs=epochs, shift=shift, n_train=n)
            print(f"  每类 {n:3d} 张  增强 ±{shift}  {epochs:3d} 轮：训练 {hist[-1]['train_acc']:6.1%}  验证（最好）"
                  f"{max(h['val_acc'] for h in hist):6.1%}  测试 {te:6.1%}")
    print("  （参照：第 10 章 MLP 在归一化字符上，每类 5 张测试 73.2%，每类 100 张 92.3%）")


# ---------------------------------------------------------------------- 7
def part7_residual():
    section("7. 深层 CNN：有无残差连接（原图，16 通道，AdamW lr=1e-3，8 轮）")
    for nb in (3, 9):  # 7 层与 19 层
        for res in (False, True):
            name = f"{2 * nb + 1} 层卷积 {'残差' if res else '普通'}"
            model, hist, te, dt = train(name, lambda: DeepCNN(nb, 16, residual=res), False, epochs=8)
            print(f"  {name:14s} 参数 {n_params(model):7,}  第 1/2/4/8 轮训练损失 "
                  + " ".join(f"{hist[e - 1]['train_loss']:.3f}" for e in (1, 2, 4, 8))
                  + f"  测试 {te:6.1%}  {dt:4.1f} 秒")


PARTS = {1: part1_conv, 2: part2_shapes, 3: part3_train, 4: part4_shift, 5: part5_filters, 6: part6_data, 7: part7_residual}

if __name__ == "__main__":
    print(f"PyTorch {torch.__version__}，使用设备 {DEV}")
    for k in [int(a) for a in sys.argv[1:] if a.isdigit()] or list(PARTS):
        PARTS[k]()
    print("\n第 12 章的演示全部完成。")
