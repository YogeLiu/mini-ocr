"""第 10 章：训练循环与单字符识别。逐段运行，对照正文阅读。

    python -m labs.ch10.train_chars            # 全部 7 个部分，GPU 上约 1–2 分钟
    python -m labs.ch10.train_chars 3 5        # 只运行第 3、5 部分
    python -m labs.ch10.train_chars --cpu
"""
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from miniocr.data.render import save_grid
from miniocr.data.tasks import CHARSET
from miniocr.models.mlp import MLP
from miniocr.torch_basics import GlyphDataset, pick_device
from miniocr.train.loop import (confusion_matrix, evaluate, fit, overfit_one_batch, predict,
                                top_confusions, train_one_epoch)

DEV = torch.device("cpu") if "--cpu" in sys.argv else pick_device()
OUT = "outputs/ch10"
os.makedirs(OUT, exist_ok=True)
_CACHE = {}


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def data(chars=CHARSET, n_train=100, align=True, distort=1.0):
    """训练、验证、测试三份数据，种子 0、1、2，互不重叠。
    align=True：先做第 2 章的字符归一化；distort：随机仿射形变的幅度（第 1 部分解释为什么需要它）。"""
    key = (chars, n_train, align, distort)
    if key not in _CACHE:
        _CACHE[key] = tuple(GlyphDataset(seed, chars, n, align=align, distort=distort)
                            for seed, n in ((0, n_train), (1, 20), (2, 50)))
    return _CACHE[key]


def n_duplicates(a, b):
    """b 里有多少张图与 a 里的某张图逐像素完全相同。"""
    seen = {v.numpy().tobytes() for v in a.x}
    return sum(v.numpy().tobytes() in seen for v in b.x)


def loaders(tr, va, te, bs=64, seed=0, sampler=None):
    g = torch.Generator().manual_seed(seed)
    return (DataLoader(tr, batch_size=bs, shuffle=sampler is None, sampler=sampler, generator=g),
            DataLoader(va, batch_size=512), DataLoader(te, batch_size=512))


def part1_data():
    section("1. 数据：训练、验证、测试")
    print("  (a) 第 7 章用过的数据：归一化、不形变。测试图与训练图重复了多少？")
    for name, al in (("原图", False), ("归一化", True)):
        tr, _, te = data(align=al, distort=0.0)
        d = n_duplicates(tr, te)
        print(f"    {name}：测试集 {len(te)} 张里，与训练集某张图逐像素完全相同的有 {d:4d} 张（{d / len(te):5.1%}）")
    tr0 = data(distort=0.0)[0]
    uniq = len({v.numpy().tobytes() for v in tr0.x[tr0.y == CHARSET.index("A")]})
    print(f"    归一化后，训练集里 100 张 'A' 只有 {uniq} 种不同的图（字号 × 是否加粗，位置被归一化消掉了）")
    print("  (b) 加上随机仿射形变（distort=1.0）后：")
    t = time.perf_counter()
    raw, _, _ = data(align=False)
    tr, va, te = data()
    print(f"    训练 {len(tr)} 张（每类 100）  验证 {len(va)} 张（每类 20）  测试 {len(te)} 张（每类 50），"
          f"原图和归一化两个版本，渲染用时 {time.perf_counter() - t:.1f} 秒")
    for name, (a, b) in (("原图", (raw, data(align=False)[2])), ("归一化", (tr, te))):
        print(f"    {name}：测试集与训练集逐像素相同的图 {n_duplicates(a, b)} 张")
    for name, ds in (("raw", raw), ("aligned", tr), ("aligned_nodistort", tr0)):
        imgs = [(255 - ds.x[i, 0].numpy() * 255).astype(np.uint8) for i in range(62)]
        save_grid(imgs, f"{OUT}/samples_{name}.png", scale=2, cols=31)
    print(f"    前 62 张训练图存到 {OUT}/samples_raw.png、samples_aligned.png、samples_aligned_nodistort.png")
    for name, ds in (("原图", raw), ("归一化", tr)):
        rows = (ds.x[:, 0] > 0.5).any(2).float().argmax(1).float()
        print(f"    {name}：墨迹最上面一行的位置 均值 {rows.mean():5.2f}  标准差 {rows.std():5.2f}；墨迹像素占比 {ds.x.mean():.3f}")


def part2_health():
    section("2. 训练前的体检")
    tr, va, te = data()
    torch.manual_seed(0)
    model = MLP([1024, 512, 62]).to(DEV)
    xb, yb = next(iter(DataLoader(tr, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(0))))
    xb, yb = xb.to(DEV), yb.to(DEV)
    with torch.no_grad():
        logits = model(xb)
        loss0 = F.cross_entropy(logits, yb).item()
    print(f"  (a) 初始损失 {loss0:.4f}，ln 62 = {math.log(62):.4f}；logits 的标准差 {logits.std():.3f}")
    print(f"      初始预测的类别分布：64 张图被预测成了 {logits.argmax(1).unique().numel()} 个不同的类别")
    losses = overfit_one_batch(model, xb, yb, torch.optim.SGD(model.parameters(), lr=0.1), steps=300)
    print("  (b) 在同一批 64 张图上训练 300 步，损失：",
          "  ".join(f"第{s}步 {losses[s - 1]:.4f}" for s in (1, 50, 100, 200, 300)))
    with torch.no_grad():
        acc = (model(xb).argmax(1) == yb).float().mean().item()
    print(f"      这一批的准确率 {acc:.0%}（能背下来，说明模型、损失、反向、更新这条链路是通的）")
    xr = torch.rand_like(xb)
    torch.manual_seed(0)
    m2 = MLP([1024, 512, 62]).to(DEV)
    l2 = overfit_one_batch(m2, xr, yb, torch.optim.SGD(m2.parameters(), lr=0.1), steps=300)
    print(f"  (c) 把输入换成随机噪声，同样 300 步：损失 {l2[0]:.4f} → {l2[-1]:.4f}（也能背下来：背下一批数据不代表学会了）")


def part3_train():
    section("3. 训练：MLP [1024, 512, 62]，SGD lr=0.1，批大小 64，30 轮")
    print("  (a) 先用原图（字符的位置、大小随机，也有形变）：")
    tr, va, te = data(align=False)
    trl, val, tel = loaders(tr, va, te)
    torch.manual_seed(0)
    model = MLP([1024, 512, 62]).to(DEV)
    hist = fit(model, trl, val, torch.optim.SGD(model.parameters(), lr=0.1), 30, DEV, log=None)
    for h in hist:
        if h["epoch"] in (1, 10, 20, 30):
            print(f"    第 {h['epoch']:2d} 轮  训练损失 {h['train_loss']:.4f}  训练准确率 {h['train_acc']:6.1%}  验证准确率 {h['val_acc']:6.1%}")
    pred, y = predict(model, tel, DEV)
    print(f"    测试准确率 {(pred == y).mean():.1%}；最常见错误",
          "  ".join(f"{a}→{b}×{n}" for a, b, n in top_confusions(confusion_matrix(pred, y, 62), CHARSET, 6)))
    print("  (b) 换成归一化后的字符（第 2 章 normalize_glyph），其他完全不变：")
    tr, va, te = data()
    trl, val, tel = loaders(tr, va, te)
    torch.manual_seed(0)
    model = MLP([1024, 512, 62]).to(DEV)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    t = time.perf_counter()
    hist = fit(model, trl, val, opt, 30, DEV, log=None)
    for h in hist:
        if h["epoch"] in (1, 2, 3, 5, 10, 15, 20, 25, 30):
            print(f"    第 {h['epoch']:2d} 轮  训练损失 {h['train_loss']:.4f}  训练准确率 {h['train_acc']:6.1%}  "
                  f"验证损失 {h['val_loss']:.4f}  验证准确率 {h['val_acc']:6.1%}")
    best = max(hist, key=lambda h: h["val_acc"])
    print(f"  用时 {time.perf_counter() - t:.1f} 秒；验证准确率最高的是第 {best['epoch']} 轮（{best['val_acc']:.1%}），已恢复这一轮的参数")
    te_loss, te_acc = evaluate(model, tel, DEV)
    print(f"  测试集：损失 {te_loss:.4f}，准确率 {te_acc:.1%}（只在最后看这一次）")
    pred, y = predict(model, tel, DEV)
    C = confusion_matrix(pred, y, 62)
    print("  最常见的 12 种错误（真实 → 预测 × 次数）：",
          "  ".join(f"{a}→{b}×{n}" for a, b, n in top_confusions(C, CHARSET, 12)))
    per = C.diagonal() / C.sum(1)
    order = np.argsort(per, kind="stable")
    print("  准确率最低的 8 个字符：", "  ".join(f"{CHARSET[i]} {per[i]:.0%}" for i in order[:8]))
    print(f"  准确率 100% 的字符有 {(per == 1).sum()} 个")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    e = [h["epoch"] for h in hist]
    ax[0].plot(e, [h["train_loss"] for h in hist], label="train")
    ax[0].plot(e, [h["val_loss"] for h in hist], label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
    ax[1].plot(e, [h["train_acc"] for h in hist], label="train")
    ax[1].plot(e, [h["val_acc"] for h in hist], label="val")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy"); ax[1].legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/curves.png", dpi=100); plt.close(fig)
    print(f"  训练曲线存到 {OUT}/curves.png")
    torch.save(model.state_dict(), f"{OUT}/mlp.pt")
    m2 = MLP([1024, 512, 62]).to(DEV)
    m2.load_state_dict(torch.load(f"{OUT}/mlp.pt"))
    print(f"  保存到 {OUT}/mlp.pt（{os.path.getsize(f'{OUT}/mlp.pt') / 1e6:.2f} MB），重新加载后测试准确率 {evaluate(m2, tel, DEV)[1]:.1%}")


def part4_overfit():
    section("4. 过拟合：训练数据少的时候")
    for n in (5, 20, 100, 400):
        tr, va, te = data(n_train=n)
        trl, val, tel = loaders(tr, va, te)
        torch.manual_seed(0)
        model = MLP([1024, 512, 62]).to(DEV)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        steps = 6000                                    # 不管数据多少，都更新 6000 步
        epochs = max(1, round(steps / len(trl)))
        hist = fit(model, trl, val, opt, epochs, DEV, log=None)
        last, best = hist[-1], max(hist, key=lambda h: h["val_acc"])
        te_acc = evaluate(model, tel, DEV)[1]
        print(f"  每类 {n:3d} 张（共 {len(tr):5d}）{epochs:4d} 轮：最后一轮 训练准确率 {last['train_acc']:6.1%} 验证 {last['val_acc']:6.1%}"
              f"  验证损失 最低 {min(h['val_loss'] for h in hist):.3f} 最后 {last['val_loss']:.3f}"
              f"  | 最好一轮 {best['epoch']:4d}，测试 {te_acc:6.1%}")
        if n == 5:
            h5 = hist
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot([h["train_loss"] for h in h5], label="train")
    ax.plot([h["val_loss"] for h in h5], label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title("5 per class"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/overfit.png", dpi=100); plt.close(fig)
    print(f"  每类 5 张时的损失曲线存到 {OUT}/overfit.png")


def _run(name, step_fn, epochs=8):
    """用每类 100 张的数据训练 epochs 轮，step_fn(model, opt, xb, yb) 负责一批的训练，返回损失。"""
    tr, va, te = data()
    trl, val, tel = loaders(tr, va, te)
    torch.manual_seed(0)
    model = MLP([1024, 512, 62]).to(DEV)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    losses = []
    for _ in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in trl:
            xb, yb = xb.to(DEV), yb.to(DEV)
            loss = step_fn(model, opt, xb, yb)
            tot += loss * len(yb); n += len(yb)
        losses.append(tot / n)
    va_acc = evaluate(model, val, DEV)[1]
    print(f"  {name:22s} 每轮训练损失 " + " ".join(f"{l:7.3f}" for l in losses) + f"   验证准确率 {va_acc:6.1%}")


def part5_bugs():
    section("5. 故意犯 5 种错误：看症状")

    def ok(model, opt, xb, yb):
        opt.zero_grad(); loss = F.cross_entropy(model(xb), yb); loss.backward(); opt.step(); return loss.item()

    def no_zero_grad(model, opt, xb, yb):
        loss = F.cross_entropy(model(xb), yb); loss.backward(); opt.step(); return loss.item()

    def double_softmax(model, opt, xb, yb):
        opt.zero_grad(); loss = F.cross_entropy(torch.softmax(model(xb), 1), yb); loss.backward(); opt.step(); return loss.item()

    def shuffled_labels(model, opt, xb, yb):
        yb = yb[torch.randperm(len(yb), device=yb.device)]
        return ok(model, opt, xb, yb)

    def lr_too_big(model, opt, xb, yb):
        for g in opt.param_groups:
            g["lr"] = 30.0
        return ok(model, opt, xb, yb)

    def mixed_batch(model, opt, xb, yb):
        xb = xb.reshape(1024, -1).T                      # 形状对了，但每一行混进了不同样本的像素
        return ok(model, opt, xb, yb)

    _run("正确", ok)
    _run("① 忘了 zero_grad", no_zero_grad)
    _run("② softmax 后再交叉熵", double_softmax)
    _run("③ 标签和图像错位", shuffled_labels)
    _run("④ 学习率太大 (30)", lr_too_big)
    _run("⑤ reshape 混了批", mixed_batch)


def part6_classes():
    section("6. 类别越多越难吗？（每类都是 100 张训练图，30 轮）")
    sets = {"10 个数字": CHARSET[:10], "26 个小写字母": CHARSET[10:36], "36 个（数字+小写）": CHARSET[:36], "62 个（全部）": CHARSET}
    for name, chars in sets.items():
        tr, va, te = data(chars)
        trl, val, tel = loaders(tr, va, te)
        torch.manual_seed(0)
        model = MLP([1024, 512, len(chars)]).to(DEV)
        fit(model, trl, val, torch.optim.SGD(model.parameters(), lr=0.1), 30, DEV, log=None)
        pred, y = predict(model, tel, DEV)
        C = confusion_matrix(pred, y, len(chars))
        acc = np.trace(C) / C.sum()
        conf = "  ".join(f"{a}→{b}×{n}" for a, b, n in top_confusions(C, chars, 4))
        if len(chars) == 62:
            C62 = C
        print(f"  {name:14s} 初始损失应为 ln {len(chars):2d} = {math.log(len(chars)):.3f}  测试准确率 {acc:6.1%}   最常见错误 {conf}")
    print("  62 类里大小写同形的字符（c/C o/O s/S v/V w/W x/X z/Z）：")
    for a in "cosvwxz":
        i, j = CHARSET.index(a), CHARSET.index(a.upper())
        C = C62
        print(f"    {a}: 对 {C[i, i]:2d} 错成 {a.upper()} {C[i, j]:2d}    {a.upper()}: 对 {C[j, j]:2d} 错成 {a} {C[j, i]:2d}")


def part7_imbalance():
    section("7. 类别不均衡：一半字符每类 100 张，另一半每类只有 5 张")
    tr, va, te = data()
    rare = set(range(31, 62))                            # 后 31 类（f..Z 之后的那一半）是"少数类"
    y = tr.y.numpy()
    keep, seen = [], {}
    for i, c in enumerate(y):
        seen[c] = seen.get(c, 0) + 1
        if c not in rare or seen[c] <= 5:
            keep.append(i)
    sub = Subset(tr, keep)
    ys = y[keep]
    counts = np.bincount(ys, minlength=62)
    print(f"  训练集 {len(sub)} 张：多数类每类 {counts[0]} 张，少数类每类 {counts[61]} 张；少数类只占 {counts[31:].sum() / len(sub):.1%}")
    print(f"  少数类是：{CHARSET[31]}…{CHARSET[61]}（{''.join(CHARSET[i] for i in sorted(rare))}）")
    tel = DataLoader(te, batch_size=512)
    val = DataLoader(va, batch_size=512)

    def report(name, loader, loss_fn=F.cross_entropy):
        torch.manual_seed(0)
        model = MLP([1024, 512, 62]).to(DEV)
        fit(model, loader, val, torch.optim.SGD(model.parameters(), lr=0.1), 30, DEV, loss_fn=loss_fn, log=None)
        pred, yt = predict(model, tel, DEV)
        C = confusion_matrix(pred, yt, 62)
        per = C.diagonal() / C.sum(1)
        pred_share = np.isin(pred, list(rare)).mean()
        print(f"  {name:18s} 总准确率 {np.trace(C) / C.sum():6.1%}  多数类 {per[:31].mean():6.1%}  少数类 {per[31:].mean():6.1%}"
              f"  测试集里被预测成少数类的比例 {pred_share:5.1%}（真实 50%）")

    g = torch.Generator().manual_seed(0)
    report("不处理", DataLoader(sub, batch_size=64, shuffle=True, generator=g))
    w = torch.tensor(len(ys) / (62 * counts), dtype=torch.float32, device=DEV)
    print(f"  加权损失：类别 k 的权重 = N / (K · n_k)，多数类 {w[0]:.3f}，少数类 {w[61]:.3f}")
    report("加权损失", DataLoader(sub, batch_size=64, shuffle=True, generator=g),
           loss_fn=lambda logits, t: F.cross_entropy(logits, t, weight=w))
    sw = (1.0 / counts[ys]).astype(np.float64)
    sampler = WeightedRandomSampler(torch.from_numpy(sw), num_samples=len(ys), replacement=True, generator=g)
    report("重采样", DataLoader(sub, batch_size=64, sampler=sampler))
    report("（参照）全部 100 张", DataLoader(tr, batch_size=64, shuffle=True, generator=g))


PARTS = {1: part1_data, 2: part2_health, 3: part3_train, 4: part4_overfit, 5: part5_bugs, 6: part6_classes, 7: part7_imbalance}

if __name__ == "__main__":
    print(f"PyTorch {torch.__version__}，使用设备 {DEV}")
    chosen = [int(a) for a in sys.argv[1:] if a.isdigit()] or list(PARTS)
    for k in chosen:
        PARTS[k]()
    print("\n第 10 章的演示全部完成。")
