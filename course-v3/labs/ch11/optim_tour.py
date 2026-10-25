"""第 11 章：优化器、学习率调度、初始化、归一化、正则化、梯度裁剪、数值精度。逐段运行，对照正文阅读。

    python -m labs.ch11.optim_tour             # 全部 8 个部分，GPU 上约 2–3 分钟
    python -m labs.ch11.optim_tour 1 4         # 只运行第 1、4 部分
    python -m labs.ch11.optim_tour --cpu
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
from torch.utils.data import DataLoader

from miniocr.models.mlp import MLP
from miniocr.torch_basics import GlyphDataset, pick_device
from miniocr.train.loop import evaluate, fit
from miniocr.train.optim import SGD, Adam, AdamW, clip_grad_norm, warmup_cosine

DEV = torch.device("cpu") if "--cpu" in sys.argv else pick_device()
OUT = "outputs/ch11"
os.makedirs(OUT, exist_ok=True)
_CACHE = {}


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def data(n_train=100, align=True):
    """第 10 章的数据：62 类，32×32，distort=1.0；训练/验证/测试种子 0/1/2。"""
    key = (n_train, align)
    if key not in _CACHE:
        _CACHE[key] = tuple(GlyphDataset(s, n_per_class=n, align=align, distort=1.0) for s, n in ((0, n_train), (1, 20), (2, 50)))
    return _CACHE[key]


def loaders(n_train=100, align=True, bs=64):
    tr, va, te = data(n_train, align)
    return (DataLoader(tr, batch_size=bs, shuffle=True, generator=torch.Generator().manual_seed(0)),
            DataLoader(va, batch_size=512), DataLoader(te, batch_size=512))


# ---------------------------------------------------------------------- 1
def part1_valley():
    section("1. 一个狭长的山谷：f(x, y) = 0.5·(x² + 25·y²)，从 (-9, 2) 出发")
    def f(w):
        return 0.5 * (w[0] ** 2 + 25 * w[1] ** 2)
    print("  Hessian = diag(1, 25)，条件数 25；梯度下降的稳定上限 lr < 2/25 = 0.08（第 7 章）")
    runs = {"SGD lr=0.07": lambda p: SGD([p], lr=0.07),
            "SGD lr=0.02": lambda p: SGD([p], lr=0.02),
            "动量 0.9 lr=0.02": lambda p: SGD([p], lr=0.02, momentum=0.9),
            "Adam lr=0.5": lambda p: Adam([p], lr=0.5)}
    paths = {}
    for name, make in runs.items():
        p = nn.Parameter(torch.tensor([-9.0, 2.0]))
        opt = make(p)
        path, hit = [p.detach().clone()], None
        for t in range(1, 201):
            opt.zero_grad()
            f(p).backward()
            opt.step()
            path.append(p.detach().clone())
            if hit is None and f(p).item() < 1e-3:
                hit = t
        paths[name] = torch.stack(path).numpy()
        print(f"  {name:16s} 第 10 步 f = {f(path[10]).item():9.4f}  第 50 步 f = {f(path[50]).item():9.5f}  "
              f"降到 1e-3 以下用了 {hit if hit else '>200'} 步")
    xs, ys = np.meshgrid(np.linspace(-10, 3, 200), np.linspace(-2.5, 2.5, 200))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.contour(xs, ys, 0.5 * (xs ** 2 + 25 * ys ** 2), levels=[0.5, 2, 8, 20, 40, 60], colors="#bbb", linewidths=0.8)
    for name, P in paths.items():
        ax.plot(P[:60, 0], P[:60, 1], ".-", ms=3, lw=1, label=name.replace("动量", "momentum"))
    ax.legend(fontsize=8); ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout(); fig.savefig(f"{OUT}/valley.png", dpi=100); plt.close(fig)
    print(f"  前 60 步的轨迹存到 {OUT}/valley.png")
    g = torch.tensor([0.001, 1.0, 1000.0])
    p = nn.Parameter(torch.zeros(3)); p.grad = g.clone()
    Adam([p], lr=0.1).step()
    print(f"  Adam 第一步：梯度 {g.tolist()} → 更新量 {(-p.detach()).tolist()}（≈ lr·符号，与梯度大小无关）")


# ---------------------------------------------------------------------- 2
def part2_optimizers():
    section("2. 四种优化器训练第 10 章的 MLP（归一化字符，15 轮）")
    trl, val, tel = loaders()
    configs = {"SGD lr=0.1": lambda ps: SGD(ps, lr=0.1),
               "SGD+动量 lr=0.1": lambda ps: SGD(ps, lr=0.1, momentum=0.9),
               "Adam lr=1e-3": lambda ps: Adam(ps, lr=1e-3),
               "AdamW lr=1e-3": lambda ps: AdamW(ps, lr=1e-3, weight_decay=0.05)}
    curves = {}
    for name, make in configs.items():
        torch.manual_seed(0)
        model = MLP([1024, 512, 62]).to(DEV)
        t = time.perf_counter()
        hist = fit(model, trl, val, make(list(model.parameters())), 15, DEV, log=None)
        curves[name] = hist
        te = evaluate(model, tel, DEV)[1]
        print(f"  {name:16s} 第 1/3/5/10/15 轮验证准确率 " + " ".join(f"{hist[e - 1]['val_acc']:6.1%}" for e in (1, 3, 5, 10, 15))
              + f"  训练 {hist[-1]['train_acc']:6.1%}  测试 {te:6.1%}  {time.perf_counter() - t:4.1f} 秒")
    torch.manual_seed(0)
    model = MLP([1024, 512, 62]).to(DEV)
    tr_raw, va_raw, te_raw = loaders(align=False)
    hist = fit(model, tr_raw, va_raw, Adam(list(model.parameters()), lr=1e-3), 30, DEV, log=None)
    print(f"  原图（不归一化）+ Adam，30 轮：训练 {hist[-1]['train_acc']:.1%}  验证 {hist[-1]['val_acc']:.1%}"
          f"  测试 {evaluate(model, te_raw, DEV)[1]:.1%}（第 10 章 SGD：测试 18.3%）")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for name, h in curves.items():
        ax.plot([r["train_loss"] for r in h], label=name.replace("动量", "mom"))
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("train loss"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/optimizers.png", dpi=100); plt.close(fig)


# ---------------------------------------------------------------------- 3
def train_steps(model, trl, opt, total, lr_at, clip=None, record=None):
    """按步训练：第 s 步先把学习率设为 lr_at(s)。record 不为 None 时记录 (损失, 裁剪前的梯度范数)。"""
    s = 0
    while s < total:
        for xb, yb in trl:
            if s >= total:
                break
            model.train()
            opt.lr = lr_at(s)
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            gn = clip_grad_norm(opt.params, clip if clip else float("inf"))
            opt.step()
            if record is not None:
                record.append((loss.item(), gn))
            s += 1


def part3_schedules():
    section("3. 学习率调度：Adam，2000 步（约 20 轮），峰值 lr = 3e-3")
    trl, val, tel = loaders()
    total, peak = 2000, 3e-3
    scheds = {"恒定": lambda s: peak,
              "阶梯（1000、1500 步各 ×0.1）": lambda s: peak * (0.1 ** ((s >= 1000) + (s >= 1500))),
              "余弦": lambda s: warmup_cosine(s, total, 0, peak),
              "预热 100 + 余弦": lambda s: warmup_cosine(s, total, 100, peak)}
    fig, ax = plt.subplots(figsize=(5, 3))
    for name, fn in scheds.items():
        torch.manual_seed(0)
        model = MLP([1024, 512, 62]).to(DEV)
        rec = []
        train_steps(model, trl, Adam(list(model.parameters()), lr=peak), total, fn, record=rec)
        va_loss, va_acc = evaluate(model, val, DEV)
        tr_loss, tr_acc = evaluate(model, trl, DEV)
        last = np.mean([r[0] for r in rec[-100:]])
        print(f"  {name:22s} 最后 100 步平均训练损失 {last:.4f}  训练准确率 {tr_acc:6.1%}  验证损失 {va_loss:.4f}  验证准确率 {va_acc:6.1%}")
        ax.plot([fn(s) for s in range(total)], label=name.split("（")[0])
    ax.set_xlabel("step"); ax.set_ylabel("lr"); fig.tight_layout()
    fig.savefig(f"{OUT}/schedules.png", dpi=100); plt.close(fig)
    print(f"  四条学习率曲线存到 {OUT}/schedules.png（图例：恒定 / 阶梯 / 余弦 / 预热+余弦）")
    print("  预热的作用（学习率较大时更明显）：峰值 lr = 1e-2")
    for warm in (0, 100):
        torch.manual_seed(0)
        model = MLP([1024, 512, 62]).to(DEV)
        rec = []
        train_steps(model, trl, Adam(list(model.parameters()), lr=1e-2), 300, lambda s: warmup_cosine(s, 2000, warm, 1e-2), record=rec)
        print(f"    预热 {warm:3d} 步：前 20 步的最大损失 {max(r[0] for r in rec[:20]):8.2f}  "
              f"最大梯度范数 {max(r[1] for r in rec[:20]):8.2f}   第 300 步验证准确率 {evaluate(model, val, DEV)[1]:6.1%}")


# ---------------------------------------------------------------------- 4
class Deep(nn.Module):
    """depth 个隐藏层、每层 width 个神经元的 MLP，可选激活、初始化方式和归一化层。
    residual=True 时，从第 2 层起每层变成 h ← h + act(norm(linear(h)))（残差连接，第 12、16 章）。"""

    def __init__(self, depth=10, width=256, act="relu", init="default", norm=None, d_in=1024, d_out=62, residual=False):
        super().__init__()
        self.residual = residual
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        d = d_in
        for _ in range(depth):
            lin = nn.Linear(d, width)
            if init == "small":
                nn.init.normal_(lin.weight, std=0.01)
            elif init == "large":
                nn.init.normal_(lin.weight, std=1.0 / math.sqrt(d) * 3)
            elif init == "xavier":
                nn.init.normal_(lin.weight, std=math.sqrt(1.0 / d))
            elif init == "kaiming":
                nn.init.normal_(lin.weight, std=math.sqrt(2.0 / d))
            if init != "default":
                nn.init.zeros_(lin.bias)
            self.layers.append(lin)
            self.norms.append(nn.BatchNorm1d(width) if norm == "bn" else nn.LayerNorm(width) if norm == "ln" else nn.Identity())
            d = width
        self.act = torch.relu if act == "relu" else torch.tanh
        self.head = nn.Linear(d, d_out)

    def forward(self, x, stats=None):
        h = x.flatten(1)
        for i, (lin, nm) in enumerate(zip(self.layers, self.norms)):
            z = self.act(nm(lin(h)))
            h = h + z if self.residual and i > 0 else z
            if stats is not None:
                stats.append(h.std().item())
        return self.head(h)


def part4_init():
    section("4. 初始化：10 层、每层 256 个神经元的 MLP，各层输出的标准差")
    tr, va, te = data()
    x = tr.x[:512].to(DEV)
    x = (x - x.mean()) / x.std()                         # 输入标准化成均值 0、标准差 1，便于比较
    for act in ("tanh", "relu"):
        for init in ("small", "xavier", "kaiming", "large", "default"):
            torch.manual_seed(0)
            m = Deep(act=act, init=init).to(DEV)
            st = []
            with torch.no_grad():
                m(x, st)
            print(f"  {act:4s} {init:8s} 第 1/2/4/6/8/10 层：" + " ".join(f"{st[i]:9.2e}" for i in (0, 1, 3, 5, 7, 9)))
    print("  各种初始化训练 300 步（Adam lr=1e-3）后的验证准确率：")
    trl, val, _ = loaders()
    for act, init in (("relu", "small"), ("relu", "kaiming"), ("relu", "large"), ("tanh", "xavier"), ("relu", "default")):
        torch.manual_seed(0)
        m = Deep(act=act, init=init).to(DEV)
        rec = []
        train_steps(m, trl, Adam(list(m.parameters()), lr=1e-3), 300, lambda s: 1e-3, record=rec)
        print(f"    {act:4s} {init:8s} 第 1 步损失 {rec[0][0]:8.3f}  第 300 步附近损失 {np.mean([r[0] for r in rec[-20:]]):.3f}"
              f"  验证准确率 {evaluate(m, val, DEV)[1]:6.1%}")


# ---------------------------------------------------------------------- 5
def part5_norm():
    section("5. 归一化层：20 层的深 MLP（PyTorch 默认初始化），普通 SGD，600 步")
    trl, val, _ = loaders()
    for norm, res in ((None, False), ("bn", False), ("ln", False), ("ln", True)):
        for lr in (0.01, 0.05):
            torch.manual_seed(0)
            m = Deep(depth=20, width=256, norm=norm, residual=res).to(DEV)
            rec = []
            train_steps(m, trl, SGD(list(m.parameters()), lr=lr), 600, lambda s: lr, record=rec)
            name = f"{str(norm)}{' + 残差' if res else ''}"
            print(f"  {name:10s} lr={lr:<5} 第 100/300/600 步附近损失 "
                  + " ".join(f"{np.mean([r[0] for r in rec[s - 20:s]]):.3f}" for s in (100, 300, 600))
                  + f"   验证准确率 {evaluate(m, val, DEV)[1]:6.1%}")
    for norm in (None, "ln", "bn"):
        torch.manual_seed(0)
        m = Deep(depth=20, width=256, norm=norm).to(DEV)
        xb, yb = next(iter(trl))
        F.cross_entropy(m(xb.to(DEV)), yb.to(DEV)).backward()
        print(f"  初始化时各层权重梯度的范数（{str(norm):4s}）第 1/6/11/20 层："
              + " ".join(f"{m.layers[i].weight.grad.norm().item():9.2e}" for i in (0, 5, 10, 19)))
    print("  BatchNorm 在训练模式和评测模式下的区别（用上面训练好的 bn 模型的写法重新训一个）：")
    torch.manual_seed(0)
    m = Deep(depth=4, width=256, norm="bn").to(DEV)
    train_steps(m, trl, SGD(list(m.parameters()), lr=0.05, momentum=0.9), 300, lambda s: 0.05)
    _, va, _ = data()
    xb, yb = va.x[:256].to(DEV), va.y[:256].to(DEV)
    with torch.no_grad():
        m.eval(); acc_eval = (m(xb).argmax(1) == yb).float().mean().item()
        one = torch.cat([m(xb[i:i + 1]) for i in range(16)]).argmax(1)
        one_ok = (one == m(xb[:16]).argmax(1)).float().mean().item()   # 必须在下面的训练模式前向之前比较
        m.train(); acc_train = (m(xb).argmax(1) == yb).float().mean().item()
        acc_small = (m(xb[:4]).argmax(1) == yb[:4]).float().mean().item()
        same = (m(xb[:8]).argmax(1) == m(xb[:8].flip(0)).flip(0).argmax(1)).float().mean().item()
    m.eval()
    print(f"    评测模式（用滑动平均）  256 张一起：准确率 {acc_eval:.1%}；逐张单独送入与一起送入的预测一致率 {one_ok:.0%}")
    print(f"    训练模式（用这一批的统计）256 张一起：准确率 {acc_train:.1%}；只送 4 张：{acc_small:.0%}")
    print(f"    （训练模式下，把 8 张图换个顺序预测不变：{same:.0%}；但批的组成一变，统计量就变）")
    print("    注意：训练模式的这几次前向也更新了滑动平均（running_mean/var），这就是评测前必须 model.eval() 的原因之一")


# ---------------------------------------------------------------------- 6
def part6_regularize():
    section("6. 正则化：每类只有 5 张训练图（第 10 章：验证 74%），Adam lr=1e-3，300 轮")
    trl, val, tel = loaders(n_train=5, bs=64)
    configs = {"无": (0.0, lambda ps: Adam(ps, lr=1e-3)),
               "Dropout 0.5": (0.5, lambda ps: Adam(ps, lr=1e-3)),
               "L2（Adam wd=1e-3）": (0.0, lambda ps: Adam(ps, lr=1e-3, weight_decay=1e-3)),
               "AdamW wd=0.5": (0.0, lambda ps: AdamW(ps, lr=1e-3, weight_decay=0.5)),
               "Dropout + AdamW": (0.5, lambda ps: AdamW(ps, lr=1e-3, weight_decay=0.5))}
    for name, (dp, make) in configs.items():
        torch.manual_seed(0)
        model = MLP([1024, 512, 62], dropout=dp).to(DEV)
        hist = fit(model, trl, val, make(list(model.parameters())), 300, DEV, log=None)
        best = max(hist, key=lambda h: h["val_acc"])
        wnorm = model.net[0].weight.norm().item()
        print(f"  {name:18s} 最后一轮 训练 {hist[-1]['train_acc']:6.1%} 验证 {hist[-1]['val_acc']:6.1%} 验证损失 {hist[-1]['val_loss']:.3f}"
              f" | 最好一轮 {best['epoch']:3d}，测试 {evaluate(model, tel, DEV)[1]:6.1%}  第一层权重范数 {wnorm:6.2f}")
    torch.manual_seed(0)
    model = MLP([1024, 512, 62], dropout=0.5).to(DEV)
    fit(model, trl, val, Adam(list(model.parameters()), lr=1e-3), 50, DEV, log=None)
    model.train()
    with torch.no_grad():
        accs = [(model(data(5)[1].x.to(DEV)).argmax(1).cpu() == data(5)[1].y).float().mean().item() for _ in range(3)]
    model.eval()
    print(f"  Dropout 模型忘了 eval()：三次评测 {', '.join(f'{a:.1%}' for a in accs)}（每次不同）；"
          f"eval() 后 {evaluate(model, val, DEV)[1]:.1%}")


# ---------------------------------------------------------------------- 7
def part7_clip():
    section("7. 梯度裁剪：6 层 MLP，SGD lr=0.3 + 动量 0.9（太大），500 步")
    trl, val, _ = loaders()
    for clip in (None, 5.0, 1.0):
        torch.manual_seed(0)
        model = Deep(depth=6, width=256).to(DEV)
        rec = []
        train_steps(model, trl, SGD(list(model.parameters()), lr=0.3, momentum=0.9), 500, lambda s: 0.3, clip=clip, record=rec)
        gn = np.array([r[1] for r in rec])
        loss = np.array([r[0] for r in rec])
        finite = np.isfinite(loss)
        print(f"  裁剪 {str(clip):4s} 梯度范数 中位数 {np.median(gn[np.isfinite(gn)]):8.2f} 最大 {np.nanmax(np.where(np.isfinite(gn), gn, np.nan)):10.2f}"
              f"  损失出现 nan/inf 的步数 {int((~finite).sum()):3d}  最后 50 步平均损失 {np.mean(loss[-50:]) if finite[-50:].all() else float('nan'):.3f}"
              f"  验证准确率 {evaluate(model, val, DEV)[1]:6.1%}")


# ---------------------------------------------------------------------- 8
def part8_precision():
    section("8. 数值精度：float32 / float16 / bfloat16")
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        fi = torch.finfo(dt)
        one = torch.tensor(1.0, dtype=dt)
        print(f"  {str(dt):15s} 位数 {fi.bits:2d}  最大 {fi.max:10.3e}  最小正规数 {fi.tiny:9.3e}  eps {fi.eps:9.3e}  "
              f"1 + 1e-3 = {(one + 1e-3).item():.6f}  70000 → {torch.tensor(70000.0).to(dt).item()}  1e-8 → {torch.tensor(1e-8).to(dt).item():.3e}")
    s = torch.tensor(0.0, dtype=torch.float16)
    for _ in range(10000):
        s += torch.tensor(1e-4, dtype=torch.float16)
    s32 = torch.tensor(0.0)
    for _ in range(10000):
        s32 += 1e-4
    print(f"  把 1e-4 累加 10000 次（应为 1.0）：float16 得 {s.item():.4f}，float32 得 {s32.item():.4f}")
    g = torch.tensor([3e-8, 1e-6, 1e-3])
    print(f"  梯度 {g.tolist()} 转成 float16：{g.half().float().tolist()}")
    print(f"  先乘 1024（loss scaling）再转 float16、再除回：{((g * 1024).half().float() / 1024).tolist()}")
    if DEV.type != "cuda":
        print("  （没有 CUDA，跳过混合精度训练的速度对比）")
        return
    print("  混合精度训练（autocast），MLP [1024, 1024, 1024, 62]，批大小 256，200 步：")
    tr, _, _ = data()
    xb_all = tr.x.to(DEV)
    yb_all = tr.y.to(DEV)
    va = data()[1]
    for mode in ("fp32", "bf16", "fp16"):
      try:
        torch.manual_seed(0)
        model = MLP([1024, 1024, 1024, 62]).to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=(mode == "fp16"))
        dt = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(mode)
        g = torch.Generator(device="cpu").manual_seed(0)
        torch.cuda.synchronize(); t = time.perf_counter()
        for step in range(200):
            idx = torch.randint(0, len(yb_all), (256,), generator=g).to(DEV)
            with torch.autocast("cuda", dtype=dt, enabled=dt is not None):
                loss = F.cross_entropy(model(xb_all[idx]), yb_all[idx])
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        torch.cuda.synchronize(); dt_s = time.perf_counter() - t
        model.eval()
        with torch.no_grad():
            acc = (model(va.x.to(DEV)).argmax(1).cpu() == va.y).float().mean().item()
        extra = f"  最终 loss scale {scaler.get_scale():.0f}" if mode == "fp16" else ""
        print(f"    {mode}：{dt_s * 1000 / 200:.2f} ms/步  最后一步损失 {loss.item():.4f}  验证准确率 {acc:.1%}{extra}")
      except torch.OutOfMemoryError:
        print(f"    {mode}：显存不足，跳过（GPU 被其他程序占用时会发生）")
        torch.cuda.empty_cache()


PARTS = {1: part1_valley, 2: part2_optimizers, 3: part3_schedules, 4: part4_init, 5: part5_norm,
         6: part6_regularize, 7: part7_clip, 8: part8_precision}

if __name__ == "__main__":
    print(f"PyTorch {torch.__version__}，使用设备 {DEV}")
    for k in [int(a) for a in sys.argv[1:] if a.isdigit()] or list(PARTS):
        PARTS[k]()
    print("\n第 11 章的演示全部完成。")
