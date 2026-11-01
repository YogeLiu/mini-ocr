"""第 4 章　训练循环与优化：配套实验脚本。正文中的“本项目观察”对应本脚本各部分。

    python -m book.part1_foundations.ch04_training.tour              # 全部 13 个部分（CPU 上约 1 分钟）
    python -m book.part1_foundations.ch04_training.tour 2 3          # 只运行第 2、3 部分
    python -m book.part1_foundations.ch04_training.tour 13 --device mps   # 训练部分在 Mac 上跑

部分与正文小节的对应：
    1  完整的训练循环（4.1）              8  梯度裁剪（4.7）
    2  梯度下降与狭长山谷（4.2）           9  初始化（4.8）
    3  动量（4.3）                       10  Dropout 与 train()/eval()（4.9）
    4  Adam（4.4）                       11  检查点续训（4.10）
    5  四种优化器的轨迹对比（4.2–4.4）     12  小批量过拟合检查（4.11）
    6  AdamW 与 L2 正则（4.5）            13  五种训练故障（4.12）
    7  预热与余弦衰减（4.6）

数据：二维"螺旋"三分类（CS231n 风格），600 个训练点、300 个验证点，全部自己生成，不依赖其他章的代码。
第 2–10 部分只做小计算，总在 CPU 上；--device 只影响第 1、12、13 部分的训练。第 11 部分固定用 CPU，
因为"逐位一致"的结论只对同一种设备成立（见正文 4.10.4）。
"""
import math
import os
import sys
import tempfile
import time

# 除非明确要求 --device cuda，否则对 PyTorch 隐藏 GPU，避免 torch.optim 的某些检查悄悄初始化 CUDA。
if "cuda" not in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import torch
import torch.nn as nn
import torch.nn.functional as F

from miniocr.device import device_from_argv
from miniocr.train.checkpoint import load_checkpoint, save_checkpoint
from miniocr.train.optim import SGD, Adam, AdamW, clip_grad_norm, warmup_cosine

OUT = "outputs/book/ch04"
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(min(8, os.cpu_count() or 1))
DEVICE = device_from_argv(default="cpu")


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def r(x, nd=4):
    return [round(v, nd) for v in x.tolist()]


# ====================================================================== 公共部分：数据、模型、训练循环
def spiral(n_per_class, n_classes=3, noise=0.2, seed=0):
    """CS231n 的螺旋数据：每类沿一条螺旋臂分布。返回 X (N, 2) float32、y (N,) int64，按类别排好序。"""
    g = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for k in range(n_classes):
        rad = torch.linspace(0.0, 1.0, n_per_class)
        t = torch.linspace(k * 4.0, (k + 1) * 4.0, n_per_class) + noise * torch.randn(n_per_class, generator=g)
        xs.append(torch.stack([rad * torch.sin(t), rad * torch.cos(t)], 1))
        ys.append(torch.full((n_per_class,), k, dtype=torch.long))
    return torch.cat(xs), torch.cat(ys)


X_TR, Y_TR = spiral(200, seed=0)          # (600, 2)、(600,)
X_VA, Y_VA = spiral(100, seed=1)          # (300, 2)、(300,)


def mlp(hidden=64, depth=2, p_drop=0.0, seed=0):
    """2 → hidden → … → 3 的 MLP，depth 是隐藏层个数。用 PyTorch 默认初始化。"""
    torch.manual_seed(seed)
    layers, d = [], 2
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        if p_drop:
            layers.append(nn.Dropout(p_drop))
        d = hidden
    layers.append(nn.Linear(d, 3))
    return nn.Sequential(*layers)


@torch.no_grad()
def evaluate(model, X, Y):
    """返回 (loss, 正确率)。调用者负责先 model.eval()。"""
    logits = model(X)
    return F.cross_entropy(logits, Y).item(), (logits.argmax(1) == Y).float().mean().item()


def train_loop(model, steps=1000, bs=64, lr=1e-2, warmup=50, wd=0.01, clip=1.0, seed=0, log_every=0,
               device="cpu", X=None, Y=None, zero_grad=True, eval_mode=True, sample="random", opt=None, norms=None):
    """本章的标准训练循环（4.1 节）。返回每一步的 loss 列表和训练后在验证集上的 (loss, 正确率)。
    zero_grad / eval_mode / sample 这几个开关只给第 13 部分复现故障用。
    注意：传入 opt 时，它的学习率每一步都会被这里的 lr（和 warmup 调度）覆盖。
    norms 是一个列表时，记录每步裁剪前的梯度全局范数（clip=0 时也记录）。"""
    X = (X_TR if X is None else X).to(device)
    Y = (Y_TR if Y is None else Y).to(device)
    model.to(device)
    opt = opt or AdamW(model.parameters(), lr=lr, weight_decay=wd)
    g = torch.Generator().manual_seed(seed)
    losses = []
    for step in range(steps):
        model.train()                                                       # ① 训练模式
        if sample == "random":                                              # ② 取一批数据
            idx = torch.randint(0, len(X), (bs,), generator=g)
        else:                                                               # "sequential"：不打乱，按顺序切
            start = (step * bs) % len(X)
            idx = torch.arange(start, start + bs) % len(X)
        xb, yb = X[idx.to(device)], Y[idx.to(device)]
        loss = F.cross_entropy(model(xb), yb)                               # ③ 前向，算 loss
        if zero_grad:
            opt.zero_grad()                                                 # ④ 清零梯度
        loss.backward()                                                     # ⑤ 反向
        if clip:
            n = clip_grad_norm(model.parameters(), clip)                    # ⑥ 裁剪梯度
        elif norms is not None:
            n = clip_grad_norm(model.parameters(), float("inf"))
        if norms is not None:
            norms.append(n)
        opt.lr = warmup_cosine(step, steps, warmup, lr) if warmup is not None else lr   # ⑦ 调学习率
        opt.step()                                                          # ⑧ 更新
        losses.append(loss.item())                                          # ⑨ 记录
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"    第 {step:4d} 步  lr {opt.lr:.5f}  loss {loss.item():.4f}")
    if eval_mode:
        model.eval()
    va = evaluate(model, X_VA.to(device), Y_VA.to(device))
    return losses, va


# ====================================================================== 1
def part1_loop():
    section("1. 完整的训练循环（4.1）：螺旋三分类，MLP 2→64→64→3，AdamW + 预热余弦 + 梯度裁剪")
    print(f"  训练集 X {tuple(X_TR.shape)}，y {tuple(Y_TR.shape)}；验证集 X {tuple(X_VA.shape)}；类别 3 个，每类 200 / 100 个")
    model = mlp()
    print(f"  参数量 {sum(p.numel() for p in model.parameters()):,}")
    model.eval()
    l0, a0 = evaluate(model, X_TR, Y_TR)
    print(f"  训练前：训练集 loss {l0:.4f}（ln 3 = {math.log(3):.4f}），正确率 {a0:.1%}")
    t = time.perf_counter()
    losses, (vl, va) = train_loop(model, steps=1000, log_every=200, device=DEVICE)
    model.eval().cpu()
    tl, ta = evaluate(model, X_TR, Y_TR)
    print(f"  训练后：训练集 loss {tl:.4f} 正确率 {ta:.1%}；验证集 loss {vl:.4f} 正确率 {va:.1%}（{time.perf_counter() - t:.1f} 秒）")
    # 画数据和决策区域
    xx, yy = torch.meshgrid(torch.linspace(-1.1, 1.1, 300), torch.linspace(-1.1, 1.1, 300), indexing="xy")
    with torch.no_grad():
        zz = model(torch.stack([xx.flatten(), yy.flatten()], 1)).argmax(1).reshape(xx.shape)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ax[0].contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5, 2.5], colors=["#d8e6f3", "#f6dcc7", "#d9ecd5"])
    for k, c in enumerate(["#2b6cb0", "#c05621", "#2f855a"]):
        m = Y_TR == k
        ax[0].scatter(X_TR[m, 0], X_TR[m, 1], s=8, c=c, label=f"类 {k}")
    ax[0].set_title("螺旋数据与训练后的决策区域"); ax[0].set_aspect("equal"); ax[0].legend(loc="lower right", fontsize=8)
    ax[1].plot(losses, lw=0.8, color="#2b6cb0"); ax[1].axhline(math.log(3), ls="--", c="gray", lw=0.8)
    ax[1].text(len(losses) * 0.6, math.log(3) + 0.03, "ln 3", color="gray")
    ax[1].set_xlabel("步"); ax[1].set_ylabel("训练 loss（单个批次）"); ax[1].set_title("学习曲线")
    fig.tight_layout(); fig.savefig(f"{OUT}/spiral_loop.png", dpi=130); plt.close(fig)
    print(f"  图：{OUT}/spiral_loop.png")


# ====================================================================== 2-5：二维二次函数
A_DIAG = (1.0, 100.0)           # f(x, y) = ½(x² + 100y²)：y 方向陡、x 方向缓，条件数 100
START = (-4.0, 1.0)
ADAM_GRID = [i / 25 for i in range(1, 51)]                    # 0.04, 0.08, …, 2.00
WIDE_GRID = [i / 10000 for i in range(5, 400, 5)]             # 0.0005, …, 0.0395（动量允许更大的 lr）
LR_GRID = [i / 10000 for i in range(5, 200, 5)] + [0.0199]     # 0.0005, 0.0010, …, 0.0195, 0.0199


def quad(p, rot=0.0):
    """f(p) = ½ pᵀ H p，H = diag(1, 100)。rot ≠ 0 时把 H 的主轴旋转 rot 弧度（第 4 部分：坐标轴不再对齐）。"""
    if rot:
        c, s = math.cos(rot), math.sin(rot)
        R = torch.tensor([[c, -s], [s, c]], dtype=p.dtype)
        q = R.T @ p
    else:
        q = p
    return 0.5 * (A_DIAG[0] * q[0] ** 2 + A_DIAG[1] * q[1] ** 2)


def hess(rot=0.0):
    """f 的 Hessian（二阶导数矩阵）。f 是二次函数，梯度就是 H θ（直接用公式算，比 autograd 快）。"""
    c, s = math.cos(rot), math.sin(rot)
    R = torch.tensor([[c, -s], [s, c]], dtype=torch.float64)
    return R @ torch.diag(torch.tensor(A_DIAG, dtype=torch.float64)) @ R.T


def run(make_opt, steps, start=START, rot=0.0, tol=1e-3, stop_at_hit=False):
    """在 f 上跑 steps 步。返回 (轨迹 (步数+1, 2), 第一次 ‖θ‖ < tol 的步数或 None)。"""
    H = hess(rot)
    p = torch.tensor(start, dtype=torch.float64, requires_grad=True)
    opt = make_opt([p])
    traj, hit = [p.detach().clone()], None
    for t in range(1, steps + 1):
        p.grad = H @ p.detach()                       # ∇f = Hθ，和 quad(p, rot).backward() 结果相同
        opt.step()
        traj.append(p.detach().clone())
        if hit is None and p.detach().norm() < tol:
            hit = t
            if stop_at_hit:
                break
        if not torch.isfinite(p).all() or p.detach().norm() > 1e6:
            break
    return torch.stack(traj), hit


def best_lr(make, steps=3000, grid=LR_GRID, **kw):
    """在学习率网格上找"到达 ‖θ‖ < 1e-3 用的步数"最少的那个。返回 (步数, lr)。"""
    return min(((run(lambda ps: make(ps, lr), steps, stop_at_hit=True, **kw)[1] or 10 ** 9), lr) for lr in grid)


def show_steps(make_opt, n=3, extra=None):
    p = torch.tensor(START, dtype=torch.float64, requires_grad=True)
    opt = make_opt([p])
    print(f"    第 0 步  θ = {r(p.detach())}  f = {quad(p).item():.4f}")
    for t in range(1, n + 1):
        opt.zero_grad()
        f = quad(p)
        f.backward()
        g = p.grad.clone()
        opt.step()
        msg = f"    第 {t} 步  g = {r(g)}"
        if extra:
            msg += "  " + extra(opt)
        print(msg + f"  →  θ = {r(p.detach())}  f = {quad(p).item():.4f}")


def part2_gd():
    section("2. 梯度下降与狭长山谷（4.2）：f(x, y) = ½(x² + 100y²)，从 (−4, 1) 出发")
    print("  梯度 ∇f = (x, 100y)。学习率 0.015，手算前 3 步：")
    show_steps(lambda ps: SGD(ps, lr=0.015))
    print("  每一步 x 乘以 (1 − 0.015×1) = 0.985，y 乘以 (1 − 0.015×100) = −0.5（变号，来回跳）")
    print("  不同学习率（稳定条件 lr < 2/100 = 0.02）：")
    for lr in (0.001, 0.005, 0.01, 0.015, 0.019, 0.0199, 0.021):
        traj, hit = run(lambda ps: SGD(ps, lr=lr), 3000)
        fx = quad(traj[-1]).item()
        state = f"{hit} 步到达 ‖θ‖ < 1e-3" if hit else ("发散" if not math.isfinite(fx) or fx > 1e3 else f"3000 步后 f = {fx:.3g}")
        print(f"    lr = {lr:<7} x 因子 {1 - lr:.4f}  y 因子 {1 - 100 * lr:+.3f}   {state}")
    traj, _ = run(lambda ps: SGD(ps, lr=0.021), 30)
    print(f"    lr = 0.021 时 30 步后 y = {traj[-1][1].item():.4g}（每步乘以 −1.1）")
    n, lr = best_lr(lambda ps, lr: SGD(ps, lr=lr))
    print(f"  学习率网格 0.0005, 0.0010, …, 0.0195, 0.0199 里最快的：{n} 步（lr = {lr}）。x 方向每步最多缩小到 {1 - lr:.4f} 倍，所以快不起来")


def part3_momentum():
    section("3. 动量（4.3）：v ← μv + g，θ ← θ − lr·v")
    print("  μ = 0.9，学习率 0.015，手算前 3 步（v 初始为 0，第一步 v = g）：")
    show_steps(lambda ps: SGD(ps, lr=0.015, momentum=0.9), extra=lambda o: f"v = {r(o.state[0])}")
    for lr in (0.005, 0.015):
        _, hit_gd = run(lambda ps: SGD(ps, lr=lr), 3000)
        _, hit_m = run(lambda ps: SGD(ps, lr=lr, momentum=0.9), 3000)
        print(f"  lr = {lr}：普通梯度下降 {hit_gd} 步到达 ‖θ‖ < 1e-3，动量 μ=0.9 {hit_m} 步")
    print("  动量允许更大的学习率（稳定条件 lr < 2(1+μ)/100），所以网格放宽到 0.0005, 0.0010, …, 0.0395，各挑最好的 lr：")
    for mu in (0.0, 0.5, 0.67, 0.9, 0.99):
        n, lr = best_lr(lambda ps, lr: SGD(ps, lr=lr, momentum=mu), grid=WIDE_GRID)
        print(f"    μ = {mu:<5} 最快 {n} 步（lr = {lr}，稳定上限 {2 * (1 + mu) / A_DIAG[1]:.4f}）")
    lo_, hi_ = A_DIAG
    mu_star = ((math.sqrt(hi_) - math.sqrt(lo_)) / (math.sqrt(hi_) + math.sqrt(lo_))) ** 2
    lr_star = (2 / (math.sqrt(hi_) + math.sqrt(lo_))) ** 2
    _, hit = run(lambda ps: SGD(ps, lr=lr_star, momentum=mu_star), 3000)
    n_grid, lr_grid = best_lr(lambda ps, lr: SGD(ps, lr=lr, momentum=mu_star), grid=WIDE_GRID)
    print(f"    理论最优（Polyak）：κ = {hi_ / lo_:g}，μ* = ((√κ−1)/(√κ+1))² = {mu_star:.4f}，lr* = (2/(√λmax+√λmin))² = {lr_star:.5f}："
          f"用 lr* {hit} 步；μ* 配网格里最好的 lr {n_grid} 步（lr = {lr_grid}）")
    p = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = SGD([p], lr=1.0, momentum=0.9)
    for _ in range(100):
        p.grad = torch.ones(1, dtype=torch.float64)
        opt.step()
    print(f"  梯度恒为 1 时，100 步后速度 v = {opt.state[0].item():.4f}（极限 1/(1−0.9) = 10）")


def part4_adam():
    section("4. Adam（4.4）：每个坐标按自己梯度的大小调整步幅")
    print("  lr = 0.3，β₁ = 0.9，β₂ = 0.999，eps = 1e-8，手算前 3 步：")

    def ex(o):
        t, m, v = o.state[0]
        mh, vh = m / (1 - o.b1 ** t), v / (1 - o.b2 ** t)
        return f"m̂ = {r(mh)}  v̂ = {r(vh, 2)}  步长 = {r(o.lr * mh / (vh.sqrt() + o.eps))}"
    show_steps(lambda ps: Adam(ps, lr=0.3), extra=ex)
    print("  第 1 步：m̂ = g，v̂ = g²，步长 = lr·g/|g| = lr·sign(g)：两个坐标都恰好走 0.3，和梯度大小无关")
    for lr in (0.1, 0.3):
        traj, hit = run(lambda ps: Adam(ps, lr=lr), 3000)
        print(f"  lr = {lr}：{hit} 步到达 ‖θ‖ < 1e-3")
    n, lr = best_lr(lambda ps, lr: Adam(ps, lr=lr), grid=ADAM_GRID)
    print(f"  Adam 在 lr 网格 0.04, 0.08, …, 2.00 里最快：{n} 步（lr = {lr}）")
    rot = math.pi / 4
    c, s_ = math.cos(rot), math.sin(rot)
    start_rot = (c * START[0] - s_ * START[1], s_ * START[0] + c * START[1])    # R @ START：起点跟着山谷一起转
    print(f"  把山谷旋转 45°（主轴不再沿坐标轴），起点也跟着转到 R·(−4, 1) = ({start_rot[0]:.4f}, {start_rot[1]:.4f})，"
          f"初始 f 仍是 {quad(torch.tensor(start_rot, dtype=torch.float64), rot).item():.2f}"
          f"（起点不转的话是 {quad(torch.tensor(START, dtype=torch.float64), rot).item():.2f}）")
    print("  先看同一个学习率下旋转前后的步数（梯度下降和动量的更新规则没有坐标轴，应当完全相同）：")
    for name, mk in (("梯度下降 lr=0.015", lambda ps: SGD(ps, lr=0.015)),
                     ("动量 μ=0.9 lr=0.015", lambda ps: SGD(ps, lr=0.015, momentum=0.9)),
                     ("Adam lr=0.3", lambda ps: Adam(ps, lr=0.3))):
        _, h0 = run(mk, 3000)
        _, h1 = run(mk, 3000, start=start_rot, rot=rot)
        print(f"    {name:20s} 对齐 {h0} 步   旋转 45° {h1} 步")
    print("  再让每种方法各自重新挑最好的学习率：")
    for name, mk, grid in (("梯度下降", lambda ps, lr: SGD(ps, lr=lr), LR_GRID),
                           ("动量 μ=0.67", lambda ps, lr: SGD(ps, lr=lr, momentum=0.67), WIDE_GRID),
                           ("动量 μ=0.9", lambda ps, lr: SGD(ps, lr=lr, momentum=0.9), WIDE_GRID),
                           ("Adam", lambda ps, lr: Adam(ps, lr=lr), ADAM_GRID)):
        n0, l0 = best_lr(mk, grid=grid)
        n1, l1 = best_lr(mk, grid=grid, start=start_rot, rot=rot)
        print(f"    {name:10s} 对齐 {n0} 步（lr={l0}）   旋转 45° {n1} 步（lr={l1}）")
    n2, l2 = best_lr(lambda ps, lr: Adam(ps, lr=lr), grid=[i / 25 for i in range(1, 201)], start=start_rot, rot=rot)
    print(f"    （Adam 旋转后的最好 lr 落在网格边上；把网格扩大到 0.04, 0.08, …, 8.00 再找：{n2} 步，lr = {l2}）")


def part5_compare():
    section("5. 四种优化器的轨迹（4.2–4.4）：同一个山谷，同一个起点，各跑 100 步")
    opts = [("梯度下降 lr=0.015", lambda ps: SGD(ps, lr=0.015), "#2b6cb0"),
            ("梯度下降 lr=0.019", lambda ps: SGD(ps, lr=0.019), "#90cdf4"),
            ("动量 μ=0.9 lr=0.015", lambda ps: SGD(ps, lr=0.015, momentum=0.9), "#c05621"),
            ("Adam lr=0.3", lambda ps: Adam(ps, lr=0.3), "#2f855a")]
    xs, ys = torch.meshgrid(torch.linspace(-4.5, 1.5, 300), torch.linspace(-3.2, 1.6, 300), indexing="xy")
    fig, axes = plt.subplots(2, 1, figsize=(8, 9), gridspec_kw={"height_ratios": [3.2, 4.8]})
    for ax, rot in zip(axes, (0.0, math.pi / 4)):
        c, s = math.cos(rot), math.sin(rot)
        u, w = c * xs + s * ys, -s * xs + c * ys
        zz = 0.5 * (u ** 2 + 100 * w ** 2)
        ax.contour(xs, ys, zz, levels=[0.1, 0.5, 2, 5, 10, 20, 40, 80], colors="#bbbbbb", linewidths=0.6)
        for name, mk, col in opts:
            traj, _ = run(mk, 100, rot=rot)
            ax.plot(traj[:, 0], traj[:, 1], "-o", ms=2, lw=1, color=col, label=name)
            if not rot:
                print(f"  {name:22s} 10 步后 θ = {r(traj[10], 3)}   100 步后 f = {quad(traj[-1]).item():.3g}")
        ax.plot(0, 0, "k*", ms=10)
        ax.set_aspect("equal"); ax.set_xlim(-4.5, 1.5); ax.set_ylim(-1.6 if not rot else -3.2, 1.6)
        ax.set_title("f = ½(x² + 100y²)，各跑 100 步" if not rot else "同一个山谷旋转 45°")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{OUT}/optimizer_paths.png", dpi=130); plt.close(fig)
    print(f"  图：{OUT}/optimizer_paths.png")


# ====================================================================== 6
def part6_adamw():
    section("6. AdamW 与 L2 正则（4.5）")
    print("  (a) SGD（不带动量）：L2 正则和解耦的权重衰减完全相同")
    torch.manual_seed(0)
    p1 = torch.randn(5, dtype=torch.float64, requires_grad=True)
    p2 = p1.detach().clone()
    opt = SGD([p1], lr=0.1, weight_decay=0.05)
    for _ in range(20):
        g = torch.randn(5, dtype=torch.float64)
        p1.grad = g.clone()
        opt.step()
        p2 = p2 * (1 - 0.1 * 0.05) - 0.1 * g           # 解耦写法
    print(f"    20 步后两种写法的最大差 {(p1.detach() - p2).abs().max().item():.1e}")

    print("  (b) Adam：两个参数，都从 1.0 出发，lr = 0.01，λ = 0.1，1000 步")
    print("      参数 A：数据梯度恒为 0（比如一个从没被用到的嵌入行）")
    print("      参数 B：数据梯度在 +10 和 −10 之间交替（平均为 0，但很大）")
    res = {}
    for name, mk in (("Adam + L2   λ=0.1", lambda ps: Adam(ps, lr=0.01, weight_decay=0.1)),
                     ("Adam + L2   λ=0.001", lambda ps: Adam(ps, lr=0.01, weight_decay=0.001)),
                     ("AdamW       λ=0.1", lambda ps: AdamW(ps, lr=0.01, weight_decay=0.1)),
                     ("AdamW       λ=0.001", lambda ps: AdamW(ps, lr=0.01, weight_decay=0.001))):
        p = torch.ones(2, dtype=torch.float64, requires_grad=True)
        opt = mk([p])
        hist = []
        for t in range(1000):
            p.grad = torch.tensor([0.0, 10.0 if t % 2 == 0 else -10.0], dtype=torch.float64)
            opt.step()
            hist.append(p.detach().clone())
        hist = torch.stack(hist)
        res[name] = hist
        print(f"    {name:20s} 第 50 步 A={hist[49, 0]:+.4f} B={hist[49, 1]:+.4f}   第 1000 步 A={hist[-1, 0]:+.4f} B={hist[-1, 1]:+.4f}")
    print(f"    对照：AdamW 每步乘以 (1 − 0.01×0.1)，1000 步后 0.999^1000 = {0.999 ** 1000:.4f}；λ=0.001 时 {(1 - 1e-5) ** 1000:.4f}")
    print("      参数 A 在 Adam + L2 下的细节（数据梯度恒为 0，只剩正则梯度 λθ）：")
    for lr, lam, marks in ((0.01, 0.1, (1, 10, 50, 100, 200)), (0.01, 0.001, (1, 10, 50, 100, 200)),
                           (0.001, 0.1, (1, 100, 1000, 2000))):
        p = torch.ones(1, dtype=torch.float64, requires_grad=True)
        opt = Adam([p], lr=lr, weight_decay=lam)
        vals, first, prev = {}, None, 1.0
        for t in range(1, 20001):
            p.grad = torch.zeros(1, dtype=torch.float64)
            opt.step()
            if t in marks:
                vals[t] = (p.item(), (prev - p.item()) / lr)
            if first is None and abs(p.item()) < 1e-3:
                first = t
            prev = p.item()
        print(f"      lr={lr} λ={lam}：" + "  ".join(f"第 {t} 步 θ={v:.6f}（这一步走了 {d:.3f}·lr）" for t, (v, d) in vals.items())
              + f"；第一次 |θ| < 1e-3 在第 {first} 步")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for name, col in (("Adam + L2   λ=0.1", "#c05621"), ("AdamW       λ=0.1", "#2f855a")):
        ax[0].plot(res[name][:, 0], color=col, label=name.replace("   ", " "))
        ax[1].plot(res[name][:, 1], color=col, label=name.replace("   ", " "), lw=0.8)
    ax[0].set_title("参数 A：数据梯度恒为 0"); ax[1].set_title("参数 B：数据梯度 ±10 交替")
    for a in ax:
        a.set_xlabel("步"); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/adamw_vs_l2.png", dpi=130); plt.close(fig)
    print(f"    图：{OUT}/adamw_vs_l2.png")

    print("  (c) 自己写的 AdamW 和 torch.optim.AdamW 在螺旋 MLP 上逐步对拍（lr=1e-2，λ=0.1，100 步）")
    m1, m2 = mlp(seed=3), mlp(seed=3)
    o1 = AdamW(m1.parameters(), lr=1e-2, weight_decay=0.1)
    o2 = torch.optim.AdamW(m2.parameters(), lr=1e-2, weight_decay=0.1)
    g = torch.Generator().manual_seed(0)
    worst = 0.0
    for _ in range(100):
        idx = torch.randint(0, 600, (64,), generator=g)
        for m, o in ((m1, o1), (m2, o2)):
            o.zero_grad()
            F.cross_entropy(m(X_TR[idx]), Y_TR[idx]).backward()
            o.step()
        worst = max(worst, max((a - b).abs().max().item() for a, b in zip(m1.parameters(), m2.parameters())))
    print(f"    100 步里参数的最大差 {worst:.1e}")


# ====================================================================== 7
SORT_V, SORT_T = 10, 12


class SortNet(nn.Module):
    """第 7 部分用的小 Transformer：嵌入 + 位置向量 → 4 层编码器 → 每个位置 10 分类。"""

    def __init__(self, d=64, layers=4, norm_first=False):
        super().__init__()
        self.emb = nn.Embedding(SORT_V, d)
        self.pos = nn.Parameter(torch.randn(SORT_T, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, batch_first=True, norm_first=norm_first)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d, SORT_V)

    def forward(self, x):
        return self.out(self.enc(self.emb(x) + self.pos))


def _sort_run(lr, warm, seed, norm_first, steps=600, bs=64):
    torch.manual_seed(seed)
    model = SortNet(norm_first=norm_first)
    opt = Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    losses = []
    for step in range(steps):
        x = torch.randint(0, SORT_V, (bs, SORT_T), generator=g)
        y = x.sort(1).values
        loss = F.cross_entropy(model(x).reshape(-1, SORT_V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.lr = warmup_cosine(step, steps, warm, lr)
        opt.step()
        losses.append(loss.item())
    return losses


def part7_schedule():
    section("7. 预热与余弦衰减（4.6）")
    T, W, base, lo = 1000, 100, 1e-3, 1e-4
    print(f"  total = {T}，warmup = {W}，base_lr = {base}，min_lr = {lo}：")
    for s in (0, 1, 49, 99, 100, 325, 550, 775, 999):
        print(f"    第 {s:3d} 步  lr = {warmup_cosine(s, T, W, base, lo):.6f}")
    lrs = [warmup_cosine(s, T, W, base, lo) for s in range(T)]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(lrs, color="#2b6cb0"); ax.axvline(W, ls="--", c="gray", lw=0.8)
    ax.set_xlabel("步"); ax.set_ylabel("学习率"); ax.set_title("预热 100 步 + 余弦衰减到 1e-4")
    fig.tight_layout(); fig.savefig(f"{OUT}/warmup_cosine.png", dpi=130); plt.close(fig)
    print(f"  图：{OUT}/warmup_cosine.png")

    print("  (a) Adam 早期的平均步幅最大：螺旋 MLP（2→64→64→3），恒定 lr = 1e-2，不预热，统计 |Δθ| / lr 的平均值")
    model = mlp(seed=0)
    opt = Adam(model.parameters(), lr=1e-2)
    g = torch.Generator().manual_seed(0)
    marks = {1, 2, 10, 100, 1000}
    for t in range(1, 1001):
        idx = torch.randint(0, 600, (64,), generator=g)
        before = [p.detach().clone() for p in model.parameters()]
        opt.zero_grad()
        F.cross_entropy(model(X_TR[idx]), Y_TR[idx]).backward()
        opt.step()
        if t in marks:
            d = torch.cat([(p.detach() - b).abs().flatten() for p, b in zip(model.parameters(), before)])
            print(f"    第 {t:4d} 步  平均 |Δθ|/lr = {d.mean().item() / 1e-2:.3f}   最大 {d.max().item() / 1e-2:.3f}")
            if t == 1:
                nz = d[d > 0] / 1e-2
                print(f"              其中 |Δθ| 恰好为 0 的参数占 {(d == 0).float().mean().item():.1%}（这一批上梯度为 0，多半是没激活的 ReLU）；"
                      f"其余参数的 |Δθ|/lr 最小 {nz.min().item():.4f}、平均 {nz.mean().item():.4f}")

    print("  (b) 预热真正起作用的地方：一个 4 层 Transformer 编码器（第 9 章才讲，这里当黑盒用 nn.TransformerEncoderLayer）")
    print("      任务：输入 12 个 0–9 的数字，每个位置输出排序后的数字（10 分类，随机猜的 loss = ln 10 = 2.303）")
    print("      Adam lr = 1e-2，600 步，批量 64，之后都余弦衰减到 0；3 个随机种子；表中是末 50 步 loss 平均")
    for norm_first, warm in ((False, 0), (False, 100), (True, 0)):
        tag = ("Pre-LN " if norm_first else "Post-LN") + (f" 预热 {warm:3d} 步" if warm else " 不预热    ")
        outs = []
        for seed in range(3):
            losses = _sort_run(1e-2, warm, seed, norm_first)
            outs.append(losses)
        print(f"    {tag}  " + "  ".join(f"种子{sd}: {sum(l[-50:]) / 50:.3f}" for sd, l in enumerate(outs))
              + "    （种子 0 的第 0/50/100/150/300 步 loss：" + " ".join(f"{outs[0][t]:.2f}" for t in (0, 50, 100, 150, 300)) + "）")
    g = torch.Generator().manual_seed(0)
    ys = torch.randint(0, SORT_V, (200000, SORT_T), generator=g).sort(1).values
    probs = torch.stack([torch.bincount(ys[:, t], minlength=SORT_V).float() / len(ys) for t in range(SORT_T)])
    ent = -(probs * probs.clamp_min(1e-12).log()).sum(1).mean().item()
    print(f"      基线：不看输入，只按\"第 t 个位置通常是几\"来猜，loss = 各位置分布的平均熵 ≈ {ent:.3f}")
    print("      对照：Post-LN、不预热、lr 降到 3e-3，种子 0 末 50 步 loss "
          f"{sum(_sort_run(3e-3, 0, 0, False)[-50:]) / 50:.3f}")

    print("  (c) 恒定学习率 vs 余弦衰减：螺旋 MLP，AdamW lr = 1e-2，1000 步，5 个种子，末 50 步训练 loss 平均")
    for seed in range(5):
        out = []
        for warm in (None, 50):
            losses, (vl, va) = train_loop(mlp(seed=seed), steps=1000, warmup=warm, seed=seed)
            out.append((sum(losses[-50:]) / 50, vl, va))
        print(f"    种子 {seed}：恒定 lr  末段 loss {out[0][0]:.4f} 验证 loss {out[0][1]:.4f} 正确率 {out[0][2]:.1%}"
              f" │ 预热+余弦 {out[1][0]:.4f} {out[1][1]:.4f} {out[1][2]:.1%}")


# ====================================================================== 8
def part8_clip():
    section("8. 梯度裁剪（4.7）")
    a = torch.zeros(2, requires_grad=True); b = torch.zeros(1, requires_grad=True)
    a.grad = torch.tensor([3.0, 4.0]); b.grad = torch.tensor([12.0])
    total = clip_grad_norm([a, b], max_norm=1.0)
    print(f"  两个参数的梯度 [3, 4] 和 [12]：全局范数 √(9+16+144) = {total:.4f}")
    print(f"  max_norm = 1 时缩放因子 1/13 ≈ {1 / 13:.4f}，裁剪后 {r(a.grad)} 和 {r(b.grad)}，新范数 {torch.cat([a.grad, b.grad]).norm():.4f}")
    a.grad = torch.tensor([3.0, 4.0]); b.grad = torch.tensor([12.0])
    total = clip_grad_norm([a, b], max_norm=20.0)
    print(f"  max_norm = 20 时（13 < 20）不变：{r(a.grad)} 和 {r(b.grad)}")
    m1, m2 = mlp(seed=5), mlp(seed=5)
    for m in (m1, m2):
        F.cross_entropy(m(X_TR), Y_TR).backward()
    n1 = clip_grad_norm(m1.parameters(), 0.1)
    n2 = torch.nn.utils.clip_grad_norm_(m2.parameters(), 0.1).item()
    diff = max((p.grad - q.grad).abs().max().item() for p, q in zip(m1.parameters(), m2.parameters()))
    print(f"  和 torch.nn.utils.clip_grad_norm_ 对拍：返回值 {n1:.6f} vs {n2:.6f}，裁剪后梯度最大差 {diff:.1e}")
    norms = []
    model = mlp(seed=0)
    opt = AdamW(model.parameters(), lr=1e-2)
    g = torch.Generator().manual_seed(0)
    for step in range(1000):
        idx = torch.randint(0, 600, (64,), generator=g)
        opt.zero_grad()
        F.cross_entropy(model(X_TR[idx]), Y_TR[idx]).backward()
        norms.append(clip_grad_norm(model.parameters(), 1.0))
        opt.lr = warmup_cosine(step, 1000, 50, 1e-2)
        opt.step()
    s = sorted(norms)
    print(f"  第 1 部分那次训练里的梯度范数：中位数 {s[500]:.3f}，最大 {s[-1]:.3f}（第 {norms.index(s[-1])} 步），"
          f"超过 1.0 被裁剪的步数 {sum(n > 1.0 for n in norms)} / 1000")


# ====================================================================== 9
def part9_init():
    section("9. 初始化（4.8）：10 层、宽 256 的网络，输入 1000 个标准正态向量，看每层输出的标准差")
    torch.manual_seed(0)
    x0 = torch.randn(1000, 256)
    fan = 256
    cases = [("tanh, std=0.01", torch.tanh, 0.01), ("tanh, std=1.0", torch.tanh, 1.0),
             ("tanh, Xavier √(1/256)", torch.tanh, math.sqrt(1 / fan)),
             ("ReLU, Xavier √(1/256)", torch.relu, math.sqrt(1 / fan)),
             ("ReLU, Kaiming √(2/256)", torch.relu, math.sqrt(2 / fan))]
    print(f"    {'设置':24s}" + "".join(f"  第{i}层" for i in (1, 2, 3, 5, 10)) + "   第1层权重梯度范数")
    for name, act, std in cases:
        torch.manual_seed(1)
        Ws = [(torch.randn(fan, fan) * std).requires_grad_() for _ in range(10)]
        h, stats = x0, []
        for W in Ws:
            h = act(h @ W)
            stats.append(h.std().item())
        h.pow(2).mean().backward()
        print(f"    {name:24s}" + "".join(f" {stats[i - 1]:7.3g}" for i in (1, 2, 3, 5, 10)) + f"   {Ws[0].grad.norm().item():.3g}")
    torch.manual_seed(1)
    h = torch.tanh(x0 @ (torch.randn(fan, fan) * 1.0))
    print(f"  tanh、std=1.0 时第 1 层输出里 |h| > 0.99 的比例 {(h.abs() > 0.99).float().mean().item():.1%}（饱和，导数 1−h² 几乎为 0）")
    print("  σ = 0.01 的 8 层 MLP（宽 128、ReLU、偏置为 0）在第一批数据（64 个点）上的梯度大小，和 Adam 的 eps = 1e-8 比较：")
    model = mlp(hidden=128, depth=8, seed=0)
    with torch.no_grad():
        for mod in model:
            if isinstance(mod, nn.Linear):
                mod.weight.normal_(0, 0.01); mod.bias.zero_()
    idx = torch.randint(0, 600, (64,), generator=torch.Generator().manual_seed(0))
    F.cross_entropy(model(X_TR[idx]), Y_TR[idx]).backward()
    lins = [mod for mod in model if isinstance(mod, nn.Linear)]
    print("    层        " + "".join(f"{i:>10d}" for i in range(1, len(lins) + 1)))
    print("    权重 |g| 中位数" + "".join(f"{l.weight.grad.abs().median().item():10.1e}" for l in lins))
    print("    偏置 |g| 中位数" + "".join(f"{l.bias.grad.abs().median().item():10.1e}" for l in lins))
    allg = torch.cat([p.grad.abs().flatten() for p in model.parameters()])
    print(f"    |g| > eps 的参数占 {(allg > 1e-8).float().mean().item():.2%}（共 {allg.numel():,} 个）")
    for gval in (1e-2, 1e-8, 1e-16):
        q = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        o = Adam([q], lr=0.01)
        q.grad = torch.full((1,), gval, dtype=torch.float64)
        o.step()
        print(f"    Adam 第 1 步，lr = 0.01，梯度 {gval:.0e}：步长 {abs(q.item()):.3e}（= lr·g/(|g|+eps)）")
    print("  对螺旋数据训练 8 个隐藏层（宽 128、ReLU）的 MLP，1000 步，只改初始化（偏置全设为 0）：")
    for opt_name in ("AdamW lr=1e-2", "SGD 动量0.9 lr=0.05"):
        for name, std_fn in (("std=0.01", lambda fi: 0.01), ("std=1.0", lambda fi: 1.0),
                             ("Kaiming √(2/fan_in)", lambda fi: math.sqrt(2 / fi))):
            model = mlp(hidden=128, depth=8, seed=0)
            with torch.no_grad():
                for mod in model:
                    if isinstance(mod, nn.Linear):
                        mod.weight.normal_(0, std_fn(mod.in_features)); mod.bias.zero_()
            model.eval()
            l0, _ = evaluate(model, X_TR, Y_TR)
            if opt_name.startswith("AdamW"):
                losses, (vl, va) = train_loop(model, steps=1000, seed=0)
            else:
                opt = SGD(model.parameters(), lr=0.05, momentum=0.9)
                losses, (vl, va) = train_loop(model, steps=1000, lr=0.05, seed=0, opt=opt)
            print(f"    {opt_name:18s} {name:20s} 初始 loss {l0:10.4g}  末 50 步平均训练 loss {sum(losses[-50:]) / 50:9.4g}"
                  f"  验证 loss {vl:9.4g}  验证正确率 {va:.1%}")


# ====================================================================== 10
def part10_dropout():
    section("10. Dropout 与 train()/eval()（4.9）")
    torch.manual_seed(0)
    drop = nn.Dropout(p=0.5)
    x = torch.ones(10)
    drop.train()
    print(f"  训练模式，输入 10 个 1，p = 0.5：{drop(x).tolist()}")
    print(f"  再调用一次（掩码不同）：        {drop(x).tolist()}")
    big = drop(torch.ones(100000))
    print(f"  10 万个 1 经过 Dropout：被置 0 的比例 {(big == 0).float().mean():.4f}，平均值 {big.mean():.4f}（保留的乘以 1/(1−p) = 2）")
    drop.eval()
    print(f"  推理模式：{drop(x).tolist()}")
    model = mlp(p_drop=0.3, seed=0)
    train_loop(model, steps=1000, seed=0)
    print(f"  带 Dropout(0.3) 的螺旋 MLP 训练 1000 步后，module.training：{[m.training for m in model][:3]} …")
    model.eval()
    l_e, a_e = evaluate(model, X_VA, Y_VA)
    model.train()
    runs = [evaluate(model, X_VA, Y_VA) for _ in range(5)]
    print(f"  eval() 下验证集：loss {l_e:.4f} 正确率 {a_e:.1%}（重复多少次都一样）")
    print("  忘了 eval()，连续评 5 次：" + "，".join(f"{a:.1%}" for _, a in runs)
          + f"；loss {min(l for l, _ in runs):.4f}–{max(l for l, _ in runs):.4f}")


# ====================================================================== 11
def _resume_run(total, cut, restore=("opt", "rng", "step"), torch_opt=False, threads=None):
    """训练 total 步；若 cut 不为 None，则在 cut 步后保存、"重启"（新建模型、优化器、生成器，并打乱全局随机数），再加载继续。
    restore 列出加载时恢复哪些东西。返回最终的参数列表。"""
    old_threads = torch.get_num_threads()
    if threads:
        torch.set_num_threads(threads)

    def build(seed):
        model = mlp(p_drop=0.2, seed=seed)
        opt = (torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01) if torch_opt
               else AdamW(model.parameters(), lr=1e-2, weight_decay=0.01))
        return model, opt

    def steps(model, opt, g, a, b):
        for step in range(a, b):
            model.train()
            idx = torch.randint(0, 600, (64,), generator=g)                    # 数据顺序来自生成器 g
            loss = F.cross_entropy(model(X_TR[idx]), Y_TR[idx])                # Dropout 用全局随机数
            opt.zero_grad()
            loss.backward()
            clip_grad_norm(model.parameters(), 1.0)
            lr = warmup_cosine(step, total, 50, 1e-2)
            if torch_opt:
                for pg in opt.param_groups:
                    pg["lr"] = lr
            else:
                opt.lr = lr
            opt.step()

    model, opt = build(0)
    g = torch.Generator().manual_seed(123)
    torch.manual_seed(7)
    if cut is None:
        steps(model, opt, g, 0, total)
    else:
        steps(model, opt, g, 0, cut)
        path = os.path.join(tempfile.mkdtemp(), "ckpt.pt")
        save_checkpoint(path, model, opt, step=cut, generators={"data": g})
        del model, opt, g
        model, opt = build(99)                                                 # "重启"：全新的对象，不同的初始参数
        g = torch.Generator().manual_seed(456)
        torch.manual_seed(2024)                                                # 新进程的全局随机数状态当然不同
        ckpt_step, _ = load_checkpoint(path, model, opt if "opt" in restore else None,
                                       generators={"data": g} if "rng" in restore else None)
        if "rng" not in restore:                                               # 只想不恢复随机数：把刚恢复的全局状态再打乱
            torch.manual_seed(2024)
        start = ckpt_step if "step" in restore else 0
        steps(model, opt, g, start, start + (total - cut))
    torch.set_num_threads(old_threads)
    return [p.detach().clone() for p in model.parameters()]


def part11_checkpoint():
    section("11. 检查点续训（4.10）：200 步不中断 vs 第 100 步保存、重启、加载、再训 100 步（CPU）")
    ref = _resume_run(200, None)

    def cmp(ps):
        eq = all(torch.equal(a, b) for a, b in zip(ref, ps))
        return eq, max((a - b).abs().max().item() for a, b in zip(ref, ps))
    cases = [("全部恢复（模型、优化器、步数、随机数）", dict()),
             ("不恢复优化器状态（m、v、t 从零开始）", dict(restore=("rng", "step"))),
             ("不恢复随机数状态（Dropout 掩码、取数据的顺序都变了）", dict(restore=("opt", "step"))),
             ("不恢复步数（学习率调度从第 0 步重新预热）", dict(restore=("opt", "rng"))),
             ("全部恢复，优化器换成 torch.optim.AdamW（两次都用它）", dict(torch_opt=True))]
    for name, kw in cases:
        if kw.get("torch_opt"):
            base = _resume_run(200, None, torch_opt=True)
            ps = _resume_run(200, 100, torch_opt=True)
            eq = all(torch.equal(a, b) for a, b in zip(base, ps))
            mx = max((a - b).abs().max().item() for a, b in zip(base, ps))
        else:
            eq, mx = cmp(_resume_run(200, 100, **kw))
        print(f"  {name:44s} 逐位一致 {str(eq):5s}  参数最大差 {mx:.2e}")
    a = _resume_run(200, None, threads=1)
    b = _resume_run(200, None, threads=8)
    eq = all(torch.equal(x, y) for x, y in zip(a, b))
    print(f"  同样不中断地训 200 步，只把线程数从 1 改成 8：逐位一致 {eq}，参数最大差 {max((x - y).abs().max().item() for x, y in zip(a, b)):.2e}")
    a = _resume_run(200, None)
    print(f"  同样设置再跑一次不中断的训练：逐位一致 {all(torch.equal(x, y) for x, y in zip(a, ref))}")


# ====================================================================== 12
def part12_overfit():
    section("12. 小批量过拟合检查（4.11）：只拿 32 个样本，反复训练 500 步，看 loss 能不能降到接近 0")
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(600, generator=g)[:32]
    xb, yb = X_TR[idx].to(DEVICE), Y_TR[idx].to(DEVICE)
    yrand = yb[torch.randperm(32, generator=g).to(DEVICE)]
    print(f"  这 32 个样本的类别计数：{torch.bincount(yb.cpu(), minlength=3).tolist()}")

    def fit(loss_fn, y, label, lr=1e-2, steps=500):
        model = mlp(hidden=64, seed=0).to(DEVICE)
        opt = Adam(model.parameters(), lr=lr)
        hist = []
        for t in range(steps):
            opt.zero_grad()
            loss = loss_fn(model(xb), y)
            loss.backward()
            opt.step()
            hist.append(loss.item())
        with torch.no_grad():
            acc = (model(xb).argmax(1) == y).float().mean().item()
        marks = "  ".join(f"{t}:{hist[t]:.4f}" for t in (0, 50, 100, 200, 499))
        print(f"    {label:34s} {marks}   批内正确率 {acc:.0%}")
        return hist
    ce = F.cross_entropy
    fit(ce, yb, "正确代码")
    fit(lambda lg, y: ce(lg.softmax(-1), y), yb, "错误：先 softmax 再 cross_entropy")
    print(f"    （先 softmax 时 loss 的理论下限 ln(e + 2) − 1 = {math.log(math.e + 2) - 1:.4f}）")
    fit(lambda lg, y: ce(lg, y), yb, "错误：学习率 1e-6（几乎没在更新）", lr=1e-6)
    fit(ce, yrand, "标签被打乱（标签和输入对不上）")


# ====================================================================== 13
def part13_faults():
    section("13. 五种训练故障（4.12）：螺旋 MLP，标准设置是 AdamW lr=1e-2、预热 50 步 + 余弦、裁剪 1.0、1000 步")
    t0 = time.perf_counter()

    def report(name, losses, va, extra=""):
        pts = "  ".join(f"{t}:{losses[t]:.3f}" for t in (0, 100, 300, 600, 999) if t < len(losses))
        print(f"  {name}\n      loss {pts}\n      验证集 loss {va[0]:.4f} 正确率 {va[1]:.1%} {extra}")
    losses, va = train_loop(mlp(p_drop=0.0, seed=0), device=DEVICE)
    report("基线（正确代码）", losses, va)

    print("\n  故障 1：学习率太大（SGD，不裁剪，不预热）")
    for lr in (0.5, 5.0, 50.0):
        m = mlp(seed=0)
        losses, va = train_loop(m, lr=lr, warmup=None, clip=0, opt=SGD(m.parameters(), lr=lr), device=DEVICE)
        nan_at = next((i for i, l in enumerate(losses) if not math.isfinite(l)), None)
        report(f"SGD lr={lr}", losses, va, f"第一次出现 nan/inf：{nan_at}")
    for lr in (0.3, 1.0):
        losses, va = train_loop(mlp(seed=0), lr=lr, device=DEVICE)
        report(f"AdamW lr={lr}（有预热和裁剪）", losses, va)

    print("\n  故障 2：忘了 opt.zero_grad()（梯度一直累加）")
    def norm_stats(ns):
        return "裁剪前梯度范数 " + "  ".join(f"{t}:{ns[t]:.3g}" for t in (0, 100, 300, 600, 999))
    ns_ok, ns_bad = [], []
    train_loop(mlp(seed=0), device=DEVICE, norms=ns_ok)
    losses, va = train_loop(mlp(seed=0), zero_grad=False, device=DEVICE, norms=ns_bad)
    report("AdamW lr=1e-2，不清零（有预热和裁剪 1.0）", losses, va)
    print(f"      不清零：{norm_stats(ns_bad)}；超过 1.0 被裁剪 {sum(n > 1.0 for n in ns_bad)} / 1000 步")
    print(f"      对照（正常清零）：{norm_stats(ns_ok)}；超过 1.0 被裁剪 {sum(n > 1.0 for n in ns_ok)} / 1000 步")
    m = mlp(seed=0)
    losses, va = train_loop(m, lr=0.05, warmup=None, clip=0, opt=SGD(m.parameters(), lr=0.05), device=DEVICE)
    report("对照：SGD lr=0.05，正常清零（不预热、不裁剪）", losses, va)
    m = mlp(seed=0)
    ns_sgd = []
    losses, va = train_loop(m, lr=0.05, warmup=None, clip=0, zero_grad=False, opt=SGD(m.parameters(), lr=0.05),
                            device=DEVICE, norms=ns_sgd)
    nan_at = next((i for i, l in enumerate(losses) if not math.isfinite(l)), None)
    report("SGD lr=0.05，不清零（不预热、不裁剪）", losses, va, f"第一次出现 nan/inf：{nan_at}")
    print(f"      {norm_stats(ns_sgd)}")
    m = mlp(seed=0)
    losses, va = train_loop(m, zero_grad=False, opt=SGD(m.parameters(), lr=0.05), device=DEVICE)
    report("反例：只把优化器换成 SGD(lr=0.05)，循环保持默认（lr=1e-2 会覆盖它，另有预热 50 步和裁剪 1.0）", losses, va)

    print("\n  故障 3：评测时忘了 model.eval()（模型带 Dropout(0.5)）")
    m = mlp(p_drop=0.5, seed=0)
    losses, va = train_loop(m, device=DEVICE)
    report("正确：训练后 eval() 再评测", losses, va)
    m.train()
    accs = [evaluate(m, X_VA.to(DEVICE), Y_VA.to(DEVICE))[1] for _ in range(5)]
    print("      同一个模型忘了 eval()，评 5 次的正确率：" + "，".join(f"{a:.1%}" for a in accs))

    print("\n  故障 4：数据没有打乱（训练集按类别排好序：类 0 的 200 个、类 1 的 200 个、类 2 的 200 个；批量 8，按顺序取）")
    print("      每个类别连续占 25 步，然后换下一个类别")
    m = mlp(seed=0)
    losses, va = train_loop(m, steps=1500, bs=8, sample="sequential", device=DEVICE)
    with torch.no_grad():
        pred = m.eval()(X_VA.to(DEVICE)).argmax(1).cpu()
    report("按顺序取，1500 步，预热 + 余弦", losses, va, f"验证集预测的类别计数 {torch.bincount(pred, minlength=3).tolist()}")
    print("      第 0–79 步每批的 loss（| 标出换类别的位置）：")
    print("      " + " ".join(("| " if t and t % 25 == 0 else "") + f"{losses[t]:.2f}" for t in range(80)))
    losses, va = train_loop(mlp(seed=0), steps=1500, bs=8, device=DEVICE)
    report("对照：同样批量 8、1500 步，每步随机取", losses, va)
    for stop in (75, 100, 125, 150):
        m = mlp(seed=0)
        train_loop(m, steps=stop, bs=8, sample="sequential", warmup=None, device=DEVICE)
        with torch.no_grad():
            pred = m.eval()(X_VA.to(DEVICE)).argmax(1).cpu()
        last = Y_TR[torch.arange((stop - 1) * 8, stop * 8) % 600]
        print(f"      恒定 lr、在第 {stop} 步停下：最后一批的类别计数 {torch.bincount(last, minlength=3).tolist()}，"
              f"验证集预测计数 {torch.bincount(pred, minlength=3).tolist()}")

    print("\n  故障 5：标签错位（先打乱数据，再把标签整体错开一位：第 i 个样本配上第 i−1 个的标签）")
    for seed in range(4):
        perm = torch.randperm(600, generator=torch.Generator().manual_seed(seed))
        losses, va = train_loop(mlp(seed=seed), X=X_TR[perm], Y=torch.roll(Y_TR[perm], 1), seed=seed, device=DEVICE)
        report(f"标签错开一位，种子 {seed}", losses, va, f"（随机猜 33.3%，ln 3 = {math.log(3):.4f}）")
    losses, va = train_loop(mlp(seed=0), X=X_TR, Y=torch.roll(Y_TR, 1), device=DEVICE)
    report("对照：数据按类别排序时错开一位（只有 3 个样本的标签变了）", losses, va)
    print(f"\n  第 13 部分共用时 {time.perf_counter() - t0:.1f} 秒")


PARTS = {1: part1_loop, 2: part2_gd, 3: part3_momentum, 4: part4_adam, 5: part5_compare, 6: part6_adamw,
         7: part7_schedule, 8: part8_clip, 9: part9_init, 10: part10_dropout, 11: part11_checkpoint,
         12: part12_overfit, 13: part13_faults}

if __name__ == "__main__":
    for k in [int(a) for a in sys.argv[1:] if a.isdigit()] or list(PARTS):
        PARTS[k]()
    print("\n第 4 章的实验全部完成。")
