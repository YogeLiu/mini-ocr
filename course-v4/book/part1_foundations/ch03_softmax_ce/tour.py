"""第 3 章　概率、softmax 与交叉熵：配套实验脚本。正文里引用的每一个数字都来自这里。

    python -m book.part1_foundations.ch03_softmax_ce.tour             # 全部 10 个部分（CPU 上十几秒）
    python -m book.part1_foundations.ch03_softmax_ce.tour 2 5         # 只运行第 2、5 部分
    python -m book.part1_foundations.ch03_softmax_ce.tour 8 --device mps   # 第 8 部分的训练放到 Mac 上

部分与正文小节的对应：
    1  logits 与 softmax（3.2）              6  (B, T, V) 与 ignore_index（3.7）
    2  溢出：fp32 与 fp16（3.3）             7  初始 loss ≈ ln C（3.8）
    3  最大似然与 NLL（3.4）                 8  动手：softmax 线性分类器（3.9）
    4  交叉熵、熵与 KL（3.5）                9  常见错误（3.10）
    5  梯度 p − y 的数值例子（3.6）          10 和 OCR 的连接：每一步一个交叉熵（3.11）
"""
import math
import os
import sys
import time

# 除非明确传入 --device cuda，否则对 PyTorch 隐藏 GPU（本机 GPU 被别的服务占用）。
if "cuda" not in sys.argv:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from miniocr.device import device_from_argv
from miniocr.nn.losses import (cross_entropy, log_softmax, naive_softmax, numerical_gradient, rel_error,
                               sequence_mean_cross_entropy, softmax, softmax_ce_grad, softmax_loss_vectorized)

OUT = "outputs/book/ch03"
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(min(8, os.cpu_count() or 1))
DEVICE = device_from_argv(default="cpu")   # 只有第 8 部分的训练用它


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def fmt(t, nd=4):
    """把张量打印成嵌套列表，保留 nd 位小数。"""
    if t.dim() == 0:
        return round(t.item(), nd)
    return [fmt(r, nd) for r in t] if t.dim() > 1 else [round(x, nd) for x in t.tolist()]


# 全章共用的 3 类例子：明天的天气，类别 0 = 下雨，1 = 多云，2 = 晴
CLASSES = ["下雨", "多云", "晴"]
Z = torch.tensor([2.0, 1.0, 0.1])
Y = 0                                      # 真实答案：下雨


# ---------------------------------------------------------------------- 1
def part1_logits():
    section("1. logits 与 softmax（3.2）：把任意实数变成概率")
    print(f"  logits z = {fmt(Z)}，形状 {tuple(Z.shape)}，类别 {CLASSES}")
    e = Z.exp()
    print(f"  第 1 步 取指数 e^z = {fmt(e)}，和 {e.sum():.4f}")
    p = e / e.sum()
    print(f"  第 2 步 归一化 p = e^z / Σ = {fmt(p)}，和 {p.sum():.4f}")
    print(f"  miniocr 的 softmax(z) = {fmt(softmax(Z))}；torch.softmax 最大差 {(softmax(Z) - Z.softmax(-1)).abs().max():.1e}")
    print(f"  所有 logits 加 100：softmax(z + 100) = {fmt(softmax(Z + 100))}（不变：只有差值有意义）")
    for s in (0.5, 2.0, 10.0):
        print(f"  logits 乘 {s:4}：softmax({s}·z) = {fmt(softmax(s * Z))}")
    print(f"  logits 的差 z0 − z1 = {Z[0] - Z[1]:.1f}，概率之比 p0 / p1 = {p[0] / p[1]:.4f} = e^1 = {math.e:.4f}")
    two = torch.tensor([1.5, 0.0])
    print(f"  两类时 softmax([1.5, 0]) = {fmt(softmax(two))}；sigmoid(1.5) = {torch.sigmoid(torch.tensor(1.5)):.4f}（二分类的 softmax 就是 sigmoid）")


# ---------------------------------------------------------------------- 2
def part2_overflow():
    section("2. 溢出（3.3）：不减最大值会怎样")
    f32, f16 = torch.finfo(torch.float32), torch.finfo(torch.float16)
    print(f"  fp32 最大 {f32.max:.4e}，ln(最大) = {math.log(f32.max):.2f}：e^z 在 z > {math.log(f32.max):.2f} 时上溢成 inf")
    print(f"  fp16 最大 {f16.max:.0f}，ln(最大) = {math.log(f16.max):.2f}：e^z 在 z > {math.log(f16.max):.2f} 时上溢成 inf")
    print(f"  fp32 最小正规数 {f32.tiny:.3e}，最小次正规数 {2.0 ** -149:.3e}（ln = {math.log(2.0 ** -149):.2f}）；"
          f"就近舍入下，比它的一半 2^-150 还小才舍入成 0：分界 ln(2^-150) = {math.log(2.0 ** -150):.2f}")
    zb = torch.tensor([-103.0, -103.5, -103.9, -103.97, -104.0])
    print(f"    实测 fp32 e^z：z = {zb.tolist()} → {[f'{v:.2e}' for v in zb.exp().tolist()]}")
    print()
    z = torch.tensor([1000.0, 999.0, 998.0])
    print(f"  【fp32】 z = {z.tolist()}")
    print(f"    e^z = {z.exp().tolist()}")
    print(f"    naive_softmax = {naive_softmax(z).tolist()}")
    print(f"    稳定 softmax（先减最大值 1000 → {(z - z.max()).tolist()}）= {fmt(softmax(z))}")
    z = torch.tensor([90.0, 88.0, 86.0])
    print(f"  【fp32】 z = {z.tolist()}：e^90 = {z.exp()[0].item()}，naive = {naive_softmax(z).tolist()}，稳定 = {fmt(softmax(z))}")
    z16 = torch.tensor([12.0, 10.0, 8.0], dtype=torch.float16)
    print(f"  【fp16】 z = {z16.tolist()}")
    print(f"    e^z = {z16.exp().tolist()}")
    print(f"    naive_softmax = {naive_softmax(z16).tolist()}")
    print(f"    稳定 softmax = {fmt(softmax(z16).float())}；同样的数在 fp32 里 naive = {fmt(naive_softmax(z16.float()))}")
    zn = torch.tensor([-1000.0, -1001.0, -1002.0])
    print(f"  【下溢】 z = {zn.tolist()}：e^z = {zn.exp().tolist()}，naive = {naive_softmax(zn).tolist()}（0/0），稳定 = {fmt(softmax(zn))}")
    zl = torch.tensor([0.0, -200.0])
    print(f"  【log 的下溢】 z = {zl.tolist()}，fp32：")
    print(f"    softmax(z) = {softmax(zl).tolist()}  →  log(softmax(z)) = {softmax(zl).log().tolist()}")
    print(f"    log_softmax(z) = {log_softmax(zl).tolist()}（精确值是 [−ln(1 + e^−200), −200 − ln(1 + e^−200)] ≈ [0, −200]）")
    print(f"    浮点舍入让 p_0 恰好等于 1：cross_entropy([[0, −200]], 标签 0) = {cross_entropy(zl[None], torch.tensor([0])).item()}")
    t = torch.tensor([0.0, -200.0], requires_grad=True)
    softmax(t).log()[1].backward()
    print(f"    对 log(softmax(z))[1] 求梯度：{t.grad.tolist()}")
    t2 = torch.tensor([0.0, -200.0], requires_grad=True)
    log_softmax(t2)[1].backward()
    print(f"    对 log_softmax(z)[1] 求梯度：{t2.grad.tolist()}（精确值 onehot(1) − p = [0 − 1, 1 − 0] = [−1, 1]）")
    zv = torch.zeros(1, 65536, dtype=torch.float16)
    m_ = zv.amax(-1, keepdim=True)
    print(f"  【fp16 的求和溢出】 z = 全 0，形状 (1, 65536)：减最大值后每项都是 1，但 Σ = {(zv - m_).exp().sum().item()}（fp16 上限 65504）")
    print(f"    只在 fp16 里减最大值：softmax 第一项 = {((zv - m_).exp() / (zv - m_).exp().sum(-1, keepdim=True))[0, 0].item()}，"
          f"log-softmax 第一项 = {((zv - m_) - (zv - m_).exp().sum(-1, keepdim=True).log())[0, 0].item()}")
    print(f"    miniocr（fp32 中间计算再转回 fp16）：softmax 第一项 = {softmax(zv)[0, 0].item():.4e}（1/65536 = {1 / 65536:.4e}），"
          f"log_softmax 第一项 = {log_softmax(zv)[0, 0].item():.4f}（−ln 65536 = {-math.log(65536):.4f}）")
    zi = torch.full((3,), -math.inf)
    print(f"  【全是 −inf】 z = {zi.tolist()}：m = −inf，z − m = {(zi - zi.amax()).tolist()}；torch.logsumexp(z) = {torch.logsumexp(zi, 0).item()}")
    r = torch.randn(4, 1000, generator=torch.Generator().manual_seed(0)) * 30
    print(f"  随机 logits（4 × 1000，标准差 30）：稳定 softmax 与 torch.softmax 最大差 {(softmax(r) - r.softmax(-1)).abs().max():.1e}，"
          f"log_softmax 最大差 {(log_softmax(r) - r.log_softmax(-1)).abs().max():.1e}")


# ---------------------------------------------------------------------- 3
def part3_likelihood():
    section("3. 最大似然与 NLL（3.4）")
    print("  例 1：抛一枚不均匀的硬币 10 次，7 次正面、3 次反面。正面概率 θ 取多少最能解释这组数据？")
    for th in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9):
        L = th ** 7 * (1 - th) ** 3
        nll = -(7 * math.log(th) + 3 * math.log(1 - th))
        print(f"    θ = {th:.1f}：似然 θ^7(1−θ)^3 = {L:.6f}   NLL = {nll:.4f}   平均 NLL = {nll / 10:.4f}")
    grid = torch.linspace(0.01, 0.99, 9801, dtype=torch.float64)
    nll = -(7 * grid.log() + 3 * (1 - grid).log())
    print(f"    在 0.01..0.99 的网格上，NLL 最小的 θ = {grid[nll.argmin()]:.4f}（数学结论：θ* = 7/10）")
    print()
    print("  例 2：两位天气预报员 5 天的预报，表里是他们给\"实际发生的天气\"的概率")
    A = torch.tensor([0.7, 0.6, 0.8, 0.5, 0.9], dtype=torch.float64)
    B = torch.tensor([0.9, 0.2, 0.95, 0.3, 0.99], dtype=torch.float64)
    for name, p in (("预报员甲", A), ("预报员乙", B)):
        print(f"    {name} p = {p.tolist()}：似然（连乘）= {p.prod():.5f}，对数似然 = {p.log().sum():.4f}，"
              f"平均 NLL = {-p.log().mean():.4f}")
    print(f"    每一天的 −ln p：甲 {fmt(-A.log())}；乙 {fmt(-B.log())}")
    print()
    print("  例 3：连乘会下溢。每个样本的概率都是 0.5：")
    for n in (100, 150, 1000, 1100):
        p32 = torch.full((n,), 0.5).prod().item()
        p64 = torch.full((n,), 0.5, dtype=torch.float64).prod().item()
        print(f"    N = {n:5d}：fp32 连乘 {p32:.3e}，fp64 连乘 {p64:.3e}，对数似然之和 = {n * math.log(0.5):.2f}")
    print()
    print("  例 4：预测对了，loss 也不是 0。−ln p 随正确类别概率 p 的变化：")
    for p in (1.0, 0.99, 0.9, 0.7, 0.5, 1 / 3, 0.1, 0.01, 1e-4):
        print(f"    p = {p:<8.4g} → −ln p = {abs(-math.log(p)):.4f}")


# ---------------------------------------------------------------------- 4
def part4_cross_entropy():
    section("4. 交叉熵、熵与 KL（3.5）")
    p = softmax(Z)
    y = torch.tensor([1.0, 0.0, 0.0])
    H = lambda q: -(q * q.clamp_min(1e-30).log()).sum()
    CE = lambda q, r: -(q * r.log()).sum()
    print(f"  模型分布 p = {fmt(p)}（来自第 1 部分），one-hot 标签 y = {y.tolist()}")
    print(f"  H(y, p) = −Σ y_i ln p_i = −ln p_0 = {CE(y, p):.4f}")
    print(f"  cross_entropy(z, 0) = {cross_entropy(Z[None], torch.tensor([0])):.4f}；"
          f"F.cross_entropy = {F.cross_entropy(Z[None], torch.tensor([0])):.4f}")
    q = torch.tensor([0.7, 0.2, 0.1])
    print(f"  真实分布 q = {fmt(q)}（比如\"同样的云图，过去 70% 下了雨\"）：")
    print(f"    熵 H(q) = {H(q):.4f}；交叉熵 H(q, p) = {CE(q, p):.4f}；KL(q‖p) = H(q, p) − H(q) = {CE(q, p) - H(q):.4f}")
    print(f"    如果 p = q：H(q, q) = {CE(q, q):.4f} = H(q)，KL = 0。交叉熵的最小值是 H(q)，不是 0")
    print(f"  均匀分布的熵：C = 3 → ln 3 = {math.log(3):.4f}；C = 10 → {math.log(10):.4f}；C = 50257 → {math.log(50257):.4f}")
    print("  诚实报告的预报员得分最好吗？真实分布 q = [0.7, 0.2, 0.1]，比较几种\"报出去的\"分布 r：")
    for r in ([0.7, 0.2, 0.1], [0.8, 0.15, 0.05], [0.6, 0.25, 0.15], [0.98, 0.01, 0.01]):
        r = torch.tensor(r)
        print(f"    r = {fmt(r, 2)}：期望 −ln 罚分 H(q, r) = {CE(q, r):.4f}；期望线性罚分 Σ q_k(1 − r_k) = {(q * (1 - r)).sum():.4f}")
    print("    （−ln 罚分在 r = q 时最小；线性罚分却鼓励把概率全押在最可能的那一类上）")
    ls = 0.1
    ys = (1 - ls) * y + ls / 3
    print(f"  标签平滑 ε = {ls}：y' = {fmt(ys)}，H(y', p) = {CE(ys, p):.4f}；"
          f"F.cross_entropy(label_smoothing=0.1) = {F.cross_entropy(Z[None], torch.tensor([0]), label_smoothing=0.1):.4f}")


# ---------------------------------------------------------------------- 5
def part5_gradient():
    section("5. 梯度 ∂L/∂z = p − y（3.6）：3 类的数值例子")
    p = softmax(Z)
    y = F.one_hot(torch.tensor(Y), 3).float()
    print(f"  z = {fmt(Z)}，正确类别 y = {Y}（{CLASSES[Y]}），one-hot {y.tolist()}")
    print(f"  p = softmax(z) = {fmt(p)}；L = −ln p_0 = {-p[0].log():.4f}")
    J = torch.diag(p) - p[:, None] * p[None, :]
    print("  softmax 的雅可比 ∂p_i/∂z_j = p_i(δ_ij − p_j)：")
    for i in range(3):
        print(f"    行 {i}: {fmt(J[i])}")
    print(f"  ∂L/∂p = −y / p = [−1/{p[0]:.4f}, 0, 0] = [{-1 / p[0]:.4f}, 0, 0]")
    g_chain = J.T @ torch.tensor([-1 / p[0], 0.0, 0.0])
    print(f"  链式法则 (∂p/∂z)ᵀ · ∂L/∂p = {fmt(g_chain)}")
    print(f"  公式 p − y = {fmt(p - y)}；各分量之和 = {(p - y).sum():.1e}")
    z = Z.clone().requires_grad_()
    L = -log_softmax(z)[Y]
    L.backward()
    print(f"  autograd：z.grad = {fmt(z.grad)}，与 p − y 最大差 {(z.grad - (p - y)).abs().max():.1e}")
    zd = Z.double()
    num = numerical_gradient(lambda t: -log_softmax(t)[Y], zd.clone())
    print(f"  数值梯度（float64，h = 1e-5）= {fmt(num, 6)}，相对误差 {rel_error(num, (softmax(zd) - y.double())):.1e}")
    print(f"  走一小步：z − 0.5·(p − y) = {fmt(Z - 0.5 * (p - y))}，新 p_0 = {softmax(Z - 0.5 * (p - y))[0]:.4f}（原来 {p[0]:.4f}），"
          f"新 L = {-log_softmax(Z - 0.5 * (p - y))[0]:.4f}")
    print("  对比：如果用均方误差 Σ(p − y)² 当损失（3.6.6）：")
    for zz in ([2.0, 1.0, 0.1], [-5.0, 5.0, 0.0], [-10.0, 10.0, 0.0]):
        zt = torch.tensor(zz, requires_grad=True)
        (-log_softmax(zt)[0]).backward()
        g_ce = zt.grad.clone()
        zt.grad = None
        ((softmax(zt) - y) ** 2).sum().backward()
        print(f"    z = {zz}：p_0 = {softmax(torch.tensor(zz))[0]:.2e}；交叉熵的梯度 {fmt(g_ce)}；均方误差的梯度 {fmt(zt.grad, 6)}")
    zb = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.0]])
    yb = torch.tensor([0, 2])
    g = softmax_ce_grad(zb, yb)
    zt = zb.clone().requires_grad_()
    F.cross_entropy(zt, yb).backward()
    print(f"  批量 N = 2：softmax_ce_grad = {fmt(g)}；F.cross_entropy 的 autograd 最大差 {(g - zt.grad).abs().max():.1e}（注意除以 N）")


# ---------------------------------------------------------------------- 6
def part6_sequences():
    section("6. (B, T, V) 与 ignore_index（3.7）")
    PAD = -100
    B, T, V = 2, 4, 5
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(B, T, V, generator=g)
    targets = torch.tensor([[1, 4, 2, 0], [3, 3, PAD, PAD]])
    print(f"  logits 形状 {tuple(logits.shape)} = (B, T, V)；targets 形状 {tuple(targets.shape)}：{targets.tolist()}")
    lp_small = torch.tensor([[-0.4, -1.4, -2.3], [-2.0, -0.2, -3.0]])
    idx = torch.tensor([0, 2])
    print(f"  gather 小例子：logp 形状 (2, 3) = {fmt(lp_small, 2)}，targets = {idx.tolist()}")
    print(f"    logp.gather(-1, targets.unsqueeze(-1)) 形状 {tuple(lp_small.gather(-1, idx[:, None]).shape)} = {fmt(lp_small.gather(-1, idx[:, None]), 2)}")
    nll = cross_entropy(logits, targets, reduction="none")
    print(f"  每个位置的 −ln p（reduction='none'，形状 {tuple(nll.shape)}）：")
    for b in range(B):
        print(f"    序列 {b}: {fmt(nll[b])}")
    valid = targets != PAD
    tok = cross_entropy(logits, targets)
    seq = sequence_mean_cross_entropy(logits, targets)
    print(f"  有效 token 数 {valid.sum().item()}；按 token 平均 = {nll.sum():.4f} / {valid.sum().item()} = {tok:.4f}")
    print(f"  按序列平均 = ({nll[0].sum():.4f}/4 + {nll[1].sum():.4f}/2) / 2 = {seq:.4f}")
    ref = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), ignore_index=PAD)
    ref2 = F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=PAD)
    print(f"  F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1)) = {ref:.4f}；"
          f"F.cross_entropy(logits.transpose(1, 2), targets) = {ref2:.4f}")
    wrong = nll.sum() / (B * T)
    print(f"  错误 1：把 ignore 位置当 0 但除以全部 B·T = {B * T}：{wrong:.4f}（被稀释，偏小）")
    t0 = targets.clone(); t0[t0 == PAD] = 0
    print(f"  错误 2：补齐位置填了 token 0 又没告诉 loss：{cross_entropy(logits, t0):.4f}（模型被要求在补齐处预测 token 0）")
    print("  全部位置都被忽略（targets 全是 −100，logits 形状 (3, 4)）：")
    za = torch.randn(3, 4, generator=g)
    ya = torch.full((3,), PAD)
    for r_ in ("mean", "sum", "none"):
        print(f"    reduction='{r_}'：miniocr {fmt(cross_entropy(za, ya, reduction=r_))}，F.cross_entropy {fmt(F.cross_entropy(za, ya, reduction=r_))}")
    zg = za.clone().requires_grad_()
    la = cross_entropy(zg, ya)
    la.backward()
    print(f"    'mean' 的 loss = {la.item()}，但对 logits 的梯度全是 0：最大 |grad| = {zg.grad.abs().max().item()}")
    print()
    print("  一个对比鲜明的例子：序列甲 8 个 token、每个 loss 0.5；序列乙 2 个 token、每个 loss 3.0")
    print(f"    按 token 平均 = (8×0.5 + 2×3.0) / 10 = {(8 * 0.5 + 2 * 3.0) / 10:.2f}；按序列平均 = (0.5 + 3.0) / 2 = {(0.5 + 3.0) / 2:.2f}")
    print("  梯度上的区别：按 token 平均时每个有效 token 的权重是 1/N_tok；按序列平均时序列 b 的每个 token 权重是 1/(B·n_b)")
    lt = logits.clone().requires_grad_()
    cross_entropy(lt, targets).backward()
    w_tok = lt.grad.abs().sum(-1)
    ls_ = logits.clone().requires_grad_()
    sequence_mean_cross_entropy(ls_, targets).backward()
    w_seq = ls_.grad.abs().sum(-1)
    print(f"    每个位置梯度的 L1 范数 / (2·(1 − p_y))，即该位置的权重：")
    p_y = softmax(logits).gather(-1, targets.clamp_min(0)[..., None]).squeeze(-1)
    print(f"    按 token：{fmt(torch.where(valid, w_tok / (2 * (1 - p_y)), torch.zeros(()))) }")
    print(f"    按序列：{fmt(torch.where(valid, w_seq / (2 * (1 - p_y)), torch.zeros(())))}")
    print("    （补齐位置的梯度精确为 0："
          f"{lt.grad[1, 2:].abs().max().item()}）")


# ---------------------------------------------------------------------- 7
def part7_init_loss():
    section("7. 初始 loss ≈ ln C（3.8）")
    g = torch.Generator().manual_seed(0)
    D, N = 64, 2048
    X = torch.randn(N, D, generator=g)
    print(f"  线性模型 scores = X W，X 形状 ({N}, {D})，每个元素 ~ N(0, 1)；标签随机")
    print(f"  {'C':>6s} {'ln C':>8s} │ " + "  ".join(f"W 标准差 {s:<5g}" for s in (0.001, 0.01, 0.1, 1.0)))
    for C in (3, 10, 100, 1000):
        y = torch.randint(0, C, (N,), generator=g)
        row = []
        for s in (0.001, 0.01, 0.1, 1.0):
            W = torch.randn(D, C, generator=g) * s
            row.append(cross_entropy(X @ W, y).item())
        print(f"  {C:6d} {math.log(C):8.4f} │ " + "  ".join(f"{v:14.4f}" for v in row))
    print("  W 标准差 1.0 时 scores 的标准差约为 √64 = 8：softmax 很尖，随机标签下 loss 远大于 ln C")
    C = 10
    y = torch.randint(0, C, (N,), generator=g)
    W = torch.zeros(D, C)
    b = torch.zeros(C); b[3] = 5.0
    print(f"  C = 10，W = 0，但偏置 b[3] = 5（模型从一开始就偏爱类别 3）：loss = {cross_entropy(X @ W + b, y):.4f}，ln 10 = {math.log(10):.4f}")
    freq = torch.tensor([0.5] + [0.5 / 9] * 9)
    N2 = 200_000
    y2 = torch.multinomial(freq, N2, replacement=True, generator=g)
    print(f"  类别不均衡（类别 0 占 50%）时，只学会\"类别频率\"的模型 loss = H(频率) = {-(freq * freq.log()).sum():.4f}，"
          f"在 {N2} 个样本上实测 {cross_entropy((freq.log())[None].expand(N2, C), y2):.4f}；比 ln 10 小")


# ---------------------------------------------------------------------- 8
def make_blobs(n_per_class, generator):
    """3 类二维高斯点云，中心在半径 2 的圆上，标准差 0.8。返回 X (3n, 2)、y (3n,)。"""
    centers = torch.tensor([[2.0, 0.0], [-1.0, 1.732], [-1.0, -1.732]])
    X = torch.cat([c + 0.8 * torch.randn(n_per_class, 2, generator=generator) for c in centers])
    y = torch.arange(3).repeat_interleave(n_per_class)
    return X, y


def add_bias(X):
    """在最后补一列 1，偏置就并进了 W 的最后一行（CS231n 的"偏置技巧"）。"""
    return torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype, device=X.device)], 1)


def make_spiral(n_per_class, generator):
    """CS231n 的三臂螺旋：线性分类器分不开的数据。返回 X (3n, 2)、y (3n,)。"""
    Xs, ys = [], []
    for c in range(3):
        r = torch.linspace(0.0, 1.0, n_per_class)
        t = torch.linspace(c * 4.0, (c + 1) * 4.0, n_per_class) + 0.2 * torch.randn(n_per_class, generator=generator)
        Xs.append(torch.stack([r * t.sin(), r * t.cos()], 1) * 3)
        ys.append(torch.full((n_per_class,), c))
    return torch.cat(Xs), torch.cat(ys)


def train_softmax(Xt, yt, lr, steps, reg, log_steps=(), seed=1):
    """全批量梯度下降训练线性 softmax 分类器。返回 (W, 每一步的 loss 列表)。"""
    W = (torch.randn(Xt.shape[1], 3, generator=torch.Generator().manual_seed(seed)) * 0.01).to(Xt.device)
    hist = []
    for step in range(steps + 1):
        loss, dW = softmax_loss_vectorized(W, Xt, yt, reg=reg)
        hist.append(loss.item())
        if step in log_steps:
            acc = ((Xt @ W).argmax(1) == yt).float().mean().item()
            print(f"    第 {step:3d} 步  loss {loss.item():.4f}  训练正确率 {acc:6.1%}")
        if step < steps:
            W = W - lr * dW
    return W, hist


def part8_classifier():
    section("8. 动手：softmax 线性分类器（3.9）")
    g = torch.Generator().manual_seed(0)
    Xtr, ytr = make_blobs(100, g)
    Xva, yva = make_blobs(100, g)
    Xtr, Xva = add_bias(Xtr), add_bias(Xva)
    print(f"  合成数据：3 类二维点云；训练集 X {tuple(Xtr.shape)}（最后一列是 1），验证集 {tuple(Xva.shape)}")

    print("  (a) 梯度检查（float64，reg = 0.1，W 随机）：")
    Wc = (torch.randn(3, 3, generator=g) * 0.01).double()
    Xd, yd = Xtr.double(), ytr
    loss, dW = softmax_loss_vectorized(Wc, Xd, yd, reg=0.1)
    num = numerical_gradient(lambda W: softmax_loss_vectorized(W, Xd, yd, reg=0.1)[0], Wc.clone())
    print(f"    loss = {loss.item():.6f}（ln 3 = {math.log(3):.6f}，差异来自正则项和 W 不为 0）")
    for i in range(3):
        print(f"    解析 dW[{i}] = {fmt(dW[i], 6)}    数值 = {fmt(num[i], 6)}")
    print(f"    最大相对误差 {rel_error(dW, num):.1e}")
    Wa = Wc.clone().requires_grad_()
    (F.cross_entropy(Xd @ Wa, yd) + 0.1 * (Wa ** 2).sum()).backward()
    print(f"    和 autograd（F.cross_entropy + L2）比：loss 差 {abs(loss.item() - (F.cross_entropy(Xd @ Wc, yd) + 0.1 * (Wc ** 2).sum()).item()):.1e}，"
          f"梯度最大差 {(Wa.grad - dW).abs().max():.1e}")
    bad = dW.clone(); bad[:, 0] *= 1.01
    print(f"    故意把解析梯度的第 0 列放大 1%：相对误差变成 {rel_error(bad, num):.1e}")
    bad = dW.clone(); bad -= 2 * 0.1 * Wc
    print(f"    故意漏掉正则项的梯度 2·reg·W：相对误差变成 {rel_error(bad, num):.1e}")

    hs = (1e-1, 1e-3, 1e-5, 1e-7, 1e-9)
    print("  (b) 数值梯度的步长 h：W 的同一个点，和解析梯度（float64）比")
    for dt in (torch.float64, torch.float32):
        rel, ab = [], []
        for h in hs:
            num_h = numerical_gradient(lambda W: softmax_loss_vectorized(W, Xd.to(dt), yd, reg=0.1)[0], Wc.to(dt).clone(), h=h)
            rel.append(rel_error(num_h.double(), dW)); ab.append((num_h.double() - dW).abs().max().item())
        print(f"    {str(dt):14s} 最大相对误差  " + "  ".join(f"h={h:.0e}: {e:.1e}" for h, e in zip(hs, rel)))
        print(f"    {'':14s} 最大绝对误差  " + "  ".join(f"h={h:.0e}: {e:.1e}" for h, e in zip(hs, ab)))
    print(f"    dW 里绝对值最小的元素 {dW.abs().min():.1e}：相对误差的最大值往往出在这种接近 0 的元素上")

    lr, steps, reg = 0.1, 300, 1e-3
    print(f"  (c) 训练：全批量梯度下降，学习率 {lr}，reg = {reg}，{steps} 步，设备 {DEVICE}")
    Xt, yt, Xv, yv = Xtr.to(DEVICE), ytr.to(DEVICE), Xva.to(DEVICE), yva.to(DEVICE)
    t0 = time.perf_counter()
    W, hist = train_softmax(Xt, yt, lr, steps, reg, log_steps=(0, 1, 5, 20, 50, 100, 300))
    dt_ = time.perf_counter() - t0
    va_loss = cross_entropy(Xv @ W, yv).item()
    va_acc = ((Xv @ W).argmax(1) == yv).float().mean().item()
    print(f"    初始 loss {hist[0]:.4f}，ln 3 = {math.log(3):.4f}")
    print(f"    验证集 loss {va_loss:.4f}，验证正确率 {va_acc:.1%}；训练用时 {dt_:.2f} 秒")
    W = W.cpu()
    p_va = softmax(Xva @ W)
    conf = p_va.max(1).values
    print(f"    验证集上最大概率的平均值 {conf.mean():.3f}；最大概率 < 0.6 的点有 {(conf < 0.6).sum().item()} 个")

    print("  (d) 同样的分类器换到三臂螺旋数据上（线性分不开）：")
    gs = torch.Generator().manual_seed(0)
    Xs_tr, ys_tr = make_spiral(100, gs)
    Xs_tr = add_bias(Xs_tr)
    Ws, hs_ = train_softmax(Xs_tr.to(DEVICE), ys_tr.to(DEVICE), lr, 1000, reg, log_steps=(0, 1000))
    Ws = Ws.cpu()
    print(f"    1000 步后训练 loss {hs_[-1]:.4f}，训练正确率 {((Xs_tr @ Ws).argmax(1) == ys_tr).float().mean():.1%}"
          f"（随机猜 33.3%）——本次配置下的结果；直线分不开螺旋，要分开它需要非线性的隐藏层（第 2 章的两层网络）")

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(hist, label="blobs"); ax[0].plot(hs_[:steps + 1], label="spiral")
    ax[0].axhline(math.log(3), ls="--", c="gray"); ax[0].text(steps * 0.6, math.log(3) + 0.02, "ln 3", color="gray")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("train loss"); ax[0].legend()
    for a, WW, XX, yy_, title in ((ax[1], W, Xva, yva, "blobs: validation + decision regions"),
                                  (ax[2], Ws, Xs_tr, ys_tr, "spiral: train + decision regions")):
        xx, yy = torch.meshgrid(torch.linspace(-4.5, 4.5, 200), torch.linspace(-4.5, 4.5, 200), indexing="xy")
        grid = add_bias(torch.stack([xx.reshape(-1), yy.reshape(-1)], 1))
        a.contourf(xx, yy, (grid @ WW).argmax(1).reshape(200, 200), alpha=0.25, levels=[-0.5, 0.5, 1.5, 2.5], cmap="viridis")
        a.scatter(XX[:, 0], XX[:, 1], c=yy_, s=8, cmap="viridis"); a.set_title(title)
    fig.tight_layout(); fig.savefig(f"{OUT}/softmax_classifier.png", dpi=120); plt.close(fig)
    print(f"  图保存在 {OUT}/softmax_classifier.png（左：loss 曲线；中：点云；右：螺旋）")


# ---------------------------------------------------------------------- 9
def part9_pitfalls():
    section("9. 常见错误（3.10）")
    z = Z[None]
    t = torch.tensor([0])
    print("  (a) 先 softmax 再交给 F.cross_entropy（它内部还会再做一次 log_softmax）：")
    p = z.softmax(-1)
    print(f"    正确 F.cross_entropy(z) = {F.cross_entropy(z, t):.4f}；错误 F.cross_entropy(softmax(z)) = {F.cross_entropy(p, t):.4f}")
    for C in (3, 10, 1000):
        floor = -math.log(math.e / (math.e + C - 1))
        print(f"    C = {C:4d}：即使模型完全确信且正确（p = one-hot），错误写法的 loss 也不低于 −ln(e/(e+C−1)) = {floor:.4f}")
    zz = z.clone().requires_grad_()
    F.cross_entropy(zz.softmax(-1), t).backward()
    z2 = z.clone().requires_grad_()
    F.cross_entropy(z2, t).backward()
    print(f"    梯度：正确 {fmt(z2.grad[0])}，错误 {fmt(zz.grad[0])}（被压小了）")
    print("  (b) 把 log_softmax 的结果交给 F.cross_entropy：")
    print(f"    F.cross_entropy(log_softmax(z)) = {F.cross_entropy(z.log_softmax(-1), t):.4f}（碰巧没错：log_softmax 再做一次不变）")
    print("  (c) 把 logits 直接交给 F.nll_loss（它假定输入已经是 log 概率）：")
    print(f"    F.nll_loss(z) = {F.nll_loss(z, t):.4f}（= −z_0，可以是负数）；F.nll_loss(log_softmax(z)) = {F.nll_loss(z.log_softmax(-1), t):.4f}")
    print("  (d) (B, T, V) 直接交给 F.cross_entropy：")
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 4, 5, generator=g)
    tg = torch.randint(0, 5, (2, 4), generator=g)
    try:
        F.cross_entropy(logits, tg)
    except Exception as e:
        print(f"    报错 {type(e).__name__}: {str(e).splitlines()[0][:90]}")
    lg = torch.randn(2, 5, 5, generator=g)
    tg5 = torch.randint(0, 5, (2, 5), generator=g)
    a = F.cross_entropy(lg, tg5)
    b = F.cross_entropy(lg.transpose(1, 2), tg5)
    print(f"    T 恰好等于 V = 5 时不报错：直接传 {a:.4f}，正确的 transpose(1, 2) {b:.4f}——数字不同，错误的那个是在\"对位置做 softmax\"")
    print("  (e) 标签是浮点数：")
    try:
        F.cross_entropy(z, torch.tensor([0.0]))
    except Exception as e:
        print(f"    报错 {type(e).__name__}: {str(e).splitlines()[0][:90]}")
    print("    （较新的 PyTorch 把浮点标签当作\"概率分布\"，形状必须和 logits 一样，否则报错）")
    print("  (f) fp16 下算 loss：")
    big = torch.tensor([[12.0, 10.0, 8.0]], dtype=torch.float16)
    print(f"    naive：−log(naive_softmax) = {(-naive_softmax(big).log()[0, 0]).item()}；稳定 log_softmax = {(-log_softmax(big)[0, 0]).item():.4f}；"
          f"fp32 参考 {F.cross_entropy(big.float(), t):.4f}")


# ---------------------------------------------------------------------- 10
def part10_ocr():
    section("10. 和 OCR 的连接（3.11）：每一步输出一个 token，loss 就是这一步的交叉熵")
    formulas = [r"\frac{a}{b}", r"x^2"]
    chars = sorted(set("".join(formulas)))
    itos = ["<pad>", "<bos>", "<eos>"] + chars
    stoi = {c: i for i, c in enumerate(itos)}
    V = len(itos)
    print(f"  玩具词表（按字符切，第 6 章再讲 BPE）：V = {V}：{itos}")
    seqs = [[stoi[c] for c in f] for f in formulas]
    T = max(len(s) for s in seqs) + 1
    PAD_ID, IGN = 0, -100
    inp = torch.full((2, T), PAD_ID)
    tgt = torch.full((2, T), IGN)
    for b, s in enumerate(seqs):
        x = [stoi["<bos>"]] + s
        y = s + [stoi["<eos>"]]
        inp[b, :len(x)] = torch.tensor(x)
        tgt[b, :len(y)] = torch.tensor(y)
    for b in range(2):
        print(f"  公式 {formulas[b]!r}：")
        print(f"    输入  {[itos[i] for i in inp[b].tolist()]}")
        print(f"    目标  {[itos[i] if i != IGN else '·' for i in tgt[b].tolist()]}（· = ignore_index）")
    print(f"  输入、目标形状都是 (B, T) = {tuple(inp.shape)}；解码器输出 logits 形状 (B, T, V) = (2, {T}, {V})")
    logits = torch.zeros(2, T, V)
    nll = cross_entropy(logits, tgt, ignore_index=IGN, reduction="none")
    print(f"  未训练、输出全 0 的模型：每一步 −ln p = {nll[0, 0]:.4f} = ln {V}；平均 loss {cross_entropy(logits, tgt):.4f}")
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(2, T, V, generator=g)
    tgt_ok = tgt.clone()
    for b in range(2):
        for t in range(T):
            if tgt[b, t] != IGN and t % 2 == 0:
                logits[b, t, tgt[b, t]] += 4.0           # 假装模型在偶数步已经"学会"了
    nll = cross_entropy(logits, tgt_ok, reduction="none")
    print("  一个\"学了一半\"的模型（偶数步给正确 token 加了 4）每一步的 loss：")
    for b in range(2):
        cells = [f"{itos[tgt[b, t]]}:{nll[b, t]:.2f}" for t in range(T) if tgt[b, t] != IGN]
        print(f"    公式 {b}: " + "  ".join(cells))
    print(f"  按 token 平均 {cross_entropy(logits, tgt_ok):.4f}；按序列平均 {sequence_mean_cross_entropy(logits, tgt_ok):.4f}")
    print(f"  ppl = e^loss：全 0 模型 e^{math.log(V):.4f} = {V}；\"学了一半\"的模型 e^{cross_entropy(logits, tgt_ok):.4f} = "
          f"{math.exp(cross_entropy(logits, tgt_ok).item()):.2f}")


PARTS = {1: part1_logits, 2: part2_overflow, 3: part3_likelihood, 4: part4_cross_entropy, 5: part5_gradient,
         6: part6_sequences, 7: part7_init_loss, 8: part8_classifier, 9: part9_pitfalls, 10: part10_ocr}

if __name__ == "__main__":
    for k in [int(a) for a in sys.argv[1:] if a.isdigit()] or list(PARTS):
        PARTS[k]()
    print("\n第 3 章的实验全部完成。")
