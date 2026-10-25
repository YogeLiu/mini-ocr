"""第 18 章：训练工程。只用 CPU，约 10 分钟。

    python -m labs.ch18.eng_tour             # 全部 6 个部分
    python -m labs.ch18.eng_tour 5           # 只运行第 5 部分（迷你缩放定律）
"""
import copy
import math
import os
import sys
import time

if "--gpu" not in sys.argv:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from miniocr.data.corpus import mixed_corpus, text_vocab
from miniocr.models.gpt import GPT, GPTConfig
from miniocr.train.engine import (flops_per_token, load_checkpoint, memory_breakdown, save_checkpoint,
                                  saved_activation_bytes, train_step)
from miniocr.train.lm import eval_loss, get_batch, train_lm

torch.set_num_threads(min(16, os.cpu_count() or 1))
OUT = "outputs/ch18"
os.makedirs(OUT, exist_ok=True)
V = text_vocab()
BASE = dict(vocab_size=len(V), block_size=256, n_layer=4, n_head=4, d_model=256)
_C = {}


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def data():
    if not _C:
        _C["tr"] = torch.tensor(V.encode(mixed_corpus(0, 20000)), dtype=torch.long)
        _C["va"] = torch.tensor(V.encode(mixed_corpus(1, 400)), dtype=torch.long)
    return _C["tr"], _C["va"]


def fmt_bytes(n):
    return f"{n / 2**20:8.1f} MB" if n < 2**30 else f"{n / 2**30:8.2f} GB"


# ---------------------------------------------------------------------- 1
def measure_tps(m, B, T, steps=4):
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    x = torch.randint(4, len(V), (B, T)); y = torch.randint(4, len(V), (B, T))
    train_step(m, opt, [(x, y)])
    t = time.perf_counter()
    for _ in range(steps):
        train_step(m, opt, [(x, y)])
    return B * T * steps / (time.perf_counter() - t)


def part1_flops():
    section("1. 计算量：每个 token 约 6N 次浮点运算")
    torch.manual_seed(0)
    cfg = GPTConfig(**BASE)
    m = GPT(cfg)
    N = m.n_params(non_embedding=True)
    T = 128
    f = flops_per_token(N, cfg.n_layer, cfg.d_model, T)
    print(f"  第 16 章的 GPT：不含嵌入的参数 N = {N:,}；6N = {6 * N:,}；注意力 12·L·T·d = {12 * cfg.n_layer * T * cfg.d_model:,}（T = {T}）")
    print(f"  每个训练 token 约 {f / 1e6:.1f} MFLOP；注意力部分占 {12 * cfg.n_layer * T * cfg.d_model / f:.0%}（T 越长占比越大）")
    tps = measure_tps(m, 32, T)
    print(f"  实测 CPU 训练速度 {tps:,.0f} token/秒 → {tps * f / 1e9:.0f} GFLOP/秒")
    tok = 3000 * 32 * 128
    print(f"  第 16 章的预训练：{tok / 1e6:.1f}M token × {f / 1e6:.1f} MFLOP = {tok * f / 1e15:.2f} PFLOP")
    # Qwen3-0.6B：不含嵌入约 0.44B 参数；官方报告预训练约 36T token
    Nq, Dq = 0.44e9, 36e12
    C = 6 * Nq * Dq
    print(f"  对照 Qwen3-0.6B：6 × 0.44B × 36T token = {C:.1e} FLOP。一块 H100 的 bf16 峰值约 1e15 FLOP/秒，")
    print(f"    按 40% 的利用率需要 {C / (0.4e15) / 86400:,.0f} 块 GPU·天——这就是为什么我们只能训练玩具模型，第 24 章起直接用别人预训练好的模型")
    print("  推导：一个 d_in × d_out 的线性层，前向每个 token 做 d_in·d_out 次乘法和同样多次加法 = 2·参数；")
    print("  反向要算对输入的梯度（又一次矩阵乘法）和对权重的梯度（再一次），共 4·参数。合计 6N")


# ---------------------------------------------------------------------- 2
def run_ckpt(m, x, y):
    """每个 Block 用 torch.utils.checkpoint 包起来：前向时不保存块内部的激活，反向时重新算一遍。"""
    h = m.tok(x)
    cos, sin = m.cos[:x.shape[1]], m.sin[:x.shape[1]]
    for b in m.blocks:
        h = checkpoint(b, h, cos, sin, use_reentrant=False)
    logits = m.head(m.norm(h))
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


def part2_memory():
    section("2. 显存都去哪了：参数、梯度、优化器状态、激活")
    torch.manual_seed(0)
    m = GPT(GPTConfig(**BASE))
    mb = memory_breakdown(m)
    print("  与参数量成正比的部分（float32）：" + "  ".join(f"{k} {v / 2**20:.1f} MB" for k, v in mb.items()) +
          f"  合计 {sum(mb.values()) / 2**20:.1f} MB = 16 字节/参数")
    print(f"  {'批 × 长':>10s} {'保存的激活':>12s} {'每个 token':>12s}   {'用激活检查点':>12s}")
    for B, T in ((8, 128), (32, 128), (32, 256), (64, 256)):
        x = torch.randint(4, len(V), (B, T)); y = torch.randint(4, len(V), (B, T))
        a, loss = saved_activation_bytes(lambda: m(x, targets=y)[1])
        loss.backward(); m.zero_grad(set_to_none=True)
        c, loss = saved_activation_bytes(lambda: run_ckpt(m, x, y))
        loss.backward(); m.zero_grad(set_to_none=True)
        print(f"  {B:4d} × {T:<4d} {fmt_bytes(a)} {a / (B * T) / 1024:9.1f} KB   {fmt_bytes(c)}")
    print("  激活与 批 × 长 成正比：每个 token 约 90 KB，4 层平均每层 22 KB ≈ 22 个 d 维的 float32 向量（归一化、注意力、SwiGLU 的 2.75d 中间结果……）。")
    print("  参数只有 3.2M（13 MB），批 64 × 256 时激活却有几百 MB：小模型、长序列时，激活是显存的大头。")
    print("  激活检查点只保存每个块的输入，反向时重新做一遍前向：本例中激活降到约 1/12，代价是多算约 1/3 的计算量")
    q = 0.6e9
    print(f"  对照：全量微调 Qwen3-0.6B，参数 + 梯度 + Adam（混合精度时还有一份 float32 主权重）≈ 16 字节 × 0.6B = {16 * q / 2**30:.1f} GB，")
    print("    还没算激活。第 28 章会用 LoRA：只训练很少的参数，梯度和优化器状态几乎为 0")


# ---------------------------------------------------------------------- 3
def part3_accum():
    section("3. 梯度累积：用小批凑出大批")
    torch.manual_seed(0)
    m1 = GPT(GPTConfig(**BASE)); m2 = copy.deepcopy(m1)
    o1 = torch.optim.AdamW(m1.parameters(), lr=1e-3); o2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    tr, _ = data()
    g = torch.Generator().manual_seed(0)
    x, y = get_batch(tr, 32, 128, g)
    l1 = train_step(m1, o1, [(x, y)])
    l2 = train_step(m2, o2, [(x[i:i + 8], y[i:i + 8]) for i in range(0, 32, 8)])
    diff = max((a - b).abs().max().item() for a, b in zip(m1.parameters(), m2.parameters()))
    print(f"  批 32 一次 vs 批 8 累积 4 次：损失 {l1:.6f} vs {l2:.6f}；更新后参数的最大差 {diff:.1e}（浮点运算顺序不同带来的误差）")
    for B, n in ((32, 1), (8, 4), (2, 16)):
        mm = GPT(GPTConfig(**BASE)); oo = torch.optim.AdamW(mm.parameters(), lr=1e-4)
        mbs = [(x[i:i + B], y[i:i + B]) for i in range(0, 32, B)]
        train_step(mm, oo, mbs)
        t = time.perf_counter()
        for _ in range(3):
            train_step(mm, oo, mbs)
        dt = (time.perf_counter() - t) / 3
        a, _ = saved_activation_bytes(lambda: mm(x[:B], targets=y[:B])[1]); mm.zero_grad(set_to_none=True)
        print(f"    小批 {B:2d} × 累积 {n:2d}：每步 {dt:.2f} 秒，激活峰值 {a / 2**20:6.1f} MB")
    print("  效果相同、激活只有 1/n，代价是小批太小时算得慢（矩阵太小，CPU/GPU 吃不饱）。")
    print("  注意：等价的前提是每个小批的 token 数相同。如果各小批里有效 token（不被 mask 的）数目不同，")
    print("  \"先在小批内平均、再除以 n\"就不等于\"全部 token 平均\"——第 26 章做监督微调时会遇到这个问题")


# ---------------------------------------------------------------------- 4
def part4_resume():
    section("4. 检查点与续训：中断后接着训练，结果与不中断完全一致")
    tr, va = data()
    cfg = dict(BASE, d_model=128)

    def fresh():
        torch.manual_seed(0)
        m = GPT(GPTConfig(**cfg))
        return m, torch.optim.AdamW(m.parameters(), lr=3e-3, betas=(0.9, 0.95)), torch.Generator().manual_seed(0)

    def go(m, opt, g, n):
        return [train_step(m, opt, [get_batch(tr, 16, 128, g)]) for _ in range(n)]

    m, opt, g = fresh()
    ref = go(m, opt, g, 120)
    m, opt, g = fresh()
    first = go(m, opt, g, 60)
    save_checkpoint(f"{OUT}/ck.pt", m, opt, 60, g)
    m2, opt2, g2 = fresh()
    load_checkpoint(f"{OUT}/ck.pt", m2, opt2, g2)
    resumed = first + go(m2, opt2, g2, 60)
    print(f"  不中断训练 120 步 vs 训练 60 步、保存、在新进程里恢复、再训练 60 步：120 个损失{'逐位相同' if resumed == ref else '不同！'}")
    m3, _, g3 = fresh()
    ck = torch.load(f"{OUT}/ck.pt", weights_only=False)
    m3.load_state_dict(ck["model"]); g3.set_state(ck["rng"])
    opt3 = torch.optim.AdamW(m3.parameters(), lr=3e-3, betas=(0.9, 0.95))   # 忘了恢复优化器状态
    bad = first + go(m3, opt3, g3, 60)
    print(f"  只恢复模型、忘了恢复优化器：第 61–65 步的损失 {[round(v, 3) for v in bad[60:65]]}")
    print(f"                       正确续训时是       {[round(v, 3) for v in ref[60:65]]}")
    print("  新的 Adam 的 m、v 都是 0，第一步按偏差修正等于 lr·sign(g)（第 11 章），每个参数都被推了满满一个学习率，")
    print("  损失跳起来，要过一阵才回到原来的轨道。所以检查点必须包括优化器状态和数据的随机状态")
    os.remove(f"{OUT}/ck.pt")


# ---------------------------------------------------------------------- 5
def power_fit(N, L):
    """拟合 L = c + a · N^(−α)：对 c 做网格搜索，每个 c 下 log(L − c) 对 log N 做直线拟合。"""
    best = None
    for c in np.linspace(0, min(L) * 0.98, 200):
        A = np.vstack([np.ones(len(N)), np.log(N)]).T
        coef, res, *_ = np.linalg.lstsq(A, np.log(np.array(L) - c), rcond=None)
        err = np.sum((np.exp(A @ coef) + c - L) ** 2)
        if best is None or err < best[0]:
            best = (err, c, math.exp(coef[0]), -coef[1])
    return best[1:]


def part5_scaling():
    section("5. 迷你缩放定律：模型越大，损失越低，但收益递减")
    tr, va = data()
    steps, B, T = 800, 32, 128
    print(f"  每个模型训练 {steps} 步 × {B} × {T} = {steps * B * T / 1e6:.1f}M 个 token（数据量固定），4 层，只改宽度 d")
    rows = []
    for d in (32, 64, 96, 128, 192, 256):
        torch.manual_seed(0)
        m = GPT(GPTConfig(**dict(BASE, d_model=d)))
        h = train_lm(m, tr, va, steps, B=B, T=T, eval_every=steps, log=None)
        N = m.n_params(non_embedding=True)
        rows.append((d, N, h[-1][2], h[-1][3]))
        print(f"    d = {d:3d}  N = {N:9,}  验证 {h[-1][2] / math.log(2):.3f} bits/字符  （{h[-1][3]:.0f} 秒，"
              f"计算量 {6 * N * steps * B * T / 1e12:.2f} TFLOP）")
    N = np.array([r[1] for r in rows], dtype=float)
    L = np.array([r[2] for r in rows]) / math.log(2)
    c, a, alpha = power_fit(N, L)
    print(f"  拟合 L(N) = {c:.3f} + {a:.2f} · N^(−{alpha:.3f})（bits/字符）")
    print(f"  外推：N = 10M 时 {c + a * 1e7 ** -alpha:.3f}，N = 100M 时 {c + a * 1e8 ** -alpha:.3f}——越来越接近 {c:.3f}")
    print("  常数 c 是\"这么多数据下，无论模型多大都降不下去的部分\"（数据量固定，大模型会过拟合或欠训练），")
    print("  也包含语言本身的不确定性（我们的语料是随机生成的，单词序列本来就无法预测）")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.loglog(N, L - c, "o", label="measured − c")
    xs = np.logspace(np.log10(N.min()), np.log10(N.max()), 50)
    ax.loglog(xs, a * xs ** -alpha, "--", label=f"a·N^-{alpha:.2f}")
    ax.set_xlabel("non-embedding params N"); ax.set_ylabel("val bits/char − c"); ax.legend(); fig.tight_layout()
    fig.savefig(f"{OUT}/scaling.png", dpi=100); plt.close(fig)
    print(f"  图存到 {OUT}/scaling.png：减去 c 之后在双对数坐标上是一条直线，这就是\"幂律\"")
    print("  Chinchilla（2022）的经验：计算量固定时，参数量和训练 token 数应同比例增长，大约每个参数 20 个 token。")
    print(f"  我们最大的模型 N = {rows[-1][1]:,}，按这个比例需要 {20 * rows[-1][1] / 1e6:.0f}M token，实际只训练了 {steps * B * T / 1e6:.1f}M——它是\"欠训练\"的")


# ---------------------------------------------------------------------- 6
def part6_throughput():
    section("6. 吞吐：批越大，每个 token 越便宜（直到算力用满）")
    torch.manual_seed(0)
    m = GPT(GPTConfig(**BASE))
    print(f"  {'批大小':>6s} {'训练 token/秒':>14s} {'推理（前向）token/秒':>20s}")
    for B in (1, 4, 16, 64):
        tps = measure_tps(m, B, 128, steps=3)
        x = torch.randint(4, len(V), (B, 128))
        with torch.no_grad():
            m(x)
            t = time.perf_counter()
            for _ in range(3):
                m(x)
            inf = B * 128 * 3 / (time.perf_counter() - t)
        print(f"  {B:6d} {tps:14,.0f} {inf:20,.0f}")
    print("  批 1 时矩阵乘法太小，大部分时间在调度和搬数据；批大了以后每次读入的权重被更多 token 共用。")
    print("  训练速度一般在批达到某个大小后不再上升：算力被用满了。再大的批只增加显存，不增加速度")


PARTS = {1: part1_flops, 2: part2_memory, 3: part3_accum, 4: part4_resume, 5: part5_scaling, 6: part6_throughput}

if __name__ == "__main__":
    for k in [int(a) for a in sys.argv[1:] if a.isdigit()] or list(PARTS):
        PARTS[k]()
    print("\n第 18 章的演示全部完成。")
