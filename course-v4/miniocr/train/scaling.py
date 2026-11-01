"""第 13 章：参数口径、计算量前沿与幂律。仅 NumPy，不依赖 scipy。

N_non_embedding 用于 Kaplan 口径；预算/前沿的 N 默认采用 memory.matmul_params
口径（含共享输出矩阵）。6ND 忽略注意力的二次项，不是实测 FLOPs。
"""
import csv
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np

from miniocr.models.gpt import GPTConfig
from miniocr.train.memory import train_flops

LADDER = {"tiny": (2, 64, 4), "small": (3, 96, 4), "medium": (4, 128, 4),
          "large": (6, 192, 6), "xl": (8, 256, 8)}
# Hoffmann et al. 2022 §3.3 第三种方法的拟合值；只用于演示，不代表本项目。
HOFFMANN2022_APPROACH3 = dict(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)


def ladder_config(size, vocab_size, context=64):
    """按阶梯名字返回 GPTConfig；所有头维都是偶数。"""
    layers, dim, heads = LADDER[size]
    return GPTConfig(vocab_size=vocab_size, block_size=context, n_layer=layers,
                     d_model=dim, n_head=heads)


def parameter_counts(cfg):
    """解析计数，支持 GQA/QK norm/LayerNorm/位置嵌入；matmul 严格沿用 memory 旧口径。

    memory.matmul_params 历史定义还包含 norm 与位置表，本函数不悄悄修正它；
    阶梯均为 RoPE，norm 的少量参数仍计入近似预算。
    """
    d, f, v, layers = cfg.d_model, cfg.ff_dim(), cfg.vocab_size, cfg.n_layer
    kv = cfg.kv_heads() * (d // cfg.n_head)
    block = 2*d*d + 2*d*kv + 3*d*f + 2*d*(2 if cfg.norm == "layernorm" else 1)
    block += 2*(d // cfg.n_head) if cfg.qk_norm else 0
    pos = cfg.block_size*d if cfg.rope and cfg.pos_emb == "learned" else 0
    total = layers*block + d*(2 if cfg.norm == "layernorm" else 1) + pos + v*d*(1 if cfg.tie_weights else 2)
    return dict(total=total, non_embedding=total-v*d-pos,
                matmul=total-v*d+(v*d if cfg.tie_weights else 0))


def ladder_table(vocab_size, tokens, context=64):
    """返回阶梯配置、三种参数数目、每 token FLOPs 与指定 D 的 6ND。"""
    rows = []
    for name in LADDER:
        cfg = ladder_config(name, vocab_size, context)
        counts = parameter_counts(cfg)
        rows.append(dict(size=name, layers=cfg.n_layer, dim=cfg.d_model, heads=cfg.n_head,
                         **counts, flops_per_token=train_flops(counts["matmul"], 1),
                         compute=train_flops(counts["matmul"], tokens),
                         attention_extra_ratio=cfg.n_layer*context*cfg.d_model/counts["matmul"]))
    return rows


def fraction_config(cfg, frac):
    """独立完整调度：round(frac*steps)，warmup 同比缩放且至少 1（不超过 steps）。"""
    if not math.isfinite(frac) or not 0 < frac <= 1:
        raise ValueError("frac 须在 (0,1] 内")
    steps = max(1, round(frac*cfg.steps))
    return replace(cfg, steps=steps, warmup=min(steps, max(1, round(frac*cfg.warmup))))


def read_ladder_points(root, log=print):
    """读取 size/frac*/ 的最终标准验证点；兼容旧 size/ 日志（标 frac=1）。

    不读取余弦中途点。不完整 run 不会建立比较协议；同训练流、分词器、验证集、
    BOS/EOS、上下文、标准验证上限和实际分母才可混拟。返回含 size/frac/N/D/C 的行。
    """
    root = Path(root)
    points, protocol, denominators = [], None, None
    paths = sorted(set(root.glob("*/frac*/evals.csv")) | set(root.glob("*/evals.csv")))
    for path in paths:
        config = json.loads((path.parent/"config.json").read_text(encoding="utf-8"))
        cfg = config["training"]
        with path.open(encoding="utf-8") as f:
            finals = [r for r in csv.DictReader(f) if int(r["step"]) == cfg["steps"]]
        if not finals:
            if log:
                log(f"跳过未完成日志={path}")
            continue
        row = finals[-1]
        current = tuple(config.get(k) for k in ("tokenizer_hash", "val_hash", "train_hash",
                                                 "data_version", "bos", "eos")) + (
                                                     cfg["context"], cfg["final_eval_formulas"])
        actual = tuple(int(row[k]) for k in ("eval_formulas", "eval_tokens", "eval_bytes"))
        seen, loss = int(row["tokens_seen"]), float(row["val_nat_per_token"])
        reason = None
        if protocol is not None and current != protocol:
            reason = "不兼容"
        elif denominators is not None and actual != denominators:
            reason = "验证分母不同"
        elif (seen != cfg["steps"]*cfg["batch"]*cfg["context"] or min(actual) <= 0 or
              actual[0] > cfg["final_eval_formulas"] or not math.isfinite(loss) or loss <= 0):
            reason = "最终点非法"
        if reason:
            if log:
                log(f"跳过{reason}的日志={path}")
            continue
        protocol, denominators = current, actual
        fractional = path.parent.name.startswith("frac")
        size = path.parent.parent.name if fractional else path.parent.name
        frac = float(path.parent.name[4:]) if fractional else 1.0
        points.append(dict(size=size, frac=frac, N=config["matmul_params"], D=seen,
                           C=6*config["matmul_params"]*seen, val_nat_per_token=loss,
                           run_dir=str(path.parent)))
    return points


def _xy(x, y, minimum=2):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (x.ndim != 1 or y.shape != x.shape or len(x) < minimum or
            not np.isfinite(x).all() or not np.isfinite(y).all() or
            np.any(x <= 0) or np.any(y <= 0) or np.ptp(np.log(x)) == 0):
        raise ValueError("x/y 须为等长正有限一维数组，x 至少有两个不同值")
    return x, y


def fit_power_law(x, y):
    """闭式 log 最小二乘 log y = log a - alpha log x；residual 为 log MSE。"""
    x, y = _xy(x, y)
    u, v = np.log(x), np.log(y)
    slope = np.dot(u-u.mean(), v-v.mean()) / np.dot(u-u.mean(), u-u.mean())
    intercept = v.mean() - slope*u.mean()
    return dict(a=float(np.exp(intercept)), alpha=float(-slope),
                residual=float(np.mean((v-intercept-slope*u)**2)))


def fit_power_law_offset(x, y):
    """网格定位后黄金分割搜索 Linf∈[0,min(y))；内层闭式拟合 log(y-Linf)。

    外层目标为原始 y 空间 MSE，避免不同偏移的 log 残差尺度不可比。
    至少三个点；窄计算区间或大噪声下偏移不可辨识，不应解释为真实极限。
    """
    x, y = _xy(x, y, 3)

    def candidate(offset):
        fit = fit_power_law(x, y-offset)
        pred = fit["a"] * x**(-fit["alpha"]) + offset
        return dict(a=fit["a"], alpha=fit["alpha"], Linf=float(offset),
                    residual=float(np.mean((pred-y)**2)))

    grid = np.linspace(0, np.min(y)*(1-1e-10), 257)
    fits = [candidate(z) for z in grid]
    best = int(np.argmin([r["residual"] for r in fits]))
    lo, hi = grid[max(0, best-1)], grid[min(len(grid)-1, best+1)]
    ratio = (math.sqrt(5)-1)/2
    for _ in range(90):
        left, right = hi-ratio*(hi-lo), lo+ratio*(hi-lo)
        if candidate(left)["residual"] < candidate(right)["residual"]:
            hi = right
        else:
            lo = left
    return min([fits[best], candidate((lo+hi)/2)], key=lambda r: r["residual"])


def compute_frontier(points, bins=12):
    """(C,L) → 各 log C 区间的最低 loss 实测点，按 C 排序（不插值、不造点）。

    同一区间内可比较不同模型/检查点；空区间跳过，保留最低点的真实 C。
    不强制单调，以免隐藏噪声；bins=None 时仅合并完全相同的 C。
    """
    p = np.asarray(list(points), dtype=float)
    if p.size == 0:
        return []
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError("前沿点须为正有限 (C,L)")
    if bins is None:
        keys = p[:, 0]
    else:
        if not isinstance(bins, int) or bins < 1:
            raise ValueError("bins 须为正整数或 None")
        edges = np.linspace(np.log(p[:, 0].min()), np.log(p[:, 0].max()), bins+1)
        keys = np.minimum(np.searchsorted(edges, np.log(p[:, 0]), side="right")-1, bins-1)
    chosen = [p[keys == k][np.argmin(p[keys == k, 1])] for k in np.unique(keys)]
    return sorted([tuple(map(float, row)) for row in chosen])


def synthetic_scaling(a=25.0, alpha=0.3, Linf=1.5, noise=0.0, seed=0, n=40):
    """返回 C∈[1e2,1e8] 和 L；noise 为 y 空间加性高斯噪声标准差。"""
    if a <= 0 or alpha <= 0 or Linf < 0 or noise < 0 or n < 3:
        raise ValueError("合成参数非法")
    c = np.geomspace(1e2, 1e8, n)
    loss = a*c**(-alpha) + Linf + np.random.default_rng(seed).normal(0, noise, n)
    return c, loss


def compare_offset_fits(x, y):
    """返回有/无不可约损失两种拟合，观察忽略 Linf 时 alpha 的偏差。"""
    return dict(power=fit_power_law(x, y), offset=fit_power_law_offset(x, y))


def chinchilla_loss(N, D, E, A, B, alpha, beta):
    """参数化损失 E + A/N^alpha + B/D^beta，支持 NumPy 数组。"""
    return E + A / np.asarray(N)**alpha + B / np.asarray(D)**beta


def chinchilla_optimal(C, E, A, B, alpha, beta):
    """约束 C=6ND 下闭式最优；返回 N_opt、D_opt、D_per_N、L_opt。"""
    if not all(math.isfinite(v) and v > 0 for v in (C, A, B, alpha, beta)) or not math.isfinite(E):
        raise ValueError("预算与系数须为正有限数（E 可为零）")
    g = (alpha*A/(beta*B))**(1/(alpha+beta))
    n = g*(C/6)**(beta/(alpha+beta))
    d = C/(6*n)
    return dict(N_opt=n, D_opt=d, D_per_N=d/n,
                L_opt=float(chinchilla_loss(n, d, E, A, B, alpha, beta)))


def chinchilla_grid(C, E, A, B, alpha, beta, points=20001):
    """独立在 log N 的 [1,C/6] 网格搜索；仅供 N,D≥1 的教学预算验证。"""
    if C <= 6 or points < 2:
        raise ValueError("网格要求 C>6 且 points>=2")
    n = np.geomspace(1, C/6, points)
    d = C/(6*n)
    loss = chinchilla_loss(n, d, E, A, B, alpha, beta)
    i = int(np.argmin(loss))
    return dict(N_opt=float(n[i]), D_opt=float(d[i]), L_opt=float(loss[i]))


def budget_steps(n_params, batch, context, *, compute=None, tokens=None):
    """恰给 compute 或 tokens；向上取整步数，同时返回实际 tokens/compute 与超额。

    batch 是累积后有效批量；6ND 的 N 由调用者显式选择口径。
    """
    if (compute is None) == (tokens is None) or min(n_params, batch, context) <= 0:
        raise ValueError("须指定一种预算，参数/批量/上下文须为正")
    d = tokens if tokens is not None else compute/(6*n_params)
    if not math.isfinite(d) or d <= 0:
        raise ValueError("预算须为正有限数")
    steps = math.ceil(d/(batch*context))
    actual = steps*batch*context
    return dict(steps=steps, requested_tokens=d, actual_tokens=actual,
                actual_compute=train_flops(n_params, actual), excess_tokens=actual-d)
