"""第 16 章起：语言模型的训练工具（第 18 章会加上梯度累积、续训等）。

    get_batch      从一长串 token 里随机截取 B 段长 T+1 的窗口：x 是前 T 个，y 是后 T 个（错开一位）
    eval_loss      在验证 token 上取若干批，求平均损失
    train_lm       AdamW + 线性预热 + 余弦衰减 + 梯度裁剪的训练循环
"""
import math
import time

import torch

from .optim import warmup_cosine


# >>> 练习 16.5：取一批训练数据
def get_batch(ids, B, T, g, device="cpu"):
    """ids 是一维 LongTensor。随机选 B 个起点 s（0 ≤ s ≤ len(ids) − T − 1，用生成器 g），
    x[b] = ids[s : s+T]，y[b] = ids[s+1 : s+T+1]。返回 (x, y)，形状都是 (B, T)，放到 device 上。
    y 就是 x 往左错开一位：位置 t 的目标是第 t+1 个 token——一次前向同时训练 T 个"预测下一个"的任务。"""
    s = torch.randint(0, len(ids) - T - 1, (B,), generator=g)
    off = torch.arange(T + 1)
    w = ids[s[:, None] + off]
    return w[:, :-1].to(device), w[:, 1:].to(device)
# <<<


@torch.no_grad()
def eval_loss(model, ids, B, T, n_batches=20, device="cpu", seed=1234):
    """固定种子取 n_batches 批，所以每次评测用的是同一批窗口，曲线不会因为抽样而抖动。"""
    was = model.training
    model.eval()
    g = torch.Generator().manual_seed(seed)
    tot = 0.0
    for _ in range(n_batches):
        x, y = get_batch(ids, B, T, g, device)
        tot += model(x, targets=y)[1].item()
    model.train(was)
    return tot / n_batches


def train_lm(model, train_ids, val_ids, steps, B=32, T=128, lr=3e-3, min_lr=3e-4, warmup=100, wd=0.1,
             clip=1.0, eval_every=200, device="cpu", seed=0, log=print):
    """返回 history：[(step, train_loss, val_loss, 秒), ...]。train_loss 是最近 eval_every 步的平均。"""
    g = torch.Generator().manual_seed(seed)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]       # 矩阵做权重衰减，向量（归一化的 γ）不做
    other = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": other, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95))
    hist, run, t0 = [], 0.0, time.perf_counter()
    model.train()
    for step in range(1, steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = warmup_cosine(step - 1, steps, warmup, lr, min_lr)
        x, y = get_batch(train_ids, B, T, g, device)
        loss = model(x, targets=y)[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        run += loss.item()
        if step % eval_every == 0 or step == steps:
            n = eval_every if step % eval_every == 0 else step % eval_every
            vl = eval_loss(model, val_ids, B, T, device=device)
            hist.append((step, run / n, vl, time.perf_counter() - t0))
            if log:
                log(f"    第 {step:5d} 步  训练 {run / n:.4f}  验证 {vl:.4f} nat/字符 = {vl / math.log(2):.3f} bits/字符  "
                    f"（{time.perf_counter() - t0:.0f} 秒，{step * B * T / (time.perf_counter() - t0):,.0f} token/秒）")
            run = 0.0
    return hist
