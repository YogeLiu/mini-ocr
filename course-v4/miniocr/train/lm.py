"""第 2.3–2.5 单元：语言模型的训练工具。

    get_batch          从一长串 token 里随机截取 B 段长 T+1 的窗口：x 是前 T 个，y 是后 T 个（错开一位）
    eval_loss          在验证 token 上取固定的若干批，求平均损失
    accumulate_grads   梯度累积：把一个大批拆成几个小批分别反向，梯度加起来（第 2.5 单元：按 token 还是按样本平均）
    TrainConfig        训练超参数，含设备、混合精度、梯度累积、检查点
    train_lm           AdamW + 预热 + 余弦衰减 + 梯度裁剪；可续训；返回每次评测的记录

所有张量都放到 cfg.device 上；模型由调用者先 .to(device)。
"""
import math
import os
import time
from dataclasses import asdict, dataclass

import torch

from ..device import autocast, synchronize
from .optim import warmup_cosine


# >>> 练习 2.3.5：取一批训练数据
def get_batch(ids, B, T, g, device="cpu"):
    """ids 是一维 LongTensor（在 CPU 上）。随机选 B 个起点 s（0 ≤ s ≤ len(ids) − T − 1，用生成器 g），
    x[b] = ids[s : s+T]，y[b] = ids[s+1 : s+T+1]。返回 (x, y)，形状都是 (B, T)，放到 device 上。
    y 就是 x 往左错开一位：位置 t 的目标是第 t+1 个 token——一次前向同时训练 T 个"预测下一个"的任务。"""
    s = torch.randint(0, len(ids) - T, (B,), generator=g)          # 上界不含：s 最大取 len(ids) − T − 1
    off = torch.arange(T + 1)
    w = ids[s[:, None] + off]
    return w[:, :-1].to(device), w[:, 1:].to(device)
# <<<


@torch.no_grad()
def eval_loss(model, ids, B, T, n_batches=20, device="cpu", seed=1234, amp_dtype=None):
    """固定种子取 n_batches 批，所以每次评测用的是同一批窗口，曲线不会因为抽样而抖动。返回 nat/token。"""
    was = model.training
    model.eval()
    g = torch.Generator().manual_seed(seed)
    tot = 0.0
    for _ in range(n_batches):
        x, y = get_batch(ids, B, T, g, device)
        with autocast(device, amp_dtype):
            tot += model(x, targets=y)[1].item()
    model.train(was)
    return tot / n_batches


# >>> 练习 2.5.1：梯度累积的两种平均
def accumulate_grads(model, micro_batches, mode="token"):
    """micro_batches：[(x, y), ...]，y 中 −100 的位置不计损失（例如补齐的位置）。对每个小批做一次前向和反向，
    让累积后的梯度等于"把所有小批拼成一个大批再算一次"的梯度：
      mode="token"：大批的损失 = 所有有效 token 的损失之和 / 所有有效 token 总数。
                    每个小批要用 sum 损失除以**全部小批**的有效 token 总数，而不是除以本小批的 token 数；
      mode="sample"：先在每条样本内部按 token 平均，再对所有样本平均（每条样本权重相同，不论多长）；
      mode="micro_mean"：常见的错误写法——每个小批按自己的 token 数平均，再除以小批个数。
                    各小批有效 token 数不同时，它既不等于 "token" 也不等于 "sample"（第 2.5 单元的故障演练）。
    model(x) 返回 (logits, _)；用 F.cross_entropy(..., reduction="none") 自己算逐 token 损失。
    调用前梯度应已清零；函数只做 backward，不做 optimizer.step()。返回大批的损失（Python 数）。"""
    import torch.nn.functional as F
    if mode == "token":
        total = sum(int((y != -100).sum()) for _, y in micro_batches)
    elif mode == "micro_mean":
        total = len(micro_batches)
    else:
        total = sum(y.shape[0] for _, y in micro_batches)
    out = 0.0
    for x, y in micro_batches:
        logits = model(x)[0]
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1),
                              ignore_index=-100, reduction="none").view(y.shape)
        valid = (y != -100).float()
        if mode == "token":
            loss = nll.sum() / total
        elif mode == "micro_mean":
            loss = nll.sum() / valid.sum().clamp(min=1) / total
        else:
            loss = ((nll * valid).sum(1) / valid.sum(1).clamp(min=1)).sum() / total
        loss.backward()
        out += loss.item()
    return out
# <<<


@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 32           # 每次更新用的序列数（= micro_batch × accum）
    accum: int = 1                 # 梯度累积的小批数；batch_size 必须能被它整除
    seq_len: int = 128
    lr: float = 3e-3
    min_lr: float = 3e-4
    warmup: int = 100
    weight_decay: float = 0.1
    clip: float = 1.0
    eval_every: int = 200
    eval_batches: int = 20
    device: str = "cpu"
    amp: str = "fp32"              # "fp32" / "bf16" / "fp16"（fp16 需要 loss scaling，第 2.5 单元）
    seed: int = 0
    ckpt_path: str = ""            # 非空时每次评测后保存检查点，启动时若已存在就续训


_DTYPES = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def _param_groups(model, wd):
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]    # 矩阵做权重衰减
    other = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]     # 归一化的 γ 不做
    return [{"params": decay, "weight_decay": wd}, {"params": other, "weight_decay": 0.0}]


def _get_global_rng(device):
    dev = torch.device(device)
    out = {"cpu": torch.get_rng_state()}
    if dev.type == "cuda":
        out["cuda"] = torch.cuda.get_rng_state()
    elif dev.type == "mps":
        out["mps"] = torch.mps.get_rng_state()
    return out


def _set_global_rng(state, device):
    torch.set_rng_state(state["cpu"])
    dev = torch.device(device)
    if dev.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"])
    elif dev.type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def train_lm(model, train_ids, val_ids, cfg: TrainConfig, log=print):
    """返回 history：[{"step", "train", "val", "sec", "tok_per_sec", "lr"}, ...]。train 是最近 eval_every 步的平均。
    续训时从检查点恢复模型、优化器、GradScaler、取数据的随机数生成器、全局随机数（dropout 用）和已有的 history，
    继续训练到 cfg.steps。续训时不要改 cfg.steps：余弦调度按总步数计算。"""
    assert cfg.batch_size % cfg.accum == 0, "batch_size 必须能被 accum 整除"
    mb = cfg.batch_size // cfg.accum
    dtype = _DTYPES[cfg.amp]
    opt = torch.optim.AdamW(_param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler(enabled=cfg.amp == "fp16")
    g = torch.Generator().manual_seed(cfg.seed)
    hist, start = [], 0
    if cfg.ckpt_path and os.path.exists(cfg.ckpt_path):
        ck = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)   # 先读到 CPU：随机数状态必须是 CPU 张量
        model.load_state_dict(ck["model"])                 # 参数按模型所在设备拷贝
        opt.load_state_dict(ck["opt"])                     # 优化器状态会自动搬到参数所在的设备
        scaler.load_state_dict(ck["scaler"])
        g.set_state(ck["rng"])
        _set_global_rng(ck["global_rng"], cfg.device)      # dropout 用的是全局随机数
        hist, start = ck["history"], ck["step"]
        if log:
            log(f"    从检查点续训：第 {start} 步")
    model.train()
    run, n_run, t0, tok0 = 0.0, 0, time.perf_counter(), 0
    for step in range(start + 1, cfg.steps + 1):
        lr = warmup_cosine(step - 1, cfg.steps, cfg.warmup, cfg.lr, cfg.min_lr)
        for gr in opt.param_groups:
            gr["lr"] = lr
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(cfg.accum):
            x, y = get_batch(train_ids, mb, cfg.seq_len, g, cfg.device)
            with autocast(cfg.device, dtype):
                loss = model(x, targets=y)[1] / cfg.accum      # 每个小批 token 数相同，按小批平均即按 token 平均
            scaler.scale(loss).backward()
            tot += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        scaler.step(opt)
        scaler.update()
        run, n_run, tok0 = run + tot, n_run + 1, tok0 + cfg.batch_size * cfg.seq_len
        if step % cfg.eval_every == 0 or step == cfg.steps:
            synchronize(cfg.device)
            sec = time.perf_counter() - t0
            vl = eval_loss(model, val_ids, mb, cfg.seq_len, cfg.eval_batches, cfg.device, amp_dtype=dtype)
            rec = dict(step=step, train=run / n_run, val=vl, sec=sec, tok_per_sec=tok0 / max(sec, 1e-9), lr=lr)
            hist.append(rec)
            if log:
                log(f"    第 {step:5d} 步  训练 {rec['train']:.4f}  验证 {vl:.4f} nat/token  "
                    f"lr {lr:.2e}  {rec['tok_per_sec']:,.0f} token/秒")
            if cfg.ckpt_path:
                torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(),
                                rng=g.get_state(), global_rng=_get_global_rng(cfg.device),
                                history=hist, step=step, cfg=asdict(cfg)), cfg.ckpt_path)
            run, n_run, t0, tok0 = 0.0, 0, time.perf_counter(), 0
    return hist


def bits_per_byte(nat_per_token, n_tokens, n_bytes):
    """把 nat/token 换成 bits/byte：不同分词器的 token 长短不同，nat/token 不能直接比；
    每字节的比特数与分词器无关，可以比较（第 2.1 单元）。"""
    return nat_per_token * n_tokens / n_bytes / math.log(2)
