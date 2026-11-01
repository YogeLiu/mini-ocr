"""自写 SFT 循环：一次 causal shift、按监督 token 加权、梯度累积、混合精度。

没有 Trainer/TRL/PEFT；模型需接受 input_ids/attention_mask/use_cache 并返回 logits。
CUDA bf16 指 fp32 参数与 AdamW 状态 + bf16 autocast，loss 累加用 fp32。
"""
from contextlib import nullcontext
import math
import random
import time

import torch
from torch.nn import functional as F

from miniocr.data.sft import collate_sft


# >>> 练习 19.3：只 shift 一次，并按有效 token 求交叉熵
def causal_loss(logits, labels, *, reduction="mean"):
    """logits[:,t] 预测 labels[:,t+1]；全 mask 拒绝，防止 NaN 被忽略。"""
    if logits.shape[:2] != labels.shape or reduction not in ("mean", "sum"):
        raise ValueError("logits/labels 形状或 reduction 非法")
    targets = labels[:, 1:]
    mask = targets != -100
    if not mask.any():
        raise ValueError("没有参与 loss 的 token")
    return F.cross_entropy(logits[:, :-1][mask].float(), targets[mask], reduction=reduction)
# <<<


def autocast_context(device, dtype):
    """仅显式 bf16 使用 autocast；CPU 默认 fp32，MPS 不冒充支持 bf16。"""
    if dtype == "fp32":
        return nullcontext()
    if dtype != "bf16" or torch.device(device).type == "mps":
        raise ValueError("此循环只支持 fp32 或 CPU/CUDA bf16")
    return torch.autocast(torch.device(device).type, dtype=torch.bfloat16)


def synchronize(device):
    """同步设备后计时，CPU 无操作。"""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    elif torch.device(device).type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def mean_loss(model, examples, pad_token_id, *, batch_size=4, dtype="fp32"):
    """固定正确 labels 的 token 加权 NLL；不会平均不同分母的 batch mean。"""
    if not examples or batch_size < 1:
        raise ValueError("评估数据不能为空，batch_size 必须为正")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    total = count = 0
    try:
        for i in range(0, len(examples), batch_size):
            batch = {k: v.to(device) for k, v in collate_sft(examples[i:i+batch_size], pad_token_id).items()}
            with autocast_context(device, dtype):
                output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
                total += causal_loss(output.logits, batch["labels"], reduction="sum").item()
            count += (batch["labels"][:, 1:] != -100).sum().item()
    finally:
        model.train(was_training)
    return dict(loss=total/count, supervised_tokens=count, n=len(examples))


# >>> 练习 19.4：自写训练循环与 token 加权梯度累积
def train_sft(model, examples, pad_token_id, *, steps=2, batch_size=2, accumulation=1,
              lr=2e-5, dtype="fp32", seed=0, callback=None):
    """固定种子打乱后循环取 batch；callback(step, record) 可在 optimizer step 后评 val。

    一个更新的分母为所有 microbatch 的监督 token 总数，而非 accumulation 个均值。
    每步只记录训练内耗时；评测/checkpoint 时间由调用方另行记录。bf16 不用 GradScaler。
    """
    if not examples or min(steps, batch_size, accumulation) < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("训练数据、规模或学习率非法")
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=.01)
    rng = random.Random(seed)
    order = []
    def next_batch():
        while len(order) < batch_size:
            indices = list(range(len(examples)))
            rng.shuffle(indices)
            order.extend(indices)
        chosen = order[:batch_size]
        del order[:batch_size]
        return collate_sft([examples[i] for i in chosen], pad_token_id)

    history = []
    model.train()
    for step in range(1, steps+1):
        synchronize(device)
        start = time.perf_counter()
        batches = [next_batch() for _ in range(accumulation)]
        count = sum((b["labels"][:, 1:] != -100).sum().item() for b in batches)
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for cpu_batch in batches:
            b = {k: v.to(device) for k, v in cpu_batch.items()}
            with autocast_context(device, dtype):
                logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False).logits
                loss_sum = causal_loss(logits, b["labels"], reduction="sum")
            if not torch.isfinite(loss_sum):
                raise FloatingPointError("non-finite loss")
            (loss_sum/count).backward()
            total += loss_sum.item()
            del logits, loss_sum
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        synchronize(device)
        record = dict(step=step, loss=total/count, supervised_tokens=count, grad_norm=float(norm),
                      seconds=time.perf_counter()-start, lr=lr)
        history.append(record)
        if callback is not None:
            callback(step, record)
    return history
# <<<
