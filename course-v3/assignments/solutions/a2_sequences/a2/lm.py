"""A2 参考实现：语言模型的取批、梯度累积、bits per char；文字行识别的编码与 CER。"""
import math

import numpy as np
import torch

from miniocr.data.text import LINE_CHARSET

from .nn import log_softmax

LINE_CLASSES = ["<blank>"] + list(LINE_CHARSET)             # 第 0 类是 CTC 的空白


def encode_line(text):
    return [LINE_CHARSET.index(c) + 1 for c in text]


def decode_line(ids):
    return "".join(LINE_CLASSES[i] for i in ids)


def edit_distance(a, b):
    """两个序列的编辑距离（插入、删除、替换各算 1）。"""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def cer(preds, refs):
    """语料级字符错误率：Σ 编辑距离 / Σ 参考长度。"""
    return sum(edit_distance(p, r) for p, r in zip(preds, refs)) / max(1, sum(len(r) for r in refs))


def get_batch(ids, batch_size, context, generator, device="cpu"):
    """从一维 token 数组 ids 里随机取 batch_size 个起点，x = ids[s : s+context]，y = ids[s+1 : s+context+1]。"""
    starts = torch.randint(0, len(ids) - context, (batch_size,), generator=generator).numpy()
    x = np.stack([ids[s:s + context] for s in starts])
    y = np.stack([ids[s + 1:s + context + 1] for s in starts])
    return torch.from_numpy(x).long().to(device), torch.from_numpy(y).long().to(device)


def accumulate_grads(model, x, y, micro_batches):
    """把一批 (x, y) 均分成 micro_batches 份，逐份前向反向，梯度累加。结果与一次算整批相同。返回整批的平均损失。"""
    model.zero_grad(set_to_none=True)
    total = 0.0
    for xs, ys in zip(x.chunk(micro_batches), y.chunk(micro_batches)):
        _, loss = model(xs, ys)
        (loss * len(xs) / len(x)).backward()                    # 每份按样本数加权，合起来才是整批的平均
        total += loss.item() * len(xs) / len(x)
    return total


@torch.no_grad()
def nll_sum(model, ids, context, device="cpu"):
    """对整段 token 求总负对数似然（nat）。按长 context+1、相邻窗口重叠 1 个 token 切开，
    除第一个 token 外，每个 token 恰好被预测一次。"""
    was = model.training
    model.eval()
    ids = torch.as_tensor(np.asarray(ids), dtype=torch.long)
    total = 0.0
    for s in range(0, len(ids) - 1, context):
        y = ids[s + 1:s + context + 1]
        x = ids[s:s + len(y)]
        logits, _ = model(x[None].to(device))
        lp = log_softmax(logits[0].float(), -1)
        total -= lp.gather(-1, y.to(device)[:, None]).sum().item()
    model.train(was)
    return total


def bits_per_char(nll_nats, n_chars):
    return nll_nats / math.log(2) / n_chars
