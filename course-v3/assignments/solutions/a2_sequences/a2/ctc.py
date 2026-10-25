"""A2 参考实现：CTC 损失（对数空间、整批向量化）、贪心解码、前缀束搜索。第 0 类是空白。"""
import math

import torch

NEG = -1e30          # 用一个很大的负数代替 −∞：全是"不可能"的状态求 logsumexp 时，梯度不会变成 nan


def _lse(*xs):
    return torch.logsumexp(torch.stack(xs), dim=0)


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction="mean"):
    """log_probs (T, N, C)；targets (N, S) 补齐的标签；input_lengths、target_lengths (N,)。
    与 F.ctc_loss 相同：reduction="mean" 先把每个样本的损失除以它的标签长度，再对批求平均；
    "sum" 求和；"none" 返回 (N,)。不可能对齐（T 太短）的样本损失为 inf。"""
    T, N, C = log_probs.shape
    S = targets.shape[1]
    L = 2 * S + 1
    ext = torch.full((N, L), blank, dtype=torch.long, device=log_probs.device)
    ext[:, 1::2] = targets
    # 能否从 s−2 跳过来：当前不是空白，且与 s−2 的符号不同
    skip = torch.zeros(N, L, dtype=torch.bool, device=log_probs.device)
    skip[:, 2:] = (ext[:, 2:] != blank) & (ext[:, 2:] != ext[:, :-2])
    lp = log_probs.gather(2, ext[None].expand(T, N, L))          # (T, N, L)：每个时间步、每个扩展位置的对数概率
    neg = log_probs.new_full((N, L), NEG)
    alpha = neg.clone()
    alpha[:, 0] = lp[0, :, 0]
    alpha[:, 1] = torch.where(target_lengths > 0, lp[0, :, 1], neg[:, 1])
    for t in range(1, T):
        a1 = torch.cat([neg[:, :1], alpha[:, :-1]], 1)
        a2 = torch.where(skip, torch.cat([neg[:, :2], alpha[:, :-2]], 1), neg)
        new = _lse(alpha, a1, a2) + lp[t]
        alpha = torch.where((t < input_lengths)[:, None], new, alpha)   # 超过自己长度的样本不再更新
    idx = 2 * target_lengths
    last = alpha.gather(1, idx[:, None]).squeeze(1)
    prev = alpha.gather(1, (idx - 1).clamp(min=0)[:, None]).squeeze(1)
    prev = torch.where(target_lengths > 0, prev, torch.full_like(prev, NEG))
    loss = -_lse(last, prev)
    loss = torch.where(loss > 1e29, torch.full_like(loss, math.inf), loss)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return (loss / target_lengths.clamp(min=1).to(loss.dtype)).mean()


def collapse(path, blank=0):
    out, prev = [], None
    for p in path:
        if p != prev and p != blank:
            out.append(int(p))
        prev = p
    return out


def greedy_decode(log_probs, blank=0):
    """log_probs (T, C) → 标签列表。"""
    return collapse(log_probs.argmax(-1).tolist(), blank)


def prefix_beam_search(log_probs, beam=8, blank=0):
    """log_probs (T, C)。每个前缀记两个数：以空白结尾的概率 pb、以非空白结尾的概率 pnb（都在对数空间）。
    每步扩展后只保留 pb ⊕ pnb 最大的 beam 个前缀。返回最可能的标签（列表）。"""
    lp = log_probs.tolist()
    ninf = -math.inf

    def add(a, b):
        if a == ninf:
            return b
        if b == ninf:
            return a
        m = max(a, b)
        return m + math.log(math.exp(a - m) + math.exp(b - m))

    beams = {(): (0.0, ninf)}
    for row in lp:
        nxt = {}
        for prefix, (pb, pnb) in beams.items():
            tot = add(pb, pnb)
            for c, p in enumerate(row):
                if c == blank:
                    b, nb = nxt.get(prefix, (ninf, ninf))
                    nxt[prefix] = (add(b, tot + p), nb)
                    continue
                ext = prefix + (c,)
                b, nb = nxt.get(ext, (ninf, ninf))
                if prefix and prefix[-1] == c:
                    nxt[ext] = (b, add(nb, pb + p))           # 重复字符之间必须隔一个空白
                    b0, nb0 = nxt.get(prefix, (ninf, ninf))
                    nxt[prefix] = (b0, add(nb0, pnb + p))     # 不隔空白：合并成同一个字符
                else:
                    nxt[ext] = (b, add(nb, tot + p))
        beams = dict(sorted(nxt.items(), key=lambda kv: -add(*kv[1]))[:beam])
    best = max(beams.items(), key=lambda kv: add(*kv[1]))
    return list(best[0])
