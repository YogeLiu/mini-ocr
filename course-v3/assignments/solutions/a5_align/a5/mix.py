"""A5 参考实现：遗忘与配比。帕累托前沿、重放、拒绝采样（RFT）的挑选、探测 KL。"""
import torch

from .nn import log_softmax


def pareto_front(points):
    """两个指标都越大越好；不被任何其他点支配的点的下标（升序）。"""
    out = []
    for i, (a, b) in enumerate(points):
        if not any(a2 >= a and b2 >= b and (a2 > a or b2 > b) for j, (a2, b2) in enumerate(points) if j != i):
            out.append(i)
    return out


def mixed_batch(rng, B, ratio, make_new, make_old):
    """round(ratio·B) 条旧任务 + 其余新任务，先旧后新排好，再按 rng.permutation(B) 打乱。返回 (样本, 是否旧任务)。"""
    n_old = int(round(ratio * B))
    items = [(make_old(rng), True) for _ in range(n_old)] + [(make_new(rng), False) for _ in range(B - n_old)]
    order = rng.permutation(B)
    return [items[i][0] for i in order], [items[i][1] for i in order]


def rft_select(rewards, G, threshold=1.0, k=1):
    """每组奖励 ≥ threshold 的回答，按奖励从高到低（相同时下标小的在前）最多取 k 个。全局下标。"""
    r = rewards.float().view(-1, G)
    out = []
    for g in range(r.shape[0]):
        order = sorted(range(G), key=lambda i: (-r[g, i].item(), i))
        out += [g * G + i for i in order if r[g, i] >= threshold][:k]
    return out


@torch.no_grad()
def probe_kl(logits_new, logits_base, mask):
    """mask 位置上 KL(p_base ‖ p_new) 的平均。"""
    lb, ln = log_softmax(logits_base.float(), -1), log_softmax(logits_new.float(), -1)
    kl = (lb.exp() * (lb - ln)).sum(-1)
    m = mask.float()
    return ((kl * m).sum() / m.sum()).item()
