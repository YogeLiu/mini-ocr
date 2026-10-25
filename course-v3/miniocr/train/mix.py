"""第 32 章：遗忘与能力配比。

    pareto_front   两个指标都越大越好时，不被任何其他点"支配"的点
    mixed_batch    一个批里按比例混入"复习"的旧任务样本（重放）
    rft_select     拒绝采样微调（RFT）：每组里挑出奖励够高的回答，当作 SFT 的数据
    probe_kl       不需要标签的遗忘指标：在一组固定的探测输入上，新模型和起点模型的输出分布差多远
"""
import torch
import torch.nn.functional as F


# >>> 练习 32.1：帕累托前沿
def pareto_front(points):
    """points：[(a, b), …]，两个指标都是越大越好。返回前沿上的点的下标（升序）。
    点 i 被点 j 支配：a_j ≥ a_i 且 b_j ≥ b_i，并且至少一个严格大于。不被任何点支配的点在前沿上。"""
    out = []
    for i, (a, b) in enumerate(points):
        if not any((a2 >= a and b2 >= b) and (a2 > a or b2 > b) for j, (a2, b2) in enumerate(points) if j != i):
            out.append(i)
    return out
# <<<


# >>> 练习 32.2：重放
def mixed_batch(rng, B, ratio, make_new, make_old):
    """一个批 B 条样本，其中 round(ratio · B) 条来自旧任务（make_old(rng)），其余来自新任务（make_new(rng)）。
    返回 (样本列表, 每条是否来自旧任务的列表)。旧样本和新样本随机打乱顺序（rng.permutation）。"""
    n_old = int(round(ratio * B))
    items = [(make_old(rng), True) for _ in range(n_old)] + [(make_new(rng), False) for _ in range(B - n_old)]
    order = rng.permutation(B)
    return [items[i][0] for i in order], [items[i][1] for i in order]
# <<<


# >>> 练习 32.3：拒绝采样
def rft_select(rewards, G, threshold=1.0, k=1):
    """rewards (N·G,)，每连续 G 个是同一道题的回答。每组里挑奖励 ≥ threshold 的回答，按奖励从高到低
    最多取 k 个（奖励相同时下标小的在前）。返回全局下标的列表（按组的顺序）。"""
    r = rewards.float().view(-1, G)
    out = []
    for g in range(r.shape[0]):
        order = sorted(range(G), key=lambda i: (-r[g, i].item(), i))
        out += [g * G + i for i in order if r[g, i] >= threshold][:k]
    return out
# <<<


# >>> 练习 32.4：探测 KL
@torch.no_grad()
def probe_kl(logits_new, logits_base, mask):
    """logits_*：(B, T, V)，两个模型在同一批探测输入上的输出；mask (B, T)：哪些位置算数。
    返回每个位置精确的 KL(base ‖ new) 按 mask 的平均：Σ_v p_base log(p_base / p_new)。
    方向：期望对起点模型求，衡量"起点模型会写的东西，新模型还认不认"。"""
    lb = F.log_softmax(logits_base.float(), -1)
    ln = F.log_softmax(logits_new.float(), -1)
    kl = (lb.exp() * (lb - ln)).sum(-1)
    m = mask.float()
    return ((kl * m).sum() / m.sum()).item()
# <<<
