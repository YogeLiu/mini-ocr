"""第 31 章：GRPO 与 DAPO 的四个改动。和第 29 章的零件（优势、KL、裁剪）一起，组成完整的一步更新。

    aggregate         逐 token 的损失 → 一个数：按样本（GRPO）、按 token（DAPO）、除以常数（Dr. GRPO）
    grpo_loss         −裁剪目标 + β·k3，按 aggregate 汇总；返回损失和监控用的统计
    overlong_penalty  DAPO 的"软超长惩罚"：回答越接近长度上限，奖励扣得越多
    keep_groups       DAPO 的"动态采样"：只保留组内奖励不全相同的组

DAPO（Yu 等，2025）相对 GRPO 的四个改动：
    1. clip-higher：裁剪的上边放宽（ε_low = 0.2，ε_high = 0.28），让低概率的好 token 能被推得更高
    2. 动态采样：去掉全对、全错（没有梯度）的组，再补采，保持每步有效的组数
    3. token 级的损失：所有 token 一起平均，而不是先每条回答平均、再对回答平均
    4. 超长奖励整形：被截断的回答不参与训练（过滤），或者按长度扣分（软惩罚）
另外 DAPO 去掉了 KL 项（β = 0）。
"""
import torch

from .pg import clipped_objective, kl_estimate


# >>> 练习 31.3：三种汇总方式
def aggregate(per_token, mask, mode="token", max_len=None):
    """per_token、mask：(B, T)。返回一个数。
        sample：每条回答先在自己的 token 上平均（Σ_t x·m / Σ_t m），再对 B 条回答平均（GRPO 原文）；
        token：Σ x·m / Σ m，整批所有回答的 token 一起平均（DAPO）；
        constant：Σ x·m / (B · max_len)，除以一个和长度无关的常数（Dr. GRPO）；max_len 为 None 时用 T。
    每条回答至少有一个 token（mask 的每行和 ≥ 1）。"""
    m = mask.float()
    if mode == "sample":
        return ((per_token * m).sum(1) / m.sum(1)).mean()
    if mode == "token":
        return (per_token * m).sum() / m.sum()
    if mode == "constant":
        return (per_token * m).sum() / (m.shape[0] * (max_len or m.shape[1]))
    raise ValueError(mode)
# <<<


# >>> 练习 31.4：GRPO 的损失
def grpo_loss(logp, old_logp, adv, mask, ref_logp=None, beta=0.0, eps_low=0.2, eps_high=None, mode="token"):
    """logp (B, T)（有梯度），old_logp、ref_logp (B, T)（不带梯度），adv (B,)，mask (B, T)。
    逐 token 的损失 = −clipped_objective(logp, old_logp, adv[:, None], eps_low, eps_high)[0]，
    β > 0 时再加 β · kl_estimate(logp, ref_logp, "k3")；用 aggregate(…, mode) 汇总。
    返回 (损失, 统计)，统计（不带梯度）：clip_frac（被裁剪的 token 占回答 token 的比例）、kl（k3 按 token 的平均，
    没有 ref_logp 时为 0）、ratio_max（ρ 的最大值，只看回答的 token）。"""
    obj, clipped = clipped_objective(logp, old_logp, adv[:, None], eps_low, eps_high)
    per = -obj
    kl = torch.zeros_like(logp)
    if ref_logp is not None:
        kl = kl_estimate(logp, ref_logp, "k3")
        if beta:
            per = per + beta * kl
    loss = aggregate(per, mask, mode)
    m = mask.bool()
    with torch.no_grad():
        stats = dict(clip_frac=clipped[m].float().mean().item(), kl=kl[m].mean().item(),
                     ratio_max=torch.exp(logp - old_logp)[m].max().item())
    return loss, stats
# <<<


# >>> 练习 31.5：软超长惩罚
def overlong_penalty(length, max_len, cache):
    """DAPO 的软超长惩罚（加到奖励上，≤ 0）：
        length ≤ max_len − cache：0
        max_len − cache < length ≤ max_len：(max_len − cache − length) / cache，从 0 线性降到 −1
        length > max_len：−1"""
    if length <= max_len - cache:
        return 0.0
    if length <= max_len:
        return (max_len - cache - length) / cache
    return -1.0
# <<<


def keep_groups(rewards, G, tol=1e-8):
    """(N·G,) → (N,) 的 bool：组内最大 − 最小 > tol 的组保留。"""
    r = rewards.float().view(-1, G)
    return (r.max(1).values - r.min(1).values) > tol
