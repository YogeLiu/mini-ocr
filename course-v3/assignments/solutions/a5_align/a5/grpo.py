"""A5 参考实现：GRPO 与 DAPO 的改动。三种汇总方式、GRPO 损失、软超长惩罚、动态采样。"""
import torch

from .pg import clipped_objective, kl_estimate


def aggregate(per_token, mask, mode="token", max_len=None):
    """sample：每条先平均再对条平均；token：所有 token 一起平均；constant：Σ / (B · max_len)，max_len 默认 T。"""
    m = mask.float()
    if mode == "sample":
        return ((per_token * m).sum(1) / m.sum(1)).mean()
    if mode == "token":
        return (per_token * m).sum() / m.sum()
    if mode == "constant":
        return (per_token * m).sum() / (m.shape[0] * (max_len or m.shape[1]))
    raise ValueError(mode)


def grpo_loss(logp, old_logp, adv, mask, ref_logp=None, beta=0.0, eps_low=0.2, eps_high=None, mode="token", max_len=None):
    """逐 token：−裁剪目标 + β·k3（β > 0 且有 ref_logp 时），按 mode 汇总。统计：clip_frac、kl（k3 的 token 平均）、ratio_max。"""
    obj, clipped = clipped_objective(logp, old_logp, adv[:, None], eps_low, eps_high)
    per = -obj
    kl = torch.zeros_like(logp)
    if ref_logp is not None:
        kl = kl_estimate(logp, ref_logp, "k3")
        if beta:
            per = per + beta * kl
    loss = aggregate(per, mask, mode, max_len)
    m = mask.bool()
    with torch.no_grad():
        stats = dict(clip_frac=clipped[m].float().mean().item(), kl=kl[m].mean().item(),
                     ratio_max=torch.exp(logp - old_logp)[m].max().item())
    return loss, stats


def overlong_penalty(length, max_len, cache):
    if length <= max_len - cache:
        return 0.0
    if length <= max_len:
        return (max_len - cache - length) / cache
    return -1.0


def keep_groups(rewards, G, tol=1e-8):
    r = rewards.float().view(-1, G)
    return (r.max(1).values - r.min(1).values) > tol
