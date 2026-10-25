"""A5 参考实现：DPO。KL 约束下的最优策略、DPO 损失（可加 chosen 的 NLL）、从组内采样挑成对数据。"""
import torch

from .nn import log_sigmoid, softmax


def optimal_policy(ref_logp, rewards, beta):
    """π* = softmax(log π_ref + R / β)。"""
    return softmax(ref_logp + rewards / beta, -1)


def dpo_loss(pol_w, pol_l, ref_w, ref_l, beta=0.1, sft_weight=0.0, n_w=None):
    """返回 (平均损失, 统计)。r = β(pol − ref)，损失 −log σ(r_w − r_l)；sft_weight > 0 时加 sft_weight·(−pol_w / n_w)。
    统计不带梯度：reward_w、reward_l、margin、acc。"""
    rw, rl = beta * (pol_w - ref_w), beta * (pol_l - ref_l)
    loss = -log_sigmoid(rw - rl)
    if sft_weight:
        loss = loss + sft_weight * (-pol_w / n_w)
    with torch.no_grad():
        stats = dict(reward_w=rw.mean().item(), reward_l=rl.mean().item(), margin=(rw - rl).mean().item(),
                     acc=(rw > rl).float().mean().item())
    return loss.mean(), stats


def pairs_from_groups(rewards, G, min_gap=0.0):
    """每组：最高的当 chosen、最低的当 rejected（并列取第一个），只保留 最高 − 最低 > min_gap 的组。全局下标。"""
    r = rewards.float().view(-1, G)
    out = []
    for g in range(r.shape[0]):
        hi, lo = int(r[g].argmax()), int(r[g].argmin())
        if r[g, hi] - r[g, lo] > min_gap:
            out.append((g * G + hi, g * G + lo))
    return out
