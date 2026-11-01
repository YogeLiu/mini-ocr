"""第 25 章：组内优势与 GRPO 损失的显式实现（用于机制教学、单元测试和 TRL 对拍）。

约定：
- 同一提示的 G 个采样组成一组；优势只在组内比较，不需要价值网络。
- 损失按每条序列对 token 取平均，再对序列取平均（与 TRL loss_type='grpo' 相同口径）。
- KL 使用 k3 估计 E_pi[(pi_ref/pi) - log(pi_ref/pi) - 1]，逐 token 计算。
"""
import torch


# >>> 练习 25.1：组内归一化优势
def group_advantages(rewards, group_size, *, scale=True, eps=1e-4):
    """rewards 形状 [P*G]，按连续的 G 个为一组（同一提示）；返回同形状优势。

    scale=True：(r - mean_g) / (std_g + eps)，std 为组内总体标准差；
    scale=False：只减组均值（Dr. GRPO 的常见做法）。组内全相同奖励时优势为 0。
    """
    if group_size < 1 or rewards.numel() % group_size:
        raise ValueError('rewards 的长度必须是 group_size 的整数倍')
    r = rewards.float().view(-1, group_size)
    centered = r - r.mean(dim=1, keepdim=True)
    if not scale:
        return centered.reshape(-1)
    return (centered / (r.std(dim=1, unbiased=False, keepdim=True) + eps)).reshape(-1)
# <<<


# >>> 练习 25.2：裁剪代理目标、KL 惩罚与每序列平均
def grpo_loss(logp_new, logp_old, logp_ref, advantages, mask, *, epsilon=.2, beta=.04):
    """逐 token 裁剪目标减去 beta*KL（k3），每条序列按自身有效 token 数平均，再对序列取平均。

    logp_new/old/ref 形状 [N,T]（old 与 ref 不反传）；advantages [N]；mask [N,T] 为响应 token。
    返回 (loss, 诊断字典)。ratio=exp(new-old)；优势为正时超过 1+epsilon 饱和，为负时低于 1-epsilon 饱和。
    """
    if not 0 < epsilon < 1 or beta < 0:
        raise ValueError('需要 0<epsilon<1 且 beta>=0')
    if logp_new.shape != mask.shape or advantages.shape[0] != logp_new.shape[0]:
        raise ValueError('形状不匹配')
    mask = mask.float()
    adv = advantages.detach().float().unsqueeze(1)
    ratio = torch.exp(logp_new - logp_old.detach())
    surrogate = torch.minimum(ratio * adv, ratio.clamp(1 - epsilon, 1 + epsilon) * adv)
    log_ref_over_new = logp_ref.detach() - logp_new
    kl = torch.exp(log_ref_over_new) - log_ref_over_new - 1
    per_token = -(surrogate - beta * kl)
    lengths = mask.sum(dim=1).clamp(min=1)
    per_sequence = (per_token * mask).sum(dim=1) / lengths
    clipped = ((ratio - 1).abs() > epsilon).float()
    denom = mask.sum().clamp(min=1)
    diagnostics = dict(clip_fraction=float((clipped * mask).sum() / denom),
                       kl_mean=float((kl.detach() * mask).sum() / denom),
                       ratio_mean=float((ratio.detach() * mask).sum() / denom))
    return per_sequence.mean(), diagnostics
# <<<
