"""第 29 章：策略梯度的零件。第 31 章的 GRPO 就是把它们拼起来。

    group_advantages        组内标准化的优势（GRPO）：A_i = (r_i − 组内均值) / (组内标准差 + eps)
    loo_advantages          留一基线（RLOO）：A_i = r_i − 其余 G − 1 个回答的平均奖励
    zero_variance_fraction  组内奖励全相同（这一组没有梯度）的组所占比例
    kl_estimate             KL(π‖π_ref) 的逐 token 估计量 k1、k2、k3
    clipped_objective       PPO 式的裁剪目标（逐 token）：min(ρ·A, clip(ρ, 1 − ε_low, 1 + ε_high)·A)

约定：rewards 是 (N·G,) 的一维张量，每连续 G 个是同一道题的 G 个回答（rollout.py 就按这个顺序返回）。
逐 token 的量都是 (B, T−1)，与第 27 章 token_logprobs 的输出对齐。
"""
import torch


# >>> 练习 29.1：组内标准化的优势
def group_advantages(rewards, G, std=True, eps=1e-6):
    """rewards (N·G,) → (N·G,)。每组减去组内均值；std=True 时再除以组内标准差 + eps（无偏标准差，unbiased=True）。
    组内奖励全相同时，优势全为 0。"""
    r = rewards.float().view(-1, G)
    a = r - r.mean(1, keepdim=True)
    if std:
        a = a / (r.std(1, keepdim=True) + eps)
    return a.view(-1)
# <<<


# >>> 练习 29.2：留一基线
def loo_advantages(rewards, G):
    """A_i = r_i − (组内其余 G − 1 个的平均)。提示：组内总和 S，其余的平均 = (S − r_i) / (G − 1)。"""
    r = rewards.float().view(-1, G)
    base = (r.sum(1, keepdim=True) - r) / (G - 1)
    return (r - base).view(-1)
# <<<


def zero_variance_fraction(rewards, G, tol=1e-8):
    r = rewards.float().view(-1, G)
    return ((r.max(1).values - r.min(1).values) < tol).float().mean().item()


# >>> 练习 29.3：KL 的三个估计量
def kl_estimate(logp, ref_logp, kind="k3"):
    """logp、ref_logp：同一批采样 token 在当前策略 π 和参考模型 π_ref 下的对数概率（同形状）。
    记 log r = ref_logp − logp（r = π_ref / π）。逐元素返回：
        k1 = −log r
        k2 = (log r)² / 2
        k3 = (r − 1) − log r       （r = exp(log r)）
    三者在 token 从 π 采出时取平均，都估计 KL(π‖π_ref)；k1、k3 无偏，k3 每个元素 ≥ 0。"""
    log_r = ref_logp - logp
    if kind == "k1":
        return -log_r
    if kind == "k2":
        return log_r ** 2 / 2
    if kind == "k3":
        return torch.exp(log_r) - 1 - log_r
    raise ValueError(kind)
# <<<


# >>> 练习 29.4：裁剪的目标
def clipped_objective(logp, old_logp, adv, eps_low=0.2, eps_high=None):
    """逐 token 返回 (目标, 是否被裁剪)，形状与 logp 相同。adv 可以是 (B, 1)（每条回答一个优势，广播到每个 token）。
        ρ = exp(logp − old_logp)
        目标 = min(ρ·A, clip(ρ, 1 − eps_low, 1 + eps_high)·A)          eps_high 为 None 时等于 eps_low
    "被裁剪"指 min 取到了裁剪的那一项、而且它与未裁剪的那一项不同（这时这个 token 的梯度为 0）。
    训练时最大化目标，所以损失是 −目标（按 mask 平均，见第 31 章）。old_logp 必须不带梯度。"""
    eps_high = eps_low if eps_high is None else eps_high
    ratio = torch.exp(logp - old_logp)
    a = ratio * adv
    b = ratio.clamp(1 - eps_low, 1 + eps_high) * adv
    obj = torch.minimum(a, b)
    return obj, (b < a)
# <<<
