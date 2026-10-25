"""A5 参考实现：策略梯度的零件。优势（组内标准化、留一）、没有梯度的组、KL 的三个估计量、裁剪的目标。
rewards 是 (N·G,)，每连续 G 个是同一道题的回答；逐 token 的量都是 (B, T−1)。"""
import torch


def group_advantages(rewards, G, std=True, eps=1e-6):
    """每组减去组内均值；std=True 时再除以（无偏）组内标准差 + eps。"""
    r = rewards.float().view(-1, G)
    a = r - r.mean(1, keepdim=True)
    if std:
        a = a / (r.std(1, keepdim=True) + eps)
    return a.view(-1)


def loo_advantages(rewards, G):
    """A_i = r_i − 其余 G − 1 个的平均。"""
    r = rewards.float().view(-1, G)
    return (r - (r.sum(1, keepdim=True) - r) / (G - 1)).view(-1)


def zero_variance_fraction(rewards, G, tol=1e-8):
    r = rewards.float().view(-1, G)
    return ((r.max(1).values - r.min(1).values) < tol).float().mean().item()


def kl_estimate(logp, ref_logp, kind="k3"):
    """log r = ref_logp − logp。k1 = −log r，k2 = (log r)²/2，k3 = (r − 1) − log r。"""
    log_r = ref_logp - logp
    if kind == "k1":
        return -log_r
    if kind == "k2":
        return log_r ** 2 / 2
    if kind == "k3":
        return torch.exp(log_r) - 1 - log_r
    raise ValueError(kind)


def clipped_objective(logp, old_logp, adv, eps_low=0.2, eps_high=None):
    """(目标, 是否被裁剪)：min(ρA, clip(ρ, 1 − ε_low, 1 + ε_high)·A)；"被裁剪" = 裁剪项严格小于未裁剪项。"""
    eps_high = eps_low if eps_high is None else eps_high
    ratio = torch.exp(logp - old_logp)
    a = ratio * adv
    b = ratio.clamp(1 - eps_low, 1 + eps_high) * adv
    return torch.minimum(a, b), b < a
