"""A3 参考实现：优化器、调度、裁剪（与 A1 相同，复制过来）。"""
import math

import torch


class AdamW(torch.optim.Optimizer):
    """与 torch.optim.AdamW 相同的更新（每个参数各自计步 t，从 1 开始）：

        p ← p · (1 − lr·λ)
        m ← β₁m + (1 − β₁)g，v ← β₂v + (1 − β₂)g²
        p ← p − lr · m̂ / (√v̂ + eps)，m̂ = m / (1 − β₁ᵗ)，v̂ = v / (1 − β₂ᵗ)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"学习率不能为负：{lr}")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["t"] += 1
                t, m, v, g = state["t"], state["m"], state["v"], p.grad
                p.mul_(1 - lr * wd)
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                m_hat = m / (1 - b1 ** t)
                v_hat = v / (1 - b2 ** t)
                p.sub_(lr * m_hat / (v_hat.sqrt() + eps))
        return loss


def lr_cosine_schedule(it, max_lr, min_lr, warmup_iters, cosine_cycle_iters):
    """it < warmup_iters：线性从 0 升到 max_lr（it/warmup_iters · max_lr）；
    warmup_iters ≤ it ≤ cosine_cycle_iters：余弦从 max_lr 降到 min_lr；之后保持 min_lr。"""
    if it < warmup_iters:
        return it / warmup_iters * max_lr
    if it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / max(1, cosine_cycle_iters - warmup_iters)
        return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (max_lr - min_lr)
    return min_lr


@torch.no_grad()
def clip_grad_norm(params, max_norm, eps=1e-6):
    """所有梯度合起来的 L2 长度超过 max_norm 时，每个梯度乘以 max_norm / (total + eps)。返回裁剪前的 total。"""
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    total = torch.sqrt(sum((g.double() ** 2).sum() for g in grads)).item()
    if total > max_norm:
        for g in grads:
            g.mul_(max_norm / (total + eps))
    return total
