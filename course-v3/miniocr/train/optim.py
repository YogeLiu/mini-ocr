"""第 11 章：自己写优化器、学习率调度和梯度裁剪，再和 PyTorch 自带的逐步对拍。

    SGD               随机梯度下降，可选动量与权重衰减（与 torch.optim.SGD 的公式相同）
    Adam              自适应学习率；decoupled=True 时就是 AdamW（与 torch.optim.Adam / AdamW 相同）
    warmup_cosine     先线性预热、再余弦衰减的学习率
    clip_grad_norm    把所有参数的梯度合起来看成一个长向量，长度超过 max_norm 就等比缩短

优化器的接口与 PyTorch 相同：opt.zero_grad()、opt.step()、opt.lr 可以随时修改。
所有更新都在 torch.no_grad() 下进行（更新参数这个动作本身不应进入计算图，第 9 章）。
"""
import math

import torch


class Optimizer:
    """公共部分：保存参数列表，提供 zero_grad。"""

    def __init__(self, params, lr):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = None


# >>> 练习 11.1：带动量的 SGD
class SGD(Optimizer):
    """对每个参数 p（梯度 g = p.grad；没有梯度的参数跳过）：

        g ← g + weight_decay · p                       （L2 正则，weight_decay=0 时不变）
        momentum = 0： p ← p − lr · g
        momentum > 0： v ← momentum · v + g（第一步 v = g），p ← p − lr · v

    v 是每个参数各自的"速度"，形状与 p 相同，保存在 self.state 里（键是参数在列表中的下标）。
    """

    def __init__(self, params, lr, momentum=0.0, weight_decay=0.0):
        super().__init__(params, lr)
        self.momentum, self.weight_decay = momentum, weight_decay
        self.state = {}

    @torch.no_grad()
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p
            if self.momentum:
                v = self.state.get(i)
                v = g.clone() if v is None else v.mul_(self.momentum).add_(g)
                self.state[i] = v
                g = v
            p.sub_(self.lr * g)
# <<<


# >>> 练习 11.2：Adam 与 AdamW
class Adam(Optimizer):
    """对每个参数 p（梯度 g），第 t 步（t 从 1 开始，每个参数各自计数）：

        decoupled=False（Adam）： g ← g + weight_decay · p
        decoupled=True（AdamW）： p ← p · (1 − lr · weight_decay)          （直接缩小参数，不经过 m、v）
        m ← β₁ m + (1 − β₁) g                       一阶矩：梯度的指数移动平均
        v ← β₂ v + (1 − β₂) g²                      二阶矩：梯度平方的指数移动平均
        m̂ = m / (1 − β₁ᵗ)，v̂ = v / (1 − β₂ᵗ)         偏差修正
        p ← p − lr · m̂ / (√v̂ + eps)

    m、v 初始为 0，与 p 同形状；self.state[i] = (t, m, v)。
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, decoupled=False):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps, self.weight_decay, self.decoupled = eps, weight_decay, decoupled
        self.state = {}

    @torch.no_grad()
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                if self.decoupled:
                    p.mul_(1 - self.lr * self.weight_decay)
                else:
                    g = g + self.weight_decay * p
            t, m, v = self.state.get(i, (0, torch.zeros_like(p), torch.zeros_like(p)))
            t += 1
            m.mul_(self.b1).add_((1 - self.b1) * g)
            v.mul_(self.b2).add_((1 - self.b2) * g * g)
            self.state[i] = (t, m, v)
            m_hat = m / (1 - self.b1 ** t)
            v_hat = v / (1 - self.b2 ** t)
            p.sub_(self.lr * m_hat / (v_hat.sqrt() + self.eps))
# <<<


def AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
    """AdamW = 解耦权重衰减的 Adam。默认 weight_decay=0.01，与 torch.optim.AdamW 相同。"""
    return Adam(params, lr, betas, eps, weight_decay, decoupled=True)


# >>> 练习 11.3：预热 + 余弦衰减
def warmup_cosine(step, total_steps, warmup_steps, base_lr, min_lr=0.0):
    """第 step 步（从 0 开始）的学习率：

        step < warmup_steps：  base_lr · (step + 1) / warmup_steps              线性上升
        否则：令 progress = (step − warmup_steps) / max(1, total_steps − warmup_steps)，
              min_lr + (base_lr − min_lr) · 0.5 · (1 + cos(π · progress))    从 base_lr 平滑降到 min_lr
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
# <<<


# >>> 练习 11.4：梯度裁剪
@torch.no_grad()
def clip_grad_norm(params, max_norm):
    """把所有参数的梯度看成一个长向量，计算它的 L2 长度 total = √(Σ_p Σ g²)。
    若 total > max_norm，每个梯度都乘以 max_norm / (total + 1e-6)；否则不变。返回裁剪前的 total（Python 数）。
    与 torch.nn.utils.clip_grad_norm_ 相同。没有梯度的参数跳过。"""
    grads = [p.grad for p in params if p.grad is not None]
    total = torch.sqrt(sum((g.double() ** 2).sum() for g in grads)).item()
    if total > max_norm:
        for g in grads:
            g.mul_(max_norm / (total + 1e-6))
    return total
# <<<
