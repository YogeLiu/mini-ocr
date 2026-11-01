"""第 3 章：概率、softmax 与交叉熵。

    naive_softmax              不减最大值的 softmax，只用来演示溢出（不要在模型里用它）
    softmax                    数值稳定的 softmax：先减去最大值，再取指数、归一化；fp16/bf16 用 fp32 中间计算
    log_softmax                数值稳定的 log-softmax：z − max − log Σ exp(z − max)
    cross_entropy              交叉熵，logits 形状 (..., V)，targets 形状 (...)；支持 ignore_index 和三种 reduction
    sequence_mean_cross_entropy 先在每条序列内按 token 平均，再对序列平均（和按 token 平均不同，见 3.7 节）
    softmax_ce_grad            softmax 交叉熵对 logits 的梯度 (p − y) / N，手推的结果
    softmax_loss_vectorized    CS231n 风格的线性 softmax 分类器：返回 (loss, dW)，不用 autograd
    numerical_gradient         中心差分的数值梯度，用来检查解析梯度
    rel_error                  两个张量的最大相对误差

本模块的函数只用张量的基本运算（max、exp、log、sum、gather、矩阵乘法），不调用
torch.softmax / F.cross_entropy；测试会拿它们和 PyTorch 的实现对拍。
"""
import torch


def naive_softmax(z, dim=-1):
    """教科书公式原样照抄：exp(z) / Σ exp(z)。logits 稍大一点就会溢出成 inf，再变成 nan（3.3 节）。"""
    e = z.exp()
    return e / e.sum(dim, keepdim=True)


# >>> 练习 3.1：稳定的 softmax
def softmax(z, dim=-1):
    """z 任意形状，沿 dim 做 softmax，返回同形状的概率（沿 dim 和为 1）。
        m = max(z)（沿 dim，keepdim）    减去最大值不改变结果（分子分母同乘 e^{−m}）
        e = exp(z − m)                   最大的那一项是 e^0 = 1，不会上溢；分母 ≥ 1，不会出现 0/0
        p = e / Σ e
    减最大值只保证每一项 ≤ 1，不保证求和不溢出：fp16 的上限是 65504，类别数 ≥ 65505 时
    全 0 的 logits 求和就会溢出。所以 fp16 / bf16 输入先转成 fp32 算，最后再转回原 dtype（3.3.4 节）。
    """
    zf = z.float() if z.dtype in (torch.float16, torch.bfloat16) else z
    m = zf.amax(dim, keepdim=True)
    e = (zf - m).exp()
    return (e / e.sum(dim, keepdim=True)).to(z.dtype)
# <<<


# >>> 练习 3.2：稳定的 log-softmax
def log_softmax(z, dim=-1):
    """log p_i = z_i − log Σ_j exp(z_j) = (z_i − m) − log Σ_j exp(z_j − m)，m = max(z)。
    不要写成 softmax(z).log()：p 下溢成 0 时会得到 −inf（3.3.5 节）。
    和 softmax 一样，fp16 / bf16 输入先转成 fp32 算，再转回原 dtype。"""
    zf = z.float() if z.dtype in (torch.float16, torch.bfloat16) else z
    m = zf.amax(dim, keepdim=True)
    s = zf - m
    return (s - s.exp().sum(dim, keepdim=True).log()).to(z.dtype)
# <<<


# >>> 练习 3.3：带 ignore_index 的交叉熵
def cross_entropy(logits, targets, ignore_index=-100, reduction="mean"):
    """logits 形状 (..., V)，最后一维是类别（注意：和 F.cross_entropy 的"类别在第 1 维"不同）；
    targets 形状 (...)，整数（long），取值 0..V−1 或 ignore_index。
        logp = log_softmax(logits)                         (..., V)
        nll  = −logp[..., targets]                         (...)，用 gather 取出正确类别那一项
        targets == ignore_index 的位置：nll 置 0，并且不计入分母
    reduction:
        "none"  返回每个位置的 nll，形状 (...)，被忽略的位置为 0
        "sum"   所有有效位置的 nll 之和
        "mean"  有效位置的平均（按 token 平均）。全部被忽略时是 0/0 = nan，和 PyTorch 一致。
    """
    logp = log_softmax(logits, dim=-1)
    valid = targets != ignore_index
    safe = torch.where(valid, targets, torch.zeros_like(targets))       # 被忽略的位置先随便取一个合法下标
    nll = -logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    nll = torch.where(valid, nll, torch.zeros_like(nll))
    if reduction == "none":
        return nll
    if reduction == "sum":
        return nll.sum()
    if reduction == "mean":
        return nll.sum() / valid.sum()
    raise ValueError(f"未知的 reduction: {reduction!r}")
# <<<


# >>> 练习 3.4：按序列平均的交叉熵
def sequence_mean_cross_entropy(logits, targets, ignore_index=-100):
    """logits 形状 (B, T, V)，targets 形状 (B, T)。
    先在每条序列内部对有效 token 求平均，得到 B 个数，再对这 B 个数求平均。
    这样每条序列的权重相同；按 token 平均（cross_entropy 的 "mean"）则是每个 token 的权重相同。
    没有任何有效 token 的序列不参加平均。"""
    nll = cross_entropy(logits, targets, ignore_index, reduction="none")    # (B, T)
    n = (targets != ignore_index).sum(-1)                                   # (B,)
    keep = n > 0
    per_seq = nll.sum(-1)[keep] / n[keep]
    return per_seq.mean()
# <<<


# >>> 练习 3.5：softmax 交叉熵对 logits 的梯度
def softmax_ce_grad(logits, targets):
    """logits 形状 (N, C)，targets 形状 (N,)。损失是 N 个样本的平均交叉熵。
    返回 ∂L/∂logits，形状 (N, C)：(softmax(logits) − onehot(targets)) / N。"""
    N = logits.shape[0]
    g = softmax(logits, dim=-1).clone()
    g[torch.arange(N), targets] -= 1
    return g / N
# <<<


# >>> 练习 3.6：向量化的 softmax 分类器损失
def softmax_loss_vectorized(W, X, y, reg=0.0):
    """CS231n 风格的线性 softmax 分类器，不用循环，也不用 autograd。
        W 形状 (D, C)，X 形状 (N, D)，y 形状 (N,)，reg 是 L2 正则系数
        scores = X @ W                         (N, C)
        loss   = mean_i[ −log softmax(scores_i)[y_i] ] + reg · Σ W²
        dW     = Xᵀ @ (P − Y) / N + 2 · reg · W     (D, C)，P 是概率，Y 是 one-hot
    返回 (loss 标量张量, dW)。"""
    scores = X @ W
    loss = cross_entropy(scores, y) + reg * (W * W).sum()
    dscores = softmax_ce_grad(scores, y)
    dW = X.T @ dscores + 2 * reg * W
    return loss, dW
# <<<


# >>> 练习 3.7：数值梯度
def numerical_gradient(f, x, h=1e-5):
    """中心差分：∂f/∂x_i ≈ (f(x + h e_i) − f(x − h e_i)) / (2h)。
    f 接收和 x 同形状的张量、返回标量（Python 数或 0 维张量）。x 会被临时修改，返回前恢复原值。
    返回和 x 同形状的梯度。建议用 float64，h 取 1e-5 左右（3.9.2 节）。"""
    grad = torch.zeros_like(x)
    flat, g = x.view(-1), grad.view(-1)
    for i in range(flat.numel()):
        old = flat[i].item()
        flat[i] = old + h
        fp = float(f(x))
        flat[i] = old - h
        fm = float(f(x))
        flat[i] = old
        g[i] = (fp - fm) / (2 * h)
    return grad
# <<<


def rel_error(a, b, eps=1e-12):
    """max |a − b| / max(|a| + |b|, eps)，逐元素算，取最大值。1e-7 以下通常说明解析梯度是对的（float64）。"""
    return ((a - b).abs() / (a.abs() + b.abs()).clamp_min(eps)).max().item()
