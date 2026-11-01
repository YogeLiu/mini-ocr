"""维护者参考接线；miniocr 参考答案只能在这里导入，学生 mechanism 不可导入。"""
import math
import torch
from miniocr.autograd import Value, numerical_grad
from miniocr.nn.losses import softmax, cross_entropy, softmax_loss_vectorized
from miniocr.train.optim import AdamW, warmup_cosine, clip_grad_norm
from miniocr.train.checkpoint import save_checkpoint, load_checkpoint


def make_value(data):
    return Value(data)


def run_numerical_grad(f, xs, h=1e-5):
    if not math.isfinite(h) or h <= 0:
        raise ValueError("h must be positive and finite")
    return numerical_grad(f, xs, h)


def _matrix(x):
    if x.ndim != 2 or min(x.shape) <= 0 or not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError("expected nonempty finite floating matrix")


def _labels(z, y):
    _matrix(z)
    if y.shape != (len(z),) or y.dtype != torch.long or y.device != z.device or ((y < 0) | (y >= z.shape[1])).any():
        raise ValueError("invalid labels")


def run_softmax(z):
    _matrix(z)
    return softmax(z)


def run_cross_entropy(logits, targets):
    _labels(logits, targets)
    return cross_entropy(logits, targets)


def run_softmax_classifier(W, X, y, reg=0.0):
    _matrix(W)
    _matrix(X)
    if X.shape[1] != W.shape[0] or not math.isfinite(reg) or reg < 0:
        raise ValueError("invalid classifier dimensions/reg")
    _labels(X @ W, y)
    return softmax_loss_vectorized(W, X, y, reg)


def run_mlp(x, w1, b1, w2, b2):
    for t in (x, w1, w2):
        _matrix(t)
    if x.shape[1] != w1.shape[0] or w1.shape[1] != w2.shape[0] or b1.shape != (w1.shape[1],) or b2.shape != (w2.shape[1],):
        raise ValueError("invalid MLP dimensions")
    hidden = x @ w1 + b1
    return torch.where(hidden > 0, hidden, torch.zeros_like(hidden)) @ w2 + b2


class Classifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes):
        super().__init__()
        if any(type(v) is not int for v in (input_dim, hidden_dim, n_classes)) or min(input_dim, n_classes) <= 0 or hidden_dim < 0:
            raise ValueError("invalid dimensions")
        self.hidden_dim = hidden_dim
        def weight(d, c):
            return torch.nn.Parameter(torch.randn(d, c) * math.sqrt(2 / d))
        def bias(c):
            return torch.nn.Parameter(torch.zeros(c))
        if hidden_dim:
            self.w1, self.b1 = weight(input_dim, hidden_dim), bias(hidden_dim)
            self.w2, self.b2 = weight(hidden_dim, n_classes), bias(n_classes)
        else:
            self.w, self.b = weight(input_dim, n_classes), bias(n_classes)

    def forward(self, x):
        if self.hidden_dim:
            return run_mlp(x, self.w1, self.b1, self.w2, self.b2)
        _matrix(x)
        if x.shape[1] != self.w.shape[0]:
            raise ValueError("invalid input dimension")
        return x @ self.w + self.b


def get_classifier(input_dim, hidden_dim, n_classes):
    return Classifier(input_dim, hidden_dim, n_classes)


def get_adamw(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
    if len(betas) != 2 or not all(math.isfinite(v) for v in (lr, *betas, eps, weight_decay)) or lr < 0 or weight_decay < 0 or eps <= 0 or any(not 0 <= b < 1 for b in betas):
        raise ValueError("invalid optimizer hyperparameters")
    return AdamW(params, lr, betas, eps, weight_decay)


def run_schedule(step, total_steps, warmup_steps, base_lr, min_lr=0.0):
    if any(type(v) is not int for v in (step, total_steps, warmup_steps)) or step < 0 or total_steps <= 0 or not 0 <= warmup_steps <= total_steps or not all(math.isfinite(v) for v in (base_lr, min_lr)) or not 0 <= min_lr <= base_lr:
        raise ValueError("invalid schedule")
    return warmup_cosine(step, total_steps, warmup_steps, base_lr, min_lr)


def run_clip_grad_norm(params, max_norm):
    if not math.isfinite(max_norm) or max_norm < 0:
        raise ValueError("invalid max_norm")
    return clip_grad_norm(params, max_norm)
