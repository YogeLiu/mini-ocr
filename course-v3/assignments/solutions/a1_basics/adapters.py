"""A1 参考实现的适配器（与 assignments/a1_basics/tests/adapters.py 的接口相同）。"""
import torch

from .a1.data import build_split, get_batch, remove_overlap
from .a1.model import CharCNN, load_checkpoint, save_checkpoint
from .a1.nn import Conv2d, Linear, batch_norm2d, conv2d, cross_entropy, log_softmax, max_pool2d
from .a1.optim import AdamW, clip_grad_norm, lr_cosine_schedule
from .a1.train import TrainConfig, train


def run_build_split(seed, n_per_class, distort, noise):
    return build_split(seed, n_per_class, distort, noise)


def run_remove_overlap(X, y, X_ref):
    return remove_overlap(X, y, X_ref)


def run_get_batch(X, y, batch_size, generator):
    return get_batch(X, y, batch_size, generator)


def run_linear(d_in, d_out, weight, bias, x):
    m = Linear(d_in, d_out)
    m.load_state_dict({"weight": weight, "bias": bias})
    return m(x)


def get_linear(d_in, d_out):
    return Linear(d_in, d_out)


def run_conv2d(x, weight, bias, stride, padding):
    return conv2d(x, weight, bias, stride, padding)


def get_conv2d(c_in, c_out, k):
    return Conv2d(c_in, c_out, k)


def run_max_pool2d(x, k):
    return max_pool2d(x, k)


def run_batch_norm2d(x, weight, bias, running_mean, running_var, training, momentum, eps):
    return batch_norm2d(x, weight, bias, running_mean, running_var, training, momentum, eps)


def run_log_softmax(x, dim):
    return log_softmax(x, dim)


def run_cross_entropy(logits, targets, label_smoothing):
    return cross_entropy(logits, targets, label_smoothing)


def get_adamw_cls():
    return AdamW


def run_lr_cosine_schedule(it, max_lr, min_lr, warmup_iters, cosine_cycle_iters):
    return lr_cosine_schedule(it, max_lr, min_lr, warmup_iters, cosine_cycle_iters)


def run_clip_grad_norm(parameters, max_norm):
    return clip_grad_norm(parameters, max_norm)


def get_char_cnn(num_classes, channels):
    return CharCNN(num_classes, channels)


def run_char_cnn(num_classes, channels, state_dict, x):
    m = CharCNN(num_classes, channels)
    m.load_state_dict(state_dict)
    return m.eval()(x)


def run_save_checkpoint(model, optimizer, iteration, out):
    save_checkpoint(model, optimizer, iteration, out)


def run_load_checkpoint(src, model, optimizer):
    return load_checkpoint(src, model, optimizer)[0]


def run_train(config, data, resume=None):
    res = train(TrainConfig(**config), data=data, resume=resume, log=lambda *a: None)
    return res["history"], res["model"]
