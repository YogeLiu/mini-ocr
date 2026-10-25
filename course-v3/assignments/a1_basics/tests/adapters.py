"""A1 的适配器：把你在 assignments/a1_basics/a1/ 里写的代码接到测试上。

测试只通过这里的函数调用你的实现。你的代码怎么组织（文件名、类名、函数名）由你决定；
在每个函数里调用你自己的实现，把结果按说明的格式返回即可。这里不应该有实质性的逻辑。
"""
from __future__ import annotations

import os
from typing import IO, BinaryIO

import numpy as np
import torch


# ---------- 问题 data ----------
def run_build_split(seed: int, n_per_class: int, distort: float, noise: float) -> tuple[np.ndarray, np.ndarray]:
    """返回 (X, y)：X float32 (N, 1, 32, 32)，墨迹 = 1、背景 = 0；y int64 (N,)，62 类。"""
    raise NotImplementedError


def run_remove_overlap(X: np.ndarray, y: np.ndarray, X_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """去掉 X 中与 X_ref 里某张图逐像素相同的样本，返回 (X, y, 去掉的个数)，保持原来的顺序。"""
    raise NotImplementedError


def run_get_batch(X: np.ndarray, y: np.ndarray, batch_size: int,
                  generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """有放回地随机取 batch_size 个样本，随机性只来自 generator。返回 (xb, yb) 两个张量。"""
    raise NotImplementedError


# ---------- 问题 linear / conv / pool / batchnorm / softmax ----------
def run_linear(d_in: int, d_out: int, weight: torch.Tensor, bias: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """构造你的 Linear(d_in, d_out)，把 weight (d_out, d_in)、bias (d_out,) 装进去，返回它对 x 的输出。"""
    raise NotImplementedError


def get_linear(d_in: int, d_out: int) -> torch.nn.Module:
    """返回一个新初始化的 Linear(d_in, d_out)（用来检查初始化）。参数名必须是 weight、bias。"""
    raise NotImplementedError


def run_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
               stride: int, padding: int) -> torch.Tensor:
    """二维卷积，结果与 F.conv2d(x, weight, bias, stride, padding) 相同，梯度也要相同。"""
    raise NotImplementedError


def get_conv2d(c_in: int, c_out: int, k: int) -> torch.nn.Module:
    """返回一个新初始化的 Conv2d(c_in, c_out, k)（用来检查初始化）。参数名必须是 weight、bias。"""
    raise NotImplementedError


def run_max_pool2d(x: torch.Tensor, k: int) -> torch.Tensor:
    """与 F.max_pool2d(x, k) 相同。"""
    raise NotImplementedError


def run_batch_norm2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, running_mean: torch.Tensor,
                     running_var: torch.Tensor, training: bool, momentum: float, eps: float) -> torch.Tensor:
    """与 F.batch_norm 相同，包括 training=True 时就地更新 running_mean 和 running_var。"""
    raise NotImplementedError


def run_log_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    raise NotImplementedError


def run_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float) -> torch.Tensor:
    """返回标量，与 F.cross_entropy(logits, targets, label_smoothing=label_smoothing) 相同。"""
    raise NotImplementedError


# ---------- 问题 adamw / schedule / clip ----------
def get_adamw_cls() -> type[torch.optim.Optimizer]:
    """返回你的 AdamW 类（不是实例）。构造参数：(params, lr, betas, eps, weight_decay)。"""
    raise NotImplementedError


def run_lr_cosine_schedule(it: int, max_lr: float, min_lr: float, warmup_iters: int, cosine_cycle_iters: int) -> float:
    raise NotImplementedError


def run_clip_grad_norm(parameters, max_norm: float) -> float:
    """就地裁剪梯度，返回裁剪前的总长度。"""
    raise NotImplementedError


# ---------- 问题 char_cnn / checkpoint ----------
def get_char_cnn(num_classes: int, channels: tuple[int, ...]) -> torch.nn.Module:
    """返回新初始化的 CharCNN；state_dict 的键必须与说明书第 4 节一致。"""
    raise NotImplementedError


def run_char_cnn(num_classes: int, channels: tuple[int, ...], state_dict: dict, x: torch.Tensor) -> torch.Tensor:
    """把 state_dict 装进你的 CharCNN，切到评测模式，返回 x 的 logits (N, num_classes)。"""
    raise NotImplementedError


def run_save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int,
                        out: str | os.PathLike | BinaryIO | IO[bytes]) -> None:
    raise NotImplementedError


def run_load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer) -> int:
    """恢复 model 和 optimizer，返回保存时的 iteration。"""
    raise NotImplementedError


# ---------- 问题 train ----------
def run_train(config: dict, data: tuple, resume: str | None = None) -> tuple[list[dict], torch.nn.Module]:
    """用 config（说明书第 6 节列出的键）在 data = (X, y, Xv, yv) 上训练，CPU 上运行。
    resume 不是 None 时，从这个检查点接着训练到 config["max_iters"]。
    config["out_dir"] 非空时，把检查点存到 out_dir/ckpt.pt。
    返回 (history, model)：history 是这次运行里每一步一条的字典，至少有 "iter"（从 1 开始）和 "loss"。"""
    raise NotImplementedError
