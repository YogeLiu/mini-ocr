"""A5 的适配器：把你在 assignments/a5_align/a5/ 里写的代码接到测试上。这里只做"接线"，不放实质逻辑。"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch


# ---------- 第一部分：策略梯度的零件 ----------
def run_group_advantages(rewards: torch.Tensor, G: int, std: bool) -> torch.Tensor:
    raise NotImplementedError


def run_loo_advantages(rewards: torch.Tensor, G: int) -> torch.Tensor:
    raise NotImplementedError


def run_zero_variance_fraction(rewards: torch.Tensor, G: int) -> float:
    raise NotImplementedError


def run_kl_estimate(logp: torch.Tensor, ref_logp: torch.Tensor, kind: str) -> torch.Tensor:
    raise NotImplementedError


def run_clipped_objective(logp, old_logp, adv, eps_low: float, eps_high: float | None) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError


# ---------- 第二部分：DPO ----------
def run_optimal_policy(ref_logp: torch.Tensor, rewards: torch.Tensor, beta: float) -> torch.Tensor:
    raise NotImplementedError


def run_dpo_loss(pol_w, pol_l, ref_w, ref_l, beta: float, sft_weight: float, n_w) -> tuple[torch.Tensor, dict]:
    raise NotImplementedError


def run_pairs_from_groups(rewards: torch.Tensor, G: int, min_gap: float) -> list[tuple[int, int]]:
    raise NotImplementedError


def run_hallucinated_caption(objs: list[dict], rng: np.random.Generator, kind: str) -> tuple[str, tuple[str, str, str]]:
    raise NotImplementedError


# ---------- 第三部分：GRPO、DAPO 与奖励 ----------
def run_aggregate(per_token: torch.Tensor, mask: torch.Tensor, mode: str, max_len: int | None) -> torch.Tensor:
    raise NotImplementedError


def run_grpo_loss(logp, old_logp, adv, mask, ref_logp, beta: float, eps_low: float, eps_high: float | None,
                  mode: str) -> tuple[torch.Tensor, dict]:
    raise NotImplementedError


def run_overlong_penalty(length: int, max_len: int, cache: int) -> float:
    raise NotImplementedError


def run_keep_groups(rewards: torch.Tensor, G: int) -> torch.Tensor:
    raise NotImplementedError


def run_well_formed(task: str, pred: str) -> bool:
    raise NotImplementedError


def run_task_reward(task: str, pred: str, ref: str) -> float:
    raise NotImplementedError


# ---------- 第四部分：采样 ----------
def get_vlm(vocab_size: int, img_id: int) -> torch.nn.Module:
    """新初始化的小 VLM（与 A3 的 get_vlm 相同：ViT 32×32、patch 8、d 32、1 层；2×2 合并连接器；GPT block_size 128、2 层、d 32）。"""
    raise NotImplementedError


def run_vlm_logits(model: torch.nn.Module, images: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def run_sample(model: torch.nn.Module, images: torch.Tensor, prompt: torch.Tensor, max_new: int, eos: int,
               temperature: float, generator: torch.Generator | None, banned: tuple = ()) -> torch.Tensor:
    raise NotImplementedError


def run_rollout(model: torch.nn.Module, items: list[tuple[np.ndarray, list[int]]], G: int, max_new: int, eos: int,
                temperature: float, generator: torch.Generator | None) -> tuple[list[list[int]], list[bool]]:
    """items：[(图像 uint8 (k, 3, H, W), 提示编号)]；图像在送进模型前按 x / 127.5 − 1 转成 float。"""
    raise NotImplementedError


def run_make_batch(prompts: list[list[int]], responses: list[list[int]], finished: list[bool], eos: int,
                   pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError


def run_policy_logprobs(model: torch.nn.Module, images: list[torch.Tensor], ids: torch.Tensor) -> torch.Tensor:
    """images：每条序列一个 float 张量 (k, 3, H, W)，按序列的顺序。"""
    raise NotImplementedError


# ---------- 第五部分：遗忘与配比 ----------
def run_pareto_front(points: list[tuple[float, float]]) -> list[int]:
    raise NotImplementedError


def run_mixed_batch(rng: np.random.Generator, B: int, ratio: float, make_new: Callable, make_old: Callable) -> tuple[list, list[bool]]:
    raise NotImplementedError


def run_rft_select(rewards: torch.Tensor, G: int, threshold: float, k: int) -> list[int]:
    raise NotImplementedError


def run_probe_kl(logits_new: torch.Tensor, logits_base: torch.Tensor, mask: torch.Tensor) -> float:
    raise NotImplementedError
