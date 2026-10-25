"""A6 的适配器：把你在 assignments/a6_data/a6/ 里写的代码接到测试上。这里只做"接线"，不放实质逻辑。"""
from __future__ import annotations

from typing import Callable

import numpy as np


# ---------- 第一部分：清洗、去重、划分 ----------
def run_latex_key(s: str) -> str:
    raise NotImplementedError


def run_canon(s: str) -> str:
    raise NotImplementedError


def run_dedup(items: list, key: Callable) -> list[int]:
    raise NotImplementedError


def run_shingles(s: str, n: int) -> set[str]:
    raise NotImplementedError


def run_jaccard(a: set, b: set) -> float:
    raise NotImplementedError


def run_minhash(sh: set[str], n_hash: int) -> np.ndarray:
    raise NotImplementedError


def run_lsh_pairs(sigs: np.ndarray, bands: int) -> set[tuple[int, int]]:
    raise NotImplementedError


def run_near_dup_clusters(sets: list[set], pairs: set, tau: float) -> list[int]:
    raise NotImplementedError


def run_group_split(groups: list, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    raise NotImplementedError


def run_decontaminate(pool_sets: list[set], test_sets: list[set], tau: float, n_hash: int, bands: int) -> list[int]:
    raise NotImplementedError

# ---------- 第二部分：多模型共识 ----------
def run_ned(a: str, b: str) -> float:
    raise NotImplementedError


def run_pairwise_agreement(preds: list[str]) -> np.ndarray:
    raise NotImplementedError


def run_triplet_consensus(preds: list[str], k: int) -> tuple[int, float, tuple]:
    raise NotImplementedError


def run_tier(score: float, hi: float, lo: float) -> str:
    raise NotImplementedError


def run_majority_vote(preds: list[str]) -> str:
    raise NotImplementedError


def run_weighted_vote(preds: list[str], weights) -> str:
    raise NotImplementedError


def run_one_coin_em(outputs: list[list[str]], iters: int) -> tuple[list[str], np.ndarray]:
    raise NotImplementedError

# ---------- 第三部分：渲染校验 ----------
def run_ink_crop(img: np.ndarray, thresh: int, min_neighbors: int) -> np.ndarray | None:
    raise NotImplementedError


def run_blur3(m: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def run_ncc(a: np.ndarray, b: np.ndarray) -> float:
    raise NotImplementedError


def run_render_score(latex: str, img: np.ndarray, size: int) -> float:
    raise NotImplementedError


def run_render_select(cands: list[str], img: np.ndarray, size: int) -> tuple[int, float, list[float]]:
    raise NotImplementedError

# ---------- 第四部分：难例挖掘与配比 ----------
def run_self_consistency(samples: list[str]) -> float:
    raise NotImplementedError


def run_formula_features(s: str) -> np.ndarray:
    raise NotImplementedError


def run_kmeans(X: np.ndarray, k: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError


def run_quota_sample(labels, weights, n: int, rng: np.random.Generator) -> np.ndarray:
    raise NotImplementedError


def run_sampling_weights(a, u, labels, f_a: float, f_u: float, tau: float, eps: float) -> np.ndarray:
    raise NotImplementedError


def run_hard_candidates(agreement, consistency, tau_a: float, tau_u: float) -> np.ndarray:
    raise NotImplementedError
