"""A6 参考实现的适配器。"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .a6 import clean, consensus, mining, text, verify


# ---------- 第一部分：清洗、去重、划分 ----------
def run_latex_key(s: str) -> str:
    return clean.latex_key(s)


def run_canon(s: str) -> str:
    return clean.canon(s)


def run_dedup(items: list, key: Callable) -> list[int]:
    return clean.dedup(items, key)


def run_shingles(s: str, n: int) -> set[str]:
    return clean.shingles(s, n)


def run_jaccard(a: set, b: set) -> float:
    return clean.jaccard(a, b)


def run_minhash(sh: set[str], n_hash: int) -> np.ndarray:
    return clean.minhash(sh, n_hash)


def run_lsh_pairs(sigs: np.ndarray, bands: int) -> set[tuple[int, int]]:
    return clean.lsh_pairs(sigs, bands)


def run_near_dup_clusters(sets: list[set], pairs: set, tau: float) -> list[int]:
    return clean.near_dup_clusters(sets, pairs, tau)


def run_group_split(groups: list, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    return clean.group_split(groups, val_frac, seed)


def run_decontaminate(pool_sets: list[set], test_sets: list[set], tau: float, n_hash: int, bands: int) -> list[int]:
    return clean.decontaminate(pool_sets, test_sets, tau, n_hash, bands)

# ---------- 第二部分：多模型共识 ----------
def run_ned(a: str, b: str) -> float:
    return text.ned(a, b)


def run_pairwise_agreement(preds: list[str]) -> np.ndarray:
    return consensus.pairwise_agreement(preds)


def run_triplet_consensus(preds: list[str], k: int) -> tuple[int, float, tuple]:
    return consensus.triplet_consensus(preds, k=k)


def run_tier(score: float, hi: float, lo: float) -> str:
    return consensus.tier(score, hi, lo)


def run_majority_vote(preds: list[str]) -> str:
    return consensus.majority_vote(preds)


def run_weighted_vote(preds: list[str], weights) -> str:
    return consensus.weighted_vote(preds, weights)


def run_one_coin_em(outputs: list[list[str]], iters: int) -> tuple[list[str], np.ndarray]:
    return consensus.one_coin_em(outputs, iters)

# ---------- 第三部分：渲染校验 ----------
def run_ink_crop(img: np.ndarray, thresh: int, min_neighbors: int) -> np.ndarray | None:
    return verify.ink_crop(img, thresh, min_neighbors)


def run_blur3(m: np.ndarray) -> np.ndarray:
    return verify.blur3(m)


def run_ncc(a: np.ndarray, b: np.ndarray) -> float:
    return verify.ncc(a, b)


def run_render_score(latex: str, img: np.ndarray, size: int) -> float:
    return verify.render_score(latex, img, size)


def run_render_select(cands: list[str], img: np.ndarray, size: int) -> tuple[int, float, list[float]]:
    return verify.render_select(cands, img, size)

# ---------- 第四部分：难例挖掘与配比 ----------
def run_self_consistency(samples: list[str]) -> float:
    return mining.self_consistency(samples)


def run_formula_features(s: str) -> np.ndarray:
    return mining.formula_features(s)


def run_kmeans(X: np.ndarray, k: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    return mining.kmeans(X, k, rng)


def run_quota_sample(labels, weights, n: int, rng: np.random.Generator) -> np.ndarray:
    return mining.quota_sample(labels, weights, n, rng)


def run_sampling_weights(a, u, labels, f_a: float, f_u: float, tau: float, eps: float) -> np.ndarray:
    return mining.sampling_weights(a, u, labels, f_a, f_u, tau, eps)


def run_hard_candidates(agreement, consistency, tau_a: float, tau_u: float) -> np.ndarray:
    return mining.hard_candidates(agreement, consistency, tau_a, tau_u)
