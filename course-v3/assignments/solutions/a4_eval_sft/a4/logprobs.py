"""A4 参考实现：回答的对数概率、两种 mask 约定、熵。logits[:, t] 预测 ids[:, t + 1]。"""
import torch

from .nn import log_softmax


def token_logprobs(logits, ids):
    """(B, T, V)、(B, T) → (B, T−1)：out[b, t] = log softmax(logits[b, t])[ids[b, t + 1]]，用 float32 算。"""
    lp = log_softmax(logits[:, :-1].float(), -1)
    return lp.gather(-1, ids[:, 1:, None]).squeeze(-1)


def labels_to_mask(targets):
    """约定一 targets (B, T)（−100 = 不算）→ 约定二 mask (B, T−1)。"""
    return (targets[:, :-1] != -100).float()


def mask_to_labels(ids, mask):
    tg = torch.where(mask.bool(), ids[:, 1:], torch.full_like(ids[:, 1:], -100))
    return torch.cat([tg, torch.full_like(ids[:, :1], -100)], 1)


def mask_from_lengths(prompt_lens, total_lens, T):
    """目标位置 j = t + 1 满足 prompt_len ≤ j < total_len 时为 1。"""
    j = torch.arange(1, T)[None]
    p = torch.as_tensor(prompt_lens)[:, None]
    n = torch.as_tensor(total_lens)[:, None]
    return ((j >= p) & (j < n)).float()


def seq_logprob(lp, mask, reduce="sum"):
    s = (lp * mask).sum(-1)
    return s if reduce == "sum" else s / mask.sum(-1).clamp(min=1)


def token_entropy(logits):
    """(B, T, V) → (B, T−1)：−Σ p log p，只取前 T−1 个位置。"""
    lp = log_softmax(logits[:, :-1].float(), -1)
    return -(lp.exp() * lp).sum(-1)
