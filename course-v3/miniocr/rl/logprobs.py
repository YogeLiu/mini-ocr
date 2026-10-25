"""第 27 章：回答的对数概率。SFT 损失、困惑度、熵、DPO、GRPO 都是这一个量的不同用法。

    token_logprobs   每个位置"下一个 token"的对数概率：(B, T, V) 的 logits 和 (B, T) 的 ids → (B, T−1)
    response_mask    由第 26 章的身份（roles）得到 (T−1,) 的 0/1：目标 token 属于回答的位置为 1
    mask_from_lengths  由提示长度和总长度得到 (B, T−1) 的回答 mask（不需要 roles 时用）
    labels_to_mask   约定一（标签里 −100 表示不算）→ 约定二（显式的 0/1 mask）
    mask_to_labels   约定二 → 约定一
    seq_logprob      把每个 token 的对数概率按 mask 求和（"sum"）或求平均（"mean"）
    token_entropy    每个位置的预测分布的熵：(B, T, V) → (B, T−1)

错位是这里最容易出错的地方：logits[:, t] 预测的是 ids[:, t + 1]。所以返回的第 t 列对应第 t + 1 个 token，
第 0 个 token（<bos>）没有对数概率，最后一个位置的 logits 预测的是序列之外的东西，被丢掉。
"""
import torch
import torch.nn.functional as F

from ..train.sft import ROLE_ANSWER


# >>> 练习 27.1：每个 token 的对数概率
def token_logprobs(logits, ids):
    """logits (B, T, V)，ids (B, T) → (B, T−1)：out[b, t] = log softmax(logits[b, t])[ids[b, t + 1]]。
    用 float32 算 log_softmax（模型用 bf16 时，概率的小差别会被精度吞掉）；用 gather 取出目标 token 的那一列。"""
    lp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    return lp.gather(-1, ids[:, 1:, None]).squeeze(-1)
# <<<


# >>> 练习 27.2：回答的 mask
def response_mask(roles):
    """roles：第 26 章 chat_roles 返回的身份列表（长 T）。返回 (T−1,) 的 float 张量：
    第 t 个元素是 1，当且仅当第 t + 1 个 token 是回答（ROLE_ANSWER，包括回答末尾的 <eos>）。"""
    return torch.tensor([1.0 if r == ROLE_ANSWER else 0.0 for r in roles[1:]])


def mask_from_lengths(prompt_lens, total_lens, T):
    """每条序列的前 prompt_lens[b] 个 token 是提示，接着到 total_lens[b] 是回答，之后是补齐。
    返回 (B, T−1)：目标位置 j = t + 1 满足 prompt_len ≤ j < total_len 时为 1。"""
    j = torch.arange(1, T)[None]
    p = torch.as_tensor(prompt_lens)[:, None]
    n = torch.as_tensor(total_lens)[:, None]
    return ((j >= p) & (j < n)).float()
# <<<


# >>> 练习 27.3：两种约定互相转换
def labels_to_mask(targets):
    """约定一：targets (B, T)，targets[b, t] 是位置 t 要预测的 token（= ids[b, t + 1]），−100 表示不算（第 22、26 章）。
    返回约定二的 (B, T−1) mask：第 t 列 = 1 当且仅当 targets[b, t] ≠ −100。（targets 最后一列总是 −100，丢掉。）"""
    return (targets[:, :-1] != -100).float()


def mask_to_labels(ids, mask):
    """约定二 → 约定一：ids (B, T)，mask (B, T−1) → targets (B, T)：targets[:, t] = ids[:, t + 1]（mask 为 1 时）或 −100，
    最后一列补 −100。"""
    tg = torch.where(mask.bool(), ids[:, 1:], torch.full_like(ids[:, 1:], -100))
    return torch.cat([tg, torch.full_like(ids[:, :1], -100)], 1)
# <<<


def seq_logprob(lp, mask, reduce="sum"):
    """lp、mask 都是 (B, T−1)。返回 (B,)：Σ lp·mask（"sum"，回答的概率的对数），或再除以 Σ mask（"mean"，每个 token 平均）。"""
    s = (lp * mask).sum(-1)
    return s if reduce == "sum" else s / mask.sum(-1).clamp(min=1)


# >>> 练习 27.4：熵
def token_entropy(logits):
    """(B, T, V) → (B, T−1)：H = −Σ_v p(v) log p(v)，p = softmax(logits[:, t])，只取前 T−1 个位置（与 token_logprobs 对齐）。
    范围 [0, ln V]：0 表示完全确定，ln V 表示均匀猜测。用 log_softmax 算 log p，避免 log(0)。"""
    lp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    return -(lp.exp() * lp).sum(-1)
# <<<
