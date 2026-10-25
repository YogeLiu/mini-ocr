"""第 29 章起：从我们的 VLM 采样回答，并整理成可以算对数概率的批。第 30–32 章都用它。

    rollout         每个提示采 G 个回答（温度采样），返回编号列表和是否以 eos 结束
    make_batch      提示 + 回答 → 右侧补齐的 ids (B, T) 和回答的 mask (B, T−1)（与 token_logprobs 对齐）
    policy_logprobs 一次完整的前向，得到每个 token 的对数概率 (B, T−1)

为什么采样之后还要再做一次前向来算 old_logp，而不直接用生成时的 logits？
    生成用 KV cache 逐个 token 地算，训练用整条序列一次算完。两者数学上相同，数值上有很小的差别（第 17 章）。
    如果 old_logp 来自生成、logp 来自训练的前向，比值 ρ 在第一步就不等于 1。用同一种前向算两者，第一步的 ρ 精确等于 1。
"""
import torch

from ..generate import generate
from .logprobs import token_logprobs


@torch.no_grad()
def rollout(vlm, images, prompts, G, max_new, eos, temperature=1.0, generator=None):
    """images：(N, 3, H, W) 或 (N, k, 3, H, W) 的 float 张量（已经归一化，和训练时一样）；prompts：N 个编号列表。
    返回 (responses, finished)：各 N·G 个，顺序是 [第 0 题的 G 个, 第 1 题的 G 个, …]。
    responses[i] 是回答的编号列表（不含 eos）；finished[i] 表示它是否在 max_new 之内写出了 eos。
    提示长度不同的分组生成（我们的 GPT 不支持左侧补齐）。"""
    vlm.eval()
    N = len(prompts)
    out = [None] * (N * G)
    groups = {}
    for i, p in enumerate(prompts):
        groups.setdefault(len(p), []).append(i)
    for idxs in groups.values():
        rows = [i for i in idxs for _ in range(G)]
        ids = torch.tensor([prompts[i] for i in rows], device=images.device)
        emb = vlm.embed(images[rows], ids)
        gen = generate(vlm.lm, ids, max_new, temperature=temperature, eos=eos, inputs_embeds=emb,
                       generator=generator)[:, ids.shape[1]:].tolist()
        k = {}
        for i, g in zip(rows, gen):
            j = i * G + k.get(i, 0)
            k[i] = k.get(i, 0) + 1
            out[j] = g
    responses, finished = [], []
    for g in out:
        if eos in g:
            responses.append(g[:g.index(eos)])
            finished.append(True)
        else:
            responses.append(g)
            finished.append(False)
    return responses, finished


def make_batch(prompts, responses, finished, eos, pad):
    """prompts：每个回答对应的提示（长度 B 的列表，已经按 G 重复好）。序列 = 提示 + 回答 + [eos]（finished 时）。
    返回 ids (B, T) 和 mask (B, T−1)：mask[b, t] = 1 当且仅当 ids[b, t + 1] 是回答的 token（包括结尾的 eos）。"""
    seqs = [p + r + ([eos] if f else []) for p, r, f in zip(prompts, responses, finished)]
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad, dtype=torch.long)
    mask = torch.zeros(len(seqs), T - 1)
    for b, (p, s) in enumerate(zip(prompts, seqs)):
        ids[b, :len(s)] = torch.tensor(s)
        mask[b, len(p) - 1:len(s) - 1] = 1.0
    return ids, mask


def policy_logprobs(vlm, images, ids):
    """(B, T) → (B, T−1)：vlm 对每个"下一个 token"的对数概率。需要梯度时在外面不要包 no_grad。"""
    return token_logprobs(vlm(images, ids)[0], ids)
