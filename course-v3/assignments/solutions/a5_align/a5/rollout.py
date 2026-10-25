"""A5 参考实现：从 VLM 采样（温度、KV cache、可复现的 generator），每题 G 个回答，整理成 (ids, mask)，算对数概率。

为什么采样之后还要再做一次前向算 old_logp：生成用 KV cache 逐个 token 算，训练用整条序列一次算，数值上有微小差别。
两者都用"整条序列的前向"算，第一步的重要性比值 ρ 才精确等于 1。
"""
import numpy as np
import torch

from .logprobs import token_logprobs
from .nn import softmax


@torch.no_grad()
def sample(vlm, images, prompt, max_new, eos, temperature=1.0, generator=None, banned=()):
    """images (B, C, H, W) 或 (B, k, C, H, W) 的 float；prompt (B, T0) 长度相同。temperature = 0 时贪心。
    每步：banned 里的 token 的 logit 设为 −∞，p = softmax(logits / temperature)，torch.multinomial(p, 1, generator=generator)。
    所有行都写出 eos 时提前停止。
    返回 (B, n) 的编号（某一行写出 eos 之后的位置内容无意义）。"""
    was = vlm.training
    vlm.eval()
    caches = [{} for _ in vlm.lm.blocks]
    logits, _ = vlm.lm(inputs_embeds=vlm.embed(images, prompt), caches=caches)
    out, done, pos = [], torch.zeros(prompt.shape[0], dtype=torch.bool, device=prompt.device), prompt.shape[1]
    for _ in range(max_new):
        last = logits[:, -1].float().clone()
        last[:, list(banned)] = float("-inf")
        if temperature == 0:
            nxt = last.argmax(-1, keepdim=True)
        else:
            nxt = torch.multinomial(softmax(last / temperature, -1), 1, generator=generator)
        out.append(nxt)
        done |= nxt[:, 0] == eos
        if done.all() or pos >= vlm.lm.cfg.block_size:
            break
        logits, _ = vlm.lm(nxt, caches=caches, pos0=pos)
        pos += 1
    vlm.train(was)
    return torch.cat(out, 1)


def to_float(x):
    return x.float() / 127.5 - 1.0


@torch.no_grad()
def rollout(vlm, items, G, max_new, eos, temperature=1.0, generator=None, device="cpu", banned=()):
    """items：[(图像 uint8 (k, 3, H, W), 提示的编号列表)]。每题采 G 个，按（图的形状, 提示长度）分组生成。
    返回 (responses, finished)：各 N·G 个，顺序 [第 0 题的 G 个, 第 1 题的 G 个, …]；回答不含 eos，finished 表示写出了 eos。"""
    out = [None] * (len(items) * G)
    groups = {}
    for i, (img, p) in enumerate(items):
        groups.setdefault((img.shape, len(p)), []).append(i)
    for (shape, _), idxs in groups.items():
        rows = [i for i in idxs for _ in range(G)]
        X = to_float(torch.from_numpy(np.stack([items[i][0] for i in rows])).to(device))
        X = X[:, 0] if shape[0] == 1 else X
        P = torch.tensor([items[i][1] for i in rows], device=device)
        gen = sample(vlm, X, P, max_new, eos, temperature, generator, banned).tolist()
        seen = {}
        for i, g in zip(rows, gen):
            out[i * G + seen.get(i, 0)] = g
            seen[i] = seen.get(i, 0) + 1
    responses, finished = [], []
    for g in out:
        responses.append(g[:g.index(eos)] if eos in g else g)
        finished.append(eos in g)
    return responses, finished


def make_batch(prompts, responses, finished, eos, pad):
    """序列 = 提示 + 回答 + [eos]（finished 时）。返回 ids (B, T)、mask (B, T−1)：ids[b, t+1] 是回答（含 eos）时为 1。"""
    seqs = [p + r + ([eos] if f else []) for p, r, f in zip(prompts, responses, finished)]
    T = max(map(len, seqs))
    ids = torch.full((len(seqs), T), pad, dtype=torch.long)
    mask = torch.zeros(len(seqs), T - 1)
    for b, (p, s) in enumerate(zip(prompts, seqs)):
        ids[b, :len(s)] = torch.tensor(s)
        mask[b, len(p) - 1:len(s) - 1] = 1.0
    return ids, mask


def policy_logprobs(vlm, images, ids):
    """images：每条序列的图像（float 张量的列表，按序列顺序），ids (B, T) → (B, T−1)。需要梯度时不要包 no_grad。"""
    return token_logprobs(vlm.lm(inputs_embeds=vlm.embed_list(images, ids))[0], ids)
