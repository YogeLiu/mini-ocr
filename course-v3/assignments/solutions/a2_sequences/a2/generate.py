"""A2 参考实现：处理 logits（温度、top-k、top-p）与带 KV cache 的生成。"""
import torch

from .nn import softmax


def process_logits(logits, temperature=1.0, top_k=0, top_p=1.0):
    """logits (B, V) → 处理后的 logits，不许采样的位置为 −∞。顺序：除以温度 → top-k → top-p。
    top_k > 0：只保留最大的 k 个。top_p < 1：按概率从大到小排，保留累计概率第一次达到 top_p 为止的那些（至少保留 1 个）。"""
    x = logits / temperature
    if top_k > 0:
        kth = x.topk(min(top_k, x.shape[-1]), dim=-1).values[:, -1:]
        x = x.masked_fill(x < kth, float("-inf"))
    if top_p < 1.0:
        s, order = x.sort(dim=-1, descending=True)
        cum = softmax(s, dim=-1).cumsum(-1)
        drop = cum - softmax(s, dim=-1) >= top_p                   # 前面的累计已经够了，这一个就不要
        drop[:, 0] = False
        x = x.masked_fill(torch.zeros_like(drop).scatter(1, order, drop), float("-inf"))   # 排序后的掩码放回原位置
    return x


@torch.no_grad()
def generate(model, idx, max_new, temperature=1.0, top_k=0, top_p=1.0, generator=None, eos=None):
    """先把提示整个送进去（预填充），之后每步只送 1 个新 token，用 KV cache 保存历史的 k、v。"""
    caches = [{} for _ in model.blocks]
    logits, _ = model(idx, caches=caches)
    out = idx
    for _ in range(max_new):
        x = process_logits(logits[:, -1], temperature, top_k, top_p)
        nxt = torch.multinomial(softmax(x, -1), 1, generator=generator)
        out = torch.cat([out, nxt], 1)
        if eos is not None and (nxt == eos).all():
            break
        if out.shape[1] >= model.cfg.block_size:
            break
        logits, _ = model(nxt, caches=caches, pos0=out.shape[1] - 1)
    return out
