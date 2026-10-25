"""第 17 章：从语言模型生成文字。

    process_logits   温度、重复惩罚、top-k、top-p：把最后一个位置的 logits 变成要采样的分布
    generate         逐个生成；use_cache=True 时先"预填充"整个提示，之后每步只算 1 个新 token（KV cache）
    beam_search      束搜索：保留累计 log 概率最高的 beam 条候选
第 22 章起 VLM 的生成也用它们：只是第一步的输入里多了图像特征（inputs_embeds）。
"""
import torch
import torch.nn.functional as F


# >>> 练习 17.1：处理 logits
def process_logits(logits, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0, prev=None):
    """logits 形状 (B, V)，返回处理后的 logits（不许采样的位置为 −∞）。按以下顺序：
    1. 重复惩罚（repetition_penalty > 1 且 prev 给出时；prev 是 (B, t) 已生成的编号）：
       对每行出现过的 token，logit > 0 的除以 penalty，< 0 的乘以 penalty（Hugging Face 的约定：总是让它变小）；
    2. 除以 temperature（> 0）；
    3. top_k > 0：每行只保留最大的 k 个，其余设为 −∞；
    4. top_p < 1：每行按概率从大到小排序，保留累计概率刚好达到 top_p 的最少几个（至少保留 1 个），其余设为 −∞。"""
    logits = logits.clone()
    if repetition_penalty != 1.0 and prev is not None and prev.numel():
        sc = logits.gather(1, prev)
        sc = torch.where(sc > 0, sc / repetition_penalty, sc * repetition_penalty)
        logits.scatter_(1, prev, sc)
    logits = logits / temperature
    if top_k > 0:
        kth = logits.topk(min(top_k, logits.shape[-1]), -1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        srt, idx = logits.sort(-1, descending=True)
        cum = srt.softmax(-1).cumsum(-1)
        drop = cum - srt.softmax(-1) >= top_p                     # 在它之前的累计概率已经够了 → 去掉
        drop[:, 0] = False
        logits = logits.masked_fill(drop.scatter(1, idx, drop), float("-inf"))
    return logits
# <<<


# >>> 练习 17.2：带 KV cache 的生成
@torch.no_grad()
def generate(model, idx, max_new, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0,
             greedy=False, eos=None, use_cache=True, generator=None, inputs_embeds=None, banned=None):
    """idx (B, T0) 提示的编号。返回 (B, T0 + 生成的长度) 的编号（遇到 eos 的行之后补 eos）。
    use_cache=True：
        第一步（预填充）：caches = 每层一个空 dict；把整个提示送进 model(idx, caches=caches)，得到最后位置的 logits；
        之后每步：只把刚生成的 1 个 token 送进去，pos0 = 当前总长度 − 1。
    use_cache=False：每步把全部序列重新送进模型（慢，但结果应完全相同）。
    inputs_embeds 给出时（第 22 章），它是提示的嵌入（可以含视觉特征），idx 只用于拼接输出和重复惩罚。
    banned：不许生成的 token 编号列表（例如特殊符号）。"""
    model.eval()
    B = idx.shape[0]
    caches = [dict() for _ in model.blocks] if use_cache else None
    out = idx
    done = torch.zeros(B, dtype=torch.bool, device=idx.device)
    prompt_len = idx.shape[1] if inputs_embeds is None else inputs_embeds.shape[1]
    for step in range(max_new):
        if step == 0 or not use_cache:
            if inputs_embeds is not None:
                gen = out[:, idx.shape[1]:]
                emb = inputs_embeds if gen.shape[1] == 0 else torch.cat([inputs_embeds, model.tok(gen)], 1)
                logits = model(inputs_embeds=emb, caches=caches)[0][:, -1]
            else:
                logits = model(out, caches=caches)[0][:, -1]
        else:
            logits = model(out[:, -1:], caches=caches, pos0=prompt_len + step - 1)[0][:, -1]
        if banned:
            logits[:, banned] = float("-inf")
        if greedy:                                            # 贪心也可以加重复惩罚（OCR 常这样用）
            nxt = process_logits(logits.float(), repetition_penalty=repetition_penalty, prev=out).argmax(-1)
        else:
            lp = process_logits(logits.float(), temperature, top_k, top_p, repetition_penalty, out)
            nxt = torch.multinomial(lp.softmax(-1), 1, generator=generator)[:, 0]
        if eos is not None:
            nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
            done |= nxt == eos
        out = torch.cat([out, nxt[:, None]], 1)
        if eos is not None and done.all():
            break
    return out
# <<<


# >>> 练习 17.3：束搜索
@torch.no_grad()
def beam_search(model, idx, max_new, beam=4, eos=None, length_penalty=1.0):
    """只处理一条提示（idx 形状 (1, T0)）。每步把每条候选扩展成 V 个，按累计 log 概率保留最高的 beam 条；
    以 eos 结尾的候选移入"完成"列表，不再扩展。最后按 累计 log 概率 / 长度^length_penalty 选最好的一条。
    为了简单，不用 KV cache（每步把所有候选整体重新算一遍）。返回 (1, T) 的编号。"""
    model.eval()
    T0 = idx.shape[1]
    beams = [(0.0, idx[0].tolist())]
    finished = []
    for _ in range(max_new):
        x = torch.tensor([s for _, s in beams], device=idx.device)
        lp = F.log_softmax(model(x)[0][:, -1].float(), -1)                # (n_beams, V)
        cand = []
        for (score, seq), row in zip(beams, lp):
            top = row.topk(beam)
            cand += [(score + v.item(), seq + [k.item()]) for v, k in zip(top.values, top.indices)]
        cand.sort(key=lambda c: -c[0])
        beams = []
        for score, seq in cand:
            if eos is not None and seq[-1] == eos:
                finished.append((score, seq))
            else:
                beams.append((score, seq))
            if len(beams) == beam:
                break
        if not beams or len(finished) >= beam:
            break
    pool = finished + beams
    best = max(pool, key=lambda c: c[0] / max(1, len(c[1]) - T0) ** length_penalty)
    return torch.tensor([best[1]], device=idx.device)
# <<<
