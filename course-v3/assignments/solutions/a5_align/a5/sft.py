"""A5 参考实现：SFT。每个 token 的身份、由身份决定的标签与权重、两种平均、补齐、打包与打包的 mask。"""
import torch

from .nn import log_softmax
from .vlm import IMAGE

ROLE_FORMAT, ROLE_IMAGE, ROLE_USER, ROLE_ANSWER = 0, 1, 2, 3


def chat_roles(messages, n_img, encode, bos, eos, img):
    """拼法与 A3 的 chat_template 相同，返回 (ids, roles)。n_img 可以是整数（每张图一样多）或列表（第 k 张图的个数）。
        <bos>、"User: "、用户消息末尾的 "\\n"、"Assistant: " → ROLE_FORMAT
        图像占位符 → ROLE_IMAGE；用户消息的文字 → ROLE_USER；助手的文字和它后面的 eos → ROLE_ANSWER"""
    ids, roles, k = [bos], [ROLE_FORMAT], 0

    def add(seq, role):
        ids.extend(seq)
        roles.extend([role] * len(seq))

    for m in messages:
        if m["role"] == "user":
            add(encode("User: "), ROLE_FORMAT)
            for i, part in enumerate(m["content"].split(IMAGE)):
                if i:
                    add([img] * (n_img if isinstance(n_img, int) else n_img[k]), ROLE_IMAGE)
                    k += 1
                add(encode(part), ROLE_USER)
            add(encode("\n"), ROLE_FORMAT)
        else:
            add(encode("Assistant: "), ROLE_FORMAT)
            add(encode(m["content"]) + [eos], ROLE_ANSWER)
    return ids, roles


def sft_targets(ids, roles, prompt_weight=0.0, format_weight=None, eos_id=None, train_eos=True):
    """位置 t 预测 ids[t + 1]，权重由 ids[t + 1] 的身份决定：回答 1、用户 prompt_weight、格式 format_weight
    （默认等于 prompt_weight）、图像 0。train_eos=False 时回答末尾的 eos 权重为 0。权重 0 的位置标签 −100；最后一位补 −100、0。"""
    fw = prompt_weight if format_weight is None else format_weight
    w_of = {ROLE_ANSWER: 1.0, ROLE_USER: prompt_weight, ROLE_FORMAT: fw, ROLE_IMAGE: 0.0}
    targets, weights = [], []
    for t in range(len(ids) - 1):
        w = w_of[roles[t + 1]]
        if not train_eos and roles[t + 1] == ROLE_ANSWER and ids[t + 1] == eos_id:
            w = 0.0
        targets.append(ids[t + 1] if w > 0 else -100)
        weights.append(w)
    return targets + [-100], weights + [0.0]


def sft_loss(logits, targets, weights=None, mode="token"):
    """token：Σ w·ce / Σ w；sample：每条样本 Σ_t w·ce / Σ_t w，再对有权重的样本平均。权重全为 0 时返回 0（可求导）。"""
    B, T, V = logits.shape
    valid = targets != -100
    lp = log_softmax(logits.float(), -1)
    ce = -lp.gather(-1, targets.clamp(min=0)[..., None]).squeeze(-1)
    w = valid.float() if weights is None else weights.float() * valid
    if mode == "token":
        return (w * ce).sum() / w.sum().clamp(min=1e-8)
    per = (w * ce).sum(1) / w.sum(1).clamp(min=1e-8)
    has = w.sum(1) > 0
    return per[has].mean() if has.any() else (w * ce).sum()


def pad_weighted(examples, pad_id):
    """[(ids, targets, weights)] → 三个 (B, T)，右侧补 pad_id / −100 / 0。"""
    T = max(len(e[0]) for e in examples)
    ids = torch.full((len(examples), T), pad_id, dtype=torch.long)
    tg = torch.full((len(examples), T), -100, dtype=torch.long)
    w = torch.zeros(len(examples), T)
    for b, (i, t, ww) in enumerate(examples):
        ids[b, :len(i)] = torch.tensor(i)
        tg[b, :len(t)] = torch.tensor(t)
        w[b, :len(ww)] = torch.tensor(ww)
    return ids, tg, w


def pack(examples, max_len, pad_id):
    """顺序装箱：当前箱子装得下就放进去，否则开新箱子。返回 dict：ids、targets、weights、seg（箱内第几条，从 1 开始，
    pad 为 0）、pos（条内位置，pad 为 0），都是 (箱子数, max_len)；order：每个箱子装了哪几条样本的编号。"""
    bins, cur, used = [], [], 0
    for k, e in enumerate(examples):
        n = len(e[0])
        if n > max_len:
            raise ValueError(f"第 {k} 条样本长 {n}，超过 max_len {max_len}")
        if used + n > max_len:
            bins.append(cur)
            cur, used = [], 0
        cur.append(k)
        used += n
    if cur:
        bins.append(cur)
    B = len(bins)
    out = dict(ids=torch.full((B, max_len), pad_id, dtype=torch.long), targets=torch.full((B, max_len), -100, dtype=torch.long),
               weights=torch.zeros(B, max_len), seg=torch.zeros(B, max_len, dtype=torch.long),
               pos=torch.zeros(B, max_len, dtype=torch.long), order=bins)
    for b, ks in enumerate(bins):
        s = 0
        for j, k in enumerate(ks):
            i, t, w = examples[k]
            n = len(i)
            out["ids"][b, s:s + n] = torch.tensor(i)
            out["targets"][b, s:s + n] = torch.tensor(t)
            out["weights"][b, s:s + n] = torch.tensor(w)
            out["seg"][b, s:s + n] = j + 1
            out["pos"][b, s:s + n] = torch.arange(n)
            s += n
    return out


def packed_mask(seg):
    """(B, T) → (B, 1, T, T)：j ≤ i 且 seg[i] = seg[j] ≠ 0；pad 位置只看自己（避免整行 False）。"""
    T = seg.shape[1]
    causal = torch.ones(T, T, dtype=torch.bool, device=seg.device).tril()
    same = (seg[:, :, None] == seg[:, None, :]) & (seg[:, :, None] != 0)
    eye = torch.eye(T, dtype=torch.bool, device=seg.device)
    return ((same & causal) | eye)[:, None]
