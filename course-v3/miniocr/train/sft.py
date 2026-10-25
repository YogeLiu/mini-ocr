"""第 26 章：VLM 的监督微调（SFT）。损失算在哪些位置、怎样平均、怎样把短样本打包进一条序列。

    ROLE_*             每个 token 的"身份"：格式（<bos>、角色前缀、换行）、视觉占位符、用户的话、助手的回答
    chat_roles         套聊天模板（与第 23 章的 apply_chat_template 相同），同时返回每个 token 的身份
    sft_targets        由身份决定每个位置的标签和权重：回答 1，用户的话 prompt_weight，其余 0
    sft_loss           带权重的交叉熵；按 token 平均（"token"）或先样本内平均、再样本间平均（"sample"）
    pad_weighted       把若干条 (ids, targets, weights) 右侧补齐成批
    pack               把若干条短样本首尾相接装进长度 ≤ max_len 的"箱子"，记录每个 token 属于第几条（seg）和条内位置（pos）
    packed_mask        打包后的注意力 mask：只能看同一条样本里、不晚于自己的位置
    packed_lm_forward  用打包的 mask 和条内位置跑第 16 章的 GPT（不改 GPT 的代码）
    embed_packed       打包序列的嵌入：按出现顺序把每张图的视觉 token 填进占位符
    steps_to           学习曲线第一次达到阈值的步数（没达到返回 None）

为什么要打包：一个批里有的样本 70 个 token、有的 200 个，右侧补齐后大量位置是 <pad>，算了也白算。
打包后几乎没有 <pad>，同样的计算量能训练更多样本；代价是要用 mask 隔开不同的样本。
"""
import torch
import torch.nn.functional as F

from ..data.instruct import IMAGE

ROLE_FORMAT, ROLE_IMAGE, ROLE_USER, ROLE_ANSWER = 0, 1, 2, 3


# >>> 练习 26.1：每个 token 的身份
def chat_roles(vocab, messages, n_img):
    """与 apply_chat_template 用同样的拼法，返回 (ids, roles)，两个等长的列表。roles[t] 是 ids[t] 的身份：
        <bos>、"User: "、用户消息末尾的 "\\n"、"Assistant: "        → ROLE_FORMAT
        <img> 占位符                                                → ROLE_IMAGE
        用户消息的文字（不含角色前缀和末尾的换行）                  → ROLE_USER
        助手消息的文字和它后面的 <eos>                              → ROLE_ANSWER"""
    img = [vocab.stoi["<img>"]] * n_img
    ids, roles = [vocab.stoi["<bos>"]], [ROLE_FORMAT]

    def add(seq, role):
        ids.extend(seq)
        roles.extend([role] * len(seq))

    for m in messages:
        if m["role"] == "user":
            add(vocab.encode("User: "), ROLE_FORMAT)
            for i, p in enumerate(m["content"].split(IMAGE)):
                if i:
                    add(img, ROLE_IMAGE)
                add(vocab.encode(p), ROLE_USER)
            add(vocab.encode("\n"), ROLE_FORMAT)
        else:
            add(vocab.encode("Assistant: "), ROLE_FORMAT)
            add(vocab.encode(m["content"]) + [vocab.stoi["<eos>"]], ROLE_ANSWER)
    return ids, roles
# <<<


def sft_targets(ids, roles, prompt_weight=0.0, format_weight=None, eos_id=None, train_eos=True):
    """返回 (targets, weights)，长度都等于 len(ids)。位置 t 预测 ids[t + 1]，权重由 ids[t + 1] 的身份决定：
    回答 1.0；用户的话 prompt_weight；格式 format_weight（默认与 prompt_weight 相同）；视觉占位符永远 0（它是我们
    自己放进去的，不需要模型"预测"）。权重为 0 的位置标签设成 −100。train_eos=False 时回答末尾的 <eos> 不计算损失
    （第 26.5 节的故障注入）。"""
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


# >>> 练习 26.2：带权重的损失与两种平均
def sft_loss(logits, targets, weights=None, mode="token"):
    """logits (B, T, V)，targets (B, T)（−100 = 不计算），weights (B, T) 浮点（None 表示全是 1）。
    每个位置的交叉熵 ce[b, t]，权重 w[b, t]（targets 为 −100 的位置权重当作 0）。
        mode="token"：  Σ w·ce / Σ w                         —— 所有计算损失的 token 一样重要，长回答的样本分量重
        mode="sample"： 先每条样本 Σ_t w·ce / Σ_t w，再对"有计算损失的位置"的样本取平均 —— 每条样本一样重要
    权重之和为 0 时返回 0（并保持可求导）。"""
    B, T, V = logits.shape
    valid = targets != -100
    ce = F.cross_entropy(logits.reshape(-1, V).float(), targets.clamp(min=0).reshape(-1), reduction="none").view(B, T)
    w = valid.float() if weights is None else weights.float() * valid
    if mode == "token":
        return (w * ce).sum() / w.sum().clamp(min=1e-8)
    per = (w * ce).sum(1) / w.sum(1).clamp(min=1e-8)
    has = w.sum(1) > 0
    return per[has].mean() if has.any() else (w * ce).sum()
# <<<


def pad_weighted(examples, pad_id):
    """[(ids, targets, weights), …] → 三个 (B, T) 张量，右侧补齐（ids 补 pad_id，targets 补 −100，weights 补 0）。"""
    T = max(len(e[0]) for e in examples)
    ids = torch.full((len(examples), T), pad_id, dtype=torch.long)
    tg = torch.full((len(examples), T), -100, dtype=torch.long)
    w = torch.zeros(len(examples), T)
    for b, (i, t, ww) in enumerate(examples):
        ids[b, :len(i)] = torch.tensor(i)
        tg[b, :len(t)] = torch.tensor(t)
        w[b, :len(ww)] = torch.tensor(ww)
    return ids, tg, w


# >>> 练习 26.3：打包
def pack(examples, max_len, pad_id):
    """examples：[(ids, targets, weights), …]，每条长度 ≤ max_len。按顺序装箱：当前箱子装得下就放进去，
    装不下就开一个新箱子（"顺序装箱"，不回头找空位）。返回 dict：
        ids, targets (B, max_len) long；weights (B, max_len) float；
        seg (B, max_len) long：第几条样本（每个箱子里从 1 开始编号，<pad> 为 0）；
        pos (B, max_len) long：样本内的位置 0, 1, 2, …（<pad> 为 0）；
        order：[[样本编号, …], …] 每个箱子里依次装了哪几条（embed_packed 用它决定图像的顺序）。"""
    bins, cur, used = [], [], 0
    for k, e in enumerate(examples):
        n = len(e[0])
        assert n <= max_len, f"第 {k} 条样本长 {n}，超过 max_len {max_len}"
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
# <<<


# >>> 练习 26.4：打包后的注意力 mask
def packed_mask(seg):
    """seg (B, T) → (B, 1, T, T) 的 bool：mask[b, 0, i, j] = True 当且仅当 j ≤ i 且 seg[b, i] = seg[b, j] ≠ 0。
    <pad> 位置（seg = 0）只允许看自己，避免整行都是 False（softmax 会得到 NaN）；这些位置不计算损失，看什么都无所谓。"""
    T = seg.shape[1]
    causal = torch.ones(T, T, dtype=torch.bool, device=seg.device).tril()
    same = (seg[:, :, None] == seg[:, None, :]) & (seg[:, :, None] != 0)
    eye = torch.eye(T, dtype=torch.bool, device=seg.device)
    return ((same & causal) | eye)[:, None]
# <<<


def packed_lm_forward(lm, x, seg, pos):
    """x (B, T, d) 是输入嵌入；seg、pos 来自 pack。逐层重复 GPT 的计算，只有两处不同：
    RoPE 用条内位置 pos 查表（每条样本都从 0 开始编号），注意力用 packed_mask。返回 logits (B, T, V)。
    与把每条样本单独送进 lm 的结果相同（测试 test_packed_equals_separate）。"""
    mask = packed_mask(seg)
    cos, sin = lm.cos[pos][:, None], lm.sin[pos][:, None]          # (B, 1, T, dh)：每个位置自己的旋转角
    from ..models.gpt import apply_rope
    for blk in lm.blocks:
        a = blk.attn
        h = blk.n1(x)
        B, T, d = h.shape
        q, k, v = a.qkv(h).view(B, T, 3, a.h, a.dh).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=a.dropout if lm.training else 0.0)
        x = x + blk.drop(a.o(y.transpose(1, 2).reshape(B, T, d)))
        x = x + blk.drop(blk.mlp(blk.n2(x)))
    return lm.head(lm.norm(x))


def embed_packed(vlm, images, ids):
    """images：(N, C, H, W)，N = 这一批里所有的图，按它们的占位符在批里出现的顺序排：第 0 行从左到右，再第 1 行……
    打包时就是"第 0 个箱子的第 1 条样本的图、第 2 条样本的图……"；一条样本有两张图（compare）时两张依次排。
    布尔下标赋值按行优先的顺序填占位符，正好和这个顺序一致。不打包的批（每行一条样本）也能用。"""
    emb = vlm.lm.tok(ids)
    v = vlm.visual_tokens(images).to(emb.dtype)                     # (N, N_v, d)
    mask = ids == vlm.img_id
    assert mask.sum().item() == v.shape[0] * v.shape[1], f"占位符 {mask.sum().item()} 个，视觉 token {v.shape[0] * v.shape[1]} 个"
    emb = emb.clone()
    emb[mask] = v.reshape(-1, v.shape[-1])
    return emb


def steps_to(curve, threshold):
    """curve：[(步数, 指标), …]，按步数排序。返回第一次 指标 ≥ threshold 的步数；从没达到返回 None。"""
    for s, v in curve:
        if v >= threshold:
            return s
    return None
