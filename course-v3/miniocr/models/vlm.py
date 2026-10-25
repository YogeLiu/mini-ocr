"""第 22 章：搭一个 VLM = 视觉编码器（第 19/20 章的 ViT）+ 连接器 + 语言模型（第 16 章的 GPT）。

    merge_2x2        把相邻的 2×2 个 patch 拼成一个向量（Qwen-VL 的 merger、SmolVLM 的 pixel shuffle 的核心）
    MLPProjector     LLaVA 式：每个 patch 过一个两层 MLP，token 数不变
    MergeProjector   合并式：先 merge_2x2（token 数变成 1/4），再过 MLP
    QFormer          查询式（BLIP-2）：n_query 个可学习的查询向量，用交叉注意力从所有 patch 里"提取"信息，token 数固定
    build_example    一条训练样本的 token 序列和标签：<bos> <img>×N 问题 答案 <eos>，只有答案部分计算损失
    VLM              把视觉 token 填进 <img> 占位符的位置，整条序列交给 GPT；generate 逐字写出答案
    set_trainable    冻结 / 解冻视觉塔、连接器、语言模型

序列格式（字符级，第 16 章的 100 个字符的字符表）：
    <bos><img><img>…<img>Q: What color is the leftmost object?\nA: red<eos>
    └─────── 不计算损失 ─────────────────────────────────────────┘└ 计算损失 ┘
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nn.attention import MultiHeadAttention


# >>> 练习 22.1：2×2 合并
def merge_2x2(x, hw):
    """x (B, N, d) 是按行排列的 patch 序列，网格 hw = (h, w)，h、w 都是偶数。
    返回 (B, N/4, 4d)：新网格 (h/2, w/2) 也按行排列；每个新 token 是原网格里一个 2×2 小方块的 4 个 patch 拼接，
    顺序为 左上、右上、左下、右下。"""
    B, N, d = x.shape
    h, w = hw
    x = x.reshape(B, h // 2, 2, w // 2, 2, d)          # (B, 行组, 组内行, 列组, 组内列, d)
    x = x.permute(0, 1, 3, 2, 4, 5)                     # (B, 行组, 列组, 组内行, 组内列, d)
    return x.reshape(B, (h // 2) * (w // 2), 4 * d)
# <<<


class MLPProjector(nn.Module):
    """LLaVA-1.5：Linear(d_v, d) → GELU → Linear(d, d)，逐个 patch 作用。"""

    def __init__(self, d_v, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_v, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, x, hw):
        return self.net(x)


class MergeProjector(nn.Module):
    """Qwen2-VL 的 PatchMerger：merge_2x2 → LayerNorm(4d_v) → Linear(4d_v, 4d_v) → GELU → Linear(4d_v, d)。"""

    def __init__(self, d_v, d):
        super().__init__()
        self.ln = nn.LayerNorm(4 * d_v)
        self.net = nn.Sequential(nn.Linear(4 * d_v, 4 * d_v), nn.GELU(), nn.Linear(4 * d_v, d))

    def forward(self, x, hw):
        return self.net(self.ln(merge_2x2(x, hw)))


class QFormerLayer(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_attn = MultiHeadAttention(d, n_head)
        self.cross_attn = MultiHeadAttention(d, n_head)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, q, patches):
        q = q + self.self_attn(self.ln1(q))                     # 查询之间互相交流
        q = q + self.cross_attn(self.ln2(q), kv=patches)         # 每个查询去所有 patch 里取信息
        return q + self.mlp(self.ln3(q))


class QFormer(nn.Module):
    """BLIP-2 的 Q-Former 的简化版：n_query 个可学习的查询，n_layer 层（自注意力 + 交叉注意力 + MLP），最后投影到 d。
    无论图有多少个 patch，输出都是 n_query 个 token。"""

    def __init__(self, d_v, d, n_query=16, n_layer=2, n_head=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, n_query, d_v) * 0.02)
        self.layers = nn.ModuleList([QFormerLayer(d_v, n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(d_v)
        self.out = nn.Linear(d_v, d)

    def forward(self, x, hw):
        q = self.query.expand(x.shape[0], -1, -1)
        for layer in self.layers:
            q = layer(q, x)
        return self.out(self.ln(q))


def n_visual_tokens(kind, n_patch, n_query=16):
    return {"mlp": n_patch, "merge": n_patch // 4, "qformer": n_query}[kind]


# >>> 练习 22.2：一条训练样本
def build_example(vocab, n_img, question, answer):
    """返回 (ids, targets)，都是长度相同的整数列表：
        ids     = [<bos>] + [<img>] × n_img + 编码("Q: " + question + "\\nA: ") + 编码(answer) + [<eos>]
        targets = ids 左移一位（targets[t] = ids[t + 1]，最后一位补 −100），
                  并且只保留"预测答案和 <eos>"的位置，其余都设成 −100（不计算损失）。
    也就是说：在 "A: " 的最后一个字符（空格）那个位置，模型要预测答案的第一个字符。"""
    head = [vocab.stoi["<bos>"]] + [vocab.stoi["<img>"]] * n_img + vocab.encode("Q: " + question + "\nA: ")
    tail = vocab.encode(answer) + [vocab.stoi["<eos>"]]
    ids = head + tail
    targets = [-100] * (len(head) - 1) + tail + [-100]
    return ids, targets
# <<<


def pad_batch(examples, pad_id):
    """[(ids, targets), …] → 两个 (B, T) 的 LongTensor，右侧补齐（ids 补 pad_id，targets 补 −100）。"""
    T = max(len(i) for i, _ in examples)
    ids = torch.full((len(examples), T), pad_id, dtype=torch.long)
    tg = torch.full((len(examples), T), -100, dtype=torch.long)
    for b, (i, t) in enumerate(examples):
        ids[b, :len(i)] = torch.tensor(i)
        tg[b, :len(t)] = torch.tensor(t)
    return ids, tg


class VLM(nn.Module):
    """vision：第 19 章的 ViT（用 patch_tokens 取每个 patch 的输出）；connector：上面三种之一；lm：第 16 章的 GPT。"""

    def __init__(self, vision, connector, lm, img_id):
        super().__init__()
        self.vision, self.connector, self.lm = vision, connector, lm
        self.img_id = img_id

    def visual_tokens(self, images):
        x, hw = self.vision.patch_tokens(images)
        return self.connector(x, hw)                             # (B, N_v, d)

    # >>> 练习 22.3：把视觉 token 填进占位符
    def embed(self, images, ids):
        """返回 (B, T, d) 的输入嵌入：先查文字嵌入 self.lm.tok(ids)，再把 ids == img_id 的位置
        按顺序换成 visual_tokens(images)（每张图 N_v 个，按行排列）。每条序列里 <img> 的个数必须正好等于 N_v。
        提示：布尔下标赋值 emb[mask] = v.reshape(-1, d) 按"先第 0 条序列、再第 1 条……"的顺序填。
        images 也可以是 (B, k, C, H, W)：每条序列 k 张图（第 23 章），先把 B·k 张图一起编码，再变形成 (B, k·N_v, d)。"""
        emb = self.lm.tok(ids)
        if images.dim() == 5:
            B, k = images.shape[:2]
            v = self.visual_tokens(images.flatten(0, 1))
            v = v.reshape(B, k * v.shape[1], v.shape[2]).to(emb.dtype)
        else:
            v = self.visual_tokens(images).to(emb.dtype)
        mask = ids == self.img_id
        assert mask.sum(1).eq(v.shape[1]).all(), f"每条序列应有 {v.shape[1]} 个 <img>，实际 {mask.sum(1).tolist()}"
        emb = emb.clone()
        emb[mask] = v.reshape(-1, v.shape[-1])
        return emb
    # <<<

    def forward(self, images, ids, targets=None):
        return self.lm(inputs_embeds=self.embed(images, ids), targets=targets)

    @torch.no_grad()
    def generate(self, images, prompt_ids, max_new=80, eos=None, **kw):
        """prompt_ids (B, T0)：<bos> <img>×N 问题 "\\nA: "（同一批里长度相同）。返回新生成的部分 (B, ≤ max_new)。"""
        from ..generate import generate
        self.eval()
        out = generate(self.lm, prompt_ids, max_new, greedy=True, eos=eos,
                       inputs_embeds=self.embed(images, prompt_ids), **kw)
        return out[:, prompt_ids.shape[1]:]


def set_trainable(vlm, vision=True, connector=True, lm=True):
    """requires_grad 开关。冻结的部分仍然参与前向和反向（梯度要穿过语言模型传到连接器），只是自己的参数不更新。"""
    for mod, on in ((vlm.vision, vision), (vlm.connector, connector), (vlm.lm, lm)):
        for p in mod.parameters():
            p.requires_grad_(on)
    return sum(p.numel() for p in vlm.parameters() if p.requires_grad)
