"""第 20 章：对比学习——CLIP 与 SigLIP。让图像编码器和文字编码器把"配对的图和文字"映射到同一个空间里相近的位置。

    TextEncoder     字符级的小 Transformer（非因果，带填充 mask），输出所有真实字符的平均
    DualEncoder     图像编码器 + 文字编码器 + 各自的线性投影 + 可学习的温度（SigLIP 还有偏置）
    clip_loss       InfoNCE：B × B 的相似度矩阵上，按行、按列各做一次 softmax 交叉熵
    siglip_loss     每一对 (图 i, 文 j) 单独做一次二分类：i = j 是正例，其余是负例
    zero_shot       用"每个类别一句话"的文字向量当分类器
    recall_at_k     检索：正确答案排在前 k 名的比例

约定：嵌入向量都已经 L2 归一化，所以点积就是余弦相似度，取值在 [−1, 1]。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit import ViTBlock


class TextEncoder(nn.Module):
    """ids (B, T)，用 pad_id 补齐。token 嵌入 + 可学习的位置表 → n_layer 个非因果块（填充位置被 mask 掉）
    → LayerNorm → 对真实字符求平均 → (B, d)。"""

    def __init__(self, vocab_size, max_len=96, d=128, n_layer=3, n_head=4, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.tok = nn.Embedding(vocab_size, d)
        nn.init.normal_(self.tok.weight, std=0.02)
        self.pos = nn.Parameter(torch.randn(max_len, d) * 0.02)
        self.blocks = nn.ModuleList([ViTBlock(d, n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(d)

    def forward(self, ids):
        mask = ids != self.pad_id
        x = self.tok(ids) + self.pos[:ids.shape[1]]
        for b in self.blocks:
            x = b(x, mask=mask)
        x = self.ln(x)
        m = mask.unsqueeze(-1).float()
        return (x * m).sum(1) / m.sum(1).clamp(min=1)


# >>> 练习 20.1：CLIP 的损失（InfoNCE）
def clip_loss(zi, zt, logit_scale):
    """zi、zt 形状 (B, D)，已归一化；第 i 张图与第 i 句话配对。logit_scale 是标量张量 s = 1/τ。
    logits = s · zi @ ztᵀ，形状 (B, B)；第 i 行的正确答案是第 i 列。
    返回 (图→文的交叉熵 + 文→图的交叉熵) / 2：前者按行做 softmax，后者按列（即对 logitsᵀ 按行）。"""
    logits = logit_scale * zi @ zt.T
    labels = torch.arange(len(zi), device=zi.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
# <<<


# >>> 练习 20.2：SigLIP 的损失
def siglip_loss(zi, zt, logit_scale, logit_bias):
    """每一对 (i, j) 是一个独立的二分类：logit_ij = s · zi_i · zt_j + b；标签 y_ij = +1（i = j）或 −1（i ≠ j）。
    损失 = −Σ_ij log σ(y_ij · logit_ij) / B（对所有 B² 对求和，再除以 B，与 SigLIP 论文相同）。
    用 F.logsigmoid 计算 log σ，避免 σ 下溢成 0 后取对数得到 −inf。"""
    logits = logit_scale * zi @ zt.T + logit_bias
    y = 2 * torch.eye(len(zi), device=zi.device) - 1
    return -F.logsigmoid(y * logits).sum() / len(zi)
# <<<


class DualEncoder(nn.Module):
    """image_encoder(x) → (B, d_img)，text_encoder(ids) → (B, d_txt)，各自线性投影到 d_emb 并归一化。
    loss="clip"：可学习的 log(1/τ)，初始 τ = 0.07（CLIP 论文），s 上限 100；
    loss="siglip"：可学习的 log s 和偏置 b，初始 s = 10、b = −10（SigLIP 论文）。
    fixed_scale 不为 None 时温度固定不学（第 20 章的对照实验）。"""

    def __init__(self, image_encoder, text_encoder, d_img, d_txt, d_emb=128, loss="clip", fixed_scale=None):
        super().__init__()
        self.image_encoder, self.text_encoder = image_encoder, text_encoder
        self.img_proj = nn.Linear(d_img, d_emb, bias=False)
        self.txt_proj = nn.Linear(d_txt, d_emb, bias=False)
        self.kind = loss
        init = math.log(1 / 0.07) if loss == "clip" else math.log(10.0)
        if fixed_scale is not None:
            self.register_buffer("log_scale", torch.tensor(math.log(fixed_scale)))
        else:
            self.log_scale = nn.Parameter(torch.tensor(init))
        if loss == "siglip":
            self.logit_bias = nn.Parameter(torch.tensor(-10.0))

    def scale(self):
        return self.log_scale.exp().clamp(max=100.0)

    def encode_image(self, x):
        return F.normalize(self.img_proj(self.image_encoder(x)), dim=-1)

    def encode_text(self, ids):
        return F.normalize(self.txt_proj(self.text_encoder(ids)), dim=-1)

    def forward(self, x, ids):
        zi, zt = self.encode_image(x), self.encode_text(ids)
        if self.kind == "clip":
            return clip_loss(zi, zt, self.scale())
        return siglip_loss(zi, zt, self.scale(), self.logit_bias)


# >>> 练习 20.3：零样本分类
def zero_shot(zi, class_zt):
    """zi (N, D) 是图像向量，class_zt (K, D) 是 K 个类别描述的文字向量（都已归一化）。
    返回每张图最相似的类别编号 (N,)。"""
    return (zi @ class_zt.T).argmax(-1)
# <<<


# >>> 练习 20.4：检索的召回率
def recall_at_k(sim, k, same=None):
    """sim (N, M)：第 i 个查询与第 j 个候选的相似度。same (N, M) 布尔矩阵：same[i, j] = True 表示候选 j
    对查询 i 来说"算对"（默认 None 表示只有 j = i 算对，此时要求 N = M）。
    返回前 k 名里至少有一个算对的候选的查询所占的比例（float）。"""
    if same is None:
        same = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    top = sim.topk(k, dim=1).indices                       # (N, k)
    return same.gather(1, top).any(1).float().mean().item()
# <<<


def encode_texts(vocab, texts, max_len, pad="<pad>"):
    """字符串列表 → (B, T) 的编号张量，T = 最长的一句（不超过 max_len），用 <pad> 补齐。"""
    ids = [vocab.encode(t)[:max_len] for t in texts]
    T = max(len(i) for i in ids)
    p = vocab.stoi[pad]
    return torch.tensor([i + [p] * (T - len(i)) for i in ids], dtype=torch.long)
