"""A4 参考实现（从 A3 复制，加了 embed_list）：连接器（MLP、2×2 合并、Q-Former）、聊天模板、把视觉 token 填进占位符的 VLM。"""
import torch
import torch.nn as nn

from .nn import LayerNorm, Linear, MultiHeadAttention, gelu

IMAGE = "<image>"


def merge_2x2(x, hw):
    """(B, N, d) 按行排列的 patch，网格 (h, w) 都是偶数 → (B, N/4, 4d)。新 token 是 2×2 小方块的 4 个 patch
    按 左上、右上、左下、右下 拼接，新网格 (h/2, w/2) 也按行排列。"""
    B, N, d = x.shape
    h, w = hw
    x = x.reshape(B, h // 2, 2, w // 2, 2, d).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(B, (h // 2) * (w // 2), 4 * d)


class MLPProjector(nn.Module):
    """LLaVA：Linear → GELU → Linear，逐个 patch 作用，token 数不变。"""

    def __init__(self, d_v, d):
        super().__init__()
        self.fc1, self.fc2 = Linear(d_v, d), Linear(d, d)

    def forward(self, x, hw):
        return self.fc2(gelu(self.fc1(x)))


class MergeProjector(nn.Module):
    """Qwen2-VL 的 PatchMerger：merge_2x2 → LayerNorm → Linear → GELU → Linear，token 数变成 1/4。"""

    def __init__(self, d_v, d):
        super().__init__()
        self.ln = LayerNorm(4 * d_v)
        self.fc1, self.fc2 = Linear(4 * d_v, 4 * d_v), Linear(4 * d_v, d)

    def forward(self, x, hw):
        return self.fc2(gelu(self.fc1(self.ln(merge_2x2(x, hw)))))


class QFormerLayer(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.n1, self.n2, self.n3 = LayerNorm(d), LayerNorm(d), LayerNorm(d)
        self.self_attn, self.cross_attn = MultiHeadAttention(d, n_head), MultiHeadAttention(d, n_head)
        self.fc1, self.fc2 = Linear(d, 4 * d), Linear(4 * d, d)

    def forward(self, q, patches):
        q = q + self.self_attn(self.n1(q))
        q = q + self.cross_attn(self.n2(q), kv=patches)
        return q + self.fc2(gelu(self.fc1(self.n3(q))))


class QFormer(nn.Module):
    """n_query 个可学习的查询，用交叉注意力从所有 patch 里取信息；无论多少个 patch，输出都是 n_query 个 token。"""

    def __init__(self, d_v, d, n_query=16, n_layer=2, n_head=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, n_query, d_v) * 0.02)
        self.layers = nn.ModuleList(QFormerLayer(d_v, n_head) for _ in range(n_layer))
        self.ln = LayerNorm(d_v)
        self.out = Linear(d_v, d)

    def forward(self, x, hw):
        q = self.query.expand(x.shape[0], -1, -1)
        for layer in self.layers:
            q = layer(q, x)
        return self.out(self.ln(q))


def chat_template(messages, n_img, encode, bos, eos, img):
    """messages：用户和助手轮流发言（从用户开始）。拼法：
        [bos]
        用户：  encode("User: ") + 内容（每个 "<image>" 换成 n_img 个 img）+ encode("\\n")
        助手：  encode("Assistant: ") + encode(内容) + [eos]
    返回 (ids, targets)：targets[t] = ids[t + 1]，只保留"预测助手内容和它后面的 eos"的位置，其余为 −100，最后一位补 −100。"""
    ids, sup = [bos], [False]
    for m in messages:
        if m["role"] == "user":
            seq = encode("User: ")
            for i, part in enumerate(m["content"].split(IMAGE)):
                seq += ([img] * n_img if i else []) + encode(part)
            seq += encode("\n")
            ids += seq
            sup += [False] * len(seq)
        else:
            head, body = encode("Assistant: "), encode(m["content"]) + [eos]
            ids += head + body
            sup += [False] * len(head) + [True] * len(body)
    targets = [ids[t + 1] if sup[t + 1] else -100 for t in range(len(ids) - 1)] + [-100]
    return ids, targets


def pad_batch(examples, pad_id):
    T = max(len(i) for i, _ in examples)
    ids = torch.full((len(examples), T), pad_id, dtype=torch.long)
    tg = torch.full((len(examples), T), -100, dtype=torch.long)
    for b, (i, t) in enumerate(examples):
        ids[b, :len(i)] = torch.tensor(i)
        tg[b, :len(t)] = torch.tensor(t)
    return ids, tg


def scatter_visual(emb, visual, mask):
    """emb (B, T, d) 文字嵌入，visual (B, n, d)，mask (B, T) 每行恰好 n 个 True。
    返回新张量：第 b 行第 k 个 True 的位置换成 visual[b, k]，其余不变。个数不对时抛出 ValueError。"""
    if not mask.sum(1).eq(visual.shape[1]).all():
        raise ValueError(f"每条序列应有 {visual.shape[1]} 个图像占位符，实际 {mask.sum(1).tolist()}")
    emb = emb.clone()
    emb[mask] = visual.reshape(-1, visual.shape[-1]).to(emb.dtype)
    return emb


class VLM(nn.Module):
    def __init__(self, vision, connector, lm, img_id):
        super().__init__()
        self.vision, self.connector, self.lm = vision, connector, lm
        self.img_id = img_id

    def visual_tokens(self, images):
        x, hw = self.vision.patch_tokens(images)
        return self.connector(x, hw)

    def embed(self, images, ids):
        """文字嵌入，再把 ids == img_id 的位置按顺序换成视觉 token。images (B, C, H, W) 或 (B, k, C, H, W)（每条 k 张图）。"""
        emb = self.lm.tok(ids)
        if images.dim() == 5:
            B, k = images.shape[:2]
            v = self.visual_tokens(images.flatten(0, 1))
            v = v.reshape(B, k * v.shape[1], v.shape[2])
        else:
            v = self.visual_tokens(images)
        return scatter_visual(emb, v, ids == self.img_id)

    def embed_list(self, images, ids):
        """images：列表，第 i 项 (k_i, C, H, W)，按占位符在批里出现的顺序排（第 0 行从左到右，再第 1 行……；
        打包时就是箱子里样本的顺序）。各项的图像大小可以不同：同样大小的拼在一起过视觉塔。"""
        groups = {}
        for i, x in enumerate(images):
            groups.setdefault(tuple(x.shape[1:]), []).append(i)
        out = [None] * len(images)
        for idx in groups.values():
            v = self.visual_tokens(torch.cat([images[i] for i in idx]))
            s = 0
            for i in idx:
                k = images[i].shape[0]
                out[i] = v[s:s + k].reshape(-1, v.shape[-1])
                s += k
        vis = torch.cat(out)
        mask = ids == self.img_id
        if int(mask.sum()) != len(vis):
            raise ValueError(f"占位符 {int(mask.sum())} 个，视觉 token {len(vis)} 个")
        emb = self.lm.tok(ids).clone()
        emb[mask] = vis.to(emb.dtype)
        return emb

    def forward(self, images, ids, targets=None):
        return self.lm(inputs_embeds=self.embed(images, ids), targets=targets)

    @torch.no_grad()
    def generate(self, images, prompt_ids, max_new=40, eos=None, banned=()):
        """贪心生成。prompt_ids (B, T0) 同一批长度相同。先把整个提示（含视觉 token）预填充进 KV cache，之后每步送 1 个 token。
        banned：永远不生成的 token（如 <pad>、<bos>、<img>：回答里出现 <img> 会让占位符的个数对不上）。"""
        was = self.training
        self.eval()
        caches = [{} for _ in self.lm.blocks]
        logits, _ = self.lm(inputs_embeds=self.embed(images, prompt_ids), caches=caches)
        out, done = [], torch.zeros(prompt_ids.shape[0], dtype=torch.bool, device=prompt_ids.device)
        pos = prompt_ids.shape[1]
        for _ in range(max_new):
            last = logits[:, -1].clone()
            last[:, list(banned)] = float("-inf")
            nxt = last.argmax(-1, keepdim=True)
            out.append(nxt)
            if eos is not None:
                done |= nxt[:, 0] == eos
                if done.all():
                    break
            if pos >= self.lm.cfg.block_size:
                break
            logits, _ = self.lm(nxt, caches=caches, pos0=pos)
            pos += 1
        self.train(was)
        return torch.cat(out, 1)


def set_trainable(vlm, vision=True, connector=True, lm=True):
    for part, on in ((vlm.vision, vision), (vlm.connector, connector), (vlm.lm, lm)):
        for p in part.parameters():
            p.requires_grad_(on)
