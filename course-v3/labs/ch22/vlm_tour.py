"""第 22 章：搭一个 VLM。第 20 章的 CLIP 视觉塔 + 连接器 + 第 16 章的 GPT，先对齐、再指令微调；比较连接器和训练方案；
最后用同样的结构做生成式 OCR。默认只用 CPU（GPU 被别的程序占用时也能跑）。

    python -m labs.ch22.vlm_tour             # 全部 6 个部分（CPU 上约 3–4 小时；需要第 16、20 章保存的模型）
    python -m labs.ch22.vlm_tour 2 6         # 第 2 部分训练并保存主模型，第 6 部分读取它
    python -m labs.ch22.vlm_tour --gpu       # 有空闲 GPU 时用 GPU
"""
import math
import os
import sys
import time

if "--gpu" not in sys.argv:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn.functional as F

from miniocr.data.corpus import text_vocab
from miniocr.data.shapes import COLORS, SHAPES, describe, draw_scene, sample_scene
from miniocr.data.text import LineDataset
from miniocr.metrics import cer
from miniocr.models.gpt import GPT, GPTConfig, apply_rope
from miniocr.models.vit import ViT, ViTConfig
from miniocr.models.vlm import (VLM, MergeProjector, MLPProjector, QFormer, build_example, n_visual_tokens, pad_batch,
                                set_trainable)
from miniocr.train.optim import warmup_cosine

DEV = "cuda" if "--gpu" in sys.argv and torch.cuda.is_available() else "cpu"
torch.set_num_threads(min(16, os.cpu_count() or 1))
OUT = "outputs/ch22"
os.makedirs(OUT, exist_ok=True)
V = text_vocab()
IMG, BOS, EOS, PAD = (V.stoi[s] for s in ("<img>", "<bos>", "<eos>", "<pad>"))
TASKS = ("caption", "count", "left", "exists")
BS = 32
_C = {}


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ---------------------------------------------------------------------- 数据
def qa(objs, task, rng):
    """一张场景图在某个任务上的 (问题, 答案)。"""
    if task == "caption":
        return "Describe the image.", describe(objs)
    if task == "count":
        return "How many objects?", str(len(objs))
    if task == "left":
        return "What color is the leftmost object?", min(objs, key=lambda o: o["box"][0])["color"]
    present = {(o["color"], o["shape"]) for o in objs}
    if rng.random() < 0.5:
        c, s = sorted(present)[int(rng.integers(len(present)))]
    else:
        absent = [(c, s) for c in COLORS for s in SHAPES if (c, s) not in present]
        c, s = absent[int(rng.integers(len(absent)))]
    return f"Is there a {c} {s}?", "yes" if (c, s) in present else "no"


def scenes(seed, n):
    if ("s", seed, n) not in _C:
        rng = np.random.default_rng(seed)
        X, objs = [], []
        for _ in range(n):
            o = sample_scene(rng, 64, 64, 1, 3)
            X.append(draw_scene(o, rng, 64, 64, noise=0.03).transpose(2, 0, 1))
            objs.append(o)
        _C[("s", seed, n)] = (torch.from_numpy(np.stack(X)), objs)
    return _C[("s", seed, n)]


def test_set(task, n=300):
    """每个任务 300 个测试样本（种子固定，所有模型用同一份）。"""
    if ("t", task) not in _C:
        X, objs = scenes(100 + TASKS.index(task), n)
        rng = np.random.default_rng(200 + TASKS.index(task))
        _C[("t", task)] = (X, [qa(o, task, rng) for o in objs])
    return _C[("t", task)]


def img_float(xb):
    xb = xb.to(DEV).float()
    return xb / 127.5 - 1.0 if xb.shape[1] == 3 else xb


# ---------------------------------------------------------------------- 模型
NEED = {"outputs/ch16/gpt_char.pt": "python -m labs.ch16.gpt_tour 3", "outputs/ch20/clip_shapes.pt": "python -m labs.ch20.clip_tour 2",
        "outputs/ch20/clip_words.pt": "python -m labs.ch20.clip_tour 6", "outputs/ch22/vlm_mlp_stage1.pt": "python -m labs.ch22.vlm_tour 2",
        "outputs/ch26/sft.pt": "python -m labs.ch26.sft_tour 2（第 2 部分会自动保存）"}


def need(path):
    """前面章节保存的模型不存在时，告诉你该先跑哪个命令（而不是抛出一长串 FileNotFoundError）。"""
    if not os.path.exists(path):
        sys.exit(f"  缺少 {path}：先运行 {NEED.get(path, '前面章节的演示')}")
    return path


def load_lm(pretrained=True):
    ck = torch.load(need("outputs/ch16/gpt_char.pt"), map_location="cpu")
    lm = GPT(GPTConfig(**ck["cfg"]))
    if pretrained:
        lm.load_state_dict(ck["model"])
    return lm


def load_vit(pretrained=True, words=False):
    if words:
        vit = ViT(ViTConfig(img_size=32, patch=8, in_ch=1, d_model=128, n_layer=4, n_head=4, pos="rope2d", pool="mean"))
        path = "outputs/ch20/clip_words.pt"
    else:
        vit = ViT(ViTConfig(img_size=64, patch=8, d_model=128, n_layer=4, n_head=4, pos="learned", pool="mean"))
        path = "outputs/ch20/clip_shapes.pt"
    if pretrained:
        sd = torch.load(need(path), map_location="cpu")
        vit.load_state_dict({k[len("image_encoder."):]: v for k, v in sd.items() if k.startswith("image_encoder.")})
    return vit


def build(connector="mlp", pretrained=True, words=False, n_query=16):
    torch.manual_seed(0)
    vit, lm = load_vit(pretrained, words), load_lm(pretrained)
    conn = {"mlp": MLPProjector(128, lm.cfg.d_model), "merge": MergeProjector(128, lm.cfg.d_model),
            "qformer": QFormer(128, lm.cfg.d_model, n_query=n_query)}[connector]
    m = VLM(vit, conn, lm, IMG).to(DEV)
    m.kind, m.n_query = connector, n_query
    return m


def n_vis(m, n_patch=64):
    return n_visual_tokens(m.kind, n_patch, m.n_query)


def n_params(mod, trainable=False):
    return sum(p.numel() for p in mod.parameters() if p.requires_grad or not trainable)


# ---------------------------------------------------------------------- 训练与评测
def train(m, steps, tasks, lr, train_n=20000, log_every=0, seed=0, data=None):
    """tasks：每一步从这些任务里均匀地随机选，组成一个批。data 给出时是 (X, [(问题, 答案), …]) 的固定数据（OCR 用）。"""
    m.train()
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    if data is None:
        X, objs = scenes(0, train_n)
    else:
        X, pairs = data
    nv = n_vis(m, 64 if data is None else 128)
    t0 = time.time()
    for step in range(steps):
        idx = rng.integers(0, len(X), BS)
        if data is None:
            ex = [build_example(V, nv, *qa(objs[i], tasks[int(rng.integers(len(tasks)))], rng)) for i in idx]
        else:
            ex = [build_example(V, nv, *pairs[i]) for i in idx]
        ids, tg = pad_batch(ex, PAD)
        for gr in opt.param_groups:
            gr["lr"] = warmup_cosine(step, steps, 100, lr, lr * 0.1)
        loss = m(img_float(X[idx]), ids.to(DEV), tg.to(DEV))[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if log_every and (step + 1) % log_every == 0:
            print(f"      第 {step + 1:4d} 步  损失 {loss.item():.4f}  {time.time() - t0:.0f} 秒")
    return time.time() - t0


@torch.no_grad()
def answer(m, X, questions, n_patch=64, max_new=None, bs=100):
    """对每张图和问题生成答案。同一批里提示必须等长：先按提示长度分组。"""
    nv = n_vis(m, n_patch)
    prompts = [build_example(V, nv, q, "")[0][:-1] for q in questions]      # 去掉 <eos>
    out = [None] * len(prompts)
    groups = {}
    for i, p in enumerate(prompts):
        groups.setdefault(len(p), []).append(i)
    for idxs in groups.values():
        for k in range(0, len(idxs), bs):
            sub = idxs[k:k + bs]
            ids = torch.tensor([prompts[i] for i in sub], device=DEV)
            gen = m.generate(img_float(X[sub]), ids, max_new=max_new or 90, eos=EOS)
            for i, g in zip(sub, gen.tolist()):
                out[i] = V.decode(g)
    return out


def evaluate(m, label="", show=True):
    res = {}
    for task in TASKS:
        X, pairs = test_set(task)
        pred = answer(m, X, [q for q, _ in pairs], max_new=90 if task == "caption" else 10)
        refs = [a for _, a in pairs]
        acc = np.mean([p == r for p, r in zip(pred, refs)])
        res[task] = (acc, cer(pred, refs) if task == "caption" else None, pred)
    if show:
        print(f"    {label:<34} 描述 全对 {res['caption'][0]:6.1%}（CER {res['caption'][1]:6.2%}）  数数 {res['count'][0]:6.1%}  "
              f"最左颜色 {res['left'][0]:6.1%}  有没有 {res['exists'][0]:6.1%}")
    return res


# ---------------------------------------------------------------------- 1
def part1_sequence():
    section("1. 一条训练样本长什么样")
    m = build("mlp")
    X, objs = scenes(0, 1)
    q, a = qa(objs[0], "left", np.random.default_rng(0))
    ids, tg = build_example(V, 64, q, a)
    show = lambda i: {IMG: "<img>", BOS: "<bos>", EOS: "<eos>"}.get(i, V.itos[i].replace("\n", "\\n"))
    print(f"  问题 '{q}'，答案 '{a}'；序列长 {len(ids)}：1 个 <bos> + 64 个 <img> + {len(ids) - 66 - len(a)} 个问题字符 + "
          f"{len(a)} 个答案字符 + 1 个 <eos>")
    print("  位置    输入        要预测的下一个 token（−100 = 不计算损失）")
    for t in list(range(0, 3)) + list(range(64, len(ids))):
        print(f"    {t:3d}   {show(ids[t]):<10}  {show(tg[t]) if tg[t] != -100 else '−100'}")
        if t == 2:
            print("    ...   （位置 1–64 都是 <img>，不计算损失）")
    print(f"  计算损失的位置只有 {sum(t != -100 for t in tg)} 个：答案的 {len(a)} 个字符 + <eos>")
    vis = m.visual_tokens(img_float(X[:1]))
    print(f"  视觉 token：ViT 输出 (1, 64, 128) → MLP 连接器 → {tuple(vis.shape)}，与文字嵌入（{m.lm.cfg.d_model} 维）同一个空间")
    print(f"  参数：ViT {n_params(m.vision):,}  连接器 {n_params(m.connector):,}  GPT {n_params(m.lm):,}")
    for name, kind in (("MLP（LLaVA 式）", "mlp"), ("2×2 合并（Qwen 式）", "merge"), ("Q-Former 16 个查询（BLIP-2 式）", "qformer")):
        mm = build(kind)
        print(f"    连接器 {name:<30} 参数 {n_params(mm.connector):>8,}  视觉 token {n_vis(mm):3d} 个")
    print(f"  描述任务的序列约 {66 + 26 + 60} 个 token，其中视觉 token 占 64 个：图像往往比文字长")


# ---------------------------------------------------------------------- 2
def part2_two_stage():
    section("2. 两阶段训练：先对齐，再指令微调")
    m = build("mlp")
    print("  阶段 0：什么都没训练（连接器是随机的）")
    evaluate(m, "未训练")
    n = set_trainable(m, vision=False, connector=True, lm=False)
    print(f"  阶段 1（对齐）：冻结 ViT 和 GPT，只训练连接器（{n:,} 个参数，占全部的 {n / n_params(m):.1%}）；只用'描述'数据，1500 步")
    sec = train(m, 1500, ("caption",), 1e-3, log_every=500)
    print(f"    用时 {sec:.0f} 秒")
    torch.save(m.state_dict(), f"{OUT}/vlm_mlp_stage1.pt")                 # 第 23 章从这里开始
    r1 = evaluate(m, "阶段 1 之后")
    print(f"    问一个没训练过的问题 '{test_set('count')[1][0][0]}' → '{r1['count'][2][0]}'（真值 '{test_set('count')[1][0][1]}'）")
    n = set_trainable(m)
    print(f"  阶段 2（指令微调）：全部解冻（{n:,} 个参数）；四种任务各占 1/4，2000 步")
    sec = train(m, 2000, TASKS, 3e-4, log_every=500)
    print(f"    用时 {sec:.0f} 秒")
    r2 = evaluate(m, "阶段 2 之后")
    torch.save(m.state_dict(), f"{OUT}/vlm_mlp.pt")
    X, pairs = test_set("caption")
    for i in range(3):
        print(f"    描述 例 {i}：真值 '{pairs[i][1]}'\n              生成 '{r2['caption'][2][i]}'")
    _C["main"] = r2


# ---------------------------------------------------------------------- 3
def part3_ablations():
    section("3. 对照：不对齐、从零训练、冻结视觉塔、冻结语言模型")
    print("  总步数都是 3500 步（批 32），评测同一份测试集")
    if "main" in _C:
        r = _C["main"]
        print(f"    {'两阶段（第 2 部分）':<34} 描述 全对 {r['caption'][0]:6.1%}（CER {r['caption'][1]:6.2%}）  数数 {r['count'][0]:6.1%}  "
              f"最左颜色 {r['left'][0]:6.1%}  有没有 {r['exists'][0]:6.1%}")
    m = build("mlp")
    sec = train(m, 3500, TASKS, 3e-4)
    evaluate(m, f"跳过阶段 1（{sec:.0f} 秒）")
    m = build("mlp", pretrained=False)
    sec = train(m, 3500, TASKS, 1e-3)
    evaluate(m, f"从零训练（随机 ViT + 随机 GPT，{sec:.0f} 秒）")
    for name, kw in (("阶段 2 冻结 ViT", dict(vision=False)), ("阶段 2 冻结 GPT", dict(lm=False))):
        m = build("mlp")
        set_trainable(m, vision=False, lm=False)
        s1 = train(m, 1500, ("caption",), 1e-3)
        set_trainable(m, **kw)
        s2 = train(m, 2000, TASKS, 3e-4)
        evaluate(m, f"{name}（{s1 + s2:.0f} 秒）")


# ---------------------------------------------------------------------- 4
def part4_connectors():
    section("4. 连接器：MLP、2×2 合并、Q-Former")
    print("  训练方案与第 2 部分相同（阶段 1：1500 步，阶段 2：2000 步）")
    if "main" in _C:
        r = _C["main"]
        print(f"    {'MLP（64 个视觉 token，第 2 部分）':<34} 描述 全对 {r['caption'][0]:6.1%}（CER {r['caption'][1]:6.2%}）  数数 {r['count'][0]:6.1%}  "
              f"最左颜色 {r['left'][0]:6.1%}  有没有 {r['exists'][0]:6.1%}")
    for kind, nq, name in (("merge", 16, "2×2 合并（16 个）"), ("qformer", 16, "Q-Former（16 个查询）"),
                           ("qformer", 4, "Q-Former（4 个查询）")):
        m = build(kind, n_query=nq)
        set_trainable(m, vision=False, lm=False)
        s1 = train(m, 1500, ("caption",), 1e-3)
        set_trainable(m)
        s2 = train(m, 2000, TASKS, 3e-4)
        evaluate(m, f"{name}（{s1 + s2:.0f} 秒）")


# ---------------------------------------------------------------------- 5
def line_data(seed, n):
    if ("l", seed, n) not in _C:
        ds = LineDataset(seed, n, noise=0.15, distort=1.0)                 # 与第 13 章相同的数据和增强
        _C[("l", seed, n)] = (ds.x, [("Read the text.", t) for t in ds.texts])
    return _C[("l", seed, n)]


def part5_ocr():
    section("5. 生成式 OCR：同一个结构读文字行")
    tr = line_data(0, 20000)
    te = line_data(2, 2000)
    print("  数据与第 13 章完全相同：训练 20000 行、测试 2000 行（32×256，噪声 + 形变）")
    m = build("merge", words=True)
    print(f"  视觉塔：第 20 章在单词图上 CLIP 预训练的 ViT（二维 RoPE，所以 32×256 的行图直接可用：4×32 = 128 个 patch）；"
          f"2×2 合并 → 32 个视觉 token；全部参数一起训练 3000 步")
    sec = train(m, 3000, None, 5e-4, data=tr, log_every=1000)
    torch.save(m.state_dict(), f"{OUT}/vlm_ocr.pt")
    X, pairs = te
    pred = answer(m, X, [q for q, _ in pairs], n_patch=128, max_new=40)
    refs = [a for _, a in pairs]
    print(f"  用时 {sec:.0f} 秒；测试 CER {cer(pred, refs):.2%}  整行全对 {np.mean([p == r for p, r in zip(pred, refs)]):.1%}"
          f"（第 13 章的 CRNN：CER 2.28%、整行全对 80.1%）")
    wrong = [(r, p) for p, r in zip(pred, refs) if p != r][:6]
    for r, p in wrong:
        print(f"    真值 '{r}'\n    生成 '{p}'")
    longer = np.mean([len(p) > len(r) + 3 for p, r in zip(pred, refs)])
    shorter = np.mean([len(p) < len(r) - 3 for p, r in zip(pred, refs)])
    print(f"  生成结果比真值长 3 个字符以上的 {longer:.1%}，短 3 个以上的 {shorter:.1%}")


# ---------------------------------------------------------------------- 6
def object_patches(o, p=8):
    x0, y0, x1, y1 = o["box"]
    return {r * 8 + c for r in range(y0 // p, (y1 - 1) // p + 1) for c in range(x0 // p, (x1 - 1) // p + 1)}


@torch.no_grad()
def lm_attention(m, emb):
    """每一层、每个头的注意力权重 (L, h, T, T)：用钩子拿到每层注意力的输入，再按第 16 章的公式重算一遍。"""
    ws, hooks = [], []
    def hook(mod, args):
        x, cos, sin = args[:3]
        B, T, d = x.shape
        q, k, _ = mod.qkv(x).view(B, T, 3, mod.h, mod.dh).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        s = q @ k.transpose(-2, -1) / math.sqrt(mod.dh)
        s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool, device=x.device).tril(), float("-inf"))
        ws.append(s.softmax(-1)[0])
    for b in m.lm.blocks:
        hooks.append(b.attn.register_forward_pre_hook(hook))
    m.lm(inputs_embeds=emb)
    for h in hooks:
        h.remove()
    return torch.stack(ws)


def part6_attention():
    section("6. 回答'最左边是什么颜色'时，语言模型在看哪些视觉 token")
    path = f"{OUT}/vlm_mlp.pt"
    if not os.path.exists(path):
        print(f"  （缺少 {path}，先运行第 2 部分）")
        return
    m = build("mlp")
    m.load_state_dict(torch.load(path))
    m.eval()
    X, objs = scenes(300, 400)
    stats = np.zeros((m.lm.cfg.n_layer, 4))
    cnt = 0
    for i in range(len(X)):
        if len(objs[i]) < 2:
            continue
        ids = torch.tensor([build_example(V, 64, "What color is the leftmost object?", "")[0][:-1]], device=DEV)
        A = lm_attention(m, m.embed(img_float(X[i:i + 1]), ids))           # (L, h, T, T)
        a = A[:, :, -1, 1:65].mean(1).cpu()                                # 最后一个位置（要说出答案的位置）对 64 个视觉 token
        li = min(range(len(objs[i])), key=lambda j: objs[i][j]["box"][0])
        left = object_patches(objs[i][li])
        other = set().union(*[object_patches(o) for j, o in enumerate(objs[i]) if j != li]) - left
        bg = set(range(64)) - left - other
        for L in range(len(a)):
            stats[L] += [a[L, list(left)].mean(), a[L, list(other)].mean() if other else 0, a[L, list(bg)].mean(),
                         A[L, :, -1, 1:65].sum(-1).mean().item()]
        cnt += 1
    print(f"  {cnt} 张有 2–3 个物体的图；每个视觉 token 平均分到的注意力（×100；64 个视觉 token 平均分时是 1/64 ≈ 1.56 × 视觉部分的总注意力）")
    print("     层    最左物体的 token   其他物体   背景     视觉 token 合计占的注意力")
    for L, (l, o, b, tot) in enumerate(stats / cnt):
        print(f"    第 {L + 1} 层      {100 * l:6.2f}        {100 * o:6.2f}   {100 * b:6.2f}        {tot:6.1%}")


PARTS = {1: part1_sequence, 2: part2_two_stage, 3: part3_ablations, 4: part4_connectors, 5: part5_ocr,
         6: part6_attention}

if __name__ == "__main__":
    want = [int(a) for a in sys.argv[1:] if a.isdigit()] or list(PARTS)
    for k in want:
        PARTS[k]()
    if want == list(PARTS):
        print("\n第 22 章的演示全部完成。")
