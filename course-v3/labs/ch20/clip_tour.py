"""第 20 章：对比学习。在 ShapesWorld（图 ↔ 描述）和 MiniDoc（单词图 ↔ 单词）上各训练一个视觉底座。
默认只用 CPU（GPU 被别的程序占用时也能跑）。

    python -m labs.ch20.clip_tour            # 全部 6 个部分（CPU 上约 1.5–2 小时；第 2、6 部分各约 25 分钟）
    python -m labs.ch20.clip_tour 2 5        # 只运行第 2、5 部分（第 5 部分用第 2 部分保存的模型）
    python -m labs.ch20.clip_tour --gpu      # 有空闲 GPU 时用 GPU

保存的模型（第 22 章用）：outputs/ch20/clip_shapes.pt、outputs/ch20/clip_words.pt
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
from miniocr.data.render import render_line
from miniocr.data.shapes import COLORS, SHAPES, SIZES, describe, draw_scene, sample_scene
from miniocr.data.text import WORDS, _token
from miniocr.models.clip import DualEncoder, TextEncoder, clip_loss, encode_texts, recall_at_k, siglip_loss, zero_shot
from miniocr.models.vit import ViT, ViTConfig
from miniocr.train.optim import warmup_cosine

DEV = "cuda" if "--gpu" in sys.argv and torch.cuda.is_available() else "cpu"
torch.set_num_threads(min(16, os.cpu_count() or 1))
OUT = "outputs/ch20"
os.makedirs(OUT, exist_ok=True)
V = text_vocab()
HELD_OUT = ("yellow", "triangle")             # 训练集里去掉所有含"黄色三角形"的场景，测试组合泛化
_C = {}


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ---------------------------------------------------------------------- 数据
def has_held_out(objs):
    return any((o["color"], o["shape"]) == HELD_OUT for o in objs)


def shape_pairs(seed, n, hold_out=True, min_objects=1, max_objects=3):
    """n 张 64×64 的场景图和它们的描述。hold_out=True 时跳过含黄色三角形的场景。返回 (X uint8 (n,3,64,64), 描述列表, 物体列表)。"""
    key = ("shapes", seed, n, hold_out, min_objects, max_objects)
    if key not in _C:
        rng = np.random.default_rng(seed)
        X, caps, all_objs = [], [], []
        while len(caps) < n:
            objs = sample_scene(rng, 64, 64, min_objects, max_objects)
            img = draw_scene(objs, rng, 64, 64, noise=0.03)
            if hold_out and has_held_out(objs):
                continue
            X.append(img.transpose(2, 0, 1))
            caps.append(describe(objs))
            all_objs.append(objs)
        _C[key] = (torch.from_numpy(np.stack(X)), caps, all_objs)
    return _C[key]


def word_pairs(seed, n, kind="mixed"):
    """n 张 32×128 的单词图（灰度，uint8 (n,1,32,128)，白底黑字）和对应的文字。
    kind="mixed"：第 13 章文字行的"词"（单词、数字、价格、日期、百分比）；"lower"：WORDS 里的小写单词；
    "num4"：四位数字。"""
    key = ("words", seed, n, kind)
    if key not in _C:
        rng = np.random.default_rng(seed)
        X, texts = [], []
        while len(texts) < n:
            if kind == "mixed":
                t = _token(rng)
            elif kind == "lower":
                t = WORDS[int(rng.integers(len(WORDS)))]
            else:
                t = f"{int(rng.integers(1000, 10000))}"
            try:
                img, _ = render_line(t, rng, H=32, W=128, noise=0.1, distort=0.5)
            except ValueError:
                continue
            X.append(img[None])
            texts.append(t)
        _C[key] = (torch.from_numpy(np.stack(X)), texts)
    return _C[key]


def to_float(xb):
    xb = xb.to(DEV).float()
    return xb / 127.5 - 1.0 if xb.shape[1] == 3 else (255.0 - xb) / 255.0          # 灰度图：墨迹 = 1


# ---------------------------------------------------------------------- 模型与训练
def make(kind="clip", fixed_scale=None, words=False):
    if words:
        img = ViT(ViTConfig(img_size=32, patch=8, in_ch=1, d_model=128, n_layer=4, n_head=4, pos="rope2d", pool="mean"))
    else:
        img = ViT(ViTConfig(img_size=64, patch=8, d_model=128, n_layer=4, n_head=4, pos="learned", pool="mean"))
    txt = TextEncoder(len(V), max_len=96, d=128, n_layer=3, n_head=4, pad_id=V.stoi["<pad>"])
    return DualEncoder(img, txt, 128, 128, 128, loss=kind, fixed_scale=fixed_scale)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def train(model, X, texts, steps, B, lr=1e-3, seed=0, log_every=0):
    torch.manual_seed(seed)
    model.to(DEV).train()
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "pos" not in n]
    rest = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "pos" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": rest, "weight_decay": 0.0}], lr=lr)
    g = torch.Generator().manual_seed(seed)
    order, k = torch.randperm(len(X), generator=g), 0
    t0 = time.time()
    for step in range(steps):
        if k + B > len(X):
            order, k = torch.randperm(len(X), generator=g), 0
        idx = order[k:k + B].tolist()
        k += B
        for gr in opt.param_groups:
            gr["lr"] = warmup_cosine(step, steps, 100, lr, lr * 0.05)
        loss = model(to_float(X[idx]), encode_texts(V, [texts[i] for i in idx], 96).to(DEV))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log_every and (step + 1) % log_every == 0:
            print(f"      第 {step + 1:4d} 步  损失 {loss.item():.3f}  温度 τ = {1 / model.scale().item():.4f}")
    return time.time() - t0


@torch.no_grad()
def embed(model, X=None, texts=None, bs=500):
    model.eval()
    out = []
    if X is not None:
        for i in range(0, len(X), bs):
            out.append(model.encode_image(to_float(X[i:i + bs])).cpu())
    else:
        for i in range(0, len(texts), bs):
            out.append(model.encode_text(encode_texts(V, texts[i:i + bs], 96).to(DEV)).cpu())
    return torch.cat(out)


def retrieval(model, X, caps):
    zi, zt = embed(model, X), embed(model, texts=caps)
    sim = zi @ zt.T
    same = torch.tensor([[a == b for b in caps] for a in caps])
    return dict(i2t1=recall_at_k(sim, 1), i2t1_same=recall_at_k(sim, 1, same), i2t5_same=recall_at_k(sim, 5, same),
                t2i1_same=recall_at_k(sim.T, 1, same), dup=(same.sum(1) > 1).float().mean().item())


PROMPTS = [f"a {s} {c} {sh}" for s in SIZES for c in COLORS for sh in SHAPES]


def single_objects(seed=2, n=2000):
    """只有一个物体的测试图（包括黄色三角形），标签是 PROMPTS 里的编号。"""
    X, caps, objs = shape_pairs(seed, n, hold_out=False, min_objects=1, max_objects=1)
    y = torch.tensor([PROMPTS.index(c) for c in caps])
    return X, y, objs


def zero_shot_report(model, indent="    "):
    X, y, objs = single_objects()
    pred = zero_shot(embed(model, X), embed(model, texts=PROMPTS))
    held = torch.tensor([has_held_out(o) for o in objs])
    parts = [p.split() for p in PROMPTS]
    attr = lambda i, k: torch.tensor([parts[int(j)][k] == parts[int(t)][k] for j, t in zip(pred, y)]).float().mean().item()
    acc = (pred == y).float()
    print(f"{indent}零样本分类（单个物体，{len(PROMPTS)} 个类别 'a <大小> <颜色> <形状>'）：总体 {acc.mean():.1%}；"
          f"大小 {attr(0, 1):.1%}  颜色 {attr(0, 2):.1%}  形状 {attr(0, 3):.1%}")
    print(f"{indent}  其中训练时见过的组合 {acc[~held].mean():.1%}（{(~held).sum().item()} 张），"
          f"从没见过的'黄色三角形' {acc[held].mean():.1%}（{held.sum().item()} 张）")
    return acc.mean().item()


# ---------------------------------------------------------------------- 1
def part1_by_hand():
    section("1. 手算：3 张图、3 句话")
    zi = F.normalize(torch.tensor([[1.0, 0.2, 0.0], [0.1, 1.0, 0.1], [0.0, 0.3, 1.0]]), dim=-1)
    zt = F.normalize(torch.tensor([[0.9, 0.1, 0.1], [0.2, 0.8, 0.4], [0.1, 0.6, 0.8]]), dim=-1)
    sim = zi @ zt.T
    print("  余弦相似度 S = zi · ztᵀ（第 i 行是第 i 张图，第 j 列是第 j 句话；对角线是配对）：")
    for r in sim:
        print("    " + "  ".join(f"{v:.3f}" for v in r.tolist()))
    for s in (1.0, 10.0, 100.0):
        L = s * sim
        row = torch.softmax(L, 1).diagonal()
        col = torch.softmax(L, 0).diagonal()
        print(f"  s = 1/τ = {s:5.0f}：按行 softmax 后配对的概率 {[round(v, 3) for v in row.tolist()]}，"
              f"按列 {[round(v, 3) for v in col.tolist()]}；CLIP 损失 {clip_loss(zi, zt, torch.tensor(s)).item():.4f}")
    for s, b in ((10.0, -10.0), (10.0, -5.0)):
        L = s * sim + b
        print(f"  SigLIP s = {s:.0f}、b = {b:.0f}：每一对的 σ(logit) =")
        for r in torch.sigmoid(L):
            print("    " + "  ".join(f"{v:.3f}" for v in r.tolist()))
        print(f"    损失 {siglip_loss(zi, zt, torch.tensor(s), torch.tensor(b)).item():.4f}")
    print(f"  随机猜的 CLIP 损失 = ln B = ln 3 = {math.log(3):.4f}")


# ---------------------------------------------------------------------- 2
def part2_train_shapes():
    section("2. 在 ShapesWorld 上训练 CLIP：图 ↔ 描述")
    X, caps, _ = shape_pairs(0, 30000)
    print(f"  训练 {len(caps)} 对（1–3 个物体，去掉了所有含黄色三角形的场景），例：'{caps[0]}'、'{caps[1]}'")
    print(f"  不同的描述 {len(set(caps))} 种；其中只有一个物体的描述只有 {len([c for c in set(caps) if ',' not in c])} 种")
    torch.manual_seed(0)
    m = make("clip")
    print(f"  模型：ViT 图像编码器 {n_params(m.image_encoder):,} 参数 + 文字编码器 {n_params(m.text_encoder):,} 参数，"
          f"投影到 128 维；批 256，3000 步")
    sec = train(m, X, caps, 3000, 256, log_every=500)
    torch.save(m.state_dict(), f"{OUT}/clip_shapes.pt")
    Xt, ct, _ = shape_pairs(1, 1000)
    r = retrieval(m, Xt, ct)
    print(f"  训练用时 {sec:.0f} 秒；最终温度 τ = {1 / m.scale().item():.4f}（s = {m.scale().item():.1f}）")
    print(f"  检索（1000 对测试，{r['dup']:.0%} 的描述在测试集里有重复）：")
    print(f"    图 → 文 R@1 = {r['i2t1']:.1%}（只有原配算对）  {r['i2t1_same']:.1%}（文字相同就算对）  "
          f"R@5 = {r['i2t5_same']:.1%}")
    print(f"    文 → 图 R@1 = {r['t2i1_same']:.1%}（文字相同就算对）；随机猜 R@1 = 0.1%")
    zero_shot_report(m)


# ---------------------------------------------------------------------- 3
def part3_temperature():
    section("3. 温度：固定 vs 可学习")
    X, caps, _ = shape_pairs(0, 30000)
    Xt, ct, _ = shape_pairs(1, 1000)
    print("  批 128，1500 步；只改温度")
    for name, fixed in (("固定 s = 1（τ = 1）", 1.0), ("固定 s = 14.3（τ = 0.07）", 1 / 0.07),
                        ("固定 s = 100（τ = 0.01）", 100.0), ("可学习（初始 τ = 0.07）", None)):
        torch.manual_seed(0)
        m = make("clip", fixed_scale=fixed)
        train(m, X, caps, 1500, 128)
        r = retrieval(m, Xt, ct)
        extra = f"，学到 τ = {1 / m.scale().item():.4f}" if fixed is None else ""
        print(f"    {name:<24} 图 → 文 R@1 = {r['i2t1_same']:6.1%}  文 → 图 R@1 = {r['t2i1_same']:6.1%}{extra}")


# ---------------------------------------------------------------------- 4
def part4_batch():
    section("4. 批大小：CLIP 与 SigLIP")
    X, caps, _ = shape_pairs(0, 30000)
    Xt, ct, _ = shape_pairs(1, 1000)
    print("  每个模型 1000 步；批越大，每一步的负例越多（B − 1 个），计算量也越大")
    for B in (16, 64, 256):
        for kind in ("clip", "siglip"):
            torch.manual_seed(0)
            m = make(kind)
            sec = train(m, X, caps, 1000, B)
            r = retrieval(m, Xt, ct)
            extra = f"  b = {m.logit_bias.item():6.2f}" if kind == "siglip" else ""
            print(f"    批 {B:3d}  {kind:<6}  图 → 文 R@1 = {r['i2t1_same']:6.1%}  文 → 图 R@1 = {r['t2i1_same']:6.1%}  "
                  f"s = {m.scale().item():5.1f}{extra}  （{sec:4.0f} 秒）")


# ---------------------------------------------------------------------- 5
@torch.no_grad()
def features(enc, X, bs=500):
    enc.eval()
    return torch.cat([enc(to_float(X[i:i + bs])).cpu() for i in range(0, len(X), bs)])


def linear_probe(Ftr, ytr, Fte, yte, k, steps=300):
    """冻结的特征上训练一个线性分类器（逻辑回归，LBFGS），返回测试准确率。特征先按训练集标准化。"""
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd
    W = torch.zeros(Ftr.shape[1], k, requires_grad=True)
    b = torch.zeros(k, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(Ftr @ W + b, ytr) + 1e-3 * W.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return ((Fte @ W + b).argmax(-1) == yte).float().mean().item()


def probe_labels(objs_list):
    count = torch.tensor([len(o) - 1 for o in objs_list])
    left = torch.tensor([list(COLORS).index(min(o, key=lambda q: q["box"][0])["color"]) for o in objs_list])
    tri = torch.tensor([int(any(q["shape"] == "triangle" for q in o)) for o in objs_list])
    big = torch.tensor([int(any(q["size"] == "large" for q in o)) for o in objs_list])
    return {"物体个数（3 类）": (count, 3), "最左物体的颜色（4 类）": (left, 4), "有没有三角形（2 类）": (tri, 2),
            "有没有大物体（2 类）": (big, 2)}


def part5_probe():
    section("5. 为什么要预训练视觉塔：冻结特征 + 线性分类器")
    path = f"{OUT}/clip_shapes.pt"
    if not os.path.exists(path):
        print(f"  （缺少 {path}，先运行第 2 部分）")
        return
    Xtr, _, otr = shape_pairs(10, 4000)
    Xte, _, ote = shape_pairs(11, 2000)
    Ltr, Lte = probe_labels(otr), probe_labels(ote)
    encs = {}
    torch.manual_seed(0)
    encs["随机初始化的 ViT"] = make("clip").image_encoder.to(DEV)
    m = make("clip")
    m.load_state_dict(torch.load(path))
    encs["CLIP 预训练的 ViT"] = m.image_encoder.to(DEV)
    sup = f"outputs/ch19/vit_left_learned.pt"
    if os.path.exists(sup):
        v = ViT(ViTConfig(img_size=64, patch=8, d_model=128, n_layer=4, n_head=4, n_classes=4))
        v.load_state_dict(torch.load(sup))
        v.head = None
        encs["第 19 章'最左颜色'监督训练的 ViT"] = v.to(DEV)
    print(f"  训练 4000 张、测试 2000 张（1–3 个物体）；特征都是 128 维，图像编码器完全冻结，只训练线性分类器")
    names = list(Ltr)
    print("    " + " " * 34 + "".join(f"{n:<16}" for n in names))
    for en, enc in encs.items():
        Ftr, Fte = features(enc, Xtr), features(enc, Xte)
        accs = [linear_probe(Ftr, Ltr[n][0], Fte, Lte[n][0], Ltr[n][1]) for n in names]
        print(f"    {en:<32}" + "".join(f"  {a:6.1%}        " for a in accs))


# ---------------------------------------------------------------------- 6
def part6_words():
    section("6. MiniDoc：单词图 ↔ 单词。用检索做'零样本 OCR'")
    X, texts = word_pairs(0, 40000)
    print(f"  训练 {len(texts)} 对 32×128 的单词图（第 13 章的'词'：单词、数字、价格、日期……），例：{texts[:6]}")
    torch.manual_seed(0)
    m = make("clip", words=True)
    sec = train(m, X, texts, 3000, 256, log_every=1000)
    torch.save(m.state_dict(), f"{OUT}/clip_words.pt")
    print(f"  训练用时 {sec:.0f} 秒；τ = {1 / m.scale().item():.4f}")
    Xl, tl = word_pairs(5, 1000, "lower")
    cands = sorted(set(WORDS))
    zt = embed(m, texts=cands)
    pred = zero_shot(embed(m, Xl), zt)
    acc = np.mean([cands[int(p)] == t for p, t in zip(pred, tl)])
    print(f"  闭集识别：1000 张小写单词图，在 {len(cands)} 个候选单词里选最相似的：准确率 {acc:.1%}")
    wrong = [(t, cands[int(p)]) for p, t in zip(pred, tl) if cands[int(p)] != t][:8]
    print(f"    认错的例子（真值 → 选中）：{wrong}")
    Xn, tn = word_pairs(6, 500, "num4")
    rng = np.random.default_rng(7)
    pool = sorted(set(tn) | {str(int(v)) for v in rng.integers(1000, 10000, 1500)})
    pred = zero_shot(embed(m, Xn), embed(m, texts=pool))
    acc_n = np.mean([pool[int(p)] == t for p, t in zip(pred, tn)])
    close = np.mean([sum(a != b for a, b in zip(pool[int(p)], t)) <= 1 for p, t in zip(pred, tn)])
    print(f"  四位数字：500 张图，在 {len(pool)} 个候选数字里选：准确率 {acc_n:.1%}；选中的数字与真值最多差 1 位的 {close:.1%}")
    print("  局限：检索只能从候选里挑，读不出候选之外的字符串。第 22 章起改成'生成'：让语言模型逐字写出答案")


PARTS = {1: part1_by_hand, 2: part2_train_shapes, 3: part3_temperature, 4: part4_batch, 5: part5_probe, 6: part6_words}

if __name__ == "__main__":
    want = [int(a) for a in sys.argv[1:] if a.isdigit()] or list(PARTS)
    for k in want:
        PARTS[k]()
    if want == list(PARTS):
        print("\n第 20 章的演示全部完成。")
