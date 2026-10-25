"""第 23 章：VLM 的任务与指令数据。从第 22 章阶段 1（对齐）的模型出发，用九种任务的混合数据做指令微调；
测量定位、多图、多轮、场景文字；比较数据配比；测量幻觉。默认只用 CPU（GPU 被别的程序占用时也能跑）。

    python -m labs.ch23.instruct_tour        # 全部 5 个部分（CPU 上约 80 分钟；需要 outputs/ch22/vlm_mlp_stage1.pt）
    python -m labs.ch23.instruct_tour 2 4    # 第 2 部分训练并保存主模型，第 4 部分读取它
    python -m labs.ch23.instruct_tour --gpu  # 有空闲 GPU 时用 GPU
"""
import os
import sys
import time

if "--gpu" not in sys.argv:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

from labs.ch22.vlm_tour import DEV, build, img_float, need
from miniocr.boxes import box_iou
from miniocr.data.corpus import text_vocab
from miniocr.data.instruct import TASKS, apply_chat_template, box_text, draw_text_scene, make_task, name, parse_box
from miniocr.data.shapes import COLORS, SHAPES, draw_scene, sample_scene
from miniocr.metrics import cer
from miniocr.models.vlm import pad_batch, set_trainable
from miniocr.train.optim import warmup_cosine

OUT = "outputs/ch23"
os.makedirs(OUT, exist_ok=True)
V = text_vocab()
EOS, PAD = V.stoi["<eos>"], V.stoi["<pad>"]
NV = 64                                               # MLP 连接器：每张图 64 个视觉 token
STEPS, BS = 4000, 32
UNIFORM = {t: 1.0 for t in TASKS}


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ---------------------------------------------------------------------- 数据
def scene(rng, lo=1, hi=3, text=False):
    objs = sample_scene(rng, 64, 64, lo, hi)
    img = draw_scene(objs, rng, 64, 64, noise=0.03)
    word = None
    if text:
        img, word, _ = draw_text_scene(img, objs, rng)
    return img.transpose(2, 0, 1), objs, word


def example(task, rng, lo=1, hi=3, neg_ground=0.2, second=None):
    """返回 (图像 uint8：(1, 3, 64, 64)，compare 是 (2, 3, 64, 64)；问答列表 [(问题, 答案), …])。
    second 不为 None 时（单图任务），在同一张图上再问一轮 second 任务（第二轮的问题里不再放图）。"""
    if task == "compare":
        a, oa, _ = scene(rng, lo, hi)
        b, ob, _ = scene(rng, lo, hi)
        return np.stack([a, b]), [make_task("compare", oa, rng, objs2=ob)]
    while True:
        img, objs, word = scene(rng, lo, hi, text=(task == "text"))
        if task != "text" or word is not None:
            break
    pairs = [make_task(task, objs, rng, word=word, neg_ground=neg_ground)]
    if second is not None:
        q, a = make_task(second, objs, rng, neg_ground=neg_ground)
        pairs.append((q.replace("<image>", ""), a))
    return img[None], pairs


def convo(pairs):
    msgs = []
    for q, a in pairs:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    return msgs


SECOND = ("count", "left", "exists", "count_color")


def sample_batch(rng, mix, neg_ground=0.2, multi_turn=0.3):
    """整个批用同一个任务（按 mix 的权重抽）；除 compare、text 外，每个样本有 multi_turn 的概率变成同一张图上的两轮对话。"""
    names = list(mix)
    p = np.array([mix[t] for t in names], dtype=float)
    task = names[int(rng.choice(len(names), p=p / p.sum()))]
    imgs, exs = [], []
    for _ in range(BS):
        two = task not in ("compare", "text") and rng.random() < multi_turn
        img, pairs = example(task, rng, neg_ground=neg_ground, second=SECOND[int(rng.integers(4))] if two else None)
        imgs.append(img)
        exs.append(apply_chat_template(V, convo(pairs), NV))
    ids, tg = pad_batch(exs, PAD)
    return torch.from_numpy(np.stack(imgs)), ids, tg


def train(m, steps, mix, lr=3e-4, neg_ground=0.2, seed=0, log_every=0):
    set_trainable(m)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    m.train()
    t0 = time.time()
    for step in range(steps):
        X, ids, tg = sample_batch(rng, mix, neg_ground)
        X = X[:, 0] if X.shape[1] == 1 else X
        for gr in opt.param_groups:
            gr["lr"] = warmup_cosine(step, steps, 100, lr, lr * 0.1)
        loss = m(img_float5(X), ids.to(DEV), tg.to(DEV))[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if log_every and (step + 1) % log_every == 0:
            print(f"      第 {step + 1:4d} 步  损失 {loss.item():.4f}  {time.time() - t0:.0f} 秒")
    return time.time() - t0


def img_float5(X):
    return img_float(X) if X.dim() == 4 else img_float(X.flatten(0, 1)).reshape(X.shape)


def load_stage1():
    m = build("mlp")
    m.load_state_dict(torch.load(need("outputs/ch22/vlm_mlp_stage1.pt")))
    return m


# ---------------------------------------------------------------------- 评测
def prompt(question, history=()):
    ids, _ = apply_chat_template(V, convo(history) + [{"role": "user", "content": question}], NV)
    return ids + V.encode("Assistant: ")


@torch.no_grad()
def ask(m, X, prompts, max_new=12, bs=100):
    """X：(N, 3, 64, 64) 或 (N, 2, 3, 64, 64) 的 uint8；prompts：编号列表。按提示长度分组生成。"""
    out = [None] * len(prompts)
    groups = {}
    for i, p in enumerate(prompts):
        groups.setdefault(len(p), []).append(i)
    for idxs in groups.values():
        for k in range(0, len(idxs), bs):
            sub = idxs[k:k + bs]
            gen = m.generate(img_float5(X[sub]), torch.tensor([prompts[i] for i in sub], device=DEV), max_new=max_new, eos=EOS)
            for i, g in zip(sub, gen.tolist()):
                out[i] = V.decode(g)
    return out


def test_examples(task, n=200, seed=None, **kw):
    rng = np.random.default_rng(1000 + TASKS.index(task) if seed is None else seed)
    ex = [example(task, rng, **kw) for _ in range(n)]
    X = torch.from_numpy(np.stack([e[0] for e in ex]))
    return (X[:, 0] if X.shape[1] == 1 else X), [e[1][0][0] for e in ex], [e[1][0][1] for e in ex]


def ground_scores(pred, refs):
    pos = [(p, r) for p, r in zip(pred, refs) if r != "none"]
    neg = [(p, r) for p, r in zip(pred, refs) if r == "none"]
    ious, bad = [], 0
    for p, r in pos:
        b = parse_box(p)
        if b is None:
            bad += 1
            ious.append(0.0)
        else:
            ious.append(box_iou(torch.tensor([b], dtype=torch.float), torch.tensor([parse_box(r)], dtype=torch.float))[0, 0].item())
    ious = np.array(ious)
    return dict(acc50=float(np.mean(ious >= 0.5)), miou=float(ious.mean()), bad=bad / max(1, len(pos)),
                none_ok=float(np.mean([p == "none" for p, _ in neg])) if neg else float("nan"), n_neg=len(neg))


def evaluate(m, label, tasks=TASKS):
    res = {}
    for task in tasks:
        X, qs, refs = test_examples(task)
        pred = ask(m, X, [prompt(q) for q in qs], max_new=90 if task == "caption" else 20)
        res[task] = dict(acc=float(np.mean([p == r for p, r in zip(pred, refs)])), pred=pred, refs=refs)
        if task == "caption":
            res[task]["cer"] = cer(pred, refs)
        if task == "ground":
            res[task].update(ground_scores(pred, refs))
    line = []
    for t in tasks:
        r = res[t]
        if t == "caption":
            line.append(f"描述 {r['acc']:.0%}/CER {r['cer']:.1%}")
        elif t == "ground":
            line.append(f"定位 IoU≥0.5 {r['acc50']:.0%}（拒答 {r['none_ok']:.0%}）")
        else:
            line.append(f"{t} {r['acc']:.0%}")
    print(f"    {label}：" + "  ".join(line))
    return res


# ---------------------------------------------------------------------- 1
def part1_template():
    section("1. 聊天模板：多轮对话、多张图，只在助手的回答上计算损失")
    rng = np.random.default_rng(0)
    img, objs, _ = scene(rng)
    pairs = [make_task("left", objs, rng), make_task("count", objs, rng)]
    pairs[1] = (pairs[1][0].replace("<image>", ""), pairs[1][1])
    ids, tg = apply_chat_template(V, convo(pairs), NV)
    text = "".join("▒" if i == V.stoi["<img>"] else {V.stoi["<bos>"]: "⟨bos⟩", EOS: "⟨eos⟩"}.get(i, V.itos[i]) for i in ids)
    print("  一段两轮对话（▒ = 一个视觉 token 的占位符，共 64 个，这里只画出前 8 个）：")
    shown = text.replace("▒" * 64, "▒" * 8 + "…")
    for line in shown.split("\n"):
        print("    " + line)
    sup = "".join(V.itos[ids[t + 1]] if tg[t] != -100 else "·" for t in range(len(ids) - 1))
    sup = sup.replace("<eos>", "⟨eos⟩")
    print(f"  计算损失的位置（· = 不计算）：{len(ids)} 个位置里只有 {sum(t != -100 for t in tg)} 个")
    print("    " + sup.replace("·" * 64, "·" * 8 + "…").replace("\n", "\\n"))
    print("  九种任务的例子：")
    for t in TASKS:
        img, [(q, a)] = example(t, rng)
        print(f"    {t:<12} 图 {img.shape[0]} 张  问：{q.replace('<image>', '[图]'):<58} 答：{a}")


# ---------------------------------------------------------------------- 2
def part2_train():
    section("2. 九种任务混合的指令微调")
    path = "outputs/ch22/vlm_mlp_stage1.pt"
    if not os.path.exists(path):
        print(f"  （缺少 {path}，先运行 python -m labs.ch22.vlm_tour 2）")
        return
    m = load_stage1()
    print(f"  从第 22 章阶段 1（只训练过连接器）的模型出发；九种任务等比例，30% 的样本是两轮对话；{STEPS} 步 × 批 {BS}，全部解冻")
    sec = train(m, STEPS, UNIFORM, log_every=1000)
    torch.save(m.state_dict(), f"{OUT}/vlm_instruct.pt")
    print(f"  用时 {sec:.0f} 秒；每个任务 200 个测试样本：")
    r = evaluate(m, "混合训练")
    g = r["ground"]
    print(f"  定位：IoU ≥ 0.5 的比例 {g['acc50']:.1%}，平均 IoU {g['miou']:.3f}，格式错误 {g['bad']:.1%}；"
          f"问不存在的物体时答 'none' 的比例 {g['none_ok']:.1%}（{g['n_neg']} 题）")
    errs = [(q, p, a) for q, p, a in zip(test_examples("ground")[1], g["pred"], g["refs"]) if a != "none" and p != a][:5]
    for q, p, a in errs:
        print(f"    {q.replace('<image>', ''):<40} 真值 {a:<16} 生成 {p}")
    for t in ("text", "compare", "refer"):
        wrong = [(p, a) for p, a in zip(r[t]["pred"], r[t]["refs"]) if p != a][:4]
        print(f"  {t} 的错误例子（真值 → 生成）：{[f'{a} → {p}' for p, a in wrong]}")
    # 多轮：第二轮问题在有第一轮历史时 vs 单独问
    rng = np.random.default_rng(77)
    Xs, single, multi, refs = [], [], [], []
    for _ in range(200):
        img, objs, _ = scene(rng)
        q1, a1 = make_task("left", objs, rng)
        q2, a2 = make_task("count", objs, rng)
        Xs.append(img)
        single.append(prompt(q2))
        multi.append(prompt(q2.replace("<image>", ""), history=[(q1, a1)]))
        refs.append(a2)
    X = torch.from_numpy(np.stack(Xs))
    s = np.mean([p == r for p, r in zip(ask(m, X, single), refs)])
    mt = np.mean([p == r for p, r in zip(ask(m, X, multi), refs)])
    print(f"  多轮：'How many objects?' 单独问 {s:.1%}；作为第二轮（第一轮问了最左颜色，图只在第一轮出现）{mt:.1%}")
    # 换一种问法：训练数据里每个任务只有一种问法
    X, objs_list = [], []
    for _ in range(200):
        img, objs, _ = scene(rng)
        X.append(img)
        objs_list.append(objs)
    X = torch.from_numpy(np.stack(X))
    print("  换一种问法（训练时每个任务只有一种固定的问法）：")
    for q, f in (("How many objects?", lambda o: str(len(o))), ("How many shapes are there?", lambda o: str(len(o))),
                 ("Count the objects.", lambda o: str(len(o))),
                 ("What color is the leftmost object?", lambda o: min(o, key=lambda z: z["box"][0])["color"]),
                 ("Which color is the object on the far left?", lambda o: min(o, key=lambda z: z["box"][0])["color"])):
        pred = ask(m, X, [prompt("<image>" + q)] * len(X))
        refs = [f(o) for o in objs_list]
        print(f"    {q:<44} 准确率 {np.mean([p == r for p, r in zip(pred, refs)]):6.1%}   例：{pred[:3]}")


# ---------------------------------------------------------------------- 3
def part3_mixture():
    section("3. 数据配比：去掉一类数据会怎样")
    path = "outputs/ch22/vlm_mlp_stage1.pt"
    if not os.path.exists(path):
        print(f"  （缺少 {path}）")
        return
    m = load_stage1()
    train(m, STEPS, UNIFORM, neg_ground=0.0)
    r = evaluate(m, "定位数据里没有'不存在'的问题", tasks=("ground", "exists"))
    g = r["ground"]
    X, qs, refs = test_examples("ground")
    neg_pred = [p for p, a in zip(g["pred"], refs) if a == "none"]
    print(f"    问不存在的物体（{len(neg_pred)} 题）：答 'none' {np.mean([p == 'none' for p in neg_pred]):.1%}，"
          f"编出一个框 {np.mean([parse_box(p) is not None for p in neg_pred]):.1%}；例：{neg_pred[:4]}")
    m = load_stage1()
    train(m, STEPS, {"ground": 1.0})
    evaluate(m, "只训练定位一种任务", tasks=("ground", "caption", "left"))
    m = load_stage1()
    train(m, STEPS, {t: (0.0 if t == "caption" else 1.0) for t in TASKS})
    evaluate(m, "去掉描述数据", tasks=TASKS)


# ---------------------------------------------------------------------- 4
def mentions(caption):
    """'a large red circle, a small blue square' → [('large', 'red', 'circle'), …]；格式不对的部分跳过。"""
    out = []
    for part in caption.split(","):
        w = part.strip().split()
        if len(w) == 4 and w[0] == "a":
            out.append(tuple(w[1:]))
    return out


def part4_hallucination():
    section("4. 幻觉：说出图里没有的东西")
    path = f"{OUT}/vlm_instruct.pt"
    if not os.path.exists(path):
        print(f"  （缺少 {path}，先运行第 2 部分）")
        return
    m = load_stage1()
    m.load_state_dict(torch.load(path))
    print("  描述里提到的物体有多少是图里没有的（CHAIR 指标的玩具版：按 颜色 + 形状 判断）")
    for lo, hi, label in ((1, 3, "1–3 个物体（与训练相同）"), (4, 5, "4–5 个物体（训练时没见过）")):
        rng = np.random.default_rng(500 + lo)
        Xs, objs_list = [], []
        for _ in range(300):
            img, objs, _ = scene(rng, lo, hi)
            Xs.append(img)
            objs_list.append(objs)
        X = torch.from_numpy(np.stack(Xs))
        caps = ask(m, X, [prompt("<image>Describe the image.")] * len(X), max_new=150)
        tot = hal = sent = miss = n_obj = 0
        for c, objs in zip(caps, objs_list):
            have = [(o["color"], o["shape"]) for o in objs]
            ms = mentions(c)
            h = sum((col, sh) not in have for _, col, sh in ms)
            tot += len(ms)
            hal += h
            sent += h > 0
            miss += max(0, len(objs) - len(ms))
            n_obj += len(objs)
        cnt = ask(m, X, [prompt("<image>How many objects?")] * len(X))
        acc = np.mean([p == str(len(o)) for p, o in zip(cnt, objs_list)])
        dist = {k: cnt.count(k) for k in sorted(set(cnt))}
        print(f"    {label}：提到的物体中幻觉 {hal / max(1, tot):.1%}，含幻觉的描述 {sent / len(caps):.1%}，"
              f"漏说的物体 {miss / n_obj:.1%}；数数准确率 {acc:.1%}，答案分布 {dist}")
    print("  '有没有'问题对三种不存在的物体答 yes 的比例（POPE 的玩具版）：")
    rng = np.random.default_rng(9)
    kinds = {"随机": [], "同形状、换颜色": [], "同颜色、换形状": []}
    Xs = []
    pos = []
    for _ in range(300):
        img, objs, _ = scene(rng)
        present = {(o["color"], o["shape"]) for o in objs}
        o = objs[int(rng.integers(len(objs)))]
        cands = {
            "随机": [(c, s) for c in COLORS for s in SHAPES if (c, s) not in present],
            "同形状、换颜色": [(c, o["shape"]) for c in COLORS if (c, o["shape"]) not in present],
            "同颜色、换形状": [(o["color"], s) for s in SHAPES if (o["color"], s) not in present],
        }
        Xs.append(img)
        pos.append(prompt(f"<image>Is there a {o['color']} {o['shape']}?"))
        for k, c in cands.items():
            pick = c[int(rng.integers(len(c)))] if c else None
            kinds[k].append(None if pick is None else prompt(f"<image>Is there a {pick[0]} {pick[1]}?"))
    X = torch.from_numpy(np.stack(Xs))
    print(f"    存在的物体（应答 yes）：yes 的比例 {np.mean([p == 'yes' for p in ask(m, X, pos)]):.1%}")
    for k, ps in kinds.items():
        idx = [i for i, p in enumerate(ps) if p is not None]
        ans = ask(m, X[idx], [ps[i] for i in idx])
        print(f"    不存在 —— {k:<12}：yes 的比例 {np.mean([a == 'yes' for a in ans]):.1%}（{len(idx)} 题）")


# ---------------------------------------------------------------------- 5
def part5_resolution_of_coords():
    section("5. 坐标的精度：定位误差有多大")
    path = f"{OUT}/vlm_instruct.pt"
    if not os.path.exists(path):
        print(f"  （缺少 {path}，先运行第 2 部分）")
        return
    m = load_stage1()
    m.load_state_dict(torch.load(path))
    X, qs, refs = test_examples("ground", n=400, seed=4242, neg_ground=0.0)
    pred = ask(m, X, [prompt(q) for q in qs])
    errs, wrong_obj = [], 0
    for p, r, x in zip(pred, refs, X):
        b, t = parse_box(p), parse_box(r)
        if b is None:
            continue
        iou = box_iou(torch.tensor([b], dtype=torch.float), torch.tensor([t], dtype=torch.float))[0, 0].item()
        if iou < 0.1:
            wrong_obj += 1
        else:
            errs.append(np.abs(np.array(b) - np.array(t)))
    e = np.array(errs)
    if len(e) == 0:
        print(f"  400 个存在的物体里没有一个框到了正确的物体附近（IoU < 0.1 的 {wrong_obj} 个，其余格式错误）：先检查第 2 部分的训练")
        return
    print(f"  400 个存在的物体：框到了别的地方（IoU < 0.1）{wrong_obj / len(pred):.1%}；其余 {len(e)} 个的坐标平均误差 "
          f"{e.mean():.2f} 像素（x0 {e[:, 0].mean():.2f}、y0 {e[:, 1].mean():.2f}、x1 {e[:, 2].mean():.2f}、y1 {e[:, 3].mean():.2f}），"
          f"误差 0 的坐标占 {np.mean(e == 0):.1%}，≤ 1 像素 {np.mean(e <= 1):.1%}")
    print("  坐标是逐个字符写出来的：'[44,13,60,29]' 一共 13 个字符，每个数字都要写对")


PARTS = {1: part1_template, 2: part2_train, 3: part3_mixture, 4: part4_hallucination, 5: part5_resolution_of_coords}

if __name__ == "__main__":
    want = [int(a) for a in sys.argv[1:] if a.isdigit()] or list(PARTS)
    for k in want:
        PARTS[k]()
    if want == list(PARTS):
        print("\n第 23 章的演示全部完成。")
