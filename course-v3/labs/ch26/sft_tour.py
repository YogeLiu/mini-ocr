"""第 26 章：VLM 的监督微调。损失算在哪里、怎样平均、要不要学 <eos>、数据量与步数、打包、分辨率与视觉 token、种子。
从第 22 章阶段 1（对齐）的模型出发，用第 23 章的九种任务。默认只用 CPU。

    python -m labs.ch26.sft_tour 1            # 序列、长度、补齐的浪费、打包的正确性：不训练，几十秒
    python -m labs.ch26.sft_tour 2            # 要不要在提示上算损失（3 种 × 3 个种子）
    python -m labs.ch26.sft_tour 3 4 5 6 7 8  # 其余实验，各自独立，可以分开跑（CPU 上一次 2000 步的训练约 10 分钟，全部约 4–5 小时）
    python -m labs.ch26.sft_tour --gpu        # 有空闲 GPU 时用 GPU

第 2–8 部分需要 outputs/ch22/vlm_mlp_stage1.pt（python -m labs.ch22.vlm_tour 2），
第 8 部分还需要 outputs/ch20/clip_shapes.pt（python -m labs.ch20.clip_tour 2）。
"""
import os
import sys
import time

if "--gpu" not in sys.argv:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

from labs.ch22.vlm_tour import DEV, build, img_float, need
from labs.ch23.instruct_tour import SECOND, ask, convo, example, prompt, test_examples
from miniocr.data.corpus import text_vocab
from miniocr.data.instruct import TASKS
from miniocr.metrics import cer
from miniocr.models.gpt import GPT, GPTConfig
from miniocr.models.vit import ViT, ViTConfig
from miniocr.models.vlm import VLM, MLPProjector, set_trainable
from miniocr.train.optim import warmup_cosine
from miniocr.train.sft import (ROLE_ANSWER, ROLE_FORMAT, ROLE_IMAGE, ROLE_USER, chat_roles, embed_packed, pack,
                               packed_lm_forward, pad_weighted, sft_loss, sft_targets, steps_to)

OUT = "outputs/ch26"
os.makedirs(OUT, exist_ok=True)
V = text_vocab()
EOS, PAD = V.stoi["<eos>"], V.stoi["<pad>"]
NV = 64
BS = 32
STEPS = 2000
STAGE1_STEPS = 1500                          # 第 8 部分：合并连接器的阶段 1（只训练连接器）
MIX = [t for t in TASKS]                    # 九种任务等比例，每个批里混在一起（第 23 章是一个批一种任务）
EVAL_TASKS = ("caption", "count", "left", "exists", "ground", "text")


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ---------------------------------------------------------------------- 数据
def make_example(rng, task=None, multi_turn=0.3, nv=NV, **kw):
    """一条样本：(图像 uint8 (k, 3, 64, 64), ids, roles, 任务名)。"""
    task = task or MIX[int(rng.integers(len(MIX)))]
    two = task not in ("compare", "text") and rng.random() < multi_turn
    img, pairs = example(task, rng, second=SECOND[int(rng.integers(4))] if two else None)
    ids, roles = chat_roles(V, convo(pairs), nv)
    return img, ids, roles, task


class Pool:
    """固定大小的训练集（n=None 表示每次都现场生成新样本，即"无限数据"）。"""

    def __init__(self, n, seed, nv=NV):
        self.rng = np.random.default_rng(seed)
        self.n, self.nv = n, nv
        self.items = [make_example(self.rng, nv=nv) for _ in range(n)] if n else None

    def batch(self, B):
        if self.items is None:
            return [make_example(self.rng, nv=self.nv) for _ in range(B)]
        return [self.items[int(i)] for i in self.rng.integers(0, self.n, B)]


def to_tensors(items, prompt_weight=0.0, train_eos=True, packed=False, max_len=512):
    """一批样本 → 模型输入。返回 (images (N, 3, 64, 64) uint8, dict)。"""
    exs = [(ids, *sft_targets(ids, roles, prompt_weight, eos_id=EOS, train_eos=train_eos)) for _, ids, roles, _ in items]
    if packed:
        p = pack(exs, max_len, PAD)
        order = [k for ks in p["order"] for k in ks]
    else:
        ids, tg, w = pad_weighted(exs, PAD)
        p, order = dict(ids=ids, targets=tg, weights=w), list(range(len(items)))
    imgs = np.concatenate([items[k][0] for k in order])          # compare 的样本贡献两张图，依次排列
    return torch.from_numpy(imgs), p


def forward_loss(m, imgs, p, mode="token"):
    emb = embed_packed(m, img_float(imgs), p["ids"].to(DEV))
    if "seg" in p:
        logits = packed_lm_forward(m.lm, emb, p["seg"].to(DEV), p["pos"].to(DEV))
    else:
        logits = m.lm(inputs_embeds=emb)[0]
    return sft_loss(logits, p["targets"].to(DEV), p["weights"].to(DEV), mode)


# ---------------------------------------------------------------------- 训练与评测
def train(m, steps, pool, lr=3e-4, seed=0, prompt_weight=0.0, train_eos=True, mode="token", packed=False,
          bs=BS, eval_every=0, eval_fn=None, log_every=0):
    """返回 (用时, 学习曲线 [(步数, eval_fn 的结果), …], 一共见过的样本数)。"""
    torch.manual_seed(seed)
    set_trainable(m)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    m.train()
    t0, curve, seen = time.time(), [], 0
    for step in range(steps):
        items = pool.batch(bs)
        seen += len(items)
        imgs, p = to_tensors(items, prompt_weight, train_eos, packed)
        for gr in opt.param_groups:
            gr["lr"] = warmup_cosine(step, steps, 100, lr, lr * 0.1)
        loss = forward_loss(m, imgs, p, mode)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if log_every and (step + 1) % log_every == 0:
            print(f"      第 {step + 1:4d} 步  损失 {loss.item():.4f}  {time.time() - t0:.0f} 秒")
        if eval_every and (step + 1) % eval_every == 0:
            curve.append((step + 1, eval_fn(m)))
            m.train()
    return time.time() - t0, curve, seen


def quick_eval(m, tasks=EVAL_TASKS, n=100):
    """每个任务 n 个测试样本的全对比例（定位用 IoU ≥ 0.5；描述另给 CER）。测试样本与第 23 章相同（固定种子）。"""
    from labs.ch23.instruct_tour import ground_scores
    res = {}
    for t in tasks:
        X, qs, refs = test_examples(t, n=n)
        pred = ask(m, X, [prompt(q) for q in qs], max_new=90 if t == "caption" else 20)
        if t == "ground":
            res[t] = ground_scores(pred, refs)["acc50"]
        else:
            res[t] = float(np.mean([a == b for a, b in zip(pred, refs)]))
        if t == "caption":
            res["caption_cer"] = cer(pred, refs)
    return res


def fmt(r, tasks=EVAL_TASKS):
    s = "  ".join(f"{t} {r[t]:5.1%}" for t in tasks)
    return s + (f"  （描述 CER {r['caption_cer']:.1%}）" if "caption_cer" in r else "")


def load_stage1():
    m = build("mlp")
    m.load_state_dict(torch.load(need("outputs/ch22/vlm_mlp_stage1.pt"), map_location="cpu"))
    return m.to(DEV)


# ---------------------------------------------------------------------- 1
def part1_sequences():
    section("1. SFT 的样本：每个 token 的身份、长度、补齐的浪费、打包（不训练）")
    rng = np.random.default_rng(0)
    img, ids, roles, task = make_example(rng, "left", multi_turn=1.0)
    mark = {ROLE_FORMAT: "f", ROLE_IMAGE: "▒", ROLE_USER: "u", ROLE_ANSWER: "A"}
    text = "".join("▒" if r == ROLE_IMAGE else V.itos[i].replace("\n", "⏎") if i not in (V.stoi["<bos>"], EOS)
                   else {V.stoi["<bos>"]: "B", EOS: "E"}[i] for i, r in zip(ids, roles))
    rs = "".join(mark[r] for r in roles)
    print("  一条两轮对话（B = <bos>，E = <eos>，▒ = 视觉占位符，64 个只画 4 个）：")
    print("    token " + text.replace("▒" * 64, "▒▒▒▒"))
    print("    身份  " + rs.replace("▒" * 64, "▒▒▒▒") + "   （f 格式，u 用户的话，A 回答）")
    counts = {k: roles.count(v) for k, v in (("格式", ROLE_FORMAT), ("视觉", ROLE_IMAGE), ("用户", ROLE_USER), ("回答", ROLE_ANSWER))}
    print(f"    {len(ids)} 个 token：{counts}")

    print("\n  九种任务各 200 条（30% 两轮）：序列长度与回答长度（token）")
    stats = {}
    for _ in range(1800):
        _, ids, roles, t = make_example(rng)
        stats.setdefault(t, []).append((len(ids), roles.count(ROLE_ANSWER)))
    print(f"    {'任务':<12} {'序列长度 平均':>12} {'最短–最长':>10} {'回答 平均':>9}")
    for t in TASKS:
        a = np.array(stats[t])
        print(f"    {t:<12} {a[:, 0].mean():12.1f} {a[:, 0].min():5d}–{a[:, 0].max():<4d} {a[:, 1].mean():9.1f}")
    allv = np.concatenate([np.array(v) for v in stats.values()])
    ans = {t: np.array(stats[t])[:, 1].mean() for t in TASKS}
    tot = sum(ans.values())
    print("  按 token 平均损失时，每种任务占梯度的份额（各任务样本数相同，份额 ∝ 回答的 token 数）：")
    print("    " + "  ".join(f"{t} {ans[t] / tot:.1%}" for t in TASKS))
    print(f"    按样本平均时每种任务都是 {1 / len(TASKS):.1%}")

    print("\n  补齐 vs 打包（随机 50 批，每批 32 条，max_len 512）：")
    pad_frac, pk_frac, rows = [], [], []
    pool = Pool(None, 1)
    for _ in range(50):
        items = pool.batch(32)
        _, p = to_tensors(items)
        pad_frac.append((p["ids"] == PAD).float().mean().item())
        _, q = to_tensors(items, packed=True)
        pk_frac.append((q["seg"] == 0).float().mean().item())
        rows.append((p["ids"].shape, q["ids"].shape))
    print(f"    补齐：每批 32 行 × 最长的那条（平均 {np.mean([r[0][1] for r in rows]):.0f} 个位置），<pad> 占 {np.mean(pad_frac):.1%}")
    print(f"    打包：每批平均 {np.mean([r[1][0] for r in rows]):.1f} 行 × 512，<pad> 占 {np.mean(pk_frac):.1%}；"
          f"总位置数是补齐的 {np.mean([r[1][0] * 512 / (r[0][0] * r[0][1]) for r in rows]):.2f} 倍")
    print(f"    全部样本的序列长度：平均 {allv[:, 0].mean():.0f}、最长 {allv[:, 0].max()}")

    print("\n  打包的正确性：一个随机初始化的小 VLM（不训练），同一批样本打包 vs 不打包，每个回答 token 的 logits：")
    torch.manual_seed(0)
    vit = ViT(ViTConfig(img_size=64, patch=8, d_model=64, n_layer=1, n_head=4, pool="mean"))
    lm = GPT(GPTConfig(vocab_size=len(V), block_size=512, n_layer=2, n_head=4, d_model=64))
    m = VLM(vit, MLPProjector(64, 64), lm, V.stoi["<img>"]).to(DEV).eval()
    items = pool.batch(8)
    with torch.no_grad():
        imgs, p = to_tensors(items)
        a = m.lm(inputs_embeds=embed_packed(m, img_float(imgs), p["ids"].to(DEV)))[0]
        la = forward_loss(m, imgs, p).item()
        imgs2, q = to_tensors(items, packed=True)
        b = packed_lm_forward(m.lm, embed_packed(m, img_float(imgs2), q["ids"].to(DEV)), q["seg"].to(DEV), q["pos"].to(DEV))
        lb = forward_loss(m, imgs2, q).item()
    va = a[p["targets"].to(DEV) != -100]
    vb = b[q["targets"].to(DEV) != -100]
    print(f"    8 条样本 → {q['ids'].shape[0]} 行；回答 token {va.shape[0]} 个 vs {vb.shape[0]} 个；logits 最大差 {(va - vb).abs().max().item():.1e}；"
          f"损失 {la:.6f} vs {lb:.6f}")
    print("  去掉打包的两个要素之一，与不打包的结果比：")
    e = embed_packed(m, img_float(imgs2), q["ids"].to(DEV))
    full = torch.arange(q["ids"].shape[1], device=DEV).expand_as(q["pos"])
    one = (q["seg"] > 0).long().to(DEV)                          # 整行当成一条样本：只剩因果 mask
    for name, seg, pos in (("位置不重新编号（整行 0, 1, 2, …）", q["seg"].to(DEV), full),
                           ("不用分块 mask（只有因果 mask）", one, q["pos"].to(DEV)), ("两个都去掉", one, full)):
        with torch.no_grad():
            vc = packed_lm_forward(m.lm, e, seg, pos)[q["targets"].to(DEV) != -100]
        print(f"    {name:<26} logits 最大差 {(va - vc).abs().max().item():.1e}")


# ---------------------------------------------------------------------- 2
def part2_prompt_loss():
    section("2. 要不要在提示上算损失：提示权重 0 / 0.1 / 1，各 3 个种子")
    print(f"  每次从阶段 1 出发训练 {STEPS} 步（批 {BS}，九种任务混合），每 500 步评测（每个任务 100 个样本）")
    res = {}
    for w in (0.0, 0.1, 1.0):
        for seed in (0, 1, 2):
            m = load_stage1()
            sec, curve, _ = train(m, STEPS, Pool(None, 10 + seed), seed=seed, prompt_weight=w, eval_every=max(1, STEPS // 4), eval_fn=quick_eval)
            res[(w, seed)] = curve
            if w == 0.0 and seed == 0:                            # 第 27、30–32、34、36 章的起点
                torch.save(m.state_dict(), f"{OUT}/sft.pt")
            print(f"    提示权重 {w:<4} 种子 {seed}（{sec:.0f} 秒）：" + " | ".join(f"{s} 步 平均 {np.mean([r[t] for t in EVAL_TASKS]):.1%}"
                                                                    for s, r in curve))
            print(f"      最后：{fmt(curve[-1][1])}")
    print("  3 个种子的平均（最后一次评测）：")
    for w in (0.0, 0.1, 1.0):
        last = [res[(w, s)][-1][1] for s in (0, 1, 2)]
        print(f"    提示权重 {w:<4} " + "  ".join(f"{t} {np.mean([r[t] for r in last]):5.1%}±{np.std([r[t] for r in last]):.1%}"
                                                   for t in EVAL_TASKS))


# ---------------------------------------------------------------------- 3
def part3_data_vs_steps():
    section("3. 数据量与步数：同样 2000 步，训练集分别有 300 / 3,000 / 30,000 条 / 无限")
    for n in (300, 3000, 30000, None):
        m = load_stage1()
        pool = Pool(n, 20)
        sec, _, seen = train(m, STEPS, pool)
        test = quick_eval(m)
        if pool.items is not None:                                # 在训练集本身上评测（模型见过这些图和问答）
            sub = pool.items[:200]
            X = torch.from_numpy(np.stack([it[0][0] for it in sub if it[0].shape[0] == 1]))
            qa = [it for it in sub if it[0].shape[0] == 1]
            prompts, refs = [], []
            for it in qa:
                ids, roles = it[1], it[2]
                first = next(i for i, r in enumerate(roles) if r == ROLE_ANSWER)
                end = first + roles[first:].index(ROLE_FORMAT) if ROLE_FORMAT in roles[first:] else len(ids)
                prompts.append(ids[:first])
                refs.append(V.decode(ids[first:end - 1]) if ids[end - 1] == EOS else V.decode(ids[first:end]))
            pred = ask(m, X, prompts, max_new=90)
            tr = float(np.mean([a == b for a, b in zip(pred, refs)]))
            trs = f"训练集上（前 200 条中的单图样本，第一轮）全对 {tr:.1%}"
        else:
            trs = "（无限数据：没有'训练集'）"
        epochs = f"每条约见过 {seen / n:.0f} 遍" if n else "每条只见一次"
        print(f"    {('无限' if n is None else f'{n:,} 条'):<10}（{epochs}，{sec:.0f} 秒）：{trs}")
        print(f"      测试：{fmt(test)}")


# ---------------------------------------------------------------------- 4
def part4_eos():
    section("4. <eos>：不学'什么时候停'会怎样")
    m = load_stage1()
    train(m, STEPS, Pool(None, 30), train_eos=False)
    for t in ("count", "left", "caption"):
        X, qs, refs = test_examples(t, n=100)
        pred = ask(m, X, [prompt(q) for q in qs], max_new=40 if t != "caption" else 90)
        exact = np.mean([a == b for a, b in zip(pred, refs)])
        prefix = np.mean([a.startswith(b) for a, b in zip(pred, refs)])
        print(f"    {t:<8} 全对 {exact:6.1%}  以正确答案开头 {prefix:6.1%}  平均输出 {np.mean([len(a) for a in pred]):.0f} 个字符"
              f"（真值平均 {np.mean([len(b) for b in refs]):.0f}）  例：{pred[0]!r}")


# ---------------------------------------------------------------------- 5
def part5_reduction():
    section("5. 按 token 平均 vs 按样本平均")
    for mode in ("token", "sample"):
        m = load_stage1()
        sec, _, _ = train(m, STEPS, Pool(None, 40), mode=mode)
        print(f"    {mode:<7}（{sec:.0f} 秒）：{fmt(quick_eval(m))}")


# ---------------------------------------------------------------------- 6
def part6_packing():
    section("6. 打包：同样的样本数，补齐 vs 打包的速度与结果")
    steps_pad = STEPS
    m = load_stage1()
    sec, _, seen = train(m, steps_pad, Pool(None, 50))
    print(f"    补齐：{steps_pad} 步 × 批 {BS} = {seen:,} 条，{sec:.0f} 秒（{sec / steps_pad * 1000:.0f} 毫秒/步）：{fmt(quick_eval(m))}")
    m = load_stage1()
    sec, _, seen = train(m, steps_pad, Pool(None, 50), packed=True)
    print(f"    打包：{steps_pad} 步 × 批 {BS} = {seen:,} 条，{sec:.0f} 秒（{sec / steps_pad * 1000:.0f} 毫秒/步）：{fmt(quick_eval(m))}")
    print("    两者见过的样本、步数、学习率完全相同；在 CPU 上打包未必更快（mask 让 PyTorch 用不了最快的注意力实现），GPU 上通常更快")


# ---------------------------------------------------------------------- 7
def part7_seeds():
    section("7. 种子：同一个配置跑 5 次，学习曲线的形状")
    task = ("count_color",)
    print(f"  只训练 count_color（按颜色数数）一种任务，{STEPS} 步，每 100 步评测 100 个样本")
    fn = lambda m: quick_eval(m, task, n=100)["count_color"]
    curves = {}
    for seed in range(5):
        m = load_stage1()
        pool = Pool(None, 60 + seed)
        pool.batch = (lambda p: (lambda B: [make_example(p.rng, "count_color") for _ in range(B)]))(pool)
        sec, curve, _ = train(m, STEPS, pool, seed=seed, eval_every=max(1, STEPS // 20), eval_fn=fn)
        curves[seed] = curve
        print(f"    种子 {seed}（{sec:.0f} 秒）：" + " ".join(f"{v:.0%}" for _, v in curve))
        print(f"      第一次 ≥ 50% 的步数 {steps_to(curve, 0.5)}，≥ 80% 的步数 {steps_to(curve, 0.8)}")
    steps = [s for s, _ in curves[0]]
    mean = [np.mean([curves[k][i][1] for k in curves]) for i in range(len(steps))]
    print("    5 个种子的平均曲线：" + " ".join(f"{v:.0%}" for v in mean))
    print("    注意：平均曲线是平滑的，但单条曲线可能是'平台 + 突然上升'；平均会把不同时间的跳变抹成一条斜坡")


# ---------------------------------------------------------------------- 8
def part8_resolution():
    section("8. 分辨率与视觉 token 数：两个不同的旋钮")
    from labs.ch22.vlm_tour import build as build22
    print("  A：原图 64×64，64 个视觉 token（MLP 连接器，第 2 部分的设置）")
    print("  B：先缩到 32×32 再放大回 64×64（细节丢了，token 数不变），64 个视觉 token")
    print("  C：原图 64×64，2×2 合并连接器，16 个视觉 token（像素不丢，token 少了 3/4）")

    def blur(X):
        x = torch.nn.functional.interpolate(X.float(), size=(32, 32), mode="area")
        return torch.nn.functional.interpolate(x, size=(64, 64), mode="nearest").round().clamp(0, 255).to(torch.uint8)

    m = load_stage1()
    train(m, STEPS, Pool(None, 70))
    print(f"    A：{fmt(quick_eval(m))}")

    m = load_stage1()
    pool = Pool(None, 70)
    orig = pool.batch
    pool.batch = lambda B: [(blur(torch.from_numpy(it[0])).numpy(), *it[1:]) for it in orig(B)]
    train(m, STEPS, pool)
    global test_examples
    saved = test_examples
    test_examples = lambda task, n=200, seed=None, **kw: (lambda X, q, r: (blur(X) if X.dim() == 4 else
                                                                         blur(X.flatten(0, 1)).view(X.shape), q, r))(*saved(task, n, seed, **kw))
    try:
        print(f"    B：{fmt(quick_eval(m))}（测试图也做同样的缩放）")
    finally:
        test_examples = saved

    m = build22("merge").to(DEV)
    set_trainable(m, vision=False, connector=True, lm=False)
    pool = Pool(None, 71, nv=16)
    pool.batch = (lambda p: (lambda B: [make_example(p.rng, "caption", multi_turn=0.0, nv=16) for _ in range(B)]))(pool)
    _train_frozen(m, STAGE1_STEPS, pool)
    import labs.ch23.instruct_tour as c23
    c23.NV, saved_nv = 16, c23.NV
    try:
        train(m, STEPS, Pool(None, 70, nv=16))
        print(f"    C：{fmt(quick_eval(m))}（阶段 1 只训练合并连接器 {STAGE1_STEPS} 步，再全部解冻 {STEPS} 步）")
    finally:
        c23.NV = saved_nv


def _train_frozen(m, steps, pool, lr=1e-3):
    """阶段 1：只训练连接器（与第 22 章相同）。"""
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    m.train()
    for step in range(steps):
        imgs, p = to_tensors(pool.batch(BS))
        for gr in opt.param_groups:
            gr["lr"] = warmup_cosine(step, steps, 100, lr, lr * 0.1)
        loss = forward_loss(m, imgs, p)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


PARTS = {1: part1_sequences, 2: part2_prompt_loss, 3: part3_data_vs_steps, 4: part4_eos, 5: part5_reduction,
         6: part6_packing, 7: part7_seeds, 8: part8_resolution}

if __name__ == "__main__":
    want = [int(a) for a in sys.argv[1:] if a.isdigit()] or list(PARTS)
    for k in want:
        PARTS[k]()
    if want == list(PARTS):
        print("\n第 26 章的演示全部完成。")
