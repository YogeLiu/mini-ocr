"""第 34 章：案例精读 Xiaomi-OCR-0。用前面写过的零件复现报告里的几个关键做法。

    python -m labs.ch34.xiaomi_tour 1 2 3 4 5   # 不训练：三元组共识、式 10 的抽样权重、锚定与 MTP 的形状和初始损失、
                                                #         Mix-RL 的奖励路由（作者的 CPU 上约 10 秒）
    python -m labs.ch34.xiaomi_tour 6           # 你自己跑：文字锚定（λ_seg）的消融
    python -m labs.ch34.xiaomi_tour 7           # 你自己跑：MTP（λ_MTP）的消融
    python -m labs.ch34.xiaomi_tour 8           # 你自己跑：解析与理解数据的配比（报告表 10 的迷你版）
    python -m labs.ch34.xiaomi_tour 9           # 你自己跑：Mix-RL vs 单任务 RL（DAPO 设置）

第 6–8 部分需要第 22 章阶段 1 的模型 outputs/ch22/vlm_mlp_stage1.pt；第 9 部分需要第 26 章的 outputs/ch26/sft.pt。
第 1 部分读作者实测的三个模型对 60 条真实公式（OleehyO/latex-formulas 第 5 个分片）的读法和标注，随仓库放在 labs/ch34/data/。
"""
import json
import os
import re
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

from miniocr.engine.clean import latex_key
from miniocr.engine.consensus import pairwise_agreement, tier, triplet_consensus
from miniocr.engine.mining import hard_candidates, quota_sample
from miniocr.metrics import ned
from miniocr.xiaomi.anchor import MaskHead, box_to_grid_mask, dice_loss, lm_hidden, ssa_loss
from miniocr.xiaomi.mixrl import kie_reward, mix_reward, sampling_weights
from miniocr.xiaomi.mtp import MTPHead, mtp_loss

torch.set_num_threads(8)
OUT = "outputs/ch34"
os.makedirs(OUT, exist_ok=True)
STEPS = int(os.environ.get("STEPS", 1500))
EXPERTS = ("GLM-OCR", "TexTeller", "Qwen3.5")
ANCHOR_MIX = ("text", "ground", "refer", "caption", "count", "count_color", "left", "exists")
UNDERSTAND = ("caption", "count", "count_color", "left", "exists", "ground", "refer")


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def norm_formula(s):
    """模型输出的公式 → 比较用的写法：剥掉 $$…$$、\\[…\\]、\\(…\\)，去掉 \\left、\\right，
    再用第 33 章的 latex_key（去环境、去间距命令、去空格），最后给单个字符的上下标补花括号（x^2 → x^{2}）。"""
    s = s.strip()
    for a, b in (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$")):
        if s.startswith(a) and s.endswith(b) and len(s) >= len(a) + len(b):
            s = s[len(a):len(s) - len(b)].strip()
    s = re.sub(r"\\(left|right)(?![a-zA-Z])\.?", "", s)
    s = latex_key(s)
    return re.sub(r"([\^_])([A-Za-z0-9])", r"\1{\2}", s)


def formula_sim(a, b):
    """公式的相似度：norm_formula 之后按字符算 1 − NED（报告用 CDM；我们的 CDM 只认第 3 章的语法，真实公式用不了）。"""
    return 1.0 - ned(norm_formula(a), norm_formula(b))


# ---------------------------------------------------------------------- 1
def part1_etc():
    section("1. 三元组共识（式 1–4）：第 33 章三个模型读 60 条真实公式的结果")
    paths = [f"labs/ch34/data/preds_{n}.json" for n in EXPERTS] + ["labs/ch34/data/real_refs.json"]
    if not all(os.path.exists(p) for p in paths):
        print("  缺少 labs/ch34/data/ 里的读法文件，跳过")
        return
    preds = [json.load(open(p))["real"] for p in paths[:3]]
    refs = json.load(open(paths[3]))
    n = len(refs)
    C = np.zeros(3)
    rows = []
    for i in range(n):
        ys = [preds[k][i] for k in range(3)]
        A = pairwise_agreement(ys, formula_sim)
        C += (A.sum(1) - 1) / 2                                     # 式 1：和其他专家的平均一致度
        med, a, _ = triplet_consensus(ys, formula_sim)
        label = ys[med]
        rows.append((a, tier(a), EXPERTS[med], norm_formula(label) == norm_formula(refs[i]), formula_sim(label, refs[i])))
    print("  式 1，每个专家和其他专家的平均一致度 C_i（60 条的平均）：" + "  ".join(f"{e} {c / n:.3f}" for e, c in zip(EXPERTS, C)))
    print("  档        条数   伪标签与数据集标注完全相同（norm_formula 之后）   与标注的平均相似度")
    for t in ("high", "medium", "low"):
        r = [x for x in rows if x[1] == t]
        if r:
            print(f"  {t:<8} {len(r):>5}          {np.mean([x[3] for x in r]):6.1%}                          {np.mean([x[4] for x in r]):.3f}")
    print(f"  中心点来自哪个专家：{dict((e, sum(x[2] == e for x in rows)) for e in EXPERTS)}")
    a = np.array([x[0] for x in rows])
    print(f"  三元组一致度 a(x) 的 10/50/90 分位：{np.percentile(a, [10, 50, 90]).round(3).tolist()}")
    print("  注意：报告用 8 个专家选三元组，这里只有 3 个，三元组就是全部；TexTeller 的训练数据包含这个数据集（第 33 章）")


# ---------------------------------------------------------------------- 2
def part2_mining():
    section("2. 多因素挖掘（式 9–11）：簇内抽样权重和难例候选")
    a = np.array([0.98, 0.95, 0.92, 0.70, 0.40, 0.99, 0.97, 0.60])
    u = np.array([0.95, 0.55, 0.30, 0.40, 0.20, 0.90, 0.35, 0.10])
    lab = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    for tau in (1.0, 0.5):
        p = sampling_weights(a, u, lab, f_a=1.0, f_u=1.0, tau=tau)
        print(f"  τ = {tau}：P(i | 簇) = {p.round(3).tolist()}")
    print(f"  难例候选（a ≥ 0.9 且 u ≤ 0.6，式 11）：{np.nonzero((a >= 0.9) & (u <= 0.6))[0].tolist()}"
          f"；第 33 章 hard_candidates(τ_a=0.9, τ_u=0.6) 得到 {np.nonzero(hard_candidates(a, u, 0.9, 0.6))[0].tolist()}"
          "（它用的是严格的 u < τ_u）")
    rng = np.random.default_rng(0)
    pick = quota_sample(lab, sampling_weights(a, u, lab), 4, rng)
    print(f"  按簇平均分 4 个名额、簇内按权重抽：{pick.tolist()}（大簇 0 和小簇 1 各 2 个）")


# ---------------------------------------------------------------------- 3、6：锚定
def anchor_item(rng, task):
    """一条样本 (图 (1, 3, 64, 64) uint8, ids, roles, 任务, 掩码 (64,) 或 None)。
    掩码：text 是单词的框，ground（物体存在时）是答案的框，refer 是问题里的框；其他任务没有掩码。"""
    from labs.ch26.sft_tour import NV, V
    from miniocr.data.instruct import draw_text_scene, make_task, parse_box
    from miniocr.data.shapes import draw_scene, sample_scene
    from miniocr.train.sft import chat_roles
    from labs.ch23.instruct_tour import convo
    while True:
        objs = sample_scene(rng, 64, 64, 1, 3)
        img = draw_scene(objs, rng, 64, 64, noise=0.03)
        word = box = None
        if task == "text":
            img, word, box = draw_text_scene(img, objs, rng)
            if word is None:
                continue
        break
    q, a = make_task(task, objs, rng, word=word, neg_ground=0.2)
    if task == "ground" and a != "none":
        box = parse_box(a)
    if task == "refer":
        box = parse_box(q[q.index("["):q.index("]") + 1])
    ids, roles = chat_roles(V, convo([(q, a)]), NV)
    return img.transpose(2, 0, 1)[None], ids, roles, task, None if box is None else box_to_grid_mask(box)


def aux_forward(m, items, heads, lam_seg=0.0, lam_mtp=0.0):
    """一批样本的损失：NTP（第 26 章的按 token 平均）+ λ_seg·(Dice + CE) + λ_MTP·L_MTP。返回 (总损失, 各项)。"""
    from labs.ch22.vlm_tour import img_float
    from labs.ch26.sft_tour import to_tensors
    from miniocr.train.sft import embed_packed, sft_loss
    imgs, p = to_tensors([it[:4] for it in items])
    emb = embed_packed(m, img_float(imgs), p["ids"])
    logits, h = lm_hidden(m.lm, emb)
    ntp = sft_loss(logits, p["targets"], p["weights"])
    stats = {"ntp": ntp.item()}
    total = ntp
    if lam_seg > 0 or "mask" in heads:
        first = (p["weights"] > 0).float().argmax(1)                # 第一个计算损失的位置 = 答案开始前的最后一个位置
        q = h[torch.arange(len(items)), first]
        vis = h[p["ids"] == m.img_id].view(len(items), -1, h.shape[-1])
        has = torch.tensor([it[4] is not None for it in items])
        tg = torch.stack([it[4] if it[4] is not None else torch.zeros(vis.shape[1]) for it in items])
        total, st = ssa_loss(ntp, heads["mask"](q, vis), tg, has, 1.0, lam_seg)
        stats.update(st)
    if lam_mtp > 0 or "mtp" in heads:
        lm_ = mtp_loss(heads["mtp"](m.lm, h, emb), p["targets"])
        total = total + lam_mtp * lm_
        stats["mtp"] = lm_.item()
    return total, stats


def part3_anchor_shapes():
    section("3. 文字锚定：掩码目标、掩码头和 MTP 头的大小、随机初始化时的各项损失")
    from labs.ch22.vlm_tour import build
    rng = np.random.default_rng(0)
    img, ids, roles, task, mask = anchor_item(rng, "text")
    print(f"  一条 text 样本：{len(ids)} 个 token，掩码里有 {int(mask.sum())} 个视觉 token（共 64 个）。8×8 网格（# = 1）：")
    for r in range(8):
        print("    " + "".join("#" if mask[r * 8 + c] else "." for c in range(8)))
    m = build("mlp", pretrained=False)
    d = m.lm.cfg.d_model
    heads = {"mask": MaskHead(d), "mtp": MTPHead(m.lm.cfg)}
    nv = sum(p.numel() for p in m.parameters())
    for k, h in heads.items():
        print(f"  {k} 头的参数 {sum(p.numel() for p in h.parameters()):,}（VLM 本身 {nv:,}）")
    torch.manual_seed(0)
    items = [anchor_item(rng, ANCHOR_MIX[i % len(ANCHOR_MIX)]) for i in range(32)]
    with torch.no_grad():
        _, st = aux_forward(m, items, heads, lam_seg=0.5, lam_mtp=0.3)
    print(f"  32 条样本（{sum(it[4] is not None for it in items)} 条有掩码），随机初始化的 VLM：" + "  ".join(f"{k} {v:.3f}" for k, v in st.items()))
    print(f"  对照：词表 {m.lm.cfg.vocab_size} 个 token 时均匀猜的交叉熵 ln V = {np.log(m.lm.cfg.vocab_size):.3f}")
    fg = np.mean([float(it[4].mean()) for it in items if it[4] is not None])
    print(f"  有掩码的样本里前景占视觉 token 的 {fg:.1%}：只用逐 token 的 CE，全猜背景也只有很小的损失，Dice 不会")
    t = torch.zeros(1, 64)
    t[0, :3] = 1
    print(f"  例：3 个前景 token，全部预测为背景（logit −5）：CE {torch.nn.functional.binary_cross_entropy_with_logits(torch.full((1, 64), -5.0), t):.3f}，"
          f"Dice {dice_loss(torch.full((1, 64), -5.0), t):.3f}")


# ---------------------------------------------------------------------- 4
def part4_mtp_configs():
    section("4. MTP：真实模型的配置里有没有 MTP 头")
    import glob
    for name, repo, keys in (("Qwen3.5-0.8B", "Qwen--Qwen3.5-0.8B", ("mtp_num_hidden_layers", "mtp_use_dedicated_embeddings")),
                             ("GLM-OCR", "zai-org--GLM-OCR", ("num_nextn_predict_layers",))):
        f = sorted(glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo}/snapshots/*/config.json")))
        if not f:
            print(f"  {name}：本机没有配置文件，跳过")
            continue
        c = json.load(open(f[0]))
        t = c.get("text_config", c)
        print(f"  {name}：" + "，".join(f"{k} = {t.get(k)}" for k in keys))
    print("  两个都带一层 MTP 头。Xiaomi-OCR-0 在持续预训练时保留了底座的这一个 MTP 头一起训练（式 13）")


# ---------------------------------------------------------------------- 5
def part5_rewards():
    section("5. Mix-RL 的奖励路由：每种任务一个打分函数，格式不对、超长一律 0")
    cases = [("text", "Total: $12.50", "Total: $12.60"),
             ("formula", "\\frac{a}{b}+x^{2}", "\\frac{a}{b}+x^2"),
             ("formula", "\\frac{a}{b}+x_{2}", "\\frac{a}{b}+x^2"),
             ("table", "<table><tr><td>1</td><td>2</td></tr></table>", "<table><tr><td>1</td><td>2</td></tr><tr><td>3</td><td>4</td></tr></table>"),
             ("page", "[title] Report\n\n[text] Sales rose 5%.", [("title", "Report"), ("text", "Sales rose 5%.")]),
             ("page", "[title] Report\n\n[text] Sales rose 5%.\n\n[text] Extra line.", [("title", "Report"), ("text", "Sales rose 5%.")]),
             ("page", "Report\nSales rose 5%.", [("title", "Report"), ("text", "Sales rose 5%.")]),
             ("kie", '{"name": "Zhang San", "date": {"issue": "2026-01-02"}}', {"name": "Zhang San", "date": {"issue": "2026-01-02", "expire": "2036-01-02"}}),
             ("kie", '{"name": "Zhang San", "date": ', {"name": "Zhang San"}),
             ("vqa", "42 km", ["42 km", "42 kilometers"]),
             ("vqa", "about forty", ["42 km"]),
             ("text", "word " * 600, "word")]
    for task, pred, ref in cases:
        print(f"  {task:<8} {mix_reward(task, pred, ref):.3f}   预测 {pred[:46]!r}")
    print("  第二个公式：x^{2} 和 x^2 渲染一样，CDM 给满分；x_{2} 位置不同，扣分")


# ---------------------------------------------------------------------- 6–8：训练指南
def aux_train(m, steps, mix, lam_seg=0.0, lam_mtp=0.0, lr=3e-4, seed=0, bs=32, log_every=250):
    """从第 22 章阶段 1 的模型出发做 SFT，可选地加锚定和 MTP 的辅助损失。mix：任务 → 比例。返回 (用时, 头)。"""
    from labs.ch26.sft_tour import warmup_cosine
    from miniocr.models.vlm import set_trainable
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d = m.lm.cfg.d_model
    heads = {}
    if lam_seg > 0:
        heads["mask"] = MaskHead(d)
    if lam_mtp > 0:
        heads["mtp"] = MTPHead(m.lm.cfg)
    set_trainable(m)
    params = [p for p in m.parameters() if p.requires_grad] + [p for h in heads.values() for p in h.parameters()]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    tasks, probs = list(mix), np.array(list(mix.values()), float)
    probs /= probs.sum()
    m.train()
    t0 = time.time()
    for step in range(steps):
        items = [anchor_item(rng, tasks[int(rng.choice(len(tasks), p=probs))]) for _ in range(bs)]
        for g in opt.param_groups:
            g["lr"] = warmup_cosine(step, steps, 100, lr, lr * 0.1)
        loss, st = aux_forward(m, items, heads, lam_seg, lam_mtp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if log_every and (step + 1) % log_every == 0:
            print(f"      第 {step + 1:4d} 步  " + "  ".join(f"{k} {v:.4f}" for k, v in st.items()) + f"  {time.time() - t0:.0f} 秒", flush=True)
    return time.time() - t0, heads


def stage1():
    from labs.ch23.instruct_tour import load_stage1
    if not os.path.exists("outputs/ch22/vlm_mlp_stage1.pt"):
        sys.exit("  缺少 outputs/ch22/vlm_mlp_stage1.pt：先运行第 22 章第 2 部分（python -m labs.ch22.vlm_tour 2，约 25 分钟），它会保存阶段 1 的模型")
    return load_stage1()


def report(m, label, sec):
    from labs.ch26.sft_tour import quick_eval
    r = quick_eval(m, tasks=("text", "ground", "count", "left", "exists", "caption"))
    print(f"    {label:<28} " + "  ".join(f"{k} {v:6.1%}" for k, v in r.items() if not k.endswith("cer")) + f"  （{sec:.0f} 秒）", flush=True)
    return r


def part6_anchor_ablation():
    section(f"6. 文字锚定的消融（你自己跑）：同样 {STEPS} 步 SFT，λ_seg = 0 / 0.5 / 2")
    mix = {t: 1.0 for t in ANCHOR_MIX}
    for lam in (0.0, 0.5, 2.0):
        m = stage1()
        sec, _ = aux_train(m, STEPS, mix, lam_seg=lam)
        report(m, f"λ_seg = {lam}", sec)


def part7_mtp_ablation():
    section(f"7. MTP 的消融（你自己跑）：同样 {STEPS} 步 SFT，λ_MTP = 0 / 0.3 / 1")
    mix = {t: 1.0 for t in ANCHOR_MIX}
    for lam in (0.0, 0.3, 1.0):
        m = stage1()
        sec, _ = aux_train(m, STEPS, mix, lam_mtp=lam)
        report(m, f"λ_MTP = {lam}", sec)


def part8_ratio():
    section(f"8. 解析与理解数据的配比（你自己跑）：OCR（text）的样本数固定，看加入理解类任务后 OCR 变好还是变差")
    settings = [("只有 text", {"text": 1.0}, 1.0),
                ("text 50% + 理解 50%", {"text": 7.0, **{t: 1.0 for t in UNDERSTAND}}, 2.0),
                ("text 29% + 理解 71%", {"text": 2.0, **{t: 0.7 for t in UNDERSTAND}}, 1 / 0.29),
                ("text 9% + 理解 91%", {"text": 1.0, **{t: 1.44 for t in UNDERSTAND}}, 1 / 0.09)]
    base = max(1, STEPS // 4)
    for label, mix, mult in settings:
        m = stage1()
        steps = int(round(base * mult))                             # text 的样本数在各设置之间相同
        sec, _ = aux_train(m, steps, mix)
        report(m, f"{label}（{steps} 步）", sec)


# ---------------------------------------------------------------------- 9
def part9_mixrl():
    section("9. Mix-RL vs 单任务 RL（你自己跑）：DAPO 设置（β = 0、token 汇总、clip-higher、动态采样），200 步")
    import labs.ch31.grpo_tour as c31
    from labs.ch32.forget_tour import load_sft
    if not os.path.exists("outputs/ch26/sft.pt"):
        sys.exit("  缺少 outputs/ch26/sft.pt：先跑第 26 章第 2 部分（python -m labs.ch26.sft_tour 2），它会自动保存这个模型")
    for label, tasks in (("单任务（只有 text）", ("text",)), ("Mix-RL（六种任务）", ("count", "count_color", "left", "exists", "ground", "text"))):
        c31.RL_TASKS = tasks
        m = load_sft()
        cfg = c31.RLConfig(steps=int(os.environ.get("RL_STEPS", 200)), lr=1e-5, beta=0.0, mode="token", eps_high=0.28,
                           dynamic=True, overlong=True, log_every=50)
        sec, _ = c31.grpo_train(m, cfg)
        report(m, label, sec)


PARTS = {1: part1_etc, 2: part2_mining, 3: part3_anchor_shapes, 4: part4_mtp_configs, 5: part5_rewards,
         6: part6_anchor_ablation, 7: part7_mtp_ablation, 8: part8_ratio, 9: part9_mixrl}

if __name__ == "__main__":
    want = [int(a) for a in sys.argv[1:] if a.isdigit()] or list(PARTS)
    for k in want:
        PARTS[k]()
    if want == list(PARTS):
        print("\n第 34 章的演示全部完成。")
