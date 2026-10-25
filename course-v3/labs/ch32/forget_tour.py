"""第 32 章：遗忘与能力配比。帕累托前沿、各方法的计算预算、不需要标签的遗忘指标、重放与配比、同一起点比较四种方法。

    python -m labs.ch32.forget_tour 1 2 3   # 不训练：帕累托前沿（v2 的实测数据）、计算预算、探测 KL（约 2 秒）
    python -m labs.ch32.forget_tour 4       # 专门化 OCR 时的遗忘与重放（需要 outputs/ch26/sft.pt，你自己跑）
    python -m labs.ch32.forget_tour 5       # 同一起点：继续 SFT / RFT / DPO / GRPO（需要 outputs/ch26/sft.pt，你自己跑）
"""
import copy
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

from miniocr.train.mix import mixed_batch, pareto_front, probe_kl, rft_select

torch.set_num_threads(8)
OUT = "outputs/ch32"
os.makedirs(OUT, exist_ok=True)
STEPS = 500
METHOD_STEPS = 200
OLD_TASKS = ("caption", "count", "count_color", "left", "exists", "ground", "refer", "compare")


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ---------------------------------------------------------------------- 1
def part1_pareto():
    section("1. 帕累托前沿：v2 版课程第 21 章的实测数据（字符级 GPT 学加法，代码困惑度衡量遗忘）")
    rows = [("全参数 1e-4", 21.0, 15.89), ("全参数 3e-4", 50.5, 40.52), ("全参数 1e-3", 55.5, 2828.16),
            ("LoRA r=8 3e-4", 13.7, 16.86), ("LoRA r=8 1e-3", 63.5, 160.18), ("LoRA r=8 3e-3", 93.8, 6629.59),
            ("只训最后两层 3e-4", 66.6, 9.08), ("只训最后两层 1e-3", 92.0, 1459.96),
            ("全参数 1e-3 + 重放", 98.6, 6.71), ("LoRA r=8 3e-3 + 重放", 85.3, 4.62)]
    for label, sub in (("没有重放的 8 个点", rows[:8]), ("全部 10 个点", rows)):
        front = pareto_front([(acc, -np.log(ppl)) for _, acc, ppl in sub])       # 准确率越高越好，困惑度越低越好
        print(f"  {label}：前沿上的点")
        for i in front:
            print(f"    {sub[i][0]:<22} 加法准确率 {sub[i][1]:5.1f}%   代码困惑度 {sub[i][2]:8.2f}")


# ---------------------------------------------------------------------- 2
def part2_budget():
    section("2. 计算预算：同样是\"一步\"，四种方法各要算多少（按实测的序列长度估算，单位：token 次前向）")
    from labs.ch26.sft_tour import NV, make_example
    from miniocr.train.sft import ROLE_ANSWER
    rng = np.random.default_rng(0)
    seqs = [make_example(rng, "text", multi_turn=0.0) for _ in range(500)]
    L = np.mean([len(s[1]) for s in seqs])
    A = np.mean([s[2].count(ROLE_ANSWER) for s in seqs])
    P = L - A
    print(f"  场景文字任务：序列平均 {L:.1f} 个 token（其中 {NV} 个视觉 token），回答平均 {A:.1f} 个 token")
    print("  约定：一次反向 ≈ 两次前向；生成时每写一个 token 是 1 个 token 的前向（KV cache），预填充是整个提示的前向")
    sft = 32 * L * 3
    gen = lambda n: n * (P + A)                                   # n 条回答：预填充 P + 逐个生成 A 个 token
    rft = gen(32 * 4) + 32 * L * 3                                # 采 32 题 × 4 个，再对选出的（至多 32 条）做 SFT
    dpo = 16 * 2 * L * 3 + 16 * 2 * L                             # 16 对：策略 2 条的前向 + 反向，参考模型 2 条的前向
    dpo_online = dpo + gen(16 * 4)                                # 在线：每步先采 16 题 × 4 个
    grpo = gen(8 * 8) + 8 * 8 * L * (1 + 1 + 2 * 3)               # 采 8 × 8，old_logp 和 ref 各一次前向，μ = 2 次前向 + 反向
    rows = [("继续 SFT（批 32）", sft), ("RFT（采 32 × 4，SFT ≤ 32 条）", rft), ("DPO 离线（16 对）", dpo),
            ("DPO 在线（16 对，每步采 16 × 4）", dpo_online), ("GRPO（8 题 × 8，μ = 2）", grpo)]
    for name, c in rows:
        print(f"    {name:<34} {c:10,.0f}   是继续 SFT 的 {c / sft:4.1f} 倍")
    print("  （下面实测一次采样和一次训练各要多久，和估算对照）")
    from labs.ch22.vlm_tour import build, img_float
    from labs.ch23.instruct_tour import prompt
    from labs.ch26.sft_tour import EOS, forward_loss, to_tensors
    from miniocr.generate import generate
    m = build("mlp")
    X = img_float(torch.from_numpy(np.stack([s[0][0] for s in seqs[:64]])))
    q = prompt("<image>What word is written in the image?")
    ids = torch.tensor([q] * 64)
    t0 = time.time()
    with torch.no_grad():
        generate(m.lm, ids, int(round(A)), temperature=1.0, inputs_embeds=m.embed(X, ids))
    t_gen = time.time() - t0
    imgs, p = to_tensors(seqs[:64])
    m.train()
    t0 = time.time()
    forward_loss(m, imgs, p).backward()
    t_sft = time.time() - t0
    print(f"  实测（随机权重的同结构 VLM，CPU）：64 条回答各生成 {int(round(A))} 个 token 用 {t_gen * 1000:.0f} 毫秒；"
          f"同样 64 条完整序列的一次前向 + 反向用 {t_sft * 1000:.0f} 毫秒；比值 {t_gen / t_sft:.2f}，"
          f"按 token 数估算的比值 {(P + A) / (3 * L):.2f}")


# ---------------------------------------------------------------------- 3
@torch.no_grad()
def part3_probe_kl():
    section("3. 不需要标签的遗忘指标：探测 KL 和困惑度（第 16 章的字符级 GPT，人为给权重加噪声，不是训练）")
    from labs.ch27.logprob_tour import char_gpt
    from miniocr.data.corpus import mixed_corpus, text_vocab
    from miniocr.rl.logprobs import token_logprobs
    V = text_vocab()
    base = char_gpt()
    text = mixed_corpus(1, 50)[:4000]                                       # 和第 27 章同一个种子的文字（预训练没见过）
    ids = torch.tensor([[V.stoi["<bos>"]] + V.encode(text[i:i + 255]) for i in range(0, 3840, 256)])
    lb = base(ids)[0]
    mask = torch.ones(ids.shape[0], ids.shape[1] - 1)
    ppl0 = torch.exp(-token_logprobs(lb, ids).mean()).item()
    print(f"  {ids.shape[0]} 段 × {ids.shape[1]} 个字符；起点的困惑度 {ppl0:.3f}")
    print(f"    {'噪声 σ（相对每个张量的标准差）':<30}{'探测 KL':>10}{'困惑度':>10}{'困惑度的变化':>12}")
    for sigma in (0.0, 0.01, 0.02, 0.05, 0.1, 0.2):
        m = copy.deepcopy(base)
        g = torch.Generator().manual_seed(0)
        for p in m.parameters():
            p.add_(sigma * p.std() * torch.randn(p.shape, generator=g))
        ln = m(ids)[0]
        ppl = torch.exp(-token_logprobs(ln, ids).mean()).item()
        print(f"    {sigma:<30}{probe_kl(ln[:, :-1], lb[:, :-1], mask):>10.4f}{ppl:>10.3f}{ppl / ppl0 - 1:>+12.1%}")


# ---------------------------------------------------------------------- 4、5：我们的 VLM
def load_sft():
    from labs.ch22.vlm_tour import build, need
    m = build("mlp")
    m.load_state_dict(torch.load(need("outputs/ch26/sft.pt"), map_location="cpu"))
    return m


def probe_set(n=64, seed=777):
    """九种任务的固定探测样本（用来算探测 KL），回答是真值。"""
    from labs.ch26.sft_tour import make_example
    rng = np.random.default_rng(seed)
    return [make_example(rng, multi_turn=0.0) for _ in range(n)]


@torch.no_grad()
def vlm_probe_kl(m, base, items):
    from labs.ch22.vlm_tour import img_float
    from labs.ch26.sft_tour import to_tensors
    from miniocr.train.sft import embed_packed
    imgs, p = to_tensors(items)
    ln = m.eval().lm(inputs_embeds=embed_packed(m, img_float(imgs), p["ids"]))[0]
    lb = base.eval().lm(inputs_embeds=embed_packed(base, img_float(imgs), p["ids"]))[0]
    return probe_kl(ln, lb, p["weights"] > 0)                    # 只看回答的位置


def sft_steps(m, steps, batch_fn, lr, log_every=100):
    from labs.ch26.sft_tour import forward_loss, to_tensors
    from miniocr.train.optim import warmup_cosine
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(0)
    t0 = time.time()
    m.train()
    for step in range(steps):
        items = batch_fn(rng)
        imgs, p = to_tensors(items)
        for gr in opt.param_groups:
            gr["lr"] = warmup_cosine(step, steps, min(50, steps // 10 + 1), lr, lr * 0.1)
        loss = forward_loss(m, imgs, p)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if log_every and (step + 1) % log_every == 0:
            print(f"      第 {step + 1:4d} 步  损失 {loss.item():.4f}  {time.time() - t0:.0f} 秒")
    return time.time() - t0


def report(m, base, probe, sec, label):
    from labs.ch26.sft_tour import quick_eval
    r = quick_eval(m)
    old = np.mean([r[t] for t in ("caption", "count", "left", "exists", "ground")])
    k = vlm_probe_kl(m, base, probe)
    print(f"    {label:<34} 场景文字 {r['text']:6.1%}   其他五个任务的平均 {old:6.1%}   探测 KL {k:.4f}   {sec:.0f} 秒")
    return r["text"], old, k


def part4_replay():
    section(f"4. 专门化 OCR（只训练场景文字）时的遗忘与重放：从第 26 章的 SFT 模型出发，每种设置 {STEPS} 步")
    if not os.path.exists("outputs/ch26/sft.pt"):
        print("  缺少 outputs/ch26/sft.pt：先跑第 26 章第 2 部分（python -m labs.ch26.sft_tour 2），它会自动保存这个模型")
        return
    from labs.ch26.sft_tour import BS, make_example
    from labs.ch28.memory_tour import lora_setup
    from miniocr.models.vlm import set_trainable
    base, probe = load_sft(), probe_set()
    results = [("出发点",) + report(base, base, probe, 0, "出发点（SFT）")]
    new = lambda rng: make_example(rng, "text", multi_turn=0.0)
    old = lambda rng: make_example(rng, OLD_TASKS[int(rng.integers(len(OLD_TASKS)))], multi_turn=0.0)
    settings = [("全参数 3e-4，重放 0", 3e-4, 0.0, "full"), ("全参数 3e-4，重放 10%", 3e-4, 0.1, "full"),
                ("全参数 3e-4，重放 30%", 3e-4, 0.3, "full"), ("全参数 3e-4，重放 50%", 3e-4, 0.5, "full"),
                ("全参数 1e-4，重放 0", 1e-4, 0.0, "full"), ("全参数 1e-3，重放 0", 1e-3, 0.0, "full"),
                ("冻结 ViT 3e-4，重放 0", 3e-4, 0.0, "freeze_vit"), ("LoRA r=16 1e-3，重放 0", 1e-3, 0.0, "lora")]
    for name, lr, ratio, how in settings:
        m = load_sft()
        if how == "freeze_vit":
            set_trainable(m, vision=False)
        elif how == "lora":
            lora_setup(m, 16)
        sec = sft_steps(m, STEPS, lambda rng: mixed_batch(rng, BS, ratio, new, old)[0], lr)
        results.append((name,) + report(m, base, probe, sec, name))
    front = pareto_front([(t, o) for _, t, o, _ in results])
    print("  帕累托前沿（场景文字、其他任务都越高越好）：" + "；".join(results[i][0] for i in front))


def part5_methods():
    section(f"5. 同一起点、同一批题（场景文字），四种继续训练的方法，各 {METHOD_STEPS} 个优化器步")
    if not os.path.exists("outputs/ch26/sft.pt"):
        print("  缺少 outputs/ch26/sft.pt：先跑第 26 章第 2 部分（python -m labs.ch26.sft_tour 2），它会自动保存这个模型")
        return
    from labs.ch22.vlm_tour import img_float
    from labs.ch23.instruct_tour import convo, example, prompt
    from labs.ch26.sft_tour import BS, EOS, NV, V, make_example
    from labs.ch30 import dpo_tour as c30
    from labs.ch31 import grpo_tour as c31
    from miniocr.metrics import ned
    from miniocr.rl.rollout import rollout
    from miniocr.train.sft import chat_roles
    base, probe = load_sft(), probe_set()
    report(base, base, probe, 0, "出发点（SFT）")

    m = load_sft()
    sec = sft_steps(m, METHOD_STEPS, lambda rng: [make_example(rng, "text", multi_turn=0.0) for _ in range(BS)], 1e-4)
    report(m, base, probe, sec, "继续 SFT（真值答案，1e-4）")

    m = load_sft()

    def rft_batch(rng, G=4):
        imgs, qs, refs = [], [], []
        for _ in range(BS):
            img, pairs = example("text", rng)
            imgs.append(img[0])
            qs.append(pairs[0][0])
            refs.append(pairs[0][1])
        X = torch.from_numpy(np.stack(imgs))
        resp, _ = rollout(m, img_float(X), [prompt(q) for q in qs], G, 16, EOS)
        r = torch.tensor([1 - ned(V.decode(t), refs[i // G]) for i, t in enumerate(resp)])
        keep = rft_select(r, G, threshold=1.0, k=1)                # 只要完全读对的
        out = []
        for j in keep:
            ids, roles = chat_roles(V, convo([(qs[j // G], V.decode(resp[j]))]), NV)
            out.append((imgs[j // G][None], ids, roles, "text"))
        return out or [make_example(rng, "text", multi_turn=0.0)]
    sec = sft_steps(m, METHOD_STEPS, rft_batch, 1e-4)
    report(m, base, probe, sec, "RFT（自己读对的当答案，1e-4）")

    m = load_sft()
    sec = c30.train_dpo(m, METHOD_STEPS, 0.1, source="text")
    report(m, base, probe, sec, "DPO（自采样对，β = 0.1，1e-5）")

    m = load_sft()
    saved = c31.RL_TASKS
    c31.RL_TASKS = ("text",)
    try:
        sec, _ = c31.grpo_train(m, c31.RLConfig(steps=METHOD_STEPS, lr=1e-5))
    finally:
        c31.RL_TASKS = saved
    report(m, base, probe, sec, "GRPO（1 − NED，1e-5）")


PARTS = {1: part1_pareto, 2: part2_budget, 3: part3_probe_kl, 4: part4_replay, 5: part5_methods}

if __name__ == "__main__":
    want = [int(a) for a in sys.argv[1:] if a.isdigit()] or list(PARTS)
    for k in want:
        PARTS[k]()
    if want == list(PARTS):
        print("\n第 32 章的演示全部完成。")
