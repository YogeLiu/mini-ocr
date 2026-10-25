"""第 31 章：GRPO 与可验证奖励。多任务奖励、三种汇总方式、超长惩罚、真实模型的组内奖励、在我们的 VLM 上做 GRPO 与 DAPO 消融。

    python -m labs.ch31.grpo_tour 1 2 3    # 奖励、汇总方式、超长惩罚：不训练，一秒
    python -m labs.ch31.grpo_tour 4        # Qwen3.5-0.8B 读文字行时采样：0/1 奖励 vs 1 − NED（只推理，CPU 上约 70 秒）
    python -m labs.ch31.grpo_tour 5        # GRPO 与 DAPO 的四个改动（需要 outputs/ch26/sft.pt，你自己跑）
"""
import copy
import os
import sys
import time
from dataclasses import dataclass, replace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

from miniocr.rl.grpo import aggregate, grpo_loss, keep_groups, overlong_penalty
from miniocr.rl.pg import group_advantages, zero_variance_fraction
from miniocr.rl.rewards import task_reward

torch.set_num_threads(8)
OUT = "outputs/ch31"
os.makedirs(OUT, exist_ok=True)
N_TEST = int(os.environ.get("N_TEST", 20))
RL_TASKS = ("count", "count_color", "left", "exists", "ground", "text")


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ---------------------------------------------------------------------- 1
def part1_rewards():
    section("1. 多任务的可验证奖励（格式不对一律 0 分）")
    cases = [("count", "2", "2"), ("count", "3", "2"), ("count", "two", "2"), ("left", "blue", "red"),
             ("left", "Red", "red"), ("exists", "yes.", "yes"), ("ground", "none", "none"),
             ("ground", "[10,10,30,30]", "none"), ("ground", "[10,10,30,30]", "[10,10,30,30]"),
             ("ground", "[14,10,34,30]", "[10,10,30,30]"), ("ground", "[0,0,63,63]", "[10,10,30,30]"),
             ("ground", "[10,10,30", "[10,10,30,30]"), ("text", "cat", "cat"), ("text", "cart", "cat"),
             ("text", "c", "cat"), ("text", "", "cat")]
    print(f"    {'任务':<8}{'真值':<18}{'回答':<18}{'奖励':>6}")
    for t, p, r in cases:
        print(f"    {t:<8}{r!r:<18}{p!r:<18}{task_reward(t, p, r):>6.3f}")


# ---------------------------------------------------------------------- 2
def part2_aggregate():
    section("2. 三种汇总方式：每个 token 的权重（4 条回答，长度 2、4、10、40；优势 +1、+1、−1、−1）")
    lens = [2, 4, 10, 40]
    adv = torch.tensor([1.0, 1.0, -1.0, -1.0])
    T = max(lens)
    mask = torch.zeros(4, T)
    for i, n in enumerate(lens):
        mask[i, :n] = 1
    print(f"    {'汇总方式':<34}" + "".join(f"{'长 ' + str(n):>10}" for n in lens))
    for mode, label in (("sample", "sample（GRPO：先按回答平均）"), ("token", "token（DAPO：所有 token 一起）"),
                        ("constant", "constant（Dr. GRPO：除以 B·T_max）")):
        x = torch.zeros(4, T, requires_grad=True)
        aggregate(x, mask, mode).backward()
        w = x.grad
        print(f"    {label:<34}" + "".join(f"{w[i, 0].item():>10.4f}" for i in range(4)) + "   ← 每个 token 的权重")
        print(f"    {'':<34}" + "".join(f"{(w[i] * mask[i]).sum().item() * adv[i].item():>+10.3f}" for i in range(4))
              + "   ← 整条回答被推动的总量（权重之和 × 优势）")


# ---------------------------------------------------------------------- 3
def part3_overlong():
    section("3. DAPO 的软超长惩罚（上限 24 个 token，缓冲区 4）")
    print("    长度  " + "".join(f"{n:>6}" for n in (10, 19, 20, 21, 22, 23, 24, 25)))
    print("    惩罚  " + "".join(f"{overlong_penalty(n, 24, 4):>6.2f}" for n in (10, 19, 20, 21, 22, 23, 24, 25)))


# ---------------------------------------------------------------------- 4：Qwen3.5 读文字行
def part4_qwen_ocr():
    section(f"4. Qwen3.5-0.8B 读文字行（放大 2×）：每行采 8 次（温度 1），两种奖励的组内方差（{N_TEST} 行）")
    from labs.ch24.hf_tour import lines
    from labs.ch29.rl_tour import qwen_sample
    from miniocr.metrics import ned
    imgs, texts = lines(N_TEST)
    q = "Read the text in the image. Output only the text, exactly as written."
    torch.manual_seed(0)
    t0 = time.time()
    exact, soft, examples = [], [], []
    for im, t in zip(imgs, texts):
        outs = qwen_sample(im, q, 8, max_new=32, scale=2, min_pixels=32 * 32)      # 与第 24、27 章相同的放大方式
        exact += [float(o == t) for o in outs]
        soft += [max(0.0, 1 - ned(o, t)) for o in outs]
        if len(examples) < 3 and len(set(outs)) > 1:
            examples.append((t, outs[:4], [round(1 - ned(o, t), 2) for o in outs[:4]]))
    E, S = torch.tensor(exact), torch.tensor(soft)
    print(f"  用时 {time.time() - t0:.0f} 秒；整行正确 {E.mean():.1%}；平均 1 − NED {S.mean():.3f}")
    for name, R in (("整行正确（0/1）", E), ("1 − NED（连续）", S)):
        g = R.view(-1, 8)
        print(f"    {name:<14} 没有梯度的组 {zero_variance_fraction(R, 8):4.0%}   组内标准差的平均 {g.std(1).mean():.3f}   "
              f"全对的组 {(g.min(1).values == 1).float().mean():4.0%}")
    both = keep_groups(E, 8) | keep_groups(S, 8)
    only_soft = keep_groups(S, 8) & ~keep_groups(E, 8)
    print(f"  0/1 奖励没有梯度、1 − NED 有梯度的组：{only_soft.float().mean():.0%}（{int(only_soft.sum())} 行）；"
          f"至少一种有梯度：{both.float().mean():.0%}")
    for t, outs, rs in examples:
        print(f"    真值 {t!r}：前 4 次 {outs}  1 − NED {rs}")


# ---------------------------------------------------------------------- 5：我们的 VLM
@dataclass
class RLConfig:
    steps: int = 200
    n_prompt: int = 8               # 每步的题目数
    G: int = 8                      # 每道题采几个回答
    mu: int = 2                     # 每批样本更新几次（> 1 时裁剪才起作用）
    lr: float = 1e-6
    beta: float = 0.04              # KL 系数（GRPO 原文 0.04；DAPO 为 0）
    eps_low: float = 0.2
    eps_high: float = 0.2           # clip-higher：0.28
    mode: str = "sample"            # 汇总方式：sample（GRPO）/ token（DAPO）/ constant（Dr. GRPO）
    std: bool = True                # 优势除以组内标准差（Dr. GRPO 去掉）
    dynamic: bool = False           # 动态采样：去掉没有梯度的组并补采
    overlong: bool = False          # 软超长惩罚
    max_new: int = 24
    cache: int = 4
    seed: int = 0
    log_every: int = 20


def sample_prompts(rng, n):
    from labs.ch23.instruct_tour import example, prompt
    imgs, ps, refs, tasks = [], [], [], []
    for _ in range(n):
        t = RL_TASKS[int(rng.integers(len(RL_TASKS)))]
        img, pairs = example(t, rng)
        imgs.append(img[0])
        ps.append(prompt(pairs[0][0]))
        refs.append(pairs[0][1])
        tasks.append(t)
    return torch.from_numpy(np.stack(imgs)), ps, refs, tasks


def collect(m, rng, cfg):
    """采一批（或者在动态采样下，采到 n_prompt 个有梯度的组为止，最多补采 4 次）。"""
    from labs.ch22.vlm_tour import img_float
    from labs.ch26.sft_tour import EOS, V
    from miniocr.rl.rollout import rollout
    got = dict(X=[], ps=[], refs=[], tasks=[], resp=[], fin=[], r=[])
    zero_frac, tries = [], 0
    while len(got["ps"]) < cfg.n_prompt and tries < (5 if cfg.dynamic else 1):
        tries += 1
        X, ps, refs, tasks = sample_prompts(rng, cfg.n_prompt)
        resp, fin = rollout(m, img_float(X), ps, cfg.G, cfg.max_new, EOS)
        r = torch.tensor([task_reward(tasks[i // cfg.G], V.decode(t), refs[i // cfg.G]) for i, t in enumerate(resp)])
        if cfg.overlong:
            r = r + torch.tensor([overlong_penalty(len(t) + f, cfg.max_new, cfg.cache) for t, f in zip(resp, fin)])
        zero_frac.append(zero_variance_fraction(r, cfg.G))
        keep = keep_groups(r, cfg.G) if cfg.dynamic else torch.ones(cfg.n_prompt, dtype=torch.bool)
        for g in torch.nonzero(keep).flatten().tolist():
            if len(got["ps"]) == cfg.n_prompt:
                break
            got["X"].append(X[g])
            got["ps"].append(ps[g])
            got["refs"].append(refs[g])
            got["tasks"].append(tasks[g])
            sl = slice(g * cfg.G, (g + 1) * cfg.G)
            got["resp"] += resp[sl]
            got["fin"] += fin[sl]
            got["r"].append(r[sl])
    if not got["ps"]:
        return None, float(np.mean(zero_frac))
    got["X"] = torch.stack(got["X"])
    got["r"] = torch.cat(got["r"])
    return got, float(np.mean(zero_frac))


def grpo_train(m, cfg):
    from labs.ch22.vlm_tour import img_float
    from labs.ch26.sft_tour import EOS, PAD
    from miniocr.rl.rollout import make_batch, policy_logprobs
    ref = None
    if cfg.beta > 0:
        ref = copy.deepcopy(m).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    t0, hist = time.time(), []
    for step in range(cfg.steps):
        b, zf = collect(m, rng, cfg)
        if b is None:
            continue
        n = len(b["ps"])
        adv = group_advantages(b["r"], cfg.G, std=cfg.std)
        rows = [i // cfg.G for i in range(n * cfg.G)]
        ids, mask = make_batch([b["ps"][i] for i in rows], b["resp"], b["fin"], EOS, PAD)
        X = img_float(b["X"])[rows]
        with torch.no_grad():
            old = policy_logprobs(m.eval(), X, ids)
            ref_lp = policy_logprobs(ref, X, ids) if ref is not None else None
        m.train()
        for _ in range(cfg.mu):
            logp = policy_logprobs(m, X, ids)
            loss, st = grpo_loss(logp, old, adv, mask, ref_lp, cfg.beta, cfg.eps_low, cfg.eps_high, cfg.mode)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        lens = mask.sum(1)
        rec = dict(step=step + 1, reward=b["r"].mean().item(), zero=zf, clip=st["clip_frac"], kl=st["kl"],
                   length=lens.mean().item(), trunc=1 - np.mean(b["fin"]),
                   by_task={t: np.mean([b["r"][g * cfg.G:(g + 1) * cfg.G].mean().item()
                                        for g in range(n) if b["tasks"][g] == t]) for t in set(b["tasks"])})
        hist.append(rec)
        if cfg.log_every and (step + 1) % cfg.log_every == 0:
            print(f"      第 {step + 1:4d} 步  奖励 {rec['reward']:.3f}  没有梯度的组 {zf:.0%}  被裁剪 {st['clip_frac']:.3f}  "
                  f"KL {st['kl']:.4f}  回答长 {rec['length']:.1f}  截断 {rec['trunc']:.0%}  {time.time() - t0:.0f} 秒")
    return time.time() - t0, hist


def part5_our_vlm():
    base = RLConfig()
    settings = [("GRPO（原文）", base),
                ("+ clip-higher（ε_high = 0.28）", replace(base, eps_high=0.28)),
                ("+ token 级汇总", replace(base, mode="token")),
                ("+ 动态采样", replace(base, dynamic=True)),
                ("+ 软超长惩罚", replace(base, overlong=True)),
                ("DAPO（四个改动 + β = 0）", replace(base, eps_high=0.28, mode="token", dynamic=True, overlong=True, beta=0.0)),
                ("Dr. GRPO（constant 汇总，不除以标准差）", replace(base, mode="constant", std=False))]
    section(f"5. 在我们的 VLM 上做 GRPO：多任务奖励，{base.steps} 步，每步 {base.n_prompt} 题 × {base.G} 个回答，"
            f"每批更新 {base.mu} 次")
    path = "outputs/ch26/sft.pt"
    if not os.path.exists(path):
        print(f"  缺少 {path}：先跑第 26 章第 2 部分（python -m labs.ch26.sft_tour 2），它会自动保存这个模型")
        return
    from labs.ch22.vlm_tour import build
    from labs.ch26.sft_tour import fmt, quick_eval

    def load():
        m = build("mlp")
        m.load_state_dict(torch.load(path, map_location="cpu"))
        return m

    print(f"  出发点：{fmt(quick_eval(load()))}")
    for name, cfg in settings:
        print(f"  {name}：")
        m = load()
        sec, hist = grpo_train(m, cfg)
        print(f"    {sec:.0f} 秒；{fmt(quick_eval(m))}")
        if hist:
            last = hist[-max(1, len(hist) // 10):]
            tasks = sorted({t for h in last for t in h["by_task"]})
            print("    最后 10% 步的平均奖励：" + "  ".join(
                f"{t} {np.mean([h['by_task'][t] for h in last if t in h['by_task']]):.2f}" for t in tasks))


PARTS = {1: part1_rewards, 2: part2_aggregate, 3: part3_overlong, 4: part4_qwen_ocr, 5: part5_our_vlm}

if __name__ == "__main__":
    want = [int(a) for a in sys.argv[1:] if a.isdigit()] or list(PARTS)
    for k in want:
        PARTS[k]()
    if want == list(PARTS):
        print("\n第 31 章的演示全部完成。")
