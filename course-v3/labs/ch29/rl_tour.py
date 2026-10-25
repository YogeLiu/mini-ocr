"""第 29 章：强化学习基础。策略梯度、基线与优势、组内标准化、KL 的估计、重要性比值与裁剪、奖励投机。只用 CPU。

    python -m labs.ch29.rl_tour 1 2 3 4    # 多臂老虎机、KL 估计量、裁剪：不训练神经网络，几十秒
    python -m labs.ch29.rl_tour 5          # Qwen3.5-0.8B 采样：组内奖励有没有方差（只推理，CPU 上约 3 分钟）
    python -m labs.ch29.rl_tour 6          # 奖励投机：用强化学习训练第 16 章的字符级 GPT（你自己跑）
"""
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

from miniocr.data.corpus import text_vocab
from miniocr.generate import generate
from miniocr.models.gpt import GPT, GPTConfig
from miniocr.rl.logprobs import token_logprobs
from miniocr.rl.pg import clipped_objective, group_advantages, kl_estimate, zero_variance_fraction

torch.set_num_threads(8)
OUT = "outputs/ch29"
os.makedirs(OUT, exist_ok=True)
V = text_vocab()
N_TEST = int(os.environ.get("N_TEST", 20))
HACK_STEPS = 100


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ---------------------------------------------------------------------- 1、2：多臂老虎机
MU = np.array([1.0, 1.5, 2.0, 5.0, 2.5])                 # 5 个拉杆的平均奖励，第 3 个（从 0 数）最好


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def sampled_gradient(theta, mu, rng, n=8, baseline="none"):
    """用 n 次拉杆估计 ∇J。∇_θ log π(a) = onehot(a) − π（softmax 策略）。"""
    pi = softmax(theta)
    a = rng.choice(len(theta), size=n, p=pi)
    r = mu[a] + rng.normal(size=n)                        # 每次的奖励 = 平均值 + 标准正态噪声
    if baseline == "none":
        adv = r
    elif baseline == "mean":
        adv = r - r.mean()
    elif baseline == "loo":
        adv = r - (r.sum() - r) / (n - 1)
    elif baseline == "group":                             # GRPO：减均值再除以标准差
        adv = group_advantages(torch.tensor(r), n).numpy()
    elif baseline == "exact":
        adv = r - (pi * mu).sum()
    else:
        raise ValueError(baseline)
    score = -np.tile(pi, (n, 1))
    score[np.arange(n), a] += 1.0
    return (adv[:, None] * score).mean(0)


def exact_gradient(theta, mu):
    pi = softmax(theta)
    return pi * (mu - (pi * mu).sum())                    # ∂J/∂θ_k = π_k (μ_k − J)


def part1_bandit_gradients():
    section("1. 多臂老虎机：采样估计的策略梯度，无偏吗？方差多大？（θ = 0，每次估计用 8 次拉杆，重复 20000 次）")
    rng = np.random.default_rng(0)
    theta = np.zeros(5)
    g = exact_gradient(theta, MU)
    print(f"  拉杆的平均奖励 μ = {MU.tolist()}；π = 均匀；J = {MU.mean():.2f}")
    print(f"  精确梯度 π_k (μ_k − J)：{np.round(g, 3).tolist()}")
    print(f"    {'基线':<8}{'平均梯度':<44}{'平均 / 精确（第 3 维）':>16}{'各维方差之和':>14}")
    for base in ("none", "mean", "loo", "group", "exact"):
        gs = np.array([sampled_gradient(theta, MU, rng, 8, base) for _ in range(20000)])
        print(f"    {base:<8}{str(np.round(gs.mean(0), 3).tolist()):<44}{gs.mean(0)[3] / g[3]:>16.3f}{gs.var(0).sum():>14.3f}")
    print("\n  所有奖励加 100（相对好坏不变）：")
    for base in ("none", "loo", "group"):
        gs = np.array([sampled_gradient(theta, MU + 100, rng, 8, base) for _ in range(20000)])
        print(f"    {base:<8}各维方差之和 {gs.var(0).sum():12.3f}   平均 / 精确（第 3 维）{gs.mean(0)[3] / g[3]:7.3f}")


def train_bandit(mu, steps, lr, baseline, seed, n=8):
    rng = np.random.default_rng(seed)
    theta = np.zeros(len(mu))
    curve = []
    for _ in range(steps):
        theta += lr * sampled_gradient(theta, mu, rng, n, baseline)
        curve.append((softmax(theta) * mu).sum())
    return np.array(curve)


def part2_bandit_training():
    section("2. 多臂老虎机的训练：奖励都加 100，不同的基线，同样的学习率和步数（20 个种子，300 步，每步 8 次拉杆）")
    print("  （这里训练的只是 5 个数 θ，不是神经网络，一共几秒钟）")
    for lr in (0.05, 0.5):
        print(f"  学习率 {lr}：")
        for base in ("none", "loo", "group"):
            curves = np.array([train_bandit(MU + 100, 300, lr, base, s) for s in range(20)]) - 100
            fin, at50 = curves[:, -1], curves[:, 49]
            print(f"    {base:<6} 第 50 步 J 的平均 {at50.mean():5.2f}；第 300 步 J 平均 {fin.mean():5.2f}，最差 {fin.min():5.2f}，"
                  f"最好 {fin.max():5.2f}（最优是 {MU.max():.1f}）")


# ---------------------------------------------------------------------- 3：KL
def char_gpt():
    from labs.ch22.vlm_tour import need
    ck = torch.load(need("outputs/ch16/gpt_char.pt"), map_location="cpu")
    m = GPT(GPTConfig(**ck["cfg"]))
    m.load_state_dict(ck["model"])
    return m.eval()


def part3_kl():
    section("3. 怎样用采样估计 KL(π‖π_ref)：k1、k2、k3")
    g = torch.Generator().manual_seed(0)
    print("  (a) 20 个类别的两个分布，从 π 采 200000 个样本：")
    pi = torch.softmax(torch.randn(20, generator=g), 0)
    for name, ref in (("π_ref 与 π 接近", torch.softmax(pi.log() + 0.3 * torch.randn(20, generator=g), 0)),
                      ("π_ref 与 π 差得远", torch.softmax(pi.log() + 2.0 * torch.randn(20, generator=g), 0))):
        exact = (pi * (pi.log() - ref.log())).sum().item()
        x = torch.multinomial(pi, 200_000, replacement=True, generator=g)
        print(f"    {name}：精确 KL = {exact:.4f}")
        for k in ("k1", "k2", "k3"):
            e = kl_estimate(pi.log()[x], ref.log()[x], k)
            print(f"      {k}  平均 {e.mean():.4f}   每个样本的标准差 {e.std():.4f}   最小 {e.min():+.4f}")

    print("\n  (b) 真实的语言模型：π = 第 16 章的字符级 GPT 在温度 0.7 下的分布，π_ref = 同一个模型在温度 1 下")
    m = char_gpt()
    torch.manual_seed(0)
    with torch.no_grad():
        start = torch.full((64, 1), V.stoi["<bos>"])
        seq = generate(m, start, 100, temperature=0.7, generator=torch.Generator().manual_seed(0))
        logits = m(seq)[0]
        lp_pi = token_logprobs(logits / 0.7, seq)              # 采样用的分布下，采到的 token 的对数概率
        lp_ref = token_logprobs(logits, seq)
        P = torch.log_softmax(logits[:, :-1] / 0.7, -1)
        R = torch.log_softmax(logits[:, :-1], -1)
        exact = (P.exp() * (P - R)).sum(-1)                    # 每个位置精确的 KL（对 100 个字符求和）
    print(f"    64 条 × 100 个字符；每个位置精确 KL 的平均 {exact.mean():.4f}，每条序列的 KL（100 个位置之和）平均 {exact.sum(1).mean():.3f}")
    for k in ("k1", "k2", "k3"):
        e = kl_estimate(lp_pi, lp_ref, k)
        print(f"      {k}  每个 token 平均 {e.mean():.4f}   每个 token 的标准差 {e.std():.4f}   最小 {e.min():+.4f}   "
              f"每条序列之和的标准差 {e.sum(1).std():.3f}")

    print("\n  (c) 把估计量直接放进损失、对它求导：期望的梯度是哪个 KL 的梯度？（20 个类别，精确求期望，不采样）")
    theta = torch.randn(20, generator=g).requires_grad_(True)
    ref_lp = torch.log_softmax(torch.randn(20, generator=g), 0)

    def grad_of(f):
        theta.grad = None
        f().backward()
        return theta.grad.clone()

    lp = lambda: torch.log_softmax(theta, 0)
    true_rev = grad_of(lambda: (lp().exp() * (lp() - ref_lp)).sum())                # ∇ KL(π‖π_ref)
    true_fwd = grad_of(lambda: (ref_lp.exp() * (ref_lp - lp())).sum())              # ∇ KL(π_ref‖π)
    print(f"    {'放进损失的量':<14}{'与 ∇KL(π‖π_ref) 的余弦':>22}{'与 ∇KL(π_ref‖π) 的余弦':>22}{'梯度的长度':>10}")
    print(f"    {'（精确）':<14}{1.0:>22.3f}{torch.cosine_similarity(true_rev, true_fwd, 0):>22.3f}{true_rev.norm():>10.3f}")
    for k in ("k1", "k2", "k3"):
        # 期望 Σ_x π(x)·k(x)，但 π(x) 这个"采样概率"不求导（采样是在 no_grad 下做的）——这就是"把 k 直接放进损失"
        gk = grad_of(lambda: (lp().exp().detach() * kl_estimate(lp(), ref_lp, k)).sum())
        cos = lambda u: torch.cosine_similarity(gk, u, 0).item() if gk.norm() > 1e-6 else float("nan")
        print(f"    {k:<14}{cos(true_rev):>22.3f}{cos(true_fwd):>22.3f}{gk.norm():>10.3f}")


# ---------------------------------------------------------------------- 4：裁剪
def part4_clip():
    section("4. 重要性比值与裁剪：目标和梯度随 ρ 怎样变（ε = 0.2）")
    rhos = torch.tensor([0.5, 0.7, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3, 1.5, 2.0])   # 避开 0.8、1.2 本身：在边界上 min 的两项相等，梯度会被平分
    for A in (1.0, -1.0):
        logp = torch.log(rhos).requires_grad_(True)
        obj, clipped = clipped_objective(logp, torch.zeros_like(rhos), torch.tensor(A))
        obj.sum().backward()
        print(f"  A = {A:+.0f}" + ("（好回答：想让 ρ 变大）" if A > 0 else "（坏回答：想让 ρ 变小）"))
        print("    ρ          " + "".join(f"{r:7.2f}" for r in rhos.tolist()))
        print("    目标       " + "".join(f"{o:7.2f}" for o in obj.tolist()))
        print("    ∂目标/∂logp" + "".join(f"{x:7.2f}" for x in logp.grad.tolist()))
        print("    被裁剪     " + "".join(f"{'是' if c else '':>6} " for c in clipped.tolist()))


# ---------------------------------------------------------------------- 5：Qwen3.5 的组内方差
def qwen_sample(img, question, G, max_new=5, temperature=1.0, **kw):
    from labs.ch24.hf_tour import qwen, qwen_inputs
    tok, m = qwen()
    inp, _ = qwen_inputs(img, question, **kw)
    rep = dict(input_ids=inp["input_ids"].repeat(G, 1), attention_mask=inp["attention_mask"].repeat(G, 1),
               pixel_values=inp["pixel_values"].repeat(G, 1), image_grid_thw=inp["image_grid_thw"].repeat(G, 1),
               mm_token_type_ids=inp["mm_token_type_ids"].repeat(G, 1))
    with torch.no_grad():
        out = m.generate(**rep, max_new_tokens=max_new, do_sample=True, temperature=temperature, top_k=0, top_p=1.0)
    return [tok.decode(o[inp["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]


def part5_qwen_groups():
    section(f"5. Qwen3.5-0.8B 在玩具任务上采样：每题 8 个回答（温度 1），组内的奖励有没有方差（每个任务 {N_TEST} 题）")
    from PIL import Image
    from labs.ch22.vlm_tour import test_set
    prompts = {"count": "How many shapes are in the image? Answer with a single digit.",
               "left": "What color is the leftmost shape? Answer with one word: red, green, blue or yellow.",
               "exists": None}
    torch.manual_seed(0)
    for task, q0 in prompts.items():
        X, pairs = test_set(task)
        t0 = time.time()
        rewards, examples = [], []
        for i in range(N_TEST):
            img = Image.fromarray(X[i].permute(1, 2, 0).numpy())
            q, a = pairs[i]
            outs = qwen_sample(img, q0 or (q + " Answer yes or no."), 8)
            r = [float(bool(o) and o.lower().strip(" .").split()[0].strip(".,") == a) for o in outs]
            rewards += r
            if 0 < sum(r) < 8 and len(examples) < 2:
                examples.append((a, outs))
        R = torch.tensor(rewards)
        per_q = R.view(-1, 8).mean(1)
        print(f"  {task:<7} 平均奖励 {R.mean():.1%}；组内全对 {(per_q == 1).float().mean():.0%}、全错 {(per_q == 0).float().mean():.0%}、"
              f"有对有错 {((per_q > 0) & (per_q < 1)).float().mean():.0%}；没有梯度的组 {zero_variance_fraction(R, 8):.0%}"
              f"（{time.time() - t0:.0f} 秒）")
        for a, outs in examples:
            print(f"      真值 {a!r}：{outs}")


# ---------------------------------------------------------------------- 6：奖励投机
def space_reward(texts):
    return torch.tensor([t.count(" ") / max(1, len(t)) for t in texts])


def train_hack(beta, steps=HACK_STEPS, n_prompt=8, G=8, length=48, lr=1e-4, seed=0, log_every=20):
    """策略 = 字符级 GPT，参考模型 = 它的一份冻结的副本。每步：从 <bos> 采 n_prompt × G 条长 length 的文字，
    奖励 = 空格的比例，组内标准化的优势，损失 = −A·log π（每个 token）+ β·k3。"""
    torch.manual_seed(seed)
    pol, ref = char_gpt(), char_gpt()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(pol.parameters(), lr=lr, weight_decay=0.0)
    gen = torch.Generator().manual_seed(seed)
    start = torch.full((n_prompt * G, 1), V.stoi["<bos>"])
    log = []
    for step in range(steps + 1):
        with torch.no_grad():
            seq = generate(pol, start, length, temperature=1.0, generator=gen)
        texts = [V.decode(s[1:].tolist()) for s in seq]
        r = space_reward(texts)
        adv = group_advantages(r, G)[:, None]
        pol.train()
        lp = token_logprobs(pol(seq)[0], seq)
        with torch.no_grad():
            rlp = token_logprobs(ref(seq)[0], seq)
        kl = kl_estimate(lp, rlp, "k3")
        if step % log_every == 0:
            log.append((step, r.mean().item(), kl.sum(1).mean().item(), texts[:3]))
            print(f"      β = {beta:<5} 第 {step:3d} 步  平均奖励（空格比例）{r.mean():.3f}  KL（每条之和）{kl.sum(1).mean():7.3f}")
        if step == steps:
            break
        loss = (-adv * lp + beta * kl).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step()
    return log


def part6_reward_hacking():
    section(f"6. 奖励投机：奖励 = 输出里空格的比例，β 不同的 KL 惩罚（字符级 GPT，{HACK_STEPS} 步）")
    for beta in (0.0, 0.05, 0.2, 1.0):
        t0 = time.time()
        log = train_hack(beta)
        print(f"    β = {beta}：{time.time() - t0:.0f} 秒；最后的 3 条输出：")
        for t in log[-1][3]:
            print(f"      {t.replace(chr(10), '⏎')!r}")


PARTS = {1: part1_bandit_gradients, 2: part2_bandit_training, 3: part3_kl, 4: part4_clip, 5: part5_qwen_groups,
         6: part6_reward_hacking}

if __name__ == "__main__":
    want = [int(a) for a in sys.argv[1:] if a.isdigit()] or list(PARTS)
    for k in want:
        PARTS[k]()
    if want == list(PARTS):
        print("\n第 29 章的演示全部完成。")
