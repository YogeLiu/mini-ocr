"""第 30 章：直接偏好优化（DPO）。推导的数值验证、似然位移、真实模型的偏好、在我们的 VLM 上做 DPO。只用 CPU。

    python -m labs.ch30.dpo_tour 1 2 3     # 手算、最优策略、似然位移：不训练神经网络，几秒
    python -m labs.ch30.dpo_tour 4         # Qwen3.5-0.8B：参考模型本来就偏好正确的描述吗（只推理，CPU 上约 70 秒）
    python -m labs.ch30.dpo_tour 5         # 在第 26 章 SFT 之后的 VLM 上做 DPO（需要 outputs/ch26/sft.pt，你自己跑）
"""
import copy
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F

from miniocr.data.prefs import hallucinated_caption
from miniocr.data.shapes import describe
from miniocr.rl.dpo import dpo_loss, optimal_policy, pairs_from_groups

torch.set_num_threads(8)
OUT = "outputs/ch30"
os.makedirs(OUT, exist_ok=True)
N_TEST = int(os.environ.get("N_TEST", 50))
DPO_STEPS = 300


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ---------------------------------------------------------------------- 1
def part1_by_hand():
    section("1. DPO 的损失与梯度的权重：h = r_w − r_l（隐含奖励之差）")
    print(f"    {'h':>6}{'损失 −log σ(h)':>16}{'梯度的权重 σ(−h)':>18}   含义")
    notes = {-4: "排反了，而且很确定：梯度最大", -1: "排反了", 0: "策略 = 参考模型（训练开始时）", 1: "排对了",
             4: "排对了，而且差距很大：几乎不再学", 10: ""}
    for h in (-4, -1, 0, 1, 4, 10):
        t = torch.tensor(float(h))
        print(f"    {h:>6}{-F.logsigmoid(t).item():>16.4f}{torch.sigmoid(-t).item():>18.4f}   {notes[h]}")
    print("  对 log π(y_w) 的梯度 = −β·σ(−h)，对 log π(y_l) 的梯度 = +β·σ(−h)：两个大小相同、方向相反")


# ---------------------------------------------------------------------- 2
def part2_optimal():
    section("2. 最优策略与 DPO：6 个候选回答，参考模型的概率 π_ref、真实奖励 R，β = 0.5")
    ref = torch.tensor([0.30, 0.25, 0.20, 0.12, 0.08, 0.05])
    R = torch.tensor([0.0, 1.0, 0.2, 2.0, 0.5, 1.5])
    beta = 0.5
    star = optimal_policy(ref.log(), R, beta)

    def run(hard, steps):
        """对所有 15 对 (i, j) 求 DPO 的期望损失。soft：i 被偏好的概率是 σ(R_i − R_j)（Bradley–Terry）；
        hard：奖励高的那个总是被偏好（确定性的标注）。表格策略：6 个 logits，从 π_ref 出发。"""
        z = ref.log().clone().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=0.05)
        I, J = torch.triu_indices(6, 6, 1)
        p = (R[I] > R[J]).float() if hard else torch.sigmoid(R[I] - R[J])
        for _ in range(steps):
            lp = torch.log_softmax(z, 0)
            h = beta * ((lp[I] - ref.log()[I]) - (lp[J] - ref.log()[J]))
            loss = (p * -F.logsigmoid(h) + (1 - p) * -F.logsigmoid(-h)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return torch.softmax(z, 0).detach(), loss.item()

    rows = [("π_ref", ref), ("R", R), ("π* = 解析解", star)]
    for steps in (200, 2000):
        pi, l = run(False, steps)
        rows.append((f"DPO，概率标注，{steps} 步", pi))
    for steps in (200, 2000, 20000):
        pi, l = run(True, steps)
        rows.append((f"DPO，确定标注，{steps} 步", pi))
    print("    " + " " * 22 + "".join(f"{'y' + str(k):>8}" for k in range(6)))
    for name, v in rows:
        print(f"    {name:<22}" + "".join(f"{x:8.3f}" for x in v.tolist()))
    pi, _ = run(False, 2000)
    imp = beta * (pi.log() - ref.log())
    print(f"  概率标注 2000 步之后：隐含奖励 β·log(π/π_ref) − 第 0 个 = {np.round((imp - imp[0]).tolist(), 3).tolist()}"
          f"；真实奖励 R − R_0 = {np.round((R - R[0]).tolist(), 3).tolist()}")
    print(f"  解析解的期望奖励 {(star * R).sum():.3f}，KL(π*‖π_ref) {(star * (star.log() - ref.log())).sum():.3f}；"
          f"π_ref 的期望奖励 {(ref * R).sum():.3f}")


# ---------------------------------------------------------------------- 3
def part3_displacement():
    section("3. 似然位移：chosen 和 rejected 很像时，DPO 可能把两者的概率一起压低（一个构造的小例子）")
    print("  4 个候选回答，策略的 logits = θ · φ(y)，θ 只有 2 个数（相当于模型的参数被所有回答共享）：")
    phi = torch.tensor([[1.0, 0.1],      # y_w：正确的回答
                        [1.0, -0.1],     # y_l：只差一点点的错误回答
                        [0.0, 3.0],      # y_o：没有出现在偏好数据里的第三种回答
                        [0.5, 0.0]])     # 另一个
    names = ["y_w（chosen）", "y_l（rejected）", "y_o（数据里没有）", "y_4"]
    theta0 = torch.tensor([2.0, 0.0])
    for i, n in enumerate(names):
        print(f"    {n:<18} φ = {np.round(phi[i].tolist(), 2).tolist()}")
    beta = 0.1
    for sft in (0.0, 1.0):
        th = theta0.clone().requires_grad_(True)
        opt = torch.optim.SGD([th], lr=1.0)
        ref = torch.log_softmax(phi @ theta0, 0)
        print(f"  {'DPO' if not sft else 'DPO + chosen 的 NLL（权重 1）'}，β = {beta}，学习率 1：")
        print(f"    {'步':>6}" + "".join(f"{n:>20}" for n in names) + f"{'隐含奖励之差 h':>16}")
        for step in range(201):
            lp = torch.log_softmax(phi @ th, 0)
            loss, st = dpo_loss(lp[0:1], lp[1:2], ref[0:1], ref[1:2], beta, sft_weight=sft, n_w=torch.ones(1))
            if step in (0, 10, 50, 200):
                print(f"    {step:>6}" + "".join(f"{p:>20.4f}" for p in lp.exp().tolist()) + f"{st['margin']:>16.3f}")
            opt.zero_grad()
            loss.backward()
            opt.step()
    print("  只用 DPO：θ 沿 φ_w − φ_l = (0, 0.2) 的方向走，y_o 在这个方向上的分量最大（3.0），它的概率涨得最快；"
          "chosen 和 rejected 一起变小，只是 rejected 掉得更多")


# ---------------------------------------------------------------------- 4：Qwen3.5
def part4_qwen_preference():
    section(f"4. Qwen3.5-0.8B：参考模型本来就偏好正确的描述吗？（{N_TEST} 张图，三种描述各算对数概率）")
    from PIL import Image
    from labs.ch27.logprob_tour import qwen_score_ids, qwen_text_ids
    from labs.ch23.instruct_tour import scene
    q = ("List every shape in the image from left to right, with size, color and shape. "
         "Answer like: a large red circle, a small blue square")
    rng = np.random.default_rng(7)
    rows = []
    t0 = time.time()
    for _ in range(N_TEST):
        img, objs, _ = scene(rng)
        im = Image.fromarray(img.transpose(1, 2, 0))
        good = describe(objs)
        sub, _ = hallucinated_caption(objs, rng, "substitute")
        add, _ = hallucinated_caption(objs, rng, "add")
        r = {}
        for name, cap in (("good", good), ("sub", sub), ("add", add)):
            lp = qwen_score_ids(im, q, qwen_text_ids(cap))
            r[name] = (lp.sum().item(), lp.mean().item(), len(lp))
        rows.append(r)
    print(f"  用时 {time.time() - t0:.0f} 秒")
    for k, label in (("sub", "换掉一个物体的颜色/形状（长度不变）"), ("add", "末尾多加一个不存在的物体（变长）")):
        ds = np.array([r["good"][0] - r[k][0] for r in rows])
        dm = np.array([r["good"][1] - r[k][1] for r in rows])
        print(f"  正确 vs {label}：")
        print(f"    按总对数概率，正确的更高：{np.mean(ds > 0):5.1%}；差值的中位数 {np.median(ds):+.2f}")
        print(f"    按每个 token 平均，正确的更高：{np.mean(dm > 0):5.1%}；差值的中位数 {np.median(dm):+.3f}")
    print(f"  三种描述的平均 token 数：正确 {np.mean([r['good'][2] for r in rows]):.1f}，换物体 "
          f"{np.mean([r['sub'][2] for r in rows]):.1f}，加物体 {np.mean([r['add'][2] for r in rows]):.1f}")


# ---------------------------------------------------------------------- 5：我们的 VLM
def pair_seq_logps(m, imgs, prompts, ws, ls):
    """一批成对数据的序列对数概率。返回 (pol_w, pol_l, n_w)：前两个 (B,)，n_w 是 chosen 的 token 数。"""
    from labs.ch26.sft_tour import EOS, PAD
    from miniocr.rl.logprobs import seq_logprob
    from miniocr.rl.rollout import make_batch, policy_logprobs
    B = len(prompts)
    ids, mask = make_batch(prompts + prompts, ws + ls, [True] * (2 * B), EOS, PAD)
    lp = seq_logprob(policy_logprobs(m, torch.cat([imgs, imgs]), ids), mask)
    return lp[:B], lp[B:], mask[:B].sum(1)


def caption_pairs(rng, B, kind="substitute"):
    from labs.ch22.vlm_tour import img_float
    from labs.ch23.instruct_tour import prompt, scene
    from labs.ch26.sft_tour import V
    imgs, ps, ws, ls = [], [], [], []
    for _ in range(B):
        img, objs, _ = scene(rng)
        imgs.append(img)
        ps.append(prompt("<image>Describe the image."))
        ws.append(V.encode(describe(objs)))
        ls.append(V.encode(hallucinated_caption(objs, rng, kind)[0]))
    return img_float(torch.from_numpy(np.stack(imgs))), ps, ws, ls


def text_pairs(m, rng, B, G=4):
    """场景文字任务：模型自己采 G 个回答，奖励 1 − NED，每组最好的当 chosen、最差的当 rejected。"""
    from labs.ch22.vlm_tour import img_float
    from labs.ch23.instruct_tour import example, prompt
    from labs.ch26.sft_tour import EOS, V
    from miniocr.metrics import ned
    from miniocr.rl.rollout import rollout
    imgs, ps, refs = [], [], []
    for _ in range(B):
        img, pairs = example("text", rng)
        imgs.append(img[0])
        ps.append(prompt(pairs[0][0]))
        refs.append(pairs[0][1])
    X = img_float(torch.from_numpy(np.stack(imgs)))
    resp, fin = rollout(m, X, ps, G, 16, EOS)
    r = torch.tensor([1 - ned(V.decode(t), refs[i // G]) for i, t in enumerate(resp)])
    pairs = pairs_from_groups(r, G, min_gap=0.1)
    if not pairs:
        return None
    k = [w // G for w, _ in pairs]
    return X[k], [ps[i] for i in k], [resp[w] for w, _ in pairs], [resp[l] for _, l in pairs], r.mean().item()


def hallucination_eval(m, n=200, seed=500):
    """描述里提到的物体有多少是图里没有的（第 23 章第 4 部分的方法），以及描述全对的比例。"""
    from labs.ch23.instruct_tour import ask, mentions, prompt, scene
    rng = np.random.default_rng(seed)
    Xs, objs_list = [], []
    for _ in range(n):
        img, objs, _ = scene(rng)
        Xs.append(img)
        objs_list.append(objs)
    caps = ask(m, torch.from_numpy(np.stack(Xs)), [prompt("<image>Describe the image.")] * n, max_new=90)
    tot = hal = 0
    for c, objs in zip(caps, objs_list):
        have = [(o["color"], o["shape"]) for o in objs]
        ms = mentions(c)
        tot += len(ms)
        hal += sum((col, sh) not in have for _, col, sh in ms)
    exact = np.mean([c == describe(o) for c, o in zip(caps, objs_list)])
    return hal / max(1, tot), exact


def train_dpo(m, steps, beta, source="caption", sft_weight=0.0, B=16, lr=1e-5, seed=0, log_every=50):
    ref = copy.deepcopy(m).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for step in range(steps):
        if source == "caption":
            imgs, ps, ws, ls = caption_pairs(rng, B)
        else:
            got = text_pairs(m, rng, B)
            if got is None:
                continue
            imgs, ps, ws, ls, _ = got
        m.train()
        pw, pl, nw = pair_seq_logps(m, imgs, ps, ws, ls)
        with torch.no_grad():
            rw, rl, _ = pair_seq_logps(ref, imgs, ps, ws, ls)
        loss, st = dpo_loss(pw, pl, rw, rl, beta, sft_weight, nw)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if log_every and (step + 1) % log_every == 0:
            print(f"      第 {step + 1:4d} 步  损失 {loss.item():.4f}  准确率 {st['acc']:.2f}  "
                  f"log π(y_w) − log π_ref(y_w) {(pw - rw).mean().item():+.3f}  "
                  f"log π(y_l) − log π_ref(y_l) {(pl - rl).mean().item():+.3f}  {time.time() - t0:.0f} 秒")
    return time.time() - t0


def part5_our_vlm():
    section(f"5. 在我们的 VLM 上做 DPO（从第 26 章 SFT 之后的模型出发，{DPO_STEPS} 步）")
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

    m = load()
    h, e = hallucination_eval(m)
    print(f"  出发点：描述中的幻觉 {h:.1%}，描述全对 {e:.1%}；{fmt(quick_eval(m))}")
    for name, kw in (("描述的幻觉对，β = 0.1", dict(beta=0.1)),
                     ("描述的幻觉对，β = 0.5", dict(beta=0.5)),
                     ("描述的幻觉对，β = 0.1，加 chosen 的 NLL", dict(beta=0.1, sft_weight=1.0)),
                     ("场景文字的自采样对（1 − NED），β = 0.1", dict(beta=0.1, source="text"))):
        print(f"  {name}：")
        m = load()
        sec = train_dpo(m, DPO_STEPS, **kw)
        h, e = hallucination_eval(m)
        print(f"    {sec:.0f} 秒；描述中的幻觉 {h:.1%}，描述全对 {e:.1%}；{fmt(quick_eval(m))}")


PARTS = {1: part1_by_hand, 2: part2_optimal, 3: part3_displacement, 4: part4_qwen_preference, 5: part5_our_vlm}

if __name__ == "__main__":
    want = [int(a) for a in sys.argv[1:] if a.isdigit()] or list(PARTS)
    for k in want:
        PARTS[k]()
    if want == list(PARTS):
        print("\n第 30 章的演示全部完成。")
