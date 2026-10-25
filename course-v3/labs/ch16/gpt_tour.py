"""第 16 章：从零写 GPT，在混合语料上预训练。默认只用 CPU（GPU 被别的程序占用时也能跑）。

    python -m labs.ch16.gpt_tour             # 全部 6 个部分
    python -m labs.ch16.gpt_tour 3           # 只运行第 3 部分（预训练，保存 outputs/ch16/gpt_char.pt）
    python -m labs.ch16.gpt_tour --gpu       # 有空闲 GPU 时用 GPU
"""
import math
import os
import sys
import time

if "--gpu" not in sys.argv:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from miniocr.data.corpus import mixed_corpus, text_vocab
from miniocr.models.gpt import GPT, GPTConfig
from miniocr.tokenize import NGramLM
from miniocr.train.lm import eval_loss, train_lm

DEV = "cuda" if "--gpu" in sys.argv and torch.cuda.is_available() else "cpu"
torch.set_num_threads(min(16, os.cpu_count() or 1))
OUT = "outputs/ch16"
os.makedirs(OUT, exist_ok=True)
V = text_vocab()
CFG = GPTConfig(vocab_size=len(V), block_size=512, n_layer=4, n_head=4, d_model=256)   # RoPE 表留到 512，训练只用 256
STEPS, B, T = 3000, 16, 256
_C = {}


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def data():
    """训练：mixed_corpus(0, 20000)，前 4000 段就是第 14 章的训练语料；验证：与第 14 章完全相同的 mixed_corpus(1, 400)。"""
    if "tr" not in _C:
        tr, va = mixed_corpus(0, 20000), mixed_corpus(1, 400)
        _C["tr_text"], _C["va_text"] = tr, va
        _C["tr"] = torch.tensor(V.encode(tr), dtype=torch.long)
        _C["va"] = torch.tensor(V.encode(va), dtype=torch.long)
    return _C["tr"], _C["va"]


# ---------------------------------------------------------------------- 1
def count(cfg):
    d, f, L, Vs = cfg.d_model, cfg.ff_dim(), cfg.n_layer, cfg.vocab_size
    attn, mlp, norms = 4 * d * d, 3 * d * f, 2 * d
    return dict(emb=Vs * d, attn=attn, mlp=mlp, norms=norms, layer=attn + mlp + norms,
                total=Vs * d + L * (attn + mlp + norms) + d)


def part1_params():
    section("1. 数参数")
    c = count(CFG)
    m = GPT(CFG)
    d, f = CFG.d_model, CFG.ff_dim()
    print(f"  配置：词表 {CFG.vocab_size}，d = {d}，{CFG.n_layer} 层，{CFG.n_head} 个头（每头 {d // CFG.n_head} 维），SwiGLU 中间维 {f}，上下文 {CFG.block_size}")
    print(f"    嵌入            {CFG.vocab_size} × {d} = {c['emb']:,}（输出层与它共享，不另算）")
    print(f"    每层注意力      4 × {d}² = {c['attn']:,}（Wq、Wk、Wv、Wo，无偏置）")
    print(f"    每层 SwiGLU     3 × {d} × {f} = {c['mlp']:,}（gate、up、down）")
    print(f"    每层两个 RMSNorm 2 × {d} = {c['norms']:,}")
    print(f"    合计            {c['emb']:,} + {CFG.n_layer} × {c['layer']:,} + {d}（最后的 RMSNorm）= {c['total']:,}")
    print(f"  m.n_params() = {m.n_params():,}  {'一致' if m.n_params() == c['total'] else '不一致！'}")
    print(f"  如果输出层不共享权重：再多 {c['emb']:,}；本模型词表小，嵌入只占 {c['emb'] / c['total']:.1%}")
    # Qwen3-0.6B 的公开配置（config.json）：不下载，只按公式算
    q = dict(V=151936, d=1024, L=28, h=16, kv=8, dh=128, f=3072)
    attn = q["d"] * q["h"] * q["dh"] * 2 + q["d"] * q["kv"] * q["dh"] * 2        # Wq、Wo 各 d×(h·dh)；Wk、Wv 各 d×(kv·dh)
    mlp = 3 * q["d"] * q["f"]
    norms = 2 * q["d"] + 2 * q["dh"]                                            # 两个 RMSNorm + q_norm、k_norm
    emb = q["V"] * q["d"]
    tot = emb + q["L"] * (attn + mlp + norms) + q["d"]
    print(f"  对照 Qwen3-0.6B（按它的 config.json 计算）：嵌入 {emb / 1e6:.1f}M，每层注意力 {attn / 1e6:.2f}M + SwiGLU {mlp / 1e6:.2f}M，")
    print(f"    {q['L']} 层合计 {q['L'] * (attn + mlp + norms) / 1e6:.1f}M，总计 {tot / 1e6:.1f}M ≈ 0.6B（与名字一致）；嵌入占 {emb / tot:.0%}")
    print(f"    它用了分组查询注意力（GQA）：16 个查询头共用 8 组键值头，Wk、Wv 只有 Wq 的一半大（第 17 章：KV cache 也小一半）")


# ---------------------------------------------------------------------- 2
def part2_init():
    section("2. 初始化体检")
    torch.manual_seed(0)
    m = GPT(CFG)
    tr, va = data()
    g = torch.Generator().manual_seed(0)
    s = torch.randint(0, len(va) - T - 1, (B,), generator=g)
    x = torch.stack([va[i:i + T] for i in s]); y = torch.stack([va[i + 1:i + T + 1] for i in s])
    with torch.no_grad():
        l_next = m(x, targets=y)[1].item()
        l_same = m(x, targets=x)[1].item()
        l_rand = m(x, targets=torch.randint(4, len(V), x.shape))[1].item()
    print(f"  初始损失：预测下一个字符 {l_next:.3f}，随机目标 {l_rand:.3f}，ln {len(V)} = {math.log(len(V)):.3f}")
    print(f"  如果目标就是当前字符本身：{l_same:.3f}——明显更低。原因：输出层与嵌入共享权重，")
    print("  最后一层的向量 h_t 里有残差带来的 e(x_t)，logit_{x_t} = ⟨norm(h_t), e(x_t)⟩ ≈ ‖e‖²·缩放，比别的 logit 大。")
    print("  所以刚初始化的模型偏爱\"重复当前字符\"；训练很快就会改掉这个偏好")
    acts = []
    hooks = [b.register_forward_hook(lambda mod, i, o: acts.append(o.std().item())) for b in m.blocks]
    with torch.no_grad():
        m(x)
    for h in hooks:
        h.remove()
    print(f"  各层输出（残差流）的标准差：{[round(a, 4) for a in acts]}；嵌入初始化标准差 0.02")
    print("  残差流每过一层都会变大一些（残差是\"加\"上去的，第 11 章）。残差支路的输出层初始化成 0.02/√(2L)，")
    print("  让每层加的量与嵌入同一数量级；层数越多每层加得越少，L 层加起来的总量大致不随 L 变化")


# ---------------------------------------------------------------------- 3
def part3_pretrain():
    section(f"3. 预训练：{STEPS} 步 × 批 {B} × 长 {T} = {STEPS * B * T / 1e6:.1f}M 个字符 token")
    tr, va = data()
    print(f"  训练语料 {len(tr):,} 个字符（2 万段），验证 {len(va):,}（与第 14 章相同）；设备 {DEV}")
    torch.manual_seed(0)
    m = GPT(CFG).to(DEV)
    hist = train_lm(m, tr, va, STEPS, B=B, T=T, eval_every=250, device=DEV)
    torch.save({"cfg": CFG.__dict__, "model": m.state_dict(), "hist": hist}, f"{OUT}/gpt_char.pt")
    by_pos(m, va)
    print(f"  每个字符 token 平均被看了 {STEPS * B * T / len(tr):.1f} 遍")
    ng = NGramLM(5, len(V), k=0.01).fit(tr.tolist())
    print(f"  对照：同样 2 万段训练的字符 5 元模型 {ng.cross_entropy(va.tolist()) / math.log(2):.3f} bits/字符；"
          f"第 14 章最好的结果（4000 段，BPE 3 元）2.404")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    st = [h[0] for h in hist]
    ax.plot(st, [h[1] / math.log(2) for h in hist], label="train")
    ax.plot(st, [h[2] / math.log(2) for h in hist], label="val")
    ax.axhline(2.404, ls="--", c="gray", label="best n-gram (ch14)")
    ax.set_xlabel("step"); ax.set_ylabel("bits / char"); ax.legend(); fig.tight_layout()
    fig.savefig(f"{OUT}/curve.png", dpi=100); plt.close(fig)
    print(f"  曲线存到 {OUT}/curve.png；模型存到 {OUT}/gpt_char.pt（第 17、22、27–29 章会用它）")


@torch.no_grad()
def by_pos(m, va, L=512):
    """把验证语料切成长 L 的块（不重叠），统计每个位置的平均损失。训练只见过 0…T−1 的位置。"""
    m.eval()
    xs = torch.stack([va[s:s + L] for s in range(0, len(va) - L - 1, L)]).to(DEV)
    ys = torch.stack([va[s + 1:s + L + 1] for s in range(0, len(va) - L - 1, L)]).to(DEV)
    lg = m(xs)[0]
    loss = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), ys.reshape(-1), reduction="none").view(len(xs), L)
    pos = loss.mean(0).cpu() / math.log(2)
    ranges = ((0, 8), (8, 64), (64, T), (T, T + 32), (T + 32, T + 128), (T + 128, L))
    print(f"  按位置统计（{len(xs)} 个长 {L} 的块；训练时只见过位置 0–{T - 1}）：")
    print("    " + "  ".join(f"位置 {a}–{b - 1}: {pos[a:b].mean():.3f}" for a, b in ranges))
    print(f"  前 {T} 个位置（与训练相同的长度）平均 {pos[:T].mean():.3f} bits/字符；开头几个字符没有上下文，损失高")
    print("  超出训练长度后损失迅速变差：RoPE 的角度 t·θ 在训练中从没取到过这么大的值，")
    print("  模型没学过怎样处理这些\"没见过的旋转\"。RoPE 只保证分数依赖相对位置，不保证能外推到更长的序列")


def load():
    if "model" not in _C:
        if not os.path.exists(f"{OUT}/gpt_char.pt"):
            part3_pretrain()
        ck = torch.load(f"{OUT}/gpt_char.pt", map_location=DEV)
        m = GPT(GPTConfig(**ck["cfg"])).to(DEV)
        m.load_state_dict(ck["model"]); m.eval()
        _C["model"] = m
    return _C["model"]


# ---------------------------------------------------------------------- 4
@torch.no_grad()
def sample(m, prompt, n, g, temp=1.0):
    """最朴素的采样：每一步把整个序列重新算一遍（第 17 章用 KV cache 加速，并讨论温度、top-k、top-p）。"""
    ids = V.encode(prompt)
    for _ in range(n):
        x = torch.tensor([ids[-T:]], device=DEV)                   # 不超过训练时的长度
        p = (m(x)[0][0, -1] / temp).softmax(-1)
        p[:4] = 0                                                  # 不生成特殊符号
        ids.append(int(torch.multinomial(p / p.sum(), 1, generator=g)))
    return V.decode(ids)


def braces_ok(body):
    depth = 0
    for ch in body:
        depth += (ch == "{") - (ch == "}")
        if depth < 0:
            return False
    return depth == 0


def part4_samples():
    section("4. 生成：和第 14 章的 n 元模型对比")
    m = load()
    g = torch.Generator(device=DEV).manual_seed(0)
    for prompt in ("# ", "| Name |", "The "):
        txt = sample(m, prompt, 200, g)
        print(f"  提示 {prompt!r}：")
        print("\n".join("    | " + l for l in txt.split("\n")[:6]))
    bad, total = 0, 0
    t = time.perf_counter()
    for _ in range(100):
        txt = sample(m, "$$", 80, g)
        if txt.count("$$") >= 2:
            total += 1
            bad += not braces_ok(txt.split("$$")[1])
    print(f"  生成 100 个以 $$ 开头的片段（{time.perf_counter() - t:.0f} 秒），{total} 个有结尾的 $$，花括号不配对 {bad} 个（{bad / max(1, total):.0%}）；"
          "第 14 章的 4 元模型是 56%")


# ---------------------------------------------------------------------- 5
def part5_ablation():
    section("5. 对照实验：去掉位置信息、不共享权重、更深 / 更宽（各 800 步）")
    tr, va = data()
    runs = [("基线（RoPE、共享权重）", CFG),
            ("不加位置信息", GPTConfig(**{**CFG.__dict__, "rope": False})),
            ("输出层不共享权重", GPTConfig(**{**CFG.__dict__, "tie_weights": False})),
            ("2 层 × d 352", GPTConfig(**{**CFG.__dict__, "n_layer": 2, "d_model": 352})),
            ("8 层 × d 176", GPTConfig(**{**CFG.__dict__, "n_layer": 8, "d_model": 176}))]
    for name, cfg in runs:
        torch.manual_seed(0)
        m = GPT(cfg).to(DEV)
        h = train_lm(m, tr, va, 800, B=B, T=T, eval_every=800, device=DEV, log=None)
        print(f"    {name:22s} 参数 {m.n_params():9,}  验证 {h[-1][2] / math.log(2):.3f} bits/字符  （{h[-1][3]:.0f} 秒）")
    print("  不加位置信息时，因果 mask 仍让模型间接知道一点顺序（位置 t 能看到 t+1 个 token，平均值的\"稀释程度\"随位置变化），")
    print("  但\"上一个字符是什么\"这种最基本的信息都很难得到，损失明显更高")


# ---------------------------------------------------------------------- 6
def part6_attention():
    section("6. 注意力在看哪里：预测 } 时，有没有头看向配对的 {")
    m = load()
    data()
    va_text = _C["va_text"]
    lines = [l for l in va_text.split("\n") if l.startswith("$$") and l.count("{") >= 3][:200]
    weights = {}
    hooks = []
    for i, b in enumerate(m.blocks):
        def hook(mod, inp, out, i=i):
            x, cos, sin = inp[0], inp[1], inp[2]
            B_, T_, d = x.shape
            q, k, _ = mod.qkv(x).view(B_, T_, 3, mod.h, mod.dh).permute(2, 0, 3, 1, 4)
            from miniocr.models.gpt import apply_rope
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            s = q @ k.transpose(-1, -2) / math.sqrt(mod.dh)
            s = s.masked_fill(~torch.ones(T_, T_, dtype=torch.bool, device=x.device).tril(), float("-inf"))
            weights[i] = s.softmax(-1)[0]
        hooks.append(b.attn.register_forward_hook(hook))
    near = torch.zeros(CFG.n_layer, CFG.n_head)            # 所有"下一个是 }"的位置：配对的 { 的权重
    m_hit = torch.zeros(CFG.n_layer, CFG.n_head)           # 嵌套的情况：配对（未关闭）的 {
    m_dec = torch.zeros(CFG.n_layer, CFG.n_head)           # 嵌套的情况：更近的、已经关闭的 {
    n_all, n_nest, dist = 0, 0, []
    with torch.no_grad():
        for line in lines:
            ids = torch.tensor([V.encode(line)], device=DEV)
            m(ids)
            stack, closed = [], []
            for t, ch in enumerate(line[:-1]):
                if ch == "{":
                    stack.append(t)
                elif ch == "}" and stack:
                    closed.append(stack.pop())
                if line[t + 1] != "}" or not stack:
                    continue
                j = stack[-1]                                   # 要预测的 } 配对的 { 在位置 j
                for L in range(CFG.n_layer):
                    near[L] += weights[L][:, t, j].cpu()
                n_all += 1
                dist.append(t - j)
                dec = [c for c in closed if c > j]              # 在 j 之后、已经关闭的 {（离 t 更近）
                if dec:
                    for L in range(CFG.n_layer):
                        m_hit[L] += weights[L][:, t, j].cpu()
                        m_dec[L] += weights[L][:, t, max(dec)].cpu()
                    n_nest += 1
    for h in hooks:
        h.remove()
    print(f"  {len(lines)} 个验证公式里，下一个字符是 }} 的位置共 {n_all} 个；配对的 {{ 平均在 {np.mean(dist):.1f} 个字符之前"
          f"（{np.mean(np.array(dist) <= 2):.0%} 的情况就在 1–2 个字符之前）")
    print("  所有位置：每个头分给配对的 { 的平均权重（离得近时，\"看最近几个字符\"的头也会得高分，不能说明问题）：")
    for L in range(CFG.n_layer):
        print(f"    第 {L + 1} 层  " + "  ".join(f"头{h} {near[L, h] / n_all:.2f}" for h in range(CFG.n_head)))
    print(f"  真正的考验：嵌套的 {n_nest} 个位置，例如 \\frac{{a^{{2}}}} 预测第二个 }} 时，配对的是 \\frac 的 {{，")
    print("  而 ^{2} 的 { 更近、但已经关闭。每个头分给\"配对的 {\" / \"更近但已关闭的 {\"的平均权重：")
    for L in range(CFG.n_layer):
        print(f"    第 {L + 1} 层  " + "  ".join(f"头{h} {m_hit[L, h] / max(1, n_nest):.2f}/{m_dec[L, h] / max(1, n_nest):.2f}"
                                             for h in range(CFG.n_head)))
    print("  如果某个头在嵌套情况下仍然把权重放在配对的 { 上、几乎不看已关闭的 {，它就真的在\"找还没关上的括号\"。")
    print("  n 元模型只能看到前 4 个字符，做不到这一点")


PARTS = {1: part1_params, 2: part2_init, 3: part3_pretrain, 4: part4_samples, 5: part5_ablation, 6: part6_attention}

if __name__ == "__main__":
    for k in [int(a) for a in sys.argv[1:] if a.isdigit()] or list(PARTS):
        PARTS[k]()
    print("\n第 16 章的演示全部完成。")
