"""第 15 章：注意力。全部在 CPU 上运行，约 3 分钟。

    python -m labs.ch15.attn_tour            # 全部 6 个部分
    python -m labs.ch15.attn_tour 4          # 只运行第 4 部分
"""
import math
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # 本章只用 CPU；连 CUDA 都不初始化（GPU 被别的程序占满时，Adam 的检查也会触发 CUDA 初始化而报错）

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from miniocr.nn.attention import LookupNet, MultiHeadAttention, attention, causal_mask

OUT = "outputs/ch15"
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(min(8, os.cpu_count() or 1))


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ---------------------------------------------------------------------- 1
def part1_hand():
    section("1. 手算：一个查询、三个键")
    q = torch.tensor([[1.0, 0.0]])
    k = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    v = torch.tensor([[10.0], [20.0], [30.0]])
    print("  q = [1, 0]；k1 = [1, 0], k2 = [0, 1], k3 = [1, 1]；v = 10, 20, 30")
    s = (q @ k.T)[0]
    print(f"  点积 q·k = {s.tolist()}；除以 √2：{[round(x, 4) for x in (s / math.sqrt(2)).tolist()]}")
    e = (s / math.sqrt(2)).exp()
    print(f"  e^score = {[round(x, 4) for x in e.tolist()]}，和 {e.sum():.4f}；权重 = {[round(x, 4) for x in (e / e.sum()).tolist()]}")
    out, w = attention(q, k, v)
    print(f"  输出 = Σ 权重 × v = {out.item():.4f}（attention() 的结果）")
    print("  和 q 方向一致的键（k1、k3）得到更大的权重；k2 与 q 正交，点积 0，权重最小但不是 0")
    print("  如果不除以 √2：权重 =", [round(x, 4) for x in (q @ k.T)[0].softmax(0).tolist()])


# ---------------------------------------------------------------------- 2
def part2_scale():
    section("2. 为什么除以 √d：点积的方差随维度增长")
    g = torch.Generator().manual_seed(0)
    print(f"  {'d':>5s} {'q·k 标准差':>10s} │ {'不缩放：最大权重':>14s} {'熵':>6s} {'q 的梯度':>9s} │ {'缩放：最大权重':>12s} {'熵':>6s} {'q 的梯度':>9s}")
    for d in (4, 16, 64, 256, 1024):
        q = torch.randn(1000, 1, d, generator=g)
        k = torch.randn(1000, 16, d, generator=g)
        up = torch.randn(1000, 1, 16, generator=g)                  # 随机的"下游梯度"，模拟损失对权重的导数
        line = f"  {d:5d} {(q @ k.transpose(1, 2)).std():10.2f} │"
        for sc in (1.0, math.sqrt(d)):
            qq = q.clone().requires_grad_()
            p = (qq @ k.transpose(1, 2) / sc).softmax(-1)
            (p * up).sum().backward()
            ent = -(p * p.clamp_min(1e-30).log()).sum(-1).mean()
            line += f" {p.max(-1).values.mean():14.3f} {ent:6.3f} {qq.grad.norm(dim=-1).median():9.2e} │"
        print(line)
    print(f"  参照：16 个键均匀分布时熵 = ln 16 = {math.log(16):.3f}；q 的梯度取 1000 个查询的中位数")
    print("  q、k 的每一维是独立的 N(0, 1) 时，q·k = Σ q_i k_i 的方差是 d，标准差 √d。")
    print("  不缩放：d 越大，分数的差距越大，softmax 越接近 one-hot（熵 → 0），饱和的 softmax 导数几乎为 0，")
    print("  梯度传不回 q 和 k（第 11 章：sigmoid / softmax 饱和）。除以 √d 后分数的标准差回到 1，各个 d 下的行为一样")


# ---------------------------------------------------------------------- 3
def part3_mask_perm():
    section("3. 因果 mask 与\"注意力不知道顺序\"")
    torch.manual_seed(0)
    T = 5
    x = torch.randn(1, T, 8)
    m = MultiHeadAttention(8, 1, causal=True)
    y = m(x)
    print("  因果自注意力的权重（第 i 行：位置 i 看各位置的权重）：")
    for row in m.last_weights[0, 0]:
        print("    " + "  ".join(f"{v:.2f}" for v in row.tolist()))
    x2 = x.clone(); x2[:, 3:] += torch.randn(1, 2, 8)
    d = (m(x2) - y).abs().amax(-1)[0]
    print(f"  改动位置 3、4 的输入后，各位置输出的最大变化：{[f'{v:.1e}' for v in d.tolist()]}")
    print("  前 3 个位置完全不变：预测第 t+1 个 token 时，模型看不到它和它之后的 token（第 16 章训练语言模型的前提）")
    m2 = MultiHeadAttention(8, 2)
    perm = torch.tensor([4, 2, 0, 3, 1])
    y1, y2 = m2(x)[:, perm], m2(x[:, perm])
    print(f"  非因果自注意力：先打乱输入再计算 vs 先计算再打乱输出，最大差 {(y1 - y2).abs().max():.1e}")
    print("  也就是说，自注意力把输入当作一个\"集合\"：\"猫追狗\"和\"狗追猫\"得到的是同一组向量（只是顺序变了）。")
    print("  要让模型知道顺序，必须把位置信息加进输入：位置嵌入（本章）或 RoPE（第 16 章）")


# ---------------------------------------------------------------------- 4
KEYS, VALS = "abcdefgh", "0123456789"
N_PAIRS = 6
# 合并格式：每一对"键值"是一个 token（a3 这样的 80 种），然后是分隔符 |，最后是要查的键（8 种）
PAIR_ITOS = [k + v for k in KEYS for v in VALS] + ["|"] + [f"?{k}" for k in KEYS]
T_PAIR = N_PAIRS + 2
# 分开格式：键和值各是一个 token
ITOS = list(KEYS) + list(VALS) + ["|"]
T = 2 * N_PAIRS + 2


def _sample(n, g):
    keys = torch.argsort(torch.rand(n, len(KEYS), generator=g), dim=1)[:, :N_PAIRS]     # 键互不相同
    vals = torch.randint(0, len(VALS), (n, N_PAIRS), generator=g)                       # 值可以重复
    qi = torch.randint(0, N_PAIRS, (n,), generator=g)
    return keys, vals, qi


def pair_batch(n, g):
    """a3 c7 h0 b3 f1 e9 | ?c → 7"""
    keys, vals, qi = _sample(n, g)
    x = torch.empty(n, T_PAIR, dtype=torch.long)
    x[:, :N_PAIRS] = keys * len(VALS) + vals
    x[:, -2] = len(KEYS) * len(VALS)
    x[:, -1] = len(KEYS) * len(VALS) + 1 + keys[torch.arange(n), qi]
    return x, vals[torch.arange(n), qi]


def lookup_batch(n, g):
    """a 3 c 7 h 0 b 3 f 1 e 9 | c → 7"""
    keys, vals, qi = _sample(n, g)
    x = torch.empty(n, T, dtype=torch.long)
    x[:, 0:2 * N_PAIRS:2] = keys
    x[:, 1:2 * N_PAIRS:2] = vals + len(KEYS)
    x[:, -2] = len(ITOS) - 1
    x[:, -1] = keys[torch.arange(n), qi]
    return x, vals[torch.arange(n), qi]


class WindowMLP(nn.Module):
    """只看最后 w 个 token 的 MLP：相当于第 14 章的 (w+1) 元语法，但用神经网络估计。w = 序列长时看全部输入。"""

    def __init__(self, vocab, w, d=64, hidden=512):
        super().__init__()
        self.w = w
        self.emb = nn.Embedding(vocab, d)
        self.net = nn.Sequential(nn.Linear(w * d, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, len(VALS)))

    def forward(self, idx):
        return self.net(self.emb(idx[:, -self.w:]).flatten(1))


def fit_lookup(model, batch_fn, steps=3000, n_train=None, bs=256, lr=2e-3, seed=0):
    g = torch.Generator().manual_seed(seed)
    fixed = batch_fn(n_train, g) if n_train else None
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    for step in range(steps):
        if fixed is None:
            x, y = batch_fn(bs, g)
        else:
            i = torch.randint(0, n_train, (bs,), generator=g)
            x, y = fixed[0][i], fixed[1][i]
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval()
    with torch.no_grad():
        xt, yt = batch_fn(5000, torch.Generator().manual_seed(999))
        acc = (model(xt).argmax(-1) == yt).float().mean().item()
    model.train()
    return acc


def part4_lookup():
    section("4. 查表任务：n 元模型做不到，一层注意力就能做到")
    g = torch.Generator().manual_seed(1)
    x, y = pair_batch(3, g)
    for i in range(3):
        print(f"    {' '.join(PAIR_ITOS[t] for t in x[i])}   → {VALS[y[i]]}")
    Vp = len(PAIR_ITOS)
    print(f"  序列长 {T_PAIR}：6 个\"键值\"token（a3 表示键 a 的值是 3），分隔符 |，最后是要查的键。随机猜 10%")
    print("  训练 3000 步，每步 256 个新生成的样本；在 5000 个测试样本上的正确率：")
    torch.manual_seed(0)
    models = {}
    for name, model in (("MLP，只看最后 3 个 token", WindowMLP(Vp, 3)),
                        ("MLP，看全部 8 个 token", WindowMLP(Vp, T_PAIR)),
                        ("注意力 1 层", LookupNet(Vp, T_PAIR, len(VALS), n_layers=1)),
                        ("注意力 1 层，无位置嵌入", LookupNet(Vp, T_PAIR, len(VALS), n_layers=1, pos=False))):
        t = time.perf_counter()
        acc = fit_lookup(model, pair_batch)
        models[name] = model
        print(f"    {name:24s} 参数 {sum(p.numel() for p in model.parameters()):7,}  正确率 {acc:6.1%}  （{time.perf_counter() - t:.0f} 秒）")
    print("  只看最后 3 个 token（第 6 对、|、查询）：只有查的恰好是第 6 个键时才能确定答案，上限约 1/6 + 5/6 × 1/10 = 25%")
    print("  这个任务与顺序无关（把 6 对打乱，答案不变），所以不加位置嵌入也行——第 3 部分说的\"集合\"性质在这里正好合适")
    explain_pair(models["注意力 1 层"])
    print("  数据量：只有 2000 个固定的训练样本（反复使用）时：")
    torch.manual_seed(0)
    for name, model in (("MLP，看全部 8 个 token", WindowMLP(Vp, T_PAIR)), ("注意力 1 层", LookupNet(Vp, T_PAIR, len(VALS), n_layers=1))):
        print(f"    {name:24s} 正确率 {fit_lookup(model, pair_batch, n_train=2000):6.1%}")
    print("  MLP 要为\"答案在第 1 个位置、第 2 个位置……\"分别学一套规则，并且要见过各种键值组合；")
    print("  注意力用同一套\"匹配 → 取值\"的规则处理所有位置：这是第 12 章卷积\"权重共享\"思想在序列上的对应物")
    print()
    print("  分开格式：键和值各是一个 token（a 3 c 7 … | c → 7），注意力 2 层，训练 3000 步：")
    torch.manual_seed(0)
    for name, model in (("MLP，看全部 14 个 token", WindowMLP(len(ITOS), T)), ("注意力 2 层", LookupNet(len(ITOS), T, len(VALS), n_layers=2))):
        print(f"    {name:24s} 正确率 {fit_lookup(model, lookup_batch):6.1%}")
    print("  注意力卡在约 33%：它学会了\"答案是序列里出现过的某个值\"，却没学会是哪一个。")
    print("  要解这个格式，第 1 层得先让每个值位置把前面的键复制过来，第 2 层再用查询去匹配；")
    print("  两层只有同时成形才有用，单独一层的改进几乎不降低损失，于是训练长时间停在平台上。")
    print("  我们试过 2 万步、不同学习率、因果 mask、单头，都没有跳出来（正文 15.7 节）")


def explain_pair(model):
    """一层注意力在查询位置上把权重放在哪里。"""
    model.eval()
    xs, ys = pair_batch(1000, torch.Generator().manual_seed(5))
    with torch.no_grad():
        model(xs)
    w = model.blocks[0].attn.last_weights[:, :, -1, :N_PAIRS]      # (1000, h, 6)：查询位置看 6 个键值 token
    qk = xs[:, -1] - (len(KEYS) * len(VALS) + 1)
    match = (xs[:, :N_PAIRS] // len(VALS) == qk[:, None]).float().argmax(1)
    hit = w[torch.arange(1000), :, match].mean(0)
    print("  查询位置分给\"键相同的那一对\"的平均权重（6 对均匀时是 0.167）：" + "  ".join(f"头{h} {p:.2f}" for h, p in enumerate(hit.tolist())))
    h = int(hit.argmax())
    lab = [PAIR_ITOS[t] for t in xs[0]]
    print(f"    第一个样本 {' '.join(lab)} → {VALS[ys[0]]}；头{h} 在 6 对上的权重 " + " ".join(f"{v:.2f}" for v in w[0, h].tolist()))
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.bar(range(N_PAIRS), w[0, h].numpy()); ax.set_xticks(range(N_PAIRS)); ax.set_xticklabels(lab[:N_PAIRS])
    ax.set_title(f"query {lab[-1]}: attention of head {h}"); fig.tight_layout()
    fig.savefig(f"{OUT}/lookup_attention.png", dpi=100); plt.close(fig)
    print(f"    图存到 {OUT}/lookup_attention.png")
    model.train()


# ---------------------------------------------------------------------- 5
def part5_cost():
    section("5. 代价：注意力的计算量和内存与长度的平方成正比")
    d = 64
    print(f"  单头，d = {d}，CPU：")
    base = None
    for T_ in (256, 512, 1024, 2048, 4096):
        x = torch.randn(1, T_, d)
        attention(x, x, x)
        t = time.perf_counter()
        for _ in range(3):
            attention(x, x, x)
        dt = (time.perf_counter() - t) / 3
        base = base or dt
        print(f"    T = {T_:5d}  用时 {dt * 1e3:8.2f} ms（{dt / base:5.1f} 倍）  权重矩阵 {T_ * T_ * 4 / 2**20:7.1f} MB（float32）")
    print("  T 翻倍，权重矩阵变成 4 倍，计算量也是 4 倍。实测的倍数并不总是 4：短序列时固定开销占比大，倍数偏小；")
    print("  权重矩阵大到放不进 CPU 缓存（几 MB 到几十 MB）之后，要反复读写内存，倍数会突然变大。")
    print("  内存访问而不是计算成为瓶颈——GPU 上同样如此，这正是 FlashAttention 要解决的问题")
    print("  一页文档的视觉 token 可能有几千个（第 24 章），这就是 VLM 要压缩视觉 token、")
    print("  以及 FlashAttention 这类\"不把整个权重矩阵存下来\"的实现存在的原因（第 18 章）")
    for d_, h in ((512, 8), (1024, 16)):
        m = MultiHeadAttention(d_, h)
        n = sum(p.numel() for p in m.parameters())
        print(f"  多头注意力 d = {d_}, {h} 个头：参数 {n:,} = 4 × ({d_}² + {d_})；与头数无关，头数只决定怎样切分 d 维")


# ---------------------------------------------------------------------- 6
def part6_cross():
    section("6. 交叉注意力：查询来自文字，键和值来自图像")
    torch.manual_seed(0)
    B, Tt, Ti, d = 2, 7, 64, 32
    text = torch.randn(B, Tt, d)                                   # 7 个文字 token
    image = torch.randn(B, Ti, d)                                  # 64 个图像特征（比如第 13 章 CNN 输出的 64 列）
    m = MultiHeadAttention(d, 4)
    y = m(text, kv=image)
    print(f"  文字 {tuple(text.shape)}，图像特征 {tuple(image.shape)} → 输出 {tuple(y.shape)}；权重 {tuple(m.last_weights.shape)} = (B, 头, 文字长度, 图像长度)")
    print("  每个文字位置得到一个\"从图像里取回的\"向量。这是 2017–2020 年注意力式 OCR 解码器的做法，")
    print("  也是 Flamingo 一类 VLM 连接视觉和语言的方式（第 22 章）。")
    print("  另一种做法（LLaVA、Qwen-VL）不用交叉注意力：把图像特征直接拼在文字前面，一起做因果自注意力（第 22 章）")
    joint = torch.cat([image, text], 1)
    print(f"  拼接后序列长 {joint.shape[1]}，自注意力的权重矩阵 {joint.shape[1]}×{joint.shape[1]}；交叉注意力只需 {Tt}×{Ti}")


PARTS = {1: part1_hand, 2: part2_scale, 3: part3_mask_perm, 4: part4_lookup, 5: part5_cost, 6: part6_cross}

if __name__ == "__main__":
    for k in [int(a) for a in sys.argv[1:] if a.isdigit()] or list(PARTS):
        PARTS[k]()
    print("\n第 15 章的演示全部完成。")
