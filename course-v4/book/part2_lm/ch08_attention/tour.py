"""第 8 章　注意力：配套实验脚本。正文里引用的每一个数字都来自这里。

    python -m book.part2_lm.ch08_attention.tour              # 全部 11 个部分，CPU 约 2 分钟
    python -m book.part2_lm.ch08_attention.tour 2 3          # 只运行第 2、3 部分
    python -m book.part2_lm.ch08_attention.tour 10 --device mps   # 第 10 部分在 Mac 上训练
    python -m book.part2_lm.ch08_attention.tour 11 --context 8   # LaTeX 上下文长度实验

部分与正文小节的对应：
    1  软查找（8.2）            6  多头：形状一步步变化（8.8）
    2  q、k、v 从哪里来（8.3）   7  注意力不知道顺序（8.9）
    3  完整的数值例子（8.4）      8  代价：长度的平方（8.10）
    4  为什么除以 √d（8.6）      9  常见错误（8.12）
    5  mask（8.7）              10 查表任务：MLP 与注意力（8.1，实验手册）
    11 回到 LaTeX：注意力能看到窗口外吗（8.11；CPU，含准备约 1 分钟）
"""
import argparse
import math
import os
import time
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ch08-matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from miniocr.device import device_from_argv
from miniocr.data.latex_corpus import right_positions
from miniocr.data.latex_lm import FormulaWindows, evaluate_nll
from miniocr.models.attention_lm import AttentionLM
from miniocr.nn.attention import LookupNet, MultiHeadAttention, attention, causal_mask

OUT = "outputs/book/ch08"
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(min(8, os.cpu_count() or 1))
DEVICE = device_from_argv(default="cpu")   # 只有第 10 部分的训练用它


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def fmt(t, nd=4):
    """把张量打印成嵌套列表，保留 nd 位小数。"""
    return [fmt(r, nd) for r in t] if t.dim() > 1 else [round(x, nd) for x in t.tolist()]


# ---------------------------------------------------------------------- 1
def part1_soft_lookup():
    section("1. 软查找（8.2）：从\"查字典\"到\"按相似度加权平均\"")
    names = ["猫", "狗", "老虎"]
    K = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]])
    V = torch.tensor([3.0, 5.0, 9.0])
    print("  字典里有 3 个条目：键是 2 维向量，值是一个数")
    for n, k, v in zip(names, K, V):
        print(f"    {n:4s} 键 {fmt(k, 2)}  值 {v.item():g}")
    print("  硬查找：查询必须和某个键完全相同。查询 [1, 0]（就是\"猫\"）→ 值 3；查询 [0.9, 0.1] 不在字典里 → 查不到")
    q = torch.tensor([0.9, 0.1])
    s = K @ q
    print(f"  软查找，查询 q = {fmt(q, 2)}：")
    print(f"    第 1 步 相似度（点积）q·k = {fmt(s)}")
    e = s.exp()
    print(f"    第 2 步 取指数 e^s = {fmt(e)}，和 {e.sum():.4f}")
    w = e / e.sum()
    print(f"    第 3 步 归一化（softmax）权重 = {fmt(w)}，和 {w.sum():.4f}")
    print(f"    第 4 步 加权平均 = " + " + ".join(f"{a:.4f}×{b:g}" for a, b in zip(w.tolist(), V.tolist()))
          + f" = {(w * V).sum():.4f}")
    q2 = torch.tensor([9.0, 1.0])
    w2 = (K @ q2).softmax(0)
    print(f"  把查询放大 10 倍 q = {q2.tolist()}：权重 {fmt(w2)}，结果 {(w2 * V).sum():.4f}——几乎变回硬查找")
    print("  相似度的尺度决定了查找有多\"硬\"：尺度越大，权重越集中在最相似的那一个上（8.6 节会用到）")


# ---------------------------------------------------------------------- 2-3：同一个三 token 例子
X = torch.tensor([[1.0, 0.0, 1.0, 0.0],
                  [0.0, 2.0, 0.0, 1.0],
                  [1.0, 1.0, 1.0, 1.0]])
WQ = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
WK = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
WV = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])


def part2_projections():
    section("2. q、k、v 从哪里来（8.3）：同一个输入乘以三个不同的矩阵")
    print(f"  3 个 token，每个是 4 维向量（d_model = 4），X 形状 {tuple(X.shape)}：")
    for i, x in enumerate(X):
        print(f"    x{i + 1} = {x.tolist()}")
    for name, W in (("W_Q", WQ), ("W_K", WK), ("W_V", WV)):
        print(f"  {name} 形状 {tuple(W.shape)}：" + "  ".join(str(r) for r in W.int().tolist()))
    Q, K, V = X @ WQ, X @ WK, X @ WV
    print(f"  Q = X W_Q，K = X W_K，V = X W_V，形状都是 {tuple(Q.shape)}：")
    for i in range(3):
        print(f"    token {i + 1}: q{i + 1} = {Q[i].tolist()}   k{i + 1} = {K[i].tolist()}   v{i + 1} = {V[i].tolist()}")
    print("  逐项验证 q1：x1 = [1, 0, 1, 0]，q1 = 1×W_Q第1行 + 0×第2行 + 1×第3行 + 0×第4行 = [1,0] + [1,0] = [2, 0]")
    print(f"  同一个 token 1：q1 = {Q[0].tolist()}，k1 = {K[0].tolist()}——\"我在找什么\"和\"我能被怎样找到\"可以不同")
    print(f"  如果只用一个矩阵（q = k），token 1 和自己的分数 q1·q1 = {Q[0] @ Q[0]:.0f} 总是不小于 0（向量和自己的点积 = 长度平方）；")
    print(f"  分开以后 q1·k1 = {Q[0] @ K[0]:.0f}，远低于它和别人的分数：token 1 可以大大降低对自己的注意力（第 3 部分算出权重只有 0.0114；严格为 0 要靠 mask）")


def part3_full_example():
    section("3. 完整的数值例子（8.4）：softmax(Q Kᵀ / √d) V")
    Q, K, V = X @ WQ, X @ WK, X @ WV
    d = Q.shape[-1]
    S = Q @ K.T
    print(f"  第 1 步 分数 S = Q Kᵀ，形状 {tuple(Q.shape)} @ {tuple(K.T.shape)} = {tuple(S.shape)}；第 i 行第 j 列 = q_i · k_j")
    for i, r in enumerate(S):
        print(f"    行 {i + 1}: {r.tolist()}")
    Ss = S / math.sqrt(d)
    print(f"  第 2 步 缩放 S / √{d}（√{d} = {math.sqrt(d):.4f}）：")
    for i, r in enumerate(Ss):
        print(f"    行 {i + 1}: {fmt(r)}")
    W = Ss.softmax(-1)
    print("  第 3 步 每一行做 softmax，得到权重矩阵 A（每行和为 1）：")
    for i, r in enumerate(W):
        print(f"    行 {i + 1}: {fmt(r)}   和 {r.sum():.4f}")
    O = W @ V
    print(f"  第 4 步 输出 O = A V，形状 {tuple(W.shape)} @ {tuple(V.shape)} = {tuple(O.shape)}：")
    for i, r in enumerate(O):
        print(f"    o{i + 1} = {fmt(r)}")
    a = W[0]
    print(f"  逐项验证 o1 = {a[0]:.4f}×{V[0].tolist()} + {a[1]:.4f}×{V[1].tolist()} + {a[2]:.4f}×{V[2].tolist()} = {fmt(O[0])}")
    out, w = attention(Q, K, V)
    ref = F.scaled_dot_product_attention(Q[None], K[None], V[None])[0]
    print(f"  miniocr 的 attention() 与手算最大差 {(out - O).abs().max():.1e}；与 PyTorch 的 SDPA 最大差 {(out - ref).abs().max():.1e}")
    print("  加上因果 mask（8.7 节）以后：")
    out_c, w_c = attention(Q, K, V, causal_mask(3))
    for i, r in enumerate(w_c):
        print(f"    行 {i + 1}: {fmt(r)}")
    print(f"    o1 = {fmt(out_c[0])}（只能看自己，就等于 v1）")


# ---------------------------------------------------------------------- 4
def part4_scale():
    section("4. 为什么除以 √d（8.6）：点积的方差随维度增长")
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
    z = torch.tensor([10.0, 0.0, 0.0])
    p = z.softmax(0)
    print(f"  softmax 饱和的例子：z = {z.tolist()} → p = {fmt(p, 6)}；∂p1/∂z1 = p1(1 − p1) = {p[0] * (1 - p[0]):.6f}")


# ---------------------------------------------------------------------- 5
def part5_mask():
    section("5. mask（8.7）：因果 mask、补齐 mask、整行被 mask")
    print("  causal_mask(4)（True = 可以看）：")
    for r in causal_mask(4).int().tolist():
        print("    " + " ".join(str(v) for v in r))
    torch.manual_seed(0)
    T = 5
    x = torch.randn(1, T, 8)
    m = MultiHeadAttention(8, 1, causal=True)
    y = m(x)
    print("  因果自注意力的权重（第 i 行：位置 i 看各位置的权重）：")
    for row in m.last_weights[0, 0]:
        print("    " + "  ".join(f"{v:.2f}" for v in row.tolist()))
    x2 = x.clone(); x2[:, 3:] += torch.randn(1, 2, 8)
    dd = (m(x2) - y).abs().amax(-1)[0]
    print(f"  改动位置 4、5 的输入后，各位置输出的最大变化：{[f'{v:.1e}' for v in dd.tolist()]}")
    print("  补齐 mask：一批里 2 个句子，长度 4 和 2，短的补 2 个无意义 token")
    q, k, v = torch.randn(2, 4, 8), torch.randn(2, 4, 8), torch.randn(2, 4, 3)
    keep = torch.tensor([[True, True, True, True], [True, True, False, False]])
    out, w = attention(q, k, v, keep[:, None, :])
    print(f"    第 2 句的权重：{fmt(w[1], 3)}")
    v2 = v.clone(); v2[1, 2:] = 100.0
    print(f"    把第 2 句补齐位置的值改成 100，输出最大变化 {(attention(q, k, v2, keep[:, None, :])[0] - out).abs().max():.1e}")
    print("  整行被 mask：softmax(−∞, −∞, −∞) = 0/0")
    q, k, v = torch.randn(3, 4), torch.randn(3, 4), torch.randn(3, 2)
    mm = torch.tensor([[True, True, False], [False, False, False], [True, False, False]])
    out, w = attention(q, k, v, mm)
    print(f"    第 2 行的权重 {w[1].tolist()}，输出 {out[1].tolist()}")
    s = (q @ k.T / 2).masked_fill(~mm, -1e9).softmax(-1)
    print(f"    用 −1e9 代替 −∞：第 2 行的权重 {fmt(s[1])}（均匀分布，不是 NaN；但这一行的输出没有意义，不能进 loss）")


# ---------------------------------------------------------------------- 6
def part6_heads():
    section("6. 多头（8.8）：形状一步步变化")
    torch.manual_seed(0)
    B, T, d, h = 2, 3, 8, 2
    dh = d // h
    x = torch.randn(B, T, d)
    m = MultiHeadAttention(d, h)
    q = m.q(x)
    print(f"  x {tuple(x.shape)} = (B, T, d)，h = {h} 个头，每头 dh = {dh}")
    print(f"  q = x W_Q                         {tuple(q.shape)}")
    qv = q.view(B, T, h, dh)
    print(f"  view(B, T, h, dh)                 {tuple(qv.shape)}  连续：{qv.is_contiguous()}")
    qt = qv.transpose(1, 2)
    print(f"  transpose(1, 2)                   {tuple(qt.shape)}  连续：{qt.is_contiguous()}")
    print(f"  第 0 个样本、第 0 个 token 的 q：{fmt(q[0, 0], 3)}")
    print(f"    头 0 拿到前 {dh} 维 {fmt(qt[0, 0, 0], 3)}；头 1 拿到后 {dh} 维 {fmt(qt[0, 1, 0], 3)}")
    y = m(x)
    w = m.last_weights
    print(f"  每个头各自做注意力，权重 {tuple(w.shape)} = (B, h, T, T)；第 0 个样本：")
    for hh in range(h):
        print(f"    头 {hh}：{fmt(w[0, hh], 2)}")
    o = torch.randn(B, h, T, dh).transpose(1, 2)
    print(f"  合并：(B, h, T, dh) → transpose(1, 2) → {tuple(o.shape)}  连续：{o.is_contiguous()}")
    try:
        o.view(B, T, d)
        print("  view 成功")
    except RuntimeError as e:
        print(f"  此时 view(B, T, d) 报错：{str(e).splitlines()[0][:70]}…")
    print(f"  reshape(B, T, d) 可以：{tuple(o.reshape(B, T, d).shape)}（必要时会复制一份连续的数据）")
    for d_, h_ in ((8, 2), (8, 4), (512, 8), (768, 12)):
        n = sum(p.numel() for p in MultiHeadAttention(d_, h_).parameters())
        print(f"  参数量 d = {d_:3d}, h = {h_:2d}：{n:>9,} = 4 × ({d_}² + {d_})")


# ---------------------------------------------------------------------- 7
def part7_order():
    section("7. 注意力不知道顺序（8.9）")
    torch.manual_seed(0)
    x = torch.randn(1, 5, 8)
    m = MultiHeadAttention(8, 2)
    perm = torch.tensor([4, 2, 0, 3, 1])
    y1, y2 = m(x)[:, perm], m(x[:, perm])
    print(f"  先打乱输入再计算 vs 先计算再打乱输出，最大差 {(y1 - y2).abs().max():.1e}")
    mc = MultiHeadAttention(8, 2, causal=True)
    mc.load_state_dict(m.state_dict())
    print(f"  加上因果 mask 以后同样比较，最大差 {(mc(x)[:, perm] - mc(x[:, perm])).abs().max():.1e}（不再相等）")
    a = torch.randn(1, 2, 8)
    print(f"  \"猫追狗\"与\"狗追猫\"：两个 token 交换位置，第一个位置的输出 vs 交换后第二个位置的输出，最大差 "
          f"{(m(a)[0, 0] - m(a[:, [1, 0]])[0, 1]).abs().max():.1e}")


# ---------------------------------------------------------------------- 8
def part8_cost():
    section("8. 代价（8.10）：计算量和内存与长度的平方成正比")
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
        print(f"    T = {T_:5d}  用时 {dt * 1e3:8.2f} ms（{dt / base:5.1f} 倍）  权重矩阵 {T_ * T_ * 4 / 2**20:7.1f} MiB（float32）")
    for T_ in (1024, 4096):
        print(f"  计算量估计 T = {T_}：Q Kᵀ 与 A V 各 2·T²·d = {2 * T_ * T_ * d / 1e6:.0f} MFLOPs；投影 4 个 d×d 矩阵 "
              f"{4 * 2 * T_ * d * d / 1e6:.0f} MFLOPs")


# ---------------------------------------------------------------------- 9
def part9_pitfalls():
    section("9. 常见错误（8.12）：mask 约定相反、分数溢出、SDPA 的位置参数")
    torch.manual_seed(0)
    q, k, v = torch.randn(1, 3, 4), torch.randn(1, 3, 4), torch.randn(1, 3, 2)
    keep = torch.tensor([True, True, False])
    mine = attention(q, k, v, keep)[0]
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)
    print(f"  (a) 我们和 SDPA 的布尔 mask 都是 True = 可以看，最大差 {(mine - ref).abs().max():.1e}")
    mha = nn.MultiheadAttention(4, 1, batch_first=True)
    _, w_wrong = mha(q, k, k, key_padding_mask=keep[None], need_weights=True)
    _, w_right = mha(q, k, k, key_padding_mask=~keep[None], need_weights=True)
    print(f"      nn.MultiheadAttention 的 key_padding_mask 是 True = 忽略。传 keep：第 1 个查询的权重 "
          f"{[round(x, 3) for x in w_wrong[0, 0].tolist()]}；传 ~keep：{[round(x, 3) for x in w_right[0, 0].tolist()]}")
    q16 = torch.full((1, 2, 64), 40.0).half()
    sc = q16 @ q16.transpose(-2, -1)
    print(f"  (b) fp16 最大 65504。q·k 在 fp32 里是 {64 * 40.0 ** 2:.0f}，在 fp16 里是 {sc[0, 0, 0].item()}；"
          f"softmax 后 {sc.float().softmax(-1)[0, 0].tolist()}")
    sc2 = (q16 / 8) @ q16.transpose(-2, -1)
    print(f"      先把 q 除以 √64 = 8 再乘：{sc2[0, 0, 0].item():.0f}，在范围内")
    x = torch.randn(1, 4, 8)
    good = F.scaled_dot_product_attention(x, x, x, is_causal=True)
    bad = F.scaled_dot_product_attention(x, x, x, None, True)     # 第 5 个位置参数是 dropout_p
    print(f"  (c) SDPA(q, k, v, None, True) 把 True 当成 dropout_p = 1：输出的最大绝对值 {bad.abs().max():.1e}；"
          f"用 is_causal=True 关键字时是 {good.abs().max():.2f}")


# ---------------------------------------------------------------------- 10
KEYS, VALS = "abcdefgh", "0123456789"
N_PAIRS = 6
PAIR_ITOS = [k + v for k in KEYS for v in VALS] + ["|"] + [f"?{k}" for k in KEYS]
T_PAIR = N_PAIRS + 2
ITOS = list(KEYS) + list(VALS) + ["|"]
T_SEP = 2 * N_PAIRS + 2


def _sample(n, g):
    keys = torch.argsort(torch.rand(n, len(KEYS), generator=g), dim=1)[:, :N_PAIRS]     # 键互不相同
    vals = torch.randint(0, len(VALS), (n, N_PAIRS), generator=g)                       # 值可以重复
    qi = torch.randint(0, N_PAIRS, (n,), generator=g)
    return keys, vals, qi


def pair_batch(n, g):
    """合并格式：a3 c7 h0 b3 f1 e9 | ?c → 7"""
    keys, vals, qi = _sample(n, g)
    x = torch.empty(n, T_PAIR, dtype=torch.long)
    x[:, :N_PAIRS] = keys * len(VALS) + vals
    x[:, -2] = len(KEYS) * len(VALS)
    x[:, -1] = len(KEYS) * len(VALS) + 1 + keys[torch.arange(n), qi]
    return x, vals[torch.arange(n), qi]


def lookup_batch(n, g):
    """分开格式：a 3 c 7 h 0 b 3 f 1 e 9 | c → 7"""
    keys, vals, qi = _sample(n, g)
    x = torch.empty(n, T_SEP, dtype=torch.long)
    x[:, 0:2 * N_PAIRS:2] = keys
    x[:, 1:2 * N_PAIRS:2] = vals + len(KEYS)
    x[:, -2] = len(ITOS) - 1
    x[:, -1] = keys[torch.arange(n), qi]
    return x, vals[torch.arange(n), qi]


class WindowMLP(nn.Module):
    """只看最后 w 个 token 的 MLP：每个位置一套独立的权重。w = 序列长时看全部输入。"""

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
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    for step in range(steps):
        if fixed is None:
            x, y = batch_fn(bs, g)
        else:
            i = torch.randint(0, n_train, (bs,), generator=g)
            x, y = fixed[0][i], fixed[1][i]
        loss = F.cross_entropy(model(x.to(DEVICE)), y.to(DEVICE))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval()
    with torch.no_grad():
        xt, yt = batch_fn(5000, torch.Generator().manual_seed(999))
        acc = (model(xt.to(DEVICE)).argmax(-1).cpu() == yt).float().mean().item()
    model.train()
    return acc


def explain_pair(model, tag="online"):
    """一层注意力在查询位置上把权重放在哪里。tag 用来区分保存的图片。"""
    model.eval().cpu()
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
    fig.savefig(f"{OUT}/lookup_attention_{tag}.png", dpi=100); plt.close(fig)
    print(f"    图存到 {OUT}/lookup_attention_{tag}.png")
    model.train()


def part10_lookup():
    section("10. 查表任务（8.1）：只看局部窗口的模型做不到，一层注意力就能做到")
    g = torch.Generator().manual_seed(1)
    x, y = pair_batch(3, g)
    for i in range(3):
        print(f"    {' '.join(PAIR_ITOS[t] for t in x[i])}   → {VALS[y[i]]}")
    Vp = len(PAIR_ITOS)
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
    explain_pair(models["注意力 1 层"])
    print("  只有 2000 个固定的训练样本（反复使用）时：")
    torch.manual_seed(0)
    for name, model in (("MLP，看全部 8 个 token", WindowMLP(Vp, T_PAIR)), ("注意力 1 层", LookupNet(Vp, T_PAIR, len(VALS), n_layers=1))):
        print(f"    {name:24s} 正确率 {fit_lookup(model, pair_batch, n_train=2000):6.1%}")
        if isinstance(model, LookupNet):           # 实验一改的就是这个模型，分析它自己的注意力
            explain_pair(model, tag="fixed2000")
    print()
    print("  分开格式（实验手册的进阶实验）：键和值各是一个 token（a 3 c 7 … | c → 7），训练 3000 步：")
    torch.manual_seed(0)
    sep = {}
    for name, model in (("MLP，看全部 14 个 token", WindowMLP(len(ITOS), T_SEP)), ("注意力 2 层", LookupNet(len(ITOS), T_SEP, len(VALS), n_layers=2))):
        print(f"    {name:24s} 正确率 {fit_lookup(model, lookup_batch):6.1%}")
        sep[name] = model
    xt, yt = lookup_batch(5000, torch.Generator().manual_seed(999))
    vals = xt[:, 1:2 * N_PAIRS:2] - len(KEYS)
    counts = torch.zeros(len(xt), len(VALS)).scatter_add_(1, vals, torch.ones_like(vals, dtype=torch.float))
    mode = counts.argmax(1)
    rand_pos = vals[torch.arange(len(xt)), torch.randint(0, N_PAIRS, (len(xt),), generator=torch.Generator().manual_seed(0))]
    m = sep["注意力 2 层"].eval().cpu()
    with torch.no_grad():
        pred = m(xt).argmax(-1)
    print(f"    基线：随机选一个值的位置 {(rand_pos == yt).float().mean():6.1%}；选出现次数最多的值 {(mode == yt).float().mean():6.1%}")
    print(f"    注意力 2 层的预测与\"选出现次数最多的值\"一致的比例：{(pred == mode).float().mean():6.1%}")


# ---------------------------------------------------------------------- 11
def evaluate_attention(model, data, indices=None, batch=256):
    """兼容第 8 章入口；共享逐目标评测保持原来的求和顺序。"""
    return evaluate_nll(model, data, indices=indices, batch=batch,
                        window_logits=lambda m, x, lengths: m(x, lengths, last_only=True))


@torch.no_grad()
def diagnose_attention(model, exp, mlp):
    from book.part2_lm.ch07_lm_basics.tour import logits

    records = [r for text in exp.val_text for r in right_positions(exp.tok, text)]
    print(f"  诊断目标=完整 right token；稳定定位的配对数={len(records)}；第 7 章参考配对数=526。")
    if len(records) != 526:
        print("  配对数不同：当前本地语料或降级样例不同，不补造配对。")
    if not records:
        print("  无可定位配对，跳过概率、干预和注意力统计。")
        return
    # right 是目标；用于预测它的查询在前一个输入 token，不能把答案喂进去。
    prefixes = [ids[:target] for _, target, _, ids in records]
    lengths = torch.tensor([min(model.context, len(s) + 1) for s in prefixes])
    x = torch.full((len(records), model.context), exp.eos, dtype=torch.long)
    for i, prefix in enumerate(prefixes):
        visible = ([exp.bos] + prefix)[-model.context:]
        x[i, :len(visible)] = torch.tensor(visible)
    targets = torch.tensor([ids[t] for _, t, _, ids in records])
    p = model(x, lengths, last_only=True).softmax(-1).gather(1, targets[:, None]).flatten()
    weights = model.attn.last_weights
    for label, select in [("≤3", lambda d: d <= 3), ("4–8", lambda d: 4 <= d <= 8), (">8", lambda d: d > 8)]:
        indices = [i for i, r in enumerate(records) if select(r[0])]
        prob = f"{p[indices].mean():.4f}" if indices else "无样本"
        print(f"  距离分桶={label}；样本位置数={len(indices)}；模型=因果注意力；目标平均概率={prob}")
    distant = [i for i, r in enumerate(records) if r[0] > 8]
    if not distant:
        print("  无距离 >8 的配对，跳过干预和注意力统计。")
        return
    altered = x.clone()
    hit, uniform, visible_count = [], [], 0
    mlp_x, mlp_changed = [], []
    for i in distant:
        _, target, source, ids = records[i]
        local_source = source + 1 - max(0, target + 1 - model.context)
        if local_source >= 0:
            altered[i, local_source] = ord("x")
            hit.append(weights[i, :, lengths[i] - 1, local_source].mean().item())
            uniform.append(1.0 / lengths[i].item())
            visible_count += 1
        else:
            # 左括号在 64 窗口外时，原模型和均匀注意力给它的权重都为零。
            hit.append(0.0)
            uniform.append(0.0)
        changed = ids.copy()
        changed[source] = ord("x")
        mlp_x.append(ids[target - 8:target])
        mlp_changed.append(changed[target - 8:target])
    p_changed = model(altered, lengths, last_only=True).softmax(-1).gather(1, targets[:, None]).flatten()
    delta = (p_changed[distant] - p[distant]).abs()
    print(f"  远距离配对数={len(distant)}；左括号仍在注意力窗口内的配对数={visible_count}；窗口外配对数={len(distant)-visible_count}。")
    print("  控制口径：只把左圆括号所在的一个 BPE token 换成字节 token x，不重新分词；非零按浮点值严格 >0 统计。")
    print(f"  控制实验：模型=因果注意力；平均概率绝对变化={delta.mean():.4f}；最大变化={delta.max():.4f}；非零比例={(delta > 0).float().mean():.4f}")
    y = targets[distant, None]
    before = logits(mlp, torch.tensor(mlp_x), exp.bos).softmax(-1).gather(1, y)
    after = logits(mlp, torch.tensor(mlp_changed), exp.bos).softmax(-1).gather(1, y)
    delta_mlp = (after - before).abs()
    print(f"  控制实验：模型=第 7 章 MLP，上下文长度=8（本次重训）；平均概率绝对变化={delta_mlp.mean():.4f}；最大变化={delta_mlp.max():.4f}；非零比例={(delta_mlp > 0).float().mean():.4f}")
    print("  注意力口径：取预测 right 的前一输入位置作为查询，不能把 right 本身作为输入泄漏答案。")
    print(f"  >8 全部配对：左括号平均注意力权重={sum(hit)/len(hit):.4f}；均匀分配对照={sum(uniform)/len(uniform):.4f}；窗口外均记零。")
    if visible_count:
        print(f"  >8 且左括号可见：左括号平均注意力权重={sum(hit)/visible_count:.4f}；均匀分配 1/可见位置数的平均值={sum(uniform)/visible_count:.4f}")


def part11_latex(context=64, steps=1200):
    from book.part2_lm.ch07_lm_basics.tour import Experiment

    section("11. 回到 LaTeX（8.11）：注意力能看到窗口外吗")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        exp = Experiment(SimpleNamespace(small=False, n_train=20000, seed=0, embed_dim=64,
                                         hidden_dim=256, steps=1200))
        exp.prepare()
        train = FormulaWindows(exp.train, exp.bos, exp.eos, context)
        val = FormulaWindows(exp.val, exp.bos, exp.eos, context)
        torch.manual_seed(0)
        model = AttentionLM(len(exp.tok), exp.bos, context=context)
        print(f"  因果注意力 LM：设备=CPU；线程数=2；种子=0；上下文长度={context}；嵌入维度=64；头数=1；层数=1；参数量={sum(p.numel() for p in model.parameters())}。")
        print("  结构：token 嵌入 + 可学习的位置嵌入 + 因果自注意力 + 输出层；无残差、归一化或前馈块。")
        print("  位置嵌入是第 10 章的预告：8.9 节说明没有位置信息的注意力不知道顺序。")
        print("  每条公式独立加 BOS/EOS；逐目标滑窗，每次最多保留 context 个输入（含尚未滑出的 BOS）；右补齐不参与注意力；窗口内位置从零开始。")
        print(f"  训练目标数={len(train)}；验证目标数={len(val)}；不截断评测目标；BOS 输出概率恒零，EOS 计入 loss，字节数只计原文。")
        print(f"  训练配置：AdamW；学习率=0.0030；批量大小=128；训练步数={steps}；每步均匀抽取目标。")
        probe = torch.randperm(len(val), generator=torch.Generator().manual_seed(1234))[:4096]
        generator = torch.Generator().manual_seed(0)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        started, running = time.perf_counter(), []
        for step in range(1, steps + 1):
            x, lengths, y = train.batch(torch.randint(len(train), (128,), generator=generator))
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x, lengths, last_only=True), y)
            loss.backward()
            opt.step()
            running.append(loss.item())
            if step % max(1, steps // 4) == 0 or step == steps:
                vl = evaluate_attention(model, val, probe)
                print(f"  注意力训练步={step}；区间训练 nat/token={sum(running)/len(running):.4f}；固定子集验证 nat/token={vl:.4f}；子集目标数={len(probe)}")
                running.clear()
        print(f"  注意力训练及中途评测用时秒={time.perf_counter()-started:.4f}")
        final = evaluate_attention(model, val)
        exp.metrics(f"因果注意力，上下文长度={context}，全量验证", final)
        print("  第 7 章既有 MLP 上下文 8 基线：验证 nat/token=3.8124；验证 bits/byte=2.7493。")
        print(f"  相对既有基线 nat/token 差值={final-3.8124:+.4f}；正值表示本次注意力模型更差。")
        mlp = exp.mlp(8)
        diagnose_attention(model, exp, mlp)
        print("  边界：单次种子、一层、短训练；能受远处输入影响不等于学会括号配对，注意力权重也不是因果解释；不宣称超过 MLP 基线。")
    finally:
        torch.set_num_threads(previous_threads)


PARTS = {1: part1_soft_lookup, 2: part2_projections, 3: part3_full_example, 4: part4_scale, 5: part5_mask,
         6: part6_heads, 7: part7_order, 8: part8_cost, 9: part9_pitfalls, 10: part10_lookup,
         11: part11_latex}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="*", type=int)
    parser.add_argument("--device", default="cpu", help="仅第 10 部分；默认 CPU")
    parser.add_argument("--context", type=int, default=64, help="第 11 部分上下文长度")
    parser.add_argument("--steps", type=int, default=1200, help="第 11 部分注意力训练步数")
    args = parser.parse_args()
    if args.context < 1 or args.steps < 1 or any(k not in PARTS for k in args.parts):
        parser.error("上下文与步数须为正，部分号须为 1–11")
    started = time.perf_counter()
    for k in args.parts or PARTS:
        start = time.perf_counter()
        if k == 11:
            part11_latex(args.context, args.steps)
        else:
            PARTS[k]()
        print(f"第 {k} 部分用时秒={time.perf_counter()-start:.4f}")
    print(f"\n第 8 章所选实验完成；tour 总用时秒={time.perf_counter()-started:.4f}")
