"""第 13 章：文字行识别（CRNN + CTC）。逐段运行，对照正文阅读。

    python -m labs.ch13.ctc_tour             # 全部 6 个部分；CPU 上约 20 分钟。训练好的模型存进 outputs/ch13，再次运行直接读取
    python -m labs.ch13.ctc_tour 2           # 只运行第 2 部分（手算 CTC，不需要 GPU）
    python -m labs.ch13.ctc_tour --cpu       # 强制用 CPU；加 --retrain 重新训练
"""
import math
import os
import sys
import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from miniocr.ctc import collapse, ctc_brute_force, ctc_forward, greedy_decode, prefix_beam_search
from miniocr.data.render import render_line, save_grid
from miniocr.data.text import LineDataset, font_width
from miniocr.metrics import align_ops, cer, exact_match
from miniocr.models.crnn import CRNN, CTCCollate, ctc_batch_loss, ctc_decode, line_vocab
from miniocr.torch_basics import pick_device

DEV = torch.device("cpu") if "--cpu" in sys.argv else pick_device()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
OUT = "outputs/ch13"
os.makedirs(OUT, exist_ok=True)
V = line_vocab()
_DATA, _MODELS = {}, {}


AUG = dict(noise=0.15, distort=1.0)          # 加噪声、模糊、倾斜：干净的渲染图太容易，两种模型都是 0 错误


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def data():
    if not _DATA:
        t = time.perf_counter()
        _DATA["tr"], _DATA["va"], _DATA["te"] = (LineDataset(0, 20000, **AUG), LineDataset(1, 1000, **AUG), LineDataset(2, 2000, **AUG))
        _DATA["sec"] = time.perf_counter() - t
    return _DATA["tr"], _DATA["va"], _DATA["te"]


def run_eval(model, ds, beam=0):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _, _, _ in DataLoader(ds, batch_size=64, collate_fn=CTCCollate(V)):
            lp = model(xb.to(DEV))
            if beam:
                a = lp.cpu().numpy()
                preds += [V.decode(prefix_beam_search(a[:, b], beam=beam)) for b in range(a.shape[1])]
            else:
                preds += ctc_decode(lp, V)
    return preds


def train(name, rnn="lstm", epochs=10, log=True):
    if name in _MODELS:
        m = _MODELS[name][0].to(DEV)
        return (m,) + _MODELS[name][1:]
    for m, *_ in _MODELS.values():
        m.cpu()
    tr, va, te = data()
    torch.manual_seed(0)
    model = CRNN(len(V), rnn=rnn).to(DEV)
    path = f"{OUT}/crnn_{name}.pt"
    if os.path.exists(path) and "--retrain" not in sys.argv:          # 训练过就直接读取（加 --retrain 重新训练）
        ck = torch.load(path, map_location=DEV)
        model.load_state_dict(ck["model"])
        _MODELS[name] = (model, ck["hist"], ck["sec"])
        return _MODELS[name]
    steps = epochs * math.ceil(len(tr) / 32)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.05)
    loader = DataLoader(tr, batch_size=32, shuffle=True, collate_fn=CTCCollate(V), generator=torch.Generator().manual_seed(0))
    hist, t0 = [], time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for xb, tg, ln, _ in loader:
            loss = ctc_batch_loss(model(xb.to(DEV)), tg, ln)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(ln); n += len(ln)
        preds = run_eval(model, va)
        hist.append((ep, tot / n, cer(preds, va.texts), exact_match(preds, va.texts)))
        if log:
            print(f"    第 {ep} 轮  训练损失 {tot / n:.4f}  验证 CER {hist[-1][2]:6.2%}  整行全对 {hist[-1][3]:6.1%}")
    _MODELS[name] = (model, hist, time.perf_counter() - t0)
    torch.save({"model": model.state_dict(), "hist": hist, "sec": _MODELS[name][2]}, path)
    return _MODELS[name]


# ---------------------------------------------------------------------- 1
def part1_alignment():
    section("1. 序列识别为什么难：字符占的宽度不固定")
    rng = np.random.default_rng(3)
    text = "Will 1100 mil fill it"
    img, boxes = render_line(text, rng, size=19)
    save_grid([img], f"{OUT}/line_example.png", scale=4, cols=1)
    print(f"  文字 {text!r}（存到 {OUT}/line_example.png）")
    print("  每个字符的墨迹框（x0, x1）和宽度：")
    print("   " + "  ".join(f"{c}:{b[0]}-{b[2]}" for c, b in zip(text, boxes) if b))
    ws = {c: font_width(c) for c in "ilm1W.0"}
    print("  19 号字的前进宽度（像素）：" + "  ".join(f"{c} {w:.1f}" for c, w in ws.items()))
    print(f"  CRNN 每 4 个像素输出一个时间步：图宽 256 → T = 64 步；这一行有 {len(text)} 个字符。"
          "哪几步对应哪个字符？——训练数据里没有这个信息")


# ---------------------------------------------------------------------- 2
def part2_hand():
    section("2. 手算 CTC：T = 3 步，类别 {空白 ε, a, b}，标签 \"ab\"")
    P = np.array([[0.5, 0.4, 0.1],
                  [0.3, 0.3, 0.4],
                  [0.6, 0.1, 0.3]])
    print("  每步的概率（列：ε, a, b）：\n" + "\n".join(f"    t={t}: {row.tolist()}" for t, row in enumerate(P)))
    total, paths = ctc_brute_force(P, [1, 2])
    names = "εab"
    for path, p in paths:
        print(f"    路径 {''.join(names[k] for k in path)}  概率 {' × '.join(f'{P[t, k]}' for t, k in enumerate(path))} = {p:.4f}")
    print(f"  共 {len(paths)} 条路径（27 条中）折叠后是 \"ab\"，总概率 {total:.4f}，CTC 损失 −ln {total:.4f} = {-math.log(total):.4f}")
    lp = np.log(P)
    print(f"  前向算法：{ctc_forward(lp, [1, 2]):.4f}   F.ctc_loss：", end="")
    print(f"{F.ctc_loss(torch.from_numpy(lp)[:, None], torch.tensor([[1, 2]]), torch.tensor([3]), torch.tensor([2]), reduction='sum').item():.4f}")
    t2, p2 = ctc_brute_force(P, [1, 1])
    print(f"  标签 \"aa\"：只有 {len(p2)} 条路径（{', '.join(''.join(names[k] for k in p) for p, _ in p2)}），概率 {t2:.4f}"
          "——两个 a 之间必须隔一个空白")
    print("  贪心解码：每步取最大 → " + "".join(names[k] for k in P.argmax(1)) + f" → 折叠后 \"{''.join(names[k] for k in collapse(P.argmax(1)))}\"")
    Q = np.array([[0.6, 0.4], [0.6, 0.4]])
    print(f"  贪心不等于最优：两步、类别 {{ε, a}}，每步 p(ε)=0.6、p(a)=0.4。"
          f"贪心路径 εε → \"\"（概率 0.36）；\"a\" 的三条路径 aε、εa、aa 之和 = {0.4 * 0.6 * 2 + 0.16:.2f}；"
          f"束搜索给出 {''.join(names[k] for k in prefix_beam_search(np.log(Q), beam=4)) or '空'}")


# ---------------------------------------------------------------------- 3
def part3_train():
    section("3. 训练 CRNN：2 万行（加噪声和形变），AdamW + OneCycle，批大小 32，10 轮")
    tr, va, te = data()
    print(f"  训练 {len(tr)} 行  验证 {len(va)} 行  测试 {len(te)} 行（渲染用时 {_DATA['sec']:.1f} 秒）；"
          f"平均 {np.mean([len(s) for s in tr.texts]):.1f} 个字符/行；字符表 {len(V)} 类（含空白）")
    torch.manual_seed(0)
    m0 = CRNN(len(V)).to(DEV)
    xb, tg, ln, _ = next(iter(DataLoader(tr, batch_size=32, collate_fn=CTCCollate(V))))
    with torch.no_grad():
        l0 = ctc_batch_loss(m0(xb.to(DEV)), tg, ln).item()
    print(f"  参数 {sum(p.numel() for p in m0.parameters()):,}；初始损失 {l0:.3f}（按标签长度平均；"
          f"粗略估计：每步均匀分布时 −ln p(一条路径) / U ≈ 64·ln {len(V)} / U ≈ {64 * math.log(len(V)) / np.mean(ln.numpy().astype(float)):.1f}，"
          "再减去 ln(路径条数)/U）")
    del m0
    model, hist, dt = train("lstm")
    print(f"  用时 {dt:.0f} 秒")
    preds = run_eval(model, te)
    print(f"  测试集：CER {cer(preds, te.texts):.2%}  整行全对 {exact_match(preds, te.texts):.1%}")
    wrong = [(p, r) for p, r in zip(preds, te.texts) if p != r]
    print(f"  错误的例子（预测 | 正确），共 {len(wrong)} 行：")
    for p, r in wrong[:8]:
        print(f"    {p!r:32s} | {r!r}")
    ops = Counter()
    for p, r in zip(preds, te.texts):
        for op, a, b in align_ops(r, p):
            if op != "=":
                ops[(op, a, b)] += 1
    print("  最常见的编辑操作（S 替换 正确→预测，D 漏掉，I 多出）：" +
          "  ".join(f"{op}{a!r}→{b!r}×{n}" if op == "S" else f"{op}{(a or b)!r}×{n}" for (op, a, b), n in ops.most_common(10)))
    t = time.perf_counter(); pb = run_eval(model, te, beam=8); tb = time.perf_counter() - t
    print(f"  前缀束搜索（beam=8）：CER {cer(pb, te.texts):.2%}  整行全对 {exact_match(pb, te.texts):.1%}  "
          f"与贪心结果不同的行 {sum(a != b for a, b in zip(pb, preds))} 行；用时 {tb:.1f} 秒（Python 实现，很慢）")


# ---------------------------------------------------------------------- 4
def part4_peaky():
    section("4. 模型自己学会了对齐：每个时间步的输出")
    model = train("lstm")[0]
    rng = np.random.default_rng(7)
    text = "Five 2024 rocks"
    img, boxes = render_line(text, rng, size=19, noise=AUG["noise"])     # 与训练数据同样的噪声（不形变，框才准确）
    x = torch.from_numpy((255 - img.astype(np.float32)) / 255)[None, None].to(DEV)
    model.eval()
    with torch.no_grad():
        lp = model(x)[:, 0].cpu().numpy()                     # (64, K)
    path = lp.argmax(1)
    print(f"  文字 {text!r}，64 个时间步的贪心路径（· 表示空白）：")
    print("    " + "".join("·" if k == 0 else V.itos[k] for k in path))
    nb = (path != 0).sum()
    print(f"  非空白的时间步 {nb} 个 / 64；解码结果 {V.decode(collapse(list(path)))!r}")
    p = np.exp(lp)
    print(f"  空白概率 > 0.9 的时间步占 {(p[:, 0] > 0.9).mean():.0%}；非空白步上最大概率的平均 {p[path != 0].max(1).mean():.3f}"
          "（\"尖峰\"：每个字符只在一两步上以很高的概率出现，其余全是空白）")
    print("  字符的墨迹中心（像素）与它被输出的时间步 × 4：")
    peaks = [t for t in range(64) if path[t] != 0 and (t == 0 or path[t] != path[t - 1])]
    chars = [(c, (b[0] + b[2]) / 2) for c, b in zip(text, boxes) if b]
    print("    " + "  ".join(f"{c}:{cx:.0f}/{t * 4 + 2}" for (c, cx), t in zip(chars, peaks)))
    fig, ax = plt.subplots(2, 1, figsize=(10, 4), gridspec_kw={"height_ratios": [1, 1.4]})
    ax[0].imshow(img, cmap="gray", aspect="auto", extent=(0, 256, 32, 0)); ax[0].set_yticks([])
    for k in sorted(set(path) - {0}):
        ax[1].plot(np.arange(64) * 4 + 2, p[:, k], label=V.itos[k])
    ax[1].plot(np.arange(64) * 4 + 2, p[:, 0], "k--", lw=0.8, label="blank")
    ax[1].set_xlim(0, 256); ax[1].set_xlabel("x (pixels)"); ax[1].legend(ncol=8, fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/peaky.png", dpi=100); plt.close(fig)
    print(f"  图存到 {OUT}/peaky.png")


# ---------------------------------------------------------------------- 5
def part5_context():
    section("5. 上下文有多重要：去掉 LSTM，每一列独立分类")
    tr, va, te = data()
    model, hist, dt = train("none", rnn="none", log=False)
    print(f"  无 RNN：参数 {sum(p.numel() for p in model.parameters()):,}，每轮验证 CER " + " ".join(f"{h[2]:.2%}" for h in hist))
    p_none = run_eval(model, te)
    p_lstm = run_eval(train("lstm")[0], te)
    print(f"  测试 CER：无 RNN {cer(p_none, te.texts):.2%}   双向 LSTM {cer(p_lstm, te.texts):.2%}")
    pairs = {"o/O": "oO", "s/S": "sS", "w/W": "wW", "c/C": "cC", "l/I": "lI", "0/O": "0O"}
    for name, preds in (("无 RNN", p_none), ("双向 LSTM", p_lstm)):
        cnt = Counter()
        for p, r in zip(preds, te.texts):
            for op, a, b in align_ops(r, p):
                if op == "S":
                    for k, (u, v) in pairs.items():
                        if {a, b} == {u, v}:
                            cnt[k] += 1
        print(f"    {name:8s} 形近字符对之间的替换错误：" + "  ".join(f"{k} {cnt[k]}" for k in pairs))
    n = Counter(c for t in te.texts for c in t)
    print(f"    （测试集里 o {n['o']} 个、O {n['O']} 个、s {n['s']} 个、S {n['S']} 个；第 10、12 章的单字符模型在这些字母上错得最多）")


# ---------------------------------------------------------------------- 6
def part6_limits():
    section("6. CTC 的限制")
    model = train("lstm")[0]
    rng = np.random.default_rng(11)
    for text in ("a", "abc 123", "the cat sat on a mat ok", "iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"):
        try:
            img, _ = render_line(text, rng, size=16, noise=AUG["noise"])
        except ValueError as e:
            print(f"  {text!r}: {e}")
            continue
        x = torch.from_numpy((255 - img.astype(np.float32)) / 255)[None, None].to(DEV)
        with torch.no_grad():
            lp = model(x)[:, 0].cpu().numpy()
        need = len(text) + sum(a == b for a, b in zip(text, text[1:]))
        print(f"  {len(text):2d} 个字符（至少需要 {need:2d} 步，T = 64）：预测 {V.decode(greedy_decode(lp))!r}")
    lp = np.log(np.full((5, 3), 1 / 3))
    print(f"  T = 5 步时，标签 \"aaa\" 至少需要 5 步（a ε a ε a）：损失 {ctc_forward(lp, [1, 1, 1]):.3f}；\"aaaa\" 需要 7 步：损失 {ctc_forward(lp, [1, 1, 1, 1])}")
    print("  CTC 假设各时间步的输出相互独立（给定图像），并且输出顺序与图像从左到右的顺序一致：")
    print("    它不能输出 x^{2} 这种需要\"先写后面的内容、再回头补括号\"的结构，也不能处理多行、表格（第 21、23 章）")


PARTS = {1: part1_alignment, 2: part2_hand, 3: part3_train, 4: part4_peaky, 5: part5_context, 6: part6_limits}

if __name__ == "__main__":
    print(f"PyTorch {torch.__version__}，使用设备 {DEV}")
    for k in [int(a) for a in sys.argv[1:] if a.isdigit()] or list(PARTS):
        PARTS[k]()
    print("\n第 13 章的演示全部完成。")
