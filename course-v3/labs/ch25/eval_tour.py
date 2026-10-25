"""第 25 章：评测。文字、公式、表格、问答的指标各有什么脾气；一个数字有多可信（置信区间、配对比较、种子）；
分桶与数据泄漏。默认只用 CPU。

    python -m labs.ch25.eval_tour             # 全部 7 个部分（CPU 上约 40 分钟，大部分时间在第 6 部分重新训练 3 个模型）
    python -m labs.ch25.eval_tour 1 2 3       # 只看指标（几秒）
    python -m labs.ch25.eval_tour --gpu       # 有空闲 GPU 时用 GPU
"""
import math
import os
import sys
import time
from collections import defaultdict

if "--gpu" not in sys.argv:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
from torch.utils.data import DataLoader

from miniocr.data.latex import gen_formula, normalize, to_latex
from miniocr.data.table import sample_table, to_html
from miniocr.eval.cdm import cdm
from miniocr.eval.stats import bootstrap_ci, mcnemar, paired_bootstrap
from miniocr.eval.teds import teds
from miniocr.metrics import anls, bleu, cer, edit_distance, exact_match, ned, vqa_accuracy, wer

DEV = "cuda" if "--gpu" in sys.argv and torch.cuda.is_available() else "cpu"
torch.set_num_threads(min(16, os.cpu_count() or 1))
OUT = "outputs/ch25"
os.makedirs(OUT, exist_ok=True)
_C = {}


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ---------------------------------------------------------------------- 1
def part1():
    section("1. 文字的指标：同一个错误，不同的指标怎么看")
    ref = "Total: $38.50 on 12-03-2024"
    cases = [("完全正确", ref),
             ("错一个字符", "Total: $38.80 on 12-03-2024"),
             ("大小写不同", "total: $38.50 on 12-03-2024"),
             ("少一个空格", "Total:$38.50 on 12-03-2024"),
             ("多一个词", "Total: $38.50 paid on 12-03-2024"),
             ("两个词换了顺序", "Total: on $38.50 12-03-2024"),
             ("漏掉后半句", "Total: $38.50"),
             ("完全无关", "the quick brown fox")]
    print(f"  参考：{ref!r}（{len(ref)} 个字符、{len(ref.split())} 个词）")
    print(f"  {'情况':<12}{'预测':<34}{'编辑距离':>6}{'CER':>8}{'NED':>7}{'WER':>7}{'BLEU-4':>8}{'BLEU-2':>8}{'全对':>5}")
    for name, p in cases:
        b4 = bleu([p.split()], [ref.split()])
        b2 = bleu([p.split()], [ref.split()], max_n=2)
        print(f"  {name:<12}{p!r:<34}{edit_distance(p, ref):>6}{cer([p], [ref]):>8.1%}{ned(p, ref):>7.3f}"
              f"{wer([p], [ref]):>7.1%}{b4:>8.3f}{b2:>8.3f}{'是' if p == ref else '否':>5}")
    print("\n  语料级 vs 逐行平均（两行：一行短的全错，一行长的全对）：")
    preds, refs = ["xx", "a long line of text here"], ["ok", "a long line of text here"]
    print(f"    语料级 CER = 总编辑距离 / 总长度 = {cer(preds, refs):.1%}；"
          f"逐行 CER 的平均 = {np.mean([cer([p], [r]) for p, r in zip(preds, refs)]):.1%}")

    print("\n  问答的指标：")
    qa = [("Paris", ["paris"]), ("Pari", ["paris"]), ("PARIS.", ["paris"]), ("The answer is Paris", ["paris"]),
          ("London", ["paris"]), ("12", ["12", "twelve"]), ("$12.00", ["12"])]
    print(f"  {'预测':<24}{'正确答案':<20}{'全对':>5}{'ANLS':>8}")
    for p, a in qa:
        print(f"  {p!r:<24}{str(a):<20}{'是' if p.lower() in a else '否':>5}{anls([p], [a]):>8.3f}")
    humans = ["red"] * 4 + ["dark red"] * 3 + ["maroon"] * 2 + ["brown"]
    print(f"  VQA 准确率（10 个人的答案：4 red、3 dark red、2 maroon、1 brown）："
          + "，".join(f"{p} {vqa_accuracy(p, humans):.2f}" for p in ("red", "dark red", "maroon", "brown", "pink")))


# ---------------------------------------------------------------------- 2
def part2():
    section("2. 公式：字符串比较 vs 归一化 vs 渲染后比较（CDM）")
    ref = "\\frac{x^{2}+1}{y_{n}}"
    cases = [("完全相同", ref),
             ("等价写法：省略花括号", "\\frac{x^2+1}{y_n}"),
             ("等价写法：\\over、加空格", "{x^{2} + 1 \\over y_{n}}"),
             ("错一个符号", "\\frac{x^{3}+1}{y_{n}}"),
             ("上标写成下标", "\\frac{x_{2}+1}{y_{n}}"),
             ("（对照）不在分式里：x^{2}+1 → x_{2}+1", None),
             ("丢了上标的结构", "\\frac{x2+1}{y_{n}}"),
             ("分式写成一行", "x^{2}+1y_{n}"),
             ("少一个花括号（无法渲染）", "\\frac{x^{2}+1}{y_{n}"),
             ("完全无关", "a=b")]
    print(f"  参考：{ref}")
    print(f"  {'情况':<26}{'原始 NED':>9}{'归一化后 NED':>13}{'CDM':>7}")
    for name, p in cases:
        if p is None:                                                   # 同样的错误，放在分式外面
            p, r0 = "x_{2}+1", "x^{2}+1"
            print(f"  {name:<26}{ned(p, r0):>9.3f}{ned(normalize(p), normalize(r0)):>13.3f}{cdm(p, r0):>7.3f}")
            continue
        try:
            n2 = f"{ned(normalize(p), normalize(ref)):.3f}"
        except ValueError:
            n2 = "无法解析"
        print(f"  {name:<26}{ned(p, ref):>9.3f}{n2:>13}{cdm(p, ref):>7.3f}")

    rng = np.random.default_rng(0)
    raw, norm_ok, c_ok = [], 0, 0
    for _ in range(300):
        tree = gen_formula(rng, level=4)
        a, b = to_latex(tree), to_latex(tree, rng)                    # 规范写法 vs 随机的等价写法
        raw.append(ned(b, a))
        norm_ok += normalize(b) == normalize(a)
        c_ok += cdm(b, a) == 1.0
    print(f"\n  300 个随机公式，各写成一种随机的等价写法：原始字符串完全相同的 {np.mean(np.array(raw) == 0):.0%}，"
          f"平均原始 NED {np.mean(raw):.3f}；归一化后相同 {norm_ok / 3:.0f}%；CDM = 1 的 {c_ok / 3:.0f}%")


# ---------------------------------------------------------------------- 3
def _perturb(rows, kind, rng):
    rows = [[dict(c) for c in r] for r in rows]
    body = list(range(1, len(rows)))
    if kind == "一格文字错一个字符":
        r = rows[int(rng.choice(body))]
        c = r[int(rng.integers(len(r)))]
        t = c["text"]
        k = int(rng.integers(len(t)))
        c["text"] = t[:k] + ("x" if t[k] != "x" else "y") + t[k + 1:]
    elif kind == "一格文字全错":
        r = rows[int(rng.choice(body))]
        r[int(rng.integers(len(r)))]["text"] = "???"
    elif kind == "两格被合并":
        r = rows[int(rng.choice(body))]
        k = int(rng.integers(len(r) - 1)) if len(r) > 1 else 0
        if len(r) > 1:
            a, b = r[k], r.pop(k + 1)
            a["text"] += " " + b["text"]
            a["colspan"] += b["colspan"]
    elif kind == "少了一行":
        rows.pop(int(rng.choice(body)))
    elif kind == "多了一列":
        for r in rows:
            r.append({"text": "0", "colspan": 1, "rowspan": 1})
    elif kind == "两行顺序颠倒":
        i = int(rng.choice(body[:-1])) if len(body) > 1 else 1
        rows[i], rows[i + 1] = rows[i + 1], rows[i]
    return rows


def part3():
    section("3. 表格：TEDS 与 TEDS-S（只比结构）")
    rng = np.random.default_rng(0)
    t = sample_table(rng, 3, 3)
    print("  一张随机表格的 HTML：", to_html(t))
    kinds = ["一格文字错一个字符", "一格文字全错", "两格被合并", "少了一行", "多了一列", "两行顺序颠倒"]
    print(f"\n  200 张随机表格（3–6 行、2–4 列），每种错误各做一次：")
    print(f"  {'错误':<18}{'TEDS':>8}{'TEDS-S':>9}{'HTML 字符串 NED':>17}")
    tables = [sample_table(rng) for _ in range(200)]
    for kind in kinds:
        s, ss, hn = [], [], []
        for tb in tables:
            p = to_html(_perturb(tb, kind, rng))
            r = to_html(tb)
            s.append(teds(p, r))
            ss.append(teds(p, r, structure_only=True))
            hn.append(ned(p, r))
        print(f"  {kind:<18}{np.mean(s):>8.3f}{np.mean(ss):>9.3f}{np.mean(hn):>17.3f}")
    print("  残缺的 HTML（只输出了前一半就停了）：", end="")
    r = to_html(tables[0])
    print(f"TEDS {teds(r[:len(r) // 2], r):.3f}")


# ---------------------------------------------------------------------- 4..7：用第 13 章的 CRNN
def crnn_preds():
    """第 13 章的测试集（2000 行）上，三种系统的预测：双向 LSTM（贪心解码）、双向 LSTM（束搜索）、无 RNN。"""
    if "preds" in _C:
        return _C["preds"]
    from labs.ch13 import ctc_tour as c13
    tr, va, te = c13.data()
    path = f"{OUT}/crnn_preds.pt"
    if os.path.exists(path):
        preds = torch.load(path)
    else:
        lstm = c13.train("lstm", log=False)[0]
        none = c13.train("none", rnn="none", log=False)[0]
        t0 = time.time()
        preds = {"LSTM 贪心": c13.run_eval(lstm, te), "LSTM 束搜索": c13.run_eval(lstm, te, beam=8),
                 "无 RNN": c13.run_eval(none, te)}
        print(f"  （生成预测用时 {time.time() - t0:.0f} 秒，存到 {path}）")
        torch.save(preds, path)
    _C["preds"] = (preds, te.texts, tr.texts)
    return _C["preds"]


def part4():
    section("4. 一个数字有多可信：bootstrap 置信区间与配对比较（第 13 章的 CRNN，测试集 2000 行）")
    preds, refs, _ = crnn_preds()
    lens = np.array([len(r) for r in refs])
    errs = {k: np.array([edit_distance(p, r) for p, r in zip(v, refs)]) for k, v in preds.items()}
    for k in preds:
        c, lo, hi = bootstrap_ci(errs[k], lens)
        a, alo, ahi = bootstrap_ci([p == r for p, r in zip(preds[k], refs)])
        print(f"  {k:<10}CER {c:6.2%}（95% 区间 {lo:.2%} – {hi:.2%}）   整行全对 {a:6.1%}（{alo:.1%} – {ahi:.1%}）")

    print("\n  两两比较（同一批 2000 行）：")
    for a, b in [("LSTM 束搜索", "LSTM 贪心"), ("无 RNN", "LSTM 贪心")]:
        d, lo, hi, frac = paired_bootstrap(errs[a], errs[b], lens)
        ca = [p == r for p, r in zip(preds[a], refs)]
        cb = [p == r for p, r in zip(preds[b], refs)]
        b10, b01, pv = mcnemar(ca, cb)
        diff_lines = sum(p != q for p, q in zip(preds[a], preds[b]))
        print(f"    {a} − {b}：CER 差 {d:+.2%}（95% 区间 {lo:+.2%} – {hi:+.2%}，差 ≤ 0 的比例 {frac:.1%}）；"
              f"输出不同的行 {diff_lines}")
        print(f"      整行全对：{a} 对而 {b} 错 {b10} 行，反过来 {b01} 行，McNemar p = {pv:.3g}")
    # 不配对的比较：把两个区间放在一起看
    c1 = bootstrap_ci(errs["LSTM 束搜索"], lens)
    c2 = bootstrap_ci(errs["LSTM 贪心"], lens)
    print(f"  不配对地看：束搜索 [{c1[1]:.2%}, {c1[2]:.2%}] 与贪心 [{c2[1]:.2%}, {c2[2]:.2%}] 的区间大部分重叠，"
          "看不出谁好；配对之后差的区间要窄得多")

    print("\n  测试集大小与区间宽度（LSTM 贪心，从 2000 行里取前 n 行）：")
    for n in (100, 300, 1000, 2000):
        c, lo, hi = bootstrap_ci(errs["LSTM 贪心"][:n], lens[:n])
        print(f"    n = {n:>4}：CER {c:6.2%}  区间 {lo:6.2%} – {hi:6.2%}  宽度 {hi - lo:6.2%}")


def part5():
    section("5. 分桶：平均数背后的分布")
    preds, refs, _ = crnn_preds()
    p = preds["LSTM 贪心"]

    def kind(tok):
        if tok.startswith("$"):
            return "价格"
        if tok.count("-") == 2 and tok.replace("-", "").isdigit():
            return "日期"
        if tok.endswith("%"):
            return "百分比"
        if tok.strip("().,:").isdigit():
            return "数字"
        if tok.startswith("("):
            return "括号里的词"
        if tok.strip(".,:").isupper() and len(tok.strip(".,:")) > 1:
            return "全大写的词"
        return "普通的词"

    print("  按行长分桶：")
    buckets = defaultdict(list)
    for a, b in zip(p, refs):
        buckets[min(len(b) // 7, 3)].append((a, b))
    names = ["1–6 个字符", "7–13", "14–20", "21–28"]
    for k in sorted(buckets):
        v = buckets[k]
        print(f"    {names[k]:<12}{len(v):>5} 行  CER {cer(*zip(*v)):6.2%}  整行全对 {exact_match(*zip(*v)):6.1%}")
    print("  按词的类型分桶（预测和参考按空格切开后逐词对齐，只统计词数相同的行）：")
    stats = defaultdict(lambda: [0, 0, 0])
    for a, b in zip(p, refs):
        wa, wb = a.split(), b.split()
        if len(wa) != len(wb):
            continue
        for x, y in zip(wa, wb):
            s = stats[kind(y)]
            s[0] += edit_distance(x, y)
            s[1] += len(y)
            s[2] += 1
    for k, (e, n, c) in sorted(stats.items(), key=lambda kv: -kv[1][0] / kv[1][1]):
        print(f"    {k:<10}{c:>6} 个词  CER {e / n:6.2%}")


def part6():
    section("6. 训练的随机性：同样的配置换 3 个种子重新训练")
    from labs.ch13 import ctc_tour as c13
    from miniocr.models.crnn import CRNN, CTCCollate, ctc_batch_loss
    preds, refs, _ = crnn_preds()
    tr, va, te = c13.data()
    lens = np.array([len(r) for r in refs])
    res = {0: preds["LSTM 贪心"]}
    for seed in (1, 2, 3):
        path = f"{OUT}/crnn_seed{seed}.pt"
        torch.manual_seed(seed)
        model = CRNN(len(c13.V), rnn="lstm").to(DEV)
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=DEV))
        else:
            t0 = time.time()
            steps = 10 * math.ceil(len(tr) / 32)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.05)
            loader = DataLoader(tr, batch_size=32, shuffle=True, collate_fn=CTCCollate(c13.V),
                                generator=torch.Generator().manual_seed(seed))
            for ep in range(10):
                model.train()
                for xb, tg, ln, _ in loader:
                    loss = ctc_batch_loss(model(xb.to(DEV)), tg, ln)
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    sched.step()
            torch.save(model.state_dict(), path)
            print(f"    种子 {seed}：训练用时 {time.time() - t0:.0f} 秒")
        res[seed] = c13.run_eval(model, te)
    vals = []
    for s, p in res.items():
        e = np.array([edit_distance(a, b) for a, b in zip(p, refs)])
        c, lo, hi = bootstrap_ci(e, lens)
        vals.append(c)
        print(f"  种子 {s}：测试 CER {c:6.2%}（bootstrap 区间 {lo:.2%} – {hi:.2%}）  整行全对 {exact_match(p, refs):6.1%}")
    print(f"  4 个种子：平均 {np.mean(vals):.2%}，标准差 {np.std(vals, ddof=1):.2%}，最好 {min(vals):.2%}、最差 {max(vals):.2%}")
    agree = np.mean([len({res[s][i] for s in res}) == 1 for i in range(len(refs))])
    print(f"  4 个模型对同一行给出相同输出的比例 {agree:.1%}")


def part7():
    section("7. 数据泄漏：测试集里有多少行在训练集里出现过")
    preds, refs, train_texts = crnn_preds()
    p = preds["LSTM 贪心"]
    S = set(train_texts)
    dup = np.array([r in S for r in refs])
    print(f"  测试集 2000 行里，有 {dup.sum()} 行（{dup.mean():.1%}）一字不差地出现在 2 万行的训练集里")
    lens = np.array([len(r) for r in refs])
    print(f"  这些行平均 {lens[dup].mean():.1f} 个字符，其他行平均 {lens[~dup].mean():.1f} 个（短的行更容易重复）")
    for name, m in [("出现过", dup), ("没出现过", ~dup)]:
        pp = [a for a, k in zip(p, m) if k]
        rr = [b for b, k in zip(refs, m) if k]
        print(f"    {name:<6}{m.sum():>5} 行  CER {cer(pp, rr):6.2%}  整行全对 {exact_match(pp, rr):6.1%}")
    print("  同样长度（≤ 10 个字符）的行里比较：")
    for name, m in [("出现过", dup & (lens <= 10)), ("没出现过", ~dup & (lens <= 10))]:
        pp = [a for a, k in zip(p, m) if k]
        rr = [b for b, k in zip(refs, m) if k]
        print(f"    {name:<6}{m.sum():>5} 行  CER {cer(pp, rr):6.2%}  整行全对 {exact_match(pp, rr):6.1%}")
    W = {w for t in train_texts for w in t.split()}
    words = [w for r in refs for w in r.split()]
    print(f"  测试集的词里，{np.mean([w not in W for w in words]):.1%} 没在训练集里出现过（大多是随机数字、价格、日期）")


def main():
    parts = [int(a) for a in sys.argv[1:] if a.isdigit()] or [1, 2, 3, 4, 5, 6, 7]
    for p in parts:
        globals()[f"part{p}"]()
    print("\n第 25 章的演示全部完成。")


if __name__ == "__main__":
    main()
