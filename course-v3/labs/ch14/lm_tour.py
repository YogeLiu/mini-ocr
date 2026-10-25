"""第 14 章：分词与 n 元语法语言模型。只用 CPU。

    python -m labs.ch14.lm_tour            # 全部 6 个部分，约 1 分钟
    python -m labs.ch14.lm_tour 3          # 只运行第 3 部分
"""
import math
import os
import sys
import time
from collections import Counter

import numpy as np

from miniocr.data.corpus import mixed_corpus
from miniocr.tokenize import BPETokenizer, NGramLM, bits_per_char, pretokenize
from miniocr.utils import Vocab

_C = {}


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def corpus():
    if not _C:
        _C["train"], _C["val"] = mixed_corpus(0, 4000), mixed_corpus(1, 400)
    return _C["train"], _C["val"]


def tok(n):
    key = ("bpe", n)
    if key not in _C:
        _C[key] = BPETokenizer.train(corpus()[0], n)
    return _C[key]


# ---------------------------------------------------------------------- 1
def part1_corpus():
    section("1. 混合语料：文字、LaTeX、Markdown")
    tr, va = corpus()
    print(f"  训练 {len(tr):,} 个字符（4000 段），验证 {len(va):,} 个字符（400 段，另一个种子）")
    print("  第一段：\n" + "\n".join("    | " + l for l in tr.split("\n\n")[0].split("\n")))
    c = Counter(tr)
    print(f"  不同的字符 {len(c)} 个；最常见的 10 个：" + "  ".join(f"{k!r}:{v}" for k, v in c.most_common(10)))
    lines = tr.split("\n")
    kinds = Counter("公式" if l.startswith("$$") else "表格" if l.startswith("|") else "列表" if l.startswith("- ")
                    else "标题" if l.startswith("# ") else "文字" for l in lines if l)
    print("  各类行的数量：" + "  ".join(f"{k} {v}" for k, v in kinds.most_common()))


# ---------------------------------------------------------------------- 2
def part2_chars():
    section("2. 字符级分词")
    tr, va = corpus()
    v = Vocab(tr, specials=())
    ids = v.encode(tr)
    print(f"  词表 {len(v)} 个（每个字符一个编号）；训练语料编码成 {len(ids):,} 个 token，每个 token 1 个字符")
    unseen = sorted(set(va) - set(tr))
    print(f"  验证语料里有训练语料没见过的字符吗？{unseen if unseen else '没有'}")
    s = "$\\frac{x^{2}}{y}$"
    print(f"  {s!r} → {len(v.encode(s))} 个 token：{v.encode(s)}")
    print("  优点：词表小、永远不会遇到不认识的词；缺点：序列长（第 15 章：注意力的计算量与长度的平方成正比）")


# ---------------------------------------------------------------------- 3
def part3_bpe():
    section("3. BPE：从字符开始，反复合并最常见的相邻对")
    tr, va = corpus()
    print("  前 15 次合并：")
    t = time.perf_counter()
    t15 = BPETokenizer.train(tr, 15, verbose=True)
    s = "Point drive science small $\\frac{y}{n+7}+8-k$."
    for n in (0, 15, 100, 300, 1000):
        tk = t15 if n == 15 else tok(n)
        toks = tk.tokens(s)
        print(f"  {n:4d} 次合并（词表 {len(tk):4d}）：{len(toks):2d} 个 token  " + "|".join(toks))
    print("  词表大小与压缩率（验证语料，平均每个 token 多少个字符）：")
    for n in (0, 100, 300, 1000, 3000):
        tk = tok(n)
        nt = len(tk.encode(va))
        print(f"    {n:5d} 次合并  词表 {len(tk):5d}  验证语料 {nt:7,} 个 token  每个 token {len(va) / nt:.2f} 个字符")
    tk = tok(1000)
    for w in (" the", " mountain", " Mountain", " MOUNTAIN", " zebra", "\\frac", "\\sqrt", "|---|---|", "2024", "$$"):
        print(f"    {w!r:14s} → {tk.tokens(w)}")
    print("  数字在预切分时就被拆成单个数字（Qwen、Llama 等模型也这样做），所以 2024 永远是 4 个 token")
    print(f"  （训练 3000 次合并用时约 {time.perf_counter() - t:.0f} 秒，包括前面几个词表）")


# ---------------------------------------------------------------------- 4
def part4_ngram():
    section("4. n 元语法语言模型：交叉熵、困惑度、bits per char")
    tr, va = corpus()
    cv = Vocab(tr, specials=())
    tests = [("字符", cv.encode(tr), cv.encode(va), len(cv))]
    for n in (300, 1000):
        tk = tok(n)
        tests.append((f"BPE {len(tk)}", tk.encode(tr), tk.encode(va), len(tk)))
    print(f"  {'分词':10s} {'n':>2s}  {'训练 nat/token':>14s} {'验证 nat/token':>14s} {'验证困惑度':>10s} {'验证 bits/char':>14s}")
    for name, a, b, V in tests:
        for n in (1, 2, 3, 5):
            lm = NGramLM(n, V, k=0.01).fit(a)
            ce_tr, ce_va = lm.cross_entropy(a[:200000]), lm.cross_entropy(b)
            print(f"  {name:10s} {n:2d}  {ce_tr:14.3f} {ce_va:14.3f} {math.exp(ce_va):10.1f} {bits_per_char(ce_va, len(b), len(va)):14.3f}")
    print(f"  参照：字符均匀分布 log2({len(cv)}) = {math.log2(len(cv)):.2f} bits/char")
    a, b = cv.encode(tr), cv.encode(va)
    print("  平滑系数 k 的影响（字符 5 元）：")
    for k in (0.0, 1e-4, 0.01, 0.1, 1.0):
        lm = NGramLM(5, len(cv), k=k).fit(a)
        ce = lm.cross_entropy(b)
        print(f"    k = {k:<6}  验证 {ce:.3f} nat/char" + ("（有没见过的 5 元组，概率为 0 → 无穷大）" if math.isinf(ce) else ""))


# ---------------------------------------------------------------------- 5
def sample(lm, itos, n, rng, prefix=()):
    out = list(prefix)
    for _ in range(n):
        h = tuple(([-1] * (lm.n - 1) + out)[-(lm.n - 1):]) if lm.n > 1 else ()
        lp = np.array([lm.logprob(h, w) for w in range(lm.V)])
        p = np.exp(lp - lp.max()); p /= p.sum()
        out.append(int(rng.choice(lm.V, p=p)))
    return "".join(itos[i] for i in out)


def part5_sample():
    section("5. 从 n 元模型采样：像，但不对")
    tr, va = corpus()
    cv = Vocab(tr, specials=())
    a = cv.encode(tr)
    rng = np.random.default_rng(0)
    for n in (2, 4, 7):
        lm = NGramLM(n, len(cv), k=1e-4).fit(a)
        txt = sample(lm, cv.itos, 220, rng)
        print(f"  字符 {n} 元：")
        print("\n".join("    | " + l for l in txt.split("\n")))
    lm7 = NGramLM(7, len(cv), k=1e-4).fit(a)
    txt = sample(lm7, cv.itos, 400, rng)
    seen = [i < 6 or lm7.ctx.get(tuple(cv.encode(txt[i - 6:i])), 0) > 0 for i in range(len(txt))]
    first = next((i for i, ok in enumerate(seen) if not ok), None)
    print(f"  7 元模型继续生成 400 个字符：第一次遇到训练中没见过的 6 字符上下文是在第 {first} 个字符；"
          f"之后的 {sum(not x for x in seen[first or 0:])} 个位置里上下文都没见过")
    print("    上下文没见过时，加 k 平滑给出的是均匀分布，于是开始乱码；乱码的上下文更不可能见过，再也回不来")
    lm = NGramLM(4, len(cv), k=1e-4).fit(a)
    bad, total = 0, 0
    for _ in range(300):
        txt = sample(lm, cv.itos, 80, rng, prefix=cv.encode("$$"))
        body = txt.split("$$")[1] if txt.count("$$") >= 2 else None
        if body is not None:
            total += 1
            depth, ok = 0, True
            for ch in body:
                depth += (ch == "{") - (ch == "}")
                ok &= depth >= 0
            bad += not (ok and depth == 0)
    print(f"  4 元模型生成 300 个以 $$ 开头的片段，其中 {total} 个有结尾的 $$；花括号不配对的 {bad} 个（{bad / max(1, total):.0%}）")
    print("  n 元模型只记得前 n−1 个字符，无法知道更早之前打开了几个 {——这需要第 15–16 章的注意力")


# ---------------------------------------------------------------------- 6
def part6_real():
    section("6. 真实的分词器：Qwen3（本机缓存，离线加载）")
    try:
        from transformers import AutoTokenizer
        qt = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", local_files_only=True)
    except Exception as e:                                     # 没有缓存或没装 transformers 时跳过
        print(f"  无法加载（{type(e).__name__}），跳过。第 24 章会正式下载真实模型。")
        return
    tr, va = corpus()
    print(f"  词表大小 {len(qt):,}（我们的 BPE：{len(tok(1000)):,}）")
    s = "Point drive science small $\\frac{y}{n+7}+8-k$.\n| Name | 2024 |"
    ids = qt.encode(s, add_special_tokens=False)
    print(f"  {s!r}\n    → {len(ids)} 个 token：" + "|".join(qt.convert_ids_to_tokens(ids)))
    for w in ("中文识别", "数学公式", "x²", "∑"):
        ids = qt.encode(w, add_special_tokens=False)
        print(f"    {w!r} → {len(ids)} 个 token：{[qt.decode([i]) for i in ids]}  （内部显示为 {qt.convert_ids_to_tokens(ids)}）")
    n_q = len(qt.encode(va, add_special_tokens=False))
    n_o = len(tok(1000).encode(va))
    print(f"  验证语料：Qwen3 {n_q:,} 个 token（{len(va) / n_q:.2f} 字符/token），我们的 BPE-1000：{n_o:,} 个（{len(va) / n_o:.2f}）")
    print("  Qwen 的压缩率更高：词表大 140 倍，英文单词几乎都有整词，还会把 }{、.\\n 这类符号组合合并成一个 token；")
    print("  我们的预切分把每个标点单独切开，标点永远合并不了")


PARTS = {1: part1_corpus, 2: part2_chars, 3: part3_bpe, 4: part4_ngram, 5: part5_sample, 6: part6_real}

if __name__ == "__main__":
    for k in [int(a) for a in sys.argv[1:] if a.isdigit()] or list(PARTS):
        PARTS[k]()
    print("\n第 14 章的演示全部完成。")
