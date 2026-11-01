"""第 0 章　导论：公式识别与评测。配套实验脚本，正文里引用的每一个数字都来自这里。

    python -m book.ch00_intro.tour            # 全部 10 个部分（CPU 上十几秒）
    python -m book.ch00_intro.tour 4 5        # 只运行第 4、5 部分

本章不训练模型，也不用 GPU。为了和其他章的脚本保持一致，仍然接受 --device 参数，但不会用到它。

部分与正文小节的对应：
    1  任务与演示集（0.1、0.3）          6  可编译率与它的近似（0.7）
    2  分词（0.4.2）                     7  渲染后比较（0.8）
    3  规范化（0.4.3–0.4.6、0.5）        8  五个指标放在一起（0.9）
    4  编辑距离（0.6.1–0.6.3）           9  常见错误（0.11）
    5  错误率：CER 与按 token（0.6.4–0.6.6） 10 M0 演练：划分、打分、错例表（0.10，实验手册）

关于数据：TexPixel 数据的使用权还没有确认，所以本章不用任何真实数据。
DEMO 里的 16 对"参考答案 / 模型预测"是作者手写的，专门设计成每个指标的盲区都能被其中某一对暴露出来。
"""
import csv
import hashlib
import json
import os
import sys

# 除非明确要求 --device cuda，否则对 PyTorch 隐藏 GPU：本机的 GPU 正被别的服务占用。
if "cuda" not in sys.argv:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

from miniocr.device import device_from_argv
from miniocr.eval.latex_norm import (apply_token_rules, is_well_formed, normalize_latex, normalized_tokens,
                                     parse_latex, serialize, tokenize_latex)
from miniocr.eval.metrics import (align, edit_distance, edit_table, error_rate, exact_match, formula_tags,
                                  format_value, main as metrics_main, sample_error_rate, score_pair,
                                  summarize, units)
from miniocr.eval.render import can_parse, compare_images, compare_render, render, save_comparison

OUT = "outputs/book/ch00"
os.makedirs(OUT, exist_ok=True)
DEVICE = device_from_argv(default="cpu")      # 本章用不到，只是保持接口一致

# 演示集：(编号, 说明, 参考答案, 模型预测, 含义是否相同)。
# P1–P7 是"写法不同、含义相同"，理想的指标应该判它们对；P8–P16 是真的错了，理想的指标应该判错，并分出轻重。
DEMO = [
    ("P1", "写法不同", r"x^{2}+y^{2}", r"x^2 + y^2", True),
    ("P2", "上下标顺序", r"\sum_{i=1}^{n}i^{2}", r"\sum^{n}_{i=1}i^2", True),
    ("P3", "省略花括号", r"\dfrac{1}{2}", r"\frac12", True),
    ("P4", "同义命令", r"a\leq b", r"a\le b", True),
    ("P5", "括号尺寸", r"\left(\frac{a}{b}\right)^{2}", r"(\frac{a}{b})^{2}", True),
    ("P6", "间距命令", r"\int_{0}^{1}f(x)\,dx", r"\int_0^1 f(x)dx", True),
    ("P7", "撇号", r"f'(x)", r"f^{\prime}(x)", True),
    ("P8", "上标认成下标", r"x^{2}+y^{2}", r"x_{2}+y^{2}", False),
    ("P9", "分子分母颠倒", r"\frac{a}{b}", r"\frac{b}{a}", False),
    ("P10", "命令认成字母", r"\alpha+\beta", r"a+\beta", False),
    ("P11", "漏了根号次数", r"\sqrt[3]{x}", r"\sqrt{x}", False),
    ("P12", "长公式错一个符号", r"\sum_{k=1}^{n}k=\frac{n(n+1)}{2}", r"\sum_{k=1}^{n}k=\frac{n(n-1)}{2}", False),
    ("P13", "中文形近字", r"S_{\text{面积}}=\pi r^{2}", r"S_{\text{面枳}}=\pi r^{2}", False),
    ("P14", "矩阵元素对调", r"\begin{pmatrix}a&b\\c&d\end{pmatrix}", r"\begin{pmatrix}a&b\\d&c\end{pmatrix}", False),
    ("P15", "缺右花括号", r"x^{2}", r"x^{2", False),
    ("P16", "重复输出", r"x^{2}", r"x^{2}x^{2}x^{2}x^{2}", False),
]
REFS = [d[2] for d in DEMO]
PREDS = [d[3] for d in DEMO]


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def yn(b):
    return "—" if b is None else ("是" if b else "否")


# ---------------------------------------------------------------------- 1
def part1_task():
    section("1. 任务与演示集（0.1、0.3）：一张公式图片 → 一段 LaTeX")
    print("  演示集（作者手写，不是真实模型的输出）：")
    for pid, title, ref, pred, same in DEMO:
        print(f"    {pid:4s} {title:10s} 参考 {ref:38s} 预测 {pred:38s} 含义相同：{yn(same)}")
    n_same = sum(d[4] for d in DEMO)
    raw_equal = sum(r == p for r, p in zip(REFS, PREDS))
    print(f"  共 {len(DEMO)} 对，其中含义相同 {n_same} 对、真的错了 {len(DEMO) - n_same} 对")
    print(f"  原始字符串完全相同的：{raw_equal} 对")
    path = os.path.join(OUT, "demo_pairs.png")
    save_comparison([(f"{d[0]} {d[1]}", d[2], d[3]) for d in DEMO], path)
    print(f"  参考答案和预测的渲染图已保存到 {path}（mathtext 渲染，见 0.7 节的说明）")


# ---------------------------------------------------------------------- 2
def part2_tokenize():
    section("2. 分词（0.4.2）：命令算一个 token，汉字一个字一个 token")
    for s in [r"\frac{a}{b} + x^2", r"S_{\text{面积}}=\pi r^{2}", r"\begin{pmatrix}a&b\\c&d\end{pmatrix}", r"a\,b\;c",
              r"\alpha+\beta"]:
        t = tokenize_latex(s)
        print(f"    {s:40s} → {len(t):2d} 个 token：{t}")
    print("  同一个公式按字符数和按 token 数（都先去掉空白）：")
    for s in [r"\alpha+\beta", r"a+b", r"S_{\text{面积}}=\pi r^{2}", r"\begin{pmatrix}a&b\\c&d\end{pmatrix}"]:
        nc = len("".join(s.split()))
        print(f"    {s:40s} 字符 {nc:3d}   token {len(tokenize_latex(s)):3d}")


# ---------------------------------------------------------------------- 3
def part3_normalize():
    section("3. 规范化（0.4.3–0.4.6、0.5）")
    print("  逐 token 规则（同义命令、间距、括号尺寸）：")
    for s in [r"\dfrac{1}{2}", r"a\le b", r"\left(\frac{a}{b}\right)", r"f(x)\,dx", r"\left. x \right|_{0}^{1}"]:
        print(f"    {s:30s} → {apply_token_rules(tokenize_latex(s))}")
    print("  语法树（解析 → 规范地写回）：")
    for s in [r"x^2", r"x^2_i", r"\frac12", r"x^ab", r"{a}^{2}", r"\sqrt[3]{x}", r"{{a+b}}^{2}", r"\frac{{a+b}}{c}",
              r"{x^2}^3"]:
        toks = apply_token_rules(tokenize_latex(s))
        tree = parse_latex(toks)
        print(f"    {s:14s} 树 {tree}")
        print(f"    {'':14s} 规范写法 {' '.join(serialize(tree))}")
    print("  一个太松的规范化器：把所有花括号都删掉")
    naive = lambda s: "".join(c for c in "".join(s.split()) if c not in "{}")
    for a, b in [(r"x^{ab}", r"x^ab"), (r"x^{2}", r"x^2")]:
        print(f"    {a:8s} vs {b:8s}  删花括号后相等：{yn(naive(a) == naive(b))}   本章的规范化后相等：{yn(normalize_latex(a) == normalize_latex(b))}")
    print(f"    x^ab 的规范写法：{normalize_latex('x^ab')}（只有 a 在上标里）")
    print("  底自己带上下标时，花括号不能去掉：")
    for a, b in [(r"{x^2}^3", r"x^{2}^{3}")]:
        print(f"    参考 {a}（合法） 规范写法 {normalize_latex(a)}   结构合法 {yn(is_well_formed(a))}")
        print(f"    预测 {b}（不合法） 规范写法 {normalize_latex(b)}   结构合法 {yn(is_well_formed(b))}"
              f"   规范化后相等 {yn(score_pair(a, b)['em_norm'])}")
    print("  演示集逐对：规范化后是否相等")
    for pid, title, ref, pred, same in DEMO:
        print(f"    {pid:4s} {title:10s} 原样相等 {yn(ref == pred)}  规范化后相等 {yn(normalize_latex(ref) == normalize_latex(pred))}  含义相同 {yn(same)}")
    em_raw, em_norm = exact_match(REFS, PREDS, normalize=False), exact_match(REFS, PREDS, normalize=True)
    print(f"  完全匹配率：原始字符串 {em_raw:.4f}（{round(em_raw * 16)}/16）；规范化后 {em_norm:.4f}（{round(em_norm * 16)}/16）")
    print(f"  含义相同的 7 对里，规范化后判对的：{sum(normalize_latex(d[2]) == normalize_latex(d[3]) for d in DEMO if d[4])} 对（P7 撇号没有被规则覆盖）")
    print(f"  P12（错一个符号）和 P16（重复输出）的完全匹配都是 0；空预测也是 0：完全匹配分不出错误的轻重")


# ---------------------------------------------------------------------- 4
def part4_edit_distance():
    section("4. 编辑距离（0.6.1–0.6.3）")
    a, b = "x^{2}", "x_2"
    D = edit_table(a, b)
    print(f"  参考 a = {a!r}（{len(a)} 个字符），预测 b = {b!r}（{len(b)} 个字符），表 D 的形状 ({len(a) + 1}, {len(b) + 1})")
    print("        " + "  ".join(f"{c:>2s}" for c in ["ε"] + list(b)))
    for i, row in enumerate(D):
        print(f"    {(['ε'] + list(a))[i]:>2s}  " + "  ".join(f"{v:2d}" for v in row))
    print(f"  编辑距离 = D[{len(a)}][{len(b)}] = {D[-1][-1]}")
    print(f"  一组最少操作（= 相同，S 替换，D 删除，I 插入）：{align(a, b)}")
    print(f"  规范化以后再比：{normalize_latex(a)!r} vs {normalize_latex(b)!r}，按字符的编辑距离 = {edit_distance(units(a, 'char'), units(b, 'char'))}")
    print(f"  另一个例子：edit_distance('kitten', 'sitting') = {edit_distance('kitten', 'sitting')}")
    print("  演示集逐对（规范化后）：编辑距离 / 参考长度")
    for pid, title, ref, pred, same in DEMO:
        s = score_pair(ref, pred)
        print(f"    {pid:4s} {title:10s} 按字符 {s['ed_char']:2d}/{s['len_char']:2d}   按 token {s['ed_token']:2d}/{s['len_token']:2d}")
    rt, pt = normalized_tokens(DEMO[8][2]), normalized_tokens(DEMO[8][3])
    print(f"  P9 的 token 对齐：{[op for op in align(rt, pt) if op[0] != '=']}")


# ---------------------------------------------------------------------- 5
def part5_error_rate():
    section("5. 错误率：CER 与按 token（0.6.4–0.6.6）")
    for unit in ("char", "token"):
        print(f"  演示集 16 对，unit={unit}：语料级错误率 {error_rate(REFS, PREDS, unit):.4f}")
    err = [d for d in DEMO if not d[4]]
    for unit in ("char", "token"):
        print(f"  只看真的错了的 9 对，unit={unit}：语料级错误率 {error_rate([d[2] for d in err], [d[3] for d in err], unit):.4f}")
    print("  语料级 vs 逐样本平均（两个样本）：")
    pairs = [(r"\pi", "n"), (DEMO[11][2], DEMO[11][3])]
    for r, p in pairs:
        print(f"    参考 {r:36s} 预测 {p:36s} token 编辑距离 {edit_distance(units(r, 'token'), units(p, 'token'))}"
              f" / 长度 {len(units(r, 'token'))} = {sample_error_rate(r, p, 'token'):.4f}")
    corpus = error_rate([r for r, _ in pairs], [p for _, p in pairs], "token")
    mean = sum(sample_error_rate(r, p, "token") for r, p in pairs) / 2
    print(f"    语料级 = 总距离 / 总长度 = {corpus:.4f}；逐样本平均 = {mean:.4f}")
    print(f"  P16（重复输出）单样本错误率：按字符 {sample_error_rate(DEMO[15][2], DEMO[15][3], 'char'):.4f}，按 token {sample_error_rate(DEMO[15][2], DEMO[15][3], 'token'):.4f}（大于 1）")
    print("  按字符和按 token 的差别（单样本错误率）：")
    for i in (9, 12, 13):
        pid, title, r, p, _ = DEMO[i]
        print(f"    {pid:4s} {title:10s} 按字符 {sample_error_rate(r, p, 'char'):.4f}   按 token {sample_error_rate(r, p, 'token'):.4f}")
    print("  错误轻重：")
    for i in (7, 8, 11):
        pid, title, r, p, _ = DEMO[i]
        print(f"    {pid:4s} {title:10s} 按 token 编辑距离 {edit_distance(units(r, 'token'), units(p, 'token'))}，单样本错误率 {sample_error_rate(r, p, 'token'):.4f}")


# ---------------------------------------------------------------------- 6
def part6_compile():
    section("6. 可编译率与它的近似（0.7）：用 mathtext 能否解析来代替")
    print("  本机没有 LaTeX。下面的\"能解析\"是 matplotlib mathtext 的结果，只是可编译率的近似。")
    for pid, title, ref, pred, same in DEMO:
        print(f"    {pid:4s} {title:10s} 参考能解析 {yn(can_parse(ref))}   预测能解析 {yn(can_parse(pred))}   预测结构合法 {yn(is_well_formed(pred))}")
    pr = sum(can_parse(p) for p in PREDS)
    rr = sum(can_parse(r) for r in REFS)
    wf = sum(is_well_formed(p) for p in PREDS)
    print(f"  预测的可解析率 {pr}/16 = {pr / 16:.4f}；参考答案的可解析率 {rr}/16 = {rr / 16:.4f}；预测的结构合法率 {wf}/16 = {wf / 16:.4f}")
    print("  mathtext 判为\"不能解析\"、但在真正的 LaTeX 里是合法写法的例子：")
    for s in [r"\frac12", r"a\le b", r"\tfrac{1}{2}", r"\begin{pmatrix}a&b\\c&d\end{pmatrix}"]:
        print(f"    {s:40s} mathtext 能解析 {yn(can_parse(s))}   先规范化再解析 {yn(can_parse(normalize_latex(s)))}")
    print(f"  能解析但错了的预测：P8 {DEMO[7][3]} 能解析 {yn(can_parse(DEMO[7][3]))}，P16 能解析 {yn(can_parse(DEMO[15][3]))}")


# ---------------------------------------------------------------------- 7
def part7_render():
    section("7. 渲染后比较（0.8）")
    A = np.array([[0, 255, 255, 0], [0, 255, 0, 0], [0, 255, 0, 0]], dtype=np.uint8)
    B = np.array([[0, 255, 255, 0], [0, 255, 255, 0], [0, 255, 0, 0]], dtype=np.uint8)
    print(f"  手算例子 1：A 形状 {A.shape}、B 形状 {B.shape}，diff = {compare_images(A, B):.4f}")
    C = np.array([[255, 0, 255, 0]], dtype=np.uint8)
    E = np.array([[255, 0, 0, 255]], dtype=np.uint8)
    print(f"  手算例子 2：C = [墨 白 墨 白]，E = [墨 白 白 墨]，diff = {compare_images(C, E):.4f}（第二笔只右移了一格）")
    img = render(r"x^{2}")
    print(f"  x^{{2}} 在 {100} dpi 下渲染成形状 {img.shape} 的灰度图；同一个公式渲染两次 diff = {compare_images(img, render(r'x^{2}')):.4f}")
    print("  演示集逐对：")
    for pid, title, ref, pred, same in DEMO:
        c = compare_render(ref, pred)
        d = "—" if c["pixel_diff"] is None else f"{c['pixel_diff']:.3f}"
        print(f"    {pid:4s} {title:10s} 渲染一致 {yn(c['image_match']):2s}  diff {d:>6s}   含义相同 {yn(same)}")
    rows = [compare_render(r, p) for r, p in zip(REFS, PREDS)]
    judged = [r for r in rows if r["image_match"] is not None]
    m = sum(r["image_match"] for r in judged)
    print(f"  能判断的 {len(judged)} 对里渲染一致 {m} 对，渲染一致率 {m / len(judged):.4f}；另有 {16 - len(judged)} 对无法判断")
    print(f"  如果把无法判断的当作不一致：{m}/16 = {m / 16:.4f}")
    path = os.path.join(OUT, "render_P5_P6_P7.png")
    save_comparison([(f"{d[0]} {d[1]}", d[2], d[3]) for d in DEMO[4:7]], path, dpi=200)
    print(f"  P5、P6、P7 的渲染对比图：{path}")


# ---------------------------------------------------------------------- 8
def part8_panel():
    section("8. 五个指标放在一起（0.9）")
    print("  每个指标的\"判对\"：完全匹配（原始 / 规范化后）、可解析、渲染一致。与\"含义相同\"一栏比较。")
    print(f"    {'':4s} {'':10s} {'含义':4s} {'原样EM':6s} {'规范EM':6s} {'tok距离':6s} {'可解析':6s} {'渲染一致':6s}")
    agree = {"em_raw": 0, "em_norm": 0, "compiles": 0, "image_match": 0}
    judged = {"em_raw": 0, "em_norm": 0, "compiles": 0, "image_match": 0}
    rows = []
    for pid, title, ref, pred, same in DEMO:
        s = score_pair(ref, pred)
        s.update(compare_render(ref, pred))
        rows.append(s)
        for k in agree:
            if s[k] is not None:
                judged[k] += 1
                agree[k] += bool(s[k]) == same
        print(f"    {pid:4s} {title:10s} {yn(same):4s} {yn(s['em_raw']):6s} {yn(s['em_norm']):6s} {s['ed_token']:6d} {yn(s['compiles']):6s} {yn(s['image_match']):6s}")
    print("  每个指标的判断和\"含义相同\"一致的对数（渲染一致只算能判断的）：")
    for k, v in agree.items():
        print(f"    {k:12s} {v}/{judged[k]}")
    s = summarize(rows)
    print("  演示集的汇总指标：")
    for k, v in s.items():
        print(f"    {k:14s} {format_value(v)}")


# ---------------------------------------------------------------------- 9
def part9_pitfalls():
    section("9. 常见错误（0.11）")
    em_pred_only = sum(r == normalize_latex(p) for r, p in zip(REFS, PREDS)) / 16
    print(f"  (a) 只规范化预测、不规范化参考：完全匹配 {em_pred_only:.4f}；两边都规范化：{exact_match(REFS, PREDS):.4f}")
    dist = sum(edit_distance(units(r, 'token'), units(p, 'token')) for r, p in zip(REFS, PREDS))
    lp = sum(len(units(p, 'token')) for p in PREDS)
    lr = sum(len(units(r, 'token')) for r in REFS)
    print(f"  (b) 分母用预测的长度：{dist}/{lp} = {dist / lp:.4f}；用参考的长度（正确）：{dist}/{lr} = {dist / lr:.4f}")
    print("      参考 x^{2}，预测把它复读 k 遍（按 token）：")
    r = "x^{2}"
    for k in (1, 2, 3, 4, 10):
        p = r * k
        d, L, Lp = edit_distance(units(r, "token"), units(p, "token")), len(units(r, "token")), len(units(p, "token"))
        print(f"      k={k:2d}  距离 {d:2d}  分母用预测长度 {d}/{Lp} = {d / Lp:.4f}   分母用参考长度 {d}/{L} = {d / L:.4f}")
    mean = sum(sample_error_rate(r, p, 'token') for r, p in zip(REFS, PREDS)) / 16
    print(f"  (c) 逐样本错误率的平均 {mean:.4f}，语料级 {error_rate(REFS, PREDS, 'token'):.4f}")
    raw_char = error_rate(REFS, PREDS, "char", normalize=False)
    print(f"  (d) 不规范化直接算按字符的错误率 {raw_char:.4f}，规范化后 {error_rate(REFS, PREDS, 'char'):.4f}")
    print(f"  (e) 把\"无法判断\"当作\"不一致\"：见第 7 部分最后两行")


# ---------------------------------------------------------------------- 10
def part10_m0_drill():
    section("10. M0 演练：划分、打分、错例表（0.10，实验手册）")
    def split_of(latex):
        h = hashlib.sha1(normalize_latex(latex).encode("utf-8")).hexdigest()
        return h, "test" if int(h, 16) % 100 < 50 else "validation"

    print("  按规范化后的参考答案取哈希来分组划分：规范化结果相同的样本一定落进同一个集合")
    for pid, _, ref, _, _ in DEMO[:3] + [DEMO[7]]:
        h, sp = split_of(ref)
        print(f"    {pid:4s} {normalize_latex(ref):34s} sha1 前 8 位 {h[:8]}  模 100 = {int(h, 16) % 100:2d}  → {sp}")
    print(f"    P1 和 P8 的参考答案相同，哈希也相同（都是 x^{{2}}+y^{{2}}），所以必定在同一个集合里")
    print("  反例：规范化器没覆盖的等价写法（P7 的两种写法）会被分开")
    for s in (DEMO[6][2], DEMO[6][3]):
        h, sp = split_of(s)
        print(f"    {s:16s} {normalize_latex(s):34s} sha1 前 8 位 {h[:8]}  模 100 = {int(h, 16) % 100:2d}  → {sp}")
    path = os.path.join(OUT, "demo_run.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for pid, title, ref, pred, _ in DEMO:
            f.write(json.dumps({"id": pid, "ref": ref, "pred": pred, "note": title}, ensure_ascii=False) + "\n")
    print(f"  把演示集写成预测文件 {path}，用命令行打分：")
    metrics_main([path, "--out", os.path.join(OUT, "demo_per_sample.csv"), "--render"])
    print("  自动打出的结构标签（只看参考答案）：")
    for pid, _, ref, _, _ in DEMO:
        print(f"    {pid:4s} {formula_tags(ref)}")


PARTS = {1: part1_task, 2: part2_tokenize, 3: part3_normalize, 4: part4_edit_distance, 5: part5_error_rate,
         6: part6_compile, 7: part7_render, 8: part8_panel, 9: part9_pitfalls, 10: part10_m0_drill}

if __name__ == "__main__":
    for k in [int(a) for a in sys.argv[1:] if a.isdigit()] or list(PARTS):
        PARTS[k]()
    print("\n第 0 章的实验全部完成。")
