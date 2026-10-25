"""第 3 章：字符、结构化输出与 LaTeX。逐段运行，对照正文阅读。

    python -m labs.ch03.latex_tour
"""
import collections
import os

import numpy as np

from miniocr.data.latex import (LatexError, gen_formula, leaves, normalize, parse, print_tree, structure_of, to_latex,
                                tokenize)
from miniocr.data.render import render_formula, sample_formula, save_grid

OUT = "outputs/ch03"


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def part1_chars():
    section("1. 字符就是整数：Unicode 码位与 UTF-8 字节")
    for ch in ["A", "a", "7", "+", "\\", "{", "é", "∑"]:
        b = ch.encode("utf-8")
        print(f"{ch!r:6s} 码位 U+{ord(ch):04X}（十进制 {ord(ch):5d}）  UTF-8 字节 {list(b)}（{len(b)} 个字节）")
    s = "x^{2}"
    print(f"\n字符串 {s!r} 有 {len(s)} 个字符：{list(s)}")
    print("对应的码位：", [ord(c) for c in s])


def part2_structured_outputs():
    section("2. 三种结构化输出：Markdown、HTML 表格、LaTeX")
    md = "# Title\n\nSome **bold** text and an inline formula \\( x^{2} \\)."
    html = "<table><tr><td>a</td><td>1</td></tr><tr><td>b</td><td>2</td></tr></table>"
    tex = "\\frac{a+1}{b}=x_{n}^{2}"
    for name, s in [("Markdown", md), ("HTML 表格", html), ("LaTeX 公式", tex)]:
        print(f"--- {name}（{len(s)} 个字符）\n{s}")
    print("\n共同点：都是纯文本，但文本里藏着结构（标题、单元格、分子分母）。OCR 模型要输出的正是这样的文本。")


def part3_tokens_and_tree():
    section("3. 记号与语法树")
    s = "\\frac{a+1}{b}=x_{n}^{2}"
    print("LaTeX:", s)
    print("记号:", tokenize(s))
    tree = parse(s)
    print("语法树（元组写法）:", tree)
    print("语法树（缩进写法）:")
    print_tree(tree)
    print("叶子（从左到右的符号）:", leaves(tree))
    print("包含的结构:", sorted(structure_of(tree)))


def part4_equivalent_writings():
    section("4. 同一棵树，多种写法")
    rng = np.random.default_rng(0)
    tree = parse("\\frac{a}{b}+x_{n}^{2}-3y")
    print("规范写法:", to_latex(tree))
    seen = collections.Counter(to_latex(tree, rng) for _ in range(200))
    print(f"随机写 200 次，得到 {len(seen)} 种不同的字符串，出现最多的 8 种：")
    for w, n in seen.most_common(8):
        print(f"  {n:3d} 次  {w!r:40s} 归一化 → {normalize(w)}")


def part5_parse_errors():
    section("5. 不合法的 LaTeX：解析器会指出错在哪里")
    for bad in ["x^", "\\frac{a}", "a \\over b", "x^2^3", "\\alpha+1", "x}"]:
        try:
            parse(bad)
            print(f"{bad!r:14s} 居然解析成功了（不应该）")
        except LatexError as e:
            print(f"{bad!r:14s} 错误：{e}")


def part6_why_normalize():
    section("6. 实验：为什么评测前必须归一化")
    print("模拟一个\"完美\"的识别模型：它每次都认对了公式（语法树完全正确），只是写法随机。")
    print("预测先写在纸上：直接比较字符串，它的\"完全匹配率\"是多少？")
    rng = np.random.default_rng(1)
    for level in (1, 2, 3, 4):
        trees = [gen_formula(rng, level) for _ in range(1000)]
        refs = [to_latex(t, rng) for t in trees]          # 数据集里的标注：也是随机写法
        preds = [to_latex(t, rng) for t in trees]         # 模型的输出：另一种随机写法
        raw = np.mean([p == r for p, r in zip(preds, refs)])
        norm = np.mean([normalize(p) == normalize(r) for p, r in zip(preds, refs)])
        print(f"  level {level}：不归一化的完全匹配 {raw:6.1%}   归一化后 {norm:6.1%}")
    print("同一个模型，评测方式不同，分数可以差几十个百分点——这就是为什么评测协议要先写清楚归一化规则。")


def part7_dataset_stats():
    section("7. 四个难度等级长什么样")
    rng = np.random.default_rng(2)
    imgs = []
    for level in (1, 2, 3, 4):
        lens, structs = [], collections.Counter()
        for _ in range(2000):
            t = gen_formula(rng, level)
            lens.append(len(to_latex(t)))
            structs.update(structure_of(t))
        ex = [to_latex(sample_formula(rng, level)[0]) for _ in range(3)]
        st = ", ".join(f"{k} {v / 20:.0f}%" for k, v in sorted(structs.items()))
        print(f"level {level}：LaTeX 平均 {np.mean(lens):5.1f} 字符，最长 {max(lens):3d}；结构出现率 [{st}]")
        print(f"         例子：{ex}")
        imgs += [sample_formula(rng, level)[1] for _ in range(2)]
    save_grid(imgs, f"{OUT}/levels.png", scale=3, cols=2)
    print(f"已保存 {OUT}/levels.png（每行一个等级，从上到下 level 1–4）")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    part1_chars()
    part2_structured_outputs()
    part3_tokens_and_tree()
    part4_equivalent_writings()
    part5_parse_errors()
    part6_why_normalize()
    part7_dataset_stats()
    print("\n第 3 章的演示全部完成。")
