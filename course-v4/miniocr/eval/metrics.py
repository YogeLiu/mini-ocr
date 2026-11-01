"""第 0 章：公式识别的字符串指标。

    edit_distance      Levenshtein 编辑距离：把 a 变成 b 最少要几次插入、删除、替换
    align              回溯编辑距离的表，列出每一步操作（看"哪里错了"）
    units              把一个 LaTeX 公式切成"计数单位"：按字符（char）或按 token（token）
    ratio              错误率的除法，统一处理分母为 0 的情况（零分母约定）
    exact_match        完全匹配率；normalize=True 时先规范化再比较
    error_rate         语料级错误率：Σ 编辑距离 / Σ 参考答案的长度。unit="char" 就是 CER，unit="token" 是按 token 的错误率
    sample_error_rate  单个样本的错误率（会超过 1，见正文 0.6.4 节）
    formula_tags       粗略地给公式打标签：分式、上下标、矩阵、中文、根号、长公式（建错例表时分桶用）
    score_pair         一对（参考答案，预测）的全部字符串指标，返回一个字典
    summarize          把很多对的结果汇总成语料级指标

refs 和 preds 的条数必须相同，否则直接报错：模型没有返回结果的样本要由调用方补成空字符串，不能悄悄少算。

命令行（M0 用它给一份预测文件打分，见 lab.md）：
    python -m miniocr.eval.metrics run.jsonl --out per_sample.csv [--render]
run.jsonl 每行一个 JSON：{"id": ..., "ref": 参考 LaTeX, "pred": 预测 LaTeX}，其他字段原样带到 CSV 里。
"""
import argparse
import csv
import json
import re

from .latex_norm import is_well_formed, normalize_latex, normalized_tokens, tokenize_latex


# >>> 练习 0.5：编辑距离
def edit_distance(a, b):
    """Levenshtein 距离：把序列 a 变成序列 b，最少需要多少次"插入一个、删除一个、替换一个"（每次代价 1）。
    a、b 可以是字符串（按字符比较），也可以是列表（按元素比较，例如 token 列表）。

    动态规划：D[i][j] = a 的前 i 个元素变成 b 的前 j 个元素的最小代价。
        D[i][0] = i（删掉 i 个），D[0][j] = j（插入 j 个），
        D[i][j] = min(D[i-1][j] + 1,                        删除 a[i-1]
                      D[i][j-1] + 1,                        插入 b[j-1]
                      D[i-1][j-1] + (a[i-1] != b[j-1]))     替换（相同则代价 0）
    只需要保留上一行，内存 O(len(b))。

    例：edit_distance("x^{2}", "x_{2}") == 1；edit_distance(["\\alpha"], ["a"]) == 1
    """
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[-1]
# <<<


def edit_table(a, b):
    """完整的动态规划表 D，形状 (len(a)+1, len(b)+1)。教学和回溯用；大规模评测用 edit_distance。"""
    D = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        D[i][0] = i
    for j in range(len(b) + 1):
        D[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    return D


def align(ref, pred):
    """把 ref 变成 pred 的一组最少操作 [(op, ref 的元素, pred 的元素)]。
    op 是 "=" 相同、"S" 替换、"D" 删除（ref 里有、pred 里没有，即漏识别）、"I" 插入（pred 里多出来的）。
    回溯时的优先顺序：对角线 → 删除 → 插入。"""
    D = edit_table(ref, pred)
    ops, i, j = [], len(ref), len(pred)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (ref[i - 1] != pred[j - 1]):
            ops.append(("=" if ref[i - 1] == pred[j - 1] else "S", ref[i - 1], pred[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            ops.append(("D", ref[i - 1], ""))
            i -= 1
        else:
            ops.append(("I", "", pred[j - 1]))
            j -= 1
    return ops[::-1]


def units(latex, unit="char", normalize=True):
    """把公式切成计数单位。
    unit="token"：规范化后的 token 列表（normalize=False 时只分词）。
    unit="char"： 把这些 token 首尾相接拼成字符串，再逐个字符切开。空白不算字符。
    例：units("\\alpha+1", "token") == ["\\alpha", "+", "1"]；units("\\alpha+1", "char") 有 8 个字符。"""
    toks = normalized_tokens(latex) if normalize else tokenize_latex(latex)
    if unit == "token":
        return toks
    if unit == "char":
        return list("".join(toks))
    raise ValueError(f"unit 只能是 'char' 或 'token'，收到 {unit!r}")


def check_same_length(refs, preds):
    """refs 和 preds 条数不同就报错。zip 会悄悄截断到较短的一边，漏掉的样本不会出现在任何分数里。"""
    if len(refs) != len(preds):
        raise ValueError(f"参考答案 {len(refs)} 条、预测 {len(preds)} 条，条数必须相同（缺失的预测请补成空字符串）")


def ratio(dist, total):
    """错误率的除法：编辑距离 / 参考长度。零分母约定：参考长度为 0（参考答案是空的）时，
    预测也是空的算 0.0（全对），预测不空算 1.0（全错）。error_rate、sample_error_rate、summarize 都用它，
    保证同一份数据从哪个入口算都得到同一个数。"""
    return dist / total if total else float(dist > 0)


def exact_match(refs, preds, normalize=True):
    """完全匹配率：预测和参考完全相同的比例。normalize=True 时比较规范化后的写法，否则比较原始字符串。"""
    check_same_length(refs, preds)
    if normalize:
        hits = sum(normalize_latex(r) == normalize_latex(p) for r, p in zip(refs, preds))
    else:
        hits = sum(r == p for r, p in zip(refs, preds))
    return hits / max(1, len(refs))


# >>> 练习 0.6：语料级错误率（CER 与按 token 的错误率）
def error_rate(refs, preds, unit="char", normalize=True):
    """语料级错误率 = 所有样本的编辑距离之和 / 所有参考答案的长度之和。
    长度和编辑距离都按 units(…, unit, normalize) 切出来的单位计算。
    注意：分母是参考答案的长度，不是预测的长度；也不是"每个样本的错误率再求平均"。
    refs 和 preds 条数不同时报错（用 check_same_length）。最后的除法用 ratio（零分母约定：
    参考全空时，预测也全空返回 0.0，否则返回 1.0）。
    例：error_rate(["ab", "cd"], ["ab", "xd"]) == 1 / 4
    """
    check_same_length(refs, preds)
    dist = total = 0
    for r, p in zip(refs, preds):
        ru, pu = units(r, unit, normalize), units(p, unit, normalize)
        dist += edit_distance(ru, pu)
        total += len(ru)
    return ratio(dist, total)


def sample_error_rate(ref, pred, unit="char", normalize=True):
    """单个样本的错误率 = 编辑距离 / 参考答案的长度（用 ratio 做除法）。预测比参考长很多时会大于 1。"""
    ru, pu = units(ref, unit, normalize), units(pred, unit, normalize)
    return ratio(edit_distance(ru, pu), len(ru))
# <<<


_CJK = re.compile(r"[㐀-鿿＀-￯　-〿]")


def formula_tags(latex):
    """粗略的结构标签（建错例表、分桶用）。只看参考答案打标签，不看预测。"""
    toks = normalized_tokens(latex)
    tags = []
    if "\\frac" in toks or "\\binom" in toks:
        tags.append("分式")
    if "^" in toks or "_" in toks:
        tags.append("上下标")
    if any(t.startswith("\\begin{") for t in toks):
        tags.append("矩阵")
    if any(_CJK.search(t) for t in toks):
        tags.append("中文")
    if "\\sqrt" in toks:
        tags.append("根号")
    if len(toks) > 40:
        tags.append("长公式")
    return tags


def score_pair(ref, pred):
    """一对公式的全部字符串指标。"""
    rt, pt = normalized_tokens(ref), normalized_tokens(pred)
    rc, pc = list("".join(rt)), list("".join(pt))
    return {
        "em_raw": ref == pred,
        "em_norm": rt == pt,
        "ed_char": edit_distance(rc, pc), "len_char": len(rc),
        "ed_token": edit_distance(rt, pt), "len_token": len(rt),
        "well_formed": is_well_formed(pred),
        "tags": formula_tags(ref),
    }


def summarize(rows):
    """rows 是 score_pair 的结果列表（可以带 render 指标）。返回语料级汇总。
    错误率用 ratio 做除法，和 error_rate 一致。没有任何能判断渲染的样本时，image_match 是 None（无法判断，不是 0）。"""
    n = max(1, len(rows))
    out = {
        "n": len(rows),
        "em_raw": sum(r["em_raw"] for r in rows) / n,
        "em_norm": sum(r["em_norm"] for r in rows) / n,
        "cer_char": ratio(sum(r["ed_char"] for r in rows), sum(r["len_char"] for r in rows)),
        "cer_token": ratio(sum(r["ed_token"] for r in rows), sum(r["len_token"] for r in rows)),
        "well_formed": sum(r["well_formed"] for r in rows) / n,
    }
    if rows and "compiles" in rows[0]:
        out["compile_rate"] = sum(r["compiles"] for r in rows) / n
        judged = [r for r in rows if r["image_match"] is not None]
        out["image_match"] = sum(r["image_match"] for r in judged) / len(judged) if judged else None
        out["image_judged"] = len(judged)
    return out


def format_value(v):
    """汇总表里的一个值：小数保留 4 位，None（无法判断）显示成 N/A。"""
    return "N/A" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def main(argv=None):
    ap = argparse.ArgumentParser(description="给一份预测文件打分（字符串指标；加 --render 再算可解析率和渲染一致率）")
    ap.add_argument("jsonl")
    ap.add_argument("--out", help="逐样本结果写到这个 CSV")
    ap.add_argument("--render", action="store_true", help="用 matplotlib mathtext 近似可编译率和渲染一致率（慢）")
    a = ap.parse_args(argv)
    rows, extra = [], []
    with open(a.jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            r = score_pair(d["ref"], d["pred"])
            if a.render:
                from .render import compare_render
                r.update(compare_render(d["ref"], d["pred"]))
            rows.append(r)
            extra.append(d)
    s = summarize(rows)
    for k, v in s.items():
        print(f"{k:14s} {format_value(v)}")
    if a.out:
        keys = list(rows[0].keys()) if rows else []
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "ref", "pred"] + keys + ["错误类型", "备注"])
            for d, r in zip(extra, rows):
                w.writerow([d.get("id", ""), d["ref"], d["pred"]]
                           + ["|".join(r[k]) if k == "tags" else r[k] for k in keys] + ["", ""])
        print(f"逐样本结果已写到 {a.out}（最后两列留给你手工填错误类型和备注）")


if __name__ == "__main__":
    main()
