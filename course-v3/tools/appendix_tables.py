"""从各章的"术语卡"和"常见错误"表格生成附录 C（术语表）和附录 B 的后半部分（各章常见错误总表）。
新增或修改章节之后重新运行（在 course-v3/ 下）：

    python -m tools.appendix_tables

附录 C 整个文件重写；附录 B 只重写 MARKER 那一行之后的部分，前面手写的内容保留。
"""
import glob
import re

PARTS = [("第 0 部分　准备", 0, 3), ("第一部分　数学", 4, 6), ("第二部分　神经网络", 7, 12), ("第三部分　序列与语言模型", 13, 18),
         ("第四部分　视觉编码器", 19, 21), ("第五部分　视觉语言模型", 22, 24), ("第六部分　评测与监督微调", 25, 28),
         ("第七部分　偏好与强化学习", 29, 32), ("第八部分　文档 OCR 与真实世界", 33, 36)]
MARKER = "<!-- 以下由 tools/appendix_tables.py 生成，不要手改 -->"


def section(text, title):
    m = re.search(r"^## " + re.escape(title) + r".*?$(.*?)(?=^## |\Z)", text, re.S | re.M)
    return m.group(1) if m else ""


def rows(block):
    """Markdown 表格 → 行的列表（跳过分隔行）。单元格里转义的 \\| 不算分隔符。"""
    out = []
    for line in block.splitlines():
        if line.startswith("|") and not re.match(r"^\|[\s\-|:]+\|$", line):
            out.append([c.strip() for c in re.split(r"(?<!\\)\|", line.strip()[1:-1])])
    return out


def chapters():
    res = {}
    for f in sorted(glob.glob("ch*.md")):
        t = open(f, encoding="utf-8").read()
        ch = int(f[2:4])
        errs = []
        for m in re.finditer(r"^##+ [\d.]*\s*常见错误.*?$(.*?)(?=^## |\Z)", t, re.S | re.M):
            errs += rows(m.group(1))
        res[ch] = {"file": f, "title": re.sub(r"^# 第 \d+ 章\s*", "", t.splitlines()[0]),
                   "terms": rows(section(t, "术语卡")), "errors": errs}
    return res


def glossary(res):
    seen = {}
    for ch in sorted(res):
        for row in res[ch]["terms"][1:]:
            if len(row) == 2:                              # 没有英文一栏的术语卡
                row = [row[0], "—", row[1]]
            if len(row) != 3:
                raise ValueError(f"第 {ch} 章的术语卡有一行不是 2 或 3 列（单元格里的 | 要写成 \\|）：{row}")
            zh, en, one = row
            if zh in seen:
                seen[zh]["chs"].append(ch)
            else:
                seen[zh] = dict(zh=zh, en=en, one=one, chs=[ch])
    out = ["# 附录 C　术语表（中英对照）", "",
           "全书各章\"术语卡\"的合集（由 `tools/appendix_tables.py` 生成）。同一个术语在几章里出现时，释义取第一次出现的那一章，"
           "\"章\"一栏列出所有出现过它的章，第一个是首次出现。",
           "前半部分按课程的部分排列，便于复习一个部分；后半部分是按英文字母排序的索引，便于查一个英文词。", "",
           f"共 {len(seen)} 个术语，来自第 {min(res)}–{max(res)} 章。", ""]
    for name, a, b in PARTS:
        items = [v for v in seen.values() if a <= v["chs"][0] <= b]
        if items:
            out += [f"## {name}（第 {a}–{b} 章）", "", "| 中文 | 英文 | 一句话 | 章 |", "|---|---|---|---|"]
            out += [f"| {v['zh']} | {v['en']} | {v['one']} | {'、'.join(map(str, v['chs']))} |" for v in items]
            out.append("")
    out += ["## 英文索引", "", "| 英文 | 中文 | 首次出现 |", "|---|---|---|"]
    for v in sorted(seen.values(), key=lambda v: v["en"].lower().lstrip("(")):
        if v["en"] and v["en"] != "—":
            out.append(f"| {v['en']} | {v['zh']} | 第 {v['chs'][0]} 章 |")
    return "\n".join(out) + "\n", len(seen)


def error_tables(res):
    out, n = [], 0
    for name, a, b in PARTS:
        chs = [c for c in sorted(res) if a <= c <= b and res[c]["errors"]]
        if not chs:
            continue
        out += [f"### {name}", ""]
        for c in chs:
            hdr, *body = res[c]["errors"]
            out += [f"**第 {c} 章　{res[c]['title']}**（[{res[c]['file']}]({res[c]['file']})）", "",
                    "| 现象 | 原因 | 怎么查 / 怎么改 |", "|---|---|---|"]
            for row in body:
                row = (row + [""] * 3)[:3]
                if hdr[0] == "错误":                       # "错误 | 症状 | 原因与修正" 的表换成 "现象 | 原因 | 怎么改" 的顺序
                    row = [row[1], row[0], row[2]]
                out.append("| " + " | ".join(row) + " |")
                n += 1
            out.append("")
    return "\n".join(out) + "\n", n


def main():
    res = chapters()
    text, n_terms = glossary(res)
    open("appendix_c_glossary.md", "w", encoding="utf-8").write(text)
    tables, n_err = error_tables(res)
    path = "appendix_b_silent_errors.md"
    head = open(path, encoding="utf-8").read().split(MARKER)[0]
    open(path, "w", encoding="utf-8").write(head + MARKER + "\n\n" + tables)
    print(f"附录 C：{n_terms} 个术语；附录 B：{n_err} 行常见错误（第 {min(res)}–{max(res)} 章）")


if __name__ == "__main__":
    main()
