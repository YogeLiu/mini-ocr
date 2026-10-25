"""第 25 章起：表格。一张表 = 若干行，每行若干个单元格 {"text", "colspan", "rowspan"}，可以转成 HTML。
第 25 章只用 HTML 来演示 TEDS；第 33 章再把表格渲染成图像。"""
import numpy as np

from .text import WORDS


def sample_table(rng, n_rows=None, n_cols=None, p_span=0.15):
    """随机生成一张表：第一行是表头（单词），其余行第一列是单词、其他列是数字；
    表头以外的行，偶尔把相邻的两个单元格合并（colspan = 2）。"""
    n_rows = int(rng.integers(3, 7)) if n_rows is None else n_rows
    n_cols = int(rng.integers(2, 5)) if n_cols is None else n_cols
    rows = [[{"text": WORDS[int(rng.integers(len(WORDS)))].capitalize(), "colspan": 1, "rowspan": 1}
             for _ in range(n_cols)]]
    for _ in range(n_rows - 1):
        row, c = [], 0
        while c < n_cols:
            if c == 0:
                text = WORDS[int(rng.integers(len(WORDS)))]
            else:
                text = f"{rng.integers(0, 1000)}" if rng.random() < 0.7 else f"{rng.random() * 100:.1f}"
            span = 2 if (c > 0 and c + 1 < n_cols and rng.random() < p_span) else 1
            row.append({"text": text, "colspan": span, "rowspan": 1})
            c += span
        rows.append(row)
    return rows


def to_html(rows):
    out = ["<table>"]
    for r in rows:
        out.append("<tr>")
        for cell in r:
            attr = f' colspan="{cell["colspan"]}"' if cell["colspan"] > 1 else ""
            attr += f' rowspan="{cell["rowspan"]}"' if cell["rowspan"] > 1 else ""
            out.append(f"<td{attr}>{cell['text']}</td>")
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)
