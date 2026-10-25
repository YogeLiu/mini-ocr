"""第 14 章起：混合语料。VLM 做 OCR 时要输出三种"语言"：普通文字、LaTeX 公式、Markdown（标题、列表、表格）。
这里用前几章的生成器拼出一份这样的语料，用于分词（第 14 章）和语言模型预训练（第 16 章）。

    sample_document(rng)   一小段"文档"：若干行，混合三种内容，以换行分隔
    mixed_corpus(seed, n)  n 段文档拼成的长字符串（段与段之间空一行）
    text_vocab()           第 16 章起语言模型（以及第 22 章的 VLM）用的字符表：可打印 ASCII + 换行 + 4 个特殊符号
"""
import numpy as np

from .latex import gen_formula, to_latex
from .text import WORDS, sample_line_text
from ..utils import Vocab

TEXT_SPECIALS = ("<pad>", "<bos>", "<eos>", "<img>")      # 补齐、开始、结束、图像占位（第 22 章）
TEXT_CHARS = "\n" + "".join(chr(c) for c in range(32, 127))


def text_vocab():
    """固定的字符表（100 个）：不依赖语料，所以第 16 章预训练的 GPT 和第 22 章的 VLM 用的编号永远一致。"""
    return Vocab(TEXT_CHARS, specials=TEXT_SPECIALS)


def _sentence(rng):
    n = int(rng.integers(4, 11))
    ws = [WORDS[int(rng.integers(len(WORDS)))] for _ in range(n)]
    ws[0] = ws[0].capitalize()
    return " ".join(ws) + str(rng.choice([".", ".", ".", "?", "!"]))


def _formula(rng):
    return to_latex(gen_formula(rng, level=int(rng.integers(1, 5))))


def _table(rng):
    cols = int(rng.integers(2, 4))
    rows = int(rng.integers(1, 4))
    head = [WORDS[int(rng.integers(len(WORDS)))].capitalize() for _ in range(cols)]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * cols) + "|"]
    for _ in range(rows):
        cells = [str(int(rng.integers(0, 1000))) if rng.random() < 0.6 else WORDS[int(rng.integers(len(WORDS)))]
                 for _ in range(cols)]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def sample_document(rng):
    """一段文档：可能有标题，然后 2–5 个块；块是句子、行内公式、独立公式、列表或表格。"""
    lines = []
    if rng.random() < 0.5:
        lines.append("# " + " ".join(WORDS[int(rng.integers(len(WORDS)))] for _ in range(int(rng.integers(1, 4)))).capitalize())
    for _ in range(int(rng.integers(2, 6))):
        r = rng.random()
        if r < 0.35:
            lines.append(_sentence(rng))
        elif r < 0.50:
            lines.append(f"{_sentence(rng)[:-1]} ${_formula(rng)}$.")
        elif r < 0.65:
            lines.append(f"$${_formula(rng)}$$")
        elif r < 0.80:
            lines += [f"- {sample_line_text(rng, max_chars=24)}" for _ in range(int(rng.integers(2, 4)))]
        elif r < 0.90:
            lines += _table(rng)
        else:
            lines.append(sample_line_text(rng))
    return "\n".join(lines)


def mixed_corpus(seed, n):
    rng = np.random.default_rng(seed)
    return "\n\n".join(sample_document(rng) for _ in range(n))
