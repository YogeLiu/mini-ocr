"""第二部分共用的 LaTeX 语料（只读本地缓存，不读取 image 列）。

标准划分协议 latex-v1：清洗后，按第 0 章规范化结果分组；解析失败则用
清洗后的原串作键。按 SHA-256(seed + 分组键) 排序，依次取完整组直到
val、test 各达到约 5000 条，剩下的组属于 train。边界组不会拆开，故条数
可能略多。固定数据快照和 seed=0 时，划分与输入顺序、n_train 无关；
扩大训练集只取 train 的更长前缀。不能先截取原始语料再划分。

验证集用于实验；测试集留到报告定稿，不在本章 tour 中评测。
没有缓存时 load_formulas 返回内置样例；它们只供流程检查，不能代表真实评测。
"""
import hashlib
import os
from pathlib import Path
import random
import re
import warnings

from miniocr.eval.latex_norm import (
    LatexSyntaxError, apply_token_rules, parse_latex, serialize, tokenize_latex,
)

SAMPLE_FORMULAS = [
    r"x^2+y^2=z^2", r"x^{2}+y^{2}=z^{2}", r"\frac{a+b}{\sqrt{x^2+1}}",
    r"\frac{a}{b}", r"\dfrac12", r"\frac{1}{2}", r"x_{i}^{2}",
    r"\sqrt{x}", r"\sqrt[3]{x+1}", r"\mathrm{d}x", r"\alpha+\beta=\gamma",
    r"\sum_{i=1}^{n}i", r"\int_0^1 x\,\mathrm{d}x", r"e^{i\pi}+1=0",
    r"\begin{pmatrix}a&b\\c&d\end{pmatrix}", r"\mathbb{R}",
    r"\left(x+y\right)", r"(x+y)", r"a_{i,j,k}+b_{longindex}",
    r"\text{面积}=ab", r"\text{模型}+\text{文字}", r"\mathbf{A}x=b",
    r"\lim_{n\to\infty}a_n", r"\binom{n}{k}", r"f(x)=123x+4",
    r"\log(xy)=\log x+\log y", r"\hat{x}+\bar{y}", r"a\leq b",
]

_OUTER = re.compile(
    r"^\\begin\s*\{\s*(align\*?|equation\*?|gather\*?|multline\*?|eqnarray\*?|displaymath|math)\s*\}"
    r"(.*)\\end\s*\{\s*\1\s*\}$", re.DOTALL,
)
_ENV = re.compile(r"\\(begin|end)\s*\{\s*([^{}]+?)\s*\}")


def clean_formula(text):
    """去掉完整的外层公式环境，统一空白；保留矩阵等内部结构。空值返回空串。"""
    text = (text or "").strip()
    while match := _OUTER.fullmatch(text):
        # 两个相邻环境不算一个外壳：第一个环境结束后还有文本就不能剥。
        depth, wraps = 0, True
        for env in _ENV.finditer(text):
            depth += 1 if env[1] == "begin" else -1
            if depth == 0 and env.end() != len(text):
                wraps = False
                break
        if not wraps or depth != 0:
            break
        text = match[2].strip()
    return " ".join(text.split())


def normalization_key(text):
    """与 normalize_latex 的成功路径相同；解析失败时退回清洗原串。"""
    text = clean_formula(text)
    try:
        return " ".join(serialize(parse_latex(apply_token_rules(tokenize_latex(text)))))
    except (LatexSyntaxError, RecursionError):
        return text


def _parquet_paths():
    """固定选择一个快照（优先本地 refs/main），避免多快照重复读数据。"""
    root = Path(os.environ.get("HF_HUB_CACHE", Path(os.environ.get(
        "HF_HOME", Path.home() / ".cache/huggingface")) / "hub")).expanduser()
    repo = root / "datasets--OleehyO--latex-formulas"
    ref = repo / "refs/main"
    candidates = []
    if ref.is_file():
        candidates.append(repo / "snapshots" / ref.read_text().strip())
    candidates.extend(sorted((repo / "snapshots").glob("*")))
    for snapshot in candidates:
        paths = sorted((snapshot / "cleaned_formulas").glob("*.parquet"))
        if paths:
            return paths
    return []


def corpus_available():
    """本地是否有 cleaned_formulas 的 parquet；不访问网络。"""
    return bool(_parquet_paths())


def load_formulas(n=None, seed=0):
    """读取、清洗并按 seed 打乱公式，返回至多 n 条；None 表示全部。

    只读取 latex_formula 列。缺缓存时警告并返回 SAMPLE_FORMULAS 的清洗样本。
    同一快照下较小 n 是较大 n 的前缀；标准划分始终读取全部再分组。
    """
    if n is not None and n < 0:
        raise ValueError("n 必须非负或为 None")
    paths = _parquet_paths()
    if paths:
        import pyarrow.parquet as pq
        texts = []
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(columns=["latex_formula"]):
                texts.extend(s for raw in batch.column(0).to_pylist() if (s := clean_formula(raw)))
    else:
        warnings.warn("本地 LaTeX 缓存不可用，降级使用 SAMPLE_FORMULAS。", stacklevel=2)
        texts = [s for raw in SAMPLE_FORMULAS if (s := clean_formula(raw))]
    random.Random(seed).shuffle(texts)
    return texts if n is None else texts[:n]


def standard_split(n_train=20000, n_val=5000, n_test=5000, seed=0, formulas=None):
    """返回 dict(train, val, test)，各值为公式列表，遵循模块的 latex-v1 协议。

    formulas 可注入测试语料，完全不读缓存。小语料需显式缩小 n_val/n_test；
    无缓存且用默认留出条数时自动各留约七分之一的样例，避免训练集为空。
    seed/n_val/n_test 是协议参数，课程正式实验请保持默认值。
    """
    if n_train is not None and n_train < 0 or n_val < 0 or n_test < 0:
        raise ValueError("划分条数必须非负，n_train 也可为 None")
    if formulas is None:
        small = not corpus_available()
        formulas = load_formulas(seed=seed)
        if small and n_val == n_test == 5000:
            n_val = n_test = max(1, len(formulas) // 7)
    groups = {}
    for raw in formulas:
        if text := clean_formula(raw):
            groups.setdefault(normalization_key(text), []).append(text)
    ordered = sorted(groups, key=lambda key: (
        hashlib.sha256(f"latex-v1\0{seed}\0{key}".encode()).digest(), key))
    out = {"train": [], "val": [], "test": []}
    for key in ordered:
        name = "val" if len(out["val"]) < n_val else "test" if len(out["test"]) < n_test else "train"
        out[name].extend(sorted(groups[key]))
    if n_train is not None:
        out["train"] = out["train"][:n_train]
    return out


def right_positions(tok, text):
    r"""匹配嵌套的 \left/\right，仅保留圆括号配对且 right 是完整 token 的位置。

    返回 (距离, right 目标下标, 左圆括号所在 token 下标, 全文 ID)。
    距离是左圆括号所在 token 与 right token 之间的 token 数。
    使用完整文本的字节偏移，绝不重新编码截断前缀。
    """
    ids = tok.encode(text, allow_special=False)
    spans, pos = [], 0
    for t in ids:
        end = pos + len(tok.vocab[t])
        spans.append((pos, end))
        pos = end
    stack, result = [], []
    # LaTeX 命令名由字母组成，不能把 \leftarrow 等命令的前缀当作括号命令。
    for m in re.finditer(r"\\(left|right)(?![a-zA-Z])\s*(\\[{}]|[^\s])", text):
        if m[1] == "left":
            stack.append(m)
        elif stack:
            left = stack.pop()
            if left[2] != "(" or m[2] != ")":
                continue
            begin = len(text[:m.start()+1].encode("utf-8"))
            target = next((i for i, span in enumerate(spans) if span == (begin, begin+5)), None)
            opening = len(text[:left.start(2)].encode("utf-8"))
            source = next((i for i, (a, b) in enumerate(spans) if a <= opening < b), None)
            if target is not None and source is not None:
                result.append((target-source-1, target, source, ids))
    return result
