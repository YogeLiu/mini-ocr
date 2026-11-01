"""第 18–25 章共用的后纠错评测；默认不抽取、不剥壳，原样评分。

EM_raw 比原串，EM_norm 比 normalize_latex；规范化不是数学等价证明。
CER 默认沿用第 0 章 normalize=True、unit='char'（规范 token 拼接后的字符），
是总编辑距离/总参考字符数，可大于 1；mean_edit_distance 使用同一口径。
broken_rate 只看原串无错样本的规范化改坏；fix_rate 看原串有错样本是否规范化
全对，所以只多了空格的输入即使照抄也可能算修好。parse_rate 只是 mathtext
子集可解析率，不是完整 LaTeX 编译率，更不是正确率。
比例给 Wilson 95% 区间；CER 按样本有放回 bootstrap，固定 seed、百分位区间。
区间只描述这个合成样本分布的抽样波动，不能证明真实 OCR 泛化或提示间显著性。
"""
from functools import lru_cache
import re

import numpy as np

from miniocr.data.postcorrect import ERROR_TYPES, build_split
from miniocr.eval.latex_norm import normalize_latex, normalized_tokens
from miniocr.eval.metrics import edit_distance, error_rate, ratio, units
from miniocr.eval.render import can_parse
from miniocr.train.pretrain import wilson_interval

_THINK = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
_PREFIX = re.compile(r"^(?:corrected(?:\s+formula)?|output|修正后|纠正后|输出)\s*[:：]\s*", re.I)
_WRAPPERS = (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$"))
# 先消费 LaTeX 命令：\$ 不是定界符，\\ 后的 $ 则仍是定界符。
_MATH_TOKEN = re.compile(r"\\[a-zA-Z]+|\\.|\$\$|\$", re.DOTALL)
_LATEX_TOKEN = re.compile(r"\\[a-zA-Z]+|\\.|.", re.DOTALL)
_PROSE = re.compile(
    r"\b[A-Za-z]+(?:\s+[A-Za-z]+){2,}\b|[\u4e00-\u9fff]{3,}[。：，！？]"
    r"|^(?:explanation\b|because\b)", re.I)


def extract_formula(text):
    r"""去 think/代码围栏，按出现顺序找 $$、\[、\(、$ 的完整数学片段。

    \$ 不作定界符；块外只有空白或 , . ; 。时，所有片段用单个空格连接；
    否则只取第一个片段（不按定界符种类排序）。片段内跨行折叠为空格。
    无片段时逐行剥 Corrected: 等标签，丢弃像自然语言的行，再用空格连接
    剩余非空行：没有反斜杠、^、_、{、}，且含至少三个连续英文单词、
    至少三个连续汉字加中文标点，或以 Explanation/Because 开头，视作说明。
    最后去一个未转义的句末 . , ; 。；保留反斜杠
    命令（包括 \\）及 \left./\right./\middle. 的不可见定界符。

    这是启发式：不理解语义、嵌套、注释或 verbatim；首块可能是引用/中间步骤。
    块外等式也会阻止拼接且被丢弃；不会恢复块间被丢弃的逗号或 LaTeX 换行。
    未配对定界符可能残留；带 LaTeX 的解释会留下，纯字母公式可能误判为说明。
    裸句末标点也可能有数学含义；不修括号、不证明正确，必须保留原输出。
    OCR 盲区：图上真实存在的句末标点也会被删除；目标若保留它，正确转写仍可能
    被判为不匹配。为保持已报告实验的冻结口径，默认行为不变，不应用抽取分数替代 raw。
    未闭合 think 块及其后续文本保守地丢弃。
    """
    text = _THINK.sub("", text)
    text = re.sub(r"<think\b[^>]*>.*$", "", text, flags=re.S | re.I)
    text = re.sub(r"```[^\n`]*\n?", "", text).strip()
    tokens = list(_MATH_TOKEN.finditer(text))
    closers = dict(_WRAPPERS)
    fragments, outside = [], []
    i = end = 0
    while i < len(tokens):
        token = tokens[i]
        right = closers.get(token.group())
        closing = next((j for j in range(i+1, len(tokens)) if tokens[j].group() == right), None) if right else None
        if closing is None:
            i += 1
            continue
        outside.append(text[end:token.start()])
        fragments.append(re.sub(r"\s*\n\s*", " ", text[token.end():tokens[closing].start()]).strip())
        end = tokens[closing].end()
        i = closing + 1
    outside.append(text[end:])
    if fragments:
        formula = " ".join(fragments) if re.fullmatch(r"[\s,.;。]*", "".join(outside)) else fragments[0]
    else:
        lines = []
        for line in text.splitlines():
            line = line.strip()
            while _PREFIX.match(line):
                line = _PREFIX.sub("", line).strip()
            if line and not (not re.search(r"[\\^_{}]", line) and _PROSE.search(line)):
                lines.append(line)
        formula = " ".join(lines)
    tail = _LATEX_TOKEN.findall(formula)
    if tail and tail[-1] in (".", ",", ";", "。"):
        invisible_delimiter = (tail[-1] == "." and len(tail) > 1
                               and tail[-2] in (r"\left", r"\right", r"\middle"))
        if not invisible_delimiter:
            formula = formula[:-1].rstrip()
    return formula


def strip_final_punct_tokens(tokens):
    r"""返回新 token 列表，去掉末尾连续的独立 . , ; token，不修改输入。

    输入应为规范化后的 token 序列；中间标点、中文句号及命令 token
    （如 \.、\ldots）不删。不判断标点的数学含义，仅供错例诊断。
    """
    tokens = list(tokens)
    while tokens and tokens[-1] in (".", ",", ";"):
        tokens.pop()
    return tokens


def em_ignoring_final_punct(target, pred):
    """双方先规范化，再去掉末尾全部 . , ; token，返回是否完全相同。

    非官方指标，只用于错例分析；官方指标仍是 EM_norm。不抽取公式，
    不证明数学等价；空序列与纯句末标点序列在此诊断下可相等。
    """
    return (strip_final_punct_tokens(normalized_tokens(target))
            == strip_final_punct_tokens(normalized_tokens(pred)))


def format_issues(text):
    """返回原始输出问题名元组；empty 指原输出 strip 后为空。

    has_explanation：多条非空行、前缀标签、常见解释词或连续英文单词/中文句子。
    围栏和合法多行公式也可能被标记；公式中的自然语言可能误报，这是格式启发式。
    """
    s = text.strip()
    issues = []
    if not s:
        issues.append("empty")
    if (len([x for x in s.splitlines() if x.strip()]) > 1 or _PREFIX.match(s)
            or re.search(r"\b(?:here|because|formula|corrected|therefore)\b|[A-Za-z]{2,}\s+[A-Za-z]{2,}|[\u4e00-\u9fff]{3,}[。：，]", s, re.I)):
        issues.append("has_explanation")
    if re.search(r"\$.*\$", s, re.S):
        issues.append("dollar_wrapped")
    if "```" in s:
        issues.append("code_fence")
    if re.search(r"</?think\b", s, re.I):
        issues.append("think_block")
    return tuple(issues)


def _proportion(successes, count):
    return dict(value=successes/count if count else None, successes=successes,
                count=count, interval=wilson_interval(successes, count))


@lru_cache(maxsize=32768)
def _parses(pred):
    return can_parse(pred)


@lru_cache(maxsize=32768)
def _pair(target, pred, normalize):
    ref_units, pred_units = units(target, "char", normalize), units(pred, "char", normalize)
    return (target == pred, normalize_latex(target) == normalize_latex(pred),
            0 if ref_units == pred_units else edit_distance(ref_units, pred_units), len(ref_units), _parses(pred))


@lru_cache(maxsize=32)
def _cer(refs, preds, normalize):
    # 直接复用第 0 章实现；缓存避免同一份原串/抽取结果反复计算。
    return error_rate(refs, preds, unit="char", normalize=normalize)


def score_postcorrect(examples, preds, *, normalize=True, bootstrap_samples=1000, bootstrap_seed=0):
    """返回 {'rows': 每样本字典列表, 'summary': 汇总字典}，不自动 extract_formula。

    所有比例为 {value, successes, count, interval}，零分母 value=None；CER 为
    {value, interval, bootstrap_samples, bootstrap_seed, normalize}。空语料 CER
    沿用 error_rate 的 0，区间为 (None,None)。bootstrap_samples=0 可禁用区间。
    按类型分桶允许一条进多个桶。preds 数量必须完全对应 examples。
    """
    examples, preds = list(examples), list(preds)
    if len(examples) != len(preds):
        raise ValueError("预测数量必须与样本数量一致，缺失输出请补空字符串")
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples 必须非负")
    rows = []
    for ex, pred in zip(examples, preds):
        raw, norm, dist, length, parses = _pair(ex.target, pred, normalize)
        rows.append(dict(id=ex.id, split=ex.split, source=ex.source, target=ex.target,
                         pred=pred, is_clean=ex.is_clean, error_types=ex.error_types,
                         em_raw=raw, em_norm=norm, edit_distance=dist, ref_length=length,
                         parses=parses, format_issues=format_issues(pred)))
    n = len(rows)
    summary = {key: _proportion(sum(r[key] for r in rows), n) for key in ("em_raw", "em_norm")}
    clean = [r for r in rows if r["is_clean"]]
    noisy = [r for r in rows if not r["is_clean"]]
    summary.update(n=n, broken_rate=_proportion(sum(not r["em_norm"] for r in clean), len(clean)),
                   fix_rate=_proportion(sum(r["em_norm"] for r in noisy), len(noisy)),
                   parse_rate=_proportion(sum(r["parses"] for r in rows), n),
                   mean_edit_distance=sum(r["edit_distance"] for r in rows)/n if n else None)
    interval = (None, None)
    if n and bootstrap_samples:
        rng = np.random.default_rng(bootstrap_seed)
        ds = np.array([r["edit_distance"] for r in rows])
        ls = np.array([r["ref_length"] for r in rows])
        values = []
        # 每次只生成 n 个下标，避免 1000×全语料的临时矩阵。
        for _ in range(bootstrap_samples):
            idx = rng.integers(0, n, n)
            values.append(ratio(int(ds[idx].sum()), int(ls[idx].sum())))
        interval = tuple(float(v) for v in np.quantile(values, [.025, .975]))
    summary["cer"] = dict(value=_cer(tuple(e.target for e in examples), tuple(preds), normalize),
                          interval=interval, bootstrap_samples=bootstrap_samples,
                          bootstrap_seed=bootstrap_seed, normalize=normalize)
    summary["by_error_type"] = {}
    for kind in ERROR_TYPES:
        bucket = [r for r in noisy if kind in r["error_types"]]
        summary["by_error_type"][kind] = _proportion(sum(r["em_norm"] for r in bucket), len(bucket))
    return dict(rows=rows, summary=summary)


def copy_baseline(examples, **score_kwargs):
    """什么都不改：pred=source，返回与 score_postcorrect 相同结构。"""
    examples = list(examples)
    return score_postcorrect(examples, [e.source for e in examples], **score_kwargs)


def oracle_baseline(examples, **score_kwargs):
    """答案上界：pred=target；parse_rate 仍可能低于 1（mathtext 子集限制）。"""
    examples = list(examples)
    return score_postcorrect(examples, [e.target for e in examples], **score_kwargs)


def evaluate(predict_fn, split="val", n=None, allow_test=False, *, seed=0, formulas=None,
             config=None, n_val=5000, n_test=5000, normalize=True, bootstrap_samples=1000, bootstrap_seed=0):
    """predict_fn 接收 source 字符串列表、返回等长预测列表；默认 val。

    test 必须显式 allow_test=True（第 0.10.1 节：选型冻结后，预先登记的候选各评一次）。
    不根据测试结果改设置；开发期间不看 test 错例，最终报告定稿后才可解封。
    守卫在读数据和调用 predict_fn 之前执行；评测协议由实验者遵守，不用磁盘全局状态模拟锁。
    """
    if split == "test" and not allow_test:
        raise ValueError("第 0.10.1 节评测协议：选型冻结后，预先登记的候选在测试集上各评一次，不据此改设置；须显式 allow_test=True")
    kwargs = {} if config is None else {"config": config}
    examples = build_split(split, n, seed, formulas=formulas, n_val=n_val, n_test=n_test, **kwargs)
    return score_postcorrect(examples, predict_fn([e.source for e in examples]), normalize=normalize,
                             bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed)
