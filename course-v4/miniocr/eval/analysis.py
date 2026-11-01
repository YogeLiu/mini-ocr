"""第 22 章共用的误差分析工具：bootstrap、配对比较、公式分桶、错例分类、泄漏近似与 test 窥视模拟。

约定：
- 所有区间只描述本次抽样（样本级有放回 bootstrap）的波动，不能代替真实 OCR 分布上的显著性。
- 比较两个模型时必须使用同一批样本、同一顺序；这里用样本下标对齐，不做 id 猜测。
- 错例分类只看一条输出，优先级固定：格式 > 不可解析 > 原样未改 > 干净样本被改坏 > 漏/多/替换。
  分类结果是启发式标签，用于找假设，不是人工标注的金标准。
"""
import re

import numpy as np

from miniocr.data.dedup import MinHasher, lsh_candidates, shingles, jaccard
from miniocr.eval.latex_norm import normalize_latex, normalized_tokens
from miniocr.eval.metrics import align
from miniocr.eval.postcorrect_eval import format_issues
from miniocr.eval.render import can_parse

FORMULA_BUCKETS = ("矩阵/数组", "分式", "上下标", "其他")
LENGTH_BUCKETS = ("短(<30字符)", "中(30-59字符)", "长(>=60字符)")
ERROR_CLASSES = ("correct", "format", "unparsable", "unchanged", "over-correction",
                 "missing", "extra", "substitution", "mixed")

_MATRIX = re.compile(r"\\begin\{(?:[pbvBV]?matrix|array|smallmatrix|cases)\}")
_FRAC = re.compile(r"\\(?:d|t)?frac\b|\\over\b")
_SCRIPT = re.compile(r"[\^_]")


def formula_bucket(target):
    """互斥的公式类型桶：矩阵/数组 > 分式 > 上下标 > 其他（按出现优先级）。"""
    if _MATRIX.search(target):
        return "矩阵/数组"
    if _FRAC.search(target):
        return "分式"
    if _SCRIPT.search(target):
        return "上下标"
    return "其他"


def length_bucket(target):
    """按参考串字符数分三档；边界固定，不依赖数据分位数。"""
    n = len(target)
    return LENGTH_BUCKETS[0] if n < 30 else LENGTH_BUCKETS[1] if n < 60 else LENGTH_BUCKETS[2]


def bootstrap_mean(values, *, samples=1000, seed=0):
    """样本均值与百分位 95% bootstrap 区间；空样本返回 None。"""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return dict(value=None, interval=(None, None), samples=samples, seed=seed)
    if samples == 0:
        return dict(value=float(x.mean()), interval=(None, None), samples=0, seed=seed)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, (samples, x.size))].mean(axis=1)
    return dict(value=float(x.mean()), interval=tuple(float(v) for v in np.quantile(means, [.025, .975])),
                samples=samples, seed=seed)


def paired_bootstrap_diff(a, b, *, samples=1000, seed=0):
    """逐样本配对差 d=b-a 的均值与 bootstrap 区间。

    同一下标抽取同一批样本，因此共同的题目难度被抵消。返回 p_b_not_better，
    即 bootstrap 中 d<=0 的比例：它是描述性的抽样比例，不是严格的假设检验 p 值。
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("配对比较要求两个等长一维非空数组")
    d = b - a
    if samples == 0:
        return dict(diff=float(d.mean()), interval=(None, None), p_b_not_better=None, n=int(d.size))
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, (samples, d.size))].mean(axis=1)
    return dict(diff=float(d.mean()),
                interval=tuple(float(v) for v in np.quantile(means, [.025, .975])),
                p_b_not_better=float((means <= 0).mean()), n=int(d.size), samples=samples, seed=seed)


def error_class(target, source, pred, *, is_clean):
    """给一条原始输出打一个互斥标签，顺序见模块文档。

    correct 使用 normalize_latex 判定；format 表示输出含解释、代码围栏、美元符等包装；
    missing/extra/substitution 由规范 token 的编辑操作决定，混合操作记为 mixed。
    """
    if normalize_latex(target) == normalize_latex(pred):
        return "correct"
    if format_issues(pred):
        return "format"
    if not can_parse(pred):
        return "unparsable"
    if pred.strip() == source.strip() and source.strip() != target.strip():
        return "unchanged"
    if is_clean:
        return "over-correction"
    kinds = {op for op, _, _ in align(normalized_tokens(target), normalized_tokens(pred)) if op != "="}
    if kinds <= {"D"}:
        return "missing"
    if kinds <= {"I"}:
        return "extra"
    if kinds <= {"S"}:
        return "substitution"
    return "mixed"


def error_table(rows):
    """rows 为 dict(target, source, pred, is_clean)；返回 {类别: 计数}，类别顺序固定。"""
    counts = {k: 0 for k in ERROR_CLASSES}
    for r in rows:
        counts[error_class(r["target"], r["source"], r["pred"], is_clean=r["is_clean"])] += 1
    return counts


def near_overlap(reference, queries, *, threshold=.8, num_perm=128, bands=16, rows=8, n=3, seed=0):
    """对每条 query 找 reference 中是否存在 Jaccard>=threshold 的近重复（LSH 召回 + 精确核验）。

    LSH 可能漏召回，因此结果是下界。返回 dict(hits=命中条数, checked=查询条数, pairs=[(q, r, J)])。
    shingle 基于规范键，与第 12 章 near_dedup 同一口径。
    """
    reference, queries = list(reference), list(queries)
    if not reference or not queries:
        return dict(hits=0, checked=len(queries), pairs=[])
    texts = reference + queries
    sets = [shingles(t, n) for t in texts]
    hasher = MinHasher(num_perm, seed)
    sigs = np.array([hasher.signature(s) for s in sets], dtype=np.uint64).reshape(-1, num_perm)
    offset = len(reference)
    hit_queries, pairs = set(), []
    for i, j in sorted(lsh_candidates(sigs, bands, rows)):
        a, b = (i, j) if i < j else (j, i)
        if a >= offset or b < offset:
            continue  # 只关心 query 与 reference 之间的配对
        q, r = b, a
        score = jaccard(sets[q], sets[r])
        if score >= threshold:
            hit_queries.add(q - offset)
            pairs.append((q - offset, r, round(score, 4)))
    return dict(hits=len(hit_queries), checked=len(queries), pairs=pairs)


def peeking_simulation(*, true_accuracy=.7, n_val=500, n_test=500, looks=(1, 2, 5, 10, 20),
                       trials=2000, seed=0):
    """模拟：候选模型真实准确率完全相同，比较三种报告方式的偏差。

    - val_select：用 val 选一个候选，只报告它的 test 准确率（无偏）。
    - test_best_of_k：对 k 个候选都看 test，报告其中最高的 test 准确率（偏乐观）。
    返回每个 k 的平均报告值与相对真值的偏差；真值未改变，偏差完全来自反复看 test。
    """
    rng = np.random.default_rng(seed)
    out = []
    for k in looks:
        val = rng.binomial(n_val, true_accuracy, size=(trials, k)) / n_val
        test = rng.binomial(n_test, true_accuracy, size=(trials, k)) / n_test
        picked_by_val = test[np.arange(trials), val.argmax(axis=1)]
        best_of_test = test.max(axis=1)
        out.append(dict(k=k, val_select_mean=float(picked_by_val.mean()),
                        test_best_of_k_mean=float(best_of_test.mean()),
                        optimism=float(best_of_test.mean() - true_accuracy)))
    return dict(true_accuracy=true_accuracy, n_val=n_val, n_test=n_test, trials=trials, rows=out)
