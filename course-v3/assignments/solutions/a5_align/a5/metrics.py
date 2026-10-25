"""A5 参考实现：文字和问答的指标。编辑距离（带回溯）、CER、WER、NED、BLEU、ANLS。"""
import math


def _table(a, b):
    D = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        D[i][0] = i
    for j in range(len(b) + 1):
        D[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    return D


def edit_distance(a, b):
    """Levenshtein 距离，只保留两行。a、b 可以是字符串或列表。"""
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[-1]


def align_ops(pred, ref):
    """把 pred 变成 ref 的一组最少操作 [(op, pred 的字符, ref 的字符)]，op ∈ "=", "S", "D", "I"。
    回溯时的优先顺序：对角线（相同或替换）→ 删除 → 插入。"""
    D = _table(pred, ref)
    ops, i, j = [], len(pred), len(ref)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (pred[i - 1] != ref[j - 1]):
            ops.append(("=" if pred[i - 1] == ref[j - 1] else "S", pred[i - 1], ref[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            ops.append(("D", pred[i - 1], ""))
            i -= 1
        else:
            ops.append(("I", "", ref[j - 1]))
            j -= 1
    return ops[::-1]


def cer(preds, refs):
    """语料级：Σ 编辑距离 / Σ 参考长度。"""
    return sum(edit_distance(p, r) for p, r in zip(preds, refs)) / max(1, sum(len(r) for r in refs))


def wer(preds, refs):
    return sum(edit_distance(p.split(), r.split()) for p, r in zip(preds, refs)) / max(1, sum(len(r.split()) for r in refs))


def ned(p, r):
    n = max(len(p), len(r))
    return edit_distance(p, r) / n if n else 0.0


def _ngrams(toks, n):
    out = {}
    for i in range(len(toks) - n + 1):
        g = tuple(toks[i:i + n])
        out[g] = out.get(g, 0) + 1
    return out


def bleu(preds, refs, max_n=4):
    """语料级 BLEU：截断的 n 元组精确率的几何平均 × 简短惩罚。任何一个 p_n 为 0 时返回 0。"""
    match, total = [0] * max_n, [0] * max_n
    c, r = sum(map(len, preds)), sum(map(len, refs))
    for p, t in zip(preds, refs):
        for n in range(1, max_n + 1):
            gp, gt = _ngrams(p, n), _ngrams(t, n)
            match[n - 1] += sum(min(k, gt.get(g, 0)) for g, k in gp.items())
            total[n - 1] += max(0, len(p) - n + 1)
    if c == 0 or min(match) == 0:
        return 0.0
    log_p = sum(math.log(m / t) for m, t in zip(match, total)) / max_n
    return (1.0 if c >= r else math.exp(1 - r / c)) * math.exp(log_p)


def anls(preds, answers, tau=0.5):
    """每题取所有正确答案里最大的 1 − ned（ned ≥ tau 记 0），小写、去首尾空白后比较，再对题目平均。"""
    tot = 0.0
    for p, ans in zip(preds, answers):
        p = p.strip().lower()
        best = 0.0
        for a in ans:
            d = ned(p, a.strip().lower())
            best = max(best, 1 - d if d < tau else 0.0)
        tot += best
    return tot / max(1, len(preds))
