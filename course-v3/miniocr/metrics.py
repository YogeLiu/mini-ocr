"""评测指标。第 13 章：编辑距离、字符错误率；第 25 章：NED、WER、BLEU、ANLS、VQA 准确率。
公式的 CDM、表格的 TEDS、置信区间在 miniocr/eval/ 里。"""
import math


# >>> 练习 13.5：编辑距离与字符错误率
def edit_distance(a, b):
    """Levenshtein 距离：把序列 a 变成 b 最少需要多少次"插入、删除、替换"（每次代价 1）。
    动态规划：D[i][j] = a[:i] 变成 b[:j] 的最小代价；
        D[i][0] = i，D[0][j] = j，
        D[i][j] = min(D[i-1][j] + 1,            删除 a[i-1]
                      D[i][j-1] + 1,            插入 b[j-1]
                      D[i-1][j-1] + (a[i-1] != b[j-1]))   替换（相同则不花代价）
    只保留两行，内存 O(len(b))。a、b 可以是字符串或列表。"""
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[-1]


def cer(preds, refs):
    """字符错误率：所有样本的编辑距离之和 / 所有参考答案的字符数之和（不是逐样本的错误率再平均）。"""
    return sum(edit_distance(p, r) for p, r in zip(preds, refs)) / max(1, sum(len(r) for r in refs))
# <<<


def exact_match(preds, refs):
    """完全正确的比例。"""
    return sum(p == r for p, r in zip(preds, refs)) / max(1, len(refs))


def align_ops(a, b):
    """回溯编辑距离的动态规划表，返回把 a 变成 b 的操作列表 [(op, a 的字符, b 的字符)]，
    op ∈ {"=", "S"（替换）, "D"（删除）, "I"（插入）}。用来统计"哪个字符被认成了哪个"。"""
    n, m = len(a), len(b)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            ops.append(("=" if a[i - 1] == b[j - 1] else "S", a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            ops.append(("D", a[i - 1], ""))
            i -= 1
        else:
            ops.append(("I", "", b[j - 1]))
            j -= 1
    return ops[::-1]


# ====================================================================== 第 25 章
def ned(p, r):
    """归一化编辑距离：编辑距离 / 两者中较长的长度，在 [0, 1] 之间；两个都为空时是 0。"""
    n = max(len(p), len(r))
    return edit_distance(p, r) / n if n else 0.0


# >>> 练习 25.1：词错误率
def wer(preds, refs):
    """词错误率：按空白切成词，再算编辑距离；和 cer 一样是"总编辑距离 / 参考答案的总词数"。"""
    return sum(edit_distance(p.split(), r.split()) for p, r in zip(preds, refs)) / max(1, sum(len(r.split()) for r in refs))
# <<<


def _ngrams(toks, n):
    out = {}
    for i in range(len(toks) - n + 1):
        g = tuple(toks[i:i + n])
        out[g] = out.get(g, 0) + 1
    return out


# >>> 练习 25.2：BLEU
def bleu(preds, refs, max_n=4):
    """语料级 BLEU（Papineni 等，2002），preds、refs 是 token 列表的列表（每条只有一个参考答案）。

    1. 对 n = 1..max_n：截断的 n 元组精确率 p_n = Σ min(预测里的次数, 参考里的次数) / Σ 预测里的 n 元组数；
    2. 简短惩罚 BP = 1（预测总长 c ≥ 参考总长 r）或 exp(1 − r / c)；
    3. BLEU = BP · exp(平均 log p_n)。任何一个 p_n 为 0 时 BLEU 是 0。
    """
    match = [0] * max_n
    total = [0] * max_n
    c = sum(len(p) for p in preds)
    r = sum(len(t) for t in refs)
    for p, t in zip(preds, refs):
        for n in range(1, max_n + 1):
            gp, gt = _ngrams(p, n), _ngrams(t, n)
            match[n - 1] += sum(min(k, gt.get(g, 0)) for g, k in gp.items())
            total[n - 1] += max(0, len(p) - n + 1)
    if c == 0 or min(match) == 0:
        return 0.0
    log_p = sum(math.log(m / t) for m, t in zip(match, total)) / max_n
    bp = 1.0 if c >= r else math.exp(1 - r / c)
    return bp * math.exp(log_p)
# <<<


# >>> 练习 25.3：ANLS
def anls(preds, answers, tau=0.5):
    """平均归一化 Levenshtein 相似度（Biten 等，2019，ST-VQA / DocVQA 的指标）。

    每个问题可以有多个正确答案（answers[i] 是字符串列表）。先都转成小写、去掉首尾空白；
    对每个正确答案算 s = 1 − ned(预测, 答案)，若 ned ≥ tau 则 s = 0；取所有正确答案里最大的 s；最后对问题求平均。
    """
    tot = 0.0
    for p, ans in zip(preds, answers):
        p = p.strip().lower()
        best = 0.0
        for a in ans:
            d = ned(p, a.strip().lower())
            best = max(best, 1 - d if d < tau else 0.0)
        tot += best
    return tot / max(1, len(preds))
# <<<


def vqa_accuracy(pred, human_answers):
    """VQAv2 / TextVQA 的准确率：10 个人的答案里有 k 个和预测相同，得分 min(k / 3, 1)。
    （官方实现还会对 10 个"去掉一个人"的子集取平均，并做大小写、标点、冠词的归一化；这里只做小写和去空白。）"""
    p = pred.strip().lower()
    return min(sum(a.strip().lower() == p for a in human_answers) / 3, 1.0)
