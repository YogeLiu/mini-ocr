"""第 12 章：预训练数据；CPU、本地语料，不训练模型。

python -m book.part2_lm.ch12_pretrain_data.tour [部分号 ...]
练习 12.1、12.2 在 miniocr.data.dedup；本脚本是实验对照。
"""
import argparse
from collections import Counter, defaultdict
from functools import cached_property
from itertools import combinations
import math
import os
from pathlib import Path
import random
import re
import time
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "2"

import numpy as np
import torch

from miniocr.data.latex_corpus import (
    SAMPLE_FORMULAS, _parquet_paths, clean_formula, corpus_available,
    load_formulas, normalization_key, standard_split,
)
from miniocr.data.dedup import (
    MinHasher, estimate_jaccard, exact_dedup, jaccard, lsh_candidates,
    lsh_probability, lsh_threshold, near_dedup, shingles,
)
from miniocr.data.latex_lm import encode_formulas, train_tokenizer
from miniocr.data.token_file import StaleCacheError, load_token_file, write_token_file
from miniocr.eval.latex_norm import is_well_formed, tokenize_latex
from miniocr.eval.render import can_parse

OUT = Path("outputs/book/ch12")
COMMAND = re.compile(r"\\[A-Za-z]+")
MATRIX = re.compile(r"\\begin\s*\{(?:[pbBvV]?matrix|smallmatrix|array|cases)\*?\}")


def line(**values):
    print("  " + "；".join(f"{k}={v:.4f}" if isinstance(v, (float, np.floating))
                          else f"{k}={v}" for k, v in values.items()), flush=True)


def short(text, limit=130):
    text = text.replace("\n", " ")
    return text[:limit] + ("…" if len(text) > limit else "")


def structures(text):
    commands = set(COMMAND.findall(text))
    return {"分式": bool(commands & {r"\frac", r"\dfrac", r"\tfrac"}),
            "上下标": "_" in text or "^" in text, "环境": r"\begin" in commands,
            "伸缩括号": bool(commands & {r"\left", r"\right"}),
            "文本": r"\text" in commands, "汉字": bool(re.search(r"[\u4e00-\u9fff]", text)),
            "矩阵类": bool(MATRIX.search(text))}


def well_formed(text):
    try:
        return is_well_formed(text)
    except RecursionError:
        return False


class Experiment:
    """只在用到时读数据；键、解析结果和近重复结果在本次运行内复用。"""
    def __init__(self, args):
        self.args = args
        self.keys, self.parsed, self.near_results = {}, {}, {}

    @cached_property
    def small(self):
        return self.args.small or not corpus_available()

    @cached_property
    def all_texts(self):
        if self.args.small:
            warnings.warn("主动降级使用 SAMPLE_FORMULAS；仅检查流程。", stacklevel=2)
            texts = [clean_formula(t) for t in SAMPLE_FORMULAS]
            random.Random(self.args.seed).shuffle(texts)
        else:
            texts = load_formulas(seed=self.args.seed)
        line(数据来源="内置样例" if self.small else "本地parquet", 全语料条数=len(texts))
        return texts

    @cached_property
    def sample(self):
        texts = self.all_texts[:self.args.sample]
        line(探索样本条数=len(texts), 抽样种子=self.args.seed)
        return texts

    def key(self, text):
        if text not in self.keys:
            self.keys[text] = normalization_key(text)
        return self.keys[text]

    @cached_property
    def split(self):
        # 协议种子固定为 0；--seed 只影响探索抽样和 MinHash。
        extra = dict(n_val=max(1, len(self.all_texts)//7),
                     n_test=max(1, len(self.all_texts)//7)) if self.small else {}
        return standard_split(formulas=self.all_texts, **extra)

    @cached_property
    def tokenizer(self):
        tok = train_tokenizer(self.split["train"])
        line(分词器训练条数=len(self.split["train"]), BPE合并=len(tok.merges), 词表=len(tok))
        return tok

    @cached_property
    def unique(self):
        return exact_dedup(self.sample)[0]

    def near(self, texts):
        key = tuple(texts)
        if key not in self.near_results:
            self.near_results[key] = near_dedup(texts, threshold=.8, num_perm=128,
                                               bands=16, rows=8, n=3, seed=self.args.seed)
        return self.near_results[key]

    def parse(self, text):
        if text not in self.parsed:
            self.parsed[text] = can_parse(text)
        return self.parsed[text]


def part1(exp):
    texts = exp.sample
    lengths = np.array([len(t) for t in texts])
    token_lengths = np.array([len(exp.key(t).split()) for t in texts])
    for name, values in (("字符", lengths), ("规范token", token_lengths)):
        q = np.quantile(values, [0, .5, .9, .99, 1])
        line(长度口径=name, 最小=q[0], p50=q[1], p90=q[2], p99=q[3], 最大=q[4])
    counts, edges = np.histogram(token_lengths, bins=[0, 16, 32, 64, 128, 256, 512, np.inf])
    for left, right, count in zip(edges, edges[1:], counts):
        line(规范token区间=f"[{left:.4f},{right:.4f})", 条数=int(count),
             直方图="█"*round(36*count/max(1, counts.max())))
    commands = Counter(c for t in texts for c in COMMAND.findall(t))
    line(命令频率前20=commands.most_common(20), 仅出现一次的命令种数=sum(v == 1 for v in commands.values()))
    flags = [structures(t) for t in texts]
    for name in flags[0]:
        line(结构=name, 比例=sum(f[name] for f in flags)/len(texts))
    examples = {"极短": sorted(texts, key=len)[:2], "极长": sorted(texts, key=len, reverse=True)[:2],
                "花括号不配对": [t for t in texts if not balanced_braces(t)][:2],
                "可疑字符或非公式文本": [t for t in texts if re.search(
                    r"https?://|[\x00-\x08�]|[A-Za-z]{16,}|(?:the|where|proof)\s", t)][:2]}
    for label, items in examples.items():
        line(异常类型=label, 找到例子=len(items))
        for text in items:
            line(样例=short(text))
    print("  极长、极短和文本启发式只是检查线索，不自动判为坏数据。")


def balanced_braces(text):
    depth = 0
    for token in tokenize_latex(text):
        depth += (token == "{") - (token == "}")
        if depth < 0:
            return False
    return depth == 0


def part2(exp):
    examples = ["  x +  y  ", "{{x}}", "x^2", "x^{2}", r"\dfrac12",
                r"\frac{1}{2}", r"a\le b", r"\left(x\right)", r"x\, +\quad y", "x^2_i"]
    for text in examples:
        line(原串=text, 规范键=exp.key(text))
    groups = defaultdict(set)
    for text in exp.sample:
        groups[exp.key(text)].add(text)
    before = sum(len(tokenize_latex(t)) for t in exp.sample)
    after = sum(len(exp.key(t).split()) for t in exp.sample)
    line(原token总数=before, 规范token总数=after, 长度变化率=(after-before)/max(1, before),
         原串不同但键相同组数=sum(len(g) > 1 for g in groups.values()))
    for text in (r"\dfrac{a", r"\frac{a"):
        line(盲区原串=text, 弱检查=well_formed(text), 回退键=exp.key(text),
             等于清洗原串=exp.key(text) == clean_formula(text))
    print("  解析失败时规范键回退到清洗原串；上面两种坏写法没有统一同义命令。")


def raw_formulas(exp):
    """load_formulas 已清洗；原串基线须另读同一快照的文本列。"""
    if exp.small:
        return list(SAMPLE_FORMULAS)
    import pyarrow.parquet as pq
    return [raw for path in _parquet_paths()
            for batch in pq.ParquetFile(path).iter_batches(columns=["latex_formula"])
            for raw in batch.column(0).to_pylist() if clean_formula(raw)]


def part3(exp):
    begun = time.perf_counter()
    texts = exp.all_texts
    raw = raw_formulas(exp)
    all_keys = [exp.key(t) for t in texts]
    groups = Counter(all_keys)
    assert len(raw) == len(texts)
    for name, unique in (("原串strip", len({t.strip() for t in raw})),
                         ("clean_formula", len(set(texts))), ("规范键", len(groups))):
        line(去重口径=name, 输入条数=len(texts), 重复条数=len(texts)-unique,
             删除比例=1-unique/len(texts))
    for key, count in sorted(groups.items(), key=lambda kv: (-kv[1], kv[0]))[:4]:
        line(规范键=short(key), 组大小=count)
    line(全语料三级计数用时秒=time.perf_counter()-begun)
    # 只打乱已计算的键，不再规范化；用同一顺序的嵌套前缀比较规模效应。
    random.Random(exp.args.seed).shuffle(all_keys)
    for size in dict.fromkeys([min(n, len(all_keys)) for n in (2000, 20000, 100000)]
                             + [len(all_keys)]):
        duplicates = size - len(set(all_keys[:size]))
        line(规模效应前缀条数=size, 规范键重复条数=duplicates,
             规范键重复比例=duplicates/max(1, size))
    split = exp.split
    keys = {name: {exp.key(t) for t in items} for name, items in split.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = len(keys[a] & keys[b])
        line(标准划分比较=f"{a}/{b}", 规范键交集=overlap)
        assert overlap == 0
    shuffled = list(texts)
    random.Random(exp.args.seed).shuffle(shuffled)
    n_val = len(split["val"])
    train_keys = {exp.key(t) for t in shuffled[n_val:]}
    leaked = sum(exp.key(t) in train_keys for t in shuffled[:n_val])
    line(错误原串随机划分验证条数=n_val, 其余全作训练=len(shuffled)-n_val,
         泄漏验证样本数=leaked, 泄漏比例=leaked/max(1, n_val))
    print("  标准核验使用 train 前20000条及完整 val/test；测试集只作泄漏审计，不参与评测。")


def part4(exp):
    base = r"x_i^2+y_i^2+z_i^2=a_i^2+b_i^2"
    pairs = [("相同", base), ("改符号", base.replace("+", "-", 1)),
             ("换变量", base.replace("x", "q")), ("完全不同", r"\int_0^1\sin(t)dt")]
    for label, other in pairs:
        a, b = shingles(base, 3), shingles(other, 3)
        values = {}
        for k in (16, 64, 256):
            hasher = MinHasher(k, exp.args.seed)
            values[f"估计k{k}"] = estimate_jaccard(hasher.signature(a), hasher.signature(b))
        line(公式对=label, 左shingle数=len(a), 右shingle数=len(b), 精确J=jaccard(a, b), **values)
    rng = random.Random(exp.args.seed)
    print("  随机对来自受控人造集合：每边75个元素，交集大小随机取0–75，避免全是低相似度。")
    random_pairs = []
    for _ in range(240):
        pool = rng.sample(range(10000000), 150)
        overlap = rng.randrange(76)
        a = {str(x) for x in pool[:75]}
        b = {str(x) for x in pool[:overlap] + pool[75:150-overlap]}
        random_pairs.append((a, b))
    for k in (16, 64, 256):
        hasher = MinHasher(k, exp.args.seed)
        errors, variances = [], []
        for a, b in random_pairs:
            truth = jaccard(a, b)
            errors.append(estimate_jaccard(hasher.signature(a), hasher.signature(b))-truth)
            variances.append(truth*(1-truth)/k)
        line(k=k, 随机对数=len(errors), 有符号误差均值=float(np.mean(errors)),
             绝对误差均值=float(np.mean(np.abs(errors))), 误差标准差=float(np.std(errors)),
             理论均方根标准差=math.sqrt(float(np.mean(variances))))
    print("  理论值基于独立随机排列；通用哈希和共享同一组系数时只是近似对照。")


def part5(exp):
    texts = exp.unique
    _, info = exp.near(texts)
    for bands, rows in ((32, 4), (16, 8), (8, 16)):
        candidates = (info["candidates"] if (bands, rows) == (16, 8)
                      else lsh_candidates(info["signatures"], bands, rows))
        line(b=bands, r=rows, 理论阈值=lsh_threshold(bands, rows),
             **{f"命中概率s{s:.1f}": lsh_probability(s, bands, rows) for s in (.5, .7, .8, .9)},
             精确去重后样本数=len(texts), 实际候选对数=len(candidates), 暴力对数=len(texts)*(len(texts)-1)//2)
    print("  b*r固定时，更多段、更少行通常增加召回与候选成本；候选不等于重复。")


def part6(exp):
    texts = exp.unique
    cached = tuple(texts) in exp.near_results
    keep, info = exp.near(texts)
    repeated = [c for c in info["clusters"] if len(c) > 1]
    line(精确去重后条数=len(texts), 候选对数=len(info["candidates"]),
         核验通过对数=len(info["accepted"]), 假阳性数=len(info["candidates"])-len(info["accepted"]),
         簇数=len(info["clusters"]), 非单例簇数=len(repeated), 删除条数=len(texts)-len(keep))
    timings = info["timings"]
    line(计时来源="复用首次实际计算记录" if cached else "本部分实际计算",
         签名用时秒=timings["signature_seconds"],
         **{"LSH+核验用时秒": timings["lsh_verify_seconds"]},
         near_dedup完整用时秒=timings["total_seconds"])
    print("  签名用时含shingle构造；LSH+核验用时含合簇。完整用时不含精确去重及下方暴力诊断。")
    for group in repeated[:3]:
        line(簇大小=len(group), 保留下标=group[0])
        for i in group[:3]:
            line(成员下标=i, 公式=short(texts[i]))
    # 明确标为人造反例，避免把自动发现的近重复误称为语义等价。
    a = r"x_1+x_2+x_3+x_4+x_5+x_6=0"
    b = a[:-1] + "1"
    line(人造语义反例甲=a, 乙=b, 精确J=jaccard(shingles(a, 3), shingles(b, 3)))
    print("  两个等式右端不同；高J也会误删。连通分量还可能由相似链串起不相似的两端。")
    count = min(2000, len(texts))
    subsets = [("随机前缀", list(range(count)))]
    if count < len(texts):
        subsets.append(("较短公式诊断", sorted(sorted(range(len(texts)), key=lambda i: len(texts[i]))[:count])))
    for label, indices in subsets:
        sets = {i: shingles(texts[i], 3) for i in indices}
        truth = {(i, j) for i, j in combinations(indices, 2) if jaccard(sets[i], sets[j]) >= .8}
        missed = truth - info["candidates"]
        line(暴力子集选择=label, 暴力核验子集条数=count, 暴力对数=count*(count-1)//2,
             真近重复对数=len(truth), LSH漏对数=len(missed),
             召回率=1-len(missed)/len(truth) if truth else "无真对，不定义")
    print("  较短子集按字符长度选择，不按LSH命中选择；两个子集的召回都不代表全体语料。")


def part7(exp):
    texts = exp.sample[:2000]
    weak = [well_formed(t) for t in texts]
    parsed = [exp.parse(t) for t in texts]
    line(过滤检查条数=len(texts), 弱检查通过率=sum(weak)/len(texts), mathtext通过率=sum(parsed)/len(texts))
    for a in (False, True):
        for b in (False, True):
            line(弱检查=a, mathtext=b, 条数=sum(x == a and y == b for x, y in zip(weak, parsed)))
    categories = {"矩阵环境": lambda t: bool(MATRIX.search(t)),
                  "文本或汉字": lambda t: structures(t)["文本"] or structures(t)["汉字"],
                  "省写或别名": lambda t: bool(re.search(r"\\le(?![A-Za-z])|\\frac\s*\d", t))}
    for label, predicate in categories.items():
        indices = [i for i, t in enumerate(texts) if predicate(t)]
        line(结构类别=label, 条数=len(indices),
             mathtext失败率=sum(not parsed[i] for i in indices)/len(indices) if indices else "无样本")
    for label, text in (("合法矩阵", r"\begin{pmatrix}a&b\\c&d\end{pmatrix}"),
                        ("合法别名", r"a\le b"), ("合法省写", r"\frac12"),
                        ("坏花括号", r"\frac{a}{b")):
        line(人造核验例=label, 公式=text, 弱检查=well_formed(text), mathtext=exp.parse(text))
    for text, good, ok in list(zip(texts, weak, parsed)):
        if good and not ok:
            line(样本中弱检查通过但mathtext拒绝=short(text))
            break
    print("  弱检查通过不证明合法。mathtext不是完整LaTeX编译器；硬过滤会系统性删除某些结构，改变分布。")


def part8(exp):
    texts = exp.sample
    lengths = [len(exp.key(t).split()) for t in texts]
    median = float(np.median(lengths))
    sources = [[], []]
    for text, length in zip(texts, lengths):
        sources[int(length > median or bool(MATRIX.search(text)))].append(text)
    if not all(sources):
        line(混合实验="样本不足以构成两个非空源", 中位长度=median)
        return
    totals = [sum(len(exp.key(t).split()) for t in source) for source in sources]
    means = [total/len(source) for total, source in zip(totals, sources)]
    matrix_rates = [sum(bool(MATRIX.search(t)) for t in source)/len(source) for source in sources]
    commands = [Counter(c for t in source for c in COMMAND.findall(t)) for source in sources]
    vocabulary = sorted(set(commands[0]) | set(commands[1]))
    target_count = sum(commands[1].values())
    target = np.array([commands[1][c]/max(1, target_count) for c in vocabulary])
    for i, source in enumerate(sources):
        line(源="短公式A" if i == 0 else "长公式或矩阵B", 条数=len(source),
             规范token总数=totals[i], 平均长度=means[i])
    budget = sum(totals)
    line(期望混合token预算=budget, 目标命令分布="源B；仅作课程对照")
    for weight in (0., .1, .3, .5):
        # 按 token 预算推导期望条数，允许分数权重，不随机采样。
        counts = [(1-weight)*budget/means[0], weight*budget/means[1]]
        total_count = sum(counts)
        rates = np.array([sum(counts[i]*commands[i][c]/len(sources[i]) for i in (0, 1))
                          for c in vocabulary])
        actual = rates/max(1, rates.sum())
        line(B的token配额=weight, B的期望条数占比=counts[1]/total_count,
             混合平均长度=budget/total_count,
             矩阵条数占比=sum(counts[i]*matrix_rates[i] for i in (0, 1))/total_count,
             命令TV距离=float(np.abs(actual-target).sum()/2) if target_count and rates.sum() else "无命令，不定义",
             A重复epoch=(1-weight)*budget/totals[0], B重复epoch=weight*budget/totals[1],
             若按条数混合同一比例_B实际token占比=weight*means[1]/((1-weight)*means[0]+weight*means[1]))
    print("  epoch是期望用量/可用量，超过1才是上采样；token指规范token，不是BPE。这里不推断模型效果。")


def part9(exp):
    tok, texts = exp.tokenizer, exp.split["train"]
    version = f"{'sample' if exp.small else 'latex-v1'}/train/n={len(texts)}"
    path = OUT / ("sample_train" if exp.small else "train")
    meta = write_token_file(path, tok, texts, version)
    mapped = load_token_file(path, tok, version)
    seqs, _ = encode_formulas(tok, texts)
    expected = np.array([i for seq in seqs for i in seq], dtype=np.uint16)
    equal = np.array_equal(mapped, expected)
    assert equal
    line(token总数=len(mapped), 文件字节数=Path(str(path)+".bin").stat().st_size,
         每token字节=mapped.dtype.itemsize, cache_key前16位=meta["cache_key"][:16],
         memmap读回一致=equal, 原文UTF8字节数=meta["text_utf8_bytes"])
    other_texts = texts[:max(1, len(texts)//2)]
    other = train_tokenizer(other_texts, num_merges=min(1000, len(tok.merges)//2))
    line(错误分词器训练条数=len(other_texts), 错误分词器合并数=len(other.merges))
    for label, tokenizer, data_version in (("换分词器", other, version), ("换数据版本", tok, version+"/changed")):
        try:
            load_token_file(path, tokenizer, data_version)
        except StaleCacheError as exc:
            line(故障=label, 捕获异常=f"StaleCacheError: {exc}")
        else:
            raise AssertionError("缓存失配没有被拒绝")
    content = [i for seq in seqs[:100] for i in seq[1:-1]]
    try:
        decoded = other.decode(content)
        line(直接错解码=short(decoded))
    except KeyError as exc:
        line(直接错解码=f"编号{exc.args[0]}不在错误词表中，无法解码")
    for text, seq in zip(texts, seqs):
        ids = seq[1:-1]
        if all(i in other.vocab for i in ids) and other.decode(ids) != text:
            line(编号未越界的原公式=short(text), 错误解码=short(other.decode(ids)))
            break
    else:
        # 小样例的合并表可能只有前缀差别；用共同编号构造明确标注的探针。
        shared = [i for i in range(min(len(tok), len(other))) if tok.vocab[i] != other.vocab[i]][:12]
        line(人造共有编号探针=shared, 正确解码=short(tok.decode(shared)), 错误解码=short(other.decode(shared)))
    print("  数据版本由调用方维护；改变数据内容必须升级版本。缓存键不是原文内容校验和。")


def part10(exp):
    # 主报告跟踪完整探索样本；慢速 mathtext 另作子集示范。
    texts = exp.sample
    line(报告原始样本数=len(texts), token口径="规范token；坏串使用规范键回退", 过滤器="教学演示，非推荐生产策略")
    previous = len(texts)

    def report(label, current, removed):
        nonlocal previous
        lengths = [len(exp.key(t).split()) for t in current]
        line(步骤=label, 剩余条数=len(current), 本步删除条数=previous-len(current),
             本步删除比例=(previous-len(current))/max(1, previous), token总数=sum(lengths),
             平均长度=sum(lengths)/max(1, len(current)),
             矩阵比例=sum(bool(MATRIX.search(t)) for t in current)/max(1, len(current)),
             文本比例=sum(structures(t)["文本"] for t in current)/max(1, len(current)))
        for text in removed[:2]:
            line(本步删除代表=short(text))
        previous = len(current)

    report("原始", texts, [])
    cleaned = [clean_formula(t) for t in texts if clean_formula(t)]
    report("清洗去空", cleaned, [t for t in texts if not clean_formula(t)])
    unique, _ = exact_dedup(cleaned)
    seen, removed = set(), []
    for text in cleaned:
        key = exp.key(text)
        if key in seen:
            removed.append(text)
        seen.add(key)
    report("规范键精确去重", unique, removed)
    keep, _ = exp.near(unique)
    kept = set(keep)
    near = [unique[i] for i in keep]
    report("近重复连通分量", near, [t for i, t in enumerate(unique) if i not in kept])
    weak = [well_formed(t) for t in near]
    good = [t for t, ok in zip(near, weak) if ok]
    report("弱语法检查", good, [t for t, ok in zip(near, weak) if not ok])
    subset = good[:2000]
    parsed = [exp.parse(t) for t in subset]
    rejected = [t for t, ok in zip(subset, parsed) if not ok]
    accepted = [t for t, ok in zip(subset, parsed) if ok]
    line(子集示范="mathtext硬过滤（子集估计）", 子集来源="弱语法检查后前至多2000条",
         子集条数=len(subset), 子集删除条数=len(rejected),
         子集删除比例=len(rejected)/len(subset) if subset else "无样本，不定义",
         子集剩余条数=len(subset)-len(rejected), 探索样本最终剩余条数=len(good))
    for label, current in (("过滤前", subset), ("过滤后", accepted)):
        lengths = [len(exp.key(t).split()) for t in current]
        line(子集阶段=label, token总数=sum(lengths),
             平均长度=sum(lengths)/max(1, len(current)),
             矩阵比例=sum(bool(MATRIX.search(t)) for t in current)/max(1, len(current)),
             文本比例=sum(structures(t)["文本"] for t in current)/max(1, len(current)))
    for text in rejected[:2]:
        line(子集删除代表=short(text))
    print("  mathtext仅作子集估计，不计入探索样本的最终剩余条数；按该比例外推不可靠，前缀与过滤会改变分布。")
    print("  输入来自已清洗的load_formulas；清洗步可能不删。报告只做数据审计，不训练、不评测模型。")


PARTS = {1: (part1, "语料探索（12.1）"), 2: (part2, "规范化（12.2）"),
         3: (part3, "精确去重（12.3）"), 4: (part4, "MinHash 原理验证（12.4）"),
         5: (part5, "LSH 取舍（12.5）"), 6: (part6, "近重复去重（12.6）"),
         7: (part7, "能否编译过滤（12.7）"), 8: (part8, "数据混合（12.8）"),
         9: (part9, "token 文件与缓存键（12.9）"), 10: (part10, "数据报告（12.10）")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="*", type=int, help="部分号1–10，默认全部")
    parser.add_argument("--seed", type=int, default=0, help="探索抽样和MinHash的种子")
    parser.add_argument("--sample", type=int, default=20000, help="探索样本条数")
    parser.add_argument("--small", action="store_true", help="主动使用内置样例")
    args = parser.parse_args(argv)
    if args.sample < 1 or any(p not in PARTS for p in args.parts):
        parser.error("部分号须为1–10；sample须为正")
    torch.set_num_threads(2)
    print(f"第12章；设备=CPU；线程数=2；种子={args.seed}；浮点数四位小数。", flush=True)
    start = time.perf_counter()
    exp = Experiment(args)
    for part in args.parts or PARTS:
        fn, title = PARTS[part]
        print("\n" + "="*70 + f"\n{part}. {title}\n" + "="*70, flush=True)
        begun = time.perf_counter()
        fn(exp)
        print(f"第 {part} 部分用时秒={time.perf_counter()-begun:.4f}", flush=True)
    print(f"第12章所选实验完成；tour总用时秒={time.perf_counter()-start:.4f}", flush=True)


if __name__ == "__main__":
    main()
