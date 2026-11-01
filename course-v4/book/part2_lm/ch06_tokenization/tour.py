"""第 6 章：分词与 BPE。所有实验只读取本地数据，计算在 CPU 上完成。

    python -m book.part2_lm.ch06_tokenization.tour [部分号 ...]
    python -m book.part2_lm.ch06_tokenization.tour 8 --n-train 1000
    python -m book.part2_lm.ch06_tokenization.tour 8 --command-blocks
    python -m book.part2_lm.ch06_tokenization.tour --small

输出：outputs/book/ch06/。辅助函数都是演示代码，不是挖空练习。
--small 使用内置公式；--command-blocks 同时替换训练和编码的预分词规则。
"""
import argparse
from collections import Counter
from functools import lru_cache
import math
import os
from pathlib import Path
import sys
import time

# 在导入任何可能使用 torch 的库之前隐藏 GPU。分词本身始终在 CPU 上。
if not any(a == "--device=cuda" or a == "--device" and sys.argv[i + 1:i + 2] == ["cuda"]
           for i, a in enumerate(sys.argv)):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import regex

from miniocr.data.latex_corpus import SAMPLE_FORMULAS, corpus_available, standard_split
from miniocr.eval.latex_norm import tokenize_latex
from miniocr.tokenizer import (
    BPETokenizer, COMMAND_PAT, PAT, _best, _merge_word, pretokenize, show_tokens, train_bpe, train_bpe_naive,
)

OUT = Path("outputs/book/ch06")
MERGE_COUNTS = (0, 250, 1000, 2000, 4000)
TOY_COUNTS = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
RANK_MERGES = [(b"b", b"c"), (b"a", b"b")]
_C = {}
_SMALL = False
_N_TRAIN = 20000


def data():
    if "data" not in _C:
        if _SMALL or not corpus_available():
            print("  提示：使用 SAMPLE_FORMULAS 小样例降级运行，结果仅用于检查流程。")
            split = standard_split(n_train=_N_TRAIN, n_val=4, n_test=4, formulas=SAMPLE_FORMULAS)
        else:
            split = standard_split(n_train=_N_TRAIN)
        print(f"  语料划分：训练条数={len(split['train'])}；固定验证条数={len(split['val'])}；"
              f"固定测试条数={len(split['test'])}；划分种子=0；协议=latex-v1")
        print(f"  本章评测：验证条数={len(split['val'][:2000])}；测试集不参与实验。")
        _C["data"] = split["train"], split["val"][:2000]
    return _C["data"]


def latex_merges(pattern=None):
    pattern = PAT if pattern is None else pattern
    key = ("merges", pattern)
    if key not in _C:
        train, _ = data()
        start = time.perf_counter()
        _C[key] = train_bpe(train, 4000, pattern=pattern)
        print(f"  共用 BPE：请求合并次数=4000；实际合并次数={len(_C[key])}；"
              f"训练用时秒={time.perf_counter() - start:.4f}")
        OUT.mkdir(parents=True, exist_ok=True)
        suffix = "commands" if pattern not in (None, PAT) else "default"
        for n in (2000, 4000):
            BPETokenizer(_C[key][:n], specials=(), pattern=pattern).save(OUT / f"bpe_{n}_{_N_TRAIN}_{suffix}.json")
        print("  说明：每次进程重新训练，避免旧缓存混入不同训练条数或 PAT 的实验。")
    return _C[key]


def tokenizer(n, pattern=None):
    pattern = PAT if pattern is None else pattern
    key = ("tok", n, pattern)
    if key not in _C:
        _C[key] = BPETokenizer(latex_merges(pattern)[:n], specials=(), pattern=pattern)
    return _C[key]


def pieces(tok, text, **kwargs):
    return show_tokens([tok.vocab[i] for i in tok.encode(text, **kwargs)])


def bytes_per_token(tok, texts):
    """总 UTF-8 字节数 / 总 token 数；空语料返回零，不插入换行或特殊 token。"""
    total_bytes = total_tokens = 0
    for text in texts:
        total_bytes += len(text.encode("utf-8"))
        total_tokens += len(tok.encode(text, allow_special=False))
    return total_bytes / total_tokens if total_tokens else 0.0


@lru_cache(maxsize=None)
def real_tokenizer(name):
    try:
        from transformers import AutoTokenizer
        if name == "gpt2":
            from miniocr.models.gpt2 import find_local_gpt2
            name = find_local_gpt2() or name
        return AutoTokenizer.from_pretrained(name, local_files_only=True)
    except (OSError, ValueError, ImportError) as exc:
        print(f"  跳过：本地分词器 {name} 不可用，原因类型={type(exc).__name__}；不会下载。")
        return None


def part1_phenomenon(pattern=None):
    tok = tokenizer(2000, pattern)
    qwen = real_tokenizer("Qwen/Qwen3.5-0.8B")
    gpt = real_tokenizer("gpt2")
    for text in [r"\frac{a+b}{\sqrt{x^2+1}}", "模型读入图像，写出文字。",
                 "The model reads the image and writes the text."]:
        print(f"  原文={text!r}")
        for name, units in [("字符级", list(text)), ("原始字节，词表大小=256", show_tokens(
                [bytes([b]) for b in text.encode()])), ("LaTeX BPE，请求合并次数=2000", pieces(tok, text))]:
            print(f"    {name}：token 数={len(units)}；切分={units}")
        for name, ref in [("Qwen", qwen), ("GPT-2", gpt)]:
            if ref is not None:
                ids = ref.encode(text, add_special_tokens=False)
                print(f"    {name}：token 数={len(ids)}；切分（词表记号）={ref.convert_ids_to_tokens(ids)}")


def granularity_stats(train, val, split):
    counts = Counter(t for text in train for t in split(text))
    n_tokens = n_oov = n_formulas = 0
    for text in val:
        units = split(text)
        missing = sum(t not in counts for t in units)
        n_tokens += len(units)
        n_oov += missing
        n_formulas += bool(missing)
    return (len(counts), sum(counts.values()) / max(1, len(train)), n_tokens / max(1, len(val)),
            n_oov / max(1, n_tokens), n_formulas / max(1, len(val)),
            sum(n == 1 for n in counts.values()) / max(1, len(counts)))


def part2_granularity(pattern=None):
    train, val = data()
    for name, split in [("字符级", list), ("空格切词", str.split), ("LaTeX token", tokenize_latex)]:
        v, tr_len, va_len, oov, formulas, tail = granularity_stats(train, val, split)
        print(f"  切法={name}；训练词表大小={v}；训练平均序列长度={tr_len:.4f}；"
              f"验证平均序列长度={va_len:.4f}；验证 OOV token 比例={oov:.4f}；"
              f"验证含 OOV 公式比例={formulas:.4f}；单次词条占词表比例={tail:.4f}")
    print("  提示：后两种切法丢弃空白；此处只比较粒度与覆盖，不声称它们能无损还原。")


def part3_utf8(pattern=None):
    for text in ["x", "²", "α", "模", "🙂", r"\alpha"]:
        raw = text.encode()
        print(f"  文本={text!r}；UTF-8 十六进制={raw.hex(' ')}；字节数={len(raw)}")
    half = "模".encode()[:2]
    try:
        half.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        print(f"  截断字节={half.hex(' ')}；strict 错误原文={exc}")
    print(f"  replace 解码结果={half.decode('utf-8', errors='replace')!r}")


def part4_pretokenize(pattern=None):
    print(f"  当前 PAT={PAT if pattern is None else pattern}")
    for text in ["I'm learning 123 things.", "模型读入图像，写出文字。", r"\frac{a}{b}",
                 r"x_{i}^{2}", r"\mathrm{d}x+123"]:
        print(f"  原文={text!r}；词块数={len(pretokenize(text, pattern))}；词块={pretokenize(text, pattern)}")
    print("  默认 PAT 把反斜杠和命令字母分开，数字按位切；BPE 无法跨词块合并。")


def toy_training(n_merges=5):
    """明确给定词块及频数；返回初始语料和每轮计数、选择、合并结果。"""
    words = {tuple(bytes([b]) for b in w.encode()): count for w, count in TOY_COUNTS.items()}
    initial, rounds = words.copy(), []
    for _ in range(n_merges):
        counts = Counter()
        for word, count in words.items():
            for pair in zip(word, word[1:]):
                counts[pair] += count
        if not counts:
            break
        best = _best(counts)
        if counts[best] < 2:
            break
        words = {_merge_word(word, best, b"".join(best)): count for word, count in words.items()}
        rounds.append((counts, best, words.copy()))
    return initial, rounds


def print_corpus(words):
    for word, count in words.items():
        print(f"    词块 token={show_tokens(word)}；频数={count}")


def part5_training(pattern=None):
    initial, rounds = toy_training()
    print("  玩具语料初始字节：")
    print_corpus(initial)
    print("  平局规则：取 bytes 元组字典序更大的相邻对；与两个训练函数一致。")
    for i, (counts, best, words) in enumerate(rounds, 1):
        print(f"  合并轮次={i}；相邻对种类数={len(counts)}；计数表显示上限=10：")
        for pair, count in sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)[:10]:
            print(f"    相邻对={show_tokens(pair)}；加权次数={count}")
        print(f"  选中对={show_tokens(best)}；次数={counts[best]}；合并后语料：")
        print_corpus(words)
    train, _ = data()
    results = []
    for name, fn in [("朴素版", train_bpe_naive), ("增量版", train_bpe)]:
        start = time.perf_counter()
        results.append(fn(train, 500, pattern=pattern))
        print(f"  算法={name}；训练条数={len(train)}；请求合并次数=500；实际合并次数={len(results[-1])}；"
              f"用时秒={time.perf_counter() - start:.4f}")
    print(f"  两版合并列表完全相同={results[0] == results[1]}")


def trace_word(word, merges, greedy=False):
    """词块编码的轨迹：初始 parts、[(所选 pair, 合并后 parts), ...]。

    greedy=True 是故意写错的从左往右策略，用来与按 rank 合并对照。
    """
    parts = tuple(bytes([b]) for b in word.encode())
    initial, steps = parts, []
    ranks = {pair: rank for rank, pair in enumerate(merges)}
    while True:
        candidates = [p for p in zip(parts, parts[1:]) if p in ranks]
        if not candidates:
            return initial, steps
        pair = candidates[0] if greedy else min(candidates, key=ranks.get)
        parts = _merge_word(parts, pair, b"".join(pair))
        steps.append((pair, parts))


def part6_encoding(pattern=None):
    for rank, pair in enumerate(RANK_MERGES):
        print(f"  合并排名（从零计）={rank}；相邻对={show_tokens(pair)}")
    for name, greedy in [("按合并排名", False), ("错误的从左往右贪心", True)]:
        initial, steps = trace_word("abc", RANK_MERGES, greedy)
        print(f"  策略={name}；初始序列={show_tokens(initial)}")
        for i, (pair, parts_) in enumerate(steps, 1):
            print(f"    步数={i}；合并对={show_tokens(pair)}；当前序列={show_tokens(parts_)}")
        print(f"  最终结果={show_tokens(steps[-1][1])}")
    print(f"  BPETokenizer 实际输出={pieces(BPETokenizer(RANK_MERGES, specials=()), 'abc')}")


def part7_specials(pattern=None):
    tok = BPETokenizer(latex_merges(pattern)[:2000], specials=("<|bos|>", "<|eos|>", "<|pad|>"), pattern=pattern)
    for text, idx in tok.special_ids.items():
        print(f"  特殊 token={text!r}；编号={idx}")
    text = "x<|eos|>y"
    for name, target, allowed in [("允许特殊 token", tok, True), ("作为普通文本", tok, False),
                                   ("没有特殊 token 机制", BPETokenizer(tok.merges, specials=(), pattern=tok.pattern), True)]:
        print(f"  情形={name}；编号序列={target.encode(text, allow_special=allowed)}；"
              f"切分={pieces(target, text, allow_special=allowed)}")
    print("  用户输入混入特殊 token 字符串可能伪造结束或消息边界；普通用户文本应使用 allow_special=False。")


def part8_latex(pattern=None):
    _, val = data()
    compression = []
    for n in MERGE_COUNTS:
        tok = tokenizer(n, pattern)
        ratio = bytes_per_token(tok, val)
        mean = sum(len(tok.encode(s)) for s in val) / len(val)
        compression.append(ratio)
        print(f"  请求合并次数={n}；实际合并次数={len(tok.merges)}；每 token 平均字节数={ratio:.4f}；"
              f"平均每条公式 token 数={mean:.4f}")
    merges = latex_merges(pattern)
    print(f"  最早合并结果，显示条数={len(merges[:30])}：{show_tokens([a + b for a, b in merges[:30]])}")
    for text in [r"\frac", r"\mathrm", r"\left(", r"\alpha", "^{2}"]:
        first = next((i for i, pair in enumerate(merges, 1) if b"".join(pair) == text.encode()), None)
        print(f"  目标={text!r}；词块={pretokenize(text, pattern)}；首次成为单 token 的合并次数={first if first else '未合并'}")
    print("  默认 PAT 的边界不能靠增加合并次数突破；命令字母可以合并，整条反斜杠命令仍可能分开。")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(MERGE_COUNTS, compression, marker="o")
    ax.set(xlabel="Requested BPE merges", ylabel="UTF-8 bytes / token", title="LaTeX validation compression")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "compression.png", dpi=140)
    plt.close(fig)
    print(f"  压缩率曲线文件={OUT / 'compression.png'}")


def part9_languages(pattern=None):
    from miniocr.data import zh_en
    root = Path(os.environ.get("HF_HUB_CACHE", Path(os.environ.get(
        "HF_HOME", Path.home() / ".cache/huggingface")) / "hub")).expanduser()
    paths = sorted((root / "datasets--YogeLiu--zh-en-translation-dataset-600K").glob("snapshots/*/data/*.parquet"))
    if _SMALL or not paths:
        print("  跳过：小样例模式或本地中英语料缓存不可用；不会下载。")
        return
    # zh_en.load_pairs 在缺缓存时会下载；这里显式读已确认的路径，再交给它划分。
    import pyarrow.parquet as pq
    table = pq.read_table(paths[0], columns=["english", "chinese"])
    pairs_ = [(e.strip(), z.strip()) for e, z in zip(table['english'].to_pylist(), table['chinese'].to_pylist())
              if e and z and e.strip() and z.strip()]
    import random
    random.Random(0).shuffle(pairs_)
    # 多留训练候选，去掉与留出集共享句子的句对后，仍能取满每种语言两万行。
    train, val, _, _ = zh_en.standard_split(pairs=pairs_[:26000], dev=2000, test=2000)
    for lang, name in [("zh", "中文"), ("en", "英文")]:
        lines = zh_en.to_lines(train, lang)[:20000]
        held = zh_en.to_lines(val, lang)
        start = time.perf_counter()
        tok = BPETokenizer(train_bpe(lines, 2000, pattern=pattern), specials=(), pattern=pattern)
        print(f"  语种={name}；训练行数={len(lines)}；验证行数={len(held)}；合并次数={len(tok.merges)}；"
              f"训练用时秒={time.perf_counter() - start:.4f}；验证每 token 平均字节数={bytes_per_token(tok, held):.4f}")
        print(f"  语种={name}；最早合并显示条数=20；合并结果={show_tokens([a+b for a,b in tok.merges[:20]])}")
        if lang == "zh":
            print(f"  LaTeX 分词器编码同一中文验证集：请求合并次数=2000；"
                  f"每 token 平均字节数={bytes_per_token(tokenizer(2000, pattern), held):.4f}")
    print("  训练语料决定分词器擅长什么；跨领域时要在同一份文本上比较。")


def unigram_metrics(tok, train, val):
    """加一平滑 unigram：以完整普通词表为支持集，不加入 BOS/EOS。"""
    counts = Counter(i for text in train for i in tok.encode(text, allow_special=False))
    vocab_size = 256 + len(tok.merges)
    denominator = sum(counts.values()) + vocab_size
    held = Counter(i for text in val for i in tok.encode(text, allow_special=False))
    n_tokens = sum(held.values())
    n_bytes = sum(len(s.encode()) for s in val)
    total_nats = sum(n * math.log(denominator / (counts[i] + 1)) for i, n in held.items())
    loss = total_nats / n_tokens
    return loss, math.exp(loss), total_nats / (math.log(2) * n_bytes)


def part10_bits(pattern=None):
    print("  换算公式：bits/byte = (nat/token × token 数) / (ln2 × 字节数)")
    train, val = data()
    print("  模型=加一平滑 unigram；训练与评测逐条编码，不插入分隔符；验证文本完全相同。")
    for n in MERGE_COUNTS:
        tok = tokenizer(n, pattern)
        loss, perplexity, bpb = unigram_metrics(tok, train, val)
        print(f"  请求合并次数={n}；实际合并次数={len(tok.merges)}；验证 nat/token={loss:.4f}；"
              f"验证困惑度={perplexity:.4f}；验证 bits/byte={bpb:.4f}")
    print("  观察 loss/token 是否随词表增大而升高；这不是单调性定理，跨分词器应比较 bits/byte。")
    for name, n_tokens, n_bytes, loss in [("A", 100, 300, 3.0), ("B", 60, 300, 4.0)]:
        print(f"  手算分词器={name}；给定 token 数={n_tokens}；给定字节数={n_bytes}；"
              f"给定 nat/token={loss:.4f}；换算 bits/byte={loss*n_tokens/(math.log(2)*n_bytes):.4f}")


def command_split_counts(text, pre_offsets, token_offsets):
    """按命令每次出现计数，返回总数、预分词切开数、词块完整但 BPE 切开数、最终切开数。

    偏移均为原文字符区间 [start, end)。命令可以只是一个较大词块/token 的子串。
    两阶段比例都以所有命令出现次数为分母，不是只以完整词块为分母。
    """
    total = pre_split = bpe_split = fragmented = 0
    for command in regex.finditer(r"\\[A-Za-z]+", text):
        total += 1
        intact = any(start <= command.start() and end >= command.end() for start, end in pre_offsets)
        split = sum(start < command.end() and end > command.start() for start, end in token_offsets) > 1
        pre_split += not intact
        bpe_split += intact and split
        fragmented += split
    return total, pre_split, bpe_split, fragmented


def bpe_token_offsets(tok, text):
    """教学分词器的普通 token 字符偏移；半个 UTF-8 字符对应同一个字符区间。"""
    char_at_byte = [i for i, char in enumerate(text) for _ in char.encode("utf-8")]
    offsets, pos = [], 0
    for idx in tok.encode(text, allow_special=False):
        end = pos + len(tok.vocab[idx])
        offsets.append((char_at_byte[pos], char_at_byte[end - 1] + 1))
        pos = end
    return offsets


def print_command_stats(name, texts, pre_offsets, token_offsets):
    counts = [command_split_counts(text, pre, final)
              for text, pre, final in zip(texts, pre_offsets, token_offsets)]
    total, pre, bpe, fragmented = map(sum, zip(*counts)) if counts else (0, 0, 0, 0)
    print(f"  分词器={name}；命令出现次数={total}；预分词已切开次数(a)={pre}；比例(a)={pre/max(1,total):.4f}；"
          f"预分词完整但 BPE 切开次数(b)={bpe}；比例(b)={bpe/max(1,total):.4f}；"
          f"最终跨多个 token 次数={fragmented}；比例={fragmented/max(1,total):.4f}")


def part11_real(pattern=None):
    _, val = data()
    tok = tokenizer(4000, pattern)
    print(f"  验证条数={len(val)}；LaTeX BPE 平均 token 数={sum(len(tok.encode(s)) for s in val)/len(val):.4f}")
    print("  统计口径=验证文本中反斜杠加英文字母命令的每次出现；按原文字符偏移统计；"
          "(a)/(b) 均除以全部命令次数，分别区分预分词边界与 BPE 合并未覆盖。")
    default = tokenizer(4000, PAT)
    print_command_stats("LaTeX BPE（默认 PAT）", val,
                        [[m.span() for m in regex.finditer(default.pattern, s)] for s in val],
                        [bpe_token_offsets(default, s) for s in val])
    examples = [r"\frac{a+b}{\sqrt{x^2+1}}", r"x\in\mathbb{R}",
                r"\begin{pmatrix}a&b\\c&d\end{pmatrix}", r"x_{i_1,i_2,\ldots,i_n}",
                r"\text{面积}=ab", r"\mathrm{d}x", r"\alpha+\beta=\gamma", r"\left(x+y\right)",
                r"\sum_{i=1}^{n}i^2", r"\int_0^1 f(x)\,\mathrm{d}x"]
    for name, model in [("Qwen", "Qwen/Qwen3.5-0.8B"), ("GPT-2", "gpt2")]:
        ref = real_tokenizer(model)
        if ref is None:
            print(f"  第 11 部分跳过 {name}：本地分词器不可用；不会下载。")
            continue
        print(f"  分词器={name}；模型名={model}；len(tokenizer)（含新增 token）={len(ref)}；"
              f"tokenizer.vocab_size（基础词表）={ref.vocab_size}")
        encoded = ref(val, add_special_tokens=False, return_offsets_mapping=True)
        print(f"  验证条数={len(val)}；{name} 平均 token 数={sum(map(len, encoded['input_ids']))/len(val):.4f}")
        pretokenizer = ref.backend_tokenizer.pre_tokenizer
        for text in [r"\frac{a}{b}", r"x\in\mathbb{R}", r"\alpha+\beta", r"{\frac{a}{b}}"]:
            pre = pretokenizer.pre_tokenize_str(text)
            final = ref.convert_ids_to_tokens(ref.encode(text, add_special_tokens=False))
            print(f"  {name} 原文={text!r}；预分词（词表记号、字符偏移）={pre}；最终 token={final}")
        print_command_stats(name, val,
                            [[offset for _, offset in pretokenizer.pre_tokenize_str(s)] for s in val],
                            encoded['offset_mapping'])
        for text in examples:
            print(f"  典型公式={text!r}")
            print(f"    {name} 切分（词表记号）={ref.convert_ids_to_tokens(ref.encode(text, add_special_tokens=False))}")
            print(f"    LaTeX BPE 切分={pieces(tok, text)}")


def encode_uncached(tok, text):
    """相同预分词、相同合并规则，唯一差别是每次重新计算词块编码。"""
    return [i for word in pretokenize(text, tok.pattern) for i in tok._encode_word(word)]


def part12_pitfalls(pattern=None):
    raw = BPETokenizer([], specials=())
    ids = raw.encode("模型")
    print(f"  逐 token 解码再拼接={''.join(raw.decode([i]) for i in ids)!r}；整体解码={raw.decode(ids)!r}")
    trained = BPETokenizer(train_bpe(["abc"] * 4, 2), specials=())
    print(f"  训练 PAT=默认；正确编码 abc={pieces(trained, 'abc')}")
    wrong = BPETokenizer(trained.merges, specials=(), pattern=r"(?s).")
    print(f"  错误编码 PAT=逐字符；错误编码 abc={pieces(wrong, 'abc')}")
    train, val = data()
    batch = (val[:100] or train[:100]) * 10
    tok = BPETokenizer(latex_merges(pattern)[:2000], specials=(), pattern=pattern)
    start = time.perf_counter()
    cached = [tok.encode(s, allow_special=False) for s in batch]
    cached_sec = time.perf_counter() - start
    start = time.perf_counter()
    uncached = [encode_uncached(tok, s) for s in batch]
    uncached_sec = time.perf_counter() - start
    print(f"  缓存对照：文本条数={len(batch)}；有缓存用时秒（含首次填充）={cached_sec:.4f}；"
          f"无缓存用时秒={uncached_sec:.4f}；输出完全相同={cached == uncached}；"
          f"无缓存与有缓存用时比={uncached_sec/max(cached_sec, 1e-12):.4f}")
    text = "  x +  y\n\t z  "
    decoded = tok.decode(tok.encode(text))
    print(f"  空白往返：原文={text!r}；解码={decoded!r}；完全相同={text == decoded}")
    print("  我们的字节分词器保留空白，不会自动多出或丢掉空格；语料清洗是独立的、有意改变空白的步骤。")


PARTS = {
    1: ("现象", part1_phenomenon), 2: ("三种粒度", part2_granularity),
    3: ("字节与 UTF-8", part3_utf8), 4: ("预分词", part4_pretokenize),
    5: ("BPE 手算与训练", part5_training), 6: ("编码必须按合并顺序", part6_encoding),
    7: ("特殊 token", part7_specials), 8: ("在 LaTeX 语料上训练", part8_latex),
    9: ("中英文对照", part9_languages), 10: ("bits/byte", part10_bits),
    11: ("真实分词器看 LaTeX", part11_real), 12: ("常见错误", part12_pitfalls),
}


def main():
    global _SMALL, _N_TRAIN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="*", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--n-train", type=int, default=20000)
    parser.add_argument("--command-blocks", action="store_true")
    args = parser.parse_args()
    if args.n_train <= 0 or any(k not in PARTS for k in args.parts):
        parser.error("训练条数须为正，部分号须在 1 至 12 之间")
    _SMALL, _N_TRAIN = args.small, args.n_train
    _C.clear()
    started = time.perf_counter()
    print("第 6 章；实际计算设备=CPU；数据访问=仅本地；浮点输出保留四位小数。")
    print(f"实验配置：请求训练条数={_N_TRAIN}；命令完整词块={args.command_blocks}")
    pattern = COMMAND_PAT if args.command_blocks else PAT
    for k in args.parts or PARTS:
        name, fn = PARTS[k]
        print("\n" + "=" * 70 + f"\n第 {k} 部分：{name}（正文 6.{k}）\n" + "=" * 70)
        start = time.perf_counter()
        fn(pattern=pattern)
        print(f"第 {k} 部分用时秒={time.perf_counter() - start:.4f}")
    print(f"\n第 6 章所选实验完成；tour 总用时秒={time.perf_counter() - started:.4f}")


if __name__ == "__main__":
    main()
