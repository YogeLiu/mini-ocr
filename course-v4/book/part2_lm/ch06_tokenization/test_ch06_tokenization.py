"""第 6 章的自动检查：python -m pytest book/part2_lm/ch06_tokenization"""
import os
import tempfile
import json

from miniocr.data.zh_en import SAMPLE_PAIRS, split_pairs, to_lines
from miniocr.tokenizer import BPETokenizer, COMMAND_PAT, PAT, pretokenize, show_tokens, train_bpe, train_bpe_naive

LINES = to_lines(SAMPLE_PAIRS) * 3


def test_pretokenize_is_lossless_and_splits_digits():
    for s in LINES + ["I'm 12 岁。", "  多个空格\n换行\t制表", "emoji 😀!"]:
        assert "".join(pretokenize(s)) == s
    assert pretokenize("I'm 12 岁。") == ["I", "'m", " 1", "2", " 岁", "。"]


def test_naive_first_merges_by_hand():
    # 加权：(a,b)=5，(x,y)=2；若每个词块只加 1，则 (a,b)=1，(x,y)=2，首对会选错。
    m = train_bpe_naive(["ab"] * 5 + ["xyc", "xyd"], 1)
    assert m == [(b"a", b"b")]
    # 次数 < 2 时停止
    assert train_bpe_naive(["xyz"], 5) == []


def test_incremental_equals_naive():
    assert train_bpe(LINES, 120) == train_bpe_naive(LINES, 120)


def test_roundtrip_chinese_english_and_unknown_text():
    tok = BPETokenizer(train_bpe(LINES, 150))
    for s in LINES + ["从没见过的字：龘靐", "mixed 中文 and English 123", "😀"]:
        assert tok.decode(tok.encode(s)) == s


def test_merges_compress_and_vocab_layout():
    m = train_bpe(LINES, 150)
    tok = BPETokenizer(m)
    s = "".join(LINES)
    assert len(tok.encode(s)) < len(s.encode("utf-8"))
    assert len(tok) == 256 + len(m) + 1
    assert tok.vocab[256] == m[0][0] + m[0][1]


def test_encode_follows_merge_rank_not_greedy_left_to_right():
    # 规则先学 (b, c)，再学 (a, b)。"abc" 应该先合 bc → [a, bc]，而不是从左到右先合 ab。
    tok = BPETokenizer([(b"b", b"c"), (b"a", b"b")], specials=())
    ids = tok.encode("abc")
    assert [tok.vocab[i] for i in ids] == [b"a", b"bc"]


def test_special_tokens_are_atomic():
    tok = BPETokenizer(train_bpe(LINES, 50))
    eot = tok.special_ids["<|endoftext|>"]
    ids = tok.encode("你好<|endoftext|>hello")
    assert ids.count(eot) == 1
    assert tok.decode(ids) == "你好<|endoftext|>hello"
    assert eot not in tok.encode("你好<|endoftext|>", allow_special=False)


def test_save_load_and_show_tokens():
    tok = BPETokenizer(train_bpe(LINES, 80))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "tok.json")
        tok.save(p)
        tok2 = BPETokenizer.load(p)
    assert tok2.encode("模型读入图像") == tok.encode("模型读入图像")
    assert show_tokens(["模".encode(), "模".encode()[:2]]) == ["模", "<e6><a8>"]


def test_split_pairs_keeps_pairs_together():
    pairs = [(f"e{i}", f"z{i}") for i in range(100)]
    tr, dv, te = split_pairs(pairs, dev=10, test=10, seed=0)
    assert len(tr) == 80 and len(dv) == 10 and len(te) == 10
    assert not set(tr) & set(dv) and not set(tr) & set(te)


def test_standard_split_removes_shared_sentences():
    from miniocr.data.zh_en import standard_split
    pairs = SAMPLE_PAIRS + [(SAMPLE_PAIRS[0][0], "另一种译法。")] * 3     # 同一个英文句子出现在多个句对里
    tr, dv, ts, n_rm = standard_split(pairs=pairs, dev=2, test=2)
    held = {s for p in dv + ts for s in p}
    assert all(e not in held and z not in held for e, z in tr)
    assert len(tr) + len(dv) + len(ts) + n_rm == len(pairs)


# 新章的语料与演示检查全部用内置数据或临时 parquet，不依赖本机缓存。
import math

import pytest

from miniocr.data import latex_corpus
from miniocr.data.latex_corpus import SAMPLE_FORMULAS, clean_formula, normalization_key
from miniocr.eval.latex_norm import normalize_latex
from book.part2_lm.ch06_tokenization.tour import (
    RANK_MERGES, TOY_COUNTS, bpe_token_offsets, bytes_per_token, command_split_counts, encode_uncached,
    granularity_stats, toy_training, trace_word, unigram_metrics,
)


@pytest.mark.parametrize("raw, expected", [
    (None, ""), (" \n\t ", ""),
    (r"\begin{align*} x^2  + y^2 \end{align*}", "x^2 + y^2"),
    ("\\begin{equation*}\n x +\t y \\end{equation*}", "x + y"),
    (r"\begin{gather}\begin{equation}x\end{equation}\end{gather}", "x"),
    (r"\begin{pmatrix}a&b\\c&d\end{pmatrix}", r"\begin{pmatrix}a&b\\c&d\end{pmatrix}"),
    (r"\begin{align}x\end{equation}", r"\begin{align}x\end{equation}"),
    (r"\begin{align}x\end{align}+\begin{align}y\end{align}",
     r"\begin{align}x\end{align}+\begin{align}y\end{align}"),
])
def test_formula_cleaning(raw, expected):
    assert clean_formula(raw) == expected


def test_normalization_success_and_parse_failure_fallback():
    for text in SAMPLE_FORMULAS:
        assert normalization_key(text) == normalize_latex(clean_formula(text))
    assert normalization_key(r"\dfrac12") == normalization_key(r"\frac{1}{2}")
    malformed = r"\dfrac{ x "
    assert normalization_key(malformed) == r"\dfrac{ x"
    assert normalization_key(malformed) != normalize_latex(malformed)


def test_formula_split_no_normalized_leakage_and_fixed_holdouts():
    formulas = SAMPLE_FORMULAS + [r"\begin{align*}x^2+y^2=z^2\end{align*}", " "]
    full = latex_corpus.standard_split(n_train=None, n_val=4, n_test=4, formulas=formulas)
    short = latex_corpus.standard_split(n_train=2, n_val=4, n_test=4, formulas=reversed(formulas))
    again = latex_corpus.standard_split(n_train=None, n_val=4, n_test=4, formulas=formulas)
    assert full == again
    assert short["val"] == full["val"] and short["test"] == full["test"]
    assert short["train"] == full["train"][:2]
    keys = {name: {normalization_key(s) for s in texts} for name, texts in full.items()}
    assert not keys["train"] & keys["val"]
    assert not keys["train"] & keys["test"]
    assert not keys["val"] & keys["test"]
    assert sum(map(len, full.values())) == len(formulas) - 1
    assert all(full.values())
    # 分组不可拆：同义写法与重复公式全在同一集合，且确实都被保留。
    equivalent = {r"x^2+y^2=z^2", r"x^{2}+y^{2}=z^{2}"}
    assert sum(bool(equivalent & set(part)) for part in full.values()) == 1


def test_local_formula_loading_reads_only_text_column(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    repo = tmp_path / "datasets--OleehyO--latex-formulas"
    shard = repo / "snapshots/revision/cleaned_formulas/train.parquet"
    shard.parent.mkdir(parents=True)
    pq.write_table(pa.table({"latex_formula": [*SAMPLE_FORMULAS, "", None],
                            "image": [b"do-not-read"] * (len(SAMPLE_FORMULAS) + 2)}), shard)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    original = pq.ParquetFile
    columns_read = []

    class TextOnlyFile:
        def __init__(self, path):
            self.file = original(path)

        def iter_batches(self, *, columns):
            columns_read.append(columns)
            assert columns == ["latex_formula"]
            return self.file.iter_batches(columns=columns)

    monkeypatch.setattr(pq, "ParquetFile", TextOnlyFile)
    assert latex_corpus.corpus_available()
    texts = latex_corpus.load_formulas(seed=7)
    assert len(texts) == len(SAMPLE_FORMULAS)
    assert latex_corpus.load_formulas(n=3, seed=7) == texts[:3]
    assert latex_corpus.load_formulas(seed=8) != texts
    assert columns_read == [["latex_formula"], ["latex_formula"], ["latex_formula"]]


def test_missing_corpus_uses_samples_without_network(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    assert not latex_corpus.corpus_available()
    with pytest.warns(UserWarning, match="SAMPLE_FORMULAS"):
        texts = latex_corpus.load_formulas()
    assert sorted(texts) == sorted(SAMPLE_FORMULAS)
    with pytest.warns(UserWarning, match="SAMPLE_FORMULAS"):
        split = latex_corpus.standard_split()
    assert all(split.values())
    assert sum(map(len, split.values())) == len(SAMPLE_FORMULAS)


def test_formula_validation_arguments():
    with pytest.raises(ValueError):
        latex_corpus.load_formulas(-1)
    with pytest.raises(ValueError):
        latex_corpus.standard_split(n_train=-1, formulas=SAMPLE_FORMULAS)
    assert latex_corpus.standard_split(formulas=[]) == {"train": [], "val": [], "test": []}


def test_toy_five_rounds_match_both_trainers():
    initial, rounds = toy_training()
    expected = [(b"s", b"t"), (b"e", b"st"), (b"o", b"w"), (b"l", b"ow"), (b"w", b"est")]
    assert [best for _, best, _ in rounds] == expected
    assert rounds[0][0][(b"s", b"t")] == 9
    assert rounds[0][0][(b"e", b"s")] == 9  # 平局取字节序更大者
    assert initial[tuple(bytes([b]) for b in b"low")] == 5
    texts = [word for word, count in TOY_COUNTS.items() for _ in range(count)]
    assert train_bpe_naive(texts, 5) == train_bpe(texts, 5) == expected


def test_rank_counterexample_traces_and_real_encoder():
    initial, ranked = trace_word("abc", RANK_MERGES)
    _, greedy = trace_word("abc", RANK_MERGES, greedy=True)
    assert initial == (b"a", b"b", b"c")
    assert ranked == [((b"b", b"c"), (b"a", b"bc"))]
    assert greedy == [((b"a", b"b"), (b"ab", b"c"))]
    tok = BPETokenizer(RANK_MERGES, specials=())
    assert tuple(tok.vocab[i] for i in tok.encode("abc")) == ranked[-1][1]


def test_bos_eos_pad_atomic_and_opt_out():
    specials = ("<|bos|>", "<|eos|>", "<|pad|>")
    tok = BPETokenizer([(b"a", b"b")], specials=specials)
    assert list(tok.special_ids.values()) == [257, 258, 259]
    for special in specials:
        assert tok.encode(special) == [tok.special_ids[special]]
        assert tok.special_ids[special] not in tok.encode(special, allow_special=False)
        assert tok.decode(tok.encode(special, allow_special=False)) == special
    text = "<|bos|>ab<|eos|><|pad|>"
    assert tok.decode(tok.encode(text)) == text
    plain = BPETokenizer(tok.merges, specials=())
    assert plain.encode(text) == tok.encode(text, allow_special=False)


def test_bytes_per_token_weighted_and_empty():
    tok = BPETokenizer([(b"a", b"b")], specials=())
    assert bytes_per_token(tok, ["ab", "模"]) == 5 / 4
    assert bytes_per_token(tok, iter(["ab", "abab"])) == 2.0
    assert bytes_per_token(tok, []) == bytes_per_token(tok, [""]) == 0.0


def test_unigram_smoothing_has_known_probabilities():
    tok = BPETokenizer([], specials=("<|eos|>",))
    loss, perplexity, bpb = unigram_metrics(tok, ["aaa"], ["ab"])
    # 分母=3 个训练 token + 256 个普通词条；p(a)=4/259，p(b)=1/259。
    expected = -(math.log(4 / 259) + math.log(1 / 259)) / 2
    assert loss == pytest.approx(expected)
    assert perplexity == pytest.approx(math.exp(expected))
    assert bpb == pytest.approx(expected / math.log(2))


def test_granularity_metrics_count_tokens_and_formula_oov():
    stats = granularity_stats(["a a b"], ["a z", "a"], str.split)
    assert stats == pytest.approx((2, 3, 1.5, 1 / 3, 1 / 2, 1 / 2))


def test_command_pattern_is_lossless_and_shared_by_train_encode():
    for text in SAMPLE_FORMULAS + [r"{\frac{a}{b}}", r"\frac{\alpha}{b}", "123", "\\", "\n\t "]:
        assert "".join(pretokenize(text, COMMAND_PAT)) == text
    assert pretokenize(r"\frac{a}{b}", COMMAND_PAT)[0] == r"\frac"
    assert pretokenize("123", COMMAND_PAT) == ["1", "2", "3"]
    texts = [r"\frac{\alpha}{b}"] * 10
    merges = train_bpe(texts, 30, pattern=COMMAND_PAT)
    assert merges == train_bpe_naive(texts, 30, pattern=COMMAND_PAT)
    tok = BPETokenizer(merges, specials=(), pattern=COMMAND_PAT)
    assert len(tok.encode(r"\frac")) == len(tok.encode(r"\alpha")) == 1
    default = BPETokenizer(merges, specials=())
    assert len(default.encode(r"\frac")) > 1
    assert len(tok.encode(r"\frac")) == 1  # 不同规则的实例交替使用，互不干扰。
    assert pretokenize(r"\frac")[0] == "\\"


@pytest.mark.parametrize("text, expected", [
    (r"{\frac{a}{b}}", ["{", r"\frac", "{", "a", "}{", "b", "}}"]),
    (r"\frac{\alpha}{b}", [r"\frac", "{", r"\alpha", "}{", "b", "}"]),
])
def test_command_pattern_protects_nested_commands(text, expected):
    assert pretokenize(text, COMMAND_PAT) == expected


def test_command_pattern_save_load_keeps_exact_ids(tmp_path):
    tok = BPETokenizer(train_bpe([r"\frac"] * 10, 10, pattern=COMMAND_PAT),
                       specials=(), pattern=COMMAND_PAT)
    before = tok.encode(r"\frac")
    assert len(before) == 1
    path = tmp_path / "commands.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.pattern == COMMAND_PAT
    assert loaded.encode(r"\frac") == before
    assert encode_uncached(loaded, r"\frac") == before


def test_legacy_file_without_pattern_uses_default(tmp_path):
    tok = BPETokenizer(train_bpe([r"\frac"] * 10, 10))
    path = tmp_path / "legacy.json"
    tok.save(path)
    saved = json.loads(path.read_text())
    assert saved.pop("pattern") == PAT
    path.write_text(json.dumps(saved))
    loaded = BPETokenizer.load(path)
    assert loaded.pattern == PAT
    assert loaded.encode(r"\frac") == tok.encode(r"\frac")
    assert loaded.special_ids == tok.special_ids


def test_command_statistics_separate_pre_boundaries_from_bpe():
    # 三次出现：预分词切开；预分词完整但 BPE 切开；整个命令在较大 token 中。
    text = r"\ab \ab {\ab}"
    pre = [(0, 1), (1, 3), (3, 7), (7, 13)]
    final = [(0, 1), (1, 3), (3, 5), (5, 7), (7, 13)]
    assert command_split_counts(text, pre, final) == (3, 1, 1, 2)
    assert command_split_counts("没有命令", [(0, 4)], [(0, 4)]) == (0, 0, 0, 0)


def test_local_token_offsets_use_character_positions_after_unicode():
    import regex
    text = "模" + r"\frac"
    raw = BPETokenizer([], specials=())
    offsets = bpe_token_offsets(raw, text)
    assert offsets[:3] == [(0, 1)] * 3
    assert offsets[3:] == [(i, i + 1) for i in range(1, len(text))]
    pre = [m.span() for m in regex.finditer(PAT, text)]
    assert command_split_counts(text, pre, offsets) == (1, 1, 0, 1)
    assert bpe_token_offsets(raw, "") == []


@pytest.mark.parametrize("available", ["gpt2", "Qwen/Qwen3.5-0.8B", None])
def test_part11_local_reference_and_missing_cache_paths(available, monkeypatch, capsys):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    from book.part2_lm.ch06_tokenization import tour

    # 临时训练一个真实 fast tokenizer，检查接口与分支；不依赖模型缓存，不下载。
    texts = [r"\frac{a}{b}", r"{\frac{\alpha}{b}}"]
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.train_from_iterator(texts * 3, trainers.BpeTrainer(
        vocab_size=280, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    ref = PreTrainedTokenizerFast(tokenizer_object=backend)
    monkeypatch.setattr(tour, "real_tokenizer", lambda name: ref if name == available else None)
    monkeypatch.setattr(tour, "data", lambda: (texts, texts))
    monkeypatch.setattr(tour, "tokenizer", lambda n, pattern=None: BPETokenizer([], specials=(), pattern=pattern))
    tour.part11_real()
    output = capsys.readouterr().out
    assert "分词器=LaTeX BPE（默认 PAT）；命令出现次数=3" in output
    for name, model in [("GPT-2", "gpt2"), ("Qwen", "Qwen/Qwen3.5-0.8B")]:
        if model == available:
            assert f"{name} 平均 token 数=" in output
            assert f"分词器={name}；命令出现次数=3" in output
            assert f"模型名={model}；len(tokenizer)（含新增 token）={len(ref)}" in output
            assert f"tokenizer.vocab_size（基础词表）={ref.vocab_size}" in output
            assert output.count(f"{name} 原文=") == 4
        else:
            assert f"第 11 部分跳过 {name}" in output


def test_uncached_encoding_and_whole_decode():
    tok = BPETokenizer(train_bpe(SAMPLE_FORMULAS, 40), specials=())
    for text in SAMPLE_FORMULAS + ["  x +  y\n\t z  ", "模型"]:
        assert encode_uncached(tok, text) == tok.encode(text)
        assert tok.decode(tok.encode(text)) == text
    raw = BPETokenizer([], specials=())
    ids = raw.encode("模")
    assert raw.decode(ids) == "模"
    assert "".join(raw.decode([i]) for i in ids) == "���"
