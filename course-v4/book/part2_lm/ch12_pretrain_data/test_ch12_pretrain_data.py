"""第 12 章：人造语料、独立算术参考和缓存故障；不读真实语料。"""
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from miniocr.data.dedup import (
    MinHasher, _stable_hash, estimate_jaccard, exact_dedup, exact_key, jaccard,
    lsh_candidates, lsh_probability, lsh_threshold, near_dedup, shingles,
)
from miniocr.data.latex_lm import encode_formulas
from miniocr.data.token_file import (
    StaleCacheError, cache_key, load_token_file, write_token_file,
)
from miniocr.tokenizer import BPETokenizer, train_bpe


def test_exact_stable_first():
    texts = ["z", "x^2", "a", "x^{2}", "z", "b"]
    assert exact_dedup(texts) == (["z", "x^2", "a", "b"], 2)
    assert exact_key("x^2") == exact_key("x^{2}")
    assert len(exact_key("x^2")) == 64


def test_shingles_and_jaccard_boundaries():
    assert shingles("", 3) == set()
    assert shingles("x", 3) == {("x",)}
    assert shingles("x+y", 2) == {("x", "+"), ("+", "y")}
    assert shingles(r"\alpha+x", 2) == {(r"\alpha", "+"), ("+", "x")}
    assert shingles("x^2", 3) == shingles("x^{2}", 3)
    assert jaccard(set(), set()) == 1.0
    assert jaccard(set(), {1}) == 0.0
    assert jaccard({1, 2}, {2, 3}) == 1/3
    with pytest.raises(ValueError):
        shingles("x", 0)


def test_minhash_determinism_and_error():
    a, b = {str(i) for i in range(75)}, {str(i) for i in range(25, 100)}
    hasher = MinHasher(512, 42)
    sig = hasher.signature(a)
    assert sig.dtype == np.uint64 and sig.shape == (512,)
    np.testing.assert_array_equal(sig, MinHasher(512, 42).signature(a))
    assert estimate_jaccard(sig, hasher.signature(a)) == 1
    assert not np.array_equal(sig, MinHasher(512, 43).signature(a))
    truth = jaccard(a, b)
    assert truth == .5
    assert abs(estimate_jaccard(sig, hasher.signature(b))-truth) < 4*math.sqrt(truth*(1-truth)/512)
    empty = hasher.signature(set())
    assert np.all(empty == np.iinfo(np.uint64).max)
    assert estimate_jaccard(empty, sig) == 0


def test_modular_multiplication_matches_python_integers():
    hasher = MinHasher(16, 7)
    assert hasher.P == (1 << 31)-1
    assert np.all((1 <= hasher.a) & (hasher.a < hasher.P))
    assert np.all(hasher.b < hasher.P)
    # 除随机系数外显式核验最大乘加，Python 整数参考不会发生 uint64 溢出。
    hasher.a[:2] = [1, hasher.P-1]
    hasher.b[:2] = [0, hasher.P-1]
    assert (hasher.P-1)**2 + hasher.P-1 < 2**62
    values = np.array([0, 1, hasher.P//2, hasher.P-2, hasher.P-1], dtype=np.uint64)
    expected = [[(int(a)*int(x)+int(b)) % hasher.P for x in values]
                for a, b in zip(hasher.a, hasher.b)]
    np.testing.assert_array_equal(hasher._hashes(values), np.array(expected, dtype=np.uint64))
    items = {("x", "+", "y"), ("y", "^", "2")}
    expected_min = [min((int(a)*(_stable_hash(s) % hasher.P)+int(b)) % hasher.P for s in items)
                    for a, b in zip(hasher.a, hasher.b)]
    np.testing.assert_array_equal(hasher.signature(items), expected_min)


def test_near_dedup_timing_includes_all_phases(monkeypatch):
    from miniocr.data import dedup
    ticks = iter([10., 14., 17.])
    monkeypatch.setattr(dedup.time, "perf_counter", lambda: next(ticks))
    _, info = near_dedup(["x", "x", "y"])
    assert info["timings"] == dict(signature_seconds=4., lsh_verify_seconds=3.,
                                   total_seconds=7.)


def test_hash_seed_independence():
    code = ("from miniocr.data.dedup import MinHasher; "
            "print(MinHasher(16, 8).signature({'甲','乙','x','y'}).tolist())")
    outputs = [subprocess.check_output([sys.executable, "-c", code],
               env={**os.environ, "PYTHONHASHSEED": seed, "CUDA_VISIBLE_DEVICES": ""})
               for seed in ("1", "97")]
    assert outputs[0] == outputs[1]


def test_lsh_candidates_and_limits():
    hasher = MinHasher(64, 0)
    a = hasher.signature({"a", "b"})
    b = hasher.signature({"c", "d"})
    assert lsh_candidates(np.array([a, a, b]), 8, 8) == {(0, 1)}
    with pytest.raises(ValueError, match="不得超过"):
        lsh_candidates(np.array([a]), 9, 8)
    with pytest.raises(ValueError):
        lsh_candidates(np.array([a]), 0, 8)
    assert lsh_candidates(np.empty((0, 64), dtype=np.uint64), 8, 8) == set()


def test_lsh_probability():
    assert lsh_probability(1, 2, 2) == 1
    assert lsh_probability(0, 2, 2) == 0
    assert lsh_probability(.5, 2, 2) == .4375
    assert lsh_threshold(16, 4) == .5
    with pytest.raises(ValueError):
        lsh_probability(1.1, 2, 2)


def test_near_dedup_first_and_exact_verification():
    a = "x_1+x_2+x_3+x_4+x_5+x_6=0"
    b = a[:-1]+"1"
    texts = ["z", a, b, a, "q"]
    assert jaccard(shingles(a, 3), shingles(b, 3)) >= .8
    keep, info = near_dedup(texts, threshold=.8, num_perm=128, bands=32, rows=4)
    assert keep == [0, 1, 4]
    assert info["clusters"] == [[0], [1, 2, 3], [4]]
    strict, _ = near_dedup(texts, threshold=1, num_perm=128, bands=32, rows=4)
    assert strict == [0, 1, 2, 4]
    assert near_dedup([])[0] == []


@pytest.fixture
def tokenizer():
    return BPETokenizer(train_bpe(["alpha+alpha", "beta+beta", "中文公式"]*3, 12),
                        specials=("<|bos|>", "<|eos|>"))


def test_token_file_round_trip(tmp_path, tokenizer):
    texts = ["x^2", "中文公式", "<|bos|>+x"]
    path = tmp_path / "tokens"
    meta = write_token_file(path, tokenizer, texts, "v1")
    mapped = load_token_file(path, tokenizer, "v1")
    seqs, n_bytes = encode_formulas(tokenizer, texts)
    assert isinstance(mapped, np.memmap)
    assert mapped.dtype == np.uint16 and not mapped.flags.writeable
    np.testing.assert_array_equal(mapped, [i for seq in seqs for i in seq])
    assert meta["formula_count"] == len(texts)
    assert meta["text_utf8_bytes"] == n_bytes
    assert meta["token_count"]*2 == Path(str(path)+".bin").stat().st_size
    saved = tmp_path / "tok.json"
    tokenizer.save(saved)
    assert cache_key(tokenizer, "v1") == cache_key(BPETokenizer.load(saved), "v1")
    assert cache_key(tokenizer, "v1", {"a": 1, "b": 2}) == cache_key(tokenizer, "v1", {"b": 2, "a": 1})


def test_stale_tokenizer_and_version(tmp_path, tokenizer):
    path = tmp_path / "tokens"
    write_token_file(path, tokenizer, ["alpha"], "v1")
    other = BPETokenizer([], specials=tokenizer.specials)
    with pytest.raises(StaleCacheError, match="分词器内容"):
        load_token_file(path, other, "v1")
    with pytest.raises(StaleCacheError, match="数据版本"):
        load_token_file(path, tokenizer, "v2")
    changed_specials = BPETokenizer(tokenizer.merges, specials=list(reversed(tokenizer.specials)))
    assert cache_key(tokenizer, "v1") != cache_key(changed_specials, "v1")
    changed_pattern = BPETokenizer(tokenizer.merges, specials=tokenizer.specials, pattern=r".")
    assert cache_key(tokenizer, "v1") != cache_key(changed_pattern, "v1")


def test_truncated_token_file(tmp_path, tokenizer):
    path = tmp_path / "tokens"
    write_token_file(path, tokenizer, ["x+y"], "v1")
    binary = Path(str(path)+".bin")
    binary.write_bytes(binary.read_bytes()[:-1])
    with pytest.raises(ValueError, match="文件大小"):
        load_token_file(path, tokenizer, "v1")


def test_tampered_cache_key(tmp_path, tokenizer):
    path = tmp_path / "tokens"
    meta = write_token_file(path, tokenizer, ["x"], "v1")
    meta["cache_key"] = "错误摘要"
    Path(str(path)+".json").write_text(json.dumps(meta))
    with pytest.raises(StaleCacheError, match="摘要"):
        load_token_file(path, tokenizer, "v1")


def test_large_vocab_rejected(tmp_path):
    class LargeTokenizer:
        def __len__(self):
            return 65536
    with pytest.raises(ValueError, match="65536"):
        write_token_file(tmp_path / "bad", LargeTokenizer(), ["x"], "v1")


def test_empty_token_file(tmp_path, tokenizer):
    path = tmp_path / "empty"
    write_token_file(path, tokenizer, [], "v1")
    result = load_token_file(path, tokenizer, "v1")
    assert result.size == 0 and not result.flags.writeable


def test_cli_invalid_part(capsys):
    from book.part2_lm.ch12_pretrain_data.tour import main
    with pytest.raises(SystemExit) as exc:
        main(["11"])
    assert exc.value.code == 2
    assert "部分号须为1–10" in capsys.readouterr().err


def test_cli_small_without_corpus(monkeypatch, capsys):
    from book.part2_lm.ch12_pretrain_data import tour
    def forbidden(*args, **kwargs):
        raise AssertionError("样例模式不得读取真实语料")
    monkeypatch.setattr(tour, "load_formulas", forbidden)
    monkeypatch.setattr(tour, "corpus_available", forbidden)
    with pytest.warns(UserWarning, match="主动降级"):
        tour.main(["--small", "2"])
    assert "第 2 部分用时秒=" in capsys.readouterr().out


def test_part6_reports_original_timing_when_cached(monkeypatch):
    from argparse import Namespace
    from book.part2_lm.ch12_pretrain_data import tour
    exp = tour.Experiment(Namespace(seed=0, sample=3, small=True))
    exp.all_texts = ["x", "y", "z"]
    _, info = exp.near(exp.unique)
    info["timings"] = dict(signature_seconds=4., lsh_verify_seconds=3., total_seconds=7.)
    records = []
    monkeypatch.setattr(tour, "line", lambda **values: records.append(values))

    def forbidden(*args, **kwargs):
        raise AssertionError("已有近重复结果应复用")

    monkeypatch.setattr(tour, "near_dedup", forbidden)
    tour.part6(exp)
    timing = next(row for row in records if "计时来源" in row)
    assert timing == {"计时来源": "复用首次实际计算记录", "签名用时秒": 4.,
                      "LSH+核验用时秒": 3., "near_dedup完整用时秒": 7.}


def test_part10_full_sample_and_separate_mathtext_subset(monkeypatch, capsys):
    from argparse import Namespace
    from book.part2_lm.ch12_pretrain_data import tour
    # 不同汉字各为一个 token，保证前 2001 条互不近似；后面两条的 shingle 集相同。
    distinct = [chr(0x4e00+i) for i in range(2001)]
    texts = distinct + ["x+y+x+y", "y+x+y+x", "{{", " ", distinct[0]]
    exp = tour.Experiment(Namespace(seed=0, sample=len(texts), small=True))
    exp.all_texts = texts
    records, parsed = [], []
    monkeypatch.setattr(tour, "line", lambda **values: records.append(values))

    def parse(text):
        parsed.append(text)
        return text != distinct[0]

    monkeypatch.setattr(exp, "parse", parse)
    tour.part10(exp)
    assert next(row["报告原始样本数"] for row in records if "报告原始样本数" in row) == 2006
    steps = {row["步骤"]: row for row in records if "步骤" in row}
    assert list(steps) == ["原始", "清洗去空", "规范键精确去重", "近重复连通分量", "弱语法检查"]
    assert [row["剩余条数"] for row in steps.values()] == [2006, 2005, 2004, 2003, 2002]
    assert parsed == distinct[:2000]
    subset = next(row for row in records if "子集示范" in row)
    assert subset["子集条数"] == 2000
    assert subset["子集删除条数"] == 1
    assert subset["子集删除比例"] == 1/2000
    assert subset["子集剩余条数"] == 1999
    assert subset["探索样本最终剩余条数"] == steps["弱语法检查"]["剩余条数"]
    assert "按该比例外推不可靠" in capsys.readouterr().out
