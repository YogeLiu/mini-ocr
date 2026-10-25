"""A2 第一部分：预切分、BPE 训练、编码解码。期望的合并规则来自快照（参考实现生成，并与第 14 章的 bpe_train 核对过）。"""
import io
import os

import pytest
import torch

from ._impl import FIXTURES
from ._impl import adapters as A

FX = None


def _fx():
    global FX
    if FX is None:
        FX = torch.load(os.path.join(FIXTURES, "bpe.pt"), weights_only=False)
    return FX


def test_pretokenize():
    assert A.run_pretokenize("The cat 42\\frac{a}") == ["The", " cat", " ", "4", "2", "\\frac", "{", "a", "}"]
    assert A.run_pretokenize("a  b\n\n| x |") == ["a", "  ", "b", "\n", "\n", "|", " x", " ", "|"]   # 按正则的先后顺序匹配
    text = _fx()["corpus"][:5000]
    assert "".join(A.run_pretokenize(text)) == text                     # 不丢、不多任何字符


def test_bpe_toy_example():
    fx = _fx()
    vocab, merges = A.run_train_bpe(fx["toy"], 96 + 6, [])
    assert [tuple(m) for m in merges] == fx["toy_merges"]
    assert len(vocab) == 102 and vocab[0] == "\n" and vocab[1] == " " and vocab[96] == "es"


def test_bpe_corpus_matches_snapshot():
    fx = _fx()
    vocab, merges = A.run_train_bpe(fx["corpus"], fx["vocab_size"], [fx["special"]])
    assert len(vocab) == fx["vocab_size"]
    assert vocab[0] == fx["special"]                                    # 特殊符号排在最前
    assert [tuple(m) for m in merges] == fx["merges"]
    assert dict(vocab) == fx["vocab"]


def test_bpe_specials_are_not_merged():
    fx = _fx()
    text = fx["special"].join(fx["corpus"].split("\n\n")[:200])
    _, merges = A.run_train_bpe(text, 96 + 1 + 100, [fx["special"]])
    assert all(c not in a + b for a, b in merges for c in "<|>"), "特殊符号被拆开参与了合并"


def test_encode_decode():
    fx = _fx()
    tok = A.get_tokenizer(fx["vocab"], fx["merges"], [fx["special"]])
    ids = tok.encode(fx["sample"])
    assert ids == fx["sample_ids"]
    assert tok.decode(ids) == fx["sample"]
    assert tok.encode(fx["special"]) == [0]
    assert tok.encode("") == []


def test_overlapping_specials_prefer_longest():
    fx = _fx()
    s, ss = "<|end|>", "<|end|><|end|>"
    vocab = {0: s, 1: ss, **{i + 2: c for i, c in enumerate(fx["vocab"][k] for k in range(1, 97))}}
    tok = A.get_tokenizer(vocab, [], [s, ss])
    assert tok.encode(ss + s + "a") == [1, 0, vocab_id(vocab, "a")]


def vocab_id(vocab, s):
    return next(i for i, v in vocab.items() if v == s)


def test_encode_iterable_matches_encode():
    fx = _fx()
    tok = A.get_tokenizer(fx["vocab"], fx["merges"], [fx["special"]])
    text = fx["corpus"][:20000]
    assert list(tok.encode_iterable(io.StringIO(text))) == tok.encode(text)
    pieces = [text[i:i + 37] for i in range(0, len(text), 37)]           # 从任意位置切开也要一致
    assert list(tok.encode_iterable(pieces)) == tok.encode(text)


def test_unknown_char_raises():
    fx = _fx()
    tok = A.get_tokenizer(fx["vocab"], fx["merges"], [])
    with pytest.raises(ValueError):
        tok.encode("café")
