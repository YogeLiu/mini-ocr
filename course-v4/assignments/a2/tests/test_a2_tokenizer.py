"""A2 第一部分：预分词、字节级 BPE 训练、编码解码。期望的合并规则来自快照（参考实现生成，并与朴素版逐条核对过）。"""
import io
import os

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
    assert A.run_pretokenize("I'm 12 岁。") == ["I", "'m", " 1", "2", " 岁", "。"]
    assert A.run_pretokenize("模型读入图像，写出文字。") == ["模型读入图像", "，", "写出文字", "。"]
    assert A.run_pretokenize("  缩进\n\n下一行") == [" ", " 缩进", "\n", "\n", "下一行"]
    text = "\n".join(_fx()["lines"][:100])
    assert "".join(A.run_pretokenize(text)) == text                   # 不丢、不多任何字符


def test_bpe_toy_example():
    fx = _fx()
    vocab, merges = A.run_train_bpe(fx["toy"], 6, [])
    assert [tuple(m) for m in merges] == fx["toy_merges"]
    assert [tuple(m) for m in merges][:2] == [(b"s", b"t"), (b"e", b"st")]   # 平局取字节序更大的一对
    assert len(vocab) == 262 and vocab[0] == b"\x00" and vocab[256] == b"st" and vocab[257] == b"est"


def test_bpe_corpus_matches_snapshot():
    fx = _fx()
    vocab, merges = A.run_train_bpe(fx["train_text"], fx["n_merges"], [fx["special"]])
    assert [tuple(m) for m in merges] == fx["merges"]
    assert len(vocab) == 256 + fx["n_merges"] + 1
    assert vocab[256 + fx["n_merges"]] == fx["special"].encode()      # 特殊符号排在最后
    assert dict(vocab) == fx["vocab"]


def test_bpe_specials_are_not_merged():
    fx = _fx()
    _, merges = A.run_train_bpe(fx["train_text"], fx["n_merges"], [fx["special"]])
    assert all(b"<|" not in a + b and b"|>" not in a + b for a, b in merges), "特殊符号被拆开参与了合并"


def test_encode_decode_snapshot():
    fx = _fx()
    tok = A.get_tokenizer(fx["vocab"], fx["merges"], [fx["special"]])
    ids = tok.encode(fx["sample"])
    assert ids == fx["sample_ids"]
    assert tok.decode(ids) == fx["sample"]
    assert tok.encode(fx["special"]) == [256 + fx["n_merges"]]
    assert tok.encode("") == []


def test_rare_chars_fall_back_to_bytes():
    fx = _fx()
    tok = A.get_tokenizer(fx["vocab"], fx["merges"], [fx["special"]])
    ids = tok.encode("龘")                                            # 语料里没有：退回 3 个字节
    assert ids == list("龘".encode())
    assert tok.decode(ids[:2]) == "�"                            # 半个汉字解码成替换字符，不抛异常


def test_overlapping_specials_prefer_longest():
    s, ss = "<|end|>", "<|end|><|end|>"
    vocab = {**{i: bytes([i]) for i in range(256)}, 256: s.encode(), 257: ss.encode()}
    tok = A.get_tokenizer(vocab, [], [s, ss])
    assert tok.encode(ss + s + "a") == [257, 256, ord("a")]


def test_encode_iterable_matches_encode():
    fx = _fx()
    tok = A.get_tokenizer(fx["vocab"], fx["merges"], [fx["special"]])
    text = "\n".join(fx["lines"][:200]) + "\n"
    assert list(tok.encode_iterable(io.StringIO(text))) == tok.encode(text)
