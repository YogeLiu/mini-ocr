"""第 14 章的自动检查。"""
import math

from miniocr.tokenize import BPETokenizer, NGramLM, bits_per_char, bpe_train, pretokenize


def test_pretokenize():
    assert pretokenize("The cat 42\\frac{a}") == ["The", " cat", " ", "4", "2", "\\frac", "{", "a", "}"]
    s = "| a | b |\n$$x^{2}$$ ok.  end"
    assert "".join(pretokenize(s)) == s
    assert " ok" in pretokenize(s) and "." in pretokenize(s)


def test_bpe_train_hand_example():
    # 经典例子：low low low low low lower lower newest newest newest newest newest newest widest widest widest
    text = " ".join(["low"] * 5 + ["lower"] * 2 + ["newest"] * 6 + ["widest"] * 3)
    merges = bpe_train(text, 4)
    # 块是 "low"、" low"、" lower"、" newest"、" widest"；出现最多的对是 ("e","s")：6 + 3 = 9 次
    assert merges[0] == ("e", "s") and merges[1] == ("es", "t")
    assert merges[2] == (" ", "n") or merges[2] == ("l", "o")


def test_tokenizer_roundtrip_and_rank_order():
    text = "abab abab abc abc ab"
    tok = BPETokenizer.train(text, 3)
    for s in ("abab", " abc", "cab ba", "x"[:0] + "ab"):
        assert tok.decode(tok.encode(s)) == s
    # 按规则顺序合并：手工构造规则 (b,c) 优先于 (a,b) 时，"abc" 应切成 a + bc
    t2 = BPETokenizer([("b", "c"), ("a", "b")], "abc")
    assert t2.tokens("abc") == ["a", "bc"] and t2.tokens("abab") == ["ab", "ab"]
    assert len(t2) == 3 + 2


def test_ngram_lm():
    ids = [0, 1, 0, 1, 0, 1]
    lm = NGramLM(2, vocab_size=2, k=0.0).fit(ids)
    assert math.isclose(lm.logprob((0,), 1), 0.0)              # 0 后面总是 1
    assert math.isclose(lm.logprob((-1,), 0), 0.0)             # 开头总是 0
    uni = NGramLM(1, vocab_size=4, k=1.0).fit([0, 0, 1])
    assert math.isclose(math.exp(uni.logprob((), 0)), 3 / 7)  # (2 + 1) / (3 + 4)
    assert math.isclose(uni.cross_entropy([3]), -math.log(1 / 7))
    total = sum(math.exp(lm.logprob((1,), w)) for w in range(2))
    assert math.isclose(total, 1.0)
    assert math.isclose(bits_per_char(math.log(2), 10, 20), 0.5)
