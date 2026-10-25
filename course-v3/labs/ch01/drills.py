"""第 1 章的 Python 基础练习：只用 Python 本身（不用 NumPy）。"""


# >>> 练习 1.1：字符统计
def char_histogram(texts):
    """统计若干个字符串里每个字符出现的次数，返回字典 {字符: 次数}。

    例如 ["ab", "b"] -> {"a": 1, "b": 2}。用 dict.get(key, 默认值) 处理"第一次见到"的字符。
    """
    counts = {}
    for t in texts:
        for c in t:
            counts[c] = counts.get(c, 0) + 1
    return counts
# <<<


def most_common(counts, k):
    """出现次数最多的 k 个 (字符, 次数)，次数相同时按字符排序（保证结果确定）。"""
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
