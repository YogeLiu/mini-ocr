"""A6 参考实现：编辑距离、NED、文字相似度（与 A4 相同）。"""


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def ned(a, b):
    return edit_distance(a, b) / max(len(a), len(b)) if (a or b) else 0.0


def text_sim(a, b):
    return 1.0 - ned(a, b)
