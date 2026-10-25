"""第 33 章：整页解析的打分。整页的输出是一串区块（类型 + 内容），先把预测的区块和真值的区块配对，再按类型打分、加权。

    match_elements   同类型的区块之间按相似度贪心配对（相似度最高的先配）
    page_score       每种类型：配上的区块按真值字数加权平均相似度，多写的区块的字数算进分母（多写要扣分）；
                     各类型再按真值里的字数占比加权求和

这是 OmniDocBench 和 Xiaomi-OCR-0 的整页奖励（按各模态的字符覆盖率加权）的简化版：真实的实现用更仔细的配对
（匈牙利算法、允许一个区块对多个区块的拆分与合并，Xiaomi-OCR-0 的 MGAM），这里只做一对一的贪心配对。
"""
from ..metrics import ned


def block_sim(kind, pred, ref):
    """文字、标题：1 − NED；公式：CDM；表格：TEDS（第 25 章）。"""
    if kind == "formula":
        from .cdm import cdm
        try:
            return cdm(pred, ref)
        except Exception:
            return 0.0
    if kind == "table":
        from .teds import teds
        try:
            return teds(pred, ref)
        except Exception:
            return 0.0
    return max(0.0, 1.0 - ned(pred, ref))


# >>> 练习 33.8：区块配对
def match_elements(pred, ref, sim=block_sim):
    """pred、ref：[(类型, 内容), …]。只在类型相同的区块之间配对。
    算出所有同类型的 (i, j) 的相似度 s = sim(类型, pred[i] 的内容, ref[j] 的内容)，按 s 从大到小（相同时 i 小、再 j 小的在前）
    依次考虑：i 和 j 都还没配过、且 s > 0 时配成一对。返回 [(i, j, s), …]（按配对的先后）。"""
    cand = [(sim(pk, pc, rc), i, j) for i, (pk, pc) in enumerate(pred) for j, (rk, rc) in enumerate(ref) if pk == rk]
    cand.sort(key=lambda t: (-t[0], t[1], t[2]))
    used_p, used_r, out = set(), set(), []
    for s, i, j in cand:
        if s > 0 and i not in used_p and j not in used_r:
            used_p.add(i)
            used_r.add(j)
            out.append((i, j, s))
    return out
# <<<


def page_score(pred, ref, sim=block_sim):
    """返回 (总分, {类型: 这一类的分数})。
    这一类的分数 = Σ_配上的对 s · len(真值内容) / (Σ_这一类的真值 len + Σ_这一类没配上的预测 len)；
    总分 = Σ_类型 这一类的分数 · (这一类真值的字数 / 真值的总字数)。真值为空时返回 (1.0 if 预测也为空 else 0.0, {})。"""
    total = sum(len(c) for _, c in ref)
    if total == 0:
        return (1.0 if not pred else 0.0), {}
    pairs = match_elements(pred, ref, sim)
    matched_p = {i for i, _, _ in pairs}
    per = {}
    for kind in sorted({k for k, _ in ref}):
        num = sum(s * len(ref[j][1]) for i, j, s in pairs if ref[j][0] == kind)
        den = sum(len(c) for k, c in ref if k == kind)
        den += sum(len(c) for i, (k, c) in enumerate(pred) if k == kind and i not in matched_p)
        per[kind] = num / den
    score = sum(per[k] * sum(len(c) for kk, c in ref if kk == k) / total for k in per)
    return score, per
