"""第 13 章：CTC（Connectionist Temporal Classification）。

模型对 T 个时间步（图像从左到右的 T 列）各输出一个 K 类分布，其中第 0 类是"空白"（blank）。
一条"路径"是每个时间步选一个类别得到的长度为 T 的序列；把路径"折叠"（先合并相邻重复，再删掉空白）得到输出文字。
CTC 损失 = −log Σ_{折叠后等于标签的所有路径} Π_t p_t(路径第 t 个类别)。

    collapse              折叠一条路径
    ctc_brute_force       枚举所有路径求概率（只用于很小的例子，验证前向算法）
    ctc_forward           前向算法（动态规划，在对数空间计算），与 F.ctc_loss 一致
    greedy_decode         每步取最大，再折叠
    prefix_beam_search    CTC 前缀束搜索
"""
import itertools
import math

import numpy as np


# >>> 练习 13.1：折叠
def collapse(path, blank=0):
    """先合并相邻的重复，再删掉空白。例如（blank=0）：[1, 1, 0, 1, 2, 2] → [1, 1, 2]。
    注意顺序：先合并再删空白，所以被空白隔开的两个 1 不会被合并。"""
    out, prev = [], None
    for p in path:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return out
# <<<


def ctc_brute_force(probs, target, blank=0):
    """probs (T, K) 是概率（不是对数）。枚举全部 K^T 条路径，把折叠后等于 target 的路径概率相加。
    返回 (总概率, 符合条件的路径列表)。只用于 T、K 很小的手算例子。"""
    T, K = probs.shape
    total, paths = 0.0, []
    for path in itertools.product(range(K), repeat=T):
        if collapse(path, blank) == list(target):
            p = float(np.prod([probs[t, k] for t, k in enumerate(path)]))
            total += p
            paths.append((path, p))
    return total, paths


def _logsumexp(*xs):
    m = max(xs)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(x - m) for x in xs))


# >>> 练习 13.2：前向算法
def ctc_forward(log_probs, target, blank=0):
    """log_probs (T, K) 是对数概率（log_softmax 的输出），target 是标签编号列表（不含空白）。返回 −log p(target)。

    扩展标签：在每个字符前后插入空白，l' = [b, y1, b, y2, b, ..., yU, b]，长度 S = 2U + 1。
    α[t][s] = 前 t+1 个时间步、折叠后恰好走到 l' 的第 s 个位置的所有路径的概率之和（取对数）。
        初始：α[0][0] = log p_0(b)，α[0][1] = log p_0(y1)（U ≥ 1 时），其余 −∞。
        递推：α[t][s] = logsumexp(α[t−1][s], α[t−1][s−1], 允许时 α[t−1][s−2]) + log p_t(l'[s])
              "允许跳过"的条件：l'[s] 不是空白，并且 l'[s] ≠ l'[s−2]（两个相同字符之间必须有空白隔开）。
        结果：log p = logsumexp(α[T−1][S−1], α[T−1][S−2])（以空白结尾或以最后一个字符结尾）。
    T 不够长（T < U + 相邻重复字符的对数）时，p = 0，返回 +∞。
    """
    T = len(log_probs)
    ext = [blank]
    for y in target:
        ext += [y, blank]
    S = len(ext)
    NEG = -math.inf
    alpha = [NEG] * S
    alpha[0] = float(log_probs[0][blank])
    if S > 1:
        alpha[1] = float(log_probs[0][ext[1]])
    for t in range(1, T):
        new = [NEG] * S
        for s in range(S):
            cands = [alpha[s]]
            if s >= 1:
                cands.append(alpha[s - 1])
            if s >= 2 and ext[s] != blank and ext[s] != ext[s - 2]:
                cands.append(alpha[s - 2])
            new[s] = _logsumexp(*cands) + float(log_probs[t][ext[s]])
        alpha = new
    ll = _logsumexp(alpha[S - 1], alpha[S - 2]) if S > 1 else alpha[0]
    return -ll
# <<<


# >>> 练习 13.3：贪心解码
def greedy_decode(log_probs, blank=0):
    """log_probs (T, K)：每个时间步取概率最大的类别，得到一条路径，再折叠。返回编号列表。"""
    return collapse(list(np.asarray(log_probs).argmax(-1)), blank)
# <<<


# >>> 练习 13.4：前缀束搜索
def prefix_beam_search(log_probs, beam=8, blank=0):
    """CTC 前缀束搜索。贪心解码只看"最可能的一条路径"；而一个输出文字的概率是很多条路径之和，
    束搜索按"前缀"（已经折叠好的输出）合并路径，保留概率最大的 beam 个前缀。

    每个前缀记两个数（对数概率）：pb = 以空白结尾的路径总概率，pnb = 以非空白结尾的路径总概率。
    每个时间步 t，对每个前缀 (prefix, pb, pnb) 和每个类别 k（概率 p = p_t(k)）：
        k = 空白：           新 pb[prefix]  += (pb + pnb) · p
        k = prefix 的最后一个字符 c：
                             新 pnb[prefix] += pnb · p              （重复的 c 被合并，前缀不变）
                             新 pnb[prefix + c] += pb · p           （中间隔着空白，才是新的 c）
        k 是其他字符：        新 pnb[prefix + k] += (pb + pnb) · p
    （"+=" 在对数空间里用 logsumexp。）然后按 pb + pnb 保留最大的 beam 个前缀。
    返回概率最大的前缀（编号元组转成列表）。
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    T, K = lp.shape
    NEG = -math.inf
    beams = {(): (0.0, NEG)}                                   # 空前缀：以空白结尾的概率为 1
    for t in range(T):
        nxt = {}

        def add(prefix, b=NEG, nb=NEG):
            ob, onb = nxt.get(prefix, (NEG, NEG))
            nxt[prefix] = (_logsumexp(ob, b), _logsumexp(onb, nb))

        for prefix, (pb, pnb) in beams.items():
            total = _logsumexp(pb, pnb)
            for k in range(K):
                p = lp[t, k]
                if k == blank:
                    add(prefix, b=total + p)
                elif prefix and k == prefix[-1]:
                    add(prefix, nb=pnb + p)
                    add(prefix + (k,), nb=pb + p)
                else:
                    add(prefix + (k,), nb=total + p)
        beams = dict(sorted(nxt.items(), key=lambda kv: -_logsumexp(*kv[1]))[:beam])
    best = max(beams.items(), key=lambda kv: _logsumexp(*kv[1]))[0]
    return list(best)
# <<<
