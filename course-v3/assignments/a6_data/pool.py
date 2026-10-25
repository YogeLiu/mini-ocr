"""A6 的原始数据池和测试集（基础设施：不是你的代码，不要改）。

一个模拟的"爬下来的公式图"数据池：每张图没有真值，只有 6 个"专家"模型各写的一个 LaTeX 答案。
数据池故意带着真实数据池常见的毛病——你的数据引擎（第 33、34 章）要把它们找出来：

    文档分组       每张图属于一篇"文档"（doc），同一篇文档里的公式风格（字号、噪声、难度）相近，还常有改了一个符号的变体
    重复           同一篇文档里同一个公式出现多次（重新渲染，像素不同），也有跨文档的复制
    测试集污染     一部分图是测试集公式的重新渲染
    看不清的图     很重的噪声，专家们基本是在乱写
    专家           alpha 很准；beta 也准，但写法不统一（x^2 / x^{2}、{a \\over b}、x^{2}_{i}）；gamma 一般；
                   delta 与 epsilon 在难图上犯**同一个**结构性错误（上下标对调、分子分母对调、形近字），像用相似数据训练出来的两个模型；
                   zeta 常把答案截断
    审核集         500 张图有人工核对过的真值（audit），可以用来估计精度、调阈值，也可以放进训练集

用法（在 course-v3/ 下）：

    from assignments.a6_data.pool import load_pool, load_test
    pool = load_pool()          # dict：images (N, 48, 192) uint8、outputs（N 行，每行 6 个字符串）、doc (N,)、audit {下标: 真值}
    test = load_test()          # dict：images (1000, 48, 192) uint8、labels（规范写法）

    python -m assignments.a6_data.pool info

规则：**你的数据引擎只能用 load_pool() 返回的东西**（和 load_test() 的标签——去污染要用）。
_truth()、planted() 是给固定训练脚本的 oracle 对照和排行榜的污染检查用的，在你的引擎里读它们等于作弊。
"""
import argparse

import numpy as np

from miniocr.data.latex import DIGITS, SYMBOLS, VARS, gen_formula, to_latex, tokenize
from miniocr.data.render import render_formula

H, W = 48, 192
SEED_POOL, SEED_TEST = 20261050, 20261060
N_DOCS, N_TEST, N_AUDIT = 1200, 1000, 500
EXPERTS = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta")
CONFUSE = {"2": "z", "z": "2", "6": "b", "b": "6", "1": "7", "7": "1"}


# ---------------------------------------------------------------- 公式与渲染
def _style(rng):
    """一篇文档的风格：字号、噪声、各难度的比例。"""
    hard = rng.random() < 0.4
    return dict(size=int(rng.integers(14, 19)),
                noise=float(rng.choice([0.0, 0.05, 0.1, 0.2, 0.3], p=[0.2, 0.3, 0.25, 0.15, 0.1])),
                levels=[0.1, 0.2, 0.35, 0.35] if hard else [0.35, 0.35, 0.2, 0.1])


def _render(tree, rng, size, noise):
    try:
        return render_formula(tree, rng, H=H, W=W, size=size, noise=noise)[0]
    except ValueError:
        return None


def _new(rng, st):
    """按文档风格生成一个放得进图里的公式。"""
    while True:
        level = int(rng.choice(4, p=st["levels"])) + 1
        tree = gen_formula(rng, level)
        img = _render(tree, rng, st["size"], st["noise"])
        if img is not None:
            return tree, img, level


def _leaves(node, out):
    kind = node[0]
    if kind == "sym":
        out.append(node)
    elif kind == "row":
        for c in node[1]:
            _leaves(c, out)
    elif kind == "script":
        for c in node[1:]:
            if c is not None:
                _leaves(c, out)
    elif kind in ("frac", "sqrt"):
        for c in node[1:]:
            _leaves(c, out)
    return out


def _map(node, f):
    """把 f 作用到每个 sym 叶子上（f 返回新的叶子），返回新树。"""
    kind = node[0]
    if kind == "sym":
        return f(node)
    if kind == "op":
        return node
    if kind == "row":
        return ("row", [_map(c, f) for c in node[1]])
    if kind == "script":
        return ("script", _map(node[1], f), *(None if c is None else _map(c, f) for c in node[2:]))
    return (kind, *(_map(c, f) for c in node[1:]))


def _variant(tree, rng):
    """改一个符号（字母换字母、数字换数字）：近似重复。"""
    leaves = [x for x in _leaves(tree, []) if x[1] in VARS + DIGITS[1:]]
    if not leaves:
        return None
    target = leaves[int(rng.integers(len(leaves)))]
    pool = VARS if target[1] in VARS else DIGITS[1:]
    new = str(rng.choice([c for c in pool if c != target[1]]))
    done = []

    def f(x):
        if x is target and not done:
            done.append(1)
            return ("sym", new)
        return x
    return _map(tree, f)


def _structural(tree):
    """delta、epsilon 共有的结构性错误：第一个同时有上下标或只有一种的 script 对调上下标；
    没有 script 就把第一个分式的分子分母对调；都没有就做形近字替换。"""
    state = {"done": False}

    def go(node):
        kind = node[0]
        if state["done"] or kind in ("sym", "op"):
            return node
        if kind == "script":
            state["done"] = True
            return ("script", node[1], node[3], node[2])
        if kind == "frac" and not _has(node, "script"):
            state["done"] = True
            return ("frac", node[2], node[1])
        if kind == "row":
            return ("row", [go(c) for c in node[1]])
        return (kind, *(go(c) for c in node[1:]))
    new = go(tree)
    if state["done"]:
        return new
    return _map(tree, lambda x: ("sym", CONFUSE.get(x[1], x[1])))


def _has(node, kind):
    if node is None or node[0] in ("sym", "op"):
        return False
    if node[0] == kind:
        return True
    kids = node[1] if node[0] == "row" else node[1:]
    return any(_has(c, kind) for c in kids)


# ---------------------------------------------------------------- 专家
def _join(toks):
    """记号接回字符串；命令后面紧跟字母时补一个空格（否则 \\over b 会变成 \\overb）。"""
    out, prev = "", ""
    for t in toks:
        if prev.startswith("\\") and t[:1].isalpha():
            out += " "
        out += t
        prev = t
    return out


def _corrupt(s, rng, p):
    """逐记号的随机错误：替换、删除、插入，各 p/3。"""
    out = []
    for t in tokenize(s):
        u = rng.random()
        if u < p / 3:
            out.append(SYMBOLS[int(rng.integers(len(SYMBOLS)))])
        elif u < 2 * p / 3:
            continue
        elif u < p:
            out += [t, SYMBOLS[int(rng.integers(len(SYMBOLS)))]]
        else:
            out.append(t)
    return _join(out)


def _experts(tree, rng, hard, unreadable):
    m = 15.0 if unreadable else (3.0 if hard else 1.0)
    canon = to_latex(tree)
    shared = hard and rng.random() < 0.7                                  # delta、epsilon 这次是否一起犯结构性错误
    wrong = to_latex(_structural(tree)) if shared else canon
    out = [_corrupt(canon, rng, min(0.9, 0.005 * m)),
           _corrupt(to_latex(tree, rng), rng, min(0.9, 0.008 * m)),
           _corrupt(canon, rng, min(0.9, 0.03 * m)),
           _corrupt(wrong, rng, min(0.9, 0.008 * m)),
           _corrupt(wrong, rng, min(0.9, 0.008 * m))]
    z = _corrupt(canon, rng, min(0.9, 0.02 * m))
    zt = tokenize(z)
    if len(zt) > 3 and rng.random() < 0.15:
        z = _join(zt[:int(rng.integers(1, len(zt) - 1))])
    out.append(z)
    return out


# ---------------------------------------------------------------- 测试集与数据池
def _test_trees():
    rng = np.random.default_rng(SEED_TEST)
    trees, imgs, levels = [], [], []
    for i in range(N_TEST):
        st = _style(rng)
        st["levels"] = [0.25] * 4
        t, img, lv = _new(rng, st)
        trees.append(t)
        imgs.append(img)
        levels.append(lv)
    return trees, np.stack(imgs), levels


_CACHE = {}


def _build():
    if "pool" in _CACHE:
        return _CACHE["pool"]
    test_trees, _, _ = _test_trees()
    rng = np.random.default_rng(SEED_POOL)
    imgs, trees, outs, docs, kinds = [], [], [], [], []
    popular = []                                                          # 跨文档被复制的公式
    big = [t for t in test_trees if len(tokenize(to_latex(t))) >= 6]
    plant_docs = set(rng.choice(N_DOCS, size=150, replace=False).tolist())
    for d in range(N_DOCS):
        st = _style(rng)
        n = int(rng.integers(5, 36))
        mine = []
        for k in range(n):
            u = rng.random()
            unreadable = False
            if mine and u < 0.08:                                         # 同文档的重复
                tree, kind = mine[int(rng.integers(len(mine)))], "dup"
            elif mine and u < 0.28:                                       # 同文档的变体
                tree, kind = _variant(mine[int(rng.integers(len(mine)))], rng), "variant"
            elif popular and u < 0.31:                                    # 跨文档的复制
                tree, kind = popular[int(rng.integers(len(popular)))], "copy"
            else:
                tree, kind = None, "new"
            if d in plant_docs and k < 3:                                 # 测试集污染
                tree, kind = big[int(rng.integers(len(big)))], "planted"
            noise = st["noise"]
            if rng.random() < 0.03:
                noise, unreadable, kind = 0.65, True, kind + "+unreadable"
            img = _render(tree, rng, st["size"], noise) if tree is not None else None
            if img is None:
                tree, img, _ = _new(rng, dict(st, noise=noise))
                kind = "new" + ("+unreadable" if unreadable else "")
            lv = _level(tree)
            hard = noise >= 0.2 or (lv == 4 and rng.random() < 0.5)
            imgs.append(img)
            trees.append(tree)
            outs.append(_experts(tree, rng, hard, unreadable))
            docs.append(d)
            kinds.append(kind)
            mine.append(tree)
            if kind == "new" and rng.random() < 0.05:
                popular.append(tree)
    N = len(imgs)
    perm = rng.permutation(N)                                              # 打乱：文档不再是连续的一段
    imgs = np.stack(imgs)[perm]
    trees = [trees[i] for i in perm]
    outs = [outs[i] for i in perm]
    docs = np.array(docs)[perm]
    kinds = [kinds[i] for i in perm]
    clean = [i for i in range(N) if "unreadable" not in kinds[i] and not kinds[i].startswith("planted")]
    audit = sorted(rng.choice(clean, size=N_AUDIT, replace=False).tolist())
    _CACHE["pool"] = dict(images=imgs, trees=trees, outputs=outs, doc=docs, kinds=kinds, audit=audit)
    return _CACHE["pool"]


def _level(tree):
    s = to_latex(tree)
    if _has(tree, "sqrt") or "(" in s:
        return 4
    if _has(tree, "frac"):
        return 3
    if _has(tree, "script"):
        return 2
    return 1


def load_pool():
    """数据池（第一次调用约 10–20 秒）。返回 dict：
    images (N, 48, 192) uint8（白底黑字）；outputs：N 个列表，每个是 6 个专家（EXPERTS 的顺序）的 LaTeX 字符串；
    doc (N,) int：所属文档；audit：{下标: 真值的规范写法}，500 条。"""
    p = _build()
    return dict(images=p["images"], outputs=[list(o) for o in p["outputs"]], doc=p["doc"].copy(),
                audit={i: to_latex(p["trees"][i]) for i in p["audit"]}, experts=EXPERTS)


def load_test():
    """测试集：1000 张图（4 个难度各约 1/4）和它们的规范写法。"""
    if "test" not in _CACHE:
        trees, imgs, levels = _test_trees()
        _CACHE["test"] = dict(images=imgs, labels=[to_latex(t) for t in trees], levels=np.array(levels))
    t = _CACHE["test"]
    return dict(images=t["images"], labels=list(t["labels"]), levels=t["levels"].copy())


def _truth():
    """**不要在你的数据引擎里用。** 数据池每张图的真值（规范写法）。"""
    return [to_latex(t) for t in _build()["trees"]]


def planted():
    """**不要在你的数据引擎里用。** 被植入的测试集公式（污染）的下标。"""
    return [i for i, k in enumerate(_build()["kinds"]) if k.startswith("planted")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["info"])
    ap.parse_args()
    p = load_pool()
    N = len(p["outputs"])
    print(f"数据池 {N:,} 张图，{len(set(p['doc'].tolist()))} 篇文档，图 {H}×{W}；审核集 {len(p['audit'])} 条；专家 {', '.join(EXPERTS)}")
    for i in list(p["audit"])[:3]:
        print(f"  #{i}  doc {p['doc'][i]}  真值 {p['audit'][i]}")
        for name, o in zip(EXPERTS, p["outputs"][i]):
            print(f"      {name:<8} {o}")
    t = load_test()
    print(f"测试集 {len(t['labels'])} 张；种子 {SEED_TEST}（数据池种子 {SEED_POOL}）")
