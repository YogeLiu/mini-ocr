"""第 33 章：文档解析系统与数据工程。两阶段与端到端的视觉 token 开销、多模型共识、渲染校验、真实数据的清洗与去重、按簇配额的挖掘。

    python -m labs.ch33.engine_tour 1 2 3   # 不训练：视觉 token、模拟专家的共识、渲染校验（约 1 分钟）
    python -m labs.ch33.engine_tour 4 5     # 不训练：真实公式数据（OleehyO/latex-formulas，本机缓存）的清洗、去重、泄漏、配额（约 2 分钟）
    python -m labs.ch33.engine_tour 6       # 你自己跑：用不同来源的标签训练第 13 章的 CRNN，比较伪标签的质量
"""
import glob
import io
import math
import os
import re
import sys
import time
from collections import Counter

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

from miniocr.data.latex import LatexError, SYMBOLS, normalize, to_latex
from miniocr.data.render import sample_formula
from miniocr.data.text import LINE_CHARSET, sample_line_text
from miniocr.engine.clean import dedup, jaccard, latex_key, lsh_pairs, minhash, shingles
from miniocr.engine.consensus import majority_vote, text_sim, tier, triplet_consensus
from miniocr.engine.mining import kmeans, quota_sample
from miniocr.engine.verify import render_score
from miniocr.metrics import ned

torch.set_num_threads(8)
OUT = "outputs/ch33"
os.makedirs(OUT, exist_ok=True)
LATEX_DIR = os.path.expanduser("~/.cache/huggingface/hub/datasets--OleehyO--latex-formulas/snapshots/"
                               "436791f14f92cbb7fdc1b27f2c316cf1f4e65445/cleaned_formulas")


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ---------------------------------------------------------------------- 1
def part1_tokens():
    section("1. 两阶段与端到端：同一页，视觉 token 各要多少（Qwen3.5 的缩放规则，第 24 章）")
    from miniocr.hf_vlm import n_qwen_tokens
    print("  A4 纸 210 × 297 毫米；每 32 × 32 像素合成 1 个视觉 token（16 像素的块，2 × 2 合并）")
    for dpi in (72, 100, 150, 200):
        h, w = round(297 / 25.4 * dpi), round(210 / 25.4 * dpi)
        print(f"    整页 {dpi:3d} dpi（{w:4d} × {h:4d} 像素）           {n_qwen_tokens(h, w):6,d} 个视觉 token")
    print("  两阶段：版面检测之后，每个区块单独送进 VLM（150 dpi 下的典型大小）")
    regions = [("一段正文（宽 16 厘米、8 行）", 960, 260), ("一行公式", 700, 90), ("一张 6 行的表格", 960, 330),
               ("一个标题", 700, 60)]
    tot = 0
    for name, w, h in regions:
        n = n_qwen_tokens(h, w)
        tot += n
        print(f"    {name:<24} {w:4d} × {h:3d}          {n:6,d} 个视觉 token")
    print(f"    上面 4 个区块合计                           {tot:6,d}（整页 150 dpi 是 {n_qwen_tokens(1754, 1240):,d}）")
    print("  小区块被放大到 min_pixels：")
    for mp in (256 * 256, 32 * 32):
        print(f"    min_pixels = {mp:6d}：一行公式 700 × 90 → {n_qwen_tokens(90, 700, min_pixels=mp):4d} 个；"
              f"一个字 40 × 40 → {n_qwen_tokens(40, 40, min_pixels=mp):4d} 个")


# ---------------------------------------------------------------------- 2
CONFUSE = {"l": "1", "o": "0", "e": "c"}


def corrupt(s, rng, p_char=0.0, confuse=False, drop_space=0.0, trunc=0.0):
    """模拟一个专家的错误：逐字的随机替换 / 删除 / 插入；系统性的形近字混淆；丢空格（写法习惯）；截断。"""
    if confuse:
        s = "".join(CONFUSE.get(c, c) for c in s)
    if drop_space:
        s = "".join(c for c in s if c != " " or rng.random() >= drop_space)
    out = []
    for c in s:
        u = rng.random()
        if u < p_char / 3:
            out.append(LINE_CHARSET[rng.integers(len(LINE_CHARSET))])
        elif u < 2 * p_char / 3:
            continue
        elif u < p_char:
            out += [c, LINE_CHARSET[rng.integers(len(LINE_CHARSET))]]
        else:
            out.append(c)
    s = "".join(out)
    if trunc and rng.random() < trunc and len(s) > 4:
        s = s[:int(rng.integers(2, len(s) - 1))]
    return s


EXPERTS = [("A 很准", dict(p_char=0.005)), ("B 准", dict(p_char=0.01)), ("C 一般", dict(p_char=0.02)),
           ("D 难图上形近字混淆", dict(p_char=0.005)), ("E 和 D 同样的毛病", dict(p_char=0.005)),
           ("F 常丢空格", dict(p_char=0.005, drop_space=0.5)), ("G 差", dict(p_char=0.05)),
           ("H 常截断", dict(p_char=0.01, trunc=0.2))]


def simulate_experts(t, rng, p_hard=0.25, hard_mult=6.0):
    """一行文字 t 在 8 个专家那里的输出。25% 的行是"难图"：所有专家的随机错误率乘以 6，
    D、E 还会发生同样的形近字混淆（l→1、o→0、e→c）——它们的错误是相关的，像是用相似的数据训练出来的两个模型。"""
    hard = rng.random() < p_hard
    row = []
    for k, (_, kw) in enumerate(EXPERTS):
        kw = dict(kw, p_char=kw["p_char"] * (hard_mult if hard else 1.0))
        row.append(corrupt(t, rng, confuse=hard and k in (3, 4), **kw))
    return row, hard


def expert_outputs(rng, n):
    truth, outs, hards = [], [], []
    for _ in range(n):
        t = sample_line_text(rng)
        row, hard = simulate_experts(t, rng)
        truth.append(t)
        outs.append(row)
        hards.append(hard)
    return truth, outs, hards


def part2_consensus():
    section("2. 多模型共识：8 个模拟的专家读同一批文字行，从它们的一致程度里挑伪标签")
    rng = np.random.default_rng(0)
    truth, outs, hards = expert_outputs(rng, 2000)
    n = len(truth)
    print(f"  {n} 行；每个专家单独的成绩（和真值比）：")
    for k, (name, _) in enumerate(EXPERTS):
        acc = np.mean([o[k] == t for o, t in zip(outs, truth)])
        sim = np.mean([text_sim(o[k], t) for o, t in zip(outs, truth)])
        print(f"    {name:<18} 整行全对 {acc:6.1%}   平均 1 − NED {sim:.4f}")

    def report(label, picks, scores=None):
        acc = np.mean([p == t for p, t in zip(picks, truth)])
        sim = np.mean([text_sim(p, t) for p, t in zip(picks, truth)])
        print(f"    {label:<30} 整行全对 {acc:6.1%}   平均 1 − NED {sim:.4f}")

    print("  几种合并方式：")
    report("多数投票（完全相同的答案）", [majority_vote(o) for o in outs])
    res = {k: [triplet_consensus(o, k=k) for o in outs] for k in (2, 3, 5, 8)}
    for k in (2, 3, 5, 8):
        report(f"{k} 个最一致的专家的中心点", [o[r[0]] for o, r in zip(outs, res[k])])

    print("  三元组共识按一致度分档（≥ 0.9 高、0.6–0.9 中、< 0.6 低）：")
    tiers = [tier(r[1]) for r in res[3]]
    picks = [o[r[0]] for o, r in zip(outs, res[3])]
    for tname in ("high", "medium", "low"):
        idx = [i for i in range(n) if tiers[i] == tname]
        if not idx:
            print(f"    {tname:<7} 0 行")
            continue
        acc = np.mean([picks[i] == truth[i] for i in idx])
        print(f"    {tname:<7} {len(idx):5d} 行（{len(idx) / n:5.1%}）   伪标签整行全对 {acc:6.1%}")
    for label, flag in (("易图", False), ("难图", True)):
        idx = [i for i in range(n) if hards[i] == flag]
        hi_ = [i for i in idx if tiers[i] == "high"]
        print(f"    {label} {len(idx):5d} 行：高档占 {len(hi_) / len(idx):5.1%}，高档里伪标签整行全对 "
              f"{np.mean([picks[i] == truth[i] for i in hi_]):6.1%}；全部伪标签整行全对 "
              f"{np.mean([picks[i] == truth[i] for i in idx]):6.1%}")
    hi = [i for i in range(n) if tiers[i] == "high"]
    wrong = [i for i in hi if picks[i] != truth[i]]
    in_de = sum(1 for i in wrong if set(res[3][i][2]) >= {3, 4})
    print(f"  高档里的错误伪标签 {len(wrong)} 个，其中 {in_de} 个的三元组里同时有 D 和 E（相关的错误）")
    for i in wrong[:4]:
        print(f"    真值 {truth[i]!r:<32} 伪标签 {picks[i]!r:<32} 三元组 {[EXPERTS[j][0][0] for j in res[3][i][2]]}")
    acc_hi = np.mean([picks[i] == truth[i] for i in hi])
    no_e = [triplet_consensus(o[:4] + o[5:]) for o in outs]
    hi2 = [i for i in range(n) if no_e[i][1] >= 0.9]
    acc2 = np.mean([(outs[i][:4] + outs[i][5:])[no_e[i][0]] == truth[i] for i in hi2])
    print(f"  去掉 E（只留一个有这种毛病的专家）：高档 {len(hi2) / n:5.1%} 的行，伪标签整行全对 {acc2:6.1%}"
          f"（8 个专家时：{len(hi) / n:5.1%}、{acc_hi:6.1%}）")


# ---------------------------------------------------------------------- 3
def _nodes(t, path=()):
    yield path, t
    if t[0] == "row":
        for i, c in enumerate(t[1]):
            yield from _nodes(c, path + (1, i))
    elif t[0] == "script":
        for j in (1, 2, 3):
            if t[j] is not None:
                yield from _nodes(t[j], path + (j,))
    elif t[0] == "frac":
        yield from _nodes(t[1], path + (1,))
        yield from _nodes(t[2], path + (2,))
    elif t[0] == "sqrt":
        yield from _nodes(t[1], path + (1,))


def _replace(t, path, v):
    if not path:
        return v
    t = list(t)
    if t[0] == "row" and path[0] == 1:
        kids = list(t[1])
        kids[path[1]] = _replace(kids[path[1]], path[2:], v)
        t[1] = kids
    else:
        t[path[0]] = _replace(t[path[0]], path[1:], v)
    return tuple(t)


def mutate(tree, rng, kind):
    """在语法树上制造一个错误：sym 换一个符号；script 上下标互换；delete 删掉一行里的一个元素。做不到时返回 None。"""
    ns = list(_nodes(tree))
    if kind == "sym":
        c = [(p, x) for p, x in ns if x[0] == "sym"]
        p, x = c[rng.integers(len(c))]
        new = x[1]
        while new == x[1]:
            new = SYMBOLS[rng.integers(len(SYMBOLS))]
        return _replace(tree, p, ("sym", new))
    if kind == "script":
        c = [(p, x) for p, x in ns if x[0] == "script"]
        if not c:
            return None
        p, x = c[rng.integers(len(c))]
        return _replace(tree, p, ("script", x[1], x[3], x[2]))
    c = [(p, x) for p, x in ns if x[0] == "row" and len(x[1]) > 1]
    if not c:
        return None
    p, x = c[rng.integers(len(c))]
    kids = list(x[1])
    del kids[rng.integers(len(kids))]
    return _replace(tree, p, ("row", kids))


def norm_sim(a, b):
    """先归一化（解析再写成规范写法），再算 1 − NED；解析失败的一方按原样比。"""
    def n(s):
        try:
            return normalize(s)
        except LatexError:
            return s
    return 1.0 - ned(n(a), n(b))


def part3_verify():
    section("3. 渲染校验：把候选的 LaTeX 重新画出来，和原图比（不需要真值）")
    rng = np.random.default_rng(1)
    kinds = {"规范写法": [], "规范写法（不做平移搜索）": [], "等价的另一种写法": [], "换了一个符号": [], "上下标互换": [],
             "删掉一个元素": []}
    ned_equiv = []
    t0 = time.time()
    for _ in range(300):
        tree, img, _ = sample_formula(rng, level=3, noise=0.05)
        canon = to_latex(tree)
        kinds["规范写法"].append(render_score(canon, img))
        kinds["规范写法（不做平移搜索）"].append(render_score(canon, img, shift=0))
        v = to_latex(tree, rng)
        kinds["等价的另一种写法"].append(render_score(v, img))
        ned_equiv.append(1 - ned(v, canon))
        for name, k in (("换了一个符号", "sym"), ("上下标互换", "script"), ("删掉一个元素", "delete")):
            t = mutate(tree, rng, k)
            if t is not None and to_latex(t) != canon:
                kinds[name].append(render_score(to_latex(t), img))
    print(f"  300 个公式（第 3 章的生成器，level 3，噪声 0.05），每个候选的渲染校验分数；{time.time() - t0:.0f} 秒")
    print(f"    {'候选':<22} {'个数':>5} {'≥ 0.99 的比例':>14} {'中位数':>8} {'最低':>7}")
    for name, s in kinds.items():
        s = np.array(s)
        print(f"    {name:<22} {len(s):5d} {np.mean(s >= 0.99):14.1%} {np.median(s):8.3f} {s.min():7.3f}")
    eq = np.array(ned_equiv)
    print(f"  等价写法和规范写法的 1 − NED：平均 {eq.mean():.3f}，{np.mean(eq < 1):.0%} 的不是 1（文本指标会扣分，渲染校验不扣）")

    section("3b. 公式的共识：一致度用原始 LaTeX 的 1 − NED，还是先归一化")
    rng = np.random.default_rng(2)
    rows = []
    for _ in range(500):
        tree, _, _ = sample_formula(rng, level=3)
        outs = []
        for p_err in (0.05, 0.1, 0.1, 0.2, 0.3):                   # 5 个专家：写法各自随机，出错的概率不同
            t = tree
            if rng.random() < p_err:
                t = mutate(tree, rng, ("sym", "script", "delete")[rng.integers(3)]) or tree
            outs.append(to_latex(t, rng))
        rows.append((to_latex(tree), outs))
    for label, sim in (("原始 LaTeX 的 1 − NED", text_sim), ("归一化之后的 1 − NED", norm_sim)):
        res = [triplet_consensus(o, sim=sim) for _, o in rows]
        tiers = Counter(tier(r[1]) for r in res)
        ok_hi = [norm_sim(o[r[0]], t) == 1.0 for (t, o), r in zip(rows, res) if r[1] >= 0.9]
        print(f"    {label:<22} 高 {tiers['high'] / len(rows):5.1%}  中 {tiers['medium'] / len(rows):5.1%}  "
              f"低 {tiers['low'] / len(rows):5.1%}   高档里伪标签（归一化后）正确 {np.mean(ok_hi):6.1%}")


# ---------------------------------------------------------------------- 4
def load_latex(n_files=1):
    import pyarrow.parquet as pq
    files = sorted(glob.glob(LATEX_DIR + "/*.parquet"))[:n_files]
    if not files:
        return None
    return sum((pq.read_table(f, columns=["latex_formula"]).column(0).to_pylist() for f in files), [])


def part4_real():
    section("4. 真实数据：OleehyO/latex-formulas（cleaned_formulas 的第 1 个分片）的清洗、去重与泄漏")
    L = load_latex()
    if L is None:
        print(f"  本机没有缓存这个数据集（{LATEX_DIR}），跳过")
        return
    n = len(L)
    print(f"  {n:,d} 条（整个 cleaned_formulas 是 552,340 条、6 个分片；许可证 openrail，README 说来自 arXiv 爬取 + im2latex-100K）")
    lens = np.array([len(s) for s in L])
    print(f"  长度（字符）：5% 分位 {np.percentile(lens, 5):.0f}，中位数 {np.median(lens):.0f}，"
          f"95% 分位 {np.percentile(lens, 95):.0f}，最长 {lens.max()}")
    env = Counter(re.match(r"\\begin\{([a-z*]+)\}", s).group(1) if s.startswith("\\begin{") else "（无）" for s in L)
    print("  外层环境：" + "，".join(f"{k} {v / n:.1%}" for k, v in env.most_common(4)))
    cmds = Counter(c for s in L for c in re.findall(r"\\[A-Za-z]+", s))
    print(f"  不同的命令 {len(cmds):,d} 种；最常见的 12 个：" + " ".join(f"{c}({v:,d})" for c, v in cmds.most_common(12)))
    once = sum(1 for v in cmds.values() if v == 1)
    print(f"  只出现过一次的命令 {once:,d} 种（{once / len(cmds):.0%}）：长尾，多半是拼写错误或个人宏")
    ok = sum(1 for s in L[:5000] if _parses(latex_key(s)))
    print(f"  前 5000 条里，能被第 3 章的迷你文法解析的：{ok}（{ok / 5000:.1%}）——真实公式远远超出我们的玩具文法")

    t0 = time.time()
    k1 = dedup(L)
    k2 = dedup(L, key=latex_key)
    print(f"  精确去重：{n:,d} → {len(k1):,d}（重复 {1 - len(k1) / n:.2%}）；"
          f"去掉环境与空格之后再去重：→ {len(k2):,d}（重复 {1 - len(k2) / n:.2%}）；{time.time() - t0:.1f} 秒")
    try:
        _image_stats()
    except Exception as e:                                              # 图像解码失败不影响后面的部分
        print(f"  （读取图像失败：{e}）")

    print("  近似重复与泄漏：取去重后的前 20,000 条，随机 90% / 10% 划分（不按组）")
    U = [latex_key(L[i]) for i in k2[:20000]]
    t0 = time.time()
    sh = [shingles(s, 5) for s in U]
    sigs = np.stack([minhash(s) for s in sh])
    t1 = time.time()
    pairs = lsh_pairs(sigs, bands=16)
    t2 = time.time()
    near = [(i, j) for i, j in pairs if jaccard(sh[i], sh[j]) >= 0.8]
    print(f"    MinHash 签名 {t1 - t0:.1f} 秒，LSH 候选对 {len(pairs):,d} 个（{t2 - t1:.1f} 秒），"
          f"其中真的 Jaccard ≥ 0.8 的 {len(near):,d} 对")
    for i, j in sorted(near, key=lambda p: -jaccard(sh[p[0]], sh[p[1]]))[:3]:
        k = next((x for x, (a, b) in enumerate(zip(U[i], U[j])) if a != b), min(len(U[i]), len(U[j])))
        lo = max(0, k - 25)
        print(f"      J = {jaccard(sh[i], sh[j]):.2f}，从第 {k} 个字符起不同：…{U[i][lo:k + 25]!r}\n"
              f"                                 …{U[j][lo:k + 25]!r}")
    perm = np.random.default_rng(0).permutation(len(U))
    val = set(perm[:len(U) // 10].tolist())
    leak = {i if i in val else j for i, j in near if (i in val) != (j in val)}
    print(f"    验证集 {len(val):,d} 条里，有 {len(leak):,d} 条（{len(leak) / len(val):.1%}）在训练集里有 Jaccard ≥ 0.8 的近似重复")


def _parses(s):
    try:
        normalize(s)
        return True
    except LatexError:
        return False


def _image_stats(n=1000):
    import pyarrow.parquet as pq
    from PIL import Image
    f = sorted(glob.glob(LATEX_DIR + "/*.parquet"))[0]
    pf, t, g = pq.ParquetFile(f), [], 0
    while len(t) < n and g < pf.num_row_groups:
        t += pf.read_row_group(g, columns=["image"]).column(0).to_pylist()
        g += 1
    t = t[:n]
    sizes = np.array([Image.open(io.BytesIO(x["bytes"])).size for x in t])
    w, h = sizes[:, 0], sizes[:, 1]
    print(f"  图像（前 {len(t)} 张）：宽 中位数 {np.median(w):.0f}（5%–95%：{np.percentile(w, 5):.0f}–{np.percentile(w, 95):.0f}），"
          f"高 中位数 {np.median(h):.0f}（{np.percentile(h, 5):.0f}–{np.percentile(h, 95):.0f}）；"
          f"宽高比 > 10 的 {np.mean(w / h > 10):.1%}")


# ---------------------------------------------------------------------- 5
FEATS = ("长度", "\\frac", "^", "_", "\\left", "矩阵", "&", "\\\\", "希腊字母")
GREEK = re.compile(r"\\(alpha|beta|gamma|delta|epsilon|varepsilon|theta|lambda|mu|nu|pi|rho|sigma|tau|phi|varphi|psi|omega|"
                   r"Gamma|Delta|Theta|Lambda|Pi|Sigma|Phi|Psi|Omega|xi|eta|zeta|chi|kappa|iota)\b")


def features(s):
    return [math.log(1 + len(s)), s.count("\\frac"), s.count("^"), s.count("_"), s.count("\\left"),
            int(bool(re.search(r"\\begin\{(array|[pbvBV]?matrix|cases)\}", s))), s.count("&"), s.count("\\\\"),
            len(GREEK.findall(s))]


def part5_quota():
    section("5. 按簇配额的抽样：从真实公式里挑 1,000 条，随机抽 vs 每簇同样多的名额")
    L = load_latex()
    if L is None:
        print("  本机没有缓存这个数据集，跳过")
        return
    keep = dedup(L, key=latex_key)
    U = [latex_key(L[i]) for i in keep]
    X = np.array([features(s) for s in U], dtype=float)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    rng = np.random.default_rng(0)
    t0 = time.time()
    labels, C = kmeans(Xs, 8, rng)
    print(f"  {len(U):,d} 条去重后的公式，9 个特征（{'、'.join(FEATS)}）标准化后 k-means 分 8 簇（{time.time() - t0:.1f} 秒）")
    rand = rng.choice(len(U), 1000, replace=False)
    quota = quota_sample(labels, np.ones(len(U)), 1000, rng)
    print(f"    {'簇':<3} {'总数':>7} {'占比':>6} {'随机 1000':>9} {'配额 1000':>9}   这一簇的特点（均值）")
    order = np.argsort([-(labels == c).sum() for c in range(8)])
    for c in order:
        m = X[labels == c].mean(0)
        desc = f"长度 {math.exp(m[0]) - 1:5.0f}  \\frac {m[1]:.1f}  矩阵 {m[5]:.0%}  & {m[6]:.1f}  希腊 {m[8]:.1f}"
        print(f"    {c:<3} {(labels == c).sum():7,d} {(labels == c).mean():6.1%} {(labels[rand] == c).sum():9d} "
              f"{(labels[quota] == c).sum():9d}   {desc}")
    small = order[-2:]
    print(f"  最小的两簇：随机抽到 {sum((labels[rand] == c).sum() for c in small)} 条，配额抽到 "
          f"{sum((labels[quota] == c).sum() for c in small)} 条")


# ---------------------------------------------------------------------- 6
def part6_guide(epochs=4, n_train=20000):
    section("6. 伪标签的质量：用不同来源的标签训练第 13 章的 CRNN（你自己跑）")
    from labs.ch13.ctc_tour import AUG, V, run_eval
    from miniocr.data.text import LineDataset
    from miniocr.metrics import cer, exact_match
    from miniocr.models.crnn import CRNN, CTCCollate, ctc_batch_loss
    from torch.utils.data import DataLoader
    tr = LineDataset(0, n_train, **AUG)
    te = LineDataset(2, 2000, **AUG)
    rng = np.random.default_rng(10)
    outs = [simulate_experts(t, rng)[0] for t in tr.texts]
    res = [triplet_consensus(o) for o in outs]
    sources = {
        "真值": (list(tr.texts), None),
        "单个专家 C（一般）": ([o[2] for o in outs], None),
        "单个专家 G（差）": ([o[6] for o in outs], None),
        "三元组中心点，全部": ([o[r[0]] for o, r in zip(outs, res)], None),
        "三元组中心点，只要高档": ([o[r[0]] for o, r in zip(outs, res)], [i for i, r in enumerate(res) if r[1] >= 0.9]),
    }
    for name, (labels, subset) in sources.items():
        idx = subset if subset is not None else list(range(len(labels)))
        noise = np.mean([labels[i] != tr.texts[i] for i in idx])
        ds = torch.utils.data.Subset(_Relabeled(tr, labels), idx)
        torch.manual_seed(0)
        model = CRNN(len(V), rnn="lstm")
        steps = epochs * math.ceil(len(idx) / 32)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.05)
        loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=CTCCollate(V),
                            generator=torch.Generator().manual_seed(0))
        t0 = time.time()
        for _ in range(epochs):
            model.train()
            for xb, tg, ln, _ in loader:
                loss = ctc_batch_loss(model(xb), tg, ln)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
        preds = run_eval(model, te)
        print(f"    {name:<22} 训练样本 {len(idx):6,d}（标签错 {noise:5.1%}）  测试 CER {cer(preds, te.texts):6.2%}  "
              f"整行全对 {exact_match(preds, te.texts):6.1%}  {time.time() - t0:.0f} 秒")


class _Relabeled(torch.utils.data.Dataset):
    def __init__(self, ds, labels):
        self.ds, self.labels = ds, labels

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds.x[i], self.labels[i]


PARTS = {1: part1_tokens, 2: part2_consensus, 3: part3_verify, 4: part4_real, 5: part5_quota, 6: part6_guide}

if __name__ == "__main__":
    want = [int(a) for a in sys.argv[1:] if a.isdigit()] or list(PARTS)
    for k in want:
        PARTS[k]()
    if want == list(PARTS):
        print("\n第 33 章的演示全部完成。")
