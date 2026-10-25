"""A6 参考实现：一条最简单的数据引擎流水线，从数据池生成一份训练数据。在 course-v3/ 下运行：

    python -m assignments.solutions.a6_data.a6.engine --out outputs/a6/engine.jsonl
    python -m assignments.solutions.a6_data.a6.engine --out ... --set hi=0.95 --set render_medium=false
    python -m assignments.solutions.a6_data.a6.engine --out ... --smoke          # 只处理前 2000 张图

步骤：规范化 → 三元组共识 → 去污染（6 个专家的答案里任何一个和测试集标签的 3-gram Jaccard ≥ 0.7，只看 ≥ 6 个记号的）→ 高档直接收、
中档用渲染校验在 6 个候选里重选（渲染分够高才收）、低档丢掉 → 审核集用真值 → 写出 {"i", "label"}。
每一步都在审核集上报告标签精度和覆盖率。
"""
import argparse
import json
import time

import numpy as np

from assignments.a6_data.pool import load_pool, load_test

from .clean import canon, decontaminate, shingles
from .mining import formula_features
from .consensus import triplet_consensus
from .verify import render_select

CFG = dict(hi=0.9, lo=0.6, render_medium=True, render_min=0.8, decon_tau=0.7, decon_min_tokens=6, use_audit=True,
           smoke=False)


def _audit_report(name, labels, keep, audit):
    ids = [i for i in audit if keep[i]]
    prec = np.mean([labels[i] == audit[i] for i in ids]) if ids else float("nan")
    print(f"  {name:<28} 审核集：覆盖 {len(ids) / len(audit):6.1%}  标签精度 {prec:.4f}")


def run(cfg, out):
    t0 = time.time()
    pool, test = load_pool(), load_test()
    N = 2000 if cfg["smoke"] else len(pool["outputs"])
    audit = {i: t for i, t in pool["audit"].items() if i < N}
    outs = [[canon(o) for o in row] for row in pool["outputs"][:N]]
    res = [triplet_consensus(row) for row in outs]
    labels = [row[m] for row, (m, _, _) in zip(outs, res)]
    score = np.array([s for _, s, _ in res])
    print(f"{N:,} 张图；三元组共识完成（{time.time() - t0:.0f} 秒）")
    keep = np.ones(N, dtype=bool)
    _audit_report("三元组中心点，全部", labels, keep, audit)

    long_ = lambda s: formula_features(s)[0] >= cfg["decon_min_tokens"]          # noqa: E731  短公式本来就常见，不算污染
    flat = [(i, o) for i in range(N) for o in set(outs[i]) if long_(o)]           # 标签有噪声：任何一个专家的答案像测试集就算
    bad = decontaminate([shingles(o) for _, o in flat], [shingles(x) for x in test["labels"] if long_(x)], cfg["decon_tau"])
    bad = sorted({flat[k][0] for k in bad})
    keep[bad] = False
    print(f"  去污染（只看 ≥ {cfg['decon_min_tokens']} 个记号的答案）：去掉 {len(bad)} 张")

    hi = score >= cfg["hi"]
    med = (score >= cfg["lo"]) & ~hi
    _audit_report("只要高档", labels, keep & hi, audit)
    accept = keep & hi
    if cfg["render_medium"]:
        n_ok = 0
        for i in np.nonzero(keep & med)[0]:
            k, s, _ = render_select(outs[i], pool["images"][i])
            if s >= cfg["render_min"]:
                labels[i] = outs[i][k]
                accept[i] = True
                n_ok += 1
        print(f"  中档 {int((keep & med).sum())} 张，渲染校验收下 {n_ok} 张（{time.time() - t0:.0f} 秒）")
    _audit_report("高档 + 校验过的中档", labels, accept, audit)
    if cfg["use_audit"]:
        for i, t in audit.items():
            labels[i], accept[i] = t, keep[i]
    with open(out, "w") as f:
        for i in np.nonzero(accept)[0]:
            f.write(json.dumps({"i": int(i), "label": labels[i]}) + "\n")
    print(f"写出 {int(accept.sum()):,} 条到 {out}（{time.time() - t0:.0f} 秒）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = dict(CFG, smoke=a.smoke)
    for item in a.set:
        k, v = item.split("=", 1)
        cfg[k] = v.lower() in ("1", "true") if isinstance(CFG[k], bool) else type(CFG[k])(v)
    run(cfg, a.out)


if __name__ == "__main__":
    main()
