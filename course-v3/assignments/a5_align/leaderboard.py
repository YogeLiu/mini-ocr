"""A5 的排行榜：既要学会新东西，又不能忘了旧的。复用 A3、A4 的两个测试集：

    ShapesWorld 问答（A3，540 题）：九个任务准确率的平均 acc
    读文字行（A4，500 行）：CER

分数 = ½·acc + ½·(1 − min(CER, 1))，两项都附 95% 区间。在 course-v3/ 下运行：

    python -m assignments.a5_align.leaderboard score shapes.jsonl lines.jsonl
两个文件都是每行一个 JSON 字符串，顺序与各自的测试集相同（见 assignments/a3_vlm/leaderboard.py、assignments/a4_eval_sft/leaderboard.py）。
"""
import argparse
import json

import numpy as np

from assignments.a3_vlm import leaderboard as shapes
from assignments.a4_eval_sft import leaderboard as lines


def score(shape_preds, line_preds, n_boot=2000):
    acc, per = shapes.score(shape_preds)
    test = shapes.load_test()
    ok = np.array([shapes.correct(t, p, a) for (t, _, _, a), p in zip(test, shape_preds)], dtype=float)
    tasks = np.array([t for t, _, _, _ in test])
    idx = np.random.default_rng(0).integers(0, len(ok), size=(n_boot, len(ok)))
    boot = np.mean([np.array([ok[i][tasks[i] == t].mean() if (tasks[i] == t).any() else np.nan for i in idx]) for t in per], 0)
    lo, hi = np.nanquantile(boot, [0.025, 0.975])
    ln = lines.score(line_preds)
    return dict(score=0.5 * acc + 0.5 * (1 - min(ln["cer"], 1.0)), acc=acc, acc_lo=float(lo), acc_hi=float(hi),
                cer=ln["cer"], cer_lo=ln["lo"], cer_hi=ln["hi"], per_task=per)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("score")
    p.add_argument("shapes")
    p.add_argument("lines")
    a = ap.parse_args()
    if a.cmd == "score":
        load = lambda f: [json.loads(x) for x in open(f, encoding="utf-8") if x.strip()]        # noqa: E731
        s = score(load(a.shapes), load(a.lines))
        print("  ".join(f"{t} {v:.3f}" for t, v in s["per_task"].items()))
        print(f"ShapesWorld 平均准确率 {s['acc']:.4f} [{s['acc_lo']:.4f}, {s['acc_hi']:.4f}]；"
              f"文字行 CER {s['cer']:.4f} [{s['cer_lo']:.4f}, {s['cer_hi']:.4f}]；分数 {s['score']:.4f}")
    else:
        print(__doc__)
