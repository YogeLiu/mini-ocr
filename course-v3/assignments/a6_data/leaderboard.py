"""A6 的排行榜：固定训练配方，比谁的数据好。在 course-v3/ 下运行：

    python -m assignments.a6_data.leaderboard check 你的数据.jsonl          # 提交前：格式、污染、规模
    python -m assignments.a6_data.leaderboard score 你的数据.jsonl outputs/a6/run1

score 会重新给 run1/test_preds.jsonl 打分（测试集 1000 个公式，规范化之后整式全对率，95% bootstrap 区间），
并核对：run1 是用**这份**数据、**没改过的** fixed_train.py 训练的；数据里植入的测试集公式不超过 5 条；
最后报告你的标签在数据池真值上的准确率（只作参考，不计分）。
"""
import argparse
import json
import os

from . import fixed_train as F
from . import pool as P

MAX_PLANTED = 5


def check(path):
    rows = F.load_data(path)
    bad = set(P.planted())
    hit = sorted({r["i"] for r in rows if r["i"] in bad})
    truth = P._truth()
    acc = sum(F.canon(r["label"]) == truth[r["i"]] for r in rows) / max(1, len(rows))
    n_unk = sum(t == F.UNK for r in rows for t in F.encode(r["label"]))
    return dict(n_rows=len(rows), n_images=len({r["i"] for r in rows}), planted=len(hit), ok=len(hit) <= MAX_PLANTED,
                label_acc=acc, n_unk=n_unk, data_sha=F._sha(path))


def score(path, run_dir):
    F.check_prereq()
    c = check(path)
    res = json.load(open(os.path.join(run_dir, "result.json")))
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    preds = [json.loads(x) for x in open(os.path.join(run_dir, "test_preds.jsonl")) if x.strip()]
    test = P.load_test()
    problems = []
    if cfg["data_sha"] != c["data_sha"]:
        problems.append("run 用的不是这份数据")
    if cfg["recipe_sha"] != F._sha(F.__file__):
        problems.append("fixed_train.py 被改过")
    if cfg.get("smoke") or len(preds) != len(test["labels"]):
        problems.append("这是 --smoke 的运行")
    if not c["ok"]:
        problems.append(f"数据里有 {c['planted']} 条植入的测试集公式（最多 {MAX_PLANTED} 条）")
    n = len(preds)
    s = F.score(preds, test["labels"][:n], test["levels"][:n])
    return dict(c, **s, track=cfg["track"], train_seconds=res.get("train_seconds"), valid=not problems, problems=problems)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("data")
    p = sub.add_parser("score")
    p.add_argument("data")
    p.add_argument("run")
    a = ap.parse_args()
    if a.cmd == "check":
        c = check(a.data)
        print(f"{c['n_rows']:,} 条，{c['n_images']:,} 张不同的图，<unk> {c['n_unk']} 个；植入的测试集公式 {c['planted']} 条"
              f"（{'通过' if c['ok'] else f'超过 {MAX_PLANTED} 条，无效'}）")
    else:
        s = score(a.data, a.run)
        print(f"赛道 {s['track']}：整式全对 {s['exact']:.4f} [{s['lo']:.4f}, {s['hi']:.4f}]；平均 1 − NED {s['sim']:.4f}；"
              f"按难度 {s['by_level']}")
        print(f"数据 {s['n_rows']:,} 条（{s['n_images']:,} 张图）；植入的测试集公式 {s['planted']} 条；"
              f"标签在真值上的准确率 {s['label_acc']:.4f}（参考，不计分）")
        print("有效" if s["valid"] else "无效：" + "；".join(s["problems"]))
