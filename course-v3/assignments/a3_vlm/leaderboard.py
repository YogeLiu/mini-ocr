"""A3 的排行榜：ShapesWorld 九个任务的问答，分数 = 九个任务准确率的平均。在 course-v3/ 下运行：

    python -m assignments.a3_vlm.leaderboard info
    python -m assignments.a3_vlm.leaderboard score preds.jsonl      # 每行一个 JSON 字符串（你的回答），顺序与测试集相同

测试集（参考实现的场景生成器生成，所有人用同一份）：data/test.npz
    img uint8 (M, 3, 64, 64)：所有图拼在一起；start、count：第 i 题用 img[start[i] : start[i]+count[i]]（compare 两张，其余一张）
    task、question、answer：字符串数组。问题里的 "<image>" 是图像占位符。

判分（与 a3/data.py 的 correct 相同）：ground 任务答案是 "none" 时要完全相同，否则解析出的框与真值 IoU ≥ 0.5 即算对；
其余任务去掉首尾空白后完全相同。
规则：**只用它打最终的分。** 调参、选检查点用你自己的验证集（种子不要用 20261022–20261099）。
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "data", "test.npz")
SEED, N_PER_TASK = 20261025, 60


def load_test():
    """返回列表 [(task, 图像 uint8 (k, 3, 64, 64), 问题, 答案)]。"""
    d = np.load(PATH)
    return [(str(t), d["img"][s:s + c], str(q), str(a))
            for t, s, c, q, a in zip(d["task"], d["start"], d["count"], d["question"], d["answer"])]


def _iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def correct(task, pred, ref):
    from miniocr.data.instruct import parse_box
    if task != "ground" or ref == "none":
        return pred.strip() == ref
    b, r = parse_box(pred), parse_box(ref)
    return b is not None and _iou(b, r) >= 0.5


def score(preds):
    """返回 (平均准确率, {任务: 准确率})。"""
    test = load_test()
    if len(preds) != len(test):
        raise ValueError(f"预测有 {len(preds)} 个，测试集有 {len(test)} 个")
    acc = {}
    for (t, _, _, a), p in zip(test, preds):
        acc.setdefault(t, []).append(correct(t, p, a))
    per = {t: float(np.mean(v)) for t, v in acc.items()}
    return float(np.mean(list(per.values()))), per


def _build():
    """（维护者用）重新生成测试集。需要参考实现。"""
    from assignments.solutions.a3_vlm.a3.data import eval_set
    data = eval_set(SEED, N_PER_TASK)
    imgs, start, count = [], [], []
    for _, img, _, _ in data:
        start.append(sum(len(i) for i in imgs))
        count.append(len(img))
        imgs.append(img)
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    np.savez_compressed(PATH, img=np.concatenate(imgs), start=np.array(start), count=np.array(count),
                        task=np.array([d[0] for d in data]), question=np.array([d[2] for d in data]),
                        answer=np.array([d[3] for d in data]))
    print(f"写入 {PATH}（{len(data)} 题，{sum(count)} 张图）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("info")
    sub.add_parser("build")
    p = sub.add_parser("score")
    p.add_argument("preds")
    a = ap.parse_args()
    if a.cmd == "build":
        _build()
    elif a.cmd == "score":
        preds = [json.loads(line) for line in open(a.preds, encoding="utf-8") if line.strip()]
        mean, per = score(preds)
        print("  ".join(f"{t} {v:.3f}" for t, v in per.items()))
        print(f"平均准确率 {mean:.4f}")
    else:
        test = load_test()
        print(f"{len(test)} 题（每个任务 {N_PER_TASK} 题），种子 {SEED}")
        for t, img, q, ans in test[::N_PER_TASK]:
            print(f"  [{t}] {img.shape[0]} 张图  {q!r} → {ans!r}")
