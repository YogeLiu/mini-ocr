"""A4 的排行榜：用你的 VLM 读文字行，按 CER 排名，必须附 95% 置信区间。在 course-v3/ 下运行：

    python -m assignments.a4_eval_sft.leaderboard info
    python -m assignments.a4_eval_sft.leaderboard score preds.jsonl     # 每行一个 JSON 字符串（读出的文字），顺序与测试集相同

测试集（参考实现的渲染器生成，所有人用同一份）：data/lines_test.npz
    img uint8 (500, 32, 256)：render_line 的输出（白底黑字，灰度），noise = 0.1，distort = 0.5
    texts：每行的标签
提问方式不限（参考实现用 "<image>Read the text."），图像怎么送进模型也不限。

分数：语料级 CER = Σ 编辑距离 / Σ 参考长度；95% 区间用逐行 bootstrap（2000 次，百分位法，种子 0）。
规则：**只用它打最终的分。** 调参、选检查点用你自己的验证集（种子不要用 20261022–20261099）。
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "data", "lines_test.npz")
SEED, N, NOISE, DISTORT = 20261026, 500, 0.1, 0.5


def load_test():
    """返回 (img uint8 (N, 32, 256)，texts 列表)。"""
    d = np.load(PATH)
    return d["img"], [str(t) for t in d["texts"]]


def _edit(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def score(preds, n_boot=2000):
    """返回 dict(cer, lo, hi, exact)。"""
    _, refs = load_test()
    if len(preds) != len(refs):
        raise ValueError(f"预测有 {len(preds)} 行，测试集有 {len(refs)} 行")
    ed = np.array([_edit(p, r) for p, r in zip(preds, refs)], dtype=np.float64)
    ln = np.array([len(r) for r in refs], dtype=np.float64)
    idx = np.random.default_rng(0).integers(0, len(ed), size=(n_boot, len(ed)))
    lo, hi = np.quantile(ed[idx].sum(1) / ln[idx].sum(1), [0.025, 0.975])
    return dict(cer=float(ed.sum() / ln.sum()), lo=float(lo), hi=float(hi), exact=float(np.mean([p == r for p, r in zip(preds, refs)])))


def _build():
    """（维护者用）重新生成测试集。"""
    from miniocr.data.render import render_line
    from miniocr.data.text import sample_line_text
    rng = np.random.default_rng(SEED)
    imgs, texts = [], []
    while len(texts) < N:
        t = sample_line_text(rng)
        try:
            g, _ = render_line(t, rng, noise=NOISE, distort=DISTORT)
        except ValueError:
            continue
        imgs.append(g)
        texts.append(t)
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    np.savez_compressed(PATH, img=np.stack(imgs), texts=np.array(texts))
    print(f"写入 {PATH}（{N} 行，{sum(map(len, texts)):,} 个字符）")


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
        s = score(preds)
        print(f"CER {s['cer']:.4f}  95% 区间 [{s['lo']:.4f}, {s['hi']:.4f}]  整行全对 {s['exact']:.3f}")
    else:
        img, t = load_test()
        print(f"{len(t)} 行，图 {img.shape[1]}×{img.shape[2]}，{sum(map(len, t)):,} 个字符；种子 {SEED}，noise={NOISE}，distort={DISTORT}")
        print("前 3 行：", t[:3])
