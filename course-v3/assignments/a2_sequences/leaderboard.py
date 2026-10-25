"""A2 的两个排行榜：文字行识别（CER）和语言模型（bits/char）。在 course-v3/ 下运行：

    python -m assignments.a2_sequences.leaderboard info
    python -m assignments.a2_sequences.leaderboard ocr preds.txt          # 每行一个预测，顺序与测试集相同
    python -m assignments.a2_sequences.leaderboard lm --nll 12345.6       # 你在测试文本上算出的总负对数似然（nat）

测试集（参考实现的渲染器生成，所有人用同一份）：
    data/ocr_test.npz：img uint8 (1000, 32, 256)，渲染器输出的原图（白底黑字）；texts：每行的标签
    data/lm_test.txt： 一段混合语料（英文、LaTeX、Markdown 表格），约 4.6 万字符

规则：**只用它们打最终的分。** 调参、选检查点用你自己的验证集。
"""
import argparse
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OCR_PATH = os.path.join(HERE, "data", "ocr_test.npz")
LM_PATH = os.path.join(HERE, "data", "lm_test.txt")
OCR_SEED, OCR_N, NOISE, DISTORT = 20261023, 1000, 0.15, 1.0
LM_SEED, LM_DOCS = 20261024, 300


def load_ocr_test():
    """返回 (X, texts)：X float32 (N, 1, 32, 256)，墨迹 = 1（与 LineDataset 相同）；texts 是字符串列表。"""
    d = np.load(OCR_PATH)
    X = 1.0 - d["img"].astype(np.float32) / 255.0
    return X[:, None], [str(t) for t in d["texts"]]


def load_lm_test():
    return open(LM_PATH, encoding="utf-8").read()


def _edit(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def score_ocr(preds):
    _, refs = load_ocr_test()
    if len(preds) != len(refs):
        raise ValueError(f"预测有 {len(preds)} 行，测试集有 {len(refs)} 行")
    return sum(_edit(p, r) for p, r in zip(preds, refs)) / sum(len(r) for r in refs)


def score_lm(nll_nats):
    return nll_nats / math.log(2) / len(load_lm_test())


def _build():
    """（维护者用）重新生成两个测试集。需要参考实现的 miniocr。"""
    from miniocr.data.corpus import mixed_corpus
    from miniocr.data.text import LineDataset
    ds = LineDataset(OCR_SEED, OCR_N, noise=NOISE, distort=DISTORT)
    X = ds.x.numpy()[:, 0]
    img = np.rint((1.0 - X) * 255).astype(np.uint8)
    assert np.array_equal(1.0 - img.astype(np.float32) / 255.0, X)
    np.savez_compressed(OCR_PATH, img=img, texts=np.array(ds.texts))
    open(LM_PATH, "w", encoding="utf-8").write(mixed_corpus(LM_SEED, LM_DOCS))
    print(f"写入 {OCR_PATH}（{len(ds.texts)} 行）和 {LM_PATH}（{len(load_lm_test()):,} 字符）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("info")
    sub.add_parser("build")
    p = sub.add_parser("ocr")
    p.add_argument("preds")
    p = sub.add_parser("lm")
    p.add_argument("--nll", type=float, required=True)
    a = ap.parse_args()
    if a.cmd == "build":
        _build()
    elif a.cmd == "ocr":
        preds = open(a.preds, encoding="utf-8").read().split("\n")
        if preds and preds[-1] == "":
            preds = preds[:-1]
        print(f"CER {score_ocr(preds):.4f}")
    elif a.cmd == "lm":
        print(f"{score_lm(a.nll):.4f} bits/char")
    else:
        X, t = load_ocr_test()
        print(f"OCR：{len(t)} 行，图 {X.shape[2]}×{X.shape[3]}，{sum(map(len, t)):,} 个字符；种子 {OCR_SEED}，noise={NOISE}，distort={DISTORT}")
        print(f"LM：{len(load_lm_test()):,} 字符；mixed_corpus({LM_SEED}, {LM_DOCS})")
