"""A1 排行榜的固定测试集与打分。在 course-v3/ 下运行：

    python -m assignments.a1_basics.leaderboard info                 # 测试集的大小和生成方式
    python -m assignments.a1_basics.leaderboard score preds.npy      # 给你的预测打分

测试集存在 data/leaderboard_test.npz 里（参考实现的渲染器生成，所有人用同一份）：
    img：uint8 (3100, 32, 32)，渲染器输出的原图（白底黑字）
    y：  int64 (3100,)，62 类，每类 50 张
用 load_test() 读进来，得到与 build_split 格式相同的 X（float32 (N, 1, 32, 32)，墨迹 = 1）。

规则：**只能用它打最终的分。** 调参、选检查点一律用你自己的验证集；看着测试集调参，分数就不再说明问题（第 25 章）。
"""
import os
import sys

import numpy as np

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "leaderboard_test.npz")
SEED, N_PER_CLASS, DISTORT, NOISE = 20261022, 50, 1.0, 0.05


def load_test():
    d = np.load(PATH)
    X = 1.0 - d["img"].astype(np.float32) / 255.0                   # 与 miniocr.data.imgops.to_float(invert=True) 相同
    return X[:, None], d["y"]


def score(preds):
    _, y = load_test()
    preds = np.asarray(preds).reshape(-1)
    if preds.shape != y.shape:
        raise ValueError(f"预测有 {preds.size} 个，测试集有 {y.size} 张")
    return float((preds == y).mean())


def _build():
    """（维护者用）重新生成测试集。需要参考实现的 miniocr。"""
    from miniocr.data.tasks import glyph_arrays
    X, y = glyph_arrays(np.random.default_rng(SEED), n_per_class=N_PER_CLASS, distort=DISTORT, noise=NOISE, flat=False)
    img = np.rint((1.0 - X) * 255).astype(np.uint8)
    assert np.array_equal(1.0 - img.astype(np.float32) / 255.0, X)
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    np.savez_compressed(PATH, img=img, y=y)
    print(f"写入 {PATH}：{len(y)} 张")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "info"
    if cmd == "build":
        _build()
    elif cmd == "score":
        print(f"准确率 {score(np.load(sys.argv[2])):.2%}")
    else:
        X, y = load_test()
        print(f"{len(y)} 张，{len(set(y.tolist()))} 类；种子 {SEED}，每类 {N_PER_CLASS} 张，distort={DISTORT}，noise={NOISE}")
