"""第 1 章：Python 与 NumPy 的演示。逐段运行，对照正文阅读。

    python -m labs.ch01.numpy_tour
"""
import time

import numpy as np

from labs.ch01.drills import char_histogram, most_common
from miniocr.data.latex import gen_formula, to_latex
from miniocr.utils import Vocab, minibatches, one_hot, pad_sequences, pairwise_sq_dists


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def part1_python():
    section("1. 列表、字典与字符统计")
    rng = np.random.default_rng(0)
    formulas = [to_latex(gen_formula(rng, 3)) for _ in range(1000)]
    print("前 3 个公式:", formulas[:3])
    h = char_histogram(formulas)
    print(f"一共 {sum(h.values())} 个字符，{len(h)} 种不同的字符")
    print("最常见的 8 种:", most_common(h, 8))


def part2_vocab():
    section("2. 字符表：文字 <-> 编号")
    v = Vocab("0123456789abcxyz+-=^_{}\\frac", specials=("<pad>", "<eos>"))
    print("字符表大小", len(v), "  itos 前 12 项:", v.itos[:12])
    s = "x^{2}+1"
    ids = v.encode(s)
    print(f"encode({s!r}) = {ids}")
    print(f"decode(...)    = {v.decode(ids)!r}")
    print(f"加上 <eos>: {ids + [v.stoi['<eos>']]} -> decode 跳过特殊符号 -> {v.decode(ids + [1])!r}")


def part3_arrays():
    section("3. 数组：形状、类型、下标")
    a = np.arange(12)
    print("np.arange(12)            ", a, " 形状", a.shape, "类型", a.dtype)
    b = a.reshape(3, 4)
    print("reshape(3, 4):\n", b)
    print("b[1]      =", b[1], "   ← 第 1 行")
    print("b[:, 2]   =", b[:, 2], "      ← 第 2 列")
    print("b[1:, 1:3]=\n", b[1:, 1:3])
    print("b[b > 6]  =", b[b > 6], "  ← 布尔下标：取出所有大于 6 的元素")
    print("b.sum() =", b.sum(), "  b.sum(axis=0) =", b.sum(axis=0), "  b.sum(axis=1) =", b.sum(axis=1))
    c = b[:, 1]
    c[0] = 100
    print("c = b[:, 1]; c[0] = 100 之后 b[0] =", b[0], " ← 切片是视图，改 c 就改了 b！")
    d = b[:, 2].copy()
    d[0] = -1
    print("d = b[:, 2].copy(); d[0] = -1 之后 b[0] =", b[0], " ← copy 之后互不影响")
    x = np.array([200, 100], dtype=np.uint8)
    print("uint8: 200 + 100 =", x[0] + x[1], "（溢出）；转成 int 后 =", int(x[0]) + int(x[1]))
    print("整数除法 7 // 2 =", 7 // 2, "  7 / 2 =", 7 / 2, "  np.array([7]) / 2 的类型:", (np.array([7]) / 2).dtype)


def part4_broadcast():
    section("4. 广播")
    col = np.array([[0], [10], [20]])            # (3, 1)
    row = np.array([1, 2, 3, 4])                 # (4,)
    print("(3,1) + (4,) -> ", (col + row).shape, "\n", col + row)
    imgs = np.ones((2, 4, 4, 3))
    mean = np.array([0.5, 0.4, 0.3])
    print("图像 (2,4,4,3) - 每通道均值 (3,) ->", (imgs - mean).shape, "  第一个像素:", (imgs - mean)[0, 0, 0])
    try:
        np.ones((3, 4)) + np.ones(3)
    except ValueError as e:
        print("(3,4) + (3,) 报错:", e)
    print("修正：(3,4) + (3,1) ->", (np.ones((3, 4)) + np.ones(3)[:, None]).shape)
    print("one_hot([2,0,1], 4):\n", one_hot([2, 0, 1], 4))
    arr, mask = pad_sequences([[5, 6, 7], [8], [9, 9]])
    print("pad_sequences([[5,6,7],[8],[9,9]]):\n", arr, "\nmask:\n", mask.astype(int))


def part5_speed():
    section("5. 为什么要向量化：同一个计算，循环与广播的速度")
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(200, 64)), rng.normal(size=(62, 64))

    t = time.perf_counter()
    D1 = np.zeros((200, 62))
    for i in range(200):
        for j in range(62):
            s = 0.0
            for k in range(64):
                s += (A[i, k] - B[j, k]) ** 2
            D1[i, j] = s
    t_loop = time.perf_counter() - t

    t = time.perf_counter()
    for _ in range(100):
        D2 = pairwise_sq_dists(A, B)
    t_vec = (time.perf_counter() - t) / 100
    print(f"三重 Python 循环 {t_loop * 1000:8.1f} ms")
    print(f"广播             {t_vec * 1000:8.3f} ms   快 {t_loop / t_vec:.0f} 倍；最大差异 {np.abs(D1 - D2).max():.1e}")


def part6_random():
    section("6. 随机数：种子决定一切")
    print("default_rng(0) 的三个数:", np.random.default_rng(0).integers(0, 100, 3))
    print("再来一次 default_rng(0):", np.random.default_rng(0).integers(0, 100, 3), " ← 完全一样")
    print("default_rng(1)          :", np.random.default_rng(1).integers(0, 100, 3))
    rng = np.random.default_rng(0)
    print("同一个 rng 连续取两次:", rng.integers(0, 100, 3), rng.integers(0, 100, 3), " ← 状态在前进")
    print("minibatches(10, 4, rng):", [b.tolist() for b in minibatches(10, 4, np.random.default_rng(0))])


def part7_debug():
    section("7. 调试：一个形状错误是怎样被发现的")

    def mean_image_buggy(imgs):
        return imgs.mean(axis=1)                 # 想对"张数"求平均，却写成了 axis=1

    def mean_image(imgs):
        assert imgs.ndim == 3, f"期望 (N, H, W)，实际 {imgs.shape}"
        out = imgs.mean(axis=0)
        assert out.shape == imgs.shape[1:], out.shape
        return out

    imgs = np.random.default_rng(0).integers(0, 256, size=(10, 32, 32)).astype(np.float32)
    print("错误版本的输出形状:", mean_image_buggy(imgs).shape, " ← (10, 32)：没有报错，但不是我们要的 (32, 32)")
    print("正确版本的输出形状:", mean_image(imgs).shape)
    try:
        mean_image(imgs[0])
    except AssertionError as e:
        print("把单张图传进去，断言立即指出问题:", e)


if __name__ == "__main__":
    part1_python()
    part2_vocab()
    part3_arrays()
    part4_broadcast()
    part5_speed()
    part6_random()
    part7_debug()
    print("\n第 1 章的演示全部完成。")
