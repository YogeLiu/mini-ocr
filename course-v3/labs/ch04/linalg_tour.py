"""第 4 章：向量、矩阵与形状。逐段运行，对照正文阅读。

    python -m labs.ch04.linalg_tour
"""
import string
import time

import numpy as np

from miniocr.data.imgops import _area_matrix, normalize_glyph, to_float
from miniocr.data.render import render_glyph
from miniocr.linalg import (cosine_similarity, linear, matmul_flops, matmul_loops, mlp_param_count,
                            pairwise_sq_dists_mm, rotation)
from miniocr.utils import pairwise_sq_dists

CHARSET = string.digits + string.ascii_letters


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def glyph_vec(ch, rng, size=16):
    return to_float(normalize_glyph(render_glyph(ch, rng)[0], size), invert=True).ravel()


def part1_vectors():
    section("1. 一张图就是一个向量")
    rng = np.random.default_rng(0)
    a1, a2, b = glyph_vec("A", rng), glyph_vec("A", rng), glyph_vec("B", rng)
    print("归一化到 16×16 的字符图拉平成向量，形状", a1.shape)
    print(f"长度 |A1| = {np.linalg.norm(a1):.2f}   |A2| = {np.linalg.norm(a2):.2f}   |B| = {np.linalg.norm(b):.2f}")
    print(f"点积 A1·A2 = {a1 @ a2:.2f}   A1·B = {a1 @ b:.2f}")
    print(f"距离 |A1-A2| = {np.linalg.norm(a1 - a2):.2f}   |A1-B| = {np.linalg.norm(a1 - b):.2f}")
    cs = cosine_similarity(np.stack([a1]), np.stack([a2, b]))[0]
    print(f"余弦 cos(A1,A2) = {cs[0]:.3f}   cos(A1,B) = {cs[1]:.3f}")


def part2_similar_chars():
    section("2. 哪些字符长得最像：62 个模板两两的余弦相似度")
    rng = np.random.default_rng(1)
    T = np.stack([np.mean([glyph_vec(c, rng) for _ in range(30)], axis=0) for c in CHARSET])   # (62, 256)
    S = cosine_similarity(T, T)                                                               # (62, 62)
    print("模板矩阵", T.shape, " 相似度矩阵", S.shape, " 对角线全是 1:", np.allclose(S.diagonal(), 1))
    iu = np.triu_indices(len(CHARSET), k=1)                     # 上三角（不含对角线）的所有 (i, j)
    order = np.argsort(-S[iu])
    print("最像的 10 对:")
    for r in order[:10]:
        i, j = iu[0][r], iu[1][r]
        print(f"   {CHARSET[i]} {CHARSET[j]}  {S[i, j]:.3f}")
    r = order[-1]
    print(f"最不像的一对: {CHARSET[iu[0][r]]} {CHARSET[iu[1][r]]}  {S[iu[0][r], iu[1][r]]:.3f}")


def part3_matmul():
    section("3. 矩阵乘法：手算、循环、NumPy")
    A = np.array([[1, 2], [3, 4], [5, 6]])
    B = np.array([[7, 8, 9], [10, 11, 12]])
    print("A (3,2) @ B (2,3) =\n", matmul_loops(A, B).astype(int))
    print("C[0,0] = 1·7 + 2·10 =", 1 * 7 + 2 * 10)
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(64, 64)), rng.normal(size=(64, 64))
    t = time.perf_counter(); C1 = matmul_loops(X, Y); t_loop = time.perf_counter() - t
    t = time.perf_counter()
    for _ in range(1000):
        C2 = X @ Y
    t_np = (time.perf_counter() - t) / 1000
    print(f"64×64 @ 64×64：循环 {t_loop * 1000:.1f} ms，NumPy {t_np * 1e6:.1f} µs，快 {t_loop / t_np:.0f} 倍，"
          f"最大差 {np.abs(C1 - C2).max():.1e}")
    print("交换律不成立：A@B 形状", (A @ B).shape, "  B@A 形状", (B @ A).shape)


def part4_linear_layer():
    section("4. 线性层：一批图 → 每类一个分数")
    rng = np.random.default_rng(0)
    x = rng.random((8, 1024))                          # 8 张 32×32 的图，拉平
    W = rng.normal(0, 0.01, size=(1024, 62))
    b = np.zeros(62)
    y = linear(x, W, b)
    print("x", x.shape, "@ W", W.shape, "+ b", b.shape, "->", y.shape)
    print("第 0 张图得分最高的类:", y[0].argmax(), "  每张图的预测:", y.argmax(1))
    tok = rng.random((2, 196, 768))
    P = rng.normal(0, 0.02, size=(768, 1024))
    print("预告第 22 章：2 张图 × 196 个视觉向量 (2,196,768) @ 投影矩阵 (768,1024) ->", linear(tok, P, 0).shape)


def part5_counting():
    section("5. 数参数、数运算量")
    for sizes in ([1024, 62], [1024, 256, 62], [1024, 512, 512, 62]):
        n = mlp_param_count(sizes)
        print(f"MLP {sizes}: {n:>9,} 个参数 = {n * 4 / 1e6:.2f} MB（float32）")
    B, d_in, d_out = 256, 1024, 256
    print(f"一批 {B} 张图过第一层 ({d_in}→{d_out})：{matmul_flops(B, d_in, d_out) / 1e6:.0f} M 次浮点运算")


def part6_geometry():
    section("6. 矩阵是一种变换")
    box = np.array([[10, 10], [30, 10], [30, 20], [10, 20]], dtype=float)   # 矩形的四个角 (x, y)
    S = np.diag([0.5, 2.0])
    print("缩放矩阵 diag(0.5, 2):\n", S, "\n四个角变成\n", box @ S.T)
    R = rotation(np.pi / 6)
    c = box.mean(0)
    rot = (box - c) @ R.T + c
    print("绕中心旋转 30°：\n", np.round(rot, 2))
    print("旋转矩阵是正交的：R.T @ R =\n", np.round(R.T @ R, 10))
    M = _area_matrix(6, 4)
    print("第 2 章的区域平均缩放也是矩阵：_area_matrix(6, 4) =\n", np.round(M, 3))
    print("M @ [10,20,30,40,50,60] =", np.round(M @ np.arange(10, 61, 10), 2))


def part7_einsum():
    section("7. 批量矩阵乘法与 einsum")
    rng = np.random.default_rng(0)
    Q, K = rng.normal(size=(4, 10, 16)), rng.normal(size=(4, 12, 16))
    s1 = Q @ K.transpose(0, 2, 1)
    s2 = np.einsum("bqd,bkd->bqk", Q, K)
    print("Q (4,10,16) @ Kᵀ (4,16,12) ->", s1.shape, "  einsum 'bqd,bkd->bqk' ->", s2.shape,
          " 相同:", np.allclose(s1, s2))
    v = np.arange(3.0)
    print("einsum 'i,i->' (点积) =", np.einsum("i,i->", v, v), "  'i,j->ij' (外积) =\n", np.einsum("i,j->ij", v, v))
    print("einsum 'ii->' (迹) =", np.einsum("ii->", np.eye(3) * 2))


def part8_memory():
    section("8. 两种两两距离：广播 vs 矩阵乘法")
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(2000, 256)), rng.normal(size=(500, 256))
    t = time.perf_counter(); D1 = pairwise_sq_dists(A, B); t1 = time.perf_counter() - t
    t = time.perf_counter(); D2 = pairwise_sq_dists_mm(A, B); t2 = time.perf_counter() - t
    print(f"广播：中间数组 {2000 * 500 * 256 * 8 / 1e9:.2f} GB，用时 {t1 * 1000:.0f} ms")
    print(f"矩阵乘法：中间数组 {2000 * 500 * 8 / 1e6:.0f} MB，用时 {t2 * 1000:.0f} ms；最大差 {np.abs(D1 - D2).max():.1e}")


if __name__ == "__main__":
    part1_vectors()
    part2_similar_chars()
    part3_matmul()
    part4_linear_layer()
    part5_counting()
    part6_geometry()
    part7_einsum()
    part8_memory()
    print("\n第 4 章的演示全部完成。")
