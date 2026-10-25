"""第 5 章：导数、梯度与链式法则。逐段运行，对照正文阅读。

    python -m labs.ch05.calculus_tour        # 图片保存到 outputs/ch05/
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from miniocr.calculus import numerical_derivative, numerical_gradient, rel_error, sigmoid

OUT = "outputs/ch05"


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def part1_slope():
    section("1. 导数是\"放大后的斜率\"：f(x) = x² 在 x = 3 处")
    f = lambda x: x ** 2
    for h in (1.0, 0.1, 0.01, 0.001):
        print(f"  h = {h:<6}  (f(3+h) - f(3)) / h = {(f(3 + h) - f(3)) / h:.6f}")
    print("  h 越小越接近 6 = 2·3")


def part2_h_tradeoff():
    section("2. h 不是越小越好：截断误差 vs 舍入误差")
    f, x, true = np.exp, 1.0, np.e
    hs = 10.0 ** -np.arange(1, 16)
    fwd = [abs((f(x + h) - f(x)) / h - true) for h in hs]
    cen = [abs(numerical_derivative(f, x, h) - true) for h in hs]
    for h, a, b in zip(hs, fwd, cen):
        print(f"  h = 1e-{int(-np.log10(h)):<2d}  前向差分误差 {a:.1e}   中心差分误差 {b:.1e}")
    print(f"  前向差分最准在 h = {hs[int(np.argmin(fwd))]:.0e}，中心差分最准在 h = {hs[int(np.argmin(cen))]:.0e}")
    plt.figure(figsize=(5, 3.5))
    plt.loglog(hs, fwd, "o-", label="forward (f(x+h)-f(x))/h")
    plt.loglog(hs, cen, "s-", label="central (f(x+h)-f(x-h))/2h")
    plt.xlabel("h"); plt.ylabel("|error|"); plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/h_tradeoff.png", dpi=120); plt.close()
    print(f"  已保存 {OUT}/h_tradeoff.png")


def part3_chain():
    section("3. 链式法则：手算一个\"迷你神经元\"的梯度")
    x, y, w, b = 2.0, 1.0, 0.5, -0.3
    z = w * x + b
    p = sigmoid(z)
    L = (p - y) ** 2
    print(f"  前向：z = w·x + b = {z:.4f}   p = σ(z) = {p:.4f}   L = (p - y)² = {L:.4f}")
    dL_dp = 2 * (p - y)
    dp_dz = p * (1 - p)
    dz_dw, dz_db = x, 1.0
    dL_dw = dL_dp * dp_dz * dz_dw
    dL_db = dL_dp * dp_dz * dz_db
    print(f"  反向：dL/dp = 2(p-y) = {dL_dp:.4f}   dp/dz = p(1-p) = {dp_dz:.4f}   dz/dw = x = {dz_dw}")
    print(f"        dL/dw = {dL_dp:.4f} × {dp_dz:.4f} × {dz_dw} = {dL_dw:.6f}   dL/db = {dL_db:.6f}")
    loss = lambda v: (sigmoid(v[0] * x + v[1]) - y) ** 2
    g = numerical_gradient(loss, np.array([w, b]))
    print(f"  数值梯度：dL/dw = {g[0]:.6f}   dL/db = {g[1]:.6f}   相对误差 {rel_error(g, [dL_dw, dL_db]):.1e}")


def bowl(v):
    """一个椭圆形的"碗"：f(w0, w1) = (w0 - 1)² + 4 (w1 + 0.5)²，最低点在 (1, -0.5)。"""
    return (v[0] - 1) ** 2 + 4 * (v[1] + 0.5) ** 2


def bowl_grad(v):
    return np.array([2 * (v[0] - 1), 8 * (v[1] + 0.5)])


def part4_gradient():
    section("4. 梯度：每个方向各求一次偏导")
    p = np.array([-1.0, 1.0])
    print(f"  f(w0, w1) = (w0-1)² + 4(w1+0.5)²  在 {p.tolist()} 处的值 {bowl(p):.2f}")
    print(f"  手算梯度 (2(w0-1), 8(w1+0.5)) = {bowl_grad(p).tolist()}")
    print(f"  数值梯度                       = {np.round(numerical_gradient(bowl, p), 6).tolist()}")


def part5_steepest():
    section("5. 负梯度方向下降最快：和 1000 个随机方向比一比")
    p, eps = np.array([-1.0, 1.0]), 1e-3
    g = bowl_grad(p)
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(1000, 2))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)                 # 单位方向
    drops = np.array([bowl(p) - bowl(p + eps * d) for d in dirs])         # 沿每个方向走 eps，函数下降多少
    best = dirs[drops.argmax()]
    neg_g = -g / np.linalg.norm(g)
    print(f"  随机方向里下降最多的方向 {np.round(best, 3).tolist()}，下降 {drops.max():.6f}")
    print(f"  负梯度方向（单位化）     {np.round(neg_g, 3).tolist()}，下降 {bowl(p) - bowl(p + eps * neg_g):.6f}")
    print(f"  两者夹角 {np.degrees(np.arccos(np.clip(best @ neg_g, -1, 1))):.2f}°")
    print(f"  沿梯度垂直方向走：下降 {bowl(p) - bowl(p + eps * np.array([-neg_g[1], neg_g[0]])):.2e}（几乎为 0：沿等高线走）")


def part6_contour():
    section("6. 等高线与梯度下降的路径")
    w0, w1 = np.meshgrid(np.linspace(-2, 3, 200), np.linspace(-2, 1.5, 200))
    Z = bowl(np.stack([w0, w1]))
    plt.figure(figsize=(5.5, 4))
    cs = plt.contour(w0, w1, Z, levels=[0.25, 1, 2, 4, 8, 16, 32], cmap="viridis")
    plt.clabel(cs, fontsize=7)
    for lr, color in ((0.05, "tab:blue"), (0.2, "tab:red")):
        p, path = np.array([-1.5, 1.2]), []
        for _ in range(25):
            path.append(p.copy())
            p = p - lr * bowl_grad(p)
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], "o-", ms=3, color=color, label=f"lr={lr}")
        print(f"  学习率 {lr}：25 步后到达 {np.round(p, 4).tolist()}，函数值 {bowl(p):.2e}")
    p = np.array([-1.0, 1.0])
    g = bowl_grad(p)
    plt.arrow(p[0], p[1], -0.08 * g[0], -0.08 * g[1], head_width=0.08, color="k")
    plt.plot([1], [-0.5], "k*", ms=12)
    plt.xlabel("w0"); plt.ylabel("w1"); plt.legend(); plt.gca().set_aspect("equal")
    plt.tight_layout(); plt.savefig(f"{OUT}/contour.png", dpi=120); plt.close()
    print(f"  已保存 {OUT}/contour.png（黑色箭头是 (-1, 1) 处的负梯度，五角星是最低点）")
    print("  学习率 0.2 时 w1 方向来回震荡：那个方向的\"坡\"陡（系数 4），一步迈过了谷底。第 7 章细讲。")


def part7_table():
    section("7. 常用导数，用数值导数逐个验证")
    x = 0.7
    rows = [("x³", lambda t: t ** 3, 3 * x ** 2), ("eˣ", np.exp, np.exp(x)), ("ln x", np.log, 1 / x),
            ("1/x", lambda t: 1 / t, -1 / x ** 2), ("√x", np.sqrt, 0.5 / np.sqrt(x)),
            ("σ(x)", sigmoid, sigmoid(x) * (1 - sigmoid(x))), ("tanh x", np.tanh, 1 - np.tanh(x) ** 2),
            ("max(0,x)", lambda t: np.maximum(0, t), 1.0)]
    for name, f, d in rows:
        print(f"  {name:9s} 公式导数 {d:.8f}   数值导数 {numerical_derivative(f, x):.8f}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    part1_slope()
    part2_h_tradeoff()
    part3_chain()
    part4_gradient()
    part5_steepest()
    part6_contour()
    part7_table()
    print("\n第 5 章的演示全部完成。")
