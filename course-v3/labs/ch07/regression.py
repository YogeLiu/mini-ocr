"""第 7 章：梯度下降训练线性回归、逻辑回归、softmax 回归。逐段运行，对照正文阅读。

    python -m labs.ch07.regression           # 图片保存到 outputs/ch07/
"""
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from miniocr.calculus import numerical_gradient, rel_error
from miniocr.data.shapes import SHAPES
from miniocr.data.tasks import CHARSET, glyph_arrays, shape_sizes
from miniocr.models.linear_np import logistic_loss_grad, mse_loss_grad, softmax_loss_grad, standardize
from miniocr.utils import minibatches

OUT = "outputs/ch07"


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def gd(loss_grad, params, lr, steps, log_every=None):
    """全批量梯度下降：params 是参数列表，loss_grad(*params) 返回 (损失, 梯度...)。返回每一步的损失。"""
    params = [np.array(p, dtype=np.float64) for p in params]
    hist = []
    for t in range(steps):
        out = loss_grad(*params)
        hist.append(out[0])
        if not np.isfinite(out[0]):
            break
        for p, g in zip(params, out[1:]):
            p -= lr * g
        if log_every and t % log_every == 0:
            print(f"    第 {t:4d} 步  损失 {out[0]:.4f}")
    return params, hist


def part1_linear():
    section("1. 线性回归：从面积猜边长")
    area, side, kind = shape_sizes(np.random.default_rng(0), n=600)
    a_te, s_te, k_te = shape_sizes(np.random.default_rng(1), n=300)
    print("  前 5 个样本 (面积, 边长, 形状):", [(int(a), int(s), SHAPES[k]) for a, s, k in zip(area[:5], side[:5], kind[:5])])
    X, Xte = np.sqrt(area)[:, None], np.sqrt(a_te)[:, None]
    (w, b), hist = gd(lambda w, b: mse_loss_grad(X, side, w, b), [np.zeros(1), 0.0], lr=0.006, steps=30001, log_every=5000)
    print(f"  梯度下降得到  边长 ≈ {w[0]:.4f}·√面积 + {b:.4f}")
    A = np.hstack([X, np.ones_like(X)])
    sol = np.linalg.lstsq(A, side, rcond=None)[0]
    print(f"  最小二乘闭式解 边长 ≈ {sol[0]:.4f}·√面积 + {sol[1]:.4f}")
    pred = Xte[:, 0] * w[0] + b
    print(f"  测试集均方误差 {np.mean((pred - s_te) ** 2):.3f}，平均绝对误差 {np.mean(np.abs(pred - s_te)):.3f} 像素")
    for k, name in enumerate(SHAPES):
        r = (pred - s_te)[k_te == k]
        print(f"    {name:8s} 平均残差 {r.mean():+.3f}")
    onehot = np.eye(3)[kind]
    X2 = np.hstack([X, onehot[:, 1:]])                                 # √面积 + "是正方形" + "是三角形"
    X2te = np.hstack([Xte, np.eye(3)[k_te][:, 1:]])
    (w2, b2), _ = gd(lambda w, b: mse_loss_grad(X2, side, w, b), [np.zeros(3), 0.0], lr=0.006, steps=30001)
    pred2 = X2te @ w2 + b2
    sol2 = np.linalg.lstsq(np.hstack([X2, np.ones((len(X2), 1))]), side, rcond=None)[0]
    print(f"  加上形状特征后的闭式解 w = {np.round(sol2[:3], 3).tolist()}，b = {sol2[3]:.3f}")
    print(f"  加上形状特征后：w = {np.round(w2, 3).tolist()}，b = {b2:.3f}；测试集平均绝对误差 {np.mean(np.abs(pred2 - s_te)):.3f} 像素")


def part2_lr():
    section("2. 学习率的四种结局（以及特征缩放）")
    area, side, _ = shape_sizes(np.random.default_rng(0), n=600)
    X = np.stack([np.sqrt(area), area], 1)                             # 两个特征，尺度相差约 15 倍
    print(f"  特征 √面积 的范围 [{X[:, 0].min():.0f}, {X[:, 0].max():.0f}]，面积 的范围 [{X[:, 1].min():.0f}, {X[:, 1].max():.0f}]")
    plt.figure(figsize=(9, 3.5))
    for j, (Xu, tag) in enumerate(((X, "raw"), (standardize(X)[0], "standardized"))):
        plt.subplot(1, 2, j + 1)
        lrs = (3e-6, 1e-5, 3.5e-5, 4e-5) if tag == "raw" else (0.01, 0.1, 0.48, 0.52)
        for lr in lrs:
            _, hist = gd(lambda w, b: mse_loss_grad(Xu, side, w, b), [np.zeros(2), 0.0], lr=lr, steps=200)
            h = np.array(hist)
            end = "发散" if (not np.isfinite(h[-1]) or h[-1] > h[0]) else f"{h[-1]:.3f}"
            print(f"  {tag:12s} lr = {lr:<6g}  第 10 步 {h[min(10, len(h) - 1)]:12.4g}  第 200 步 {end}")
            plt.semilogy(np.clip(h, 1e-3, 1e6), label=f"lr={lr:g}")
        plt.title(tag); plt.xlabel("step"); plt.ylabel("MSE"); plt.legend(fontsize=7); plt.ylim(1e-1, 1e6)
    plt.tight_layout(); plt.savefig(f"{OUT}/lr_outcomes.png", dpi=120); plt.close()
    print(f"  已保存 {OUT}/lr_outcomes.png")


def part3_logistic():
    section("3. 逻辑回归：区分 b 和 d")
    Xtr, ytr = glyph_arrays(np.random.default_rng(0), chars="bd", n_per_class=200, size=16, align=True)
    Xte, yte = glyph_arrays(np.random.default_rng(1), chars="bd", n_per_class=200, size=16, align=True)
    Xtr, Xte, ytr, yte = Xtr.astype(np.float64), Xte.astype(np.float64), ytr.astype(float), yte.astype(float)
    w0 = np.random.default_rng(0).normal(0, 0.01, 256)
    g_num = numerical_gradient(lambda v: logistic_loss_grad(Xtr[:20], ytr[:20], v, 0.0)[0], w0)
    g_ana = logistic_loss_grad(Xtr[:20], ytr[:20], w0, 0.0)[1]
    print(f"  梯度检查（20 个样本，256 个权重）：相对误差 {rel_error(g_num, g_ana):.1e}")
    (w, b), hist = gd(lambda w, b: logistic_loss_grad(Xtr, ytr, w, b), [np.zeros(256), 0.0], lr=0.5, steps=301,
                      log_every=100)
    acc = lambda X, y: np.mean(((X @ w + b) > 0) == (y == 1))
    print(f"  训练集准确率 {acc(Xtr, ytr):.1%}，测试集准确率 {acc(Xte, yte):.1%}")
    plt.figure(figsize=(3, 3))
    m = np.abs(w).max()
    plt.imshow(w.reshape(16, 16), cmap="RdBu_r", vmin=-m, vmax=m); plt.colorbar(); plt.axis("off")
    plt.title("w (red: votes for d)"); plt.tight_layout(); plt.savefig(f"{OUT}/logistic_weights.png", dpi=120); plt.close()
    print(f"  已保存 {OUT}/logistic_weights.png（把 256 个权重排回 16×16）")


def part4_softmax():
    section("4. softmax 回归：62 类字符")
    Xtr, ytr = glyph_arrays(np.random.default_rng(0), n_per_class=60, size=16, align=True)
    Xva, yva = glyph_arrays(np.random.default_rng(1), n_per_class=20, size=16, align=True)
    Xtr, Xva = Xtr.astype(np.float64), Xva.astype(np.float64)
    print(f"  训练集 {Xtr.shape}，验证集 {Xva.shape}")
    rng = np.random.default_rng(0)
    W, b = np.zeros((256, 62)), np.zeros(62)
    print(f"  初始损失 {softmax_loss_grad(Xtr, ytr, W, b)[0]:.4f}（ln 62 = {np.log(62):.4f}）")
    t0 = time.perf_counter()
    for epoch in range(30):
        for idx in minibatches(len(Xtr), 64, rng):
            _, gW, gb = softmax_loss_grad(Xtr[idx], ytr[idx], W, b)
            W -= 0.5 * gW
            b -= 0.5 * gb
        if epoch % 5 == 4 or epoch == 0:
            Ltr = softmax_loss_grad(Xtr, ytr, W, b)[0]
            Lva = softmax_loss_grad(Xva, yva, W, b)[0]
            acc_tr = np.mean((Xtr @ W + b).argmax(1) == ytr)
            acc_va = np.mean((Xva @ W + b).argmax(1) == yva)
            print(f"    第 {epoch + 1:2d} 轮  训练损失 {Ltr:.4f} 准确率 {acc_tr:.1%}   验证损失 {Lva:.4f} 准确率 {acc_va:.1%}")
    print(f"  用时 {time.perf_counter() - t0:.1f} 秒（CPU，NumPy）")
    pred = (Xva @ W + b).argmax(1)
    conf = {}
    for t, p in zip(yva, pred):
        if t != p:
            conf[(CHARSET[t], CHARSET[p])] = conf.get((CHARSET[t], CHARSET[p]), 0) + 1
    print("  最常见的混淆:", sorted(conf.items(), key=lambda kv: -kv[1])[:8])


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    part1_linear()
    part2_lr()
    part3_logistic()
    part4_softmax()
    print("\n第 7 章的演示全部完成。")
