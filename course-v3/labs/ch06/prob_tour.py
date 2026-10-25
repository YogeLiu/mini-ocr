"""第 6 章：概率、交叉熵与最大似然。逐段运行，对照正文阅读。

    python -m labs.ch06.prob_tour
"""
import collections

import numpy as np

from miniocr.data.latex import gen_formula, to_latex
from miniocr.prob import cross_entropy, entropy, kl_divergence, log_softmax, softmax


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def part1_mean_var():
    section("1. 求和、均值、方差：掷骰子")
    x = np.arange(1, 7)
    p = np.full(6, 1 / 6)
    mean = (p * x).sum()
    var = (p * (x - mean) ** 2).sum()
    print(f"  理论：E[X] = Σ p·x = {mean:.4f}   Var[X] = Σ p·(x-E)² = {var:.4f}   标准差 {np.sqrt(var):.4f}")
    rng = np.random.default_rng(0)
    for n in (10, 100, 10000, 1000000):
        s = rng.integers(1, 7, size=n)
        print(f"  掷 {n:>7} 次：样本均值 {s.mean():.4f}   样本方差 {s.var():.4f}")


def part2_char_dist():
    section("2. 从计数到概率：公式里的字符分布")
    rng = np.random.default_rng(0)
    text = "".join(to_latex(gen_formula(rng, 3)) for _ in range(2000))
    c = collections.Counter(text)
    chars = sorted(c)
    p = np.array([c[ch] for ch in chars], dtype=float)
    p /= p.sum()
    print(f"  {len(text)} 个字符，{len(chars)} 种；概率最高的 5 个:",
          [(chars[i], round(float(p[i]), 4)) for i in np.argsort(-p)[:5]])
    H = entropy(p)
    print(f"  熵 H = {H:.4f} nat = {H / np.log(2):.4f} bit；均匀分布（{len(chars)} 种）的熵 = {np.log(len(chars)):.4f} nat")
    return chars, p


def part3_softmax():
    section("3. softmax：分数 -> 概率")
    z = np.array([2.0, 1.0, 0.1])
    print("  logits", z.tolist(), "-> softmax", np.round(softmax(z), 4).tolist())
    print("  所有 logits 加 100:", np.round(softmax(z + 100), 4).tolist(), " ← 不变")
    print("  所有 logits 乘 2  :", np.round(softmax(z * 2), 4).tolist(), " ← 更\"尖\"")
    for T in (0.5, 1.0, 2.0, 10.0):
        print(f"  温度 T = {T:<4}: softmax(z / T) = {np.round(softmax(z / T), 4).tolist()}")
    big = np.array([1000.0, 1001.0, 1002.0])
    with np.errstate(over="ignore", invalid="ignore"):
        naive = np.exp(big) / np.exp(big).sum()
    print("  logits [1000, 1001, 1002]：直接算", naive.tolist(), "  减最大值后", np.round(softmax(big), 4).tolist())
    tiny = np.array([0.0, -800.0])
    with np.errstate(divide="ignore"):
        print("  log(softmax([0, -800])) =", np.log(softmax(tiny)).tolist(), "  log_softmax =", log_softmax(tiny).tolist())


def part4_ce_values():
    section("4. 交叉熵：-log(给正确答案的概率)")
    for q in (0.99, 0.9, 0.5, 0.1, 0.01, 0.001):
        print(f"  正确类的概率 {q:<6} -> 损失 {-np.log(q):.4f}")
    for K in (2, 10, 62, 100, 151936):
        print(f"  {K:>6} 类均匀猜测的损失 ln K = {np.log(K):.4f}")
    rng = np.random.default_rng(0)
    z = rng.normal(0, 0.01, size=(1000, 62))
    y = rng.integers(0, 62, 1000)
    print(f"  随机初始化（logits 很小）的模型，在 1000 个样本上的交叉熵 = {cross_entropy(z, y):.4f}")
    z_bad = rng.normal(0, 10, size=(1000, 62))
    print(f"  logits 很大的随机模型（\"自信地乱猜\"）= {cross_entropy(z_bad, y):.4f}")


def part5_mle():
    section("5. 最大似然：抛 10 次硬币，7 次正面，p 应该是多少？")
    heads, n = 7, 10
    ps = np.linspace(0.01, 0.99, 99)
    lik = ps ** heads * (1 - ps) ** (n - heads)
    ll = heads * np.log(ps) + (n - heads) * np.log(1 - ps)
    print(f"  似然最大的 p = {ps[lik.argmax()]:.2f}，对数似然最大的 p = {ps[ll.argmax()]:.2f}（= 7/10）")
    for p in (0.5, 0.6, 0.7, 0.8):
        print(f"  p = {p}: 似然 {p ** 7 * (1 - p) ** 3:.6f}   对数似然 {7 * np.log(p) + 3 * np.log(1 - p):.4f}")
    rng = np.random.default_rng(0)
    probs = rng.uniform(0.1, 0.9, size=2000)
    print(f"  2000 个 0.1–0.9 之间的概率连乘 = {np.prod(probs)}（下溢成 0）；它们的 log 之和 = {np.log(probs).sum():.1f}")


def part6_kl():
    section("6. KL 散度与交叉熵 = 熵 + KL")
    p = np.array([0.7, 0.2, 0.1])
    for name, q in (("q1 接近 p", np.array([0.6, 0.25, 0.15])), ("q2 均匀", np.full(3, 1 / 3)),
                    ("q3 很偏", np.array([0.1, 0.1, 0.8]))):
        ce = -(p * np.log(q)).sum()
        print(f"  {name:8s} KL(p||q) = {kl_divergence(p, q):.4f}  KL(q||p) = {kl_divergence(q, p):.4f}"
              f"   交叉熵 H(p,q) = {ce:.4f} = H(p) {entropy(p):.4f} + KL {kl_divergence(p, q):.4f}")


def part7_unigram(chars, p_train):
    section("7. 一个最简单的\"语言模型\"：用训练集的字符频率预测测试集")
    rng = np.random.default_rng(99)
    test = "".join(to_latex(gen_formula(rng, 3)) for _ in range(500))
    idx = {c: i for i, c in enumerate(chars)}
    ids = np.array([idx[c] for c in test if c in idx])
    K = len(chars)
    uni = np.log(p_train)[ids]
    print(f"  测试集 {len(ids)} 个字符")
    print(f"  均匀模型   的平均 -log p = {np.log(K):.4f} nat   困惑度 e^H = {K:.1f}")
    print(f"  字符频率模型的平均 -log p = {-uni.mean():.4f} nat   困惑度 = {np.exp(-uni.mean()):.1f}")
    for alpha in (0.0, 0.5, 2.0):
        q = (p_train + alpha / K) / (1 + alpha)
        print(f"  把频率与均匀分布混合（alpha={alpha}）: {-np.log(q)[ids].mean():.4f} nat")


if __name__ == "__main__":
    part1_mean_var()
    chars, p = part2_char_dist()
    part3_softmax()
    part4_ce_values()
    part5_mle()
    part6_kl()
    part7_unigram(chars, p)
    print("\n第 6 章的演示全部完成。")
