"""附录 F：用数值核对附录里的每一个推导。每一节打印"推导给出的值"和"直接算出的值"，两者应当一致。

    python -m labs.appendix_f.math_checks          # 全部（几秒）
    python -m labs.appendix_f.math_checks 3 7      # 只核对 F.3、F.7
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def f1():
    section("F.1 softmax + 交叉熵对 logits 的梯度 = p − y")
    z = torch.randn(5, requires_grad=True)
    y = 2
    F.cross_entropy(z[None], torch.tensor([y])).backward()
    p = torch.softmax(z.detach(), -1)
    onehot = F.one_hot(torch.tensor(y), 5)
    print("autograd：", z.grad.numpy().round(6))
    print("p − y   ：", (p - onehot).numpy().round(6))


def f2():
    section("F.2 中心差分的误差 ∝ h²，单边差分的误差 ∝ h")
    f, x, true = np.sin, 1.0, np.cos(1.0)
    for h in (1e-1, 1e-2, 1e-3):
        one = (f(x + h) - f(x)) / h - true
        two = (f(x + h) - f(x - h)) / (2 * h) - true
        print(f"h = {h:g}   单边误差 {one:+.2e}   中心误差 {two:+.2e}   中心误差 / h² = {two / h**2:+.4f}（推导：f‴(x)/6 = −cos(1)/6 = {-np.cos(x) / 6:+.4f}）")


def f3():
    section("F.3 Adam 的偏差修正：梯度恒为 g 时，修正后 m̂ = g、v̂ = g²")
    g, b1, b2 = 0.3, 0.9, 0.999
    m = v = 0.0
    for t in range(1, 6):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        print(f"t = {t}   m = {m:.4f}   m̂ = {m / (1 - b1**t):.4f}   v = {v:.6f}   v̂ = {v / (1 - b2**t):.4f}   这一步的更新 / lr = {m / (1 - b1**t) / (np.sqrt(v / (1 - b2**t)) + 1e-8):.4f}")


def f4():
    section("F.4 KL ≥ 0，H(p, q) = H(p) + KL(p‖q)")
    p = torch.softmax(torch.randn(6), -1)
    q = torch.softmax(torch.randn(6), -1)
    H = -(p * p.log()).sum()
    Hpq = -(p * q.log()).sum()
    kl = (p * (p / q).log()).sum()
    print(f"H(p) = {H:.6f}   H(p, q) = {Hpq:.6f}   KL(p‖q) = {kl:.6f}   H(p) + KL = {H + kl:.6f}")
    print(f"KL(q‖p) = {(q * (q / p).log()).sum():.6f}（不对称）")
    print(f"下界：Σ p (1 − q/p) = 1 − Σ q = {(p * (1 - q / p)).sum():.2e} ≤ KL")


def f5():
    section("F.5 初始化：Var(z) = n σ_w² E[h²]；ReLU 用 σ_w² = 2/n 才能保持方差")
    n, N = 512, 4000
    x = torch.randn(N, n)
    for s2, name in ((1 / n, "σ_w² = 1/n"), (2 / n, "σ_w² = 2/n")):
        h, out = x, []
        for layer in range(1, 11):
            z = h @ (torch.randn(n, n) * np.sqrt(s2))
            h = torch.relu(z)
            if layer in (1, 5, 10):
                out.append(f"第 {layer:2d} 层 {z.var():.4f}")
        print(f"ReLU，{name}：Var(z)  " + "   ".join(out) + f"   （推导：第 1 层 = n σ_w² = {n * s2:g}，之后每层乘 n σ_w² / 2 = {n * s2 / 2:g}）")
    h = torch.relu(x)
    W = torch.randn(n, n) * np.sqrt(2 / n)
    print(f"一层核对：n σ_w² E[h²] = {n * (2 / n) * (h**2).mean():.4f}，实测 Var(hW) = {(h @ W).var():.4f}")


def f6():
    section("F.6 注意力为什么除以 √d：q·k 的方差 = d")
    for d in (16, 64, 256):
        q, k = torch.randn(100000, d), torch.randn(100000, d)
        s = (q * k).sum(-1)
        print(f"d = {d:3d}   Var(q·k) = {s.var():7.2f}   Var(q·k/√d) = {(s / np.sqrt(d)).var():.3f}")


def f7():
    section("F.7 策略梯度：E[R ∇log π] = ∇E[R]；基线不改变期望")
    theta = torch.randn(4, requires_grad=True)
    R = torch.tensor([1.0, 0.0, 0.5, 2.0])
    pi = torch.softmax(theta, -1)
    (pi * R).sum().backward()
    exact = theta.grad.clone()
    logp = torch.log_softmax(theta.detach(), -1)
    pi = logp.exp()
    grads = torch.eye(4) - pi                     # ∇_θ log π(y) 的第 y 行 = onehot(y) − π
    rng = np.random.default_rng(0)
    ys = rng.choice(4, size=200000, p=pi.numpy())
    for b in (0.0, 1.0, 100.0):
        g = ((R[ys] - b)[:, None] * grads[ys])
        print(f"基线 b = {b:5.1f}   估计 {g.mean(0).numpy().round(4)}   每个样本的方差之和 {g.var(0).sum():9.3f}")
    print(f"精确梯度          {exact.numpy().round(4)}")


def f8():
    section("F.8 KL 的三个估计量：期望与梯度（r = π_ref / π，y ~ π）")
    theta = torch.randn(5, requires_grad=True)
    ref = torch.log_softmax(torch.randn(5), -1)
    logp = torch.log_softmax(theta, -1)
    pi = logp.exp()
    log_r = ref - logp
    k = {"k1": -log_r, "k2": log_r**2 / 2, "k3": log_r.exp() - 1 - log_r}
    kl_f = (pi * (logp - ref)).sum()                        # KL(π‖π_ref)
    kl_r = (ref.exp() * (ref - logp)).sum()                  # KL(π_ref‖π)
    print(f"KL(π‖π_ref) = {kl_f.item():.6f}   E[k1] = {(pi * k['k1']).sum().item():.6f}   E[k3] = {(pi * k['k3']).sum().item():.6f}   E[k2] = {(pi * k['k2']).sum().item():.6f}")
    g_f = torch.autograd.grad(kl_f, theta, retain_graph=True)[0]
    g_r = torch.autograd.grad(kl_r, theta, retain_graph=True)[0]
    # "把估计量放进损失"：对采样到的 y 求 ∇k(y)，y 本身不求导；期望 = Σ_y π(y) ∇k(y)
    for name in ("k1", "k2", "k3"):
        g = sum(pi[y].detach() * torch.autograd.grad(k[name][y], theta, retain_graph=True)[0] for y in range(5))
        print(f"E[∇{name}] = {g.numpy().round(5)}")
    print(f"∇KL(π‖π_ref) = {g_f.numpy().round(5)}")
    print(f"∇KL(π_ref‖π) = {g_r.numpy().round(5)}")


def f9():
    section("F.9 带 KL 约束的最优策略：π* ∝ π_ref · exp(R/β)")
    beta = 0.5
    ref = torch.softmax(torch.randn(4), -1)
    R = torch.tensor([1.0, 0.2, 0.0, 0.7])
    star = ref * (R / beta).exp()
    Z = star.sum()
    star = star / Z

    def J(pi):
        return (pi * R).sum() - beta * (pi * (pi / ref).log()).sum()

    print(f"J(π*) = {J(star):.6f}   β log Z = {beta * Z.log():.6f}")
    best = max(J(torch.softmax(torch.randn(4) * 3, -1)) for _ in range(20000))
    print(f"随机试 2 万个策略，最大的 J = {best:.6f}（不超过 J(π*)）")
    other = torch.softmax(torch.randn(4), -1)
    print(f"任取一个 π：J(π) = {J(other):.6f}，β log Z − β KL(π‖π*) = {beta * Z.log() - beta * (other * (other / star).log()).sum():.6f}")
    implied = beta * (star / ref).log()
    print(f"反推的奖励 β log(π*/π_ref) = {implied.numpy().round(4)}，减去真实奖励 = {(implied - R).numpy().round(4)}（全是同一个常数 −β log Z）")


def f10():
    section("F.10 InfoNCE 对相似度的梯度 = (p − y) / (τB)")
    tau = 0.07
    img = F.normalize(torch.randn(4, 8), dim=-1)
    txt = F.normalize(torch.randn(4, 8), dim=-1).requires_grad_(True)
    s = img @ txt.T
    s.retain_grad()
    F.cross_entropy(s / tau, torch.arange(4)).backward()
    p = torch.softmax(s.detach() / tau, -1)
    print("autograd 第 0 行：", s.grad[0].numpy().round(5))
    print("(p − y)/(τB)   ：", ((p[0] - F.one_hot(torch.tensor(0), 4)) / tau / 4).numpy().round(5))


def f11():
    section("F.11 MinHash：P(两个集合的最小哈希相同) = Jaccard；LSH 分段：1 − (1 − J^r)^b")
    rng = np.random.default_rng(0)
    A, B = set(range(0, 60)), set(range(20, 100))
    J = len(A & B) / len(A | B)
    U = 1000
    hits = 0
    trials = 20000
    for _ in range(trials):
        perm = rng.permutation(U)
        hits += min(perm[list(A)]) == min(perm[list(B)])
    print(f"Jaccard = {J:.4f}   最小哈希相同的比例 = {hits / trials:.4f}")
    r, b = 4, 16
    for s in (0.3, 0.5, 0.8, 0.9):
        sim = (rng.random((20000, b, r)) < s).all(-1).any(-1).mean()
        print(f"J = {s}   公式 1 − (1 − J^r)^b = {1 - (1 - s**r)**b:.4f}   模拟 = {sim:.4f}")


def f12():
    section("F.12 bootstrap：平均数的标准误 ≈ σ / √n")
    rng = np.random.default_rng(0)
    x = (rng.random(200) < 0.7).astype(float)          # 200 个样本，正确率约 70%
    boots = np.array([rng.choice(x, size=len(x)).mean() for _ in range(5000)])
    print(f"正确率 {x.mean():.3f}   bootstrap 标准误 {boots.std():.4f}   公式 √(p(1−p)/n) = {np.sqrt(x.mean() * (1 - x.mean()) / len(x)):.4f}")
    print(f"95% 区间（百分位法）[{np.percentile(boots, 2.5):.3f}, {np.percentile(boots, 97.5):.3f}]   正态近似 ±1.96 SE = ±{1.96 * boots.std():.3f}")


def f13():
    section("F.13 Dice 损失的梯度：前景只有 3 个 token，背景从 61 个增加到 1021 个")
    eps = 1.0
    for n in (64, 1024):
        z = torch.full((1, n), -4.6)                # 背景基本判对了：p ≈ 0.01
        z[0, :3] = 0.0                              # 前景还没学会：p = 0.5
        t = torch.zeros(1, n)
        t[0, :3] = 1
        zd = z.clone().requires_grad_(True)
        p = torch.sigmoid(zd)
        dice = (2 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)
        (1 - dice).backward()
        Nn = 2 * (p * t).sum().item() + eps
        M = (p.sum() + t.sum()).item() + eps
        s = 0.25                                    # p(1 − p)，p = 0.5
        zb = z.clone().requires_grad_(True)
        bce = F.binary_cross_entropy_with_logits(zb, t)
        bce.backward()
        print(f"共 {n:4d} 个 token   Dice 损失 {1 - dice.item():.3f}，前景 token 的梯度 {zd.grad[0, 0]:+.4f}（公式 {-(2 * M - Nn) / M**2 * s:+.4f}）"
              f"   BCE 损失 {bce.item():.4f}，前景 token 的梯度 {zb.grad[0, 0]:+.5f}（公式 (p − 1)/N = {-0.5 / n:+.5f}）")


def f15():
    section("F.15 InfoNCE 的互信息下界：ln B − L ≤ I(X; Y)")
    rng = np.random.default_rng(0)
    K = 8
    joint = np.full((K, K), 0.2 / (K * K - K))
    np.fill_diagonal(joint, 0.8 / K)                # 80% 的概率 y = x，其余均匀
    px, py = joint.sum(1), joint.sum(0)
    I = (joint * np.log(joint / np.outer(px, py))).sum()
    f_opt = np.log(joint / np.outer(px, py))         # 最好的打分函数：log p(x, y) / (p(x) p(y))
    f_bad = np.eye(K) * 1.0                          # 一个更差的打分函数
    print(f"I(X; Y) = {I:.4f}   ln K = {np.log(K):.4f}")
    flat = joint.ravel()
    for B in (2, 8, 64, 256):
        n = min(4000, 8_000_000 // (B * B))          # 批数：B 大时少抽几批，控制内存
        idx = rng.choice(K * K, size=(n, B), p=flat)  # 每批 B 对 (x, y)，按联合分布抽
        x, y = idx // K, idx % K
        for name, f in (("最优", f_opt), ("较差", f_bad)):
            s = f[x[:, :, None], y[:, None, :]]           # s[b, i, j] = f(x_i, y_j)
            logp = s - np.logaddexp.reduce(s, axis=-1, keepdims=True)
            per_batch = -np.diagonal(logp, axis1=1, axis2=2).mean(-1)
            L, se = per_batch.mean(), per_batch.std() / np.sqrt(n)
            print(f"B = {B:3d}   {name}的打分   L = {L:.4f}   ln B − L = {np.log(B) - L:.4f} ± {2 * se:.4f}（{n} 批）")


PARTS = {1: f1, 2: f2, 3: f3, 4: f4, 5: f5, 6: f6, 7: f7, 8: f8, 9: f9, 10: f10, 11: f11, 12: f12, 13: f13, 15: f15}

if __name__ == "__main__":
    want = [int(a) for a in sys.argv[1:] if a.isdigit()] or list(PARTS)
    for i in want:
        PARTS[i]()
