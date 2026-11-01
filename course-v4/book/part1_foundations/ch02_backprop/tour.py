"""第 2 章　导数、计算图与反向传播：配套实验脚本。正文里引用的每一个数字都来自这里。

    python -m book.part1_foundations.ch02_backprop.tour            # 全部 9 个部分（CPU 上约 1 分钟）
    python -m book.part1_foundations.ch02_backprop.tour 4 6        # 只运行第 4、6 部分
    python -m book.part1_foundations.ch02_backprop.tour 8 --device mps   # 第 8 部分的 PyTorch 训练放到 Mac 上

部分与正文小节的对应：
    1  一个旋钮：试探与导数；数值求导的代价（2.1）   6  数值梯度检查：步长表、相对误差、折点（2.8）
    2  计算图与链式法则（2.3、2.4）                 7  PyTorch autograd 的几个开关（2.9）
    3  同一个变量用了两次：梯度相加（2.5）           8  忘记清零梯度（2.9.6）
    4  手算两层网络：Value 与 PyTorch 对拍（2.6）    9  用 Value 训练 moons 分类器（2.10）
    5  Value 引擎：拓扑排序、顺序错误、递归深度（2.7）

第 1–7 部分只用 Python 的 float 和 CPU 上的 PyTorch；第 9 部分用纯 Python 的 Value，和设备无关；
只有第 8 部分的 PyTorch 训练会用到 --device。
"""
import math
import os
import random
import statistics
import sys
import time

# 除非明确传入 --device cuda，否则对 PyTorch 隐藏 GPU。
if "cuda" not in sys.argv:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from miniocr.autograd import MLP, Value, graph_nodes, numerical_grad, rel_error, zero_grad
from miniocr.device import device_from_argv

OUT = "outputs/book/ch02"
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(min(8, os.cpu_count() or 1))
DEVICE = device_from_argv(default="cpu")   # 只有第 8 部分用它


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ---------------------------------------------------------------------- 2.6 节的两层小网络（第 4 部分和测试共用）
# 输入 x 有 2 维；隐藏层 2 个 relu 神经元；输出层 1 个线性神经元；损失 L = (ŷ − y)²。
NET = dict(x1=2.0, x2=1.0,
           w11=1.0, w12=-1.0, b1=0.5,      # 隐藏神经元 1：z1 = w11·x1 + w12·x2 + b1
           w21=0.5, w22=1.0, b2=-1.0,      # 隐藏神经元 2：z2 = w21·x1 + w22·x2 + b2
           v1=1.0, v2=-1.0, c=0.5,         # 输出：ŷ = v1·a1 + v2·a2 + c
           y=2.0)
PARAM_NAMES = ["w11", "w12", "b1", "w21", "w22", "b2", "v1", "v2", "c"]


def two_layer_values(net=NET):
    """用 Value 搭 2.6 节的网络。返回一个字典：所有输入、参数、中间量和损失（都是 Value）。"""
    v = {k: Value(val, label=k) for k, val in net.items()}
    v["z1"] = v["w11"] * v["x1"] + v["w12"] * v["x2"] + v["b1"]
    v["z2"] = v["w21"] * v["x1"] + v["w22"] * v["x2"] + v["b2"]
    v["a1"], v["a2"] = v["z1"].relu(), v["z2"].relu()
    v["yhat"] = v["v1"] * v["a1"] + v["v2"] * v["a2"] + v["c"]
    v["e"] = v["yhat"] - v["y"]
    v["L"] = v["e"] ** 2
    return v


def two_layer_torch(net=NET):
    """同一个网络，用 PyTorch 的标量张量搭。参数 requires_grad=True，x 也设为 True 以便看 ∂L/∂x。"""
    t = {k: torch.tensor(val, dtype=torch.float64, requires_grad=(k != "y")) for k, val in net.items()}
    z1 = t["w11"] * t["x1"] + t["w12"] * t["x2"] + t["b1"]
    z2 = t["w21"] * t["x1"] + t["w22"] * t["x2"] + t["b2"]
    a1, a2 = torch.relu(z1), torch.relu(z2)
    yhat = t["v1"] * a1 + t["v2"] * a2 + t["c"]
    L = (yhat - t["y"]) ** 2
    return t, L


def two_layer_loss(net):
    """纯 float 版本，给数值梯度用。"""
    z1 = net["w11"] * net["x1"] + net["w12"] * net["x2"] + net["b1"]
    z2 = net["w21"] * net["x1"] + net["w22"] * net["x2"] + net["b2"]
    yhat = net["v1"] * max(z1, 0.0) + net["v2"] * max(z2, 0.0) + net["c"]
    return (yhat - net["y"]) ** 2


# ---------------------------------------------------------------------- moons 数据（第 8、9 部分和测试共用）
def make_moons(n, noise=0.1, seed=0):
    """两个交错的半月形。偶数下标是上面的月牙（标签 +1），奇数下标是下面的月牙（标签 −1）。
    最后把坐标平移 (−0.5, −0.25)，让数据大致以原点为中心。返回 (点的列表, 标签列表)。"""
    rng = random.Random(seed)
    X, Y = [], []
    for i in range(n):
        t = rng.uniform(0.0, math.pi)
        if i % 2 == 0:
            px, py, lab = math.cos(t), math.sin(t), 1
        else:
            px, py, lab = 1.0 - math.cos(t), 0.5 - math.sin(t), -1
        X.append([px + rng.gauss(0.0, noise) - 0.5, py + rng.gauss(0.0, noise) - 0.25])
        Y.append(lab)
    return X, Y


def hinge_loss(model, X, Y):
    """平均合页损失 mean(max(0, 1 − y·s))。返回 (损失 Value, 打分列表)。第 3 章会换成交叉熵。"""
    scores = [model(x) for x in X]
    losses = [(1 - y * s).relu() for s, y in zip(scores, Y)]
    return sum(losses) * (1.0 / len(X)), scores


def sq_loss(model, X, Y):
    """平均平方损失 mean((s − y)²)。处处光滑，适合做梯度检查。"""
    return sum((model(x) - y) ** 2 for x, y in zip(X, Y)) * (1.0 / len(X))


def accuracy(scores, Y):
    return sum((s.data > 0) == (y > 0) for s, y in zip(scores, Y)) / len(Y)


# ---------------------------------------------------------------------- 1
def part1_knob():
    section("1. 一个旋钮（2.1）：试探着拧一下 vs 直接算导数；数值求导的代价")
    x, y, w = 2.0, 3.0, 0.5
    L = lambda w: (w * x - y) ** 2
    print(f"  模型 ŷ = w·x，一个样本 x = {x:g}，y = {y:g}；损失 L(w) = (w·x − y)²；当前 w = {w:g}，L = {L(w):g}")
    for dw in (0.1, 0.01, 0.001):
        print(f"    w 加 {dw:<6g} → L = {L(w + dw):.6f}，变化 {L(w + dw) - L(w):+.6f}，变化 / {dw:g} = {(L(w + dw) - L(w)) / dw:+.4f}")
    print(f"  导数 dL/dw = 2(w·x − y)·x = {2 * (w * x - y) * x:g}：负号说明 w 应该调大")
    print()
    print("  数值求导的代价：中心差分每个参数要做 2 次前向；反向传播一次前向 + 一次反向就得到全部梯度")
    X, Y = make_moons(4, seed=1)
    for sizes in ([2, 8, 8, 1], [2, 16, 16, 1]):
        m = MLP(sizes, seed=0)
        P = m.parameters()

        def f(vals):
            for p, v in zip(P, vals):
                p.data = v
            return sq_loss(m, X, Y).data

        vals = [p.data for p in P]
        t0 = time.perf_counter()
        num = numerical_grad(f, vals)
        t_num = time.perf_counter() - t0
        f(vals)
        t0 = time.perf_counter()
        loss = sq_loss(m, X, Y)
        zero_grad(P)
        loss.backward()
        t_bp = time.perf_counter() - t0
        ana = [p.grad for p in P]
        print(f"    MLP {sizes}：参数 {len(P):4d} 个，数值求导 {2 * len(P):4d} 次前向 {t_num * 1000:8.1f} ms；"
              f"反向传播 {t_bp * 1000:6.1f} ms；慢 {t_num / t_bp:5.0f} 倍；两者最大相对误差 {rel_error(ana, num):.1e}")
    print("  （本项目观察，CPU；计时受机器负载影响。参数越多，差距越大：数值求导的次数和参数个数成正比）")


# ---------------------------------------------------------------------- 2
def part2_graph():
    section("2. 计算图与链式法则（2.3、2.4）")
    print("  一条链：y = (3x + 1)²，在 x = 1 处")
    x = 1.0
    u = 3 * x + 1
    yv = u ** 2
    print(f"    前向：u = 3x + 1 = {u:g}，y = u² = {yv:g}")
    print(f"    局部导数：du/dx = 3，dy/du = 2u = {2 * u:g}；链式法则 dy/dx = {2 * u:g} × 3 = {2 * u * 3:g}")
    x2 = 1.001
    y2 = (3 * x2 + 1) ** 2
    print(f"    验证：x 加 0.001 → y = {y2:.6f}，变化 {y2 - yv:.6f} ≈ {2 * u * 3:g} × 0.001 = {2 * u * 3 * 0.001:g}")
    print()
    print("  一张图：L = (a·b + c)²，a = 2，b = −3，c = 10")
    a, b, c = Value(2.0, label="a"), Value(-3.0, label="b"), Value(10.0, label="c")
    d = a * b; d.label = "d"
    e = d + c; e.label = "e"
    L = e ** 2; L.label = "L"
    print(f"    前向：d = a·b = {d.data:g}，e = d + c = {e.data:g}，L = e² = {L.data:g}")
    L.backward()
    print(f"    反向：∂L/∂L = 1 → ∂L/∂e = 2e = {e.grad:g} → 加法原样分发：∂L/∂d = {d.grad:g}，∂L/∂c = {c.grad:g}")
    print(f"          → 乘法交换：∂L/∂a = ∂L/∂d · b = {a.grad:g}，∂L/∂b = ∂L/∂d · a = {b.grad:g}")
    ta, tb, tc = (torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (2.0, -3.0, 10.0))
    ((ta * tb + tc) ** 2).backward()
    print(f"    PyTorch：a.grad = {ta.grad.item():g}，b.grad = {tb.grad.item():g}，c.grad = {tc.grad.item():g}")
    print()
    print("  几种节点怎样处理流回来的梯度（上游梯度记为 g）：")
    print("    加法 z = x + y：x、y 都拿到 g（分发器）")
    print("    乘法 z = x·y：x 拿到 g·y，y 拿到 g·x（交换器）")
    print("    relu：输入 > 0 时原样通过，否则变成 0（开关）")
    for z in (0.0, 1.0, 2.0, 3.0):
        t = math.tanh(z)
        print(f"    tanh 在 z = {z:g}：tanh = {t:.4f}，局部导数 1 − tanh² = {1 - t * t:.4f}")
    print("    tanh 的输入绝对值越大，局部导数越接近 0（阀门快关死了，梯度很难流过去）")


# ---------------------------------------------------------------------- 3
def _buggy_ops():
    """返回 (加法, 乘法) 两个"写成 = 而不是 +="的错误版本，用来临时替换 Value 的方法。"""
    def add(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad = out.grad
            other.grad = out.grad
        out._backward = _backward
        return out

    def mul(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad
        out._backward = _backward
        return out
    return add, mul


def part3_fork():
    section("3. 同一个变量用了两次（2.5）：梯度相加")

    def run():
        x = Value(3.0)
        f = x * x
        f.backward()
        x2, y2 = Value(2.0), Value(5.0)
        g = x2 * y2 + x2
        g.backward()
        return x.grad, x2.grad

    good = run()
    orig = Value.__add__, Value.__mul__
    Value.__add__, Value.__mul__ = _buggy_ops()
    try:
        bad = run()
    finally:
        Value.__add__, Value.__mul__ = orig
    print(f"  f = x·x，x = 3：正确的 df/dx = 2x = 6；用 +=：{good[0]:g}；写成 =：{bad[0]:g}")
    print(f"  g = x·y + x，x = 2，y = 5：正确的 dg/dx = y + 1 = 6；用 +=：{good[1]:g}；写成 =：{bad[1]:g}")
    tx = torch.tensor(3.0, requires_grad=True)
    (tx * tx).backward()
    print(f"  PyTorch：x·x 在 x = 3 处 x.grad = {tx.grad.item():g}")
    print("  直接验证：f(3.001) − f(3) = "
          f"{3.001 ** 2 - 9:.6f} ≈ 6 × 0.001。两条路径各贡献 x = 3，加起来才是 6")


# ---------------------------------------------------------------------- 4
def part4_two_layer():
    section("4. 手算两层网络（2.6）：Value、PyTorch、数值梯度三方对拍")
    v = two_layer_values()
    print("  输入、参数和目标：" + "，".join(f"{k} = {NET[k]:g}" for k in NET))
    print("  前向：")
    for k in ("z1", "a1", "z2", "a2", "yhat", "e", "L"):
        print(f"    {k:5s}= {v[k].data:g}")
    v["L"].backward()
    print("  反向（∂L/∂·）：")
    for k in ("L", "e", "yhat", "c", "v1", "v2", "a1", "a2", "z1", "z2",
              "w11", "w12", "b1", "w21", "w22", "b2", "x1", "x2"):
        print(f"    ∂L/∂{k:5s}= {v[k].grad:g}")
    t, L = two_layer_torch()
    L.backward()
    diff = max(abs(v[k].grad - t[k].grad.item()) for k in PARAM_NAMES + ["x1", "x2"])
    print(f"  PyTorch：L = {L.item():g}；9 个参数和 x1、x2 的梯度与 Value 的最大差 {diff:.1e}")
    keys = PARAM_NAMES + ["x1", "x2"]

    def f(vals):
        return two_layer_loss({**NET, **dict(zip(keys, vals))})

    num = numerical_grad(f, [NET[k] for k in keys])
    print(f"  数值梯度（中心差分，h = 1e-5）与 Value 的最大相对误差 {rel_error([v[k].grad for k in keys], num):.1e}")
    print("  x1 被两个隐藏神经元同时用到：∂L/∂x1 = ∂L/∂z1·w11 + ∂L/∂z2·w21 = "
          f"({v['z1'].grad:g})×{NET['w11']:g} + ({v['z2'].grad:g})×{NET['w21']:g} = {v['x1'].grad:g}")
    print()
    for lr in (0.02, 0.1):
        new = dict(NET)
        for k in PARAM_NAMES:
            new[k] = NET[k] - lr * v[k].grad
        nv = two_layer_values(new)
        print(f"  沿负梯度走一步，学习率 {lr:g}：新参数 " + "，".join(f"{k} = {new[k]:.4g}" for k in PARAM_NAMES))
        print(f"    新的前向：z1 = {nv['z1'].data:.4g}，z2 = {nv['z2'].data:.4g}，ŷ = {nv['yhat'].data:.4g}，"
              f"L 从 {v['L'].data:g} 变成 {nv['L'].data:.4g}")
    gsq = sum(v[k].grad ** 2 for k in PARAM_NAMES)
    print(f"  一阶近似预测的下降量：学习率 × Σ(梯度²) = 0.02 × {gsq:g} = {0.02 * gsq:.4g}")
    print()
    dead = dict(NET, w21=-1.0, y=3.0)
    dv = two_layer_values(dead)
    dv["L"].backward()
    print("  变体：w21 = −1，y = 3。隐藏神经元 2 的输入为负，relu 输出 0：")
    print(f"    z2 = {dv['z2'].data:g}，a2 = {dv['a2'].data:g}，ŷ = {dv['yhat'].data:g}，L = {dv['L'].data:g}")
    print("    梯度：" + "，".join(f"∂L/∂{k} = {dv[k].grad:g}" for k in ("v2", "a2", "z2", "w21", "w22", "b2")))


# ---------------------------------------------------------------------- 5
def part5_engine():
    section("5. Value 引擎（2.7）：拓扑排序、顺序错误、重复调用 backward、递归深度")
    a, b, c = Value(2.0, label="a"), Value(-3.0, label="b"), Value(10.0, label="c")
    d = a * b; d.label = "d"
    e = d + c; e.label = "e"
    L = e ** 2; L.label = "L"
    topo, seen = [], set()

    def build(v):
        if v not in seen:
            seen.add(v)
            for p in v._prev:
                build(p)
            topo.append(v)
    build(L)
    print("  L = (a·b + c)² 的拓扑顺序（每个节点都排在它的输入之后）：" + " → ".join(v.label for v in topo))
    print("  反向传播按逆序调用 _backward：" + " → ".join(v.label for v in reversed(topo)))
    # 错误顺序：按拓扑顺序（正序）调用
    for v in topo:
        v.grad = 0.0
    L.grad = 1.0
    for v in topo:
        v._backward()
    print(f"  如果按正序调用：a.grad = {a.grad:g}，b.grad = {b.grad:g}，c.grad = {c.grad:g}（正确是 −24、16、8）")
    print("    原因：轮到 a、b、c 时，它们的下游还没把梯度传下来，读到的上游梯度是 0")
    print()
    v = two_layer_values()
    nodes = graph_nodes(v["L"])
    leaves = [n for n in nodes if not n._prev]
    print(f"  2.6 节两层网络的计算图：{len(nodes)} 个节点，其中叶子 {len(leaves)} 个"
          f"（2 个输入、9 个参数、目标 y，以及减法里生成的常数 −1）")
    v["L"].backward()
    g1 = v["w11"].grad
    v["L"].backward()
    print(f"  在同一张图上连续调用两次 backward：w11.grad 从 {g1:g} 变成 {v['w11'].grad:g}（叶子节点累加，中间节点每次重算）")
    print()
    x = Value(1.0)
    s = x
    for _ in range(5000):
        s = s + 1.0
    try:
        s.backward()
        print("  5000 次加法串成的链：backward 成功")
    except RecursionError as err:
        print(f"  5000 次加法串成的链：backward 报 RecursionError（{err}）。Python 默认递归深度上限 {sys.getrecursionlimit()}")


# ---------------------------------------------------------------------- 6
def part6_gradcheck():
    section("6. 数值梯度检查（2.8）：步长表、相对误差、折点")
    w0 = 0.7

    def f64(w):
        return (math.tanh(1.5 * w) - 0.3) ** 2

    t = math.tanh(1.5 * w0)
    exact = 2 * (t - 0.3) * (1 - t * t) * 1.5
    print(f"  函数 f(w) = (tanh(1.5w) − 0.3)²，在 w = {w0:g} 处；解析导数 = {exact:.10f}")
    w32 = torch.tensor(w0, dtype=torch.float32)

    def f32(w):
        return ((torch.tanh(1.5 * w) - 0.3) ** 2)

    print(f"  {'步长 h':>8s} | {'前向差分 float64':>16s} | {'中心差分 float64':>16s} | {'中心差分 float32':>16s}")
    for k in range(1, 13):
        h = 10.0 ** -k
        fwd = (f64(w0 + h) - f64(w0)) / h
        cen = (f64(w0 + h) - f64(w0 - h)) / (2 * h)
        h32 = torch.tensor(h, dtype=torch.float32)
        cen32 = ((f32(w32 + h32) - f32(w32 - h32)) / (2 * h32)).item()
        print(f"  {h:8.0e} | {rel_error(fwd, exact):16.1e} | {rel_error(cen, exact):16.1e} | {rel_error(cen32, exact):16.1e}")
    eps64, eps32 = sys.float_info.epsilon, torch.finfo(torch.float32).eps
    print(f"  机器精度：float64 {eps64:.1e}，float32 {eps32:.1e}")
    print(f"  理论上的最佳步长：前向差分约 √ε（float64 {eps64 ** 0.5:.0e}），"
          f"中心差分约 ε^(1/3)（float64 {eps64 ** (1 / 3):.0e}，float32 {eps32 ** (1 / 3):.0e}）")
    print()
    X, Y = make_moons(5, seed=2)
    m = MLP([2, 4, 4, 1], seed=1)
    P = m.parameters()

    def loss_value():
        return sq_loss(m, X, Y)

    def f(vals):
        for p, v in zip(P, vals):
            p.data = v
        return loss_value().data

    vals = [p.data for p in P]
    num = numerical_grad(f, vals)
    f(vals)

    def analytic():
        loss = loss_value()
        zero_grad(P)
        loss.backward()
        return [p.grad for p in P]

    errs = [rel_error(a, n) for a, n in zip(analytic(), num)]
    print(f"  MLP [2, 4, 4, 1]（tanh，平方损失，{len(P)} 个参数，5 个样本）：")
    print(f"    正确实现：相对误差 最大 {max(errs):.1e}，中位数 {statistics.median(errs):.1e}")
    orig_tanh = Value.tanh

    def bad_tanh(self):                       # 错误：导数写成 1 − x²（用了输入），应该是 1 − tanh(x)²
        out = Value(math.tanh(self.data), (self,), "tanh")

        def _backward():
            self.grad += (1.0 - self.data ** 2) * out.grad
        out._backward = _backward
        return out
    Value.tanh = bad_tanh
    try:
        errs_b = [rel_error(a, n) for a, n in zip(analytic(), num)]
    finally:
        Value.tanh = orig_tanh
    print(f"    tanh 导数写成 1 − x²：相对误差 最大 {max(errs_b):.1e}，中位数 {statistics.median(errs_b):.1e}")
    orig_pow = Value.__pow__

    def bad_pow(self, k):                     # 错误：忘了乘上游梯度 out.grad
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1)
        out._backward = _backward
        return out
    Value.__pow__ = bad_pow
    try:
        errs_p = [rel_error(a, n) for a, n in zip(analytic(), num)]
    finally:
        Value.__pow__ = orig_pow
    print(f"    幂的反向忘了乘 out.grad：相对误差 最大 {max(errs_p):.1e}，中位数 {statistics.median(errs_p):.1e}")
    print()
    for x0 in (2e-6, 0.5):
        a = Value(x0)
        a.relu().backward()
        n = numerical_grad(lambda v: max(v[0], 0.0), [x0])[0]
        print(f"  relu 在 x = {x0:g}：解析导数 {a.grad:g}，数值导数（h = 1e-5）{n:.4f}，相对误差 {rel_error(a.grad, n):.1e}")
    print("  x 离折点 0 比 h 还近时，x + h 和 x − h 落在折点两侧，数值导数既不是 0 也不是 1")
    print()
    print(f"  两个梯度都接近 0 时：解析 0、数值 1e-11，相对误差 {rel_error(0.0, 1e-11):.1e}；"
          f"解析 1e-9、数值 1.1e-9，相对误差 {rel_error(1e-9, 1.1e-9):.1e}——这时要改看绝对误差")


# ---------------------------------------------------------------------- 7
def part7_torch_flags():
    section("7. PyTorch autograd 的几个开关（2.9）")
    x = torch.tensor(3.0, requires_grad=True)
    w = torch.tensor(2.0)
    y = w * x
    z = y ** 2
    print(f"  x requires_grad={x.requires_grad}，w requires_grad={w.requires_grad}")
    print(f"  y = w·x：requires_grad={y.requires_grad}，grad_fn={type(y.grad_fn).__name__}，is_leaf={y.is_leaf}")
    print(f"  z = y²：grad_fn={type(z.grad_fn).__name__}；x.is_leaf={x.is_leaf}，x.grad_fn={x.grad_fn}")
    print(f"  backward 之前：x.grad = {x.grad}")
    z.backward()
    print(f"  z.backward() 之后：x.grad = {x.grad.item():g}（dz/dx = 2·w·x·w = 2·{w.item():g}²·{x.item():g} = {2 * 4 * 3:g}）；w.grad = {w.grad}")
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yg = y.grad
    print(f"  中间结果 y.grad = {yg}（非叶子节点默认不保存梯度；有警告：{bool(caught)}）")
    try:
        z.backward()
    except RuntimeError as err:
        msg = str(err).split(".")[0]
        print(f"  再调用一次 z.backward()：RuntimeError：{msg}.")
    x = torch.tensor(3.0, requires_grad=True)
    z = (2 * x) ** 2
    z.backward(retain_graph=True)
    g1 = x.grad.item()
    z.backward()
    print(f"  加 retain_graph=True 后连续两次 backward：x.grad 从 {g1:g} 变成 {x.grad.item():g}（累加）")
    x.grad = None
    print(f"  清零：x.grad = None 之后 x.grad = {x.grad}")
    v = torch.tensor([1.0, 2.0], requires_grad=True)
    try:
        (v * 2).backward()
    except RuntimeError as err:
        print(f"  对非标量调用 backward()：RuntimeError：{err}")
    c = torch.tensor(1.0)
    try:
        (c * 2).backward()
    except RuntimeError as err:
        print(f"  对不需要梯度的张量调用 backward()：RuntimeError：{err}")
    x = torch.tensor(3.0, requires_grad=True)
    with torch.no_grad():
        q = x * 2
    print(f"  torch.no_grad() 里算出的 q：requires_grad={q.requires_grad}；x.detach() 的 requires_grad={x.detach().requires_grad}")


# ---------------------------------------------------------------------- 8
def part8_forget_zero():
    section("8. 忘记清零梯度（2.9.6）")
    w = torch.tensor(0.5, requires_grad=True)
    print("  同一个样本（x = 2，y = 3，L = (w·x − y)²），w 固定不更新，连续 4 次 backward，不清零：")
    for i in range(4):
        L = (w * 2.0 - 3.0) ** 2
        L.backward()
        print(f"    第 {i + 1} 次之后 w.grad = {w.grad.item():g}")
    print("  每次的真实梯度都是 −8，.grad 里却是它们的和")
    print()
    torch.manual_seed(0)
    X, Y = make_moons(100, seed=0)
    Xt = torch.tensor(X, device=DEVICE)
    Yt = torch.tensor(Y, dtype=torch.float32, device=DEVICE)

    def net():
        g = torch.Generator().manual_seed(0)
        sizes = [2, 8, 8, 1]
        ps = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            s = 1.0 / math.sqrt(a)
            ps.append(((torch.rand(a, b, generator=g) * 2 - 1) * s).to(DEVICE).requires_grad_())
            ps.append(torch.zeros(b, device=DEVICE, requires_grad=True))
        return ps

    def forward(ps):
        h = Xt
        for i in range(0, len(ps) - 2, 2):
            h = torch.tanh(h @ ps[i] + ps[i + 1])
        return (h @ ps[-2] + ps[-1]).squeeze(-1)

    print(f"  moons 100 个点，MLP [2, 8, 8, 1]（tanh），合页损失，SGD 学习率 0.5，设备 {DEVICE}：")
    logs = {}
    for zero in (True, False):
        ps = net()
        hist = []
        t0 = time.perf_counter()
        for step in range(201):
            s = forward(ps)
            loss = torch.relu(1 - Yt * s).mean()
            if zero:
                for p in ps:
                    p.grad = None
            loss.backward()
            gnorm = math.sqrt(sum((p.grad ** 2).sum().item() for p in ps))
            acc = ((s > 0) == (Yt > 0)).float().mean().item()
            if step in (0, 10, 50, 100, 200):
                hist.append((step, loss.item(), acc, gnorm))
            with torch.no_grad():
                for p in ps:
                    p -= 0.5 * p.grad
        logs[zero] = (hist, time.perf_counter() - t0)
    for zero, (hist, sec) in logs.items():
        print(f"    {'每步清零' if zero else '从不清零'}（201 步用时 {sec:.3f} 秒）：")
        for step, loss, acc, g in hist:
            print(f"      第 {step:3d} 步  损失 {loss:.4f}  准确率 {acc:5.1%}  .grad 的范数 {g:9.3f}")
    print("  （本项目观察。从不清零时 .grad 是历史梯度的累加，相当于带着越来越大的\"惯性\"乱冲）")


# ---------------------------------------------------------------------- 9
def part9_moons():
    section("9. 用 Value 训练 moons 分类器（2.10）")
    X, Y = make_moons(100, seed=0)
    Xva, Yva = make_moons(200, seed=2)     # 验证集：实验四调学习率、平移等设置时用它比较
    Xte, Yte = make_moons(200, seed=1)     # 测试集：设置定下来以后只评一次
    m = MLP([2, 8, 8, 1], seed=0)
    P = m.parameters()
    print(f"  训练集 100 个点，验证集 200 个点，测试集 200 个点（三个不同的随机种子）；"
          f"MLP [2, 8, 8, 1]，tanh，参数 {len(P)} 个")
    print("  全批量梯度下降，学习率 1.0，合页损失 mean(max(0, 1 − y·s))")
    loss0, _ = hinge_loss(m, X, Y)
    print(f"  第 1 步计算图：{len(graph_nodes(loss0))} 个节点")
    # 和 PyTorch 对拍第 1 步的梯度
    zero_grad(P)
    loss0.backward()
    Ws = []
    for layer in m.layers:
        Ws.append(torch.tensor([[n.w[i].data for n in layer.neurons] for i in range(len(layer.neurons[0].w))],
                               dtype=torch.float64, requires_grad=True))
        Ws.append(torch.tensor([n.b.data for n in layer.neurons], dtype=torch.float64, requires_grad=True))
    h = torch.tensor(X, dtype=torch.float64)
    for i in range(0, len(Ws) - 2, 2):
        h = torch.tanh(h @ Ws[i] + Ws[i + 1])
    s = (h @ Ws[-2] + Ws[-1]).squeeze(-1)
    lt = torch.relu(1 - torch.tensor(Y, dtype=torch.float64) * s).mean()
    lt.backward()
    tg = []
    for li, layer in enumerate(m.layers):
        W, B = Ws[2 * li], Ws[2 * li + 1]
        for j, n in enumerate(layer.neurons):
            tg += [W.grad[i, j].item() for i in range(len(n.w))] + [B.grad[j].item()]
    print(f"  第 1 步：Value 损失 {loss0.data:.6f}，PyTorch 损失 {lt.item():.6f}；"
          f"{len(P)} 个梯度最大差 {max(abs(p.grad - g) for p, g in zip(P, tg)):.1e}")
    t0 = time.perf_counter()
    curve = []
    for step in range(101):
        loss, scores = hinge_loss(m, X, Y)
        zero_grad(P)
        loss.backward()
        for p in P:
            p.data -= 1.0 * p.grad
        curve.append(loss.data)
        if step % 10 == 0:
            print(f"    第 {step:3d} 步  损失 {loss.data:.4f}  训练准确率 {accuracy(scores, Y):5.1%}"
                  f"  已用 {time.perf_counter() - t0:5.1f} 秒")
    elapsed = time.perf_counter() - t0
    va_scores = [m(x) for x in Xva]
    te_scores = [m(x) for x in Xte]
    print(f"  训练 101 步共 {elapsed:.1f} 秒（本项目观察，CPU，纯 Python）；验证准确率 {accuracy(va_scores, Yva):.1%}")
    print(f"  测试准确率 {accuracy(te_scores, Yte):.1%}（实验四调设置时只看验证准确率，选型冻结后再看这一行，不据此改设置）")
    # 画决策边界
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(curve)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("hinge loss")
    axes[0].set_title("training loss")
    gx = [i / 20 for i in range(-36, 37)]
    gy = [i / 20 for i in range(-26, 27)]
    Z = [[m([a, b]).data for a in gx] for b in gy]
    axes[1].contourf(gx, gy, Z, levels=[-100, 0, 100], colors=["#f4c7c3", "#c6dbef"], alpha=0.8)
    axes[1].contour(gx, gy, Z, levels=[0], colors="k", linewidths=1)
    axes[1].scatter([p[0] for p in X], [p[1] for p in X], c=["#1f77b4" if y > 0 else "#d62728" for y in Y], s=12)
    axes[1].set_title("decision boundary (train points)")
    fig.tight_layout()
    path = os.path.join(OUT, "moons_value_mlp.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  图片已保存：{path}")


PARTS = {1: part1_knob, 2: part2_graph, 3: part3_fork, 4: part4_two_layer, 5: part5_engine,
         6: part6_gradcheck, 7: part7_torch_flags, 8: part8_forget_zero, 9: part9_moons}

if __name__ == "__main__":
    for k in [int(a) for a in sys.argv[1:] if a.isdigit()] or list(PARTS):
        PARTS[k]()
    print("\n第 2 章的实验全部完成。")
