"""第 8 章：自己写的自动求导。逐段运行，对照正文阅读。

    python -m labs.ch08.autograd_tour
"""
import math
import time

import numpy as np

from miniocr.data.tasks import glyph_arrays
from miniocr.nn.micrograd import MLP, Value, cross_entropy_values, graph_nodes, zero_grad


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def part1_neuron():
    section("1. 第 5 章的迷你神经元，这次让计算机自己求导")
    x, y = Value(2.0), Value(1.0)
    w, b = Value(0.5), Value(-0.3)
    z = w * x + b
    p = z.sigmoid()
    L = (p - y) ** 2
    L.backward()
    print(f"  L = {L.data:.4f}   dL/dw = {w.grad:.6f}   dL/db = {b.grad:.6f}   （第 5 章手算：-0.294268、-0.147134）")
    nodes = graph_nodes(L)
    print(f"  这个小小的计算一共产生了 {len(nodes)} 个节点：")
    topo, seen = [], set()

    def build(v):
        if v not in seen:
            seen.add(v)
            for q in v._prev:
                build(q)
            topo.append(v)
    build(L)
    names = {id(x): "x", id(y): "y", id(w): "w", id(b): "b", id(z): "z", id(p): "p", id(L): "L"}
    for i, v in enumerate(topo):
        print(f"    {i:2d}  {names.get(id(v), ''):2s} op={v.op or '输入':6s} data={v.data:9.4f}  grad={v.grad:9.4f}")


def part2_accumulate():
    section("2. 为什么梯度要用 += 累加")
    a = Value(3.0)
    f = a * a + a
    f.backward()
    print(f"  f = a·a + a，a = 3：自动求导 df/da = {a.grad}，手算 2a + 1 = 7")

    class BadValue(Value):
        def __mul__(self, other):
            other = other if isinstance(other, Value) else BadValue(other)
            out = BadValue(self.data * other.data, (self, other), "*")

            def _backward():
                self.grad = other.data * out.grad              # 错误：= 而不是 +=
                other.grad = self.data * out.grad
            out._backward = _backward
            return out
    a = BadValue(3.0)
    f = a * a
    f.backward()
    print(f"  用 = 代替 += 时，f = a·a 的 df/da = {a.grad}（应该是 2a = 6）：同一个节点作为两个输入，第二次赋值覆盖了第一次")


def part3_zero_grad():
    section("3. 忘了清零梯度")
    w = Value(1.0)
    for step in range(3):
        L = (w * 2.0 - 1.0) ** 2
        L.backward()
        print(f"  w = 1，第 {step} 次反向（不清零）：w.grad = {w.grad:.2f}（每次都应是 4(2w-1) = 4）")


def part4_order():
    section("4. 为什么要按拓扑顺序的逆序")
    x = Value(2.0)
    a = x * 1.0                                               # 一个中间节点（有自己的 _backward）
    b, c = a * 3, a.exp()
    d = b * c
    d.backward()
    right = x.grad
    for v in (x, a, b, c, d):
        v.grad = 0.0
    d.grad = 1.0
    for v in (d, b, a, c):                                    # 一个错误的顺序：a 在 c 之前反向
        v._backward()
    print(f"  d = (3a)·e^a，a = x·1，x = 2：正确的 dd/dx = {right:.4f}（= 3e² + 3·2·e² = {9 * math.exp(2):.4f}）")
    print(f"  错误顺序 (d, b, a, c) 得到 {x.grad:.4f}（= 3e² = {3 * math.exp(2):.4f}）：a 把梯度传给 x 时，c 那一份还没到")


def part5_train():
    section("5. 用 Value 搭的 MLP 识别 10 个数字（8×8 对齐）")
    Xtr, ytr = glyph_arrays(np.random.default_rng(0), chars="0123456789", n_per_class=30, size=8, align=True)
    Xte, yte = glyph_arrays(np.random.default_rng(1), chars="0123456789", n_per_class=30, size=8, align=True)
    rng = np.random.default_rng(0)
    net = MLP([64, 16, 10], rng)
    params = net.parameters()
    print(f"  训练集 {len(Xtr)} 张，测试集 {len(Xte)} 张；网络 [64, 16, 10]，{len(params)} 个参数")
    sample_loss = cross_entropy_values(net(list(Xtr[0])), int(ytr[0]))
    print(f"  一个样本的计算图有 {len(graph_nodes(sample_loss))} 个节点；初始损失 {sample_loss.data:.4f}（ln 10 = {math.log(10):.4f}）")
    lr, bs = 0.5, 25
    t0 = time.perf_counter()
    for epoch in range(6):
        order = rng.permutation(len(Xtr))
        total = 0.0
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            zero_grad(params)
            loss = sum((cross_entropy_values(net(list(Xtr[i])), int(ytr[i])) for i in idx), Value(0.0)) / len(idx)
            loss.backward()
            for p in params:
                p.data -= lr * p.grad
            total += loss.data * len(idx)
        if True:
            acc = np.mean([np.argmax([o.data for o in net(list(x))]) == t for x, t in zip(Xte, yte)])
            print(f"    第 {epoch + 1:2d} 轮  训练损失 {total / len(Xtr):.4f}  测试准确率 {acc:.1%}  （已用 {time.perf_counter() - t0:.0f} 秒）")
    per = (time.perf_counter() - t0) / (6 * len(Xtr))
    print(f"  平均每个样本（前向 + 反向）{per * 1000:.2f} ms。第 7 章的 NumPy softmax 回归处理 3720×30 个样本只要 0.3 秒")


if __name__ == "__main__":
    part1_neuron()
    part2_accumulate()
    part3_zero_grad()
    part4_order()
    part5_train()
    print("\n第 8 章的演示全部完成。")
