"""第 2 章的自动检查：python -m pytest book/part1_foundations/ch02_backprop"""
import math
import random

import pytest
import torch

from miniocr.autograd import MLP, Value, graph_nodes, numerical_grad, rel_error, zero_grad


def test_mul_backward():
    """练习 2.1：乘法把上游梯度乘以"对方"的值；和普通数混用也要能求导。"""
    x, y = Value(3.0), Value(-4.0)
    z = x * y
    z.backward()
    assert z.data == -12.0 and x.grad == -4.0 and y.grad == 3.0
    a = Value(2.0)
    b = 5 * a * 3           # __rmul__ 和 __mul__ 混用
    b.backward()
    assert b.data == 30.0 and a.grad == 15.0


def test_pow_tanh_relu_backward():
    """练习 2.2：三个一元运算的局部导数。"""
    x = Value(1.5)
    (x ** 3).backward()
    assert math.isclose(x.grad, 3 * 1.5 ** 2)
    x = Value(2.0)
    (x ** -1).backward()                                   # 除法就是乘以 x^(−1)
    assert math.isclose(x.grad, -0.25)
    x = Value(0.0)
    y = x ** 0                                             # x⁰ 恒为 1，导数是 0；不能去算 0 ** −1
    y.backward()
    assert y.data == 1.0 and x.grad == 0.0
    x = Value(0.8)
    t = x.tanh()
    t.backward()
    assert math.isclose(t.data, math.tanh(0.8)) and math.isclose(x.grad, 1 - math.tanh(0.8) ** 2)
    for v, g in ((2.0, 1.0), (-2.0, 0.0), (0.0, 0.0)):    # 约定 relu 在 0 处的导数取 0，和 PyTorch 一致
        x = Value(v)
        r = x.relu()
        r.backward()
        assert r.data == max(v, 0.0) and x.grad == g


def test_same_variable_used_twice_accumulates():
    """2.5 节：一个变量被用了两次，各条路径的梯度要相加。"""
    x = Value(3.0)
    (x * x).backward()
    assert x.grad == 6.0
    x, y = Value(2.0), Value(5.0)
    (x * y + x).backward()
    assert x.grad == 6.0 and y.grad == 2.0
    x = Value(-1.5)
    (x + x + x).backward()
    assert x.grad == 3.0


def test_backward_uses_topological_order():
    """练习 2.3：L = (a·b + c)²；以及一个"菱形"图：x 经两条长短不同的路径汇合。"""
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    L = (a * b + c) ** 2
    L.backward()
    assert (a.grad, b.grad, c.grad) == (-24.0, 16.0, 8.0)
    x = Value(0.5)
    u = x * 2.0
    v = (u * 3.0).tanh()                                   # 长路径
    out = v * u                                            # u 被用了两次
    out.backward()
    t = math.tanh(3.0)
    expected = 2.0 * (3.0 * (1 - t * t) * 1.0 + t)         # d(out)/du · du/dx
    assert math.isclose(x.grad, expected, rel_tol=1e-12)


def test_backward_twice_accumulates_only_leaves():
    """叶子节点的 grad 累加（和 PyTorch 一样），中间节点每次 backward 都重算。"""
    w, x = Value(2.0), Value(3.0)
    h = w * x
    L = h * h
    L.backward()
    assert w.grad == 2 * 6.0 * 3.0 and h.grad == 12.0
    L.backward()
    assert w.grad == 2 * 36.0 and h.grad == 12.0
    zero_grad([w, x])
    assert w.grad == 0.0 and x.grad == 0.0
    x = Value(2.0)                                         # 根本身是叶子：也要累加（PyTorch 同样得到 2）
    x.backward()
    x.backward()
    assert x.grad == 2.0


def test_two_layer_network_matches_chapter_and_torch():
    """2.6 节手算的两层网络：前向、所有梯度和正文的表一致，并和 PyTorch 一致。"""
    from book.part1_foundations.ch02_backprop.tour import PARAM_NAMES, two_layer_torch, two_layer_values
    v = two_layer_values()
    v["L"].backward()
    fwd = {k: v[k].data for k in ("z1", "a1", "z2", "a2", "yhat", "L")}
    assert fwd == dict(z1=1.5, a1=1.5, z2=1.0, a2=1.0, yhat=1.0, L=1.0)
    grads = {k: v[k].grad for k in PARAM_NAMES + ["x1", "x2", "z1", "z2"]}
    assert grads == dict(w11=-4.0, w12=-2.0, b1=-2.0, w21=4.0, w22=2.0, b2=2.0, v1=-3.0, v2=-2.0, c=-2.0,
                         x1=-1.0, x2=4.0, z1=-2.0, z2=2.0)
    t, L = two_layer_torch()
    L.backward()
    for k in PARAM_NAMES + ["x1", "x2"]:
        assert math.isclose(t[k].grad.item(), v[k].grad, abs_tol=1e-12)


def test_matches_torch_on_random_mlp():
    """随机初始化的 MLP：Value 的每个参数梯度和 PyTorch（float64）一致。"""
    m = MLP([3, 5, 4, 1], seed=7)
    P = m.parameters()
    rng = random.Random(1)
    X = [[rng.uniform(-1, 1) for _ in range(3)] for _ in range(4)]
    Y = [1.0, -1.0, 0.5, 0.0]
    loss = sum((m(x) - y) ** 2 for x, y in zip(X, Y)) * 0.25
    zero_grad(P)
    loss.backward()
    tp = [torch.tensor(p.data, dtype=torch.float64, requires_grad=True) for p in P]
    k, layers = 0, []
    for li, layer in enumerate(m.layers):
        rows = []
        act = "tanh" if li < len(m.layers) - 1 else "none"   # 按约定写死，不读待测模型的 n.act
        for n in layer.neurons:
            rows.append((tp[k:k + len(n.w)], tp[k + len(n.w)], act))
            k += len(n.w) + 1
        layers.append(rows)
    tl = 0
    for x, y in zip(X, Y):
        h = [torch.tensor(xi, dtype=torch.float64) for xi in x]
        for rows in layers:
            out = []
            for ws, b, act in rows:
                z = sum(wi * hi for wi, hi in zip(ws, h)) + b
                out.append(torch.tanh(z) if act == "tanh" else z)
            h = out
        tl = tl + (h[0] - y) ** 2
    (tl * 0.25).backward()
    assert math.isclose(loss.data, (tl * 0.25).item(), rel_tol=1e-12)
    assert max(abs(p.grad - t.grad.item()) for p, t in zip(P, tp)) < 1e-12


def test_mlp_output_layer_is_linear():
    """练习 2.5：最后一层不加激活。固定权重后输出要能大于 1，偏置的梯度是 1，输出层权重的梯度是隐藏层的输出。"""
    m = MLP([2, 3, 1], seed=0)
    for n in m.layers[0].neurons:
        n.w[0].data, n.w[1].data, n.b.data = 0.5, -0.5, 0.1
    out_n = m.layers[1].neurons[0]
    for w in out_n.w:
        w.data = 2.0
    out_n.b.data = 1.5
    x = [0.4, -0.2]
    h = math.tanh(0.5 * 0.4 - 0.5 * -0.2 + 0.1)
    s = m(x)
    assert math.isclose(s.data, 3 * 2.0 * h + 1.5, rel_tol=1e-12) and s.data > 1
    zero_grad(m.parameters())
    s.backward()
    assert out_n.b.grad == 1.0
    assert all(math.isclose(w.grad, h, rel_tol=1e-12) for w in out_n.w)


def test_numerical_grad_and_rel_error():
    """练习 2.4：中心差分要准到 O(h²)，不能改动传入的列表；相对误差的定义。"""
    f = lambda v: v[0] ** 3 + math.sin(v[1]) * v[0]
    xs = [1.2, 0.4]
    g = numerical_grad(f, xs)
    assert xs == [1.2, 0.4]
    exact = [3 * 1.2 ** 2 + math.sin(0.4), math.cos(0.4) * 1.2]
    assert rel_error(g, exact) < 1e-8
    g_big = numerical_grad(lambda v: v[0] ** 3, [1.0], h=1e-2)
    assert math.isclose(g_big[0], 3 + 1e-4, rel_tol=1e-9)        # 中心差分对 x³ 的误差恰好是 h²
    assert rel_error(2.0, 1.0) == 0.5
    assert rel_error([1.0, 3.0], [1.0, 2.0]) == pytest.approx(1 / 3)
    assert rel_error(0.0, 0.0) == 0.0


def test_gradcheck_value_against_numerical():
    """用数值梯度检查 Value 搭的一个光滑函数（tanh 网络）。"""
    m = MLP([2, 3, 1], seed=3)
    P = m.parameters()
    X, Y = [[0.3, -0.7], [1.1, 0.2]], [0.5, -0.2]

    def loss():
        return sum((m(x) - y) ** 2 for x, y in zip(X, Y))

    def f(vals):
        for p, v in zip(P, vals):
            p.data = v
        return loss().data

    vals = [p.data for p in P]
    num = numerical_grad(f, vals)
    f(vals)
    L = loss()
    zero_grad(P)
    L.backward()
    assert rel_error([p.grad for p in P], num) < 1e-6


def test_mlp_structure_and_training():
    """练习 2.5：参数个数、输出类型；在 4 个点上训练 30 步，损失要明显下降。"""
    m = MLP([2, 8, 8, 1], seed=0)
    assert len(m.parameters()) == 2 * 8 + 8 + 8 * 8 + 8 + 8 + 1
    out = m([0.1, -0.2])
    assert isinstance(out, Value)
    assert isinstance(MLP([2, 3, 2], seed=0)([0.1, 0.2]), list)
    X, Y = [[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]], [1.0, -1.0, 1.0, -1.0]   # 标签就是 x1 的符号
    P = m.parameters()
    losses = []
    for _ in range(30):
        L = sum((m(x) - y) ** 2 for x, y in zip(X, Y)) * 0.25
        zero_grad(P)
        L.backward()
        for p in P:
            p.data -= 0.1 * p.grad
        losses.append(L.data)
    assert losses[-1] < 0.5 * losses[0]
    assert len(graph_nodes(L)) > len(P)


def test_tanh_variant_answers():
    """习题 H3 的答案：2.6 节网络的隐藏层换成 tanh。"""
    from book.part1_foundations.ch02_backprop.tour import NET
    v = {k: Value(x) for k, x in NET.items()}
    z1 = v["w11"] * v["x1"] + v["w12"] * v["x2"] + v["b1"]
    z2 = v["w21"] * v["x1"] + v["w22"] * v["x2"] + v["b2"]
    a1, a2 = z1.tanh(), z2.tanh()
    yhat = v["v1"] * a1 + v["v2"] * a2 + v["c"]
    L = (yhat - v["y"]) ** 2
    L.backward()
    got = [a1.data, a2.data, yhat.data, L.data, yhat.grad, z1.grad, z2.grad,
           v["w11"].grad, v["w21"].grad, v["v1"].grad, v["x1"].grad]
    want = [0.9051, 0.7616, 0.6436, 1.8399, -2.7129, -0.4902, 1.1393, -0.9805, 2.2787, -2.4556, 0.0794]
    assert all(abs(g - w) < 1e-4 for g, w in zip(got, want))
