"""第 8 章的自动检查：每种运算的梯度都和数值导数对照。"""
import math

import numpy as np

from miniocr.calculus import numerical_gradient, rel_error
from miniocr.nn.micrograd import MLP, Value, cross_entropy_values, graph_nodes, zero_grad


def _num(fn, xs):
    return numerical_gradient(lambda a: fn(*[Value(t) for t in a]).data, np.array(xs, dtype=float))


def _auto(fn, xs):
    vs = [Value(x) for x in xs]
    fn(*vs).backward()
    return [v.grad for v in vs]


def test_mul_and_add():
    f = lambda a, b: a * b + a
    assert _auto(f, [2.0, -3.0]) == [-3.0 + 1.0, 2.0]


def test_each_op_matches_numeric():
    cases = [
        (lambda a, b: a * b, [1.5, -2.0]),
        (lambda a, b: a / b, [1.5, -2.0]),
        (lambda a, b: a - b ** 3, [0.7, 1.3]),
        (lambda a: a.exp(), [0.3]),
        (lambda a: a.log(), [2.5]),
        (lambda a: a.tanh(), [0.4]),
        (lambda a: a.relu(), [0.8]),
        (lambda a: a.relu(), [-0.8]),
        (lambda a: a.sigmoid(), [0.2]),
        (lambda a, b: (a * b).tanh() + (a / b).exp(), [0.5, 1.5]),
    ]
    for fn, xs in cases:
        assert rel_error(_auto(fn, xs), _num(fn, xs)) < 1e-6, fn


def test_reused_node_accumulates():
    """a 被用了 3 次：f = a·a + a，∂f/∂a = 2a + 1。忘记用 += 时这里会错。"""
    a = Value(3.0)
    f = a * a + a
    f.backward()
    assert a.grad == 7.0


def test_topological_order_diamond():
    """菱形图：b、c 都由 a 算出，d 由 b、c 算出。a 必须在 b、c 都把梯度传完之后才反向。"""
    a = Value(2.0)
    b, c = a * 3, a.exp()
    d = b * c
    d.backward()
    assert math.isclose(a.grad, 3 * math.exp(2) + 6 * math.exp(2))


def test_zero_grad_and_mlp():
    rng = np.random.default_rng(0)
    net = MLP([3, 4, 2], rng)
    assert len(net.parameters()) == 3 * 4 + 4 + 4 * 2 + 2
    x = [0.5, -1.0, 2.0]
    loss = cross_entropy_values(net(x), 1)
    loss.backward()
    g1 = [p.grad for p in net.parameters()]
    zero_grad(net.parameters())
    assert all(p.grad == 0.0 for p in net.parameters())
    loss = cross_entropy_values(net(x), 1)
    loss.backward()
    assert np.allclose(g1, [p.grad for p in net.parameters()])
    w = net.parameters()[0]
    old = w.data

    def f(t):
        w.data = t
        return cross_entropy_values(net(x), 1).data
    num = (f(old + 1e-5) - f(old - 1e-5)) / 2e-5
    w.data = old
    assert abs(num - g1[0]) < 1e-6
    assert len(graph_nodes(loss)) > 50
