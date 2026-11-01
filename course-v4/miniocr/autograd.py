"""第 2 章：一个标量自动求导引擎（思路来自 Andrej Karpathy 的 micrograd）。

每个 Value 是一个数，外加它在计算图里的"来历"：

    data       这个数的值（前向计算的结果）
    grad       最终输出（通常是损失 L）对它的导数 ∂L/∂(这个数)；调用 backward() 之后才有意义
    _prev      它是由哪些 Value 直接算出来的（计算图里指向它的那些节点）
    _backward  一个函数：已知 self.grad 时，按链式法则把梯度"分摊"给 _prev 里的节点
    op         产生它的运算名，只用来打印计算图

反向传播 = 先做拓扑排序，再按逆序依次调用每个节点的 _backward。

本模块还提供：
    numerical_grad   用中心差分估计梯度，用来检查解析梯度（2.8 节）
    rel_error        两个梯度之间的相对误差
    zero_grad        把一组参数的梯度清零
    Neuron / Layer / MLP   用 Value 搭的多层感知机（2.10 节）
"""
import math
import random


class Value:
    def __init__(self, data, _prev=(), op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_prev)
        self._backward = lambda: None
        self.op = op
        self.label = label                           # 可选的名字，打印计算图时用

    def __repr__(self):
        name = f"{self.label}=" if self.label else ""
        return f"Value({name}{self.data:.4g}, grad={self.grad:.4g})"

    # ------------------------------------------------------------------ 基本运算：每个都要写自己的 _backward
    def __add__(self, other):
        """z = x + y：∂z/∂x = 1，∂z/∂y = 1。加法把上游梯度原样复制给两个输入。"""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # 用 += 而不是 =：同一个节点可能被用了好几次，各条路径传回来的梯度要相加（2.5 节）。
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    # >>> 练习 2.1：乘法的反向传播
    def __mul__(self, other):
        """z = x · y：∂z/∂x = y，∂z/∂y = x。乘法把上游梯度乘以"对方"的值，交叉分给两个输入。"""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    # <<<

    # >>> 练习 2.2：幂、tanh、relu 的反向传播
    def __pow__(self, k):
        """z = x^k，k 是普通的数（不是 Value）：∂z/∂x = k · x^(k−1)。
        k = 0 时 z 恒为 1，导数是 0；单独处理，免得 x = 0 时去算 0 ** −1 而报 ZeroDivisionError。"""
        assert isinstance(k, (int, float)), "只支持常数次幂"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            local = 0.0 if k == 0 else k * self.data ** (k - 1)
            self.grad += local * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        """z = tanh x：∂z/∂x = 1 − z²。用前向已经算好的 z，不必再算一次 tanh。"""
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t * t) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        """z = max(0, x)：x > 0 时 ∂z/∂x = 1，否则为 0（约定 x = 0 处的导数取 0，和 PyTorch 一致）。"""
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out
    # <<<

    # ------------------------------------------------------------------ 组合运算：由基本运算拼出来，不需要自己的 _backward
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1 if isinstance(other, Value) else 1.0 / other)

    def __radd__(self, other):                       # 3 + Value：Python 先试 int.__add__，失败后调用 Value.__radd__(3)
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Value(other) - self

    def __rtruediv__(self, other):
        return Value(other) / self

    # ------------------------------------------------------------------ 反向传播
    # >>> 练习 2.3：拓扑排序与反向传播
    def backward(self):
        """从 self（通常是损失 L）出发做反向传播，把 ∂L/∂v 累加进图里每个叶子节点 v 的 grad。

        1. 拓扑排序：深度优先搜索，一个节点在它所有输入之后才加入 topo（"先有输入，后有输出"）。
        2. 中间节点（有 _prev 的节点）的 grad 先清零：它们只是本次反向传播的草稿纸。
           叶子节点（参数、输入）的 grad 不清零，和 PyTorch 一样会累加，所以训练时每步要先 zero_grad。
        3. self.grad += 1（∂L/∂L = 1）。用 += 而不是 =：如果 self 是中间节点，第 2 步已把它清零，结果就是 1；
           如果 self 本身是叶子（比如直接调用 x.backward()），也要和其他叶子一样累加。
        4. 按 topo 的逆序调用 _backward：轮到一个节点时，用到它的所有节点都已经把梯度传给它了。
        """
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for p in v._prev:
                    build(p)
                topo.append(v)
        build(self)
        for v in topo:
            if v._prev:
                v.grad = 0.0
        self.grad += 1.0
        for v in reversed(topo):
            v._backward()
    # <<<


def zero_grad(params):
    """把一组参数的梯度清零。每次 backward 之前都要做，否则这一步的梯度会和以前的累加在一起。"""
    for p in params:
        p.grad = 0.0


def graph_nodes(root):
    """root 的计算图里的所有节点（用于统计节点数和打印）。"""
    out, stack, seen = [], [root], set()
    while stack:
        v = stack.pop()
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
        stack.extend(v._prev)
    return out


# ====================================================================== 数值梯度检查
# >>> 练习 2.4：数值梯度与相对误差
def numerical_grad(f, xs, h=1e-5):
    """用中心差分估计 f 在 xs 处的梯度。

    f  接收一个 float 列表、返回一个 float 的普通函数（不用 Value）。
    xs float 列表。
    返回和 xs 等长的列表，第 i 个元素是 (f(x + h·e_i) − f(x − h·e_i)) / (2h)。
    中心差分的误差是 O(h²)，float64 下 h 取 1e-5 左右最合适（2.8 节）。
    """
    xs = [float(x) for x in xs]
    grads = []
    for i in range(len(xs)):
        old = xs[i]
        xs[i] = old + h
        fp = f(xs)
        xs[i] = old - h
        fm = f(xs)
        xs[i] = old                                   # 一定要还原，否则后面各维都在错误的点上求导
        grads.append((fp - fm) / (2 * h))
    return grads


def rel_error(a, b, eps=1e-12):
    """相对误差 |a − b| / max(|a|, |b|, eps)。a、b 可以是数，也可以是等长的列表（取各分量中的最大值）。"""
    if isinstance(a, (list, tuple)):
        return max((rel_error(x, y, eps) for x, y in zip(a, b)), default=0.0)
    return abs(a - b) / max(abs(a), abs(b), eps)
# <<<


# ====================================================================== 用 Value 搭一个多层感知机
class Neuron:
    """一个神经元：act(Σ_i w_i x_i + b)。act 取 "tanh"、"relu" 或 "none"。"""

    def __init__(self, n_in, rng, act="tanh"):
        s = 1.0 / math.sqrt(n_in)
        self.w = [Value(rng.uniform(-s, s)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.act = act

    def __call__(self, x):
        z = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.act == "tanh":
            return z.tanh()
        if self.act == "relu":
            return z.relu()
        return z

    def parameters(self):
        return self.w + [self.b]


# >>> 练习 2.5：层与多层感知机
class Layer:
    """n_out 个神经元，每个都接收同样的 n_in 个输入，输出一个长度为 n_out 的列表。"""

    def __init__(self, n_in, n_out, rng, act="tanh"):
        self.neurons = [Neuron(n_in, rng, act) for _ in range(n_out)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """sizes = [n_in, h1, ..., n_out]。隐藏层用 act（默认 tanh），最后一层不加激活。
    n_out = 1 时直接返回一个 Value，否则返回列表。"""

    def __init__(self, sizes, seed=0, act="tanh"):
        rng = random.Random(seed)
        n = len(sizes) - 1
        self.layers = [Layer(a, b, rng, act if i < n - 1 else "none")
                       for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
# <<<
