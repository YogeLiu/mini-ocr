"""第 8 章：一个只有一百多行的自动求导引擎（思路来自 Andrej Karpathy 的 micrograd）。

每个 Value 保存：
    data       这个数的值（前向计算的结果）
    grad       最终输出（损失）对它的导数 ∂L/∂(这个数)，反向传播后才有意义
    _prev      它是由哪些 Value 算出来的（计算图里的"父节点"）
    _backward  一个函数：已知 self.grad 时，把梯度按链式法则"分给"父节点

反向传播 = 按拓扑顺序的逆序，依次调用每个节点的 _backward。
"""
import math


class Value:
    def __init__(self, data, _prev=(), op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_prev)
        self._backward = lambda: None
        self.op = op                                     # 只用来打印计算图

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # ------------------------------------------------------------------ 基本运算
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # z = x + y：∂z/∂x = 1，∂z/∂y = 1。用 += 而不是 =：同一个节点可能被用了好几次，梯度要累加。
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    # >>> 练习 8.1：乘法的反向传播
    def __mul__(self, other):
        """z = x · y：∂z/∂x = y，∂z/∂y = x。"""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    # <<<

    def __pow__(self, k):
        """z = x^k（k 是普通的数，不是 Value）：∂z/∂x = k·x^(k-1)。"""
        assert isinstance(k, (int, float)), "只支持常数次幂"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    # >>> 练习 8.2：exp、log、tanh、relu 的反向传播
    def exp(self):
        """z = e^x：∂z/∂x = e^x = z。"""
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        """z = ln x：∂z/∂x = 1/x。"""
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        """z = tanh x：∂z/∂x = 1 - z²。"""
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        """z = max(0, x)：x > 0 时 ∂z/∂x = 1，否则为 0。"""
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out
    # <<<

    # ------------------------------------------------------------------ 由基本运算组合出来的运算（不需要自己的 _backward）
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1 if isinstance(other, Value) else 1.0 / other)

    def __radd__(self, other):              # 让 3 + Value 也能用：Python 会调用 Value.__radd__(3)
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Value(other) - self

    def __rtruediv__(self, other):
        return Value(other) / self

    def sigmoid(self):
        return 1 / (1 + (-self).exp())

    # ------------------------------------------------------------------ 反向传播
    # >>> 练习 8.3：拓扑排序与反向传播
    def backward(self):
        """从 self（通常是损失）出发做反向传播，填好计算图里每个节点的 grad。

        1. 拓扑排序：用深度优先搜索，保证每个节点排在它所有父节点之后（"先有父，后有子"）；
        2. self.grad = 1（∂L/∂L = 1）；
        3. 按拓扑顺序的逆序调用每个节点的 _backward：轮到一个节点时，用到它的所有子节点都已经把梯度传给它了。
        """
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for p in v._prev:
                    build(p)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
    # <<<


def zero_grad(params):
    """把一组参数的梯度清零。每次反向传播前都要做，否则梯度会和上一步的累加在一起。"""
    for p in params:
        p.grad = 0.0


def graph_nodes(root):
    """计算图里的所有节点（用于统计和打印）。"""
    out, stack = [], [root]
    seen = set()
    while stack:
        v = stack.pop()
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
        stack.extend(v._prev)
    return out


# ====================================================================== 用 Value 搭一个多层感知机
class Neuron:
    """一个神经元：y = act(Σ w_i x_i + b)。"""

    def __init__(self, n_in, rng, act="tanh"):
        s = 1.0 / math.sqrt(n_in)
        self.w = [Value(rng.uniform(-s, s)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.act = act

    def __call__(self, x):
        z = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return {"tanh": z.tanh, "relu": z.relu, "none": lambda: z}[self.act]()

    def parameters(self):
        return self.w + [self.b]


# >>> 练习 8.4：层与多层感知机
class Layer:
    """n_out 个神经元，每个都接收同样的 n_in 个输入。"""

    def __init__(self, n_in, n_out, rng, act="tanh"):
        self.neurons = [Neuron(n_in, rng, act) for _ in range(n_out)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """sizes = [n_in, h1, ..., n_out]。隐藏层用 tanh，最后一层不加激活（输出 logits）。"""

    def __init__(self, sizes, rng):
        self.layers = [Layer(a, b, rng, act="tanh" if i < len(sizes) - 2 else "none")
                       for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
# <<<


def cross_entropy_values(logits, label):
    """一个样本的交叉熵：logits 是 Value 的列表。用第 6 章的减最大值技巧（最大值当作常数）。"""
    m = max(z.data for z in logits)
    exps = [(z - m).exp() for z in logits]
    return (sum(exps[1:], exps[0])).log() - (logits[label] - m)
