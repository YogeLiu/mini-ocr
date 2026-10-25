# 第 9 章　PyTorch 基础

## 本章在全景图中的位置

第 8 章的自动求导是对的，但太慢：每个数一个 Python 对象，一个样本 45 毫秒。**PyTorch** 把同样的思想做成了工业级的工具：计算图的每个节点是一个**张量**（多维数组），每个运算用 C++/CUDA 实现，可以在 GPU 上运行。

从本章开始，本书的所有模型都用 PyTorch 写。本章逐一对照前几章亲手写过的东西：张量 ↔ NumPy 数组（第 1 章），autograd ↔ `Value`（第 8 章），`nn.Linear` ↔ 线性层（第 4 章），`F.cross_entropy` ↔ 交叉熵（第 6 章），`DataLoader` ↔ `minibatches`（第 1 章）。最后用 PyTorch 重做第 8 章的任务：结果相同，速度快了一千多倍。

第 35 章用 Hugging Face 的库训练真实 VLM 时，底层也全是本章的这些部件。

## 学习目标

1. 能创建、索引、变形张量，在张量与 NumPy 数组之间转换，并知道哪些操作共享内存；
2. 能用 autograd 求梯度，正确使用 `requires_grad`、`backward`、`zero_grad`、`no_grad`、`detach`；
3. 能用 `nn.Module` 和 `nn.Parameter` 写一个层，知道 `parameters()`、`state_dict()`、`to(device)` 做了什么；
4. 能写一个 `Dataset`，用 `DataLoader` 产生小批量；
5. 能把计算放到 GPU 上，正确地测量 GPU 的耗时，并判断什么时候 GPU 反而更慢。

## 前置检查

1. 第 8 章训练循环的四步是什么？（答案：清零 → 前向 → 反向 → 更新。）
2. 交叉熵损失的输入是概率还是 logits？（答案：我们的实现接受 logits，内部做 log-softmax，第 6 章。）
3. 运行 `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`，能看到版本号吗？（第 0 章的环境。）

---

## 9.1 张量

### 和 NumPy 几乎一样

```python
import torch
a = torch.arange(12).reshape(3, 4)
```

`python -m labs.ch09.torch_tour` 第 1 部分：

```
torch.arange(12).reshape(3, 4):
 tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
形状 (3, 4) 类型 torch.int64 设备 cpu
a[1, :] [4, 5, 6, 7]   a[:, 2] [2, 6, 10]   a.sum(dim=0) [12, 15, 18, 21]
```

下标、切片、广播、`reshape`、`@`、`sum`……和第 1、4 章的 NumPy 用法完全相同。几个小区别：

| NumPy | PyTorch | 说明 |
|---|---|---|
| `axis=0` | `dim=0` | 两者都接受，PyTorch 习惯写 `dim` |
| `x.astype(np.float32)` | `x.float()` 或 `x.to(torch.float32)` | |
| `np.concatenate` | `torch.cat` | |
| `x[None]` / `np.expand_dims` | `x.unsqueeze(0)` | 插入长度为 1 的维 |
| `x.copy()` | `x.clone()` | |
| 默认小数类型 `float64` | 默认小数类型 **`float32`** | 深度学习用 32 位就够，速度快一倍、内存省一半 |

```
默认的小数类型: torch.float32    torch.tensor([1, 2]) 的类型: torch.int64
```

`torch.tensor([1, 2])` 是整数张量。如果把它用作模型输入，和 float32 的权重相乘会报类型错误。**创建输入数据时注意类型**：图像用 float32，类别标签用 int64（`F.cross_entropy` 要求标签是 int64）。

### 与 NumPy 互相转换

```
torch.from_numpy 共享内存：改张量后 NumPy 数组变成 [7.0, 0.0, 0.0]
torch.tensor 复制一份：改张量后 NumPy 数组仍是 [7.0, 0.0, 0.0]
```

- `torch.from_numpy(数组)`：**共享内存**，不复制，改一个另一个也变（和第 1 章的视图一样）；
- `torch.tensor(数组)`：复制一份。演示里先用 `from_numpy` 把第 0 个元素改成了 7，再用 `torch.tensor` 复制出 `t2`、把 `t2[1]` 改成 9：数组里没有出现 9，仍是 `[7.0, 0.0, 0.0]`；
- `张量.numpy()`：反方向，共享内存（只能用于 CPU 上、不需要梯度的张量）；
- `张量.item()`：只有一个元素的张量 → Python 数。打印损失时常用 `loss.item()`。

### 连续性

```
a.T 是视图，contiguous 吗？ False   a.T.view(-1) 会报错，a.T.reshape(-1) = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
```

第 1 章"深入一步"讲过，转置只是交换步长，内存里的数据不动，所以转置后的张量**不连续**（not contiguous）。PyTorch 有两个变形函数：

- `view`：只改变"怎么看"，要求内存连续，不连续时报错；
- `reshape`：能 view 就 view，不能就复制一份。

不确定时用 `reshape`，或者先 `.contiguous()`。第 15 章写多头注意力时会频繁遇到这个问题。

## 9.2 autograd

### 与 Value 一一对应

第 2 部分，第 5、8 章的迷你神经元：

```python
x, y = torch.tensor(2.0), torch.tensor(1.0)
w = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(-0.3, requires_grad=True)
z = w * x + b
p = torch.sigmoid(z)
L = (p - y) ** 2
L.backward()
```

```
  L = 0.1101   L.grad_fn = PowBackward0   它的上一步 SubBackward0
  dL/dw = -0.294268   dL/db = -0.147134   （第 5、8 章：-0.294268、-0.147134）
  x.requires_grad = False，x.grad = None（输入不需要梯度，PyTorch 不为它算）
```

| 第 8 章的 Value | PyTorch |
|---|---|
| `v.data` | 张量本身（或 `t.detach()`） |
| `v.grad` | `t.grad` |
| `v._prev` + `v._backward` | `t.grad_fn`（一个对象，记录了运算类型和输入） |
| `L.backward()`：拓扑排序 + 逆序调用 | `L.backward()`：同样的事情，用 C++ 实现 |
| 每个节点都算梯度 | 只为 `requires_grad=True` 的张量及其下游算梯度 |

`L.grad_fn` 是 `PowBackward0`（平方运算的反向函数），它的 `next_functions` 指向上一步的 `SubBackward0`（减法）——这就是 PyTorch 记录的计算图，和我们的 `_prev` 一样。

**`requires_grad`**：只有参数需要梯度。输入数据 x 默认 `requires_grad=False`，PyTorch 就不为它计算和保存梯度，节省时间和内存。`nn.Parameter`（9.3 节）默认 `requires_grad=True`。

### 同样的三条规则

```
  不清零再反向一次：w.grad = -0.588536（累加了！）
```

PyTorch 的梯度也是**累加**的，所以每步都要清零（第 8 章 8.4 节）。清零的方式：`optimizer.zero_grad()`（9.5 节），或者手动 `w.grad = None`。

```
  torch.no_grad() 里算出的 z2.requires_grad = False（不记录计算图：评测和参数更新时用）
  detach() 得到的张量 requires_grad = False（从计算图上"剪断"）
```

- **`with torch.no_grad():`** 里的运算不记录计算图。两个用途：(a) **评测**时只需要前向，不记录图可以省下大量内存；(b) **更新参数**时，`w -= lr * w.grad` 这个运算本身不应该进入计算图；
- **`t.detach()`** 返回一个共享数据、但不连在计算图上的张量。用于"我要用这个值，但不希望梯度从这里传回去"。第 20 章的对比学习、第 31 章的 GRPO（旧策略的概率当作常数）都会用到。

### 验证：autograd 与手推梯度、与 Value

练习 9.3 用 autograd 求第 7 章 softmax 回归的梯度：

```python
def softmax_regression_grads(X, y, W, b):
    Xt = torch.as_tensor(X, dtype=torch.float64)
    yt = torch.as_tensor(y, dtype=torch.int64)
    Wt = torch.tensor(W, dtype=torch.float64, requires_grad=True)
    bt = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    loss = F.cross_entropy(Xt @ Wt + bt, yt)
    loss.backward()
    return loss.item(), Wt.grad.numpy(), bt.grad.numpy()
```

测试 `test_autograd_matches_hand_derived` 检查它和第 7 章手推的 `softmax_loss_grad` 完全一致；`test_micrograd_matches_torch` 把第 8 章 Value 版 MLP 的权重复制到 PyTorch 里，检查两者的梯度逐个相等。**三种方法——手推、自己写的自动求导、PyTorch——给出了同样的梯度。** 从现在起，我们可以放心地依赖 `loss.backward()`。

注意 `F.cross_entropy` 的输入是 **logits**，不是概率；它内部做了 log-softmax（第 6 章）。这是 PyTorch 用户最常犯的错误之一：在模型最后加一个 softmax，再传给 `F.cross_entropy`——相当于做了两次 softmax，模型还能训练，但效果明显变差，而且不报错。

## 9.3 nn.Module

### 练习 9.1：自己写一个全连接层

```python
class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        bound = 1.0 / math.sqrt(in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(out_features).uniform_(-bound, bound)) if bias else None

    def forward(self, x):
        y = x @ self.weight.T
        return y + self.bias if self.bias is not None else y
```

要点：

- **继承 `nn.Module`**，并在 `__init__` 里先调用 `super().__init__()`，否则后面的参数登记会出错；
- **`nn.Parameter`**：把张量包装成"参数"。赋值给 `self.xxx` 时，Module 会自动把它登记下来。登记的参数会：出现在 `parameters()` 里（交给优化器）；随 `model.to(device)` 一起搬到 GPU；随 `state_dict()` 一起保存；
- **`forward`** 定义前向计算。调用时写 `layer(x)` 而不是 `layer.forward(x)`：`__call__` 除了调用 `forward`，还会处理一些钩子（hooks）；
- `torch.empty(...)` 分配内存但不初始化，`.uniform_(a, b)` 原地填入均匀随机数。末尾带下划线的方法都是**原地**操作；
- **权重形状是 `(out, in)`**，计算 `x @ W.T`。这是 PyTorch 的约定（第 4 章提到过）；初始化范围 `±1/√in` 与 `nn.Linear` 相同。

```
  MyLinear 的参数: [('weight', (62, 1024)), ('bias', (62,))]
  nn.Linear 的参数: [('weight', (62, 1024)), ('bias', (62,))]  ← 权重是 (out, in)
```

测试 `test_my_linear_matches_nn_linear` 检查：同样的权重下，`MyLinear` 和 `nn.Linear` 的输出完全相同，对三维输入也适用。

### 组合模块

```python
net = nn.Sequential(nn.Flatten(), nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 62))
```

```
  一个 MLP:
 Sequential(
  (0): Flatten(start_dim=1, end_dim=-1)
  (1): Linear(in_features=1024, out_features=256, bias=True)
  (2): ReLU()
  (3): Linear(in_features=256, out_features=62, bias=True)
)
  参数总数: 278334 （第 4 章 mlp_param_count([1024,256,62]) = 278334）
  输入 (8, 1, 32, 32) -> 输出 (8, 62)
  state_dict 的键: ['1.weight', '1.bias', '3.weight', '3.bias']
```

- `nn.Sequential` 把若干模块串起来，依次调用；
- `nn.Flatten()` 把 `(B, 1, 32, 32)` 拉平成 `(B, 1024)`（保留第 0 维批）；
- 参数总数和第 4 章手算的完全一样；
- **`state_dict()`** 是一个"名字 → 张量"的字典，包含所有参数（以及第 11 章会遇到的 BatchNorm 统计量这类"缓冲区"）。保存模型就是保存它：`torch.save(net.state_dict(), "model.pt")`；加载：`net.load_state_dict(torch.load("model.pt"))`。名字里的 `1.`、`3.` 是模块在 Sequential 里的序号。第 35 章加载 Hugging Face 的模型时，看到的也是这样的名字（比如 `model.layers.0.self_attn.q_proj.weight`）。

更复杂的模型（第 12 章起）会写成自定义的 `nn.Module` 子类，在 `__init__` 里创建子模块，在 `forward` 里写清楚数据怎么流动。

## 9.4 Dataset 与 DataLoader

### 练习 9.2：Dataset

PyTorch 用两个类处理数据：

- **`Dataset`**：回答两个问题——"一共有多少个样本"（`__len__`）和"第 i 个样本是什么"（`__getitem__`）；
- **`DataLoader`**：从 Dataset 里取样本，打乱、分批、拼成张量（可以用多个进程并行加载）。

```python
class GlyphDataset(Dataset):
    def __init__(self, seed, chars=CHARSET, n_per_class=100, size=32, align=False, noise=0.0, distort=0.0):
        X, y = glyph_arrays(np.random.default_rng(seed), chars, n_per_class, size, align, noise, flat=False,
                            distort=distort)
        self.x = torch.from_numpy(X).unsqueeze(1)          # (N, 1, size, size)：加上通道维
        self.y = torch.from_numpy(y)
        self.chars = chars

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]
```

`glyph_arrays` 是第 7 章用过的渲染函数（`distort` 是随机形变的幅度，第 10 章才用到，这里原样传下去即可）。这里不拉平，而是保留 `(1, H, W)` 的形状：通道维在前，第 12 章的卷积网络需要它（第 2 章的 CHW 约定）。种子 `seed` 决定了渲染哪些图——**训练、验证、测试用不同的种子**，保证互不重叠。

```
  渲染 1240 张字符图用时 0.06 秒
  ds[5]：图像 (1, 32, 32) torch.float32，标签 5（字符 '5'）
  一批：图像 (64, 1, 32, 32)，标签 (64,)，前 10 个标签 [15, 58, 3, 44, 51, 20, 27, 8, 11, 31]
  一轮有 20 批（1240 / 64 向上取整）
```

`DataLoader(ds, batch_size=64, shuffle=True)` 每轮打乱顺序，每次取 64 个样本，把 64 个 `(1, 32, 32)` 的图像叠成 `(64, 1, 32, 32)`、64 个标签叠成 `(64,)`。最后一批只有 `1240 − 19 × 64 = 24` 个样本（`drop_last=True` 可以丢掉它，第 1 章的 `minibatches` 也有这个选项）。

`generator=torch.Generator().manual_seed(0)` 固定了打乱的随机性，保证可重复（第 1 章）。

**我们的数据很小，构造时就全部渲染好放在内存里。** 真实的数据集（几百万张图）放不下，`__getitem__` 里才去读磁盘上的第 i 张图、做预处理；这时 `DataLoader(num_workers=4)` 用 4 个子进程并行读取，避免 GPU 等数据。第 18 章会讨论数据加载成为瓶颈的情况。

## 9.5 优化器

第 8 章更新参数是手写的：`p.data -= lr * p.grad`。PyTorch 把它包装成**优化器**：

```python
opt = torch.optim.SGD(net.parameters(), lr=0.5)
...
opt.zero_grad()            # 所有参数的 .grad 清零（第 8 章的 zero_grad）
loss.backward()            # 反向
opt.step()                 # 所有参数 p ← p − lr · p.grad（在 no_grad 下执行）
```

为什么要包装？因为更好的更新规则（动量、Adam，第 11 章）需要为每个参数保存额外的状态，优化器负责管理这些状态。`torch.optim.SGD` 就是最朴素的梯度下降；第 11 章会自己实现 SGD 和 Adam，再和 PyTorch 的版本对拍。

## 9.6 GPU

### 搬到 GPU

```python
DEV = torch.device("cuda")                 # Apple 芯片用 "mps"，没有就用 "cpu"
net = net.to(DEV)                          # 所有参数搬到 GPU（原地，对 Module 而言）
xb = xb.to(DEV)                            # 数据也要搬过去（返回新张量，对张量而言）
```

**参与同一个运算的张量必须在同一个设备上**，否则报错 `Expected all tensors to be on the same device`。模型在 GPU 上、数据在 CPU 上，是最常见的情形。

`miniocr/torch_basics.py` 里的 `pick_device()` 自动选择可用的设备。第 9–13 章的演示脚本默认用 `pick_device()` 选出的设备（有 GPU 就用 GPU），加 `--cpu` 参数强制只用 CPU；从第 15 章起，脚本默认只用 CPU，第 16 章起需要 GPU 时加 `--gpu`（附录 D.4 有完整的列表）。

### 速度：矩阵乘法

第 5 部分：

```
  GPU：NVIDIA GeForce RTX 5080
    256×256   CPU     0.04 ms ( 0.80 TFLOPS)   GPU   0.010 ms (  3.25 TFLOPS)   快     4 倍
   1024×1024  CPU     1.23 ms ( 1.75 TFLOPS)   GPU   0.079 ms ( 27.14 TFLOPS)   快    16 倍
   2048×2048  CPU    14.08 ms ( 1.22 TFLOPS)   GPU   0.464 ms ( 37.04 TFLOPS)   快    30 倍
  把 16 MB 的张量从内存拷到显存：0.79 ms（19.8 GB/s）
```

（作者测量时，这块 GPU 上同时运行着另一个占用约 14 GB 显存的服务，所以本书的 GPU 演示都控制在 1 GB 显存以内；绝对速度可能比独占时略低。CPU 的计时每次运行波动很大：再运行一次，1024 和 2048 的 CPU 时间分别是 2.07 ms 和 9.43 ms，倍数变成 26 倍和 20 倍。GPU 的计时很稳定。所以这类倍数只看**量级**，不要在意具体数字。）

TFLOPS 是"每秒万亿次浮点运算"（第 4 章：n×n 的矩阵乘法需要 2n³ 次运算）。几点观察：

1. **矩阵越大，GPU 的优势越大**：256×256 只快 4 倍，2048×2048 快 30 倍。小矩阵时，GPU 的数千个计算单元大部分在闲着，而每次启动一个 GPU 运算（叫一个 kernel）本身就有几微秒的固定开销；
2. **CPU 也不慢**：现代 CPU 用多核和向量指令，可以达到 1–2 TFLOPS。第 4 章用循环写的矩阵乘法每秒只有几百万次运算；
3. **数据搬运有代价**：16 MB 拷贝要 0.79 ms，比在 GPU 上做一次 2048×2048 的矩阵乘法（0.46 ms）还慢。所以要尽量让数据**留在 GPU 上**，不要来回搬。

### 测量 GPU 耗时的陷阱

```
  不同步就计时：0.098 ms（只测到了"提交任务"）；同步后计时：0.680 ms
```

GPU 的运算是**异步**的：Python 代码把任务提交给 GPU 后立刻返回，不等它算完。如果直接用 `time.perf_counter()` 计时，测到的只是"提交任务"的时间。正确的做法是在计时前后调用 `torch.cuda.synchronize()`，等 GPU 真正算完。第 18 章做性能分析时会反复用到这个规则。

另外，第一次调用某个运算时，GPU 要做一些初始化（加载 kernel、分配内存），会特别慢。所以测速前先"热身"（`bench` 函数里先调用一次再计时）。

### 小模型在 GPU 上反而慢

第 6 部分用 PyTorch 重做第 8 章的任务（[64, 16, 10] 的 MLP，8×8 数字）：

```
  （GPU）6 轮用时 0.07 秒，平均每个样本 41 µs（第 8 章的 Value 版：约 45000 µs）
  （CPU）6 轮用时 0.03 秒，平均每个样本 16 µs（第 8 章的 Value 版：约 45000 µs）
```

两个结论：

1. **比 Value 版快了一千到三千倍**，而训练曲线几乎一样：作者在 CPU 上再跑一次，6 轮的训练损失从 2.01 降到 0.24（第 8 章是 2.03 → 0.20），测试准确率在第 4 轮达到 100%，之后在 97.7%–100% 之间波动。每次运行的数字略有不同（DataLoader 的打乱和初始化与第 8 章不同）；
2. **CPU 比 GPU 快**！这个网络只有 1210 个参数，一批 25 个样本，每个运算都极小。GPU 上每个运算的启动开销（几微秒）远大于实际计算时间，再加上每批数据要从内存拷到显存。**GPU 只在运算足够大时才划算。** 第 10 章的模型大了几百倍，GPU 就会明显更快。

## 9.7 自己动手

### 必做

1. **完成练习 9.1–9.3**，使 `python -m pytest labs/ch09` 全部通过。
2. **张量练习**：不运行，写出下列结果的形状，再验证：`torch.zeros(8, 3, 32, 32).flatten(1)`、`torch.zeros(8, 1024).view(8, 32, 32).unsqueeze(1)`、`torch.cat([torch.zeros(2, 3), torch.ones(4, 3)], dim=0)`、`torch.stack([torch.zeros(3)] * 5, dim=1)`。
3. **autograd 练习**：用 PyTorch 求 `f(a, b, c) = (a + b)(b + c)` 在 (1, 2, 3) 处的梯度，和第 8 章自己动手第 2 题的手算结果对照。
4. **设备练习**：故意把模型放在 GPU、数据放在 CPU，读报错信息。然后用 `--cpu` 运行演示第 6 部分，再把网络改成 `[64, 512, 512, 10]`、批大小改成 256，比较 CPU 和 GPU 的速度。从多大的网络开始 GPU 更快？

### 选做

5. 写一个 `nn.Module` 子类 `TwoLayerMLP(d_in, d_hidden, d_out)`，在 `forward` 里用 `F.relu`。打印它的 `state_dict()` 的键，和 `nn.Sequential` 版本比较命名方式。
6. 用 `torch.autograd.gradcheck` 检查 `MyLinear`（输入和参数都要是 float64）。
7. 研究 `DataLoader` 的 `num_workers`：对 `GlyphDataset`（已在内存中），`num_workers=0` 和 `num_workers=4` 哪个快？为什么？如果把渲染放到 `__getitem__` 里呢？

## 9.8 常见错误

| 现象 | 原因 | 怎么查 |
|---|---|---|
| `Expected all tensors to be on the same device` | 模型和数据不在同一个设备 | 每批数据都 `.to(DEV)`；模型 `.to(DEV)` |
| `expected scalar type Long but found Int` / `Float` | 标签不是 int64，或输入不是 float32 | `y.long()`、`x.float()` |
| `view size is not compatible with input tensor's size and stride` | 对不连续的张量用了 `view` | 用 `reshape`，或先 `.contiguous()` |
| 损失下降但效果很差 | 模型输出了 softmax 再传给 `F.cross_entropy` | 模型最后一层输出 logits |
| 显存越用越多直到 OOM | 把 `loss`（带计算图）存进了列表，计算图无法释放 | 记录用 `loss.item()` |
| 评测时显存不够 | 评测没有用 `torch.no_grad()`，记录了计算图 | 评测包在 `with torch.no_grad():` 里 |
| `Can't call numpy() on Tensor that requires grad` | 需要梯度的张量不能直接转 NumPy | `t.detach().cpu().numpy()` |
| GPU 测速结果小得不可思议 | 没有 `torch.cuda.synchronize()` | 计时前后同步，并先热身 |
| 每次运行结果不同 | 没有固定 `torch.manual_seed` 和 DataLoader 的 generator | 固定所有种子 |

## 9.9 真实世界对照

- **Hugging Face 的模型都是 `nn.Module`**。`AutoModelForCausalLM.from_pretrained(...)` 返回的就是一个大的 Module，`model.state_dict()` 的键名对应权重文件里的张量名，`model.to("cuda")` 把几亿个参数搬到 GPU。第 24、35 章打开 Qwen3.5 和 SmolVLM 时，你会看到层层嵌套的 Module：`model.visual.blocks[0].attn.qkv` 这样的路径；
- **数据管道**：真实 VLM 训练的 Dataset 的 `__getitem__` 要读图、做第 2 章的预处理（缩放、归一化）、把文字分词、拼接成聊天模板（第 23 章），再由一个 `collate_fn` 把长短不一的样本补齐成一批（第 1 章的 `pad_sequences`）。这部分代码往往比模型本身还长；
- **精度**：PyTorch 默认 float32。真实的大模型训练用 bf16（第 11 章），显存减半、速度翻倍。第 0 章环境检查里，GPU 和 CPU 的矩阵乘法相差 3×10⁻⁵，就是 float32 舍入误差的表现；
- **异步执行**：GPU 的异步特性让 Python 可以"提前"准备下一批数据，CPU 和 GPU 同时工作。但任何需要从 GPU 取回结果的操作（`.item()`、`.cpu()`、`print`）都会强制同步。训练循环里每步都 `print(loss.item())` 会拖慢训练，所以通常每隔几十步才记录一次。

## 9.10 深入一步

### `requires_grad` 怎么传播

一个运算的输出是否需要梯度，取决于它的输入：**只要有一个输入 `requires_grad=True`，输出就是 `requires_grad=True`**，并且有 `grad_fn`。所以只需要把参数标记为需要梯度，所有由参数算出来的东西（每一层的输出、损失）都会自动进入计算图。

只有**叶子张量**（用户直接创建的，不是由运算得到的）的 `.grad` 会被保留下来。中间结果的梯度在反向传播中用完就丢掉了，以节省内存。如果确实需要看某个中间结果的梯度（比如调试梯度消失），要先调用 `t.retain_grad()`。

### 原地操作与计算图

`x.add_(1)`、`x[0] = 5`、`x += 1` 这些原地操作会修改张量的数据。如果这个张量在反向传播时还要用到（比如它是某个 `tanh` 的输出，`tanh` 的反向需要它），原地修改会破坏梯度计算。PyTorch 给每个张量维护一个"版本号"，检测到这种情况会报错：`one of the variables needed for gradient computation has been modified by an inplace operation`。规则：**对需要梯度的张量，避免原地操作**；更新参数时在 `torch.no_grad()` 里做（优化器的 `step()` 就是这样）。

### 为什么 `nn.Linear` 的权重是 (out, in)

内存布局的原因。`y = x Wᵀ`，第 j 个输出是 x 与 W 的第 j **行**的点积。W 存成 `(out, in)` 时，每个输出神经元的全部权重在内存里是连续的一行，按行读取效率高（第 1 章"深入一步"）。这也和数学书上 `y = Wx`（W 是 `(out, in)`、x 是列向量）的写法一致。

## 9.11 横向对照：本书前后实现的同一件事

| 功能 | 第 1–8 章（NumPy / Value） | PyTorch |
|---|---|---|
| 数组 | `np.ndarray` | `torch.Tensor` |
| 矩阵乘法 | `@` | `@` / `torch.matmul` |
| 线性层 | `linear(x, W, b)`（第 4 章） | `nn.Linear` / `MyLinear` |
| softmax / log-softmax | `miniocr.prob`（第 6 章） | `F.softmax` / `F.log_softmax` |
| 交叉熵 | `cross_entropy`（第 6 章） | `F.cross_entropy` |
| 梯度 | 手推（第 7 章）/ `Value.backward`（第 8 章） | `loss.backward()` |
| 清零梯度 | `zero_grad(params)`（第 8 章） | `opt.zero_grad()` |
| 参数更新 | `p -= lr * g` | `opt.step()` |
| 小批量 | `minibatches`（第 1 章） | `DataLoader` |
| 数值梯度检查 | `numerical_gradient`（第 5 章） | `torch.autograd.gradcheck` |

每一行左边的东西你都亲手写过。遇到 PyTorch 的行为不确定时，回到左边的实现，问自己"它会怎么做"——答案通常一样。

## 9.12 本章小结

- 张量 ≈ NumPy 数组 + 设备 + 自动求导；默认 float32；`from_numpy` 共享内存，`tensor` 复制；不连续的张量用 `reshape`；
- autograd：参数 `requires_grad=True`，`loss.backward()` 填好 `.grad`；梯度累加，每步清零；评测和更新用 `no_grad`；`detach` 剪断计算图；
- 三种方法（手推、Value、PyTorch）给出相同的梯度；
- `nn.Module` + `nn.Parameter`：参数自动登记；`forward` 定义计算；`state_dict` 保存；`nn.Linear` 的权重是 `(out, in)`；
- `F.cross_entropy` 的输入是 logits，标签是 int64；
- `Dataset` 定义"有多少、第 i 个是什么"，`DataLoader` 负责打乱和分批；训练、验证、测试用不同种子；
- GPU：模型和数据都要 `.to(DEV)`；运算越大 GPU 越划算，小模型 CPU 可能更快；测速要同步和热身；少在 CPU 和 GPU 之间搬数据。

## 术语卡

| 中文 | 英文 | 一句话 |
|---|---|---|
| 张量 | tensor | PyTorch 的多维数组 |
| 设备 | device | 张量所在的位置：cpu、cuda、mps |
| 连续 | contiguous | 内存里按行优先顺序紧密排列 |
| 叶子张量 | leaf tensor | 用户直接创建的张量（如参数） |
| grad_fn | grad_fn | 记录张量由哪个运算产生，用于反向传播 |
| 模块 | nn.Module | 包含参数和前向计算的组件 |
| 参数 | nn.Parameter | 会被登记、训练、保存的张量 |
| 状态字典 | state_dict | 名字到参数张量的字典 |
| 数据集 / 数据加载器 | Dataset / DataLoader | 定义样本 / 打乱分批取样本 |
| 优化器 | optimizer | 管理参数更新规则的对象 |
| kernel | kernel | GPU 上执行的一个运算程序 |
| 同步 | synchronize | 等待 GPU 上提交的任务全部完成 |

## 自测题

**概念**

1. `torch.from_numpy` 和 `torch.tensor` 有什么区别？各在什么时候用？
2. 为什么评测时要用 `torch.no_grad()`？不用会怎样？
3. `nn.Parameter` 和普通张量有什么区别？如果在 `__init__` 里写 `self.w = torch.randn(3, 3)`（没有包 Parameter），会发生什么？
4. 为什么一个很小的网络在 GPU 上训练反而比 CPU 慢？

**计算**

5. `nn.Sequential(nn.Flatten(), nn.Linear(3*32*32, 512), nn.ReLU(), nn.Linear(512, 100))` 有多少参数？输入 `(16, 3, 32, 32)` 时输出的形状是什么？
6. 一个 1240 个样本的数据集，`batch_size=64`，一轮有几批？`drop_last=True` 时呢？
7. 4096×4096 的矩阵乘法有多少次浮点运算？如果 GPU 达到 40 TFLOPS，需要多少毫秒？

**代码**

8. 找出下面训练循环中的所有问题：
   ```python
   net = MyNet()
   opt = torch.optim.SGD(net.parameters(), lr=0.1)
   net.to("cuda")
   losses = []
   for xb, yb in loader:
       out = torch.softmax(net(xb), dim=1)
       loss = F.cross_entropy(out, yb)
       loss.backward()
       opt.step()
       losses.append(loss)
   ```
9. 写一个函数 `evaluate(net, ds, device)`，返回模型在 Dataset 上的准确率（一次性前向全部数据，注意 no_grad 和设备）。
10. 下面这行为什么报错？怎么改？`w = torch.randn(3, requires_grad=True); w += 1`

## 参考答案

1. `from_numpy` 共享内存，不复制，适合大数组、只读的场合；`tensor` 复制一份，之后两者独立，适合需要独立修改或不确定是否共享的场合。
2. 评测只需要前向。不用 `no_grad` 时，PyTorch 会为所有运算记录计算图、保存中间结果，浪费大量内存和时间，大模型评测时可能因此显存不够。结果本身不受影响。
3. `nn.Parameter` 会被 Module 自动登记：出现在 `parameters()` 里被优化器更新、随 `to(device)` 搬移、随 `state_dict()` 保存，并且默认 `requires_grad=True`。普通张量不会被登记：优化器看不到它（不会被训练），`net.to("cuda")` 不会搬它（之后与 GPU 上的数据运算会报设备错误），保存模型时也不包含它。
4. 每个 GPU 运算都有固定的启动开销（几微秒），数据还要从内存拷到显存；小网络的每个运算实际计算量极小，开销占了主导。CPU 没有这些开销。
5. `3072 × 512 + 512 + 512 × 100 + 100 = 1,572,864 + 512 + 51,200 + 100 = 1,624,676`。输出 `(16, 100)`。
6. `⌈1240/64⌉ = 20` 批；`drop_last=True` 时 `⌊1240/64⌋ = 19` 批。
7. `2 × 4096³ ≈ 1.37 × 10¹¹` 次；`1.37 × 10¹¹ / 4 × 10¹³ ≈ 3.4 ms`。
8. (a) 数据没有搬到 GPU（`xb, yb = xb.to("cuda"), yb.to("cuda")`）；(b) 对模型输出做了 softmax 再传给 `F.cross_entropy`（应直接传 logits）；(c) 没有 `opt.zero_grad()`，梯度累加；(d) `losses.append(loss)` 保存了带计算图的张量，内存会一直增长，应 `loss.item()`。（`net.to("cuda")` 在创建优化器之后调用，对 `nn.Module` 是原地的，参数对象不变，所以没有问题。）
9. ```python
   def evaluate(net, ds, device):
       net.eval()
       with torch.no_grad():
           pred = net(ds.x.to(device)).argmax(1).cpu()
       return (pred == ds.y).float().mean().item()
   ```
   （数据很大时应该用 DataLoader 分批，避免显存不够。`net.eval()` 在第 11 章讲 BatchNorm 和 Dropout 时会解释。）
10. `w` 是需要梯度的叶子张量，对它做原地操作会报错（`a leaf Variable that requires grad is being used in an in-place operation`）。如果是想更新参数，写 `with torch.no_grad(): w += 1`；如果是想得到一个新张量，写 `w2 = w + 1`。
