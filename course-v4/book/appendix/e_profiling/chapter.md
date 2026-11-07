# 附录 E 性能分析

本附录了解即可。读完你能用 PyTorch profiler 找出一段训练代码最慢的部分，并判断一段计算偏向计算受限还是带宽受限。

实验在 `tour.py`，全部在 CPU 上运行（AMD Ryzen 9 9950X，PyTorch 2.14.1，默认两线程），不写 trace 文件。用时随机器和负载变化，所以本附录只说名次和大致比例，测试也只断言故意放的慢点排第一。文中“第 k 部分”指 tour 的第 k 部分。

> **回顾：** 15.7 讲过算术强度、计算受限和访存受限（带宽受限是同一个意思）。16.10 讲过 FLOPs 和 MFU。

---

## E.1 动机

训练比预想的慢。可能是模型太大，可能是数据加载，也可能是某一行顺手记日志的 Python 循环。

凭感觉猜很容易猜错。MFU 只告诉你用上了几成算力，没告诉你时间花在哪里。这一步要靠测量。

## E.2 直觉

分析性能像查一天的时间去哪了。先粗分几大块：上班、通勤、吃饭。哪一块大，再细看这一块。

profiler 的两个视角：

- **span**：你自己划的段落，如前向、反向、优化器。回答“哪一段慢”。
- **算子**：PyTorch 自动记录的 `aten::*`。回答“这一段里哪个操作慢”。

找到以后还要问：快不起来是因为算不动，还是因为数据搬不动？这决定该换什么。

## E.3 推导

### E.3.1 总时间和自身时间

一个操作可能调用别的操作。

- **总时间**（`cpu_time_total`）：包含子操作，适合比较 span。
- **自身时间**（`self_cpu_time_total`）：不含子操作，适合比较算子。

用总时间排算子，外层的包装操作会排在前面，看不出真正的热点。

### E.3.2 只统计中间几步

第一步常有懒初始化。`schedule(wait=1, warmup=1, active=3)` 表示跳过 1 步，再预热 1 步（记录但丢弃），只统计 3 步。

### E.3.3 profiler 会改变它要测的东西

每记录一次算子都有开销，所以算子调用越多，被放大得越多。只要执行路径和步数固定（像本章的训练循环），**次数**这一列就不受影响，是可靠的线索。一个操作被调用几万次，往往说明有 Python 循环。

### E.3.4 计算受限还是带宽受限

回到 15.7.1 的公式：

算术强度 I = FLOPs / 访存字节数

机器平衡点（machine balance）是峰值 FLOP/s 除以内存带宽（字节/s），意思是这台机器每搬一个字节，最多能配多少次运算。按 roofline 模型，性能上界取峰值算力和 I × 带宽两者中较小的一个，两者相等的地方正好是 I 等于这个比值：

- I 大于这个比值：倾向**计算受限**，上界是峰值算力。
- I 小于这个比值：倾向**带宽受限**，上界是 I × 带宽。

两个可以手算的例子（数学结论）：

- y += a·x：每元素 2 FLOPs，读 x、读 y、写 y 共 12 字节（fp32），I = 1/6。
- n × n 矩阵乘：2n³ FLOPs，三个矩阵各进出一次共 12n² 字节，I = n/6。

矩阵乘假设了每个矩阵只进出一次，实际可能更多。

**实测怎样判断：**

1. **算 I**，和机器的比值对照。
2. **加线程**，计算受限的负载通常接近线性增长。带宽受限的负载很快饱和。
3. **换工作集大小**，数据放得进缓存时快、放不进时慢，说明卡在访存。
4. **在 GPU 上**，可以分别调核心频率和显存频率，看哪个影响速度（待验证，本书未做）。

这些只是证据，还算不上证明。线程不增长也可能是算子没有并行，或者调度开销太大。

## E.4 数值例子

### E.4.1 抓一个故意放的慢点

tiny 模型，V = 128，B = 4，T = 32。每步调一次 `slow_weight_norm`，用 Python 循环逐元素算嵌入的平方和，共 128 × 64 = 8,192 个元素。

统计 3 步（本项目观察，第 1 部分，CPU，两线程）：

| 名次 | span | 次数 |
|---:|---|---:|
| 1 | `span/weight_norm`（慢点） | 3 |
| 2 | `span/backward` | 3 |
| 3 | `span/forward` | 3 |
| 4 | `span/optimizer` | 3 |

慢点占这四段总时间的八成上下（几次运行）。这次运行按自身时间排算子，前三名是 `aten::select`、`aten::as_strided`、`aten::item`。

看次数就能定位：`aten::select` 49,158 次，`aten::item` 24,639 次。3 × 8,192 = 24,576，和 `item` 的次数几乎相同，多出的 63 次来自训练循环的其他调用。`weight[i, j]` 索引两维，每个元素两次 `select`，3 × 8,192 × 2 = 49,152，多 6 次（本项目观察，第 1 部分；手算只解释主要部分）。

换成向量化写法后（第 2 部分）：

```python
model.tok.weight.detach().double().square().sum().item()
```

`span/weight_norm` 从第 1 名掉到第 4 名，占比只有 1% 上下。这次运行里 `aten::mm` 成为自身时间最长的算子，调用 135 次。两种写法的结果都是 3.253085，差 3.1 × 10⁻¹⁴（第 3 部分）。

不开 profiler 再量一次（第 3 部分），慢写法的三步总时间仍是快写法的数倍。开 profiler 时慢点占比更高，有一部分来自 E.3.3 说的放大。**修之前修之后，都要在不开 profiler 时量一次真实耗时。**

这里只保证慢点排第一。其他名次都随机器和负载变化，本书不作断言。

### E.4.2 算术强度

（数学结论，第 4 部分。）

| 操作 | I（FLOPs/字节） |
|---|---:|
| y += a·x | 0.167 |
| 矩阵乘 n = 16 | 2.67 |
| 矩阵乘 n = 256 | 42.7 |
| 矩阵乘 n = 512 | 85.3 |
| 矩阵乘 n = 1024 | 170.7 |
| Qwen3-0.6B 预填充，T = 512（15.7 粗算） | 512 |
| Qwen3-0.6B 解码，T_past = 512，B = 1 | 1.05 |
| Qwen3-0.6B 解码，T_past = 512，B = 32 | 13.6 |

机器的比值要你自己填（待验证）。本书不替你的机器给数字。

### E.4.3 加线程试一试

三种负载，线程数 1、2、4（本项目观察，第 5 部分）。下面只写趋势，具体数字每次会变，这里概括三次运行：

| 负载 | I | 线程从 1 加到 4 |
|---|---:|---|
| 矩阵乘，n = 512 | 85.3 | GFLOP/s 约增长到 3 倍 |
| 大工作集 y += a·x（x、y 共 256 MiB） | 0.167 | 三次运行都不到 2 倍，波动较大 |
| 小工作集 y += a·x（x、y 共 32 KiB） | 0.167 | 基本不变 |

这次运行里，矩阵乘的 GFLOP/s 是 y += a·x 的十几倍以上。

怎样读这张表：

- 矩阵乘的表现符合计算受限。
- 大工作集的 y += a·x 增长较少，和带宽受限相符。但这个实验不能单独证明，因为 PyTorch 内部的并行策略也会影响结果。
- 小工作集不随线程变化，很可能是数组太小、没有并行，加上 Python 调用开销（待验证）。所以它不能当计算受限的例子。

单线程时大工作集比小工作集慢，差距每次运行都不一样，换工作集大小这条证据在这组实验里信号不强。换一台机器或换一个负载，结果可能不同。

## E.5 代码

用 `record_function` 划 span：

```python
opt.zero_grad(set_to_none=True)
with record_function("span/forward"):
    loss = model(x, targets=y)[1]
with record_function("span/backward"):
    loss.backward()
with record_function("span/optimizer"):
    opt.step()
with record_function("span/weight_norm"):
    return loss.item(), norm_fn(model)
```

开启 profiler，每步调 `prof.step()`：

```python
with profile(activities=[ProfilerActivity.CPU],
             schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=1)) as prof:
    for _ in range(wait+warmup+active):
        train_step(*setup, norm_fn)
        prof.step()
```

读结果：

```python
rows = [(e.key, e.cpu_time_total, e.count) for e in prof.key_averages() if e.key in SPANS]
```

运行时终端可能多出几行 `USDT ... profiler_start`，以及 `ERROR ... gpuGetDeviceCount failed`。本项目的日志里有这些行，统计结果仍然正常产出。本机是在隐藏 GPU 的情况下运行的。

在你自己的代码上，步骤是：

1. 划 4 到 6 个大 span，把数据加载也划进去。
2. 跑 `wait + warmup + active` 步，按总时间排 span。
3. 对最大的 span，按自身时间和次数看算子。
4. 改完以后不开 profiler，比较每步耗时。

要看内存，加 `profile_memory=True`。要看形状，加 `record_shapes=True`。要定位代码行，加 `with_stack=True`。GPU 上加 `ProfilerActivity.CUDA`，并用 trace 文件看时间线；这时 `cpu_time_total` 统计的是主机端耗时，不能直接代表 GPU kernel 耗时，定位 GPU 热点要看设备（CUDA）时间列。这些本书没有运行（待验证）。

## E.6 常见错误

| 错误 | 现象 | 怎样发现 |
|---|---|---|
| 统计了第一步 | 初始化开销排在前面 | 用 `schedule` 跳过 |
| 用总时间排算子 | 外层包装排第一，看不出热点 | 按自身时间排 |
| 只看开 profiler 时的比例 | 高估小操作多的代码 | 不开 profiler 再量一次 |
| GPU 上不同步就计时 | 时间都算到了后面的操作 | 计时前后调 `miniocr.device.synchronize` |
| 只测一两步 | 名次跳动 | 多测几步，看稳定的部分 |
| 没把数据加载划进去 | 看不到等数据的时间 | 给数据加载也划一个 span |
| 把名次当成跨机器的结论 | 换机器就对不上 | 在目标机器上重测 |

## E.7 小结

- 先划 span，用总时间（含子操作）找哪一段慢；再对最慢的一段，用自身时间（不含子操作）和次数看算子。用 `schedule` 跳过第一步的懒初始化。
- 调用次数巨大通常意味着 Python 循环。本例的慢点排第一，向量化后结果不变、名次掉到最后。
- profiler 会放大调用多的代码，修完要不开 profiler 复测。
- 算术强度 I = FLOPs / 访存字节数。按 roofline 模型，I 高于机器平衡点（峰值 FLOP/s ÷ 带宽）倾向计算受限，低于它倾向带宽受限。fp32 下 y += a·x 的 I 是 1/6，n × n 矩阵乘是 n/6（按每个矩阵只进出一次）。
- I 只是线索，还要用加线程等实验找证据。本例矩阵乘随线程增长明显，大工作集的 y += a·x 增长较少。
- 最常见的错误：用总时间排算子，外层包装排第一；只看开 profiler 时的比例，高估调用多的小操作。

## E.8 精读

**1. Williams, Waterman, Patterson, 2009, *Roofline: An Insightful Visual Performance Model for Multicore Architectures*。** 读模型定义部分。

- 屋顶的斜坡和平顶各对应什么？把 E.4.2 的几个操作放到图上。

**2. Pope et al., 2022, *Efficiently Scaling Transformer Inference*。** 15.12 已列为必读。

- 作者怎样判断解码的瓶颈？和 E.3.4 的方法有什么不同？

**3. PyTorch Profiler 官方教程。** 读 `schedule`、`profile_memory` 和 trace 导出的部分。

- 教程怎样用 `key_averages` 排序？和 E.5 的写法对照。
