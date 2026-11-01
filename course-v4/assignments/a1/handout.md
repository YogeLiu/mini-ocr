# A1：从零写自动求导和训练循环，训练公式符号分类器

**前置**：第 2、3、4、5 章。**算力**：机制题和冒烟只要 CPU；训练层的设备配置待校准（第 8 节）。**总分**：100。

A1 分三层：机制题（60 分）、书面题（15 分）、训练层（25 分）。你从一个空的包开始，自己写自动求导、softmax、交叉熵、MLP、AdamW、学习率调度、梯度裁剪和检查点，再把它们接成一个完整的训练流程，训练 24 类公式符号分类器。

你要交的东西：

1. `assignments/a1/mechanism/` 里的代码，包括训练脚本 `train.py`；
2. `assignments/a1/tests/adapters.py` 里的接线（测试和评测脚本通过它调用你的代码）；
3. 训练好的检查点、日志、冻结文件和测试记录；
4. 实验报告 `writeup.md`（模板见 `writeup_template.md`）。

## 1. 规则

### 1.1 能用什么、不能用什么

`mechanism/` 是**机制实现部分**（`rules.toml` 里列出），受自动检查（`tests/test_a1_rules.py`），子目录也会检查。

- **可以用**：PyTorch 的张量运算（`@`、`exp`、`log`、`sum`、`amax`、`where`……）、自动求导、`torch.nn.Module`、`torch.nn.Parameter`、`torch.nn.ModuleList`、`torch.nn.ModuleDict`、`torch.nn.Sequential`、`torch.optim.Optimizer`（作为基类）、Python 标准库、NumPy；
- **不能用**：`torch.nn` 里的现成层（`Linear`……）、`torch.nn.functional` 的任何函数、`torch.softmax`、`torch.log_softmax`、张量的 `.softmax()`/`.log_softmax()` 方法、`torch.optim` 里的现成优化器、`torch.nn.init`、`torch.nn.utils.clip_grad_norm_`；
- **不能 import** `miniocr` 里除 `miniocr.data` 和 `miniocr.device` 之外的任何模块。`miniocr` 的其余部分就是参考答案；
- **不能 import 其他作业**，也不能在 `mechanism/` 里用 `assignments.` 开头的绝对路径。`mechanism/` 内部用相对导入（`from .optim import ...`），训练脚本不能导入 `assignments.a1.tests`。可以复制你自己写过的代码。

有两道题连张量的自动求导也不能用：

- `value`：`Value` 自己做反向传播；
- `softmax_classifier`：`dW` 自己推、自己写。

其他部分正常用张量 autograd。训练符号模型时，`loss.backward()` 是允许的。

`adapters.py` 只接线，算法都写在 `mechanism/` 里。检查器用语法树查找，抓不住所有绕过方式；规则以本节为准，算法放在哪、`dW` 有没有暗用 autograd，由人工审阅。测试文件里用了 `F.cross_entropy` 等做对照，那是测试，不受限制。

**章节演示不能照搬。** 第 5 章 5.9.1 节的 `SymbolCNN` 用了 `nn.Conv2d`、`nn.Linear` 和 `F.max_pool2d`，5.9.2 节的 `fit` 用了 `torch.optim.AdamW` 和 `F.cross_entropy`，第 5 章实验脚本里的 `jitter_batch` 用了 `F.affine_grid` 和 `F.grid_sample`。这些都不能进 `mechanism/`。A1 交付线性模型和两层 MLP，卷积留给后面的作业。

**数据工具可以用。** `miniocr.data.symbols` 的 `make_dataset`、`group_split`、`average_hash`、`cross_duplicates`、`nearest_pixel_distance`、`augment` 都可以直接调用。这些工具内部用了 `F`，这不代表你自己可以用 `F`。

### 1.2 AI 助手

可以问概念、问报错、让 AI 审阅你的代码、要分级提示。**不要让它替你写 `mechanism/` 里的代码**，写的时候建议关闭 AI 自动补全。报告由你自己写。

### 1.3 数字要诚实

报告里的每个数字都要能在日志里找到：运行 id、配置、设备、日志行、代码位置。只跑过一次的结论要标明。3 步冒烟和章节演示的数字不算训练结果。

### 1.4 验证集和测试集

规则来自第 0 章 0.10.1 节：

- **验证集用来做决定。** 调参、选模型、看错例、提假设，都只用它，可以反复用；
- **测试集是留出集。** 选型冻结以后，预先登记的候选可以各评一次，只报告数字和置信区间，不根据结果改任何设置；
- **开发期不看测试集的错例。** 最终报告定稿以后才可以解封；
- **已经根据测试集改了设置怎么办？** 在报告里写明，再补一份新的留出集（第 5 章 5.8.3 节）。

补充留出集（下文的 fresh）和测试集同样对待。

## 2. 准备

```bash
cd course-v4                                              # 或 make_student 生成的 my-course/
python -m pytest -q assignments/a1/tests                  # 一开始：规则和评测协议的独立测试通过，其余因 NotImplementedError 失败
python -m pytest -q assignments/a1/tests/test_a1_rules.py # 只跑规则检查
```

始终不要设置环境变量 `A_SOLUTIONS`。设了以后，测的就不是你的代码。

建议的代码结构（不强制）：

```
mechanism/
  value.py        Value、数值梯度
  losses.py       softmax、交叉熵、手推梯度的线性分类器
  model.py        两层 MLP 的前向、分类器 Module
  optim.py        AdamW、调度、裁剪
  checkpoint.py   保存和恢复
  train.py        训练脚本
```

机制题按行为评测，不要求内部类名和别人一样。检查点里的 `state_dict` 键名是公开格式，必须遵守。

统一约定（除特别说明，下面的张量函数都遵守）：

- 浮点输入保持 dtype 和 device；
- N、D、H、C 都是正数；
- 非法形状或参数抛 `ValueError`。

每题的完整契约写在 `tests/adapters.py` 的文档字符串里。

## 3. 第一部分：自动求导

### 问题（value）：标量自动求导（10 分）

实现 `make_value(data)`，返回标量 `Value`。第 2 章 2.3–2.7 节。

- 属性 `data`、`grad` 可读写；
- 支持 `+ - * /`（含反向运算，如 `3 - x`、`2 / x`）、常数指数的 `**`、`tanh()`、`relu()`、`backward()`；
- `backward()` 先清中间节点的梯度，再累加到叶子（2.7.7 节）。同一张图反向两次，叶子梯度翻倍，中间梯度不会重复传播；对叶子自己调用 `backward()` 也累加；
- 一个节点被用多次时，梯度相加（2.5.4 节）；
- ReLU 在零处导数取 0；`0 ** 0` 的值是 1，梯度是 0；
- 非数值输入抛 `TypeError` 或 `ValueError`，变量指数抛 `TypeError` 或 `AssertionError`，除零抛 `ZeroDivisionError`。

测试会检查 (2 × −3 + 10)² = 16，三个输入的梯度分别是 −24、16、8。

### 问题（numerical_grad）：数值梯度（3 分）

实现 `run_numerical_grad(f, xs, h=1e-5)`。第 2 章 2.8 节。

`f` 接收浮点列表，返回浮点数。用中心差分，返回同长度的列表，不修改 `xs`。`h` 必须有限且大于 0。f = x²y 在 (2, −3) 的梯度是 (−12, 4)。

### 问题（graph_reasoning）：计算图（3 分，书面）

x = 3，u = x²，L = u + u + x。算出 L 和 dL/dx。反向传播时，为什么不能用 `=` 写回梯度？同一张图连续调用两次 `backward()`，x 和 u 的梯度应该是什么？

## 4. 第二部分：softmax 与交叉熵

### 问题（softmax）：稳定的 softmax（5 分）

实现 `run_softmax(z)`，输入有限浮点 (N, C)，在最后一维计算。第 3 章 3.2–3.3 节。

- 先减去每行最大值（3.3.4 节）；
- fp16、bf16 输入用 fp32 做中间计算，再转回原 dtype。测试用 70000 类的 fp16 全零输入，分母求和不能溢出；
- logits 整体加 10000，结果不变；
- 错误的秩、空张量、整数、NaN 都要拒绝。

### 问题（cross_entropy）：交叉熵（5 分）

实现 `run_cross_entropy(logits, targets)`，返回可反传的标量平均 CE。第 3 章 3.4–3.6 节。

- `logits` 是有限浮点 (N, C)，`targets` 是 long (N,)，且 0 ≤ y < C，和 `logits` 在同一设备；
- 不经过概率，直接用 log-sum-exp（3.3.5 节）。±1000 的 logits 要保持稳定，损失和梯度都要和 PyTorch 对上；
- 只做普通分类。3.7 节的 `ignore_index` 和 3.5.6 节的标签平滑不在这个接口里。

N = 2、C = 2、logits 全零时，损失是 ln 2，梯度元素是 ±1/4。

### 问题（softmax_classifier）：手推梯度的线性分类器（7 分）

实现 `run_softmax_classifier(W, X, y, reg=0.0)`，返回 `(loss, dW)`。第 3 章 3.9 节。

- W 是 (D, C)，X 是 (N, D)，y 是 (N,)；
- `loss` = 平均 CE + reg · ΣW²；
- `dW` = Xᵀ(P − Y)/N + 2·reg·W，**必须手写**。测试在 `no_grad` 下调用；
- 偏置靠给 X 补一列 1 实现，所以正则也作用到 W 的最后一行；
- 维度不合或 reg < 0 要拒绝。

### 问题（ce_derivation）：推导（3 分，书面）

推导平均 CE 对 logits 和对 W 的梯度。减去最大值为什么不改变 softmax？为什么不能先 softmax 再 log？

## 5. 第三部分：两层 MLP

### 问题（mlp）：MLP 与分类器（8 分）

第 2 章 2.6、2.10 节，第 4 章 4.8 节，第 5 章 5.9 节。2.10 节用的是标量 `Value`，5.9 节用的是现成层，这里的接口以下面为准。

实现 `run_mlp(x, w1, b1, w2, b2)`：

- x 是 (N, D)，w1 是 (D, H)，b1 是 (H,)，w2 是 (H, C)，b2 是 (C,)；
- 输出 `relu(x @ w1 + b1) @ w2 + b2`，形状 (N, C)，最后不加 softmax；
- 对全部输入保留计算图。ReLU 可以用 `torch.where` 写。零处导数必须是 0；
- 注意权重方向是 (D, H)，和 `torch.nn.Linear` 的 (H, D) 相反。

实现 `get_classifier(input_dim, hidden_dim, n_classes)`，返回 `Module`：

- 输入 (N, D)，输出 logits (N, C)；
- `hidden_dim = 0` 是线性模型，`state_dict` 键名是 `w`、`b`；否则是两层 MLP，键名是 `w1`、`b1`、`w2`、`b2`；
- 偏置初始化为零，权重随机非零，具体分布你自己定，写进报告；
- 输入维度不对、`input_dim` 或 `n_classes` 不是正数、`hidden_dim` 为负都要拒绝。

### 问题（mlp_accounting）：算账（3 分，书面）

D = 1024，隐藏宽度 H = h，C = 24。写出两层 MLP 的参数量公式，并和线性模型比较。你怎样初始化隐藏层和输出层？权重能不能全取 0？ReLU 在零点的梯度怎样约定？报告里还要用你实际的 h 代入，和代码数出的结果对照。

## 6. 第四部分：优化与检查点

### 问题（adamw）：AdamW（8 分）

实现 `get_adamw(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)`，返回优化器。第 4 章 4.4–4.5 节。

- 要有 `step()`、`zero_grad()` 和可写的 `.lr`；
- 解耦衰减：先 θ ← θ(1 − ηλ)，m 和 v 只用数据梯度更新（4.5.3 节）；
- ε 在根号外面；
- `grad` 为 `None` 的参数整个跳过：不衰减，也不计数；`zero_grad()` 把梯度置为 `None`；
- 所有有梯度的参数都衰减，包括偏置。没有参数组接口；
- lr 或 weight_decay 为负、eps ≤ 0、beta 不在 [0, 1) 都要拒绝。

测试手算首步：θ = 2、g = 4、lr = 0.1、betas = (0.5, 0.5)、eps = 0.1、decay = 0.2 时，结果是 2 × 0.98 − 0.1 × 4/4.1。然后与 `torch.optim.AdamW` 在双精度下对拍 5 步，其中 lr 会变，部分步的梯度是 `None`。

### 问题（schedule）：预热 + 余弦（4 分）

实现 `run_schedule(step, total_steps, warmup_steps, base_lr, min_lr=0.0)`。第 4 章 4.6 节。

- s 从 0 开始；s ≥ T 返回 min_lr；s < W 返回 base·(s + 1)/W；其余按余弦，进度为 (s − W)/max(1, T − W)；
- W = 0 和 W = T 都合法；
- 负步数、T = 0、W > T、min > base、NaN 都要拒绝。

T = 6、W = 2、base = 1、min = 0.2 时，s = 0、1、2、4、6、7 分别得到 0.5、1、1、0.6、0.2、0.2。

### 问题（clip）：梯度裁剪（3 分）

实现 `run_clip_grad_norm(params, max_norm)`，**返回裁剪前的范数**。第 4 章 4.7 节。

- 全局 L2 范数，超过上限时原地乘 max_norm/(norm + 1e-6)；
- 跳过 `None`，没有梯度时返回 0；max_norm 为负要拒绝。

梯度 [3, 4] 和 [12] 的全局范数是 13。

### 问题（checkpoint）：检查点（7 分）

实现下面两个函数。第 4 章 4.10 节。

```python
save_checkpoint(path, model, optimizer, step, generators=None, extra=None)
load_checkpoint(path, model, optimizer=None, generators=None, map_location="cpu")   # 返回 (step, extra)
```

- `step` 是**已完成的步数**，续训从 `range(step, total_steps)` 接着走；
- 保存模型、优化器的配置和状态（m、v、计数，以及非默认的超参）、Python/NumPy/torch 全局随机状态、具名生成器（如 batch、aug）的状态、`extra`；
- 文件至少含键 `model`、`step`、`extra`；
- 先写临时文件再替换，保证原子性（4.10.3 节）。测试验证恢复行为，不模拟断电；原子性由人工审阅；
- 文件不存在抛 `FileNotFoundError`。

测试在 CPU 上做对照：保存后连续走 3 步，和新对象恢复后走 3 步，loss 和所有参数都要逐位相同。

CUDA 和 MPS 上的续训没有测试覆盖。4.10.3 节的 `rng_state` 没有存 MPS 的全局随机状态，如果你在 MPS 上用了随机算子，自己补上并验证。

### 问题（adamw_resume）：AdamW 与续训（3 分，书面）

Adam 加 L2 为什么不等于 AdamW？检查点漏存优化器状态、取批或增广的随机状态、步数，分别会怎样？什么条件下才能讨论“逐位一致”？

## 7. 训练层任务

训练层共 25 分，按第 5 章 5.9.3 节的七步做。分工如下：

| 七步 | 题目 | 分 |
|---|---|---|
| 1 数据、2 泄漏审计 | train_data | 5 |
| 3 训练循环、5 学习曲线 | train_loop（曲线在第 9 节） | 8 |
| 4 调参 | exp_tuning（第 9 节） | 7 |
| 6 冻结后测试、7 记录 | final_report（第 10 节） | 5 |

### 问题（train_data）：数据与泄漏审计（5 分）

**数据。** 24 类、32×32 灰度图，全部本地生成，不需要下载。

- 用 `make_dataset` 生成，用 `group_split` 按来源组做 8:1:1 划分。图像展平成 1024 维，直接用原始 [0, 1] 像素；
- 不做归一化。评测脚本直接把原始像素喂给模型，类别和划分比例也不能换；
- 另用一个生成种子生成补充留出集 fresh，传 `group_offset=24*sources_per_class`，让来源编号不重叠；
- 来源数和每来源张数待校准。5.9.3 节提到的每类 200–400 个来源可以作为探索起点，它不是已校准的配置。

你的 `train.py` 必须和评测脚本用同一种方式取数据，否则两边的划分对不上：

```python
d = make_dataset(sources_per_class=S, per_source=K, seed=SEED, jitter=J, noise=N)
train_idx, val_idx, _ = group_split(d["groups"], fractions=(.8, .1, .1), seed=SPLIT_SEED)
```

`make_dataset` 只能整体生成，所以开发进程的内存里会短暂有 test 的图片。规则管的是你的行为：丢掉 test 的下标，不对它做预测、统计、画图或查看。fresh 在开发期不生成。

**泄漏审计。** `--ckpt` 模式会打印 `leakage` 字段（第 10 节有命令），内容是来源交集、哈希检出率、像素检出率和门限。它的底层是 `average_hash` → `cross_duplicates`，以及 `nearest_pixel_distance` 算出的像素 RMS。

- 开发期只审计 train-val。5.9.3 节第 2 步提到的 train-test 审计，放到冻结之后由 `--final-test` 输出，只给汇总，不给图和下标；
- 第 5 章 5.11 节的“抽查标出来的样本”，开发期也只能用在 train-val 上；
- 划分正确时，来源交集应为 0。这时的检出率是误报参照（5.5.4 节），这些样本不是该删的证据；
- 检测门限（哈希 ≤ 2、像素 RMS ≤ 0.05）是固定的工程示例，没有校准过，也不是训练达标门槛；
- 如果研究确实要换检测规则，在碰 test 之前，用验证集或已知来源的实验定好，然后再冻结。不得因为 test 的检出率改数据或设置。

### 问题（train_loop）：可续训的训练脚本（8 分，含第 9 节的学习曲线）

在 `mechanism/train.py` 里写训练脚本，要求：

- **参数化**：设备（`cpu`、`mps`、`cuda`）、数据、模型宽度、种子、batch、总步数、lr、weight_decay、warmup、min_lr、裁剪上限；
- **用你自己的零件**：你的分类器、CE、AdamW、调度、裁剪；
- **每步的顺序**：按调度设 `opt.lr`，清梯度，前向，反向，裁剪，更新；
- **随机性**：取批和增广各用一个独立生成器（5.9.2 节解释了原因）。增广只用在训练集上，它的随机状态也要存；
- **增广**：可以调 `miniocr.data.symbols.augment`，也可以用基本张量运算自己写。如果用 NumPy 的生成器，它的状态也要存；
- **评测**：`model.eval()` 加 `no_grad`，覆盖整个训练集和验证集，不要用“前若干张”代替（数据按类别排序，前若干张只覆盖部分类别）；
- **检查点**：定期原子保存，包含配置、步数、历史曲线、优化器和随机状态；
- **日志**：每次评测记录步数、训练 loss、验证 loss、验证准确率、lr、梯度范数、耗时、样本/秒，存成 JSON。

**检查点格式。** 评测脚本读取 `model`、`step`、`extra`，其中 `extra` 必须包含：

```python
extra["model_cfg"] = {"input_dim": 1024, "hidden_dim": h, "n_classes": 24}   # h = 0 是线性模型
extra["data_cfg"] = {
    "sources_per_class": ..., "per_source": ..., "seed": ..., "split_seed": ...,
    "jitter": ..., "noise": ..., "fresh_sources_per_class": ..., "fresh_seed": ...,
}
```

`data_cfg` 的 8 个键都必填，不能多也不能少，`seed` 和 `fresh_seed` 必须不同。模型权重用默认浮点精度保存，能被 `load_state_dict` 加载。

**中断续训对照。** 在同一环境里做一次：不中断训到 T 步，和中断在第 c 步、恢复后再训到 T 步，比较最终参数和 loss。报告写实测结果。换设备或环境时，按 4.10.4 节退一步，比较曲线在统计上是否一致。

**自检。** 机制题都接好以后，跑一次冒烟，确认接线能走通训练、保存、加载和验证集评测：

```bash
python -m assignments.a1.evaluate --smoke --run-dir /tmp/a1-smoke
```

冒烟只在 CPU 上走 3 步，用的是极小的数据（每类 2 个来源）。它只检查流程。它的参数不能抄进正式配置，它的数字不能写成训练结果。

## 8. 门槛

| 配置 | 模型/宽度 | 数据规模/增广 | batch/总步数 | lr/warmup/min_lr/decay/clip | dtype/加速方式 | 准确率门槛 |
|---|---|---|---|---|---|---|
| CUDA | 待校准 | 待校准 | 待校准 | 待校准 | 待校准 | 待校准 |
| MPS | 待校准 | 待校准 | 待校准 | 待校准 | 待校准 | 待校准 |
| CPU 兜底 | 待校准 | 待校准 | 待校准 | 待校准 | 待校准 | 待校准（校准时可放宽） |

门槛目前全部**待校准**，由你第一次完整训练后确定，再填进这张表和第 5 章 5.9.3 节的表。这里没有已验证的速度、显存、效果或设备等价性结论。

- 冒烟和章节演示的数字不能作为校准依据；
- 定门槛只看验证集结果，并且在 `--final-test` 之前写下来。看了 test 再定门槛，等于用 test 做决定；
- 5.9.3 节的表比的是测试准确率。这个比较只读出结果，不能反过来改任何设置；
- 还没做完整训练时，报告里就写“待校准”；
- 目前门槛不单独计分。训练层的 25 分看流程和证据。

## 9. 实验

每个实验都写：假设 → 配置差异 → 结果（表或图）→ 解释。只在验证集上比较。

验证集评测命令（结果打印到标准输出，自己存下来）：

```bash
python -m assignments.a1.evaluate --ckpt runs/mlp-s0.pt > runs/mlp-s0.val.json
```

`--device` 和 `--batch-size` 可选，默认 `cpu` 和 128。

### 问题（exp_tuning）：只用验证集调参（7 分）

- 先写假设，再跑。第 5 章 5.7 节的做法：一次只改一项；
- 至少比较 **3 组设置**，每组至少 **3 个配对种子**。可选的变量：宽度、weight_decay、增广强度；
- 线性模型（`hidden_dim=0`）也要训练，作为基线，不计入这 3 组；
- 所有最终候选必须共用同一个 `data_cfg`。增广强度通过训练时的在线增广来改，不要改 `jitter`、`noise`；
- 记录每个种子的结果、平均和范围。**失败和发散的运行也保留**；
- 选型指标和停止规则在碰 test 之前定好，写进报告。

### 问题（exp_curves）：学习曲线（计入 train_loop）

画出训练 loss、验证 loss 和验证准确率随步数的三条曲线，同时记录 lr、梯度范数、耗时和样本吞吐。解释：

- 模型处于欠拟合还是过拟合（5.6 节）；
- loss 和准确率走势不一致时，哪个说了算（5.6.4 节）；
- 你怎样选定检查点和停止步数。

## 10. 冻结与最终测试

### 问题（leakage_protocol）：协议（3 分，书面）

为什么要按来源分组划分？置信区间为什么也要按来源计算？test 的结果不理想时，你能做什么、不能做什么？

### 问题（final_report）：冻结后测试与记录（5 分）

**第一步：登记候选。** 验证集上的选型全部做完以后，一次性写好候选清单。这份清单至少要有线性基线和你选中的 MLP；选中设置的其他种子也建议一起登记，这样不会变成在 test 上挑种子。路径相对于这个 JSON 文件：

```json
{"linear-s0": "linear-s0.pt", "mlp-s0": "mlp-s0.pt"}
```

**第二步：冻结。** 这一步不访问测试数据。

```bash
python -m assignments.a1.evaluate --freeze runs/candidates.json --run-dir runs/final --device cpu
```

冻结文件 `frozen.json` 记下检查点的 SHA256、模型和数据配置、设备、评测批量、检测门限、协议版本和源码哈希。它有这些约束：

- `frozen.json` 只能创建一次，之后不能追加候选；
- 所有候选必须用同一个 `data_cfg`；
- 同一份检查点不能换名字重复登记；
- 源码哈希在参考模式下也包含学习者的 `tests/adapters.py`；训练脚本 `mechanism/train.py` 已包含在机制源码范围内。
- 冻结后改动 `mechanism/`、`tests/adapters.py`、`evaluate.py`、`miniocr/data/symbols.py` 或检查点，最终评测都会拒绝运行。这时还没有占用机会，把文件恢复原样才能继续。恢复不了就在报告里写明，并按本讲义 1.4 节补一份新的留出集。

冻结前先用同样的 `--device` 对每个候选跑一次 `--ckpt`，确认它们都能加载。

**第三步：最终评测。** 每个候选各一次：

```bash
python -m assignments.a1.evaluate --final-test --candidate mlp-s0 --run-dir runs/final
```

- 脚本先写 `TEST_STARTED_<检查点哈希>.json`，再访问数据，所以**失败也占用机会**；
- 第一次启动会写 `REGISTRY_SHA256`，钉住整份候选清单；
- 结果写进 `TEST_RESULT_<检查点哈希>.json`；
- `--final-test` 不接受 `--device` 和 `--batch-size`，只用冻结时的值；
- 同一项研究保留同一个 `run-dir`。不要删记录，也不要换目录重来。这是本地审计协议，靠你守规矩，它防不住有意篡改；
- `run-dir` 不能放在名为 `outputs` 的目录下，脚本会拒绝。

**结果怎么读。** A1 用符号分类面板，协议版本 `a1-symbol-classification-v1`。

- test 和 fresh 各报告 `n`、`n_groups`、平均 CE、准确率、`accuracy_ci95` 和 `leakage`；
- 没有按类别的切片，也不输出预测和错例；
- 区间用来源 cluster bootstrap：有放回地抽来源组，组内图片一起进入，重复 2000 次，种子 0，取 95% 分位区间；
- 这个区间不代表训练随机性，也不保证跨数据域的泛化；
- 第 0 章 0.9.3 节的 EM、CER、可编译率、渲染一致率是公式识别的指标，A1 不报告它们。

最终报告定稿后才可以解封做错例分析。评测脚本不提供这个功能，A1 也不要求做。

**记录。** 报告里完整写出配置、运行 id、代码版本或源码哈希、设备、系统、PyTorch 版本、dtype、种子、数据生成和划分参数、预处理、训练时长和日志位置（格式参考 5.7.5 节）。门槛状态按第 8 节写。

## 11. 交付清单与评分

| 项 | 内容 | 分 |
|---|---|---|
| 机制题 | 测试全部通过（含规则测试）：value 10、numerical_grad 3、softmax 5、cross_entropy 5、softmax_classifier 7、mlp 8、adamw 8、schedule 4、clip 3、checkpoint 7 | 60 |
| 书面题 | graph_reasoning、ce_derivation、mlp_accounting、adamw_resume、leakage_protocol，各 3 分 | 15 |
| train_data | 数据配置、划分、来源交集和近重复汇总 | 5 |
| train_loop | `train.py`、中断续训对照、学习曲线和停止理由 | 8 |
| exp_tuning | 至少 3 组设置 × 3 个种子的全部验证结果 | 7 |
| final_report | 冻结清单、启动记录、test 和 fresh 的汇总与区间、完整记录 | 5 |
| 合计 | | 100 |
| 代码可追溯 | 随机抽报告里 3 个数字，能给出运行 id、配置、设备、日志行、代码位置 | 通过 / 不通过 |

测试通过不等于学术诚信的证明。算法是否放在 `mechanism/`、`dW` 是否手推、检查点是否原子保存、报告结论是否受 test 影响，都要人工审阅。
