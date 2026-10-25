# A1：基础——从零训练一个字符识别器

| | |
|---|---|
| 位置 | 第 12 章之后（里程碑 A） |
| 预计用时 | 15–25 小时 |
| 算力 | 写代码、跑测试：CPU。实验：CPU 也能做完，GPU 更快。排行榜：一张租用的 24 GB 显卡，训练时间不超过 10 分钟 |
| 交付 | ① `a1/` 里的代码，`tests/adapters.py` 接好后全部测试通过；② 书面报告 `writeup.md`（从 `writeup_template.md` 复制）；③ 排行榜记录 |
| 满分 | 100 分：代码 53 分，书面 9 分，实验 32 分，排行榜 6 分 |

第 1–12 章里，你一次写一个函数，每个函数都有现成的签名和说明。这个作业不一样：**你从一个空的包开始，自己决定怎么组织代码**，最后得到一个能训练、能续训、能复现、能给出排行榜成绩的完整训练系统。测试只规定行为，不规定写法。

你会用到的知识：第 2 章（图像就是数组）、第 6 章（交叉熵）、第 8–9 章（自动求导、PyTorch）、第 10 章（训练循环、体检清单、5 种故障）、第 11 章（优化器、调度、裁剪、BN）、第 12 章（卷积、im2col、感受野）。

---

## 1. 规则

### 1.1 能用什么、不能用什么

`a1/` 里的代码：

| 可以用 | 不可以用 |
|---|---|
| 张量运算（`torch.zeros`、`@`、`.sum`、`.amax`、`torch.exp`、`torch.log`……）和 autograd | `torch.nn.functional` 的任何东西 |
| `nn.Module`、`nn.Parameter`、`nn.ModuleList`、`nn.ModuleDict`、`nn.Sequential`（只当容器用） | `torch.nn` 里的层和损失：`nn.Linear`、`nn.Conv2d`、`nn.BatchNorm2d`、`nn.MaxPool2d`、`nn.ReLU`、`nn.CrossEntropyLoss`…… |
| 继承 `torch.optim.Optimizer` 写你自己的优化器 | `torch.optim` 里现成的优化器和调度器；`torch.nn.utils.clip_grad_norm_` |
| `torch.save`、`torch.load` | `Tensor.unfold`、`F.unfold`（im2col 自己写）；`torch.log_softmax`、`torch.softmax`、`torch.conv2d` 之类的函数版 |
| `miniocr.data` 里的**数据生成器**（渲染器、`glyph_arrays`） | `miniocr` 里的其他东西（模型、训练、优化器……）——它们就是你要写的 |
| NumPy、标准库 | 预训练权重 |

`tests/test_a1_rules.py` 会用语法树检查 `a1/` 里的代码。它查不出所有绕过的办法，**规则以本节为准**。测试代码里可以用 `torch.nn.functional`，因为那是拿来对照的标准答案。

### 1.2 关于 AI 助手

和第 0.10 节相同：可以让 AI 解释概念、解释报错、帮你检查自己写的代码，**不要让它替你写 `a1/` 里的代码**。画图、命令行参数解析这类样板代码可以请它帮忙，但你要能逐行解释。

### 1.3 参考实现

`assignments/solutions/` 里有一份参考实现。`make_student` 生成的 `my-course/` 里不会复制这个目录。如果你直接在 `course-v3/` 里做，**提交之前不要打开它**。卡住时先重读第 8–12 章，再按第 0.10 节"卡住了怎么办"的办法排查。

---

## 2. 准备

```
assignments/a1_basics/
├── handout.md                 本说明
├── a1/                        你的代码（现在只有 __init__.py）
├── tests/
│   ├── adapters.py            ← 你要填的"接线"：测试通过它调用你的代码
│   └── test_a1_*.py           测试（不要改）
├── leaderboard.py             排行榜：固定测试集、打分
├── data/leaderboard_test.npz  排行榜测试集
└── writeup_template.md        书面报告模板
```

所有命令都在 `course-v3/`（或 `my-course/`）下运行：

```bash
python -m pytest -q assignments/a1_basics/tests          # 现在大部分是失败：NotImplementedError
python -m pytest -q assignments/a1_basics/tests -k conv  # 只跑名字里带 conv 的测试
```

**前提**：在 `my-course/` 里做时，第 2 章的练习（尤其是 2.6 区域平均缩放、2.7 归一化）必须已经完成，因为数据生成器要用到它们。否则数据相关的测试会报 `NotImplementedError: 练习 2.7`。

**适配器怎么用**：比如 `adapters.run_conv2d(x, weight, bias, stride, padding)`，你在 `a1/` 的某个文件里写好卷积，然后在适配器里 `from assignments.a1_basics.a1.你的文件 import 你的函数` 并调用它。适配器里只做"接线"：导入、构造、把权重装进去、返回结果，不放实质逻辑。

建议的顺序与本文的章节顺序相同：数据 → 层 → 模型 → 优化 → 训练 → 实验 → 排行榜。每写完一部分就跑对应的测试。

---

## 3. 数据

所有数据都来自 `miniocr.data.tasks.glyph_arrays`：62 个字符（`0-9a-zA-Z`），32×32 灰度图，不做第 2 章的字符归一化（`align=False`），用 `flat=False` 保留图的形状。

### 问题（data）：生成、去重、取批（6 分）

写三个函数，接到 `run_build_split`、`run_remove_overlap`、`run_get_batch`：

1. **生成**：给定 `seed, n_per_class, distort, noise`，用 `np.random.default_rng(seed)` 调用 `glyph_arrays`，返回 `X`：float32 `(N, 1, 32, 32)`，墨迹 = 1、背景 = 0；`y`：int64 `(N,)`。同一个种子必须得到完全相同的数据。
2. **去重**：去掉 `X` 中与参照集 `X_ref` 某张图**逐像素完全相同**的样本，保持原顺序，返回剩下的 `(X, y)` 和去掉的个数。要求 O(N) 而不是 O(N²)（提示：把每张图变成可以放进集合的东西）。
3. **取批**：从 `(X, y)` 里**有放回地**随机取 `batch_size` 个，随机性只来自传入的 `torch.Generator`，返回两个张量。

> 为什么取批要用自己的 `Generator`，而不是全局随机数？想一想续训：从检查点恢复以后，下一批应该和"没中断时的下一批"一模一样。第 7 节的测试会检查这一点。

**交付**：代码，`test_a1_data.py` 全部通过。

### 问题（data_dup）：重复样本（2 分，书面）

训练集用种子 100、每类 100 张，验证集用种子 101、每类 10 张，在下面四种设置下各算两个数：训练集里有多少张互不相同的图，验证集里有多少张与训练集重复。

| | noise=0 | noise=0.05 |
|---|---|---|
| distort=0 | | |
| distort=1 | | |

**交付**：填好表，两三句话解释：(a) 为什么 distort=0、noise=0 时会有重复；(b) noise=0.05 时"重复数是 0"能不能说明验证集是干净的？逐像素比较的去重方法有什么盲区？（第 33 章会讲近似重复检测。）

---

## 4. 层与损失

### 问题（linear）：线性层（2 分）

`Linear(d_in, d_out, bias=True)`：参数 `weight` 的形状是 `(d_out, d_in)`，`bias` 的形状是 `(d_out,)`，输出 `x Wᵀ + b`，`x` 可以有任意多个前导维。初始化：`weight` 用 Kaiming 正态分布，均值 0、标准差 `√(2 / d_in)`；`bias` 初始为 0。

**交付**：`run_linear`、`get_linear`；`test_linear`、`test_init_scale` 通过。

### 问题（conv2d）：卷积（8 分）

1. `unfold(x, kh, kw, stride, padding)`：im2col，`(N, C, H, W) → (N, C·kh·kw, L)`，`L` 是输出位置的个数。排列顺序与 `F.unfold` 相同：通道在最外层，窗口内先行后列。**不许用 Python 循环遍历输出位置**（那样 32×32 的图要循环一千次）；遍历窗口内的 `kh × kw` 个偏移是允许的，每个偏移一次取出一整片。
2. `conv2d(x, w, b, stride, padding)`：用 `unfold` 加一次矩阵乘法实现。结果和梯度都要与 `F.conv2d` 一致。
3. `Conv2d(c_in, c_out, k, stride=1, padding=0, bias=True)`：模块版，`weight` 的形状是 `(c_out, c_in, k, k)`，Kaiming 正态初始化，标准差 `√(2 / (c_in·k·k))`。

梯度不需要手写：只要前向全由张量运算组成，autograd 会算。但你要确认它算的是对的，测试会和 `F.conv2d` 的梯度对照。

**交付**：`run_conv2d`、`get_conv2d`；`test_conv2d`（5 种配置）通过。

### 问题（max_pool）：最大池化（2 分）

`max_pool2d(x, k)`：窗口 k×k、步长 k；高或宽不是 k 的倍数时，丢掉放不下一个窗口的边缘（与 `F.max_pool2d(x, k)` 相同）。提示：一次 `reshape` 加一次 `amax` 就够了。

**交付**：`run_max_pool2d`；`test_max_pool2d` 通过。

### 问题（batchnorm）：批归一化（6 分）

函数版 `batch_norm2d(x, weight, bias, running_mean, running_var, training, momentum, eps)` 和模块版 `BatchNorm2d(c)`，行为与 `F.batch_norm` 完全相同：

- `training=True`：用**本批**每个通道的均值和**有偏**方差归一化；同时**就地**更新 `running ← (1 − momentum)·running + momentum·本批统计量`，其中方差用**无偏**估计；
- `training=False`：用 `running_mean`、`running_var` 归一化；
- 最后乘 `weight`、加 `bias`。

模块版：参数 `weight`（初始 1）、`bias`（初始 0）；缓冲区（`register_buffer`）`running_mean`（初始 0）、`running_var`（初始 1）。模块用 `self.training` 决定走哪条分支。

> 有偏和无偏两种方差都要用，这是 PyTorch 的约定。这个细节写错，训练时一切正常，评测时准确率却莫名其妙地低一截。第 11 章讲过这类"静默错误"，附录 B 有完整清单。

**交付**：`run_batch_norm2d`；`test_batch_norm2d`（两种模式）通过。

### 问题（softmax_ce）：log-softmax 与交叉熵（4 分）

1. `log_softmax(x, dim)`：数值稳定，先减去最大值。logits 是 ±1000 时也不能出现 `inf` 或 `nan`。
2. `cross_entropy(logits, targets, label_smoothing=0.0)`：logits `(N, K)`，返回批内平均。带标签平滑时，每个样本的损失是 `(1 − ε)·(−log p_y) + ε·(1/K)·Σ_k(−log p_k)`。

**交付**：`run_log_softmax`、`run_cross_entropy`；`test_log_softmax_stable`、`test_cross_entropy` 通过。

---

## 5. 模型：CharCNN

```
输入 (N, 1, 32, 32)
→ [Conv 3×3, padding 1, 无偏置 → BN → ReLU → MaxPool 2]   channels[0]   → 16×16
→ [Conv 3×3, padding 1, 无偏置 → BN → ReLU → MaxPool 2]   channels[1]   → 8×8
→ [Conv 3×3, padding 1, 无偏置 → BN → ReLU → MaxPool 2]   channels[2]   → 4×4
→ 全局平均池化（对高和宽求平均）                          (N, channels[2])
→ Linear                                                   (N, num_classes)
```

默认 `channels = (32, 64, 128)`、`num_classes = 62`。`state_dict()` 的键必须**恰好**是（`i = 0, 1, 2`）：

```
convs.{i}.weight
bns.{i}.weight   bns.{i}.bias   bns.{i}.running_mean   bns.{i}.running_var
head.weight      head.bias
```

用 `nn.ModuleList` 存三个卷积和三个 BN，把线性层命名为 `head`，就能得到这些名字。卷积为什么不要偏置？因为紧接着的 BN 会减去均值，偏置被抵消了。

### 问题（char_cnn）：搭模型（4 分）

**交付**：`get_char_cnn`、`run_char_cnn`；`test_state_dict_keys_and_param_count`、`test_char_cnn_matches_reference` 通过。后一个测试用 `torch.nn` 搭了同样的网络，给 BN 设了随机的运行统计量，和你的模型在评测模式下逐值比较。

### 问题（conv_accounting）：算账（4 分，书面）

对默认配置（`channels = (32, 64, 128)`、62 类）手算，写出过程：

1. 每一层的参数量和总参数量。（写完代码后用 `sum(p.numel() …)` 核对，但要先手算。）
2. 一张图前向需要多少次乘加（MAC）？每个卷积层各占多少？哪一层最贵？为什么第 2、3 层的通道数翻倍，计算量却差不多？
3. 最后一个卷积层的一个输出像素，在输入图上的感受野有多大？（第 12.3 节的递推公式。）它比 32 大还是小？对"认出整个字符"意味着什么？
4. batch 128、float32 时，第一个卷积层的输出这一个张量占多少字节？与全部参数占的字节数比一比。

---

## 6. 优化

### 问题（adamw）：AdamW（6 分）

继承 `torch.optim.Optimizer` 写 `AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)`，更新与 `torch.optim.AdamW` 相同（第 11 章）。要求：

- 超参数放进 `defaults`，在 `step()` 里从 `self.param_groups` 读取。这样训练循环修改 `group["lr"]` 就能实现学习率调度；
- 每个参数的状态（步数 t、m、v）放在 `self.state[p]` 里，这样 `state_dict()` 和 `load_state_dict()` 会自动保存和恢复它们。

**交付**：`get_adamw_cls`；`test_adamw_matches_torch`（中途改学习率，逐步对拍 12 步）、`test_adamw_state_roundtrip` 通过。

### 问题（adamw_accounting）：优化器的显存（3 分，书面）

1. float32 训练时，每个参数要占多少字节？分成参数本身、梯度、m、v 四项来算。默认 CharCNN 一共要多少？
2. 对比第 4 问：这个模型训练时，显存主要花在参数和优化器状态上，还是花在激活值上？batch 从 128 增大到 1024，哪一部分会跟着变？
3. （思考）到了第 28 章的 VLM，结论会反过来吗？

### 问题（schedule）：预热加余弦（2 分）

`lr_cosine_schedule(it, max_lr, min_lr, warmup_iters, cosine_cycle_iters)`：

- `it < warmup_iters`：`it / warmup_iters · max_lr`（注意 it = 0 时是 0）；
- `warmup_iters ≤ it ≤ cosine_cycle_iters`：`min_lr + ½(1 + cos(π·(it − warmup_iters)/(cosine_cycle_iters − warmup_iters)))·(max_lr − min_lr)`；
- `it > cosine_cycle_iters`：`min_lr`。

这个定义和第 11 章的 `warmup_cosine` 有一处不同，找出来。

**交付**：`run_lr_cosine_schedule`；`test_lr_cosine_schedule` 通过。

### 问题（clip）：梯度裁剪（2 分）

`clip_grad_norm(params, max_norm)`：所有梯度合起来的 L2 长度超过 `max_norm` 时，每个梯度乘以 `max_norm / (total + 1e-6)`；没有梯度的参数跳过；返回裁剪前的长度。

**交付**：`run_clip_grad_norm`；`test_clip_grad_norm` 通过。

---

## 7. 训练

### 问题（checkpoint）：检查点（3 分）

`save_checkpoint(model, optimizer, iteration, out)` 和 `load_checkpoint(src, model, optimizer) → iteration`。`out`、`src` 可能是路径，也可能是打开的文件对象（`torch.save`、`torch.load` 两种都支持）。

**交付**：`run_save_checkpoint`、`run_load_checkpoint`；`test_checkpoint_resume_is_exact` 通过：训练 3 步、存盘、读回一个新模型，两边再各训练 3 步，参数要完全相同。

### 问题（train）：训练脚本（8 分）

写一个训练函数和命令行入口。配置至少包括下面这些键（`run_train` 的测试会传入其中一部分）：

| 键 | 含义 | 建议默认值 |
|---|---|---|
| `n_per_class`、`val_per_class` | 训练集、验证集每类的张数 | 100、10 |
| `distort`、`noise` | 渲染参数 | 1.0、0.05 |
| `train_seed`、`val_seed` | 数据种子；**不要用 20261022–20261099**（留给排行榜） | 100、101 |
| `channels` | CharCNN 的通道数 | (32, 64, 128) |
| `batch_size` | | 128 |
| `max_lr`、`min_lr`、`warmup_iters`、`lr_decay_iters` | 学习率调度；`lr_decay_iters` 为 0 时等于 `max_iters` | 3e-3、3e-4、100、0 |
| `max_iters` | 训练到第几步 | 2000 |
| `weight_decay`、`clip`、`label_smoothing` | `clip=0` 表示不裁剪 | 0.01、1.0、0 |
| `eval_every` | 每隔几步在验证集上评测一次 | 200 |
| `seed` | 初始化和取批的种子 | 0 |
| `device` | `cpu` 或 `cuda`。**默认 `cpu`**（本书的约定，见附录 D.4） | cpu |
| `out_dir` | 日志和检查点的目录；空字符串表示不写 | "" |

要求：

1. **可复现**：同样的配置和数据，两次运行的每一步损失完全相同；换一个 `seed` 就不同。
2. **能续训**：从检查点恢复后，结果与不中断完全相同。检查点除了模型和优化器，还要存什么？（提示：第 3 节的 `Generator`。）测试的做法是：一次跑 6 步；另一次先跑 3 步存盘，再从检查点跑到第 6 步。两边的 6 个损失和最后的参数要一致。`lr_decay_iters` 就是为这个设的：先跑 3 步时调度也必须按"总共 6 步"来算。
3. **评测**：每 `eval_every` 步、以及最后一步，在验证集上算损失和准确率。评测时切到 `eval()` 并关掉梯度，评测完切回 `train()`。
4. **日志**：`out_dir` 非空时，每一步往 `out_dir/log.jsonl` 写一行 JSON（至少有 `iter`、`loss`、`lr`，评测的步再加 `val_loss`、`val_acc`、`elapsed`），并把配置存成 `config.json`。实验要画的图都从这个文件画。
5. **命令行**：至少支持下面这些用法（具体的名字由你定，在报告里写清楚）：
   - 用默认配置训练，从命令行覆盖任意一个键；
   - `--smoke`：极小的配置（几个通道、几十步），CPU 上几秒钟跑完，用来检查流程；
   - 从检查点续训；
   - 用检查点对一个 `.npz` 文件做预测，存成 `.npy`（排行榜要用）。

**交付**：`run_train`；`test_train_is_reproducible`、`test_resume_matches_uninterrupted` 通过。冒烟命令能跑通。

---

## 8. 实验

所有实验都写进 `writeup.md`。每个实验都要：**先写预测，再运行**；附上从 `log.jsonl` 画出的图；说明你看到了什么，与预测有什么不同，为什么。只有一次运行的结论要标明"单次运行"；需要比较大小的结论至少跑 2 个种子（第 25 章）。

算力：这些实验都能在 CPU 上做完。每次运行先用 `--smoke` 确认能跑通，再上正式配置。在租用的 GPU 上做的话，按第 9 节的流程来，把显卡型号和每次运行的耗时记进报告。

### 问题（exp_debug）：亲手制造 5 种故障（6 分）

在训练脚本里加一个只在实验时打开的开关，分别制造下面 5 种错误，各训练 300 步：

1. 忘了 `zero_grad`；
2. 标签被打乱（每个 batch 内随机重排 `y`）；
3. 学习率乘以 100；
4. 图和标签错位：取批时把 `xb` 先 `reshape(batch, -1)` 再转置回原形状之类的操作，让每张"图"混进了别的样本的像素（第 10 章的⑤）；
5. 在 logits 上先做一次 softmax 再传给 `cross_entropy`（"双重 softmax"）。

**交付**：一张 5 + 1 条曲线的训练损失图（加上正常的一条），一张验证准确率图，一张表："症状 → 从曲线上怎么认出来 → 原因"。运行前先预测：哪几种错误的训练损失看起来"正常"，只有验证集能暴露问题？

### 问题（exp_lr）：学习率扫描（6 分）

固定其他配置，`max_lr` 取 1e-4、3e-4、1e-3、3e-3、1e-2、3e-2、1e-1、3e-1（`min_lr = max_lr / 10`），各跑 2 个种子。

**交付**：(a) 训练损失曲线；(b) 最终验证准确率对 `max_lr` 的图（横轴取对数，画出两个种子）；(c) 损失开始发散的学习率是多少？最好的学习率离"发散的边缘"有多远？(d) 关掉预热（`warmup_iters=0`）再跑最好的两个学习率，有区别吗？

### 问题（exp_batch）：batch 大小（4 分）

batch 取 16、64、256、1024。比较两种做法：学习率保持不变；学习率按 batch 的平方根放大。为了公平，横轴用"见过的样本数"，并记录每次运行的墙钟时间。

**交付**：图和一段分析：batch 变大时，"每秒处理的样本数"和"达到同一准确率需要的样本数"各怎么变？在 CPU 和 GPU 上结论一样吗？

### 问题（exp_data）：数据量（6 分）

`n_per_class` 取 5、10、20、50、100、200、400，每个都训练到收敛（步数要随数据量调整；说明你怎么判断"收敛"），各跑 2 个种子。

**交付**：(a) 测试错误率对训练集大小的图，两个坐标轴都取对数；(b) 看起来像一条直线吗？粗略估计斜率；(c) 数据量小的时候，训练损失和验证损失的差距有多大？什么设置能缩小它（下一题）？

### 问题（exp_ablation）：消融（6 分）

以你最好的配置为基准，每次只改一处：

1. 去掉三个 BN（换成恒等映射，卷积加上偏置）；
2. `label_smoothing = 0.1`；
3. 加上数据增强：训练时随机平移 ±2 像素（自己实现，不用 torchvision）；
4. `weight_decay` 取 0 和 0.1。

在 `n_per_class = 20` 和 `200` 两种数据量下各做一遍。

**交付**：一张表（各配置、两种数据量下的验证准确率，均值 ± 两个种子的差），一段分析：哪些改动在数据少时更有用？为什么？

### 问题（exp_bn_eval）：BN 的训练模式和评测模式（4 分）

拿一个训练好的模型，在验证集上用 4 种方式评测：`eval()` 模式；`train()` 模式且 batch = 1024；`train()` 模式且 batch = 8；`train()` 模式且 batch = 1。

**交付**：一张表，加上解释：为什么 `train()` 模式下结果会随 batch 大小变化？batch = 1 时发生了什么？（提示：第三个 BN 的输入是 8×8。batch = 1 时，每个通道只剩 64 个数用来求均值和方差，而且全来自同一张图。）

---

## 9. 排行榜

### 问题（leaderboard）：10 分钟能训练到多准（6 分）

目标：在 `data/leaderboard_test.npz`（3100 张，每类 50 张，distort=1.0，noise=0.05）上的准确率尽量高。

**规则**

1. **硬件和时间**：一张租用的 24 GB 显卡（RTX 4090、L4、A10 这一级），从训练开始到结束的墙钟时间不超过 **10 分钟**，评测时间不算。生成数据的时间也不算，但训练图像总数不超过 50 万张。报告里写明显卡型号。
2. **数据**：只能用 `miniocr.data` 的渲染器生成，加上你自己写的增强。种子不能用 20261022–20261099。**不能用测试集训练、调参或选检查点**，测试集只用来打最终的分。
3. **代码**：第 1.1 节的限制照样适用。在这个前提下，结构、优化器的超参数、增强、混合精度（`torch.autocast`）、`torch.compile` 都可以自由改。
4. **打分**：`python -m assignments.a1_basics.leaderboard score 你的预测.npy`。

**GPU 的租用流程**：

1. 在本机 CPU 上用 `--smoke` 跑通，代码写好、测试全部通过；
2. 选一个带 CUDA 的 PyTorch 镜像，按 `requirements.txt` 固定版本；
3. 开机后先跑一次 `--smoke --set device=cuda`，再跑正式配置；
4. 结果（`log.jsonl`、`config.json`、检查点、预测）在关机前拷回本机；
5. 开机前先写下这次最多跑几次、最多用几小时；把费用记进报告。

没有条件租 GPU 时，可以参加 **CPU 赛道**：同样的规则，训练时间改成 30 分钟，写明 CPU 型号。

**交付**：

- `leaderboard.md` 加一行：日期、准确率、显卡（或 CPU）、训练耗时、一句话说明做法、配置文件的位置；
- 报告里一段话：你试过哪些改动，哪些有用、哪些没用，有用的那些为什么有用；附上最终运行的学习曲线。

---

## 10. 交付清单与评分

| 部分 | 问题 | 分数 |
|---|---|---|
| 代码 | data 6、linear 2、conv2d 8、max_pool 2、batchnorm 6、softmax_ce 4、char_cnn 4、adamw 6、schedule 2、clip 2、checkpoint 3、train 8 | 53 |
| 书面 | data_dup 2、conv_accounting 4、adamw_accounting 3 | 9 |
| 实验 | exp_debug 6、exp_lr 6、exp_batch 4、exp_data 6、exp_ablation 6、exp_bn_eval 4 | 32 |
| 排行榜 | leaderboard 6（按要求完成、记录完整、分析清楚即可得满分，不按名次给分） | 6 |
| **合计** | | **100** |

代码部分以测试为准：一个问题对应的测试全部通过，才得这个问题的分。书面和实验部分自己对照"交付"一条条检查：图有没有坐标轴标签和单位？结论有没有标明运行了几次？预测写在运行之前了吗？

完成后回到第 0.11 节的里程碑 A：你写的训练系统能识别单个字符，并且你能从曲线上认出 5 种故障。接下来是第 13 章。A2 在第 18 章之后。
