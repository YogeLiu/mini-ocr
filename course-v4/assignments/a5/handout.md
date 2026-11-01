# A5：从卷积到小公式 OCR

**前置**：第 28–31 章；第 11 章的 GPT（或你的 A2）；第 0 章练习 0.1–0.6（2.4 节说明原因）。**算力**：机制题和冒烟只要 CPU；训练层的设备配置待校准（第 7 节）。**总分**：100。

A5 分三层：机制题（80 分）、书面题（5 分）、训练层（15 分）。你从一个空的包开始，手写卷积的前向和反向、patch 嵌入、位置插值、双向视觉 block、小 ViT、CLIP 损失和连接器。然后用这些零件从随机初始化训练一个小公式 OCR：ViT 空间 token → 连接器 → GPT 视觉前缀。最后在 latex-v1 上评测：val 用来做决定，test 在选型冻结后评。

你要交的东西：

1. `assignments/a5/mechanism/` 里的代码；
2. `assignments/a5/tests/adapters.py` 里的接线；
3. 训练和分析代码（放在 `mechanism/` 外面）、配置、日志和评测结果；
4. 实验报告 `writeup.md`（模板见 `writeup_template.md`）。

第 28–31 章的 tour 用的是参考实现。它们可以用来对照和理解流程，产出的日志和检查点不算 A5 的结果。第 31 章 31.7、31.8 节的数字来自历史运行，只作参照：不是门槛，也不是设备配置。

A5 的结果可以填进里程碑 M2 的交付（第 31 章 31.9 节，lab 第 5 节）：结构图、训练曲线、val 与 test 结果、saliency 分析和 val 错例。报告模板和这份清单一一对应。

## 1. 规则

### 1.1 能用什么、不能用什么

`mechanism/` 是**机制实现部分**（`rules.toml` 里列出），受自动检查（`tests/test_rules.py`），子目录也会检查。

- **可以用**：NumPy；PyTorch 的基础张量运算（`reshape`、`permute`、`matmul`、`einsum`、`logsumexp`、`erf`、`clamp`……）、自动求导（卷积反向例外）、`torch.nn.Module`、`torch.nn.Parameter`、`torch.nn.ModuleList`、`torch.nn.Sequential`、Python 标准库；
- **不能用**：现成的层、卷积、注意力、损失、激活、归一化、插值和优化器。具体包括 `torch.nn` 里的其他类、`torch.nn.functional` 的任何函数、`torch.conv2d`、`unfold`、`torch.softmax`、`torch.log_softmax`、张量的 `.softmax()` 和 `.log_softmax()` 方法、`torch.layer_norm`、`torch.gelu`、`torch.optim` 里的现成优化器；
- **不能 import** `miniocr` 里除 `miniocr.data` 和 `miniocr.device` 之外的任何模块。`miniocr.nn.cnn`、`miniocr.nn.vit`、`miniocr.nn.contrastive`、`miniocr.models.formula_ocr` 都不能进 `mechanism/`；
- **不能 import 其他作业**。`mechanism/` 内部用相对导入。可以复制你以前写过的代码。

LayerNorm、GELU 和稳定的 softmax 都要展开写。线性层也要自己写。

`adapters.py` 只接线，实质的机制写在 `mechanism/` 里。训练层的 `get_ocr` 和 `targets_for` 除外，它们可以接到你自己写的训练层代码（5.2 节）。

**下面这些能通过自动检查，但一样禁用：**

| 写法 | 为什么禁用 |
|---|---|
| `torch.conv1d`、`torch.conv_transpose2d`、`torch.bilinear`、`torch.fft` 拼卷积 | 现成卷积或它的替身 |
| `np.convolve`、`np.lib.stride_tricks.sliding_window_view` | 现成卷积和 unfold 的 NumPy 版本 |
| `torch.relu`、`torch.selu`、`torch.dropout`、`x.relu()` | 现成激活；本作业的结构也用不到 |
| `torch.instance_norm`、`torch.group_norm` | 现成归一化 |
| `torch.grid_sampler`、`torch.affine_grid_generator` | 现成插值 |
| `torch.cosine_similarity`、`torch.pixel_unshuffle`、`torch.distributions` | 现成的相似度、切块和 log-prob |
| M1、M2 里的 `torch.autograd.grad`、`.backward()`、`torch.func` | 卷积的反向要显式写链式法则 |
| `scipy`、`torchvision`、`timm`、`einops`、`transformers` 进 `mechanism/` | 机制题要你自己写，框架只能放在 `mechanism/` 外面 |
| `importlib`、`__import__`、`getattr` 绕过导入检查 | 检查器是语法检查，反射看不出来，规则仍然有效 |
| `adapters.py` 导入 `miniocr.nn`、`miniocr.models` 再转发 | `tests/` 不是受检目录，人工审阅接线是否只转发到你的代码 |

检查器抓不住所有绕过方式：动态导入、别名转移、把实现藏到外部，都要靠人工审阅。规则以本节为准。

**章节代码不能照搬。** 第 29 章的 `patch_embed`、`interpolate_positions`、`VisionBlock`、`TinyViT` 和第 31 章的 `Connector` 调用了 `F` 或 `nn` 里的现成层。章节练习允许这样写，比大作业宽。放进 `mechanism/` 时，你要展开底层运算。第 28 章的 `conv_forward`、`conv_backward` 和第 30 章的 `clip_loss` 本来就手写，你自己做过的练习可以复制过来，再对照本讲义的接口修改。

**框架实验放在 `mechanism/` 外面。** 训练和分析代码放到别的目录（例如 `assignments/a5/training/`）。这个目录不受自动检查，可以用 `torch.optim` 和高效的张量计算，但你要能逐行解释。它可以导入你的 `assignments.a5.mechanism`。调用现成层不算完成机制题。

### 1.2 A5 和章节的差别

| 项 | 章节 | A5 |
|---|---|---|
| patch 投影 | 练习 29.1 有 `method="conv"` 分支 | 没有这个参数，只写手工投影。测试用卷积做独立对拍 |
| 位置插值 | 调 `F.interpolate` | 用张量运算写双三次，结果和 `F.interpolate` 对拍 |
| 非法形状 | 部分输入由底层抛 `RuntimeError` | 一律抛 `ValueError` |
| 第 30 章的范围 | CLIP、SigLIP、双塔训练、R@K | 只测对称 InfoNCE。其余仍是章节内容，A5 不交、不自动评分，也不设门槛 |
| 连接器 | 第 31 章的同宽结构 | 严格采用同宽结构：LN → Linear → GELU → Linear。不要求异宽连接器或其他变体 |
| CLS | 分类 `TinyViT` 有 CLS | M6 的分类 ViT 有 CLS。OCR 的空间编码器没有 CLS，不能把分类 logits 当视觉 token |
| 评测面板 | 第 0 章 0.9.3 节有完整 OCR 面板 | 用第 31–35 章的 OCR 精简版，见 6.2 节。这不算完成完整面板 |
| test 启动记录 | 第 31 章 tour 一次运行一个 `TEST_STARTED.json` | 每个预登记候选一个标记，见第 8 节 |

### 1.3 AI 助手

可以问概念、问报错、让 AI 审阅你的代码、要分级提示。**不要让它替你写 `mechanism/` 里的代码**，写的时候建议关闭 AI 自动补全。报告由你自己写。

### 1.4 数字要诚实

报告里的每个数字都要能在日志里找到：运行 id、配置、设备、日志行、代码位置。只跑过一次的结论要标明“单次”，不能把单种子结论写成普遍事实。没有跑完的栏目写“未运行”，不填推测的结果。CPU 冒烟和章节演示的数字不算训练结果，也不能用来定门槛。

### 1.5 验证集和测试集

规则来自第 0 章 0.10.1 节：

- **验证集用来做决定。** 选超参、选 checkpoint、看错例、提假设，都只用它，可以反复用；
- **测试集是留出集。** 选型冻结以后，预先登记的候选可以各评一次，只报告数字和区间，不根据结果改任何设置；
- **开发期不看测试集的错例。** 最终报告定稿以后才可以解封。A5 的评测脚本不提供解封功能，报告里也不写 test 错例；
- **已经根据测试集改了设置怎么办？** 在报告里写明，再补一份新的、没看过的测试集。

A5 的 test 评测是**必做**：训练门槛定义在 latex-v1 的 test 上。流程见第 8 节。

## 2. 准备

### 2.1 命令

```bash
cd course-v4                                                # 或 make_student 生成的 my-course/
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 python -m pytest -q assignments/a5/tests  # 一开始：规则和部分评测协议测试通过，其余因 NotImplementedError 失败
CUDA_VISIBLE_DEVICES="" python -m pytest -q assignments/a5/tests/test_rules.py      # 只跑规则检查
```

始终不要设置环境变量 `A_SOLUTIONS`。设了以后，测的就不是你的代码。

默认的 `python -m pytest` 不收集 `assignments/`，要写明路径。

测试不下载任何东西，不用 GPU，也不读真实的公式图和 test 数据。

### 2.2 代码结构

建议的结构（不强制）：

```
mechanism/
  conv.py         M1、M2：NumPy 卷积的前向和反向
  layers.py       自己的 Linear、LayerNorm、GELU
  patch.py        M3、M4：patch 嵌入、位置插值
  vit.py          M5、M6：视觉 block、小 ViT
  clip.py         M7：对称 InfoNCE
  connector.py    M8：连接器
  ocr.py          OCR 模型（没有框架依赖时建议放这里，见 8.3 节）
training/         不受自动检查
  prepare.py      准备 train、val 缓存和泄漏检查
  train.py        训练和保存
  analyze.py      val 上的视觉依赖对照、saliency 和错例
```

机制题按行为评测，不要求类名和别人一样。接口的函数名和参数名必须和 `tests/adapters.py` 一致。模块的参数名也必须一致，每题写在下面。

### 2.3 统一约定

- 形状不匹配、几何非法、非正整数配置都抛 `ValueError`；
- M3–M8 保留输入的 dtype。测试用 float64 做数值梯度检查，模块要能 `.double()`；
- M1、M2 用 NumPy float64；
- 不就地修改输入；
- 每题的完整契约写在 `tests/adapters.py` 的文档字符串和本讲义里。

### 2.4 数据

**主任务只用开源公式的原图。** 数据是 `latex-v1` 划分下的本地缓存：来源是 OleehyO/latex-formulas 的 `cleaned_formulas`，许可 openrail，公式来自 arXiv 和 im2latex-100K（第 31 章 31.2.6 节，lab 第 0 节）。只在本地学习时使用，不分发原始数据、转换后的数据和权重。本作业不下载。

合成渲染数据只做 CPU 冒烟，或者作为单独标注的可选消融。它的结果不能和开源数据混报（第 31 章 31.2.7 节）。

```python
from miniocr.data.contrastive_pairs import prepare_pairs
from miniocr.data.latex_corpus import corpus_available

if not corpus_available():
    raise SystemExit("缺本地语料")
data = prepare_pairs(DATA_DIR, {"train": TRAIN_N, "val": VAL_N}, source="open", max_length=MAX_LENGTH)
train_keys = {r["key_sha256"] for r in data["train"]["manifest"]}
val_keys = {r["key_sha256"] for r in data["val"]["manifest"]}
assert not train_keys & val_keys
```

每个划分是一个字典：`images`（N, 1, 32, 160）、`ids`、`manifest`（`id`、`formula`、`key_sha256`、`split`）、`source`、`split`、`max_length`。

- **先划分，再取前缀。** 完整语料先按规范化键分组划分，再筛长度，再按 parquet 的固定顺序取前 N 条（第 31 章 31.2.3 节）。不要自己先抽样；
- **这是短公式子集。** 只保留非空、UTF-8 长度不超过 `max_length` 的公式。它不代表完整公式分布，也不代表 TexPixel（31.2.6 节）；
- **图像预处理固定。** 裁白边、保宽高比双线性缩放、留边、补 0，得到 32×160 的墨迹画布（31.2.3 节）；
- **缓存和配置绑定。** 同一个目录的条数、来源、`max_length` 不能变。要改就换新目录，否则 `prepare_pairs` 抛 `ValueError`；
- **不要自己传 `allow_test=True`。** test 图片只由第 8 节的入口解码；
- **`make_student` 目录。** 划分依赖第 0 章练习 0.1–0.4，评分依赖练习 0.1–0.6。没做完会抛 `NotImplementedError`。先让 `python -m pytest -q book/ch00_intro` 里对应的测试通过。

### 2.5 自动测试管什么

自动测试用手算数字、固定随机种子、独立的 PyTorch 算子和 float64 数值梯度。没有二进制 fixtures。测试里用到的 `F.conv2d`、`F.interpolate`、`F.scaled_dot_product_attention`、`F.cross_entropy` 只做对拍，你的实现不能调用它们。

| 内容 | 怎样检查 |
|---|---|
| 卷积前向和反向 | 手算值、`F.conv2d` 对拍、每个坐标的中心差分、缓存快照 |
| patch 嵌入 | 手算值、步长为 p 的卷积对拍、`gradcheck` |
| 位置插值 | `F.interpolate` 对拍、常数表、CLS 不变、对象身份 |
| 视觉 block、小 ViT | 独立算子组成的对照实现、全部参数和输入的梯度、置换等变、`state_dict` 往返 |
| CLIP | 手算值、`F.cross_entropy` 对拍、平移、转置、配对置换不变、极端值 |
| 连接器 | 对照实现、逐 token 独立、手算输出 |
| `targets_for`、`get_ocr` | 手算编码和 mask、loss 和 CE 一致、全部参数有有限梯度、因果性、图像依赖、往返 |
| 评测门禁 | 假 checkpoint 字节，不加载任何 test 数据 |
| 规则 | 语法树检查，抓不住反射 |
| 训练质量、门槛、saliency、错例分析 | **报告审查**，自动测试不管 |

测试通过不等于训练层合格。

## 3. 机制题

### 问题（A5-M1）：卷积前向（8 分）

实现 `conv_forward(x, weight, bias, stride=1, padding=0)`，返回 `(y, cache)`。第 28 章 28.2–28.4 节，练习 28.2。

- NumPy float64。`x` 是 NCHW，`weight` 是 OIHW，`bias` 是 (O,)；
- 做互相关，不翻转核（第 28 章 28.2.1 节）；
- 核可以是矩形。`stride`、`padding` 是标量整数，只对 H、W 补 0；
- 输出尺寸分别是 $\lfloor(\text{size}+2p-k)/s\rfloor+1$（28.3.2 节）；
- `cache` 要保留输入和权重的快照。调用者后面原地修改 `x`、`weight`，反向结果不能变；
- 通道数不对、偏置形状不对、步长为 0、padding 为负、核超过补过的输入，都抛 `ValueError`。

手算例子：1…9 排成 3×3 输入，2×2 全 1 核，偏置 0.5，输出是 [[12.5, 16.5], [24.5, 28.5]]。

测试还用多样本、多通道、矩形核和 `F.conv2d` 对拍，几何取 `(stride, padding)` = (1, 0)、(2, 1)、(3, 2)。

自查（不计分）：H = 3、k = 2、s = 3、p = 2 时，输出高是多少？

### 问题（A5-M2）：卷积反向（12 分）

实现 `conv_backward(dout, cache)`，返回 `(dx, dw, db)`。第 28 章 28.5 节，练习 28.3。

- 不能用现成卷积、`unfold` 和自动求导。显式写链式法则；
- `dx` 在窗口重叠处累加，再裁掉 padding。p = 0 时也要正确；
- `dout` 和 `y` 形状不同抛 `ValueError`。

手算例子（M1 的数，上游梯度全 1）：

| 量 | 值 |
|---|---|
| `dx` | [[1, 2, 1], [2, 4, 2], [1, 2, 1]] |
| `dw` | [[12, 16], [24, 28]] |
| `db` | [4] |

测试还在 M1 的三种几何下，对 `x`、`weight`、`bias` 的每个坐标做 float64 中心差分（eps = 1e-5），并和 PyTorch 自动求导对拍。

自查（不计分）：补的那圈 0 为什么要从 `dx` 裁掉？

### 问题（A5-M3）：patch 嵌入（10 分）

实现三个函数。第 29 章 29.2 节，练习 29.1。

- `patch_grid(height, width, patch, pad=False)` 返回 `(gh, gw)`。`pad=False` 必须整除；`pad=True` 只算向上取整的网格，不裁切也不改图。尺寸和 patch 都是正整数；
- `patchify(x, patch)`：NCHW → (B, gh·gw, C·p·p)。网格行优先，patch 内按 (C, pH, pW)。只接受完整 patch；
- `patch_embed(x, weight, bias=None)`：`weight` 是 (D, C, p, p)，`bias` 是 (D,)，输出 (B, N, D)。`patchify` 后乘 `weight.flatten(1).T`。

手算例子：

- `x` 是 1…16 排成的 (1, 2, 2, 4)，`patchify(x, 2)` 是 [[1, 2, 5, 6, 9, 10, 13, 14], [3, 4, 7, 8, 11, 12, 15, 16]]。这个例子区分通道顺序；
- 取第一通道，核 [[0, 1], [0, 0]]，偏置 0.5，`patch_embed` 输出 [[2.5], [4.5]]。这个核不对称，能抓住转置错误；
- `patch_grid(5, 7, 2, True)` 是 (3, 4)。`patch_grid(3, 4, 2)` 抛 `ValueError`。

错误的维数、不能整除、通道不对、矩形权重、偏置形状不对，都抛 `ValueError`。

测试还和步长为 p 的卷积对拍前向和梯度，并做 `gradcheck`。

自查（不计分）：练习 29-H2 的核 [[1, 0], [0, 1]] 为什么查不出行列顺序错误？

### 问题（A5-M4）：二维位置插值（10 分）

实现 `interpolate_positions(pos, old_grid, new_grid, has_cls=True)`。第 29 章 29.5 节，练习 29.2。

- `pos` 是 (1, old_h·old_w + CLS, D)。新旧网格都显式传两个正整数，不从 token 数开平方；
- CLS 不参与插值，原样拼回；
- 同网格返回**原张量对象**；
- 不能调用 `F.interpolate`，用张量运算写。

空间部分是双三次，`align_corners=False`，不启用 antialias。沿一个轴：

- 目标坐标 $d$ 的源坐标是 $s=(d+0.5)\cdot\text{old}/\text{new}-0.5$；
- 取 $\lfloor s\rfloor-1,\dots,\lfloor s\rfloor+2$ 四个源格，下标越界时取边缘值；
- 权重是下面的核在 $t=s-\text{源格下标}$ 处的值，$a=-0.75$：

$$W(t)=\begin{cases}(a+2)|t|^3-(a+3)|t|^2+1, & |t|\le1\\ a|t|^3-5a|t|^2+8a|t|-4a, & 1<|t|<2\\ 0, & \text{其他}\end{cases}$$

两个轴的权重相乘，对 16 个源格求和。

手算例子：常数表恒为 3、CLS 为 99，2×3 插值到 4×2，空间部分仍然全是 3，CLS 仍是 99。

错误的表长和非正的网格抛 `ValueError`。

测试还在 2×3 → 3×4、含和不含 CLS 时，和 `F.interpolate` 对拍前向和 `pos` 的梯度，并做 `gradcheck`。

自查（不计分）：新网格 8×18 时，用 √ 推网格会发生什么？（29-D3）

### 问题（A5-M5）：双向视觉 block（10 分）

实现 `get_vision_block(width, heads=4)`，返回 `nn.Module`，输入输出都是 (B, T, D)。第 29 章 29.7 节，练习 29.3。

结构（无 dropout）：

1. pre-LN（eps = 1e-5）→ QKV → 双向 scaled dot-product attention → `proj` → 残差；
2. pre-LN → Linear(D, 4D) → 精确 GELU → Linear(4D, D) → 残差。

- LayerNorm 对最后一维用总体方差，$(x-\mu)/\sqrt{\sigma^2+\varepsilon}$，再乘 γ 加 β；
- 精确 GELU 是 $\frac{x}{2}\bigl(1+\mathrm{erf}(x/\sqrt2)\bigr)$；
- `qkv` 输出按 Q、K、V 顺序分组，每头 D/heads 维。分数除以 $\sqrt{D/\text{heads}}$；
- 没有任何 mask；
- softmax 要展开。`scores - scores.logsumexp(-1, keepdim=True)` 再 `exp` 是稳定的写法。

参数名必须是（线性权重的形状是 (out, in)）：

| 参数 | 含义 |
|---|---|
| `norm1.weight`、`norm1.bias`、`norm2.weight`、`norm2.bias` | LN 的 γ、β |
| `qkv.weight`、`qkv.bias` | (3D, D)、(3D,) |
| `proj.weight`、`proj.bias` | (D, D)、(D,) |
| `mlp.0.weight`、`mlp.0.bias` | (4D, D)、(4D,) |
| `mlp.2.weight`、`mlp.2.bias` | (D, 4D)、(D,) |

宽度或头数非正、不能整除，都抛 `ValueError`。

手算例子：全部参数清零时，输出等于输入。

测试还和独立算子组成的对照实现比较输出、输入和每个参数的梯度，并检查 token 置换等变、改动最后一个 token 会影响第一个 token。

自查（不计分）：误加因果 mask 时，置换等变测试为什么会失败？（29-D4）

### 问题（A5-M6）：小 ViT（10 分）

实现 `get_vit(cfg)`，返回 `nn.Module`。第 29 章 29.1.3、29.6 节，练习 29.3。

- `cfg` 的键是 `image_size`、`patch`、`width`、`depth`、`heads`、`classes`，都是正整数。初始位置网格是正方形，`image_size` 要能被 `patch` 整除；
- 输入是灰度图，可以是能整除的矩形，分辨率可变；
- 流程：patch 嵌入 → 拼 CLS → 加插值后的位置表 → `depth` 个 block → 最终 LN → 取 CLS → 线性分类头。输出未归一化的 logits (B, classes)；
- 初始化不计分，但模型要能训练。

参数名必须是：

| 参数 | 形状 |
|---|---|
| `weight` | (D, 1, p, p) |
| `bias` | (D,) |
| `cls` | (1, 1, D) |
| `pos` | (1, grid² + 1, D) |
| `blocks.i.<M5 的键>` | 同 M5 |
| `norm.weight`、`norm.bias` | (D,) |
| `head.weight`、`head.bias` | (classes, D)、(classes,) |

零深度、通道不是 1、尺寸不能整除，都抛 `ValueError`。

手算例子：全部参数清零，再把 `head.bias` 设成 [1, 2, 3]，输出就是这个向量。

测试还用 4×4 初始网格、4×6 输入的完整对照实现，比较全部参数和输入的梯度，并做 `gradcheck` 和 `state_dict` 往返。

自查（不计分）：手算例子里，为什么只有 `head.bias` 留在输出里？

### 问题（A5-M7）：CLIP 损失（10 分）

实现 `clip_loss(scores)`，返回标量。第 30 章 30.3、30.4 节，练习 30.1。

- `scores` 是已经除过温度的非空 B×B 浮点矩阵，正例在对角线。一批里的规范化键要互不相同，这是数据的责任；
- 公式（30.3.3 节）：

$$L=\frac12\Bigl[\frac1B\sum_i\Bigl(\mathrm{logsumexp}_j S_{ij}-S_{ii}\Bigr)+\frac1B\sum_j\Bigl(\mathrm{logsumexp}_i S_{ij}-S_{jj}\Bigr)\Bigr]$$

- 可以用 `logsumexp`。不能调用 `cross_entropy`、`softmax`、`log_softmax`；
- 空矩阵、非方阵、维数不对，都抛 `ValueError`。

手算例子：全零分数、B = 4 时 $L=\ln4$，梯度是 $(\tfrac14-I)/4$。B = 1 时 L = 0。

测试还用非对称随机矩阵和两个方向的 `F.cross_entropy` 对拍前向和梯度，并做 `gradcheck`。整体加 10000、转置、配对同时置换，损失都不变。分数到 ±1000 时，损失和梯度仍然有限。

自查（不计分）：为什么所有分数相同时 $L=\ln B$？

### 问题（A5-M8）：连接器（10 分）

实现 `get_connector(width)`，返回 `nn.Module`。第 31 章 31.3 节，练习 31.1。

- 输入输出都是 (B, P, D)：LayerNorm（eps = 1e-5）→ Linear(D, D) → 精确 GELU → Linear(D, D)；
- 不池化，不打乱 token；
- 参数名：`net.0.weight`、`net.0.bias`（LN），`net.1.weight`、`net.1.bias`、`net.3.weight`、`net.3.bias`（线性）；
- 宽度非正、输入维度不对，都抛 `ValueError`。

手算例子：全部参数清零，再把 `net.3.bias` 设成 [1, 2, 3, 4]，每个 token 的输出都是这个向量。

测试还和对照实现比较前向、全部参数和输入的梯度，并检查改动一个 token 不影响其他 token。

自查（不计分）：改动一个 token 为什么不影响其他 token？

## 4. 书面题

每题 1 分。

### 问题（A5-W1）：卷积的梯度为什么累加（1 分）

卷积的 `dx` 为什么要累加？3×3 输入上的 2×2 核，中心像素被几个窗口使用？写出每个像素被覆盖次数组成的矩阵。第 28 章 28.5.1–28.5.3 节。

### 问题（A5-W2）：patch 大小和预算（1 分）

32×160 的图，patch 从 8 改成 4。空间 token 数怎样变？不含 CLS 的视觉自注意力分数元素数怎样变？加上文字以后，总序列的分数还能说恰好 16 倍吗？用 max_length = 64 算一个数。第 29 章 29.3.1 节，第 31 章 31.3.3 节。

### 问题（A5-W3）：CLS 和因果 mask（1 分）

为什么不对 CLS 插值？为什么不能给视觉 block 用因果 mask？OCR 为什么用全部空间 token，不用 CLS？第 29 章 29.5.4、29.6、29.7.1 节，练习 29-C3、29-C8，第 31 章 31.3 节。

### 问题（A5-W4）：CLIP 的导数和温度（1 分）

写出 CLIP 损失对 $S_{ij}$ 的导数。$S=C/\tau$ 时，对 $C_{ij}$ 的导数是什么？温度改变什么，不改变什么？检索模型为什么写不出图库外的 LaTeX？第 30 章 30.4.3、30.5、30.12.2 节，概念题 30.4（c）。

### 问题（A5-W5）：连接器和 saliency（1 分）

连接器为什么不做全局池化？saliency 能不能证明因果依赖？要证明模型依赖图像，还要看什么？这些证据的边界在哪里？第 31 章 31.3、31.8.3、31.8.6 节。

## 5. 训练层任务

### 问题（A5-T1）：小公式 OCR 的训练、对照和验收（15 分）

| 子项 | 内容 | 分 |
|---|---|---|
| T1-a | 接线、数据审计、长训练前的检查（5.1–5.3 节） | 5 |
| T1-b | 完整曲线、val 选型、消融和成本（5.4、5.5 节，第 6 节） | 5 |
| T1-c | val 上的 saliency、错例和视觉依赖对照（5.6 节）；门槛冻结、test 评测和人工验收（第 7、8 节） | 5 |

**任务目标**：从随机初始化训练小公式 OCR，门槛定义在 latex-v1 的 test 上。门槛还没有校准，状态见第 7 节。

本作业不提供完整训练程序。

### 5.1 模型和数据边界

- **结构**：ViT 空间 token → 连接器 → GPT 视觉前缀。文字位置能看到全部视觉 token（第 31 章 31.3 节）。全部参数随机初始化，没有预训练；
- **复用**：视觉 block 和连接器用你自己的机制实现。GPT 可以复制你之前写的代码（A2），不能跨作业 import，也不能接 `miniocr.models` 或 `miniocr.nn` 的参考实现；
- **高效替代**：训练层可以用框架优化器和高效的张量计算。如果把某个零件换成高效实现，报告要写明替换范围，并附和机制版本的数值对拍；
- **可选消融**：交叉注意力接口可以另做（第 31 章 31.3.1 节）。主接口是前缀拼接；
- **数据**：train 只用来训练，val 只用来做决定。主训练集来自 2.4 节的开源原图。

### 5.2 训练层接口

**`targets_for(texts, max_length)`** 返回 `(inputs, labels)`，形状都是 (B, max_length + 1)。第 31 章 31.2.3、31.4 节。

- token 是 UTF-8 字节 + 1。PAD = 0，BOS = 257，EOS = 258，词表 259；
- `inputs` = BOS + 字节，右补 0。`labels` = 字节 + EOS，右补 −100；
- 空串和超长输入抛 `ValueError`，不能截断。

手算例子，`targets_for(['x', 'é'], 3)`：

| | 第 1 条 | 第 2 条 |
|---|---|---|
| `inputs` | [257, 121, 0, 0] | [257, 196, 170, 0] |
| `labels` | [121, 258, −100, −100] | [196, 170, 258, −100] |

**`get_ocr(cfg)`** 返回模型。第 31 章 31.3–31.5 节。

- `cfg` 的键是 `width`、`vision_depth`、`decoder_depth`、`patch`、`max_length`。`model.config` 保存这些键；
- 画布固定 32×160，视觉 token 数 P = (32/patch)·(160/patch)；
- `forward(images, inputs, labels=None)` 返回 `(文字位置的 logits, loss)`。logits 是 (B, max_length + 1, 259)。视觉前缀的标签是 −100，loss 等于有效文字 token 的交叉熵；
- `generate(images)` 贪心输出字符串列表。禁止生成 PAD 和 BOS，遇到 EOS 停，最多 max_length + 1 步。不做提示抽取，也不做后处理；
- `cfg` 里没有 `heads`。第 31 章的结构两侧都是 4 头，而且要求 `width` 是 8 的倍数，这样 RoPE 的每头维度是偶数。你的 GPT 有别的约束时，写进报告；
- 测试会用 `width=8`、`patch=16` 的小配置，你的模型要接受它。

**检查点格式**固定为：

```python
torch.save({"cfg": cfg, "state_dict": model.state_dict()}, path)
```

评测脚本用 `get_ocr(ckpt["cfg"])` 重建模型。消融要多传一个开关时，`get_ocr` 可以接受带默认值的可选键，把它写进 `cfg`。

训练接口的自动测试只查编码、mask、形状、因果性、图像依赖和往返。它们不检验训练质量。

### 5.3 长训练之前的检查

先用真实的 train 样本走一遍 前向 → 反向 → 更新 → 保存 → 加载，通过了再开长训练。报告这些：

- **第一步 loss**：随机初始化时应接近 ln 259 ≈ 5.557（第 31 章 31.4 节，练习 31.2）；
- **梯度**：所有参数都有有限梯度，连接器和 patch 权重的梯度不为 0（31.3.6 节）；
- **mask**：视觉前缀和补齐不计入 loss，有效标签数和手算一致；
- **保存重载**：同一批 val 上逐条比较，记录不同条数；
- **小数据过拟合**：极少几条 train 样本能学到按图输出；换图输出会变（练习 31.2 的 `test_learns_two_image_conditioned_examples`）；
- **泄漏检查**：train 和 val 的 `key_sha256` 没有交集。这里不要碰 test；
- **冒烟**：6.3 节。

这段的数字不写进训练结果。

### 5.4 训练和选型

- **规则先写**：训练前写定 val 选型规则。建议用固定 val 前缀上的最小 CER，并列时取较早的步（第 31 章 31.6.3 节）。选型子集和最终 val 要分清（31.6.6 节）；
- **曲线**：记录每步 loss、学习率、梯度范数，以及每次评测的 val 指标。把训练 loss 和 val CER 画在一起（31.7.6 节）；
- **冻结**：选中的 checkpoint 在完整 val 上评一次，记下 SHA256；
- **解释**：训练样本数和平均遍数一起写（31.7.3 节）。写明种子数，单种子不能当普遍事实；
- **成本**：实测设备、精度、峰值内存、耗时、训练步数。章节的耗时和显存不能抄。

### 5.5 消融

patch 预算（第 29 章 29.3 节，第 31 章 31.7.4 节）和连接器（第 31 章 31.3.2 节）至少做一项。只用 val 比较，写明同时变化的因素。消融的结论只在你的配置下成立。

要把消融登记成 test 候选，它和主候选的 `max_length` 必须相同（8.2 节）。

### 5.6 只在 val 上做的分析

下面都只用 val（第 31 章 31.8 节，lab 第 4 节）。这些分析你要自己写，评测脚本不输出。可以用 `assignments.a5.evaluate.score(refs, preds, "val")` 算同一个面板。

- **视觉依赖对照**：同一 checkpoint、同一贪心解码，比较原图、中央遮挡、循环错配、全白图（31.8.3、31.8.4 节）。先推导再解读：确定性模型下，错配的 parse_rate 必然和原图相同；
- **saliency**：实现目标 token 的 log 概率对输入像素的梯度（练习 31.3），不能在参数上留下 `.grad`。至少取第一个字节和 EOS 两个位置。这是给定正确前文下的局部导数，不是因果证据；
- **错例表**：按 31.8.6 节的顺序分类：标签不可见、解码失控、视觉混淆、结构错误、说不清。每类附样本数和例子；
- **归因限制**：遮挡、错配、白图是分布外输入。它们能说明模型整体依赖图像，不能逐条归因，也不能单独评价编码器的质量。

禁止在开发期看 test 错例，也禁止根据 test 重新选模型。

## 6. 统一评测

### 6.1 验证集评测

```bash
CUDA_VISIBLE_DEVICES="" python -m assignments.a5.evaluate --ckpt /tmp/my-a5/model.pt --data-dir /tmp/my-a5/data --n 20 --device cpu \
    > /tmp/my-a5/val.json
```

- 默认只读 val。没有缓存时，它用本地语料生成 `val.pt`；
- 结果是一行 JSON，写到 stdout，自己保存成开发日志；
- `--n` 默认 20，`--batch-size` 默认 4。这两个默认值只为调用方便，不是训练配置。正式比较时固定一个 `--n`，同一个 `--data-dir` 不能换条数；
- 用你自己的 `get_ocr`。还没接线时报 `NotImplementedError`；
- 路径不能在 `outputs/` 下（含符号链接解析后的位置），脚本会拒绝。运行数据和结果不进教材仓库；
- 评测固定 fp32，没有 autocast。你混合精度训练时，在报告里写明最终评测的精度，不能根据 test 切换。

### 6.2 输出字段

主面板在 `metrics` 里，是第 0 章 0.9.3 节的 OCR 精简版：

| 字段 | 面板指标 | 口径 |
|---|---|---|
| `em_norm` | EM_norm | 规范化后完全匹配，附成功数、分母和 Wilson 95% 区间 |
| `cer` | CER | 规范化字符的语料级距离，1000 次固定 seed = 0 的 bootstrap 区间；可能大于 1 |
| `parse_rate` | parse_rate | mathtext 子集可解析率，Wilson 区间。不是完整 LaTeX 编译率，也不说明对错 |
| `em_raw` | EM_raw | 不规范化的完全匹配，并列参考 |
| `n` | 样本数 | |

外层还有 `seconds`、`device`、`torch`、`python`、`checkpoint_sha256`、`code_sha256`、`split`、`seed`、`utc`。`threshold` 固定是 `待校准`。

脚本不输出 CER_token 和渲染一致率，报告也不能把缺项写成 0。区间只描述当前子集的抽样波动，不涵盖训练种子、数据选择和设备差异（第 31 章 31.6.3 节）。峰值内存脚本不记录，要自己量。

### 6.3 CPU 冒烟

```bash
CUDA_VISIBLE_DEVICES="" python -m assignments.a5.evaluate --smoke --run-dir /tmp/fresh-a5
```

要求 `--run-dir` 是新目录。它用渲染的 2 条 train、2 条 val 做一次更新，再做保存、加载、生成。它永远不读 test。

配置 `width=8`、`depth=1`、`patch=16`、`max_length=16` 是程序自检的常量，不是教学建议。冒烟的结果不是训练结果，也不能用来定门槛。没接好 `get_ocr` 和 `targets_for` 时报 `NotImplementedError`。

## 7. 门槛与设备配置

下表全部待校准。A5 没有可以直接启动长训练的预设。每套配置要另存独立的配置文件和运行 id。

| 项 | CUDA | MPS | CPU |
|---|---|---|---|
| `width`、`vision_depth`、`decoder_depth`、`patch`、`max_length` | 待校准 | 待校准 | 待校准 |
| batch、梯度累积、训练步数、学习率、warmup、衰减、裁剪 | 待校准 | 待校准 | 待校准 |
| train_n、val_n、test_n、seed、val 频率、精度 | 待校准 | 待校准 | 待校准 |
| 时间、峰值内存、训练效果 | 待实测，不预填 | 待实测，不预填 | 待实测，不预填 |
| EM_norm、CER 等门槛 | 待校准 | 待校准 | 待校准（可以适度放宽，事先记录） |

第 31 章某次 GPU 实验的配置不能当成这三套配置，它的数字也不能当门槛。

门槛由你第一次完整训练后确定：

- 定门槛**只看验证集结果**，并且**写在 test 评测之前**。看了 test 再定门槛，等于用 test 做决定；
- 冒烟和章节演示的数字不能作为校准依据；
- 门槛写明指标、方向、数值、用哪套配置、依据的运行 id 和日期；
- 门槛只对主候选（8.1 节）判定，不能看完 test 再换一个过线的候选；
- 保留校准配置、曲线和理由；
- 还没做完整训练时，报告里就写“待校准”，训练层写“未验收”；
- 运行成功不等于通过。主候选没过线就写没过线，不事后改门槛。要再试，先在报告里写明，再补一份新的、没看过的留出集（第 0 章 0.10.1 节）。

## 8. 冻结与最终测试（必做）

### 8.1 冻结之前

全部选型在 val 上做完以后，依次做这几件事：

1. 定候选：写下要登记的候选和登记理由，指定一个**主候选**；
2. 写门槛：把门槛写进报告第 1 节，记下日期。`FROZEN.json` 不记录门槛，所以再存一份门槛文件，记下它的 SHA256；
3. 确认缓存：`train.pt`、`val.pt` 已经在 `data_dir` 里。冻结后它们不能变。冻结时还没有的 `val.pt`，后面再生成也会让启动被拒绝；
4. 试运行：每个候选用计划里的 `device`、`batch_size` 跑一次 val。test 启动后任何失败都占用次数（8.4 节）。

### 8.2 写计划

```json
{"candidates": {"base": "/absolute/base.pt", "ablation": "/absolute/ablation.pt"},
 "data_dir": "/absolute/private-cache", "test_n": 500, "device": "cpu", "batch_size": 8, "seed": 0}
```

这里的数量只展示文件格式，正式数量待校准。

- 候选名只用字母、数字、下划线和连字符。每个候选要有自己的检查点文件，内容相同的别名会被拒绝；
- **所有候选的 `max_length` 必须相同。** 冻结时会用 CPU 读取各检查点的 `cfg.max_length`，检查它们是相同的正整数，并将该值写入 `FROZEN.json`。不一致会立即报错，不生成冻结记录，也不占用任何候选的次数；
- `test_n` 取保守的值。短公式筛选后凑不够也会失败。不要为了查条数去统计 test。

### 8.3 冻结

```bash
python -m assignments.a5.evaluate --freeze-plan plan.json --run-dir /absolute/private-final
```

它独占创建 `FROZEN.json` 和 `FROZEN.sha256.json`，登记候选名、路径、checkpoint SHA256、评测代码 SHA256、train 和 val 缓存的哈希、源 parquet 的路径、大小和修改时间、精度、种子和解码规则。已有的冻结文件不能覆盖。整份报告共用一个 final 目录。

**代码哈希的范围。** `code_sha256` 覆盖 `evaluate.py`、`mechanism/`、`tests/adapters.py` 和 `miniocr/`，以及 `assignments/a5/training/` 下递归的所有 `.py` 源码（包括相对路径），参考模式也覆盖学习者的训练源码。冻结和结果记录中的 `training_code` 列出目录状态与逐文件 SHA256：目录不存在时允许继续，明确记为 `missing`、文件表为空；存在但无 Python 文件记为 `present`、文件表为空。冻结后新增、删除、重命名或修改训练源码，以及目录从缺失变为存在，都会改变 `code_sha256`，在启动前拒绝且不占次数。非 Python 文件及目录外依赖仍须在报告中单独记录。

**数据的范围。** parquet 只绑定元数据，不算内容哈希。这是效率和审计强度的折中。报告仍要写数据来源、快照和许可。

### 8.4 逐个评测

```bash
python -m assignments.a5.evaluate --run-dir /absolute/private-final --candidate base --split test --final-test
python -m assignments.a5.evaluate --run-dir /absolute/private-final --candidate ablation --split test --final-test
```

- test 用登记的配置，不用命令行的 `--n`、`--device` 等开发参数；
- 开关缺失、候选没登记、代码、权重或 train、val 缓存变了，都拒绝。这些拒绝发生在启动之前，不占次数；
- 核对通过后，脚本先独占创建 `TEST_STARTED_<checkpoint_sha256>.json`，再加载模型、读 test。**失败也占用次数**，模型加载失败、显存不够、凑不够条数都算。同一候选不能重试，并发启动也会被拒绝；
- 其他预登记候选仍可各评一次；
- 第一次生成的 `test.pt` 的哈希写进 `TEST_DATA.sha256.json`。后面的候选必须用同一份缓存；
- 成功时写 `TEST_RESULT_<checkpoint_sha256>.json`，只有聚合数字和区间。没有逐条预测、错例表、test saliency 和分桶；
- 真的遇到硬件故障，在报告里写明事件，不悄悄重跑。

门禁只防手滑。手工删标记、另建 run 目录、同时改冻结文件和摘要，都能绕过。所以规定：不删记录，不换目录重评，不按 test 改设置。发现违规，按第 0 章 0.10.1 节在报告里写明，再换新的留出集。

## 9. 交付清单与评分

| 项 | 内容 | 分 |
|---|---|---|
| 机制题 | 测试全部通过（含规则测试）：A5-M1 8、A5-M2 12、A5-M3 10、A5-M4 10、A5-M5 10、A5-M6 10、A5-M7 10、A5-M8 10 | 80 |
| 书面题 | A5-W1 到 A5-W5，每题 1 分 | 5 |
| T1-a | `get_ocr`、`targets_for` 接线和训练代码；数据来源、划分和泄漏检查；长训练前的检查；保存重载 | 5 |
| T1-b | 结构图和形状表；训练和 val 曲线、运行 id；val 选型规则和面板；消融；实测的时间和内存 | 5 |
| T1-c | 只来自 val 的 saliency、错例和视觉依赖对照；门槛和校准依据的冻结记录；预登记候选和启动记录；test 聚合面板；人工验收。运行成功不自动给分，未校准就是“未验收” | 5 |
| 合计 | | 100 |

要提交的报告内容以 `writeup_template.md` 为准。还要提交 `FROZEN.json`、`FROZEN.sha256.json`、`TEST_DATA.sha256.json`、每个候选的 `TEST_STARTED_*.json` 和 `TEST_RESULT_*.json`、开发日志和代码版本，并声明没有根据 test 改设置。缺这些文件，T1-c 按不可追溯扣分。

权重和数据不用放进仓库，留在本地，登记的哈希要对得上。

测试通过不等于学术诚信的证明。算法是否放在 `mechanism/`、`adapters.py` 是否只有接线、报告结论是否受 test 影响，都要人工审阅。
