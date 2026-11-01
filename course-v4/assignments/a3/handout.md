# A3：SFT 与 LoRA，训练 OCR 后纠错模型

**前置**：第 18–21 章；第 0 章练习 0.1–0.6（2.4 节说明原因）。**算力**：机制题和冒烟只要 CPU；训练层的设备配置待校准（第 7 节）。**总分**：100。

A3 分三层：机制题（60 分）、书面题（20 分）、训练层（20 分）。你从一个空的包开始，自己写 SFT 的监督信号和 LoRA 层，再把它们用到真实模型上，训练一个 OCR 后纠错模型：输入带噪的 LaTeX，输出干净的 LaTeX。最后用统一脚本和提示基线、照抄比较。

你要交的东西：

1. `assignments/a3/mechanism/` 里的代码；
2. `assignments/a3/tests/adapters.py` 里的接线；
3. 训练和预测代码（放在 `mechanism/` 外面）、评测配置、日志和评测结果；
4. 实验报告 `writeup.md`（模板见 `writeup_template.md`）。

第 19–21 章的 tour 用的是参考实现。它们可以用来对照和理解流程，产出的日志和检查点不算 A3 的结果。第 21 章 21.10.3 节的观察来自一次历史运行，只作参照：不是门槛，也不能代替你的结果。

A3 的结果可以填进项目报告 R2（第 19 章 19.8.4 节，第 21 章 21.10.4 节）。填的时候写明：这里的噪声是文本上合成的，不能证明真实图像 OCR 也能泛化（第 18 章 18.1.5 节）。

## 1. 规则

### 1.1 能用什么、不能用什么

`mechanism/` 是**机制实现部分**（`rules.toml` 里列出），受自动检查（`tests/test_rules.py`），子目录也会检查。

- **可以用**：PyTorch 的张量运算（`@`、`exp`、`logsumexp`、`amax`、`gather`、`uniform_`……）、自动求导、`torch.nn.Module`、`torch.nn.Parameter`、`torch.nn.ModuleList`、`torch.nn.ModuleDict`、`torch.nn.Sequential`、`torch.optim.Optimizer`（作为基类）、Python 标准库；
- **不能用**：`torch.nn` 里的现成层（`Linear`、`Embedding`……）、`torch.nn.functional` 的任何函数、`torch.softmax`、`torch.log_softmax`、张量的 `.softmax()`、`.log_softmax()` 方法、`torch.optim` 里的现成优化器、`torch.nn.init`、`torch.nn.utils.clip_grad_norm_`；
- **不能 import** `miniocr` 里除 `miniocr.data` 和 `miniocr.device` 之外的任何模块。`miniocr.prompting` 也不能进 `mechanism/`，模板调用要自己写；
- **不能 import 其他作业**。`mechanism/` 内部用相对导入。可以复制你以前写过的代码。

`adapters.py` 只接线，算法都写在 `mechanism/` 里。检查器用语法树查找，抓不住所有绕过方式；规则以本节为准，`adapters.py` 里有没有藏算法由人工审阅。

**章节代码不能照搬。** 第 19 章的 `causal_loss` 用了 `F.cross_entropy`。第 20 章的 `LoRALinear` 用了 `nn.Linear`、`F.linear`、`nn.init.kaiming_uniform_`，`merged` 还返回 `nn.Linear`。这些都不能进 `mechanism/`。所以 A3 的接口和章节不同：`get_lora` 接收权重张量，`merged()` 返回你自己写的层。章节函数对一些非法输入不检查，或者抛其他异常；A3 按 `adapters.py` 的契约抛 `ValueError`。

**数据工具有一部分不能用。** `miniocr.data.sft` 能通过 import 检查。其中 `SFTExample` 和 `IGNORE_INDEX` 可以用，`encode_sft` 和 `collate_sft` 是参考答案，不能调用。

**训练放在 `mechanism/` 外面。** 训练和预测代码放到别的目录（例如 `assignments/a3/training/`），可以用 transformers、PEFT、TRL，这个目录不受自动检查，但你要能逐行解释。它可以导入你的 `assignments.a3.mechanism`。调用 TRL 不算完成机制题。

### 1.2 AI 助手

可以问概念、问报错、让 AI 审阅你的代码、要分级提示。**不要让它替你写 `mechanism/` 里的代码**，写的时候建议关闭 AI 自动补全。报告由你自己写。

### 1.3 数字要诚实

报告里的每个数字都要能在日志里找到：运行 id、配置、设备、日志行、代码位置。只跑过一次的结论要标明“单次”。没有跑完的栏目写“未运行”，不填推测的结果。CPU 冒烟和章节演示的数字不算训练结果，也不能用来定门槛。

### 1.4 验证集和测试集

规则来自第 0 章 0.10.1 节：

- **验证集用来做决定。** 选 checkpoint、提示、rank、目标模块、学习率、看错例、提假设，都只用它，可以反复用；
- **测试集是留出集。** 选型冻结以后，预先登记的候选可以各评一次，只报告数字和区间，不根据结果改任何设置；
- **开发期不看测试集的错例。** 最终报告定稿以后才可以解封；
- **已经根据测试集改了设置怎么办？** 在报告里写明，再补一份新的、没看过的测试集。

## 2. 准备

### 2.1 命令

```bash
cd course-v4                                           # 或 make_student 生成的 my-course/
python -m pytest -q assignments/a3/tests               # 一开始：规则和评测协议的独立测试通过，其余因 NotImplementedError 失败
python -m pytest -q assignments/a3/tests/test_rules.py # 只跑规则检查
```

始终不要设置环境变量 `A_SOLUTIONS`。设了以后，测的就不是你的代码。

默认的 `python -m pytest` 不收集 `assignments/`，要写明路径。

测试不下载任何东西，也不加载真实模型或真实 tokenizer。PEFT 是必需依赖：没装时 PEFT 对拍测试会报错，不会跳过。

### 2.2 代码结构

建议的结构（不强制）：

```
mechanism/
  sft.py          模板、联合编码、变长整理
  loss.py         因果损失
  lora.py         LoRA 层和合并层
training/         不受自动检查
  train.py        训练和保存
  predict.py      评测用的预测器
```

机制题按行为评测，不要求类名和别人一样。公开的属性名（`input_ids`、`labels`、`prompt_length`、`base.weight`、`base.bias`、`A`、`B`、`merged()`）必须遵守。

### 2.3 统一约定

- 非法输入抛 `ValueError`；
- 输入不就地修改；
- 每题的完整契约写在 `tests/adapters.py` 的文档字符串和本讲义里。

### 2.4 数据

数据是 `postcorrect-v1`：在 `latex-v1` 的公式上合成六种 OCR 式错误（第 18 章 18.1 节）。划分继承 `latex-v1`，噪声 seed 默认 0。

评测脚本读 JSONL 文件。用下面的方式生成，样本 ID 保留原序号，不要重新编号：

```python
from miniocr.data.latex_corpus import corpus_available
from miniocr.data.postcorrect import build_split, save_jsonl, check_no_leak

if not corpus_available():
    raise SystemExit("缺本地语料")
train = build_split("train", n=N_TRAIN)
val = build_split("val", n=N_VAL)        # val 的前 N_VAL 条
save_jsonl(val, "assignments/a3/runs/val.jsonl")
print(check_no_leak({"train": train, "val": val}))
```

- **先查语料。** 没有本地缓存时，`standard_split` 只发警告，然后降级成内置样例。你的脚本要先检查，没有缓存就退出。语料的来源和缓存位置见第 6 章本章导读；
- **泄漏检查。** `check_no_leak` 按规范键查 train 和 val 的交集，应当是 0。这里不要传 test。规范键只查得到规范化后相同的公式，近重复用 MinHash/LSH 另查，只能给下界（第 12 章 12.6 节，第 22 章 22.5.2 节）；
- **内存里的 test。** `build_split` 内部会完整划分，开发进程的内存里会短暂有 test。规则管的是你的行为：不对它做预测、统计或查看；
- **N_VAL 要固定。** 所有比较用同一个文件。提示基线和模型必须是同一批 source；
- **`make_student` 目录。** 划分依赖第 0 章练习 0.1–0.4，评分依赖练习 0.1–0.6。没做完会抛 `NotImplementedError`。先让 `python -m pytest -q book/ch00_intro` 里对应的测试通过。

也可以用有许可、分组划分明确的真实后纠错数据。真实 TexPixel 数据的使用权没有确认时，不要放进仓库（第 0 章 0.10.2 节）。

### 2.5 自动测试管什么

自动测试只测层接口，用手工 tokenizer、手算数字和小网络。

| 内容 | 怎样检查 |
|---|---|
| 模板、联合编码、EOS、拒收、变长整理 | 手工 tokenizer 和手算值 |
| 因果损失 | 手算值、PyTorch 交叉熵对拍 |
| LoRA 层和合并层 | 手算值、数值梯度、和 PEFT 对拍 tiny 随机网络（单层 Linear 加 tanh） |
| 评测协议 | 小文件和手算值，不读真实 test |
| 模型级挂载、adapter 保存和重载、真实 Qwen3 上的对拍 | **报告审查**，自动测试不管 |
| TRL 的数据路径和 loss 路径 | **报告审查**，自动测试不管 |
| 低精度合并、梯度累积的训练循环 | **报告审查**，自动测试不管 |

测试通过不等于训练层合格。

## 3. 机制题

### 问题（A3-M1）：模板和监督边界（12 分）

实现 `run_encode_sft(tokenizer, source, target, *, max_length=512)`。第 19 章 19.2–19.5 节，第 21 章 21.2.2 节。

返回对象有三个属性：`input_ids` 和 `labels` 是 `tuple[int, ...]`，`prompt_length` 是 `int`。

1. 消息固定：system 是 `Correct OCR errors in the following LaTeX formula.`，user 是 `source`；
2. 调用 `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)`，参数不多不少；
3. 用 `tokenizer.encode(text, add_special_tokens=False)` 编码提示，再联合编码 `提示 + target`。提示必须是联合结果的非空真前缀，否则抛 `ValueError`；
4. 末尾追加 `tokenizer.eos_token_id`；
5. `labels` 不提前移位：提示位置是 −100，答案和 EOS 是原 ID；
6. 先计入 EOS，再判断长度。超过 `max_length` 时整条返回 `None`，不截断，不伪造 EOS。

要抛 `ValueError` 的情况：空白答案、tokenizer 没有 EOS、`max_length < 2`、边界跨 token 合并、答案编码后为空。空白答案和 EOS 要在编码之前查。

手算例子：提示编码成 [1, 2]，答案编码成 [3]，EOS = 9。结果 `input_ids = (1, 2, 3, 9)`，`labels = (−100, −100, 3, 9)`，`prompt_length = 2`。`max_length = 4` 接受，`max_length = 3` 返回 `None`。

### 问题（A3-M2）：变长 batch（10 分）

实现 `run_collate(examples, pad_token_id, *, multiple=1)`。第 19 章 19.6.1–19.6.4 节，第 21 章 21.3–21.4 节。

- 输入是可迭代对象，元素有 `input_ids` 和 `labels` 属性；不修改输入；
- 返回字典，键是 `input_ids`、`labels`、`attention_mask`，都是 CPU 上的 long 张量，形状 (B, T)；
- 右补齐到“最长长度向上取整到 `multiple` 的倍数”；
- 补齐位置的值：`input_ids` 是 pad，`labels` 是 −100，`attention_mask` 是 0。真实位置的 `attention_mask` 是 1；
- **按位置屏蔽，不按 ID 屏蔽。** pad 等于 EOS 时，真实 EOS 仍然要受监督。

要抛 `ValueError` 的情况：空批、`pad_token_id` 是 `None`、`multiple < 1`、样本长度小于 2、`input_ids` 和 `labels` 长度不等、`labels[1:]` 全是 −100。

手算例子：两条样本 ids = (1, 2, 9)、labels = (−100, 2, 9) 和 ids = (1, 9)、labels = (−100, 9)，pad = 9，`multiple = 4`：

```
input_ids       [[1, 2, 9, 9],       [1, 9, 9, 9]]
labels          [[-100, 2, 9, -100], [-100, 9, -100, -100]]
attention_mask  [[1, 1, 1, 0],       [1, 1, 0, 0]]
```

### 问题（A3-M3）：因果损失与 token 平均（10 分）

实现 `run_causal_loss(logits, labels, *, reduction="mean")`。第 19 章 19.3、19.6.5 节，第 21 章 21.6 节。

- `logits` 是 (B, T, V) 浮点张量，`labels` 是 (B, T) long 张量；
- `logits[:, :-1]` 和 `labels[:, 1:]` 配对，只移一次，忽略 −100；
- 交叉熵自己写，数值要稳定：大 logits 下不能溢出（第 3 章 3.3.4 节）；
- `mean` 除以全部有效 token 数，`sum` 是总和；保留自动求导；
- 要抛 `ValueError` 的情况：维度或形状不符、非法 `reduction`、没有有效目标、除 −100 以外的负标签、大于等于 V 的标签。

手算例子：`logits` 全零，形状 (1, 4, 2)，`labels = [[−100, −100, 1, 0]]`。`mean` 是 ln 2，`sum` 是 2 ln 2。`mean` 对 `logits` 的梯度逐位置是 [0, 0]、[1/4, −1/4]、[−1/4, 1/4]、[0, 0]。

测试还和 PyTorch 的交叉熵对拍值和梯度，并检查两条样本（1 个和 4 个有效 token）用 `sum` 分别求和再除以 5，等于整批的 `mean`。

这道题没有单独的梯度累积接口。你的训练循环要用 sum ÷ 全部有效 token 数（第 19 章 19.6.5 节），报告里写明代码位置。

### 问题（A3-M4）：LoRA 层（18 分）

实现 `get_lora(weight, bias=None, *, rank=4, alpha=8)`。第 20 章 20.2–20.5、20.7 节。

返回一个 `Module`：

- `base.weight` 形状 (out, in)，`base.bias` 形状 (out,) 或 `None`。复制传入的张量，不共享存储，并且冻结；
- `A` 形状 (rank, in)，随机非零；`B` 形状 (out, rank)，全零。只有 A、B 可训练，dtype 和 device 跟 `weight` 一致；
- `forward(x)` 支持任意前导维：

```
x @ W.T + b + (alpha / rank) * (x @ A.T) @ B.T
```

`nn.Linear` 和 `nn.init` 都不能用，基座自己写一个小 `Module`。A 可以用 `uniform_` 初始化，第 20 章 20.4.1 节用的范围是 ±1/√in。

要抛 `ValueError` 的情况：`weight` 不是非空的二维浮点张量、`bias` 尺寸不对、`rank` 不是正整数、`alpha` 不是正的有限数。不限制 `rank ≤ min(in, out)`。

手算例子：W = [[1, 2], [3, 4]]，b = [1, −1]，A = [[1, −1]]，B = [[2], [3]]，rank = 1，alpha = 2，x = [2, 1]。输出是 [9, 15]。对输出求和再反向：dx = [14, −4]，dA = [[20, 10]]，dB = [[2], [2]]，基座没有梯度。

测试还检查：

- B = 0 时输出等于基座，第一步 dA = 0，dB 非零；
- 改动传入的 `weight` 不影响基座；
- 三维输入的形状；
- 可训练参数量等于 r(in + out)；
- 输入的数值梯度（`gradcheck`）；
- 非零 B 下和 PEFT 对拍前向、A 梯度、B 梯度、输入梯度。

### 问题（A3-M5）：合并及 PEFT 对拍（10 分）

实现 M4 返回对象的 `merged()`。第 20 章 20.6、20.10、20.14 节。

- 返回一个独立的冻结可调用层，有 `weight` 和 `bias`，保持 dtype、device 和 `training` 状态；
- 不要求是 `nn.Linear`，这个类在机制层里禁用；
- 不能原地修改原层，反复调用不能累加。

手算例子：M4 的数字合并后，weight = [[5, −2], [9, −2]]。连续合并两次结果相同。改第一份不影响第二份，也不影响原层。原层的 `state_dict` 不变。

和 PEFT 对拍的设置：tiny 随机网络（单层 Linear 加 tanh，不是下载的语言模型），`LoraConfig(r=2, lora_alpha=5, target_modules=["proj"], lora_dropout=0, bias="none")`，非零随机 A、B，比前向、输入梯度、A 梯度、B 梯度和 `merge_and_unload` 后的输出。

## 4. 书面题

### 问题（A3-W1）：移位、边界与 EOS（5 分）

M1 手算例子的四个位置，分别由谁预测？哪些位置的标签或 logits 用不上？为什么不能分别编码提示和答案再拼接？pad 等于 EOS 时，按 token ID 屏蔽会怎样？第 19 章 19.2–19.5 节。

### 问题（A3-W2）：梯度累积与提示梯度（5 分）

两个 microbatch 分别有 1 个和 4 个目标，token 损失之和是 S₁、S₂。为什么不能平均各自的 mean？提示的 label 是 −100，是否意味着提示的表示没有梯度？第 19 章 19.4、19.6.5 节，第 21 章 21.6 节。

### 问题（A3-W3）：初始化、参数量与内存（5 分）

推导 LoRA 对一个 batch 的 A、B 梯度，说明 B = 0 时第一步发生什么。推导可训练参数量 r(in + out)。为什么减小 r 不一定明显降低峰值显存？第 20 章 20.3、20.4、20.7、20.8 节。

### 问题（A3-W4）：公平选型与失败诊断（5 分）

为什么 loss 降低不足以证明纠错成功？从自写循环迁移到 TRL 时，怎样保持可比？第 19 章 19.9 节，第 21 章 21.3–21.10 节。

## 5. 训练层任务

### 问题（A3-T1）：训练、基线与完整报告（20 分）

| 子项 | 内容 | 分 |
|---|---|---|
| T1-a | 可复现训练、保存和重载（5.1–5.2 节） | 6 |
| T1-b | 验证选型、四指标和基线（5.3 节，第 6 节） | 6 |
| T1-c | 消融与成本分析（5.4 节） | 4 |
| T1-d | 门槛校准记录、可追溯交付（第 7 节，第 9 节） | 4 |

**任务目标**：在验证集上，模型的 EM_norm 超过你选定的提示基线，同时讨论 CER、改坏率和 fix_rate 的变化。这个目标还没有校准成门槛，门槛的状态见第 7 节。

### 5.1 训练

- **数据**：`postcorrect-v1` 的 train，或 2.4 节说的真实数据。输入只有带噪文本，目标是干净 LaTeX。用规范键分组，查规范键交集和近重复（2.4 节）。记录数据版本、噪声 seed 和配置、样本 ID 清单或其哈希、拒收比例（第 19 章 19.6.4 节）；
- **模型**：建议用 Qwen3-0.6B 的本地快照。tiny 随机模型只能当管线冒烟；
- **路线二选一**：
  - 第 19 章的自写循环，加第 20 章的模型级挂载。零件用你 `mechanism/` 里的 M1–M4，挂载代码放在 `mechanism/` 外面；
  - 第 21 章的 TRL + PEFT。数据用 M1 预编码的 `input_ids` 和 `labels`（21.2.2 节）。

本作业不提供完整训练程序。

**走 TRL 路线时**：对照你安装的版本的文档和源码，查数据准备、packing 和 loss 路径（21.4.2、21.6.3 节）。CUDA 上如果走融合 LM head，要在 CUDA 上另做 loss、梯度和一步更新的对拍（21.6.4、21.6.5 节），CPU 上的对拍不能代替它。

### 5.2 长训练之前的检查

先用真实 train 样本走一遍 前向 → 反向 → 保存 → 加载，通过了再开长训练。报告这些：

- **模板和边界**：模板原文，一条样本的 token 边界（第 19 章 19.2.2 节），EOS、pad、长度上限和拒收计数；
- **监督信号**：第一批的有效 token 数、提示监督数 = 0、EOS 监督数等于样本数（第 19 章 19.9.5 节，第 21 章 21.3、21.4.4 节）；
- **可训练参数**：名字、数量和公式（第 20 章 20.7 节，第 21 章 21.5 节）；
- **和 PEFT 对拍**：真实模型或你实际挂载路径上的前向、梯度、合并误差（第 20 章 20.10 节）。自动测试只对拍了 tiny 网络；
- **保存重载**：同时保存基座身份、rank、alpha、目标模块（第 20 章 20.5.2 节）；重载后在同一批 val 上逐条比较输出（第 21 章 21.7.3 节）。

### 5.3 选型和基线

- **只看 val**：checkpoint、提示和超参只用验证集选；
- **规则先写**：选型规则、超参探索范围和停止规则都在跑之前写定（第 20 章 20.11.1 节、第 21 章 21.8.1 节有例子）；
- **提示基线**：至少跑零样本和一项带 train 示例的提示（第 18 章 18.5.1 节），示例只能取自 train。选哪一个作为基线的规则也要预先写明。提示基线和模型用同一批 source、同一套生成设置和精度；
- **照抄**：评测脚本自动给出，作为锚点。照抄比不过的结果要照实写（第 18 章 18.2.1 节）；
- **曲线**：报告训练 loss 和固定数据上的 NLL 或 val 指标（第 21 章 21.9 节）。

### 5.4 消融和成本

- **rank 或目标模块**：至少比较两种配置，固定训练预算和评测数据，说明参数量和成本的差别（第 20 章 20.7、20.11 节）；
- **监督故障**：选一项（label 错一位、提示计入 loss、漏 EOS），在 val 上诊断。比较时一律用正确 labels 算 val NLL，再看生成指标（第 19 章 19.9 节）；
- **成本**：实测训练、生成、评测的时间、吞吐和峰值内存，写明统计方法（第 20 章 20.8.5 节，第 21 章 21.10.4 节）。单次实验注明单次；
- **错例分析只在 val 上做**：按错误类型分桶，区分提示输出的格式错误和真正改错（第 18 章 18.6.1 节）。分析脚本放在 `mechanism/` 外面。

## 6. 统一评测

### 6.1 预测器和配置

评测脚本通过配置文件调用你的预测器：

```python
def predict(sources: list[str], config: dict) -> list[str]: ...
```

- 只收到 source，收不到答案；
- 自己加载本地模型，执行 config 里的提示和生成设置；
- 返回等长的原始解码文本：只解码新生成的部分，去掉特殊 token，不抽取公式，也不剥 `$` 和代码块。主指标是原始输出的 EM_norm（第 21 章 21.2.3 节）；
- 不能记录 test 的逐条输出，也不能拿 source 去查 target。

配置是一个 JSON 文件：

```json
{
  "predictor": "assignments.a3.training.predict:predict",
  "kwargs": {},
  "artifacts": {"本地文件路径": "该文件的 sha256"},
  "run_id": "",
  "model_revision": "",
  "prompt": "",
  "generation": {},
  "seed": 0
}
```

- 八个字段都必须有；
- `artifacts` 列出所有权重、tokenizer 文件和预测器代码，一个文件一行，不能写目录。任何一个文件改了，脚本就拒绝运行；
- **改了任何一项，都要重新在验证集上选型**；
- 配置里的参数必须在预测器里真正执行，不能只写着看。这一点靠人工审查。

SHA256 用评测模块自己算：`from assignments.a3.evaluate import digest`。

### 6.2 验证集评测

```bash
python -m assignments.a3.evaluate --data assignments/a3/runs/val.jsonl \
    --config selected.json --baseline prompt.json > assignments/a3/runs/selected.val.json
```

`--baseline` 必须给。比较两种提示时，把一种放在 `--config`，另一种放在 `--baseline`。在 `/tmp` 里用自造的两条 val 和小随机模型走通流程可以，它的数字不能写进结果。

输出里有 `model`、`prompt_baseline`、`copy_baseline` 三组面板，还有 `em_norm_gain_over_prompt`。输出里的 `threshold` 固定写“待校准”，你的校准记录写在报告里。

### 6.3 文本后纠错面板

用第 0 章 0.9.3 节的文本后纠错面板，定义见第 18 章 18.2 节：

| 字段 | 面板指标 | 口径 |
|---|---|---|
| `em_norm` | EM_norm | 规范化后完全匹配的比例，附成功数、分母和 Wilson 95% 区间 |
| `cer` | CER | 规范化字符的语料级距离，1000 次固定 seed 的样本 bootstrap 区间；可能大于 1 |
| `broken_rate` | 改坏率 | 输入本来正确的样本里，预测和目标不同的比例 |
| `fix_rate` | fix_rate | 输入有错的样本里，预测和目标相同的比例 |

- 比例的分母是 0 时，值是 `None`；
- “本来正确”按原串 `source == target` 定义。只多了空格的样本算有错，照抄也可能把它“改对”，所以照抄的 fix_rate 可能不是 0；
- `em_norm_gain_over_prompt` 只是差值，不是显著性检验。区间重叠时不要夸大结论；
- `protocol` 字段含评分和规范化代码的哈希，报告里要写；
- 脚本不输出 `parse_rate`。它只说明语法，不说明对错。需要诊断时在 val 上另外列，并和标准答案的可解析率一起读（第 0 章 0.9.3 节）；
- CER_token 和渲染一致率不属于这版面板，不要报告。

## 7. 门槛与设备配置

下表全部待校准。A3 没有可以直接启动长训练的预设。

| 项 | CUDA | MPS | CPU |
|---|---|---|---|
| 模型规模、本地快照 | 待校准 | 待校准 | 待校准 |
| 精度、长度、微批、累积 | 待校准 | 待校准 | 待校准 |
| rank、alpha、目标模块 | 待校准 | 待校准 | 待校准 |
| lr、优化器设置、步数、样本数 | 待校准 | 待校准 | 待校准 |
| 检查点和评测间隔、生成预算 | 待校准 | 待校准 | 待校准 |
| 门槛 | 待校准 | 待校准 | 待校准（校准时可放宽） |
| 预计耗时、内存 | 待校准 | 待校准 | 待校准 |

真实模型优先用 CUDA。MPS 和 CPU 是否可行，要实测。小随机模型只是管线冒烟。

门槛由你第一次完整训练后确定：

- 定门槛**只看验证集结果**，并且**写在 test 评测之前**。看了 test 再定门槛，等于用 test 做决定；
- 冒烟和章节演示的数字不能作为校准依据。第 21 章 21.10.3 节的观察也不能直接当门槛；
- 校准时，建议把照抄也作为一个参照（21.10.3 节）。没写进校准记录的，就不算门槛；
- 保留校准配置、曲线和理由；
- 还没做完整训练时，报告里就写“待校准”；
- 目前门槛不单独计分。训练层的 20 分看流程和证据。

## 8. 冻结与最终测试（可选）

A3 不强制评 test，这一步不计分。要评，就按下面做。没有评的话，报告写“未评”。

**第一步：选型。** 全部在 val 上选完，门槛写进报告。

**第二步：准备 test 文件。** 如果有数据所有者，由他提供文件和 SHA256。自学时你自己就是数据所有者：用脚本生成，只输出哈希，不打开文件。

```python
from miniocr.data.postcorrect import build_split, save_jsonl
from assignments.a3.evaluate import digest

path = "assignments/a3/runs/test.jsonl"
save_jsonl(build_split("test", n=N_TEST), path)
print(digest(path))
```

**第三步：写候选清单并冻结。**

```python
import json
from assignments.a3.evaluate import digest, protocol_id

manifest = {
    "protocol": protocol_id(),
    "data_sha256": digest("assignments/a3/runs/test.jsonl"),
    "candidates": {"lora-r4": digest("selected.json"), "prompt": digest("prompt.json")},
}
with open("candidates.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
```

```bash
python -m assignments.a3.evaluate --freeze candidates.json
```

候选绑定配置文件的 SHA256。提示基线如果要评 test，也必须单独登记。清单写入 `assignments/a3/test_ledger/FROZEN.json`，只能写一次，不能覆盖。

**第四步：逐个评测。**

```bash
python -m assignments.a3.evaluate --data assignments/a3/runs/test.jsonl --config selected.json \
    --split test --final-test --candidate lora-r4
```

约束：

- `--final-test` 不能少。test 上不能带 `--baseline`；
- 启动记录 `<配置哈希>.STARTED.json` 在读数据、加载模型和生成之前写好，**失败也占用次数**，不自动重试。所以先确认数据文件和预测器都没问题。数据哈希不符也算一次失败；
- 同一配置不能改名重评，改过的配置不在清单里，也不能评。评测协议的代码变了，也会拒绝；
- 结果写进 `<配置哈希>.RESULT.json`，只有四项指标的总体数字和区间，不输出逐条预测、错例或错误类型分桶；
- 某个候选失败以后不再评它，在报告里写明。

守卫只防本地手滑。它防不住删日志、改 ledger、伪造配置身份，也防不住预测器自己泄漏数据，不要换目录或删记录重评。报告要提交固定的 ledger 路径、完整日志和冻结文件，由人工审查。

不要用 test 的结果挑模型，也不要因此改设置（1.4 节）。

## 9. 交付清单与评分

| 项 | 内容 | 分 |
|---|---|---|
| 机制题 | 测试全部通过（含规则测试）：A3-M1 12、A3-M2 10、A3-M3 10、A3-M4 18、A3-M5 10 | 60 |
| 书面题 | A3-W1 5、A3-W2 5、A3-W3 5、A3-W4 5 | 20 |
| T1-a | 训练和预测代码、配置、数据记录、长训练前的检查、保存重载对照 | 6 |
| T1-b | 选型规则、曲线、val 评测 JSON、提示基线和照抄的四指标表及区间 | 6 |
| T1-c | rank 或目标模块对比、监督故障诊断、val 错例分析、实测成本 | 4 |
| T1-d | 门槛校准记录；哈希、依赖版本和日志完整。随机抽报告里 3 个数字，要能给出运行 id、配置、设备、日志行、代码位置 | 4 |
| 合计 | | 100 |

要提交的报告内容以 `writeup_template.md` 为准。评过 test 时，再加上固定的 ledger 路径、`FROZEN.json`、启动记录、结果文件和完整日志，并声明没有据此改设置。test 本身不计分；评了却缺这些文件，T1-d 按不可追溯扣分。

权重和数据不用放进仓库，留在本地，配置里的哈希要对得上。

测试通过不等于学术诚信的证明。算法是否放在 `mechanism/`、`adapters.py` 是否只有接线、配置是否真的被执行、报告结论是否受 test 影响，都要人工审阅。
