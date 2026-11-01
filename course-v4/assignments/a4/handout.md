# A4：DPO 与 GRPO，用偏好和奖励继续训练后纠错模型

**前置**：第 23–25 章；A3 或第 19–21 章的 SFT 流程；第 0 章练习 0.1–0.6（2.4 节说明原因）。第 34 章 34.3–34.5 节用于反思。**算力**：机制题和冒烟只要 CPU；训练层的设备配置待校准（第 7 节）。**总分**：100。

A4 分三层：机制题（75 分）、书面题（10 分）、训练层（15 分）。你从一个空的包开始，自己写 DPO 和 GRPO 的损失、优势、KL 估计、奖励和探针。然后从同一个 SFT 起点训练 DPO 和 GRPO，和起点、继续 SFT 比较。任务仍是小文本模型上的 OCR 后纠错：输入带噪的 LaTeX，输出干净的 LaTeX。A4 不交付 VLM 训练。

你要交的东西：

1. `assignments/a4/mechanism/` 里的代码；
2. `assignments/a4/tests/adapters.py` 里的接线；
3. 训练和预测代码（放在 `mechanism/` 外面）、评测配置、日志和评测结果；
4. 实验报告 `writeup.md`（模板见 `writeup_template.md`）。

第 23、25 章的 tour 用的是参考实现。它们可以用来对照和理解流程，产出的日志和检查点不算 A4 的结果。第 23 章 23.8 节、第 25 章 25.8 节的数字来自历史运行，只作参照：不是门槛，也不是设备配置。

A4 的结果可以填进项目报告 R2 的 DPO 行和 GRPO 行（第 23 章 23.10 节，第 25 章 25.9 节）。填的时候写明：这里的噪声是文本上合成的，不能证明真实图像 OCR 也能泛化（第 18 章 18.1.5 节）。

## 1. 规则

### 1.1 能用什么、不能用什么

`mechanism/` 是**机制实现部分**（`rules.toml` 里列出），受自动检查（`tests/test_rules.py`），子目录也会检查。

- **可以用**：PyTorch 的基础张量运算（`logaddexp`、`logsumexp`、`expm1`、`gather`、`std`、`minimum`、`clamp`、`masked_fill`……）、自动求导、`torch.nn.Module`、`torch.nn.Parameter`、Python 标准库。本题用不到模块类；
- **不能用**：`torch.nn` 里的现成层和损失、`torch.nn.functional` 的任何函数（包括 `F.logsigmoid`）、`torch.softmax`、`torch.log_softmax`、张量的 `.softmax()`、`.log_softmax()` 方法、`torch.optim` 里的现成优化器；
- **不能 import** `miniocr` 里除 `miniocr.data` 和 `miniocr.device` 之外的任何模块。`miniocr.train.preference`、`miniocr.train.grpo`、`miniocr.eval` 都不能进 `mechanism/`；
- **不能 import 其他作业**。`mechanism/` 内部用相对导入。可以复制你以前写过的代码。

`adapters.py` 只接线，算法都写在 `mechanism/` 里。

**下面这些能通过自动检查，但一样禁用：**

| 写法 | 为什么禁用 |
|---|---|
| `torch.distributions`（`Categorical`、`kl_divergence`……） | 现成的 log-prob 和 KL |
| `trl`、`peft`、`transformers`、`scipy`、`sklearn` 进 `mechanism/` | 机制题要你自己写，框架只能放在 `mechanism/` 外面 |
| `importlib`、`__import__`、`getattr` 绕过导入检查 | 检查器是语法检查，反射看不出来，规则仍然有效 |
| `adapters.py` 导入 `miniocr.train`、`miniocr.eval` 或 TRL 再转发 | `tests/` 不是受检目录，人工审阅接线是否只转发到你的 `mechanism/` |

检查器抓不住所有绕过方式。规则以本节为准。

**奖励工具走回调。** `reward` 的解析器、规范化器、编辑率和格式检测器由调用方传入。`mechanism/` 不能偷偷 import `miniocr.eval`。这四个工具本身不属于本题实现。

**章节代码不能照搬。** 第 23 章的 `bradley_terry_loss` 用了 `F.logsigmoid`。第 24 章的 `sequence_log_prob` 用了 `log_softmax`。这些都不能进 `mechanism/`。接口上的其他差别见 1.2 节。

**框架实验放在 `mechanism/` 外面。** 训练和预测代码放到别的目录（例如 `assignments/a4/training/`），可以用 transformers、PEFT、TRL。这个目录不受自动检查，但你要能逐行解释。它可以导入你的 `assignments.a4.mechanism`。调用 TRL 不算完成机制题。

### 1.2 本作业测什么

**A4 测的是正文的数学目标，不声称和 TRL 逐梯度一致。** 正文和 TRL 的差别列在第 24 章 24.10 节、第 25 章 25.11 节。和对拍有关的差别如下：

| 项 | A4（正文的数学目标） | TRL 1.15.0 |
|---|---|---|
| 标准差 | 总体标准差，`correction=0` | 样本标准差 |
| 聚合 | 每条回答按自身长度平均，再对回答平均 | `loss_type` 默认 `'dapo'` |
| KL 梯度 | 原始 k3 的梯度 | `use_bias_correction_kl` 默认 `True`，多乘一个比率 |
| 越界率 | 所有 $\lvert r-1\rvert>\varepsilon$ 的 token | 只数梯度为 0 的两格 |

**A4 的接口比章节代码严格。**

| 项 | 章节代码 | A4 |
|---|---|---|
| 参考模型的 detach | `dpo_loss` 由调用者冻结 | 接口内部 detach 参考值；GRPO 内部 detach 旧策略、参考、优势 |
| 形状不匹配 | 部分输入会广播 | 一律抛 `ValueError` |
| dtype | 部分函数转成 float32 | 保留输入 dtype，float64 用来做数值梯度 |
| mask | 先算 `exp` 再乘 mask | 先屏蔽无效位置，被屏蔽的极端值不能污染 loss 和梯度 |

**第 34 章只用于反思。** 那里的图像奖励分 0、0.2、0.5、1.0 四档，无效组可能整组跳过（34.3.2 节，34.11 节）。A4 用第 25 章的文本奖励，零优势的组仍然贡献 KL。两个训练目标不混用，也不交付 VLM 训练。

### 1.3 AI 助手

可以问概念、问报错、让 AI 审阅你的代码、要分级提示。**不要让它替你写 `mechanism/` 里的代码**，写的时候建议关闭 AI 自动补全。报告由你自己写。

### 1.4 数字要诚实

报告里的每个数字都要能在日志里找到：运行 id、配置、设备、日志行、代码位置。只跑过一次的结论要标明“单次”，不能把单种子结论写成普遍事实。没有跑完的栏目写“未运行”，不填推测的结果。CPU 冒烟和章节演示的数字不算训练结果，也不能用来定门槛。

### 1.5 验证集和测试集

规则来自第 0 章 0.10.1 节：

- **验证集用来做决定。** 选超参、修奖励、选 checkpoint、看错例、提假设，都只用它，可以反复用；
- **测试集是留出集。** 选型冻结以后，预先登记的候选可以各评一次，只报告数字和区间，不根据结果改任何设置；
- **开发期不看测试集的错例。** 最终报告定稿以后才可以解封；
- **已经根据测试集改了设置怎么办？** 在报告里写明，再补一份新的、没看过的测试集。

下文的 dev 就是验证集。评测脚本写 `--split dev`，JSONL 里的 `split` 字段写 `val`。

## 2. 准备

### 2.1 命令

```bash
cd course-v4                                                # 或 make_student 生成的 my-course/
CUDA_VISIBLE_DEVICES="" python -m pytest -q assignments/a4/tests               # 一开始：规则和部分评测协议测试通过，其余因 NotImplementedError 失败
CUDA_VISIBLE_DEVICES="" python -m pytest -q assignments/a4/tests/test_rules.py # 只跑规则检查
```

始终不要设置环境变量 `A_SOLUTIONS`。设了以后，测的就不是你的代码。

默认的 `python -m pytest` 不收集 `assignments/`，要写明路径。

测试不下载任何东西，不用 GPU，也不加载真实模型或真实 tokenizer。

### 2.2 代码结构

建议的结构（不强制）：

```
mechanism/
  preference.py   Bradley-Terry、DPO、响应 log-prob
  grpo.py         组内优势、KL 估计、GRPO 损失
  reward.py       奖励和静态探针
training/         不受自动检查
  train_dpo.py    DPO 训练和保存
  train_grpo.py   GRPO 训练和保存
  predict.py      评测用的工厂
```

机制题按行为评测，不要求类名和别人一样。八个接口的函数名和参数名必须和 `tests/adapters.py` 一致。

### 2.3 统一约定

- 浮点张量要非空、有限、同设备、同 dtype；
- 形状不匹配、非法标量、非二值 mask 抛 `ValueError`，不能悄悄广播；
- 保留输入的 dtype。测试用 float64 做数值梯度检查；
- 输入不就地修改；
- 接口里 `beta`、`epsilon`、`eps` 的默认值是手算题的约定，不是训练推荐值；
- 每题的完整契约写在 `tests/adapters.py` 的文档字符串和本讲义里。

### 2.4 数据

数据是 `postcorrect-v1`：在 `latex-v1` 的公式上合成六种 OCR 式错误（第 18 章 18.1 节）。划分继承 `latex-v1`，噪声 seed 默认 0。

评测脚本读 JSONL，每行的字段是 `id`、`split`、`source`、`target`、`is_clean`。`save_jsonl` 写的字段里没有 `is_clean`，不能直接给评测脚本用。用下面的方式生成，样本 ID 保留原序号，不要重新编号：

```python
import json
from miniocr.data.latex_corpus import corpus_available
from miniocr.data.postcorrect import build_split, check_no_leak


def write_eval_jsonl(examples, path):
    with open(path, "w", encoding="utf-8") as f:
        for e in examples:
            row = dict(id=e.id, split=e.split, source=e.source, target=e.target, is_clean=e.is_clean)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if not corpus_available():
    raise SystemExit("缺本地语料")
train = build_split("train", n=N_TRAIN)
val = build_split("val", n=N_VAL)        # val 的前 N_VAL 条
write_eval_jsonl(val, "assignments/a4/runs/dev.jsonl")
print(check_no_leak({"train": train, "val": val}))
```

- **先查语料。** 没有本地缓存时，`standard_split` 只发警告，然后降级成内置样例。你的脚本要先检查，没有缓存就退出。语料的来源和缓存位置见第 6 章本章导读；
- **`is_clean` 按原串定义。** 它等于 `source == target`，和第 18 章 18.2 节的面板一致。脚本会核对；
- **ID 要唯一。** 重复 ID 和空文件都会被拒绝；
- **泄漏检查。** `check_no_leak` 按规范键查 train 和 val 的交集，应当是 0。这里不要传 test。规范键只查得到规范化后相同的公式，近重复用 MinHash/LSH 另查，只能给下界（第 12 章 12.6 节，第 22 章 22.5.4 节）；
- **内存里的 test。** `build_split` 内部会完整划分，开发进程的内存里会短暂有 test。规则管的是你的行为：不对它做预测、统计或查看；
- **N_VAL 要固定。** 所有比较用同一个文件；
- **`make_student` 目录。** 划分依赖第 0 章练习 0.1–0.4，评分和奖励工具依赖练习 0.1–0.6。没做完会抛 `NotImplementedError`。先让 `python -m pytest -q book/ch00_intro` 里对应的测试通过。

也可以用有许可、分组划分明确的真实后纠错数据。报告要写数据来源和使用权。真实 TexPixel 数据的使用权没有确认时，不要放进仓库（第 0 章 0.10.2 节）。

### 2.5 自动测试管什么

自动测试只测层接口，用手算数字、精确枚举、数值梯度和小文件。没有二进制 fixtures。

| 内容 | 怎样检查 |
|---|---|
| Bradley-Terry、DPO | 手算值、数值梯度、参考值没有梯度 |
| 响应 log-prob | 手算值、PyTorch 交叉熵的梯度对拍、数值梯度 |
| 组内优势 | 第 25 章手算表、第 34 章稀疏奖励、数值梯度 |
| KL 估计 | 两点分布精确加权、`1e-8` 处的精度、数值梯度 |
| GRPO 损失 | 第 25 章两张梯度表、裁剪四格、mask、数值梯度 |
| 奖励和探针 | 确定性回调隔离真实渲染器 |
| 评测协议 | 临时合成文件，不读真实 test |
| 规则 | 语法树检查，抓不住反射 |
| TRL 配置、真实模型上的数据路径和 loss 路径 | **报告审查**，自动测试不管 |
| 训练效果、门槛、投机验收 | **报告审查**，自动测试不管 |

测试通过不等于训练层合格。

## 3. 机制题

### 问题（A4-M1）：Bradley-Terry 损失（5 分）

实现 `bradley_terry_loss(chosen, rejected)`。第 23 章 23.3、23.6.1 节。

- 输入是两个等形的 [B] 浮点张量，返回逐对损失 [B]，不求平均；
- 稳定计算 $\log(1+e^{\text{rejected}-\text{chosen}})$。`torch.log(torch.sigmoid(z))` 在分差很大时会出问题；
- 形状不同抛 `ValueError`，不能广播。

手算例子：chosen = 2、rejected = 0.5，分差 1.5，损失 $\log(1+e^{-1.5})=0.2014$。分差 ±1000 时损失分别是 0 和 1000，不能溢出。

测试还做 float64 数值梯度。

自查（不计分）：为什么 chosen 分数越高，loss 越小？

### 问题（A4-M2）：DPO 损失（10 分）

实现 `dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=.1)`。第 23 章 23.5、23.6.2 节，23.8.4 节。

- 四个等形的 [B] 张量，都是响应 token 的 log-prob 之和；
- 返回 `(loss, reward_chosen, reward_rejected)`，每项都是 [B]；
- 隐式奖励是 `beta * (policy - ref)`。两个参考值在接口内部 detach；
- `beta` 非正或 NaN 抛 `ValueError`。形状不同也抛。

手算例子：

| 策略 (chosen, rejected) | 参考 | β | 隐式奖励 | loss | chosen 的梯度 |
|---|---|---:|---|---|---:|
| (−5, −6) | (−5, −6) | 0.1 | (0, 0) | ln 2 | −0.05 |
| (−5, −6) | (−5, −6) | 0.5 | (0, 0) | ln 2 | −0.25 |
| (−3, −7) | (−5, −6) | 0.5 | (1, −0.5) | $\log(1+e^{-1.5})$ | $-0.5/(1+e^{1.5})$ |

rejected 的梯度和 chosen 反号，参考值没有梯度。

自查（不计分）：为什么 margin 变好，不保证 chosen 的质量变好？

### 问题（A4-M3）：响应 log-prob（7 分）

实现 `response_logprob(logits, labels)`。第 23 章 23.5.4、23.6.3 节，第 24 章 24.2.4、24.7.2 节，第 19 章 19.3.3 节。

- `logits` 是 [B, T, V] 浮点张量，`labels` 是 [B, T] long 张量，T ≥ 2；
- `logits[:, :-1]` 和 `labels[:, 1:]` 配对。−100 表示提示或 pad；
- 返回 [B]，是有效 token 的和，不平均。EOS 如果是有效动作，必须计入；
- 全部被屏蔽的序列返回 0；
- 除 −100 以外的越界标签、形状不符、T < 2 抛 `ValueError`。

手算例子：词表大小 2，`labels = [[−100, 1, −100, 0]]`。位置 0 的 logits 是 $[0,\ln 3]$，预测标签 1，概率 0.75。位置 2 的 logits 是 $[0,0]$，预测标签 0，概率 0.5。结果是 $\log 0.75+\log 0.5$。位置 1 和最后一个位置的 logits 梯度是 0。

测试还和 PyTorch 交叉熵的和对拍梯度。交叉熵只用来做对拍，你的实现不能调用它。

自查（不计分）：为什么 DPO 不按长度平均？

### 问题（A4-M4）：组内优势（10 分）

实现 `group_advantages(rewards, group_size, *, scale=True, eps=1e-4)`。第 25 章 25.2 节，第 34 章 34.3.3–34.3.4 节。

- 连续 G 个奖励属于同一个提示。输入和输出都是 [P·G]；
- `scale=True` 返回 (R − 组均值) / (组内总体标准差 + eps)，`scale=False` 只减均值；
- 保留 `rewards` 的梯度，用来做数值验证。训练时在损失里 detach；
- 长度不能被 G 整除、G 不是正整数、`eps` 非正、维度不是 1 都抛 `ValueError`。

手算例子：R = [1, 0.1, 0.1, 0]，均值 0.3，方差 0.165。优势精确等于 [0.7, −0.2, −0.2, −0.3] / ($\sqrt{0.165}$ + 0.0001)，约为 [1.7229, −0.4922, −0.4922, −0.7384]。

测试还检查：

- 在后面接一个常数组 [7, 7, 7, 7]，这一组优势全为 0，第一组不变。这证明没有跨提示归一化；
- G = 1 时优势全为 0；
- 七个 0 加一个 0.2，`eps = 1e-12`，优势是 $[-1/\sqrt7,\dots,\sqrt7]$。低奖励也被放大成和 1.0 一样的优势。

自查（不计分）：总体标准差和样本标准差有什么区别？

### 问题（A4-M5）：KL 估计器（10 分）

实现 `kl_estimators(log_ratio)`。第 24 章 24.5 节，第 25 章 25.3.2 节。

- 输入是非空的有限浮点张量，形状任意。$x=\log q-\log p$，样本来自 p；
- 返回同形字典：`k1 = -x`，`k2 = x²/2`，`k3 = expm1(x) - x`。保留梯度；
- NaN 和空输入抛 `ValueError`。

手算例子：p = [0.2, 0.8]，q = [0.6, 0.4]。

| 样本 | ρ = q/p | k1 | k2 | k3 |
|---|---:|---:|---:|---:|
| 1 | 3 | −ln 3 | (ln 3)²/2 | 2 − ln 3 |
| 2 | 0.5 | ln 2 | (ln 2)²/2 | −0.5 + ln 2 |

按 p 加权后，k1 和 k3 都等于 KL(p‖q)，k2 不等。k3 逐点非负。`x = 1e-8` 时 k3 约为 $5\times10^{-17}$，先算 `exp` 再减 1 会丢掉这个精度。

自查（不计分）：k3 在数值上无偏，为什么直接反传不等于 KL(p‖q) 的梯度？

### 问题（A4-M6）：GRPO 损失（18 分）

实现 `grpo_loss(logp_new, logp_old, logp_ref, advantages, mask, *, epsilon=.2, beta=.04)`。第 24 章 24.6 节，第 25 章 25.1–25.3、25.6–25.7 节。

- `logp_new`、`logp_old`、`logp_ref` 是 [N, T]，`advantages` 是 [N]，`mask` 是二值的 [N, T]（整数或浮点都要接受）；
- 旧策略、参考、优势在接口内部 detach；
- 逐 token 的损失是 $-[\min(r\hat A,\mathrm{clip}(r)\hat A)-\beta\,k3]$，$r=\exp(\text{new}-\text{old})$，k3 的 $\log\rho=\text{ref}-\text{new}$；
- 每条回答除以自己的有效 token 数，再对全部 N 条取平均。空回答贡献 0，但仍然算在 N 里；
- 零优势的组仍然贡献 KL；
- 被屏蔽位置的极端有限值（例如 1000）不能让 loss 或梯度变成 inf 或 NaN；
- 返回 `(loss, diagnostics)`。`loss` 是标量。`diagnostics` 是 float 字典，键是 `clip_fraction`、`kl_mean`、`ratio_mean`，都按有效 token 平均。`clip_fraction` 数所有 $\lvert r-1\rvert>\varepsilon$ 的 token，不分方向。全部被屏蔽时，诊断全为 0；
- 形状不符、非二值 mask、`epsilon` 不在 (0, 1) 内、`beta` 为负抛 `ValueError`。

手算例子用第 25 章 25.3.3、25.3.4 节的两张表：优势是 M4 的那一组，长度 [2, 3, 1, 2]，mask 是 `[[1,1,0],[1,1,1],[1,0,0],[1,1,0]]`，参考等于旧策略。

| 比率 | loss | `clip_fraction` | `kl_mean` | `ratio_mean` |
|---|---:|---:|---:|---:|
| 全部为 1 | 0 | 0 | 0 | 1 |
| [[1, 1.5], [1, 1, 1], [1.1], [0.5, 1]] | 约 −0.0473 | 0.25 | 约 0.0479 | 1.0125 |

全部为 1 时 loss 是 0，每个 token 的梯度却是 $-\hat A_i/(4L_i)$。更新后的梯度表（用未舍入的优势）：

| 回答 | 对 $\log\pi_\theta$ 的梯度 |
|---|---|
| 1 | $[-\hat A_1/8,\ 0.04(1-1/1.5)/8]$ |
| 2 | 每个 token $-\hat A_2/12$ |
| 3 | $(-1.1\hat A_3+0.04(1-1/1.1))/4$ |
| 4 | $[-0.005,\ -\hat A_4/8]$ |

裁剪四格（`beta = 0`，`epsilon = 0.2`，单 token）：

| $\hat A$ | r | 对 $\log r$ 的梯度 | `clip_fraction` |
|---:|---:|---:|---:|
| 1 | 1.5 | 0 | 1 |
| 1 | 0.5 | −0.5 | 1 |
| −1 | 0.5 | 0 | 1 |
| −1 | 1.5 | 1.5 | 1 |

只有往有利方向走过头时，策略项的梯度才是 0。越界率把四格都算上。

测试还做 float64 数值梯度，并检查 `logp_old`、`logp_ref`、`advantages` 都没有梯度。

自查（不计分）：一批样本只更新一次时，`clip_fraction` 为什么常常是 0？

### 问题（A4-M7）：奖励（10 分）

实现 `reward(completion, source, target, *, parse, normalize, error_rate, format_issues, mode='layered')`。第 25 章 25.4 节，第 34 章 34.4 节。

四个回调的签名：`parse(text) -> bool`，`normalize(text) -> str`，`error_rate(target, text) -> 非负有限 float`，`format_issues(text) -> 问题集合`（空集合表示没有问题）。返回 float。

`layered` 按下面的顺序判断：

1. 空串（strip 后）或解析失败：0；
2. 有格式问题：0.1；
3. 规范化后和目标相同：1；
4. `base = error_rate(target, source)`，`base <= 0`：0.1；
5. 其余：$0.1+0.4\cdot\mathrm{clip}((\text{base}-\text{pred})/\text{base},0,1)$。

`parse_only` 故意写坏：非空而且能解析就给 1。

文本不是字符串、模式不认识、编辑率是 NaN 或负数，都抛 `ValueError`。

手算例子（确定性回调，目标 `abcd`，输入 `abxy`）：

| 输出 | 奖励 |
|---|---:|
| `abxd`，错误率减半 | 0.3 |
| `abxy`（复制）或 `zzzz`（更差） | 0.1 |
| 空串，或解析失败的 `INVALID` | 0 |
| `a b c d`（规范化后相同） | 1 |
| 有格式问题 | 0.1 |

输入本来就干净（`abcd`）、输出 `abxd`：0.1。`x` 在 `parse_only` 下得 1，在 `layered` 下得 0.1。

**这是第 25 章的历史奖励，已知有缺陷。** 不能解析的标准答案也得 0。测试固定这个盲区，不是说它正确。缺陷的影响见第 25 章 25.4.5 节，第 34 章 34.4.3 节。

自查（不计分）：先挡解析、再判全对，有什么风险？

### 问题（A4-M8）：静态探针（5 分）

实现 `hack_probes(source, target)`。第 25 章 25.5 节，第 34 章 34.5 节。

返回 `list[(name, text)]`，六项，顺序固定：

| name | text |
|---|---|
| `empty` | 空串 |
| `single_symbol` | `x` |
| `copy_source` | `source` |
| `copy_with_prose` | `Here is the corrected formula: ` + `source` |
| `repeat_source` | 空格连接三份 `source` |
| `gold_answer` | `target` |

不是字符串抛 `ValueError`。这题没有模型前向，也没有梯度要求。

单字符不能一律判成投机：`x` 恰好就是目标时，奖励是 1。

**静态探针不调用模型，不能据此判定通过。** 它只说明奖励有没有漏洞。探针结果不随训练变化，`probe_before == probe_after` 只说明奖励规则没变（第 34 章 34.5.3 节）。模型有没有钻进去，要看 5.6 节的模型输出。

## 4. 书面题

每题 2 分。

### 问题（A4-W1）：DPO 梯度（2 分）

推导 DPO 损失对 $\log\pi_\theta(y_w)$ 和 $\log\pi_\theta(y_l)$ 的梯度。说明参考模型为什么不参与求导。用第 23 章 23.5.6 节的情形 C 解释：chosen 和 rejected 的概率可以同时下降，loss 还在降。这时要看什么？第 23 章 23.5.5、23.5.6、23.8.4 节。

### 问题（A4-W2）：组内基线的偏差（2 分）

证明含自身的组均值满足

$$\mathbb E[(R_i-\bar R)\,s_i]=\frac{G-1}{G}\,\mathbb E[R_i\,s_i]$$

其中 $s_i=\nabla_\theta\log\pi_\theta(y_i)$。写明用到了哪个独立性和哪个零期望。为什么不能说普通组均值完全无偏？再除以标准差，除了尺度还改变了什么？第 24 章 24.3.3、24.3.8 节，第 25 章 25.2.6、25.2.7 节。

### 问题（A4-W3）：GRPO 手算（2 分）

手算第 25 章 25.3.4 节的完整损失和逐 token 梯度。可以显示到四位小数，但中间量不要舍入：评分用未舍入的中间量。解释两个饱和的 token 为什么还有梯度，方向是什么。第 25 章 25.3.2–25.3.4 节。

### 问题（A4-W4）：k3 的值与梯度（2 分）

区分 k3 作为 KL 值的估计和直接反传得到的梯度。写出 k3 对 $\log p$ 的导数，说明它的期望对应哪个方向的 KL。说明采样方向和支持集条件：为什么要求 p、q 同正支持？如果样本改从 q 采，还能直接取同一个均值吗？用 M5 的两点分布算一个数说明。第 24 章 24.5.3、24.5.7 节，第 25 章 25.3.2 节。

### 问题（A4-W5）：反投机证据和 test 协议（2 分）

设计一组反投机证据。静态探针能说明什么，不能说明什么？还要看哪些开发集证据？再写出 test 协议：什么时候可以评、谁可以评几次、评完不能做什么、开发期不能看什么。第 25 章 25.5 节，第 34 章 34.5 节，第 0 章 0.10.1 节。

## 5. 训练层任务

### 问题（A4-T1）：同起点训练、对照与反投机验收（15 分）

| 子项 | 内容 | 分 |
|---|---|---|
| T1-a | 训练链路和配置、数据审计（5.1–5.4 节） | 5 |
| T1-b | 完整曲线和同起点对照（5.3–5.5 节，第 6 节） | 5 |
| T1-c | 奖励门槛和反投机验收（5.6 节，第 7 节） | 5 |

**任务目标**：从同一个 SFT 起点完成 DPO 和 GRPO 的可运行训练，比较四个臂：SFT 起点、继续 SFT、DPO、GRPO。这个目标还没有校准成门槛，门槛的状态见第 7 节。

本作业不提供完整训练程序。

### 5.1 模型和数据边界

- **模型**：本地的小文本模型，具体模型和可训练范围待校准。第 23、25 章的设置可以参照，不是已校准的配置。tiny 随机模型只能当管线冒烟；
- **起点**：四个臂共用同一个 SFT 起点，记录它的哈希。起点可以来自你的 A3，或者第 19–21 章的流程；
- **train 的用途**：chosen、rejected 只来自 train。GRPO 的采样和奖励只用 train；
- **dev 的用途**：超参、奖励修订、选择和错例分析只用 dev；
- **框架**：走 TRL 时，把 `loss_type`、`beta`、`scale_rewards`、`num_iterations`、`use_bias_correction_kl` 和版本写进记录。TRL 的 GRPO 默认配置和正文不同（第 25 章 25.7.6、25.11 节）。

### 5.2 长训练之前的检查

先用本地样本走一遍 前向 → 反向 → 保存 → 加载，通过了再开长训练。报告这些：

- **DPO 第一步**：策略等于参考时，loss 应当是 ln 2（第 23 章 23.5.6 节）；
- **GRPO 第一步**：比率为 1，KL 接近 0。看清裁剪在你的配置下会不会触发（第 25 章 25.3.6 节）；
- **gold 自评**：把标准答案原样交给奖励函数，统计得分分布（第 25 章 25.4.5 节，第 34 章 34.4 节）；
- **静态探针表**：六个探针分别在两种奖励下的均值（第 25 章 25.5.3 节）；
- **受控反例**：`parse_only` 只用来暴露漏洞。不能拿它的高奖励宣称通过；
- **短程检查**：建议先跑一小段，确认奖励、优势、KL 都正常（第 34 章 34.6 节）。这段的数字不写进结果。

### 5.3 DPO

- **偏好对**：可以用第 23 章 23.7 节的流程，从 train 上的模型采样构造。记录提示数、候选数、丢弃条件和最终对数；
- **质量统计**：chosen 完全正确的比例、长度统计（23.7.3、23.7.5 节）；
- **曲线**：DPO loss，chosen 和 rejected 的 log-prob 和隐式奖励，margin，reward accuracy（23.8.2 节）；
- **不只看 loss**：chosen 的绝对 log-prob 要和同一批的参考值比（23.8.4 节）。

### 5.4 GRPO

- **奖励版本**：训练用哪个版本，写清楚。历史分层奖励有已知缺陷；
- **奖励修订**：修订奖励时，只依据 dev 和 train 上的自评，写明版本和理由。用统一的版本重评所有开发候选，不能让不同候选用不同的奖励比高低（第 25 章 25.4.6 节）；
- **脚本的 `reward_mean` 始终是历史分层奖励。** M7 的 `layered` 分支不要改，测试会查。修订版另写一个函数，自己算分，两个数都报告；
- **曲线**：奖励、有效组比例、优势分布、KL、`clip_fraction`、输出长度（第 25 章 25.8.2 节）。

### 5.5 对照和选型

- **同起点对照**：继续 SFT 必须有。只和起点比，任何多训练的方法都可能赢（第 25 章 25.8.1 节）；
- **规则先写**：选型规则、超参探索范围和停止规则都在跑之前写定；
- **配对比较**：四个臂用同一批 dev，用配对 bootstrap 比（`miniocr.eval.analysis.paired_bootstrap_diff`，第 22 章 22.3 节）。评测脚本不输出逐条预测，配对比较要自己在 dev 上生成预测；
- **局限**：写明 DPO、GRPO、继续 SFT 的数据量和算力不相等（第 25 章 25.8.1 节）。写明种子数，单种子不能当普遍事实（第 25 章 25.8.7 节）；
- **错例分析只在 dev 上做**（第 22 章 22.6 节）；
- **成本**：实测设备、精度、峰值内存、耗时、生成 token 数、训练步数（第 25 章 25.8.4 节）。

### 5.6 反投机证据

静态探针、gold 自评、平均奖励都不够。一个平均奖励掩盖不了下面这些，它们都要解释：

- 静态探针发现的漏洞；
- gold 覆盖缺陷：标准答案拿不到满分；
- 奖励上升，主指标下降或不动（第 25 章 25.5.6 节，第 34 章 34.5.4 节）；
- 模型实际输出的退化：`empty_rate`、`wrong_single_symbol_rate`、`copy_source_rate`、`mean_char_length`（6.3 节）。干净输入上照抄是对的，要分开看（第 25 章 25.5.4 节）；
- 逐样本对照（第 34 章 34.5.5 节）。

**投机探针的通过标准由你在看 test 之前冻结，写进报告。** 标准要写明看哪些指标、和哪个基线比、方向和容差。本讲义不给数字。脚本也不判通过。没有写进冻结记录的，不算标准。

## 6. 统一评测

### 6.1 工厂和配置

评测脚本通过配置文件调用你的工厂：

```python
def load(checkpoint: str, device: str, settings: dict):
    ...
    def predict(sources: list[str]) -> list[str]:
        ...
    return predict
```

- 工厂只收到 source，收不到答案。工厂不能私自加载数据，也不能输出 test 原文，这一点人工审查；
- 自己加载本地模型，执行 `settings` 里的生成设置；
- 返回等长的原始解码文本：只解码新生成的部分，去掉特殊 token，不抽取公式，也不剥 `$` 和代码块；
- 工厂模块必须是能定位到的本地 Python 文件。

配置是一个 JSON 文件：

```json
{
  "factory": "assignments.a4.training.predict:load",
  "checkpoint": "/绝对路径/model.pt",
  "device": "cpu",
  "settings": {},
  "protocol_version": "你的协议版本"
}
```

- `checkpoint` 必须是包含实际模型参数的**单个文件**，可以在框架训练后打包。不能用指向可变远程模型的空 manifest 代替；
- `settings` 写全部生成参数和依赖版本。分词器、辅助依赖、数据来源和规范化版本写进 `settings` 和报告，保证可复现；
- `protocol_version` 不能为空；
- **改了任何一项，都要重新在验证集上选型**。

### 6.2 验证集评测

```bash
python -m assignments.a4.evaluate --config run.json --data assignments/a4/runs/dev.jsonl --device cpu \
    > assignments/a4/runs/selected.dev.json
```

`--device` 可选 `cpu`、`mps`、`cuda`，必须和配置里的 `device` 相同。在 `/tmp` 里用自造的两条数据和小随机模型走通流程可以，它的数字不能写进结果。

脚本的 `threshold_status` 固定是 `待校准`，`training_pass` 固定是 `null`。训练是否通过，由人工对照你冻结的标准验收。

### 6.3 输出字段

主面板在 `panel` 里，定义见第 0 章 0.9.3 节、第 18 章 18.2 节：

| 字段 | 面板指标 | 口径 |
|---|---|---|
| `em_norm` | EM_norm | 规范化后完全匹配的比例，附成功数、分母和 Wilson 95% 区间 |
| `cer` | CER | 规范化字符的语料级距离，1000 次固定 seed 的样本 bootstrap 区间；可能大于 1 |
| `broken_rate` | 改坏率 | 输入本来正确的样本里，预测和目标不同的比例 |
| `fix_rate` | fix_rate | 输入有错的样本里，预测和目标相同的比例 |

- 比例的分母是 0 时，值是 `None`；
- `panel` 里的 `parse_rate` 只作诊断。它只说明语法，不说明对错；
- `reward_mean` 是历史分层奖励的均值，`reward_interval` 是 1000 次固定 seed 的样本 bootstrap 区间。

dev 还输出下面这些，test 不输出：

| 字段 | 含义 |
|---|---|
| `static_probe_rewards` | 六个静态探针的奖励均值。不调用模型 |
| `gold_reward_mean` | 标准答案的奖励均值 |
| `model_diagnostics.empty_rate` | 模型输出 strip 后为空的比例 |
| `model_diagnostics.wrong_single_symbol_rate` | 模型输出只有一个字符而且不对的比例 |
| `model_diagnostics.copy_source_rate` | 模型输出和 source 逐字相同的比例 |
| `model_diagnostics.mean_char_length` | 模型输出的平均字符数 |
| `probe_status` | 固定提示：静态探针不证明模型通过 |

没有内置照抄基线。想要这个锚点，就写一个把 `sources` 原样返回的工厂，同样走一遍评测。

## 7. 门槛与设备配置

下表全部待校准。A4 没有可以直接启动长训练的预设。

| 设备 | 模型与数据方案 | 训练和生成参数 | 奖励及反投机门槛 |
|---|---|---|---|
| CUDA | 本地小文本模型和 SFT 起点；具体模型和可训练范围待校准 | 精度、长度、G、batch、累积、学习率、beta、epsilon、训练步数、生成设置、样本数全部待校准 | 待校准 |
| MPS | 本地可容纳的小模型；具体模型和可训练范围待校准 | 同上；设备兼容性和精度待校准 | 待校准 |
| CPU | 缩小模型和训练规模；具体模型和可训练范围待校准 | 同上；运行时间和参数待校准 | 待校准（校准时可单独放宽） |

正文某次 GPU 实验的配置不能当成这三套配置。

门槛由你第一次完整训练后确定：

- 定门槛**只看验证集结果**，并且**写在 test 评测之前**。看了 test 再定门槛，等于用 test 做决定；
- 冒烟和章节演示的数字不能作为校准依据。第 23 章 23.8 节、第 25 章 25.8 节的观察也不能直接当门槛；
- 冻结的内容包括奖励门槛和 5.6 节的投机探针通过标准；
- 保留校准配置、曲线和理由；
- 还没做完整训练时，报告里就写“待校准”，训练层写“未验收”；
- 运行成功不等于通过。T1-c 的 5 分不会因为脚本跑通就给。

## 8. 冻结与最终测试（可选）

A4 不强制评 test，这一步不计分。要评，就按下面做。没有评的话，报告写“未评”。

**第一步：选型。** 全部在 dev 上选完。门槛和投机探针的通过标准写进报告。

**第二步：准备 test 文件。** 如果有数据所有者，由他提供文件和 SHA256。自学时你自己就是数据所有者：用脚本生成，只输出哈希，不打开文件。这里的 `write_eval_jsonl` 是 2.4 节的函数。

```python
from miniocr.data.postcorrect import build_split
from assignments.a4.evaluate import sha256

path = "assignments/a4/runs/test.jsonl"
write_eval_jsonl(build_split("test", n=N_TEST), path)
print(sha256(path))
```

**第三步：写登记表 `frozen.json`。** 每个候选绑定配置、检查点和工厂文件的哈希。

```python
import json
from assignments.a4.evaluate import sha256


def identity(config, factory_py):
    with open(config, encoding="utf-8") as f:
        checkpoint = json.load(f)["checkpoint"]
    return dict(config_sha256=sha256(config), checkpoint_sha256=sha256(checkpoint), factory_sha256=sha256(factory_py))


registry = {
    "frozen": True,
    "test_sha256": sha256("assignments/a4/runs/test.jsonl"),
    "candidates": {
        "grpo": identity("grpo.json", "assignments/a4/training/predict.py"),
        "sft-continued": identity("sft.json", "assignments/a4/training/predict.py"),
    },
}
with open("frozen.json", "w", encoding="utf-8") as f:
    json.dump(registry, f, ensure_ascii=False, indent=2)
```

**第四步：逐个评测。**

```bash
python -m assignments.a4.evaluate --config grpo.json --data assignments/a4/runs/test.jsonl --device cpu \
    --split test --final-test --registry frozen.json --candidate grpo
```

约束：

- `--final-test`、`--registry`、`--candidate` 一个都不能少；
- 脚本先核对开关、配置和登记，写下 `started`，然后才打开 test。**失败也占用次数**，不自动重试。数据哈希不符也算一次失败；
- 日志目录固定是 `assignments/a4/final_test_runs/`。第一次启动时，整份登记表写进 `registry.json`，之后不能追加候选，也不能改；
- 同一候选不能再启动。同一个检查点换候选名也会被拒绝，所以每个候选要有自己的检查点文件。不同的预登记候选可以各评一次；
- `events.jsonl` 记录 `started`、`completed`、`failed`。`completed` 只含汇总。`failed` 不写异常内容，因为异常可能含 test 原文；
- 输出只有总体汇总，没有逐条预测、错误类型分桶、静态探针和模型退化诊断；
- 真的遇到硬件故障，在报告里写明事件，不悄悄重跑。

守卫只防本地手滑。它防不住删日志、改源码、改间接依赖，也防不住工厂自己泄漏数据。不要清日志换身份重试。报告要提交登记表、完整日志和代码版本，由人工审查。

不要用 test 的结果挑模型，也不要因此改设置（1.5 节）。

## 9. 交付清单与评分

| 项 | 内容 | 分 |
|---|---|---|
| 机制题 | 测试全部通过（含规则测试）：A4-M1 5、A4-M2 10、A4-M3 7、A4-M4 10、A4-M5 10、A4-M6 18、A4-M7 10、A4-M8 5 | 75 |
| 书面题 | A4-W1 到 A4-W5，每题 2 分 | 10 |
| T1-a | 训练和预测代码、配置、数据来源和划分记录、去重和泄漏检查、长训练前的检查、保存重载 | 5 |
| T1-b | DPO 和 GRPO 的完整曲线、运行 id；四个臂的 dev 主面板、区间和配对比较；局限和成本 | 5 |
| T1-c | gold 自评、静态探针、模型退化指标、奖励和主指标的背离分析；门槛和投机探针标准的冻结记录；人工验收。运行成功不自动给分，未校准就是“未验收” | 5 |
| 合计 | | 100 |

要提交的报告内容以 `writeup_template.md` 为准。评过 test 时，再加上 `frozen.json`、`final_test_runs/` 下的 `registry.json` 和 `events.jsonl`、代码版本，并声明没有据此改设置。test 本身不计分；评了却缺这些文件，T1-c 按不可追溯扣分。

权重和数据不用放进仓库，留在本地，登记的哈希要对得上。

测试通过不等于学术诚信的证明。算法是否放在 `mechanism/`、`adapters.py` 是否只有接线、配置是否真的被执行、报告结论是否受 test 影响，都要人工审阅。
