# 第 19 章　习题：SFT 的监督信号

做题前先跑一遍本章测试，确认环境正常（在 `course-v4/` 下）：

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q book/part3_post_training/ch19_sft
```

习题里标"本项目观察"的数字来自 `outputs/book/ch19/gpu/` 下的结果文件（RTX 4090 48GB 租用，混合 bf16）；标"假设"的数字是为了手算方便编的。

---

## 一、概念题

**19.1** `input_ids`、`attention_mask`、`labels` 三个张量各回答什么问题？对一个**提示** token、一个**答案** token、一个 **padding** 位置，分别写出它在 `attention_mask` 和 `labels` 里的值。

**19.2** 本章的约定是"数据整理时不移位，loss 函数里移一次"。
(a) 如果你的同事在数据整理时就把 labels 左移了一位，而 loss 函数仍然是 `causal_loss`，会发生什么？
(b) 如果团队决定统一在数据整理时移位，`causal_loss` 应该怎样改才正确？

**19.3** 写出 SFT 的训练目标（用 p(y|x) 表示）。把提示也算进 loss 后，目标变成了什么？多出来的那一项对后纠错任务意味着什么？

**19.4** "提示不计 loss，所以提示部分的参数不会被更新。"这句话对吗？说明理由。

**19.5** 为什么答案结尾要追加 EOS？本章的故障对照里，漏掉 EOS 后，微型模型停不下来，Qwen3-0.6B 却照常停下。怎样解释这个差别？能不能据此说"漏 EOS 无害"？

**19.6** 为什么训练时用右 padding，生成时用左 padding？如果生成时也用右 padding，会出现什么问题？

**19.7** 一条样本"提示 + 答案 + EOS"超过 `max_length` 时，本章为什么选择整条拒收，而不是截断？拒收数为什么必须写进报告？

**19.8** 主训练的 val NLL 在第 128 步就降到了 0.026，可 val 前 128 条的 EM_norm 只有 92/128（本项目观察）。这两个数矛盾吗？给出两条理由。

**19.9** 四组故障对照在评估时，不论训练时 labels 怎样构造，一律用**正确 labels** 算 val NLL。为什么不直接比较各组的训练 loss？

**19.10** 本章报告的 SFT 结果是 val 前 500 条上的 365/500。
(a) 为什么说这个数字对"选步"有轻微乐观偏差？
(b) 为什么本章不评测试集？什么时候评？

---

## 二、手算题

下面几题用一个**假设的** byte tokenizer：EOS 的 ID 是 1，pad 的 ID 是 0。一条样本的提示编码后是 5 个 token，记作 p0…p4；答案 `x^2` 编码为 `[124, 98, 54]`。

**19.11 构造三个张量。**
(a) 写出这条样本的 `input_ids` 和 `labels`（各 9 个元素，提示 token 用 p0…p4 表示）。
(b) `causal_loss` 中，哪几个位置的 logits 参与 loss？各自的目标是什么？一共几项？
(c) 分别写出 shift、prompt、no_eos 三种故障下的 `labels`，并按 `tour.py` 里 `prepare` 的定义算出 `supervised_tokens`、`supervised_prompt_tokens`、`supervised_terminal_eos` 三个计数。（`prepare` 数的是 `labels[1:]` 中不为 −100 的个数、`labels[1:prompt_length]` 中不为 −100 的个数，以及 `labels[-1] == EOS` 的样本数。）

**19.12 一个 padding 的 batch。** 第二条样本 B 的提示也是 5 个 token（q0…q4），答案是 `a`（ID 101）。把上题的样本 A 和 B 用 `collate_sft(..., pad_token_id=0, multiple=4)` 组成一个 batch。
(a) 批宽 T 是多少？
(b) 写出 B 的三行张量。
(c) `causal_loss` 里，`logits[:, :-1]` 和 `labels[:, 1:]` 的形状各是多少（词表大小记为 V）？掩码挑出后剩几行？
(d) 如果改成 `pad_token_id=1`（与 EOS 相同），却用 `labels[input_ids == pad_token_id] = -100` 来屏蔽 padding，batch 里会少掉哪些监督？

**19.13 跳一格预测。** 一条序列的 input 是 `[a, b, c, d, e]`，前两个是提示，`c d e` 是答案（e 是 EOS）。
(a) 正确的 labels 是什么？位置 1、2、3 的 logits 各预测哪个 token？
(b) shift 故障下的 labels 是什么？位置 0、1、2 的 logits 各预测哪个 token？用一句话概括"位置 t 的 logits 被训练去预测什么"。

**19.14 梯度累积的平均。** 一次更新有两个 microbatch（**假设的数字**）：

| microbatch | 序列 | 受监督 token 数 | loss 之和 |
|---|---|---:|---:|
| 1 | 甲 | 4 | 2.0 |
| 1 | 乙 | 6 | 1.2 |
| 2 | 丙 | 30 | 3.0 |
| 2 | 丁 | 40 | 4.0 |

分别算出：(a) 两个 microbatch mean 的平均；(b) 四条序列各自 mean 的平均；(c) 总和除以总数。哪一个等于把四条序列拼成一个大 batch 算 mean？在 (a) 的算法下，microbatch 1 的一个 token 的权重是 microbatch 2 的一个 token 的几倍？

**19.15 微型模型的 loss 从哪来。** 微型模型的四条训练样本是 `x_2→x^2`、`y_2→y^2`、`a_2→a^2`、`b_2→b^2`，每条答案受监督 4 个 token（3 个字符 + EOS）。本项目观察：correct 组训练后 loss 是 0.347，no_eos 组是 0.463，no_eos 组在正确 labels 上的 NLL 是 4.04。
(a) 假设模型完全学会了格式，但第一个字母只能在 4 个里均匀地猜，算出 correct 组和 no_eos 组的 loss，和观察值比较。
(b) no_eos 组在正确 labels 上的 NLL 多了一项 EOS。假设其余 3 项的损失与 (a) 中相同，估算模型在答案结尾给 EOS 的概率大约是多少。

**19.16 监督提示稀释了多少。** 本项目观察：故障对照的 1022 条训练样本，correct 组受监督 66096 个 token，prompt 组 156680 个，其中提示 90584 个。
(a) prompt 组里，答案 token 占 loss 分母的比例是多少？
(b) 平均每条样本的提示有多少个 token？答案（含 EOS）有多少个？（提示：`supervised_prompt_tokens` 数的是 `labels[1:prompt_length]`，每条样本少数了哪一个位置？）

**19.17 显存粗算。** Qwen3-0.6B 有 596,049,920 个参数，词表大小 151936。
(a) fp32 权重、fp32 梯度、AdamW 两个 fp32 状态，一共多少 GiB？
(b) 一个 microbatch 8 条、批宽 256 时，bf16 的 logits 张量有多大（GiB）？
(c) 本章主训练实测峰值 21.27 GiB（本项目观察）。(a)(b) 之外，还有哪些东西占显存？

---

## 三、编程题

练习入口都在代码里用 `# >>> 练习 19.x` 和 `# <<<` 标出。删掉函数体自己重写，再跑对应测试。

**练习 19.1（必做）实现 `encode_sft`。** 文件 `miniocr/data/sft.py`。要求：用 `render_chat` 渲染提示；联合编码 prompt+target 并核对前缀；追加 EOS；超长返回 None；空答案报错；支持四种 `variant`。
验证：`test_answer_boundary_and_eos`、`test_faults_have_precise_semantics`；有本地 Qwen3 权重时还有 `test_real_qwen_template_boundary_and_no_duplicate_specials`。

**练习 19.2（必做）实现 `collate_sft`。** 同一文件。要求：右 padding；padding 位置的 attention 为 0、label 为 −100，**按位置**写入；支持 `multiple`；拒绝没有可学 token 的样本。
验证：`test_padding_when_pad_equals_eos_and_alignment`（它故意让 pad ID 等于 EOS ID）。

**练习 19.3（必做）实现 `causal_loss`。** 文件 `miniocr/train/sft.py`。要求：只移一次；支持 `mean` 和 `sum`；全部为 −100 时报错；交叉熵在 fp32 里算。
验证：`test_causal_loss_independent_oracle_and_gradient_mask`、`test_hf_causal_lm_loss_uses_same_single_shift`；再跑 tour 部分 3，看"实现loss"和"逐token手算"是否相等。

**练习 19.4（必做）实现 `train_sft`。** 同一文件。要求：固定种子打乱取 batch；梯度累积按受监督 token 总数加权；梯度裁剪 1；记录每步 loss、token 数、裁剪前梯度范数和用时。
验证：`test_token_weighted_accumulation_matches_one_batch`、`test_tiny_training_reduces_correct_answer_nll`。

**练习 19.5（必做）逐 token 打印一条样本。** 跑 tour 部分 1（lab 第 1 节）。没有本地权重时用默认的 byte tokenizer，有权重时加 `--real`。把输出整理成和正文 19.2.2 节一样的表格，标出参与 loss 的位置和预测它的位置。再自己挑一条 val 样本（用 `build_split('val', 5)` 取），重复一次。

**练习 19.6（必做）训练前的监督检查。** 不看 `tour.py`，自己写一个函数 `check_supervision(examples, eos_id)`，返回受监督 token 总数、提示中受监督的 token 数、末尾 EOS 受监督的样本数。用 byte tokenizer 对四种 `variant` 各构造 4 条样本，打印结果，与手算题 19.11(c) 和 `tour.py` 的 `prepare` 对照。

**练习 19.7（选做）多轮样本的 assistant mask。** 写 `encode_multiturn(tokenizer, messages)`：对每一个 assistant 回答都做监督。提示：逐轮渲染"到这一轮 assistant 开头为止"的前缀，用 19.2.3 节的方法核对边界；注意 Qwen3 模板对历史轮次和当前轮次的处理不同（19.10 节）。自检：把所有 label 不为 −100 的片段解码出来，应当恰好是各轮 assistant 内容加 EOS。

**练习 19.8（选做，需要 GPU）把漏 EOS 推到极限。** 以 `run_gpu.sh` 里 no_eos 组的故障对照命令为起点（参数说明见 lab 第 4、5 节），加大步数或学习率，观察"达到预算"数和平均生成长度何时开始变化。把你的设备、精度、步数、学习率和结果记进自己的实验记录。本书不提供这组数字，结果由你自己跑出来。

---

## 四、调试题

**19.18** 同学的训练日志里，train loss 从 15 多降到了 0.2，看起来很顺利；可生成的公式全是乱码，像是"漏了一半字符"，而且常常写到生成上限。他的 collate 函数里有这样一段：

```python
labels = torch.full_like(ids, -100)
labels[row, :n-1] = torch.tensor(ex.labels[1:])
```

问题出在哪里？在训练之前，哪一个计数就能发现它？

**19.19** 另一位同学用一个没有 pad token 的模型做 SFT，按网上的写法设了 `tokenizer.pad_token = tokenizer.eos_token`，再用 `DataCollatorForLanguageModeling(tokenizer, mlm=False)` 组 batch，提示部分也自己设成了 −100。训练后模型回答正确，但每次都写到 `max_new_tokens` 才停。最可能的原因是什么？怎样改？

**19.20** 一个训练脚本把 batch 从 `8 × 累积 2` 改成 `4 × 累积 4`，有效 batch 都是 16，loss 曲线却明显变了，而且答案短的样本似乎学得更快。检查代码发现：

```python
for b in batches:
    loss = causal_loss(model(**b).logits, b["labels"])   # reduction="mean"
    (loss / accumulation).backward()
```

解释现象，并给出修改。

**19.21** 有人这样算答案边界：

```python
prompt = render_chat(tok, zero_shot(source))
prompt_length = len(prompt)
labels = [-100] * prompt_length + ids[prompt_length:]
```

训练能跑，但模型学不会任务。错在哪里？用 19.2.2 节的样本说明 `prompt_length` 实际是多少、应该是多少，以及后果。

**19.22** 训练时用 `render_chat(tok, messages)`（默认 `enable_thinking=False`），评测时为了"让模型想得更清楚"改成了 `enable_thinking=True`。val NLL 很好，生成却经常先写一大段英文再给公式，EM 大跌。解释原因。

**19.23** 同学的报告里写道："SFT 后抽取口径的 EM_norm 只有 110/500，比照抄的 191/500 还低，SFT 失败了。"请指出问题，并说明应该怎样报告。（提示：本项目观察，val 前 500 条里有 361 条目标以句号、逗号或分号结尾。）再看同一批 val 上的少样本 + 格式约束提示：原始口径 EM_norm 25/500，抽取后 32/500（本项目观察，`baselines/baseline-few_shot_strict.json`）。为什么提示基线经过抽取分数**升高**，SFT 经过抽取分数却**降低**？

---

## 五、闭卷手写

不看书和代码，在纸上写出：

**19.24** `encode_sft` 的核心部分（约 10 行）：渲染、联合编码、边界检查、追加 EOS、超长拒收、构造 labels。

**19.25** `causal_loss`（约 6 行），包括移位、掩码、全为 −100 时报错。

**19.26** 带梯度累积的一次更新（约 10 行），要求按受监督 token 总数加权。

**19.27** 画出这条样本的逐 token 表（列：位置、token、label、是否参与 loss、由哪个位置的 logits 预测）：提示是 `[A, B, C]`，答案是 `[d, e]`，末尾加 EOS。

---

## 答案与提示

**19.1** `input_ids`：每个位置是哪个 token；`attention_mask`：是真 token 还是 padding；`labels`：要不要学、目标是谁。提示 token：attention 1，label −100。答案 token：attention 1，label 是它自己的 ID。padding：attention 0，label −100。

**19.2** (a) 移了两次，位置 t 的 logits 被训练去预测位置 t+2 的 token（19.9.2 节）。训练 loss 照样下降，生成会跳 token。(b) loss 函数里不再移位，直接对齐 `logits` 和 `labels` 的同一位置，并且要确保最后一个位置的 label 是 −100。两种约定都行，但全项目只能选一种，并用测试固定下来（如 `test_hf_causal_lm_loss_uses_same_single_shift`）。

**19.3** 目标是 log p(y|x) = Σ_j log p(y_j | x, y_<j)，取负并除以答案 token 数。监督提示后变成 log p(x, y) = log p(x) + log p(y|x)。多出的 log p(x) 是"生成用户输入"的能力，对后纠错就是"生成带噪 OCR 公式"，没用，而且方向相反。

**19.4** 不对。以提示 token 为目标的那几项没有 loss（提示的最后一个位置例外：它预测第一个答案 token，照常有 loss），但答案 token 通过注意力读取提示的表示，梯度会流回提示位置的隐藏状态和所有层的权重。被屏蔽的只是"以提示 token 为目标的 −log p"那几项。

**19.5** 模型靠"在最后一个答案 token 之后预测 EOS"来学会停；不监督这一项，训练就不提供"停下"的信号。微型模型从随机初始化开始，从没学过停，所以不停；Qwen3 在自己的后训练里已经学会在 assistant 回答后写 `<|im_end|>`，本章 80 步、lr 2e-5 没有把这个习惯抹掉。不能说无害：信号确实没了，换更长的训练、更大的学习率或不太会停的基础模型，结果可能不同（待验证）。而且 no_eos 组的正确 labels NLL 也略高（0.058 对 0.048，本项目观察）。

**19.6** 训练时右 padding，真实 token 从位置 0 开始，因果注意力让它们看不到右边的 pad，pad 位置的 label 是 −100。生成时每条样本要在"最后一个位置"之后接新 token；右 padding 会让短样本的新 token 接在 pad 后面。左 padding 让所有样本的真实结尾对齐在最右边。

**19.7** 截断会丢掉答案后半段和 EOS，模型学到的是残缺的目标，也失去了"停下"的监督（直接截断不会产生"在截断处预测 EOS"的训练项）；截断后强加 EOS 更糟，等于明确教它"半截就是完整答案"。拒收损失的是覆盖面，被拒的总是最长的样本；拒收多时，模型对长公式见得少，读者需要知道这一点。本章主训练拒收 3/4096（本项目观察）。

**19.8** 不矛盾。① NLL 是每个 token 的平均，一条答案平均约 74 个 token，大多数是照抄、几乎零损失，决定对错的只有少数位置，平均值会被大量"容易 token"压低。② NLL 用 teacher forcing，每一步都喂正确前文；EM 是自回归生成，一步错了后面都在错的前文上继续。

**19.9** 各组训练时的目标不同：prompt 组的分母多了 90584 个提示 token，shift 组的目标是"下下个 token"。训练 loss 衡量的是"拟合自己的目标有多好"，不同目标之间没有可比性（prompt 组的训练 loss 在真实模型上更高、在微型模型上更低，方向都能反）。固定正确 labels 的 NLL 衡量的是同一件事：模型在正确答案上的预测能力。

**19.10** (a) 选 checkpoint 用的 val 前 128 条，正是这 500 条的前 128 条。在参与过选择的数据上报分，会偏乐观一点。(b) 本章仍在用 val 比较和选型，未评测试集。按第 0 章 §0.10.1，选型冻结后，预先登记的候选可各评一次，只报告数字和置信区间，不据此改任何设置。lab 第 0 节末尾给出了选中 checkpoint 的评测命令。开发期不看 test 错例，最终报告定稿后才可解封。

**19.11** (a) `input_ids = [p0, p1, p2, p3, p4, 124, 98, 54, 1]`，`labels = [−100, −100, −100, −100, −100, 124, 98, 54, 1]`。
(b) `labels[1:]` 中不为 −100 的是原位置 5、6、7、8，由位置 4、5、6、7 的 logits 预测，目标依次是 124、98、54、1，共 4 项。
(c)
- shift：`labels = [−100, −100, −100, −100, 124, 98, 54, 1, −100]`。`supervised_tokens` = 4，`supervised_prompt_tokens` = 1（位置 4，在 `labels[1:5]` 里），`supervised_terminal_eos` = 0（最后一个 label 是 −100）。
- prompt：`labels = input_ids`。`supervised_tokens` = 8（`labels[1:]` 共 8 个，全不为 −100），`supervised_prompt_tokens` = 4（位置 1–4），`supervised_terminal_eos` = 1。
- no_eos：`input_ids = [p0, …, p4, 124, 98, 54]`，`labels = [−100×5, 124, 98, 54]`。`supervised_tokens` = 3，`supervised_prompt_tokens` = 0，`supervised_terminal_eos` = 0。
和 `fault_*/train_config.json` 对照：真实 shift 组提示监督数恰好等于样本数 1022，EOS 监督数 0，规律相同。

**19.12** (a) A 长 9，B 长 7（5 + 1 + EOS），最长 9，向上取整到 4 的倍数是 12。
(b) `input_ids = [q0, q1, q2, q3, q4, 101, 1, 0, 0, 0, 0, 0]`；`attention_mask = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]`；`labels = [−100×5, 101, 1, −100×5]`。
(c) `logits[:, :-1]` 是 `(2, 11, V)`，`labels[:, 1:]` 是 `(2, 11)`。A 有 4 个、B 有 2 个受监督 token，挑出后是 `(6, V)`。
(d) 两条样本的真实 EOS（ID 1）都会被当成 pad 设成 −100，少掉 2 个 EOS 监督，模型学不到"停"。A 也有 3 个 padding（B 有 5 个），但误屏蔽按 ID 进行，两条样本的真实 EOS 都被误伤。

**19.13** (a) labels = `[−100, −100, c, d, e]`。位置 1 预测 c，位置 2 预测 d，位置 3 预测 e。
(b) shift 后 labels = `[−100, c, d, e, −100]`。loss 让位置 t 的 logits 预测 `labels[t+1]`：位置 0 预测 c，位置 1 预测 d，位置 2 预测 e。概括：位置 t 的 logits 被训练去预测 input 中位置 t+2 的 token。

**19.14** microbatch 1：10 个 token，和 3.2，mean 0.32；microbatch 2：70 个 token，和 7.0，mean 0.10。
(a) (0.32 + 0.10) / 2 = 0.21。
(b) 各序列 mean：0.5、0.2、0.1、0.1，平均 0.225。
(c) (3.2 + 7.0) / 80 = 0.1275。
只有 (c) 等于拼成一个大 batch 的 mean。(a) 中 microbatch 1 的一个 token 权重是 1/2 × 1/10 = 1/20，microbatch 2 的是 1/2 × 1/70 = 1/140，前者是后者的 7 倍。

**19.15** (a) 均匀猜 4 个字母的那一项是 ln 4 ≈ 1.386，其余项约为 0。correct 组每条 4 项：1.386 / 4 ≈ 0.347，和观察值 0.347 吻合。no_eos 组每条只有 3 项（没有 EOS）：1.386 / 3 ≈ 0.462，和观察值 0.463 吻合。
(b) 正确 labels 有 4 项，总损失约 4 × 4.04 = 16.16，减去字母那一项 1.386，EOS 那一项约 14.8，p(EOS) ≈ e^(−14.8)，约 4 × 10⁻⁷。模型几乎认为"这里绝不该停"。这是在假设下的估算，不是实测的 EOS 概率。

**19.16** (a) 66096 / 156680 ≈ 0.422。
(b) `labels[1:prompt_length]` 少数了位置 0，所以每条样本的提示长度 = 计数 + 1：90584 / 1022 + 1 ≈ 89.6 个 token。答案含 EOS：66096 / 1022 ≈ 64.7 个 token。

**19.17** (a) 596,049,920 × 16 字节 = 9,536,798,720 字节 ≈ 8.88 GiB。
(b) 8 × 256 × 151936 × 2 字节 = 622,329,856 字节 ≈ 0.58 GiB。
(c) 每一层的激活值（注意力、MLP 的中间结果，要留到反向时用）、交叉熵前转成 fp32 的那部分 logits、logits 的梯度、autocast 产生的 bf16 权重副本、CUDA 分配器的碎片等。第 20 章会细算。

**19.18** collate 时又把 labels 左移了一位（`ex.labels[1:]` 写到 `:n-1`），loss 函数再移一次，变成跳一格预测。训练前看 `supervised_terminal_eos`：应当等于样本数，这里会是 0；`supervised_prompt_tokens` 也会变成每条 1 个。逐 token 打印一条样本也能一眼看出：第一个非 −100 的 label 落在了提示的最后一个位置。

**19.19** pad 和 EOS 是同一个 ID，`DataCollatorForLanguageModeling` 按 ID 把等于 pad 的位置设为 −100，每条样本真正的 EOS 也被屏蔽了，模型没学过"停"。改法：按位置屏蔽 padding（像 `collate_sft` 那样）；或者自己构造好 labels 后用只补齐、不改 labels 的 collator（如 `DataCollatorForSeq2Seq`）。还有一个隐藏的坑：`DataCollatorForLanguageModeling(mlm=False)` 会重新执行 `labels = input_ids.clone()`，把你事先设好的提示 −100 也覆盖掉，所以这位同学其实连提示也在监督。只换一个独立的 pad token 能救回 EOS，却救不回提示 mask；要么换成保留既有 labels 的 collator，要么在 collate 之后重建提示 mask。改完用"pad = EOS"的设置跑一遍 `test_padding_when_pad_equals_eos_and_alignment` 式的检查。

**19.20** 每个 microbatch 先求 mean 再平均，是"平均的平均"：受监督 token 少的 microbatch，每个 token 权重更大；改变 microbatch 的划分，权重就跟着变，所以曲线变了，短答案被放大。改法：先数这次更新所有 microbatch 的受监督 token 总数 `count`，每个 microbatch 用 `reduction="sum"`，再 `(loss_sum / count).backward()`（19.6.5 节）。

**19.21** `len(prompt)` 是**字符数**，不是 token 数。19.2.2 节那条样本的提示文本有 159 个字符，却只有 32 个 token（本项目观察）。于是 `prompt_length = 159`，比整条序列的 40 个 token 还长：`ids[159:]` 是空的，labels 变成 159 个 −100，既没有任何受监督 token，长度也和 `input_ids` 对不上。`collate_sft` 会因为"长度不匹配或没有可预测的监督 token"直接报错；如果用的是不做检查的代码，提示短、答案长的样本可能还剩一小段受监督，模型学到的是错位的、残缺的目标。改法：用 token 层面的前缀长度（19.2.3 节）。

**19.22** 训练时，答案接在空思考块 `<think>\n\n</think>\n\n` 之后；打开思考后，生成前缀里没有这个空块，模型要在 `<think>` 后面自己写思考内容，这是 SFT 从没练过的上下文，于是回到原来"先讲一段再给答案"的习惯。val NLL 用的是训练时的模板，所以看不出问题。训练和推理必须用同一个模板（19.2.4 节）。

**19.23** 抽取器会去掉句末标点，而 361/500 条目标保留了句末标点，SFT 本来逐字正确的输出被"抽"错了。SFT 的输出是干净的一行公式，应该报原始输出口径：365/500，高于照抄的 191/500（本项目观察）。报告里原始和抽取两套分数都保留，写明各自口径，不能择优混用，也不能把抽取口径当成 SFT 的成绩。提示基线的输出常带解释和 `$`（格式约束提示的 500 条里仍有 340 条包了 `$`，本项目观察），抽取器去掉这些包装后有一部分变对了，收益大于去标点的损失，所以分数升高；SFT 的输出本来就没有包装，抽取带来的只有损失：主要是去掉句末标点，另外抽取器遇到公式内部的 `$…$`（如 `\mbox{in $V^*$}`）时会只取这一小段、丢掉其余公式（`final/final-val.jsonl` 第 118 行被抽成 `V^*`，第 215 行被抽成 `r=1`，本项目观察），所以分数降低。同一个抽取器对不同方法的影响方向可以相反，这正是两套口径都要报的理由。

**19.24–19.27** 对照正文 19.3.5、19.3.6、19.6.5 节的代码和 19.2.2 节的表格自查。19.27 的答案：位置 0–2 是 A、B、C，label 全为 −100；位置 3、4、5 是 d、e、EOS，label 是它们自己；参与 loss 的是位置 3、4、5，分别由位置 2、3、4 的 logits 预测；位置 5 自己的 logits 不参与 loss。
