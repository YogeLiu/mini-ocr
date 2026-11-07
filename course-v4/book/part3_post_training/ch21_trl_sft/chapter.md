# 第 21 章 用 TRL 做 SFT 实验

## 本章导读

第 19 章你亲手写了 SFT：chat template、label mask、EOS、按 token 加权的梯度累积。第 20 章你亲手写了 LoRA，还和 PEFT 逐位对拍过。现在该换上工具了。真实项目里，没人每次都从头写训练循环，大家用 Hugging Face 的 TRL（Transformer Reinforcement Learning）库，它的 `SFTTrainer` 几十行配置就能跑起一次 LoRA SFT。

框架省掉的是体力活，省不掉检查。`SFTTrainer` 替你做了数据整理、loss、优化器、梯度累积、保存，每一件都可能和你以为的不一样，而且出错时训练照样跑、loss 照样降。所以本章不细讲 TRL 的几百个参数，只走一遍七项**检查清单**（21.1.3 节）：看第一批数据，确认 completion-only 和可训练参数，把 loss 和自写循环对拍，保存后重载复现，在固定 val 上比较微调前后，最后用数据说明训练 loss 下降不等于新样本变好。

主实验走完这张清单，结果是这样（本项目观察，RTX 4090 租用）：在 val 前 200 条上，选中的第 400 步 EM_norm 是 131/200，照抄是 73/200，最强的提示是 8/200；test 评一次得 123/200；代价是改坏了 7/59 条无错输入。第 400 步以后，训练集上的 NLL 继续下降，val 不再改善。

学完本章，你应该能做到：

1. 用 TRL 的 `SFTTrainer` + PEFT 在 Qwen3-0.6B 上做一次 LoRA SFT。
2. 逐项执行这张检查清单，说出每一项查的是什么、不查会出什么事。
3. 保存 adapter 和完整配置，让别人（或一个月后的你）能复现。
4. 在固定 val 上比较微调前后，解释为什么训练 loss 一路下降而 val 指标在中途就停了。

**和前面章节的关系。** 本章不引入新的数学：label mask 和 EOS 来自第 19 章，LoRA 的公式和参数量来自第 20 章，评测脚本和提示基线来自第 18 章。本章新学的是怎样确认框架做的就是你学过的那件事。

**和作业、项目的关系。** 本章的实验是作业 A3 的**训练层**：OCR 后纠错在验证集上超过提示基线（门槛待校准）。它也是项目报告 R2 结果表里 **TRL LoRA** 那一行（第 18 章填提示基线，第 19、20 章填全参 SFT 和自写 LoRA）。

**怎样读。** 21.1 到 21.2 节搭骨架；21.3 到 21.9 节每节讲清单上的一项；21.10 节把本章结果和第 18 到 20 章放在一起比；21.11 节概括 TRL 的其他参数。动手步骤都在 lab.md 里，正文只引用 lab 的部分号。

---

## 21.1 为什么用框架，又为什么不能全信框架

### 21.1.1 自写循环做到了什么，框架多给了什么

第 19、20 章的自写代码已经能训练，那为什么还要换 TRL？因为一个能长期用的训练实验还需要很多周边功能：

- 断点保存与恢复、日志、回调（每隔多少步做点事）；
- 混合精度、梯度检查点、多卡并行；
- 各种数据格式（纯文本、对话、prompt-completion）的自动处理；
- 和 PEFT、transformers 的版本配套。

这些自己写也能写，但每一样都要调试。TRL 把它们包好了，而且整个社区都在用，出了问题容易查到。

### 21.1.2 框架替你做的事，就是可能出错的事

打个比方。自写循环像手动挡：每一次换挡都是你踩的离合，哪里不对你心里有数。框架像自动挡：开起来轻松，但它在什么时候换挡、换到几挡，你不看仪表盘就不知道。车照样往前走，出了问题也不会停下来告诉你。

具体到 SFT，框架会替你做下面几件事。每一件都对应第 19 章讲过的一个坑：

| 框架替你做的事 | 第 19 章的坑 | 出错时训练会报错吗 |
|---|---|---|
| 套 chat template、分词 | 训练和推理模板不一致 | 不会 |
| 决定哪些 token 计 loss | 提示也计入 loss；label 错一位 | 不会 |
| 补 padding、截断 | pad 被监督；截断砍掉 EOS | 不会 |
| 梯度累积时怎样平均 loss | "平均的平均"让短答案权重变大 | 不会 |
| 挂 LoRA、冻结基座 | 基座没冻住，变成全参训练 | 不会，只是更慢、更占显存 |

最后一列全是"不会"。第 19 章 19.9 节的故障对照已经演示过：label 错一位、监督提示、漏 EOS，训练 loss 都不报警。所以检查只能靠你主动去看。

### 21.1.3 检查清单一览

下面这张表是本章的地图。每一行是一节，"证据"一列写的是结果存在哪个文件里。

| 检查项 | 怎么查，什么算通过 | 证据（tour 输出文件） | 节 |
|---|---|---|---|
| 第一批 token 与 label | 从 `trainer.get_train_dataloader()` 取一批，逐位置打印 id、token、label、attention；提示和 pad 的 label 是 −100，答案和末尾 EOS 的 label 等于自己的 id | `first-batch.json`、`training-first-batch.json` | 21.3 |
| completion-only 生效 | 不看开关，看 labels：第一个监督位置等于提示长度，每条的监督数等于答案 token 数（含 EOS），提示监督数是 0 | 同上 + `audit_batch` 的断言 | 21.4 |
| 可训练参数 | 从 `trainer.model` 数 `requires_grad` 的参数：个数等于 Σ r × (in + out)，名字全部含 `lora_` | `trainable-parameters.json` | 21.5 |
| loss 计算方式 | 同一模型、同一批数据，TRL 和自写 `causal_loss` 各算一遍，loss、梯度、一步更新后的参数在容差内相等，且参数真的动了 | `part2.json` 的 `parity` | 21.6 |
| 保存与复现 | 重载 adapter，用完全相同的生成设置在同一批 val 上重新生成，逐条输出相同 | `reload-verified.json`、`environment-*.json` | 21.7 |
| 固定 val 上比较 | 微调前后用同一批题、同一评测脚本、同一解码和精度，选择规则在跑之前写好 | `stages.json`、`*-val-metrics.json` | 21.8 |
| loss 降 ≠ 变好 | 每隔固定步数在固定的 train 探针和 val 上算 NLL，和 val EM 分开画，看 train 与 val 是否分叉 | `stages.json`、`trainer-history.json` | 21.9 |

前四项在训练**之前**做（lab 第 1、2 部分），后三项在训练**之后**做（lab 第 3 到 6 部分）。数据和 loss 没确认之前，跑出来的分数不能信。

每一项的思路都一样：查框架处理之后的对象（dataloader 吐出的 batch、`trainer.model`、重载后的模型），而不是你传进去的那个；拿它和一个独立的参照比（第 19 章的编码、第 20 章的公式、自写的 loss、保存前的输出）；把比较写成断言，不靠肉眼。

学习率调度和训练日志不单列一项，但也要看：调度器类型和 warmup 在 21.2.4 节的配置表里核对，实际用的学习率和日志 loss 的含义在 21.6.6 节核对。

---

## 21.2 一次 TRL SFT 的骨架

### 21.2.1 三个对象

用 TRL 做 LoRA SFT，你要准备三个对象：

- **`SFTConfig`**：训练怎么跑。它继承自 transformers 的 `TrainingArguments`，所以学习率、batch、步数、精度这些参数和普通 `Trainer` 一样，另外加了 SFT 专用的（比如 `completion_only_loss`、`packing`、`max_length`）。
- **`LoraConfig`**（来自 PEFT）：LoRA 挂在哪、rank 多大。
- **`SFTTrainer`**：把模型、分词器、数据集和上面两个配置装在一起，调用 `.train()` 开跑。

本章的代码在 `miniocr/train/trl_sft.py` 的 `make_trainer`。去掉细节，骨架就是这样：

```python
config = SFTConfig(output_dir=..., max_steps=steps,
    per_device_train_batch_size=batch_size, gradient_accumulation_steps=accumulation,
    learning_rate=lr, lr_scheduler_type='constant', ...,
    max_length=None, packing=False, padding_free=False, completion_only_loss=True,
    dataset_kwargs={'skip_prepare_dataset': True}, remove_unused_columns=False, ...)
SFTTrainer(model=model, args=config, processing_class=tokenizer,
    train_dataset=Dataset.from_list(dataset_rows(examples)),
    peft_config=LoraConfig(r=rank, lora_alpha=2*rank, lora_dropout=0.,
                           bias='none', target_modules=['q_proj','v_proj'],
                           task_type='CAUSAL_LM'))
```

注意 `peft_config` 是直接交给 `SFTTrainer` 的。你传进去的是普通的 Qwen3 模型，`SFTTrainer` 在内部把它包成 PEFT 模型。这一步你看不见，所以 21.5 节要专门检查。

### 21.2.2 把第 19 章的数据交给框架

TRL 能吃好几种数据格式。最省事的是直接给 prompt 和 completion 两段文字，让 TRL 自己套模板、分词、做 mask。本章**故意不这样做**，而是把第 19 章 `encode_sft` 已经编码好的 `input_ids` 和 `labels` 直接交给它：

```python
def dataset_rows(examples):
    rows = []
    for ex in examples:
        collate_sft([ex], 0)  # 使用既有合同校验，不重新实现 mask。
        rows.append(dict(input_ids=list(ex.input_ids), labels=list(ex.labels)))
    if not rows:
        raise ValueError('没有训练样本')
    return rows
```

再配上 `dataset_kwargs={'skip_prepare_dataset': True}`，告诉 TRL："数据我已经准备好了，你别再套模板、别再截断。"

为什么这样做？理由有三：

1. **模板只有一个来源。** 第 18 到 20 章的评测用的就是 `encode_sft` 背后的那个非思考模板。训练也用它，训练和推理就不可能对不上。
2. **mask 由你负责，而且你已经测过。** 第 19 章的 `encode_sft` 有完整的测试（联合编码、核对前缀边界、EOS）。让框架再做一遍，就多了一个需要核对的实现。
3. **对拍才有意义。** 21.6 节要拿 TRL 的 loss 和自写 `causal_loss` 比。两边吃的是同一批 `input_ids/labels`，比的才是 loss 本身。

`dataset_rows` 只做一件事：格式转换。它调用 `collate_sft([ex], 0)` 借用第 19 章已有的合同检查（形状一致、不能全是 −100），而不是自己再写一遍 mask 逻辑。这是练习 21.1 的一部分。

这个选择还需要两个配置配合：

- `max_length=None`：第 19 章的 `encode_sft` 已经把超过 384 个 token 的样本**整条拒收**了，不需要框架再截断。截断会砍掉答案结尾的 EOS，第 19 章 19.6.4 节讲过。
- `remove_unused_columns=False`：别让 Trainer 按模型 `forward` 的签名删列，`labels` 列必须留着。

### 21.2.3 实验设置

本章后面的实测数字，除非特别说明，都来自同一次运行。为了不每次都写一长串，下文用"**主实验**"指代它：

> **主实验**：本项目观察，RTX 4090（租用）；Qwen3-0.6B 本地快照，权重、LoRA 参数和 AdamW 状态都是 FP32，训练和生成用 bf16 autocast；postcorrect-v1 数据（合成噪声，seed 0），train 前 2048 条、长度过滤后接受 2023 条，val 前 200 条；LoRA 挂 q_proj、v_proj，r = 32，α = 64，dropout 0；800 步，microbatch 2 × 累积 4（有效 batch 8），学习率 2 × 10⁻⁴ 常数，AdamW（weight decay 0.01，裁剪 1），梯度检查点开启；每 200 步保存并评一次 val。原始文件在 `outputs/book/ch21/gpu/`。

几点补充：

- **数据身份**：`data-manifest.json` 记了 train 和 val 样本 ID 的 SHA-256。val 的哈希和第 20 章 `full/data-manifest.json` 里的一致，所以本章的 val 和第 20 章是**同一批 200 条**（本项目观察）。
- **训练量**：名义上 800 × 8 = 6400 次样本，约 3.16 轮（数学结论：6400 / 2023）；选中的第 400 步名义上 3200 次样本，约 1.58 轮。实际处理量略少：`dataloader_drop_last=false`，2023 条按 microbatch 2 切成 1012 批，每轮最后一批只有 1 条，`Trainer` 又按轮划分更新（每轮 253 次更新、2023 条），所以第 400 步实际是 2023 + 147 × 8 = 3199 次，第 800 步是 3 × 2023 + 41 × 8 = 6397 次（数学结论，由配置推算）。轮数不受影响：400 / 253 ≈ 1.58，800 / 253 ≈ 3.16。
- **生成**：贪心、非思考模板、左 padding、每条最多 192 个新 token、生成批大小 8，和第 20 章一致。
- **两个评分口径都存**：原始输出（raw）和经第 18 章 `extract_formula` 抽取后的（extracted）。主指标在跑之前就定为**原始输出的 EM_norm**（21.8 节解释为什么）。
- **CPU / MPS / 5080 没有跑同规模实验**。lab 第 0 节给了 CPU 小规模命令，只验证流程，不能当效果结论。

### 21.2.4 我们动了哪些参数

`SFTConfig` 有一百多个字段（`sft-config.json` 里全存着）。本章显式设置的只有下面这些，按用途分组：

| 用途 | 字段 | 本章取值 | 为什么 |
|---|---|---|---|
| 训练量 | `max_steps`、`per_device_train_batch_size`、`gradient_accumulation_steps` | 800、2、4 | 有 `max_steps` 时，`num_train_epochs`（文件里是默认的 3.0）不起作用 |
| 优化 | `learning_rate`、`lr_scheduler_type`、`warmup_steps`、`weight_decay`、`optim` | 2e-4、constant、0、0.01、adamw_torch | 和第 20 章一致，便于比较。调度不设时默认是 `linear`，见 21.6.6 节 |
| 精度 | `bf16`、`fp16` | True、False | 权重仍是 FP32，计算走 bf16 autocast |
| 省显存 | `gradient_checkpointing` | True（`use_reentrant=False`） | 用算力换激活显存 |
| 数据 | `max_length`、`packing`、`padding_free`、`skip_prepare_dataset`、`remove_unused_columns` | None、False、False、True、False | 21.2.2 节 |
| loss | `completion_only_loss` | True | 对本章的数据路径**不起作用**，21.4 节解释 |
| 可复现 | `seed`、`data_seed` | 0、0 | |
| 日志 | `logging_steps`、`logging_first_step`、`report_to` | 10、True、none | 不往外部服务发数据 |
| 保存 | `save_strategy` | no | 由回调在每 200 步自己保存 adapter |

**没有显式设置的字段，用的都是这个版本的默认值。** 例如 AdamW 的 eps 是 1e-8、β 是 (0.9, 0.999)、梯度裁剪 1.0（`sft-config.json`）。默认值会随版本变，所以 21.7 节要把完整配置存下来。

---

## 21.3 检查一：打印第一个 batch 的 token 与 label

### 21.3.1 要看送进模型的那一批

数据出错时训练不报警，最直接的办法是在第一步之前，把要送进模型的那一批数据逐 token 打出来，用眼睛看一遍。

你自己的数据集长什么样不算数，要看的是经过框架的 dataloader 和 collator 以后变成了什么。中间可能被补 pad、被截断、被重新套模板、被打包（packing）。所以要从 `trainer.get_train_dataloader()` 里取，不从你自己的列表里取。

### 21.3.2 步骤

1. 用和正式训练**完全相同**的配置建好 trainer。
2. `batch = next(iter(trainer.get_train_dataloader()))`。
3. 对 batch 里的每一行，逐位置打印：位置、token id、token 文字、label、attention。
4. 用肉眼和断言各查一遍：
   - 提示部分（system、user、assistant 开头、空的思考块）的 label 全是 −100；
   - 答案的第一个 token 开始，label 等于 input_id；
   - 答案最后一个 token 是 EOS，它的 label 不是 −100；
   - padding 位置 attention = 0、label = −100。
5. 数一数这一批一共监督了多少个 token。

### 21.3.3 实例：主实验的第一批

主实验正式训练的第一批有 2 条样本，补齐到 263 个位置（本项目观察，主实验，`training-first-batch.json`）。

**第 1 条**是一条无错样本：带噪输入和目标完全相同，模型要学的是原样抄回来。它的真实长度是 231，后面补了 32 个 pad。把位置 122 到 130 和末尾几个位置摘出来：

| 位置 | token | label | attention |
|---:|---|---:|---:|
| 122 | `assistant` | −100 | 1 |
| 123 | `Ċ`（换行） | −100 | 1 |
| 124 | `<think>` | −100 | 1 |
| 125 | `ĊĊ` | −100 | 1 |
| 126 | `</think>` | −100 | 1 |
| 127 | `ĊĊ` | −100 | 1 |
| **128** | **`P`** | **47** | 1 |
| 129 | `=\` | 34433 | 1 |
| … | … | … | … |
| 230 | `<\|im_end\|>` | **151645** | 1 |
| 231 | `<\|endoftext\|>` | −100 | **0** |
| … | （pad，一直到 262） | −100 | 0 |

读这张表：

- 位置 0 到 127 是提示：system 消息 "Correct OCR errors in the following LaTeX formula."、user 消息里的带噪公式、`assistant` 开头，以及非思考模板自动加的空思考块 `<think>\n\n</think>\n\n`。全部 −100。
- 位置 128 是答案的第一个 token `P`，label 就是它自己的 id 47。**label 不预先移位**：第 19 章 19.3.3 节说过，移位在 loss 函数里做一次。
- 位置 230 是 `<|im_end|>`，id 151645，这就是 Qwen3 对话模板里的 EOS。它被监督了，模型才学得会写完就停。
- 位置 231 起是 pad。Qwen3 的 pad 是 `<|endoftext|>`（id 151643），和 EOS 不是同一个 token，所以屏蔽 pad 不会误伤 EOS（第 19 章 19.5.3 节的陷阱这里不存在）。

**第 2 条**是一条有错样本，真实长度正好 263，没有 pad。答案从位置 140 开始，到位置 262 的 `<|im_end|>` 结束。对比输入和答案能看到两处错：带噪输入里 `\frac{1}` 后面少了分母，目标是 `\frac{1}{Q(D,x)-P(x)}`；带噪输入里的 `\frac{l}{n}`，目标是 `\frac{1}{n}`。

监督 token 数：第 1 条 103 个（含 EOS），第 2 条 123 个，这一批合计 226 个（本项目观察，主实验，`training-first-batch.json` 的 `supervised_tokens`）。

### 21.3.4 对照代码：`audit_batch`

把用眼睛看的那几条写成断言，就是 `audit_batch`（`miniocr/train/trl_sft.py`，练习 21.1）。逐行看它查了什么：

```python
ids, labels, attention = (batch[k].cpu() for k in ('input_ids', 'labels', 'attention_mask'))
assert ids.shape == labels.shape == attention.shape
assert (labels[attention == 0] == -100).all()                 # pad 不监督
assert torch.equal(ids[labels != -100], labels[labels != -100])  # 监督位置 label == id，没有错位
```

第三个断言能抓住 label 错一位：如果有人把 labels 预先左移了一位，监督位置上的 label 就会等于**下一个** token 的 id，和 input_id 对不上。

接下来逐行和第 19 章编码的原样本对照：

```python
expected = {tuple(e.input_ids): e for e in examples}
for i in range(len(ids)):
    active = torch.where(labels[i] != -100)[0]
    assert len(active) and active[0] > 0                    # 有监督，且不是从位置 0 开始
    end = int(attention[i].sum()) - 1                       # 右 padding：最后一个真实位置
    reference = expected[tuple(ids[i, :end+1].tolist())]    # 找回这条样本的原始编码
    assert int(active[0]) == reference.prompt_length        # 答案起点 = 提示长度
    assert labels[i, :end+1].tolist() == list(reference.labels)
    assert labels[i, end] == tokenizer.eos_token_id         # 最后一个真实 token 是 EOS 且被监督
    assert (labels[i, active[0]:end+1] != -100).all()       # 答案中间没有洞
```

注意 `end = attention.sum() − 1` 依赖**右 padding**。训练用右 padding、生成用左 padding，是第 19 章 19.6.1 节的约定；TRL 的 collator 在训练时也是右补齐（看实例表里 pad 都在末尾）。

最后，逐 token 记录写成 `rows`，监督 token 数按 `labels[:, 1:]` 数：

```python
return dict(supervised_tokens=int((labels[:,1:] != -100).sum()), rows=rows, ...)
```

为什么从位置 1 开始数？因为位置 0 的 label 没有任何 logits 去预测它（因果对齐：位置 t 的 logits 预测位置 t+1 的 token）。这和 transformers 的 `Trainer` 数分母的方式一致（21.6 节）。

tour 的第 2 部分还多做了一步：用同样的两条样本，分别经过 TRL 的 collator 和第 19 章的 `collate_sft`，要求得到的三个张量**逐元素相等**（`torch.testing.assert_close`）。这证明 TRL 在这条路径上没有改动你的数据。

### 21.3.5 会出什么错

| 错误 | 现象 | 怎样发现 |
|---|---|---|
| 从自己的列表里打印，而不是从 dataloader | 你看到的是对的，送进模型的是另一回事 | 只认 `get_train_dataloader()` 的输出 |
| 只打印 `decode` 后的文字 | 看不到 label，也看不到特殊 token 的位置 | 逐位置打印 id、token、label、attention 四列 |
| 只看第一条 | 第一条恰好没有 pad，pad 的问题看不出来 | batch 里每一行都过一遍断言 |
| 以为 EOS 是 `<\|endoftext\|>` | 断言失败或 EOS 未被监督 | 打印模板末尾，看对话模板实际用哪个 token 收尾 |
| 打印用的配置和正式训练不同 | 例如打印时 batch=1，看不到 pad | 第一批要用正式配置的 trainer 取（tour 第 3 部分保存的 `training-first-batch.json` 就是这样取的） |

最后一行是本实验里真实存在的区别：tour 第 2 部分的检查 trainer 用的是 batch 1，所以 `first-batch.json` 里只有一条、没有 pad；第 3 部分才用正式配置（batch 2）取了一次，保存为 `training-first-batch.json`。两份都要看。

> **停下来想一想：** 如果第一批的两条样本恰好一样长、都没有 pad，你的检查漏掉了什么？怎样构造一批数据，保证 pad 那条断言真的被执行到？

---

## 21.4 检查二：确认 completion-only loss 真的生效

### 21.4.1 开关不等于效果

`SFTConfig` 里有一个开关 `completion_only_loss=True`，字面意思是只在 completion（答案）上算 loss。很多人打开它就放心了。可开关只表明意图，completion-only 是否生效，要看送进模型的 labels。

### 21.4.2 这个版本里，开关到底管什么

我们读了实例上安装的 TRL 1.15.0 源码（`trl/trainer/sft_trainer.py`），并对照它的文档字符串。在纯文本 SFT 这条路径上，labels 的来源是这样定的：

1. 如果数据集里**已经有 `labels` 列**，原样使用。
2. 如果没有，就在准备数据集阶段从 `completion_mask` 和 `assistant_masks` 列构造：被这些 mask 标为 0 的位置设为 −100。`completion_mask` **只在 `completion_only_loss=True` 时**才参与。
3. 如果这些列都没有，labels 默认等于 `input_ids`，也就是**整条序列都计 loss**。
4. 设了 `skip_prepare_dataset=True` 时，第 2 步整个跳过，也不做截断；collator 只按原样补齐 labels（用 −100 补）。

把本章的数据路径代进去：我们有 `labels` 列，又跳过了准备阶段，所以走的是第 1 条。**`completion_only_loss=True` 在本章根本没被用到。** 它留在配置里只是表明意图，保证只学答案的，是第 19 章 `encode_sft` 写好的 labels。

这一版还有一个保护：如果你跳过了准备阶段，数据里却只有 `completion_mask` 而没有 `labels`，TRL 会直接报错，提示"会在整条序列上训练"。但如果你连 `completion_mask` 也没给，只给了 `input_ids`，它不会报错，labels 就等于 `input_ids`，提示被悄悄监督了。

这些行为都是**版本相关**的。源码注释里就写着，构造 labels 的逻辑是从 collator 挪到数据准备阶段的；换一个版本，同样的配置可能走另一条路。所以：按你安装的版本查文档和源码，然后**用第一批数据验证**。

### 21.4.3 怎样查：主实验的第一批

查三个数：`audit_batch` 输出里的 `active[0]`（第一个被监督的位置）是否等于提示长度；每条样本被监督的 token 数是否等于答案的 token 数（含 EOS）；提示部分被监督的 token 数，应当是 0。

主实验第一批第 1 条：提示长度 128，答案（含 EOS）103 个 token，第一个被监督的位置正好是 128，监督数正好 103；第 2 条：提示长度 140，第一个监督位置 140，监督 123 个（本项目观察，主实验，`training-first-batch.json`）。提示部分被监督的 token 数是 0。

如果 completion-only 没生效，会是什么样？那第 1 条的监督数会变成 230（231 个位置减去位置 0，数学结论），其中 127 个属于提示。21.3.3 节那张表里，位置 0 到 127 的 label 列会变成各自的 token id，而不是 −100。一眼就能看出来。

### 21.4.4 对照代码

`audit_batch` 里专门查这一项的是两句：

```python
assert int(active[0]) == reference.prompt_length
assert labels[i, :end+1].tolist() == list(reference.labels)
```

第一句查答案起点；第二句更严格，要求整行 labels 和第 19 章编码时完全一致。返回值里的 `completion_only=True` 只有在所有断言都通过后才会出现，它不是从配置里抄来的。

测试 `test_ch21_collation_masks_eos_and_padding` 故意弄坏了两次：把提示位置的 label 改成一个 token id（模拟提示被监督），再把末尾 EOS 的 label 改成 −100（模拟 EOS 没被监督），两次都要求 `audit_batch` 抛出 `AssertionError`。一个检查只有在能抓住错误时才有用，所以要拿坏数据测它。

### 21.4.5 会出什么错

| 错误 | 现象 | 怎样发现 |
|---|---|---|
| 只设了 `completion_only_loss=True`，数据只有 `input_ids` 又跳过准备 | 提示被监督，训练照样跑；训练 loss 可能变高也可能变低，方向取决于提示有多难预测（第 19 章 19.9.3 节），不能据此判断 mask 对不对 | 第一批的提示位置 label 不是 −100 |
| 换了 TRL 版本，labels 的构造路径变了 | 同一份配置，监督 token 数变了 | 每次升级后重跑 tour 第 2 部分，比较 `supervised_tokens` |
| 让 TRL 自己套模板，模板和评测不同 | 训练 loss 正常，生成时格式异常 | 把训练第一批的文字和评测时 `render_chat` 的输出对比 |
| 答案起点算错一个 token（例如把 `ĊĊ` 也算进答案） | EM 下降不多，但输出开头常多一个换行 | `active[0]` 与 `prompt_length` 对比 |

---

## 21.5 检查三：可训练参数

### 21.5.1 可能错在哪

`peft_config` 交给 `SFTTrainer` 以后，包装模型、冻结基座都在框架里完成。可能出的错有两类：

- **该冻的没冻**：你以为在做 LoRA，其实在做全参训练。不会报错，只是更慢、更占显存，保存出来的 adapter 也不完整（基座的改动没存）。
- **该挂的没挂**：`target_modules` 的名字写错或换了模型架构，LoRA 只挂上了一部分，甚至一个都没挂。

所以训练前要核对两件事：**个数**和**名字**。

### 21.5.2 先用公式算

第 20 章 20.7 节的公式：LoRA 可训练参数 = Σ r × (in + out)，对每个挂了 LoRA 的线性层求和。Qwen3-0.6B 有 28 层，隐藏维 1024；`q_proj` 是 1024 → 2048，`v_proj` 是 1024 → 1024（第 20 章 20.7.2 节）。r = 32 时：

- 每层 q：32 × (1024 + 2048) = 98,304
- 每层 v：32 × (1024 + 1024) = 65,536
- 每层合计 163,840；28 层合计 **4,587,520**（数学结论）

α 不影响参数个数，只影响缩放（第 20 章 20.2.4 节）。

### 21.5.3 再看实测

`trainable-parameters.json` 记录了正式 trainer 里的模型（本项目观察，主实验）：

- 可训练参数 4,587,520，和公式一致。
- 可训练参数的名字一共 112 个，正好是 28 层 × 2 个模块 × 2 个矩阵（A、B）（数学结论）。每个名字都含 `lora_`，例如 `base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight`。名字前缀 `base_model.model` 说明模型确实被包成了 PEFT 模型。
- 参数总数 600,637,440。这个数**包括 LoRA 参数**。减掉 4,587,520，基座是 596,049,920，和第 19 章全参训练时的参数量一致（数学结论）。所以可训练比例是 4,587,520 / 596,049,920 ≈ 0.77%。

第 20 章扫描里的 "r16 q,k,v,o" 碰巧也是 4,587,520 个参数（第 20 章 20.11.2 节）。同样的参数预算，那边挂四个模块、rank 16，这边挂两个模块、rank 32。21.10 节比较时会用到。

### 21.5.4 对照代码

tour 里的 `counts` 函数：

```python
def counts(model):
    return dict(total=sum(p.numel() for p in model.parameters()),
                trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                names=[n for n,p in model.named_parameters() if p.requires_grad])
```

第 2 部分还加了一句断言：

```python
assert all('lora_' in n for n in result['parameters']['names'])
```

只看个数是不够的。比如有人把 `lm_head` 也设成可训练（PEFT 的 `modules_to_save` 可以做到），个数会多出一大截，但光看个数，要自己算一遍才知道多出的是哪些参数，看名字一眼就知道。

测试 `test_ch21_trainer_update_and_adapter_reload` 还查了另一面：训练一步以后，所有**冻结**参数和训练前**逐位相同**（`atol=0, rtol=0`），而至少一个可训练参数变了。判断冻没冻住，不能只看 `requires_grad`，要看值有没有动。

### 21.5.5 会出什么错

| 错误 | 现象 | 怎样发现 |
|---|---|---|
| 忘了传 `peft_config` | 可训练参数 ≈ 6 亿，显存暴涨 | `trainable` 远大于公式值 |
| `target_modules` 名字不匹配 | PEFT 报找不到模块；或只匹配到一部分 | 名字个数 ≠ 层数 × 模块数 × 2 |
| 打印的是包装前的模型 | 可训练参数等于总数 | 从 `trainer.model` 取，而不是从你传进去的变量取 |
| 把"总参数"当成"基座参数" | 比例算错 | 总数 − 可训练 = 基座，和已知值对一下 |

---

## 21.6 检查四：loss 怎样算，与自写循环对拍

### 21.6.1 为什么要对拍

前三项查的是数据对不对。这一项查的是框架拿到数据以后，算的是不是你以为的那个 loss。可能的差别有三处：

1. **移位**：位置 t 的 logits 预测位置 t+1 的 label，移没移、移了几次。
2. **分母**：一次更新有好几个 microbatch，loss 除以什么。
3. **实现**：这一版 TRL 在 CUDA 上用一个融合 LM head（fused LM head）算 loss，不走普通的 logits → 交叉熵。

最可靠的验证是**对拍**：同一个模型、同一批数据，一边用 TRL 算，一边用第 19 章的 `causal_loss` 算，比 loss、比梯度、再比一次优化器更新。

### 21.6.2 分母：按 token 加权，跨 microbatch 求和

先回顾第 19 章 19.6.5 节的结论：一次更新的 loss 应当是这次更新**所有** microbatch 里**所有**被监督 token 的平均。设第 k 个 microbatch 的 token 损失之和是 S_k，监督 token 数是 n_k，那么

  loss = (S₁ + S₂ + … ) / (n₁ + n₂ + …)

而不是 (S₁/n₁ + S₂/n₂ + …) / K（"平均的平均"）。

transformers 的 `Trainer` 是这样实现的：开始一次更新前，先把这次要用的 K 个 microbatch 都取出来，数出总监督 token 数 N（按 `labels[..., 1:] != -100` 数，位置 0 不算），作为 `num_items_in_batch` 传给每个 microbatch 的 `compute_loss`。每个 microbatch 算 S_k / N 并反向传播。K 次反向的梯度相加，正好是上面那个公式的梯度（数学结论：求导是线性的）。这一步成立，靠的是 N 在取数据时就数好了，是一个和参数无关的常数，所以 ∇(S₁/N) + ∇(S₂/N) + … = ∇((S₁ + S₂ + …)/N)。如果每个 microbatch 各除各的 n_k，加起来就不再是这个式子。

**一个手算例子。** 用 tour 第 1 部分的两条微型样本：第 1 条 labels 是 (−100, −100, 5, 1)，第 2 条是 (−100, −100, 7, 8, 9, 1)。按 `labels[1:]` 数，各有 2 个和 4 个监督 token，N = 6。假设两条的 token 损失之和分别是 S₁ = 1.0、S₂ = 4.0（假设的数字，只为演示）：

- 按 token 加权：(1.0 + 4.0) / 6 ≈ 0.833。
- 平均的平均：(1.0/2 + 4.0/4) / 2 = (0.5 + 1.0) / 2 = 0.75。

两者不同。按 token 加权时每个 token 的权重都是 1/6，平均的平均却让短样本的每个 token 权重变成 1/4，长样本的每个 token 只有 1/8。后纠错的答案长短差很多（主实验第一批就是 103 对 123 个），所以这个差别是真实存在的。

### 21.6.3 实现：融合 LM head

普通的做法是：模型输出整张 logits（batch × 长度 × 词表），再算交叉熵。Qwen3 的词表有 151,936 个 token（第 19 章 19.7.5 节），整张 logits 很占显存。

TRL 1.15 的 `SFTTrainer` 在建 trainer 时给模型装了一个融合 LM head（源码里的 `add_fused_lm_head`）。训练时它只对 **label 不是 −100 的位置**分块投影到词表，直接返回每个位置的 log 概率，不生成整张 logits。然后：

```python
per_token_loss = -outputs.log_probs  # zero on ignored positions
num_tokens = outputs.label_mask.sum() if num_items_in_batch is None else num_items_in_batch
loss = per_token_loss.sum() / torch.as_tensor(num_tokens).clamp(min=1)
```

（摘自安装的 `trl/trainer/sft_trainer.py` 的 `compute_loss`。）这就是 21.6.2 节的 S_k / N。

融合 head 的计算顺序和先算完整 logits 不同，浮点舍入会有微小差别。所以对拍要设容差，不要求逐位相同。

这个实现依赖 Triton，**只在 CUDA 上能用**。本章代码在 CPU/MPS 上用一个 `SFTTrainer` 子类，只把 `compute_loss` 换回 transformers 原生的 causal LM loss（同样接收 `num_items_in_batch`），数据、优化器和训练循环仍由 TRL/Trainer 管。所以，**在本地 CPU 上跑通的对拍，验证的不是 GPU 上的融合 loss**，融合 loss 的对拍必须在 CUDA 上做。

### 21.6.4 对拍的步骤

`loss_step_parity`（`miniocr/train/trl_sft.py`，练习 21.2）这样做：

1. 记下所有可训练参数的初值。
2. **TRL 一侧**：对每个 microbatch 调 `trainer.compute_loss(model, b, num_items_in_batch=N)`，反向；记下 loss 之和与梯度；裁剪梯度到 1；用一个新的 AdamW 走一步；记下更新后的参数。
3. 把参数恢复到初值。
4. **自写一侧**：对每个 microbatch 用 `causal_loss(logits, labels, reduction='sum') / N`，其余同上。
5. 比三样：loss、每个参数的梯度、一步更新后的每个参数。还要求参数真的动了（`max_parameter_delta > 0`），防止两边都没更新、所以相等。
6. 无论成功失败，最后都把参数恢复到初值，不污染后面的正式训练。

还有两处细节：

**为什么要比一次更新，不只比梯度？** 因为优化器也是框架行为的一部分，而且 Adam 会放大某些差别。下一段就是例子。

**为什么对拍时 AdamW 的 eps 用 1e-6，而正式训练是默认的 1e-8？** Adam 的第一步，每个分量的更新大约是 lr × g / (|g| + eps)。|g| 远大于 eps 时，这个比值接近 ±1，对 g 的微小舍入差不敏感；可当某个分量的 g 接近 0 时，融合 head 和完整矩阵乘的一点点舍入差，就可能让比值从 +1 翻到 −1，更新差出 2 × lr（数学结论）。eps 取大一些，能把这些"接近 0 的分量"压住，让对拍比的是有意义的差别。两边用同一个 eps，比较是公平的；但要记住，这是**对拍专用**的设置，不是正式训练的设置。

### 21.6.5 实例：两组对拍结果

**微型模型（tour 第 1 部分）。** 随机初始化的小 Qwen3，两条样本，监督 6 个 token，FP32，在 GPU 冒烟运行里走的是 TRL 的融合 loss（本项目观察，RTX 4090 租用，`smoke/part1.json`）：

| 量 | 值 |
|---|---|
| TRL loss | 4.1562796 |
| 自写 loss | 4.1562793 |
| loss 差 | 2.4 × 10⁻⁷ |
| 梯度最大差 | 4.5 × 10⁻⁸ |
| 一步更新后参数最大差 | 2.9 × 10⁻¹¹ |
| 参数最大变化 | 1.0 × 10⁻⁴ |

**真实 Qwen3-0.6B（tour 第 2 部分）。** 为了控制显存，取两条最短的训练样本，各成一个 microbatch，共监督 13 个 token，FP32，r = 32 的 LoRA（本项目观察，主实验同一实例，`part2.json` 的 `parity`）：

| 量 | 值 |
|---|---|
| TRL loss | 0.5969221 |
| 自写 loss | 0.5969204 |
| loss 差 | 1.7 × 10⁻⁶ |
| 梯度最大差 | 3.9 × 10⁻⁶ |
| 一步更新后参数最大差 | 4.4 × 10⁻⁷ |
| 参数最大变化 | 1.0 × 10⁻⁴ |

怎样读：

- loss 和梯度的差都在 10⁻⁶ 量级，远小于它们本身，说明**移位、分母、mask 三件事两边一致**。如果分母不一致（比如 TRL 按每个 microbatch 自己的 token 数除），两个 microbatch 监督数不同，loss 会差出百分之几甚至几十，不是 10⁻⁶。
- "参数最大变化"是 1.0 × 10⁻⁴，正好等于对拍用的学习率。这是 Adam 第一步的特点：更新幅度约等于 lr，和梯度大小无关（数学结论）。它证明两边都**真的更新了**。
- 真实模型的差比微型模型大。一个可能的原因是词表从 64 变成 151,936、模型有 28 层，分块求和与逐层累积的舍入更多（推测，未单独验证）。两组都在设定的容差（loss 2 × 10⁻⁵）以内。

### 21.6.6 训练日志里记的是什么

对拍确认了每一步优化的是什么。还要知道日志里记的是什么，否则 21.9 节会读错曲线。

- `Trainer` 每次更新返回的 loss，是上面那个跨 microbatch 的 token 加权平均（各 microbatch 的 S_k / N 相加）。
- 本章 `logging_steps=10`，`trainer-history.json` 里每 10 步一条，记的是**上次记录以来各步的平均**；`logging_first_step=True` 让第 1 步单独记一条并清空累计值，所以第 10 步那条其实是第 2 到 10 步共 9 步的平均，从第 20 步起才是完整的 10 步窗口（本项目观察：`trainer-history.json` 前两条的 step 是 1 和 10；`trainer.py` 按 `global_step - _globalstep_last_logged` 相除）。
- 每一步用的是**不同的随机 batch**。所以日志 loss 的起伏，一部分是因为模型在变，一部分只是因为题目在变。
- 每条记录还带着 `learning_rate` 和 `grad_norm`。`learning_rate` 是核对学习率调度最直接的证据：本章设了 `constant`、warmup 0，所以每条训练记录都应当是 2 × 10⁻⁴。`TrainingArguments` 的 `lr_scheduler_type` 默认是 `linear`（本章用的 transformers 5.19.0），忘了设的话，warmup 为 0 时学习率从 2 × 10⁻⁴ 线性降向 0，到第 400 步前后只剩约一半，训练结束时降到 0（数学结论）。训练照样跑，只是越往后更新越小，和第 20 章的常数学习率也不再可比。`grad_norm` 是记录那一步裁剪前的梯度范数，不是 10 步的平均。它大于裁剪阈值 1，只说明那一步被裁剪了，想知道多少步被裁剪，要逐步统计。

主实验第 1 步的日志 loss 是 0.496，裁剪前梯度范数 9.13；第 800 步那一条是 0.0091（本项目观察，主实验，`trainer-history.json`）。

### 21.6.7 会出什么错

| 错误 | 现象 | 怎样发现 |
|---|---|---|
| 自定义 `compute_loss` 时忽略了 `num_items_in_batch`，每个 microbatch 各自取平均 | token 权重变成"平均的平均"；这一版 `Trainer` 以为 loss 已按总数缩放、不再除以累积步数，于是日志 loss 约放大 K 倍（梯度也约放大 K 倍，但裁剪和 Adam 会掩盖幅度） | 两个监督数不同的 microbatch 对拍，loss 差远超舍入；日志 loss 和 `mean_loss` 差出约 K 倍 |
| 在 CPU 上对拍通过，就以为 GPU 也对 | GPU 走的是另一个 loss 实现 | 对拍记录里写明设备与是否融合 |
| 对拍时没恢复参数 | 正式训练从一个已经走过一步的点开始 | 对拍前后保存 state_dict 比较（测试 `test_ch21_fused_loss_gradient_and_adam_parity` 就这样查） |
| 只比 loss，不比更新 | 梯度符号问题、优化器配置问题漏掉 | 比到一步更新后的参数 |
| 拿对拍用的 eps 去训练 | 实际训练配置和记录不一致 | 正式配置以 `sft-config.json` 为准 |
| 没设 `lr_scheduler_type` | 学习率按默认的 `linear` 一路降到 0，和常数学习率的实验不可比 | 看 `trainer-history.json` 每条记录的 `learning_rate` |

---

## 21.7 检查五：保存 adapter 与完整配置，并复现

### 21.7.1 复现要回答两个问题

一个实验结果，如果过一个月你自己都复现不出来，它就只是一个传说。复现要回答：

1. **存够了吗？** 只存 adapter 权重是不够的。同一个 adapter，换一个模板、换一个生成参数、换一个基座快照，输出都会变。
2. **存的是对的吗？** 存下来以后，要真的加载一次、真的再生成一次，并且**逐条**比对输出，而不只比一个汇总分数。

### 21.7.2 要存什么

原则是：凡是会改变输出的，都要存。本章的实验目录里有这些：

| 类别 | 文件 | 存了什么 |
|---|---|---|
| adapter | `step-*/`（`save_pretrained`） | LoRA 权重和 `adapter_config.json`（rank、α、目标模块）；同目录还存了 tokenizer |
| 训练配置 | `sft-config.json` | `SFTConfig` 的全部字段，包括所有默认值 |
| 生成配置 | `base-generation-config.json`、`environment-*.json` 的 `generation` | 贪心、非思考、左 padding、192 个新 token、批大小 8 |
| 数据身份 | `data-manifest.json` | 数据版本、划分、噪声 seed、请求/接受条数、train 和 val 的 ID 哈希 |
| 环境 | `environment-*.json` | torch、transformers、trl、peft、accelerate、datasets 的版本；基座快照路径（含 commit 哈希）；GPU 型号；参与运行的 8 个代码文件的 SHA-256；命令行参数 |
| 选择 | `selection.json` | 选择规则、被选中的步、训练耗时和显存峰值 |
| 第一批数据 | `training-first-batch.json` | 21.3 节的逐 token 记录 |

对照第 20 章 20.15 节：那里说 `save_pretrained` 保存 adapter "待验证，本章未用"。本章用了，并且做了下面的重载验证。

### 21.7.3 怎样验证

lab 第 4 部分做这件事：

1. 重新加载冻结的基座（FP32），用 `PeftModel.from_pretrained` 挂上选中的 `step-400/`，`is_trainable=False`。
2. 先检查这次的精度、种子、n、长度上限、生成上限、生成批大小，和做选择时**完全一致**；任何一项不同就报错。
3. 在同一批 val 上重新生成。
4. 逐条比较这次的输出和训练中第 400 步保存的输出。全部相同才写 `reload-verified.json`。

为什么要逐条比？因为汇总分数相同不代表输出相同。两次各错 69 条，可能错的不是同一批（一边多对 3 条，另一边多对另外 3 条），EM 照样相等。

### 21.7.4 实例

主实验的重载结果（本项目观察，RTX 4090 租用，`reload-verified.json`、`reloaded-val-metrics.json`）：

- `predictions_identical: true`，200 条输出逐条相同。
- 重载后的原始 EM_norm 是 131/200 = 0.655，CER 0.0155，和第 400 步训练时的记录相同。
- 生成用时 249 秒，第 400 步训练中那次是 290 秒。时间不同而输出相同，正常：同一张卡上还跑着别的实验，用时只能看量级。

两点限定：

- **重载是在同一台租用机器上做的。** 换一台机器能不能复现，本章**没有实测**（待验证）。换 GPU 型号或驱动后，bf16 下的贪心生成可能在个别样本上分叉（待验证）。lab 第 4 部分列出了换机器时要带走的东西：整个实验目录里的配置和数据身份文件、对应的基座快照、选中的 `step-*` 目录。
- **环境文件抓到了一处代码差异。** 训练那次运行（`environment-2-3-4.json`）和重载/测试那次运行（`environment-4-5-6.json`）记录的 `tour.py` 和 `miniocr/train/trl_sft.py` 的 SHA-256 不一样，说明两次之间这两个文件被改过；现在仓库里的版本和后一次一致（本项目观察）。数据、评测、提示、`encode_sft` 等其余 6 个文件的哈希两次相同。重载验证通过，说明这次改动没有影响评测路径；但"用现在的代码从头训练能得到同一个 adapter"并没有被验证过。如果没记代码哈希，这个差异根本发现不了。

### 21.7.5 对照代码

保存发生在回调里：

```python
class Stages(TrainerCallback):
    def on_step_end(self, training_args, state, control, **kwargs):
        if state.global_step % args.eval_every and state.global_step != args.steps: return
        ...
        model.save_pretrained(path)    # 只存 adapter
        tok.save_pretrained(path)
        metrics = evaluate(args, model, tok, val, name+'-val')
        ...
```

对 PEFT 模型调 `save_pretrained`，存的只是 adapter（LoRA 权重和它的配置），不是 6 亿参数的基座。所以复现时**必须**有同一个基座快照，`environment-*.json` 里的 `base_snapshot` 就记着它的 commit 哈希。

重载时的参数一致性检查：

```python
for key in ('dtype','seed','n','max_length','max_new_tokens','eval_batch_size'):
    if getattr(args,key) != saved[key]: raise ValueError(...)
```

注意这张清单里**没有**训练参数（步数、batch、累积）。重载只做评测，训练参数不影响它。主实验的第二次运行（4、5、6 部分）命令行里的训练参数就和第一次不一样（`environment-4-5-6.json` 里 steps 是 2、batch 是 1），这不影响重载结果。读别人的实验时，以环境文件记录的参数为准，不要只看文档里的命令。

### 21.7.6 会出什么错

| 错误 | 现象 | 怎样发现 |
|---|---|---|
| 只复制 adapter 权重 | 别人加载后输出不同（模板、生成参数不同） | 复现清单里每一项都要有对应文件 |
| 基座快照变了（例如重新下载到新 commit） | 加载不报错，输出漂移 | 比对 `base_snapshot` 里的哈希 |
| 只比较汇总 EM | 逐条输出已经不同却没发现 | 逐条比对，写 `predictions_identical` |
| 重载时生成批大小不同 | 左 padding 长度变了，bf16 下个别输出改变 | 重载前检查生成参数（本章代码会直接报错） |
| 改了代码不记录 | 无法判断差异来自代码还是随机性 | 每次运行记代码哈希 |

---

## 21.8 检查六：在固定 val 上比较微调前后

### 21.8.1 固定指什么

"微调后比微调前好"这句话，只有在**同一把尺子**上量才成立。第 18 章 18.4 节的流程在这里全部适用，具体到本章：

- **同一批题**：val 前 200 条，ID 哈希记在 `data-manifest.json`。
- **同一个评测脚本**：`score_postcorrect`，同一套规范化、同一种区间（EM 用 Wilson 95% 区间，CER 用 1000 次 bootstrap）。报告的是第 0 章定义的文本后纠错面板：EM_norm、CER、改坏、fix_rate（定义见第 18 章 18.2 节）。
- **同一套生成设置**：21.2.3 节列的那些。
- **同一个模型精度**：基线和 SFT 都是 FP32 权重 + bf16 autocast。
- **事先写好的选择规则**：val 原始 EM_norm 越高越好；同分时 CER 越低越好；再同分选较早的步数。规则写在代码里，跑之前就定了。

微调前的对照不止一个。本章在同一批 val 上跑了三个对照：

- **零样本**：第 18 章的零样本提示，未微调的 Qwen3-0.6B。
- **严格 3-shot**：第 18 章的"少样本 + 格式约束"提示，3 条示例从 train 用固定种子取出，ID 存在 `few-shot-ids.json`。这是第 18 章最强的提示。
- **照抄**：原样输出带噪输入。它不用模型，却是最难打败的基线（第 18 章 18.5.3 节）。

### 21.8.2 结果

**本项目观察，主实验，val 前 200 条（有错 141 条、无错 59 条），原始输出口径。** 来源：`baseline-val-metrics.json`、`strict-3shot-val-metrics.json`、`copy-val.json`、`stages.json`。EM_norm 写成"成功数/200 [Wilson 95% 区间]"。

| 方法 | EM_norm | CER | 改坏 | fix_rate |
|---|---|---:|---:|---:|
| 零样本（未微调） | 0/200 [0.000, 0.019] | 1.062 | 59/59 | 0/141 |
| 严格 3-shot（未微调） | 8/200 [0.020, 0.077] | 0.144 | 54/59 | 3/141 |
| 照抄 | 73/200 [0.301, 0.434] | 0.0263 | 0/59 | 14/141 |
| SFT 第 200 步 | 128/200 [0.571, 0.703] | 0.0140 | 5/59 | 74/141 |
| **SFT 第 400 步（选中）** | **131/200 [0.587, 0.717]** | 0.0155 | 7/59 | 79/141 |
| SFT 第 600 步 | 127/200 [0.566, 0.699] | 0.0155 | 6/59 | 74/141 |
| SFT 第 800 步 | 128/200 [0.571, 0.703] | 0.0195 | 7/59 | 76/141 |

"改坏"是 59 条本来正确的输入里被改错了几条；fix_rate 是 141 条有错输入里被改对了几条（和第 18 到 20 章的定义相同）。零样本一行和第 20 章同一批 val 上"未微调基座"那一行完全相同（0/200，CER 1.062，第 20 章 20.12.2 节），这本身就是一次小小的复现检查。

### 21.8.3 怎样读

**结论一：SFT 远超所有未微调的对照。** 选中的第 400 步，EM_norm 区间 [0.587, 0.717] 和照抄的 [0.301, 0.434] 隔得很远，和最强提示的 [0.020, 0.077] 更远。CER 从照抄的 0.0263 降到 0.0155。

配对看更清楚。对第 400 步重载后保存的逐条输出（`reloaded-val-predictions.json`）和照抄逐条对照重新计数（本项目观察，对已有逐条输出重新计数，没有重新跑模型）：两者都对 66 条，只有 SFT 对 65 条，只有照抄对 7 条。131 − 73 = 58 = 65 − 7（数学结论）。那 7 条"照抄对、SFT 错"的，正好就是 SFT 改坏的 7 条无错输入。

**结论二：代价是改坏。** 照抄的改坏率是 0（它什么都不动，数学结论）。SFT 改坏了 7/59。看几条被改坏的输入（val 可以看）：把 `C_o` 改成了 `C_0`（把字母 o 当成 OCR 混淆"修"成了数字 0），把 `\operatorname{Ric(g)}` 改成了更常见的 `\operatorname{Ric}(g)`，把 `H_-` 改成了 `H^-`。模型学会了"像 OCR 错误的地方就改"，但不知道"原文本来就可能这么写"。第 22 章会把这类错例系统地分类。

**结论三：四个 checkpoint 的差别在噪声以内。** 四个 EM_norm 是 128、131、127、128 条，最大差 4 条（2 个百分点），而每个区间宽约 0.13。选中第 400 步是因为规则这样定，不是因为它"显著更好"。CER 反而是第 200 步最低。这种情况下，被选中的那个分数天然偏乐观：你是从四个里挑了最高的。

**结论四：要报原始输出，不报抽取后的。** 抽取后的 EM_norm 在四个 checkpoint 上只有 0.20 到 0.21（`stages.json` 的 `extracted`）。原因是第 18 章的 `extract_formula` 会去掉公式末尾的一个标点，而这批 val 里有 140 条目标以句号、逗号或分号结尾。对第 400 步的逐条输出重新计数：原始输出对、抽取后变错的有 89 条，其中 88 条的目标以这类标点结尾（本项目观察，对 `reloaded-val-predictions.json` 重新计数）。SFT 的输出本来就是一行干净公式，不需要抽取。主指标在跑之前就定为原始输出，两个口径都存下来了，不能看到结果后换成好看的那个。

### 21.8.4 test 只评一次

全书的测试集规则是：选型冻结后，预先登记的候选可各评一次；不据此改任何设置；开发期不看 test 错例，最终报告定稿后才可解封做错例分析。本章只评选中的这一个设置。

选择和重载验证都结束、`selection.json` 冻结以后，lab 第 5 部分在 test 前 200 条上**评了一次**选中的设置（本项目观察，主实验同一实例，`final-test-metrics.json`）：

- 原始 EM_norm 123/200 = **0.615** [0.546, 0.680]，CER 0.0265 [0.0167, 0.0376]。
- 改坏 5/63，fix_rate 65/137。

val 0.655，test 0.615。差距在区间之内，方向和 21.8.3 节结论三一致：从四个 checkpoint 里按 val 挑出来的分数偏乐观，test 没参与选择，是更诚实的估计。

评一次是靠代码保证的，不靠自觉：

- 不加 `--final-test` 不评；没有 `reload-verified.json` 不评。
- 评之前**独占创建** `FINAL_TEST_STARTED.json`（Python 的 `open(..., 'x')`，文件已存在就失败）。之后再调用，直接报错"禁止反复评测"，失败了也不自动重试。
- test 只存汇总，不存逐条输出，开发期想看 test 错例也没处看。

测试 `test_ch21_test_guard_never_loads_model_without_permission` 检查这几条守卫都在模型加载**之前**生效。和第 20 章一样，这只防止本目录里的手滑，不是全局的访问控制。第 20 章也在按同一规则取的 test 前 200 条上评过一次另一个配置；两次都没有根据 test 结果改过任何设置。

### 21.8.5 会出什么错

| 错误 | 现象 | 怎样发现 |
|---|---|---|
| 基线和 SFT 用了不同的 val 子集 | 差距看起来很大，其实一部分是题目难度不同 | 比对 ID 哈希 |
| 基线和 SFT 用了不同精度或生成参数 | 同上 | 基线在同一次运行里重跑，而不是抄旧表 |
| 看了结果再选口径 | 报出来的是最有利的那一列 | 主指标在代码里写死，两个口径都存 |
| 只报点估计 | 128 和 131 被说成"有提升" | 每个比例都带区间；配对比较留给第 22 章 |
| 用 test 挑 checkpoint | 报出来的 test 分数不再是无偏估计 | test 守卫；test 只存汇总 |

---

## 21.9 检查七：训练 loss 下降不等于新样本变好

### 21.9.1 现象

先看主实验的四组数字，放在同一张表里（本项目观察，主实验；日志 loss 来自 `trainer-history.json`，是截至该步的 10 步平均，第 1 步那条只有它自己，见 21.6.6 节；两个 NLL 来自 `stages.json`）：

| 步 | 有效轮数 | 日志 train loss | 固定 train 探针 NLL | val NLL | val EM_norm |
|---:|---:|---:|---:|---:|---|
| 1 | 0.004 | 0.496 | 未测 | 未测 | 未测 |
| 200 | 0.79 | 0.048 | 0.0208 | 0.0327 | 128/200 |
| 400 | 1.58 | 0.022 | 0.0091 | **0.0319** | **131/200** |
| 600 | 2.37 | 0.0089 | 0.0044 | 0.0364 | 127/200 |
| 800 | 3.16 | 0.0091 | 0.0025 | 0.0349 | 128/200 |

（有效轮数 = 步数 × 8 / 2023，数学结论。）

两件事同时在发生：

- **训练集上一路变好**：固定 train 探针 NLL 从第 200 步的 0.0208 降到第 800 步的 0.0025，降了约 8 倍（数学结论，由表中数字算出）。
- **val 上在第 400 步就停了**：val NLL 在第 400 步最低（0.0319），之后回升到 0.0364 和 0.0349；val EM_norm 在 127 到 131 之间晃，没有趋势。

只看训练日志，很容易以为第 800 步最好。实际上从第 400 步以后，多训的那约 3200 次样本只让模型把训练集背得更熟。`learning-curves.png` 把左边的两条 NLL 曲线和右边的 val EM 分开画，就是为了让这个分叉一眼可见。

### 21.9.2 为什么会这样

打个比方：你把一本练习册刷了三遍，练习册上的题几乎全对，但考试的新题并没有因为第三遍而多对几道。第三遍练的是"记住这些题"，不是"会做这类题"。

具体有三个原因，前两个在第 19 章 19.7.3 节已经见过：

1. **过拟合**：train 只有 2023 条，800 步过了 3 轮多。模型对见过的样本越来越有把握（train 探针 NLL 一直降），对没见过的样本却不再进步，甚至更自信地犯错（val NLL 回升）。
2. **token 平均 vs 整条对错**：NLL 是每个 token 的平均。一条答案平均六七十个 token，绝大部分是照抄，几乎零损失；决定 EM 的只有改错的那一两个位置。NLL 的小变化和 EM 的变化不一一对应。
3. **teacher forcing vs 自回归**：NLL 在每一步都喂正确的前文；生成时前面错一步，后面就在错的前文上继续。

还有一个本章特有的原因：**日志 loss 本身就是噪声很大的量**。每一步用的是不同的随机 batch，第 600 步的 0.0089 和第 800 步的 0.0091 谁高谁低，主要取决于抽到了哪些题。所以本章另设了"固定 train 探针"：用固定的 train 前 128 条（8494 个监督 token）、固定的正确 labels 算 NLL，不同步之间才能比较。

### 21.9.3 怎样在实验里看到它

步骤：

1. 每隔固定步数（本章 200 步），在**同一批** train 样本和**同一批** val 样本上算 token 加权 NLL。用第 19 章的 `mean_loss`，它按 token 总数加权，不会"平均的平均"。
2. 同时在 val 上生成，算 EM 和 CER。
3. 把 train 探针 NLL、val NLL、val EM 分开画，不要画在一个纵轴上（一个是 token 级的 loss，一个是序列级的比例，不是同一个量）。
4. 如果 train 探针还在降，而 val NLL 已经回升、val EM 不动，就说明后面的训练主要在拟合训练集。

对照代码（tour 第 3 部分的回调）：

```python
nll = mean_loss(model, val_encoded, tok.pad_token_id, batch_size=args.batch_size, dtype=args.dtype)
# 同一固定 train 前缀的 NLL，避免拿不同 batch 的 loss 误判过拟合。
train_nll = mean_loss(model, encoded[:min(128,len(encoded))], tok.pad_token_id,
                      batch_size=args.batch_size, dtype=args.dtype)
```

一个小细节：val NLL 只用了 197 条（13,472 个监督 token），因为有 3 条 val 样本编码后超过 384 个 token，`encode_sft` 把它们拒收了；而 val 生成用的是全部 200 条（本项目观察，`stages.json`）。这两个量的样本集合不完全相同，要写明。

### 21.9.4 那该怎么办

- **按 val 选 checkpoint，而不是取最后一步**。本章每 200 步存一次 adapter，按事先定好的规则选，选中了第 400 步。
- **存够中间 checkpoint**。如果只存最后一步，你就只能交第 800 步。
- **调训练量要在 val 上调**。这次的数据提示：在 2023 条 train 上，1.5 轮左右以后 val 不再改善。想再提高，更可能要靠更多或更好的数据，而不是更多步数（这是对下一步的假设，不是结论，第 22 章会教你怎样从错例里提出这类假设）。
- **不要拿 test 来判断是不是过拟合了**。那是 val 的工作。

> **停下来想一想：** 第 400 步和第 800 步的 val EM 只差 3 条，val NLL 却差了约 9%。如果选择规则换成"val NLL 最低"，会选到哪一步？如果规则换成"CER 最低"呢？为什么选择规则一定要在看结果之前定？

### 21.9.5 会出什么错

| 错误 | 现象 | 怎样发现 |
|---|---|---|
| 拿日志 train loss 当过拟合信号 | 被 batch 噪声误导 | 用固定探针 |
| 把 NLL 和 EM 画在一张图、一个纵轴上 | 看起来"同步下降"，其实量纲不同 | 分开画 |
| 只存最后一步 | 无法回退到更好的步 | 回调里每隔固定步保存 |
| train 探针和 val 用了不同的 mask 或模板 | 两条曲线不可比 | 两者都用 `encode_sft` 编码、`mean_loss` 计算 |

---

## 21.10 与第 18 到 20 章的对比：R2 里的 TRL LoRA 一行

### 21.10.1 同一批题上的七行

第 20 章 20.12.2 节已经把第 19 章全参 SFT 的逐条记录在 val 前 200 条上重新统计过。21.2.3 节确认了本章的 val 和第 20 章是同一批 200 条（ID 哈希一致）。所以下面这张表的每一行，都是在**同一批 200 道题、同一个评测脚本、原始输出口径**上量的：

| 方法 | 来源 | EM_norm | CER | 改坏 | fix_rate |
|---|---|---|---:|---:|---:|
| 照抄 | 本章 `copy-val.json` | 73/200 = 0.365 | 0.0263 | 0/59 | 14/141 |
| 零样本提示 | 本章 `baseline-val-metrics.json` | 0/200 = 0.000 | 1.062 | 59/59 | 0/141 |
| 严格 3-shot 提示 | 本章 `strict-3shot-val-metrics.json` | 8/200 = 0.040 | 0.144 | 54/59 | 3/141 |
| 全参 SFT（第 19 章，第 512 步） | 第 20 章 20.12.2 节 | 155/200 = 0.775 | 0.0114 | 2/59 | 98/141 |
| 自写 LoRA r64 q,k,v,o（第 20 章选中） | 第 20 章 20.11.2 节 | 134/200 = 0.670 | 0.0136 | 2/59 | 77/141 |
| 自写 LoRA r16 q,k,v,o（第 20 章） | 第 20 章 20.11.2 节 | 129/200 = 0.645 | 0.0190 | 6/59 | 76/141\* |
| **TRL LoRA r32 q,v（本章第 400 步）** | 本章 `stages.json` | **131/200 = 0.655** | 0.0155 | 7/59 | 79/141 |

\* 第 20 章没有直接记录这一格，这里由汇总计数推出：全对的 129 条由两部分组成，原串无错且没改坏的 59 − 6 = 53 条，加上修好的有错条，所以修好 129 − 53 = 76 条，fix_rate = 76/141 ≈ 0.539。其他各行也满足这个恒等式，可以自己验算。

全部是本项目观察，RTX 4090（租用），FP32 权重 + bf16 autocast。test 一列只有两行有数：第 20 章选中配置 124/200 = 0.620，本章 123/200 = 0.615；第 19 章没有评 test。

注意第 18 章 18.5.2 节的提示表不能直接放进来：那是 val 前 2000 条、抽取后口径、直接加载 bf16 权重。本章在同一次运行里用 SFT 的精度和口径**重跑**了提示基线，才放进这张表。

### 21.10.2 哪些相同，哪些不同

和第 20 章选中的 LoRA 比，相同的有：基座、val、评测脚本、生成设置、精度口径、学习率 2 × 10⁻⁴ 常数、有效 batch 8、长度上限 384、AdamW 和裁剪。不同的有：

| 项 | 第 20 章 r64 q,k,v,o | 本章 r32 q,v |
|---|---|---|
| 训练框架 | 自写循环 | TRL `SFTTrainer`（CUDA 上是融合 loss） |
| 可训练参数 | 18,350,080 | 4,587,520 |
| train 条数（接受） | 4096（4049） | 2048（2023） |
| 选中点的训练量 | 400 步 × 8 = 3200 次样本，约 0.79 轮 | 400 步 × 8 ≈ 3200 次样本（实际 3199，见 21.2 节），约 1.58 轮 |
| 选什么 | 6 个配置里选 1 个 | 4 个 checkpoint 里选 1 个 |
| 梯度检查点 | 未在本章核对 | 开启 |
| 数据顺序 | 自写循环的固定种子打乱 | Trainer 的采样器 |

所以**这不是一个单变量对照**。"TRL 和自写哪个好"这个问题，这张表回答不了：框架、rank、目标模块、训练集大小、选择方式同时变了。能说的只有：

- **TRL 版本没有做坏。** 它在同一批 val 上的结果（131/200）落在第 20 章六个自写 LoRA 配置的范围里（121/200 到 134/200），区间和它们全部重叠。尤其是和参数量完全相同的 r16 q,k,v,o（129/200）相比，只差 2 条。这从结果上旁证了框架做的就是你学过的那件事，前提是 21.3 到 21.6 节的检查都通过了。
- **全参 SFT 仍然最好**（155/200），但第 20 章 20.12.3 节讲过的混杂因素（训练量、学习率、选择方式）在这里依然存在，而且本章的 train 只有第 19 章的一半。
- **两次 test 几乎一样**（124 和 123 条），差 1 条，在噪声以内。

想把框架和自写比干净，需要固定 train 条数、步数、rank、目标模块和数据顺序，只换训练框架。这是练习里的一道选做题。

### 21.10.3 作业 A3 的训练层

A3 训练层的要求是"OCR 后纠错在验证集上超过提示基线"。**门槛待校准**：按课程规则，门槛由你第一次完整训练后确定。可以参考的观察是：本章在这批 val 上，最强的提示基线是 8/200，照抄是 73/200，TRL LoRA 是 131/200（本项目观察，主实验）。校准门槛时，建议把照抄也作为一个参照，因为在这个任务上，打败提示很容易，打败照抄才说明模型真的学会了"只改错的地方"。

### 21.10.4 填进 R2

R2 的结果表里，第 18 章填了提示基线，第 19、20 章各填了一行。本章这一行除了第 19 章 19.8.4 节列出的那些字段，还要多写：

- **框架和版本**：TRL 1.15.0、PEFT 0.21.2、transformers 5.19.0、accelerate 1.15.0、datasets 5.1.0、torch 2.14.1（`environment-*.json`）。
- **loss 路径**：CUDA 上是 TRL 的融合 LM head；附上 21.6 节的对拍结果。
- **数据路径**：预编码 labels + `skip_prepare_dataset=True`；第一批的监督 token 数，提示监督数 = 0。
- **可训练参数**：4,587,520，名字全部是 `lora_`。
- **选择**：在 val 前 200 条上，按事先定好的规则从 4 个 checkpoint 选出第 400 步。
- **复现**：`reload-verified.json` 通过；代码哈希在两次运行之间变过（21.7.4 节），照实写。
- **test**：评了一次，0.615 [0.546, 0.680]；写明"只评一次，未据此修改任何设置"。
- **成本**：训练加中间评测墙钟 2663 秒（约 0.74 GPU 小时），其中四次 val 生成合计约 1149 秒（`stages.json` 四个 `seconds` 之和，数学结论）；PyTorch 本进程峰值 allocated 3.36 GiB、reserved 5.74 GiB（`selection.json` 里的字节数换算，1 GiB = 2³⁰ 字节，数学结论）。同一张卡上有其他实验并行，用时只能看量级（本项目观察，主实验）。

---

## 21.11 TRL 参数概览（概括）

`SFTConfig` 的字段很多，而且每个版本都在变（本章就碰上了 labels 的构造从 collator 挪到数据准备阶段这样的变化）。不需要背，知道有哪几类、去哪里查就够了。下表按用途分组，只列名字和作用；**具体含义和默认值，请按你安装的版本查 TRL 文档，并在源码里核对。**

| 类别 | 代表字段 | 作用 |
|---|---|---|
| 继承自 `TrainingArguments` | 学习率、调度、batch、累积、步数/轮数、精度、梯度检查点、保存、日志、种子 | 和 transformers `Trainer` 共用 |
| 数据格式 | `dataset_text_field`、`dataset_kwargs`、`max_length`、`truncation_mode` | 纯文本、对话、prompt-completion 怎样转成 token |
| 打包 | `packing`、`padding_free` | 把多条短样本拼进一行，省掉 pad；会改变注意力的边界，要格外检查 |
| 监督范围 | `completion_only_loss`、`assistant_only_loss` | 只在答案或助手回合上计 loss；生效与否看 labels |
| loss 类型 | `loss_type` | 本章是默认的 `nll`；还有其他变体（待验证，本章未用） |
| 内核与加速 | `use_liger_kernel` 等 | 替换部分算子；本版 TRL 已自带融合 LM head |
| 模型加载 | `model_init_kwargs` | 传模型名字而不是模型对象时用 |

一个实用的做法：每次实验都把 `trainer.args.to_dict()` 整个存成 JSON（本章的 `sft-config.json`），升级版本后 diff 一下，就知道哪些默认值变了。

---

## 21.12 常见错误

现象和发现办法都在各节末尾的表里。后果最重的几条，索引如下：

- 只打开开关，不看 labels（21.4 节）。
- 在自己的列表上检查，不看 dataloader 的输出（21.3 节）。
- 把总参数当基座参数，或者不看可训练参数的名字（21.5 节）。
- 在 CPU 上对拍通过，就以为 GPU 上的融合 loss 也对（21.6 节）。
- 只存 adapter，或者只比汇总分数（21.7 节）。
- 用 test 挑 checkpoint，或者看了结果再换口径（21.8 节）。
- 拿训练日志 loss 判断越训越好（21.9 节）。

---

## 21.13 和真实框架的对应

本章用的就是真实框架，所以这一节换个方向：把第 19、20 章自写的部件和 TRL / PEFT / transformers 里做同一件事的部件对应起来（版本 TRL 1.15.0、PEFT 0.21.2、transformers 5.19.0；标"待验证"的是本章没有实测的部分）：

| 自写（第 19、20 章） | 框架里的对应 | 本章怎样确认 |
|---|---|---|
| `encode_sft`：模板、mask、EOS | TRL 的数据准备（自己套模板、从 `completion_mask` 建 labels） | 本章绕开它，直接交 labels（21.2.2 节） |
| `collate_sft`：右 padding、label 补 −100 | TRL 的 `DataCollatorForLanguageModeling` | 两边逐元素相等（tour 第 1、2 部分） |
| `causal_loss` + 按 token 总数加权 | `SFTTrainer.compute_loss`（CUDA 上融合 LM head）+ `Trainer` 的 `num_items_in_batch` | loss、梯度、一次更新对拍（21.6 节） |
| `train_sft` 的循环与梯度累积 | `Trainer.train` | 测试检查一步后可训练参数变、冻结参数不变 |
| `inject_lora` | `SFTTrainer(peft_config=...)` 内部包装成 PEFT 模型 | 可训练参数个数与名字（21.5 节） |
| 回调里评 val、保存 | `TrainerCallback.on_step_end` | `stages.json` |
| 保存 adapter | PEFT 的 `save_pretrained` | 重载逐条复现（21.7 节） |
| 加载 adapter | `PeftModel.from_pretrained(base, dir)` | 同上 |
| （未实现）packing、padding-free、多卡 | `packing`、`padding_free`、accelerate 分布式 | 待验证，本章未用 |
| （未实现）合并后推理 | PEFT 的 `merge_and_unload` | 待验证，本章未用；第 20 章自写合并已验证 |

第 19、20 章先从零写一遍，就是为了让框架的部件有东西可对。

---

## 21.14 小结

用 TRL 的 `SFTTrainer` + PEFT 做 LoRA SFT 只要几十行配置，但框架替你做的事出错时，训练照样跑、loss 照样降，所以每一项都要拿证据确认。主实验里，`completion_only_loss` 在这条数据路径上没有起作用，监督范围由你交给它的 labels 决定；CUDA 上的融合 loss 与自写 `causal_loss` 对得上；adapter 重载后 200 条 val 输出逐条一致。在固定的 val 前 200 条上，选中的第 400 步 EM_norm 是 131/200，远超照抄的 73/200 和最强提示的 8/200，代价是改坏 7/59 条无错输入；test 只评一次得 123/200（本项目观察，RTX 4090 租用）。固定 train 探针 NLL 一路降到第 800 步，val 在第 400 步以后不再改善，所以 checkpoint 要按 val 选，不看训练 loss。

复习时抓住下面几条：

- **检查清单的共同做法**：查框架处理之后的对象（dataloader 吐出的 batch、`trainer.model`、重载后的模型），不查你传进去的那个；拿它和独立的参照比；把比较写成断言。七项是：第一批 token 与 label、completion-only、可训练参数、loss 对拍、保存与复现、固定 val 比较、train 与 val 分叉。
- **数据与 mask**：本章把第 19 章 `encode_sft` 编好的 `input_ids`、`labels` 直接交给 TRL，并设 `skip_prepare_dataset=True`。数据里已有 `labels` 列时 TRL 原样使用，`completion_only_loss` 不起作用；既没有 `labels` 也没有 mask 时，labels 等于 `input_ids`，提示被监督而且不报错。所以要看第一批：提示和 pad 是 −100，第一个监督位置等于提示长度，末尾 EOS 被监督。
- **可训练参数**：个数 = Σ r × (in + out)，r = 32 挂 q、v 时是 32 × (3072 + 2048) × 28 = 4,587,520；名字个数 = 层数 × 模块数 × 2，名字全部含 `lora_`。总参数包括 LoRA，减掉可训练参数才是基座。
- **loss 与分母**：一次更新的 loss = (S₁ + S₂ + …) / N，N 是这次更新所有 microbatch 里按 `labels[..., 1:]` 数出的监督 token 总数，事先数好再传给每个 microbatch。对拍要比 loss、梯度和一步更新后的参数，还要确认参数真的动了；CPU 上的对拍不能代替 CUDA 上融合 LM head 的对拍。
- **调度与日志**：transformers 默认的调度是 `linear`，要用常数学习率就得显式设 `constant`，再看日志里的 `learning_rate` 是否始终是设定值。日志 loss 是上次记录以来各步的平均，每步的 batch 不同，噪声大，判断过拟合要用固定的 train 探针和 val。
- **复现**：除了 adapter，还要存完整配置、生成设置、数据 ID 哈希、版本、基座快照和代码哈希；重载后逐条比输出，不只比汇总分数。
- **最常见的错误**：只打开 `completion_only_loss` 不看 labels；从自己传进去的模型或列表上检查，而不是从 trainer 里取；按训练 loss 或 test 挑 checkpoint。

---

## 21.15 精读

1. **TRL 文档：SFT Trainer**（与安装版本对应的 v1.15.0 页面，lab 第 2 节给了链接：<https://huggingface.co/docs/trl/v1.15.0/en/sft_trainer>）。
   - 读讲数据集格式（纯文本、对话、prompt-completion）和只在 completion 上算 loss 的部分，再读页面末尾 `SFTConfig` / `SFTTrainer` 的 API 参考。
   - 带着这几个问题读：
     - 文档说数据集"已有 `labels` 列时原样使用"。结合 21.4 节，在什么情况下打开 `completion_only_loss` 也救不了被监督的提示？
     - 文档里 packing 怎样处理不同样本之间的注意力边界？如果打开它，21.3 节的检查要加哪一条？
     - 找到文档里所有带默认值的字段，哪几个一旦变了会改变 21.6 节对拍的结果？

2. **PEFT 文档：LoRA**（<https://huggingface.co/docs/peft>，与 0.21.2 对应）。
   - 读 `LoraConfig` 的参数说明（`r`、`lora_alpha`、`target_modules`、`lora_dropout`、`bias`、`modules_to_save`），以及保存和用 `PeftModel.from_pretrained` 加载 adapter 的部分。
   - 带着这几个问题读：
     - `modules_to_save` 会让哪些参数变成可训练的？它会怎样改变 21.5 节的参数个数？保存出的 adapter 会多出什么？
     - 加载 adapter 时，PEFT 怎样确认基座和 adapter 是配套的？如果基座快照换了，它会不会报错？（对照 21.7.6 节）

3. **源码**（选读）：安装目录下的 `trl/trainer/sft_trainer.py`，读 `compute_loss` 和构造 labels 的那一段；以及 `transformers/trainer.py` 里的 `_get_num_items_in_batch`。问题：为什么分母按 `labels[..., 1:]` 数，而不是按整个 `labels` 数？
