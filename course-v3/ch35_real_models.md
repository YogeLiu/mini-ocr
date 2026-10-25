# 第 35 章　搬到真实模型

## 本章在全景图中的位置

前面 34 章，我们自己写了每一个零件：ViT、连接器、语言模型、SFT 的标签与 mask、LoRA、回答的对数概率、DPO、GRPO。第 24 章打开了两个真实的 VLM（Qwen3.5-0.8B、SmolVLM-256M），看清了它们的 processor、视觉塔和语言模型，但只做了推理。

本章把**训练**也搬到真实模型上。要回答的问题很实际：

- transformers、peft、trl 这三个库，各对应我们自己写的哪一部分？它们的默认值是什么？
- 真实模型的聊天模板比我们的复杂（Qwen3.5 在回答前插了一段空的"思考"标记），回答的 mask 怎样才能定对？
- LoRA 加在哪些层上？默认的 `q_proj/k_proj/v_proj/o_proj` 对所有模型都合适吗？
- 在动手训练之前，怎样做一次**阻断项检查**：模型能不能加载、前向反向能不能跑、需要的 kernel 有没有、显存够不够？

本章的第 1–5 部分都是实测的（日志在 `outputs/ch35/`）。在这个过程中我们发现了两个"阻断项"：一个是 Qwen3.5 的层结构让默认的 LoRA 目标只覆盖了四分之一的层；另一个是本书用的 trl 版本里三个 trainer 都只能在 GPU 上跑。第 6–8 部分是真实模型的训练，由你自己跑（用 CPU 也能跑，只是慢；有 GPU 的话加 `--gpu`）。

## 学习目标

学完本章，你应该能够：

1. 说出 transformers、peft、trl 的主要类分别对应本书的哪些函数，读出 trl 配置的默认值并判断它们和你的意图是否一致；
2. 用"提示是完整序列的前缀"的方法定出回答的 mask，并解释为什么不能把提示和回答分别分词再拼起来；
3. 为一个新模型列出 LoRA 的目标层，检查它们覆盖了多少个 decoder 层；
4. 对一个新模型做阻断项检查：kernel、前向反向、B = 0 的恒等性、合并后的一致性、内存与速度；
5. 用我们自己的训练循环（第 26、31 章的损失）训练真实的 VLM，并说出它和 trl 的 trainer 相比各自的取舍。

## 前置检查

1. 第 24 章：Qwen3.5 的语言模型由哪两种注意力层组成？各有几层？
2. 第 26 章：回答的 mask 为什么要包括回答末尾的结束符？
3. 第 27 章：logits 的第 t 个位置预测的是第几个 token？
4. 第 28 章：LoRA 的 B 为什么初始化为 0？合并（merge）是什么意思？
5. 第 31 章：GRPO 的组内标准化在哪种情况下给出的优势全为 0？

---

## 35.1 三个库与我们自己的代码

| 要做的事 | 本书的代码 | 库 |
|---|---|---|
| 模型的前向、`generate` | `miniocr.models.vlm`、`generate.py`（第 17、22 章） | transformers：`AutoModelForImageTextToText` |
| 图像预处理、聊天模板 | `hf_vlm.qwen_preprocess`、`expand_image_tokens`（第 24 章） | transformers：`AutoProcessor`、`apply_chat_template` |
| 回答的 mask | `train.sft.chat_roles`（第 26 章）、`hf_train.answer_mask_by_prefix` | trl：`SFTConfig(assistant_only_loss / completion_only_loss)` |
| LoRA | `models.lora`（第 28 章）、`hf_train.lora_target_names` | peft：`LoraConfig`、`get_peft_model`、`merge_and_unload` |
| 回答的对数概率 | `rl.logprobs`（第 27 章）、`hf_train.hf_answer_logp` | trl 的 trainer 内部（融合的 LM head） |
| DPO | `rl.dpo.dpo_loss`（第 30 章） | trl：`DPOTrainer`、`DPOConfig` |
| GRPO / DAPO | `rl.grpo.grpo_loss`、`rl.pg.group_advantages`（第 29、31 章） | trl：`GRPOTrainer`、`GRPOConfig` |
| 奖励 | `rl.rewards.task_reward`（第 31 章）、`hf_train.trl_reward` | trl：`GRPOTrainer(reward_funcs=[...])` |

这张表就是第 1 部分打印的第一段。三个库的分工：transformers 管模型本身（结构、权重、processor、生成），peft 管"只训练一小部分参数"（LoRA 等），trl 管训练流程（SFT、DPO、GRPO 的数据处理、损失、日志）。

### 默认值：直接从配置类读

第 1 部分实例化了三个配置类，读出它们的默认值（transformers 5.19.0、peft 0.21.2、trl 1.15.0）：

```
GRPOConfig: learning_rate=1e-06, beta=0.0, epsilon=0.2, epsilon_high=None, loss_type='dapo', scale_rewards='group',
            num_generations=8, num_iterations=1, temperature=1.0, max_completion_length=512, mask_truncated_completions=False
DPOConfig:  learning_rate=1e-06, beta=0.1, loss_type=['sigmoid']
SFTConfig:  learning_rate=2e-05, assistant_only_loss=False, completion_only_loss=None, packing=False, max_length=1024
```

逐个对照前面的章节：

- **GRPO 的 `beta=0.0`**：默认**不加 KL**，和 DAPO 一样（第 31 章）。GRPO 原文的 β = 0.04；如果你想复现原文，必须自己设；
- **`loss_type='dapo'`**：默认按 token 汇总（第 31 章练习 31.3 的 `mode="token"`），不是 GRPO 原文的"每条回答先平均"；
- **`scale_rewards='group'`**：优势除以组内的标准差（第 29 章的 `group_advantages(std=True)`）。Dr. GRPO 建议去掉它（第 31 章）；
- **`num_iterations=1`**：每批样本只更新一次，也就是 μ = 1。第 31 章讲过，μ = 1 时重要性比值恒为 1，裁剪不起作用；
- **`epsilon_high=None`**：没有 clip-higher（第 31 章），要用的话设 `epsilon_high=0.28`；
- **`mask_truncated_completions=False`**：超长被截断的回答仍然算损失。DAPO 的超长处理（第 31 章练习 31.5）需要你自己开；
- **DPO 的 `beta=0.1`、`loss_type=['sigmoid']`**：就是第 30 章的标准 DPO。"DPO + SFT"（第 30 章的 `sft_weight`）在这个版本里写成 `loss_type=["sigmoid", "sft"]` 加 `loss_weights`；
- **SFT 的 `assistant_only_loss=False`**：默认**整段对话都算损失**，包括提示！第 26 章讲过，只在回答上算损失是 VLM 的 SFT 的标准做法；用 trl 的话要自己打开 `assistant_only_loss=True`（它要求聊天模板支持标出助手的部分）或者用 `completion_only_loss` 配合"提示 / 回答"两列的数据格式。

**默认值随版本变化**。这一节的数字只对 trl 1.15.0 成立：旧版本的 GRPO 默认 β 不是 0，DPO 的"加 SFT 项"以前叫 `rpo_alpha`。用任何一个训练库之前，都要像第 1 部分这样把配置打印出来，逐项和你的意图对照。

## 35.2 聊天模板与回答的 mask

### 真实模型的模板

第 26 章我们自己定了模板：`<user> 问题 <assistant> 回答 <eos>`。真实模型的模板由它的分词器（或 processor）里的 Jinja 模板决定。第 2 部分打印了两个模型的模板：

```
Qwen3.5 生成提示的结尾：            '|im_start|>assistant\n<think>\n\n</think>\n\n'
Qwen3.5 训练用的完整对话的结尾：    'ssistant\n<think>\n\n</think>\n\nTotal 42.50<|im_end|>\n'
SmolVLM 生成提示的结尾：            '.<end_of_utterance>\nAssistant:'
```

Qwen3.5 是一个"会思考"的模型家族：它的模板在助手的回答前面插了一段**空的思考**（`<think>\n\n</think>\n\n`）。推理时，`add_generation_prompt=True` 生成的提示已经包括这一段，模型直接从回答开始写。

### 用前缀定出回答

最稳妥的做法：

1. 只有提示（带生成提示）的文字，分词得到 `prompt_ids`；
2. 提示 + 回答 + 结束符的文字，分词得到 `full_ids`；
3. **检查 `prompt_ids` 正好是 `full_ids` 的前缀**；是的话，前缀之后的 token 就是回答，算损失。

### 练习 35.1：`answer_mask_by_prefix`

```python
def answer_mask_by_prefix(prompt_ids, full_ids):
    n = len(prompt_ids)
    if n > len(full_ids) or list(full_ids[:n]) != list(prompt_ids):
        k = next((i for i, (a, b) in enumerate(zip(prompt_ids, full_ids)) if a != b), min(n, len(full_ids)))
        raise ValueError(f"提示不是完整序列的前缀：第 {k} 个 token 起不同")
    return [0] * n + [1] * (len(full_ids) - n)
```

为什么要检查前缀？因为分词不是逐字的：BPE 会把相邻的字符合并成一个 token。提示的最后几个字符和回答的前几个字符，在拼起来之后可能被合并成一个新的 token，这时"提示单独分词的长度"不再是回答开始的位置。如果不检查，mask 会悄悄地错开一个或几个 token：回答的第一个 token 不算损失，或者提示的最后一个 token 被算进去——训练照样能跑，损失照样下降，只是学的东西有一点不对。这是附录 B 里的一种典型的静默错误。

另一种常见的做法是"把提示和回答**分别**分词再拼起来"。它总能"通过"前缀检查，但拼出来的 token 序列不是模型推理时看到的序列（推理时模型看到的是整段文字的分词结果），训练和推理就不一致了。

### 第 2 部分：两个模型上的结果

```
Qwen3.5：提示 22 个 token，完整 30 个；算损失的 token：['Total', 'Ġ', '4', '2', '.', '5', '0', '<|im_end|>']
SmolVLM：
  回答前加 ' ' ：前缀成立；算损失的 token：['ĠTotal', 'Ġ', '4', '2', '.', '5', '0', '<end_of_utterance>']
  回答前加 ''  ：前缀成立；算损失的 token：['Total', 'Ġ', '4', '2', '.', '5', '0', '<end_of_utterance>']
  'Assistant: 42'  分词 → ['Ass', 'istant', ':', 'Ġ', '4', '2']
  'Assistant:42'   分词 → ['Ass', 'istant', ':', '4', '2']
```

读这几行：

- Qwen3.5 的空思考段在前缀里，不算损失：模型推理时这一段是提示给的，不需要学着写它。算损失的是回答和 `<|im_end|>`（第 26 章：结束符必须学，否则模型不会停）；
- SmolVLM 的提示以 `Assistant:` 结尾，回答前面加不加空格，前缀检查都能通过——但得到的 token 不同：`ĠTotal`（带空格的 Total）和 `Total`。哪个对？看模板：用 `apply_chat_template` 生成一个带回答的完整对话（回答是 `hello world`），得到的结尾是 `Assistant: hello world<end_of_utterance>\n`，冒号后面有一个空格。所以要加空格，否则训练时模型学的是"冒号后面直接接文字"，和它预训练时见过的格式不一致；
- 数字被逐位切开（`4`、`2`）：这两个分词器都把数字拆成单个数字，和第 14 章讨论过的做法一致。

**规则**：mask 用前缀法定，回答前的空格、换行按模板的完整对话来定（用 `apply_chat_template` 生成一个带回答的完整对话，看回答前面是什么）。

## 35.3 LoRA 加在哪些层上

### peft 的接口

```python
from peft import LoraConfig, get_peft_model
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=names))
```

`target_modules` 可以是名字的后缀（比如 `["q_proj", "v_proj"]`，匹配所有以它们结尾的层），也可以是完整名字的列表。用后缀很方便，但有一个陷阱：**视觉塔里的层也叫 `q_proj`**。

### 练习 35.2：`lora_target_names`

```python
def lora_target_names(model, suffixes=("q_proj", "k_proj", "v_proj", "o_proj"), skip=("vision", "visual")):
    out = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        parts = name.split(".")
        if parts[-1] in suffixes and not any(s in p for p in parts for s in skip):
            out.append(name)
    return out
```

返回完整名字的列表，只包括语言模型里的层。为什么要跳过视觉塔？第 28 章的讨论：OCR 的微调常常需要视觉塔也学（它要学会看清字），但这应该是一个**有意识的选择**，而不是因为名字碰巧相同而被选中。两种做法都可以试，但要知道自己在做哪一种。

### 第 3 部分：SmolVLM-256M

```
名字以 q_proj 结尾的层共 42 个，其中视觉塔里 12 个；lora_target_names 选出 120 个（只在语言模型里）
LoRA r = 16：可训练 1,843,200 / 258,328,128（0.71%）
```

如果用后缀 `["q_proj", "k_proj", "v_proj", "o_proj"]`，视觉塔的 12 层也会被加上 LoRA。

手算 1,843,200：SmolVLM 的语言模型隐藏维度 576，9 个注意力头、3 个 KV 头，每头 64 维，所以 q、o 是 576 → 576，k、v 是 576 → 192（第 15 章的 GQA）。一个 d_in → d_out 的线性层加上秩 r 的 LoRA，多 r × (d_in + d_out) 个参数（A 是 r × d_in，B 是 d_out × r，第 28 章）：

```
q、o：16 × (576 + 576) = 18,432，两个共 36,864
k、v：16 × (576 + 192) = 12,288，两个共 24,576
每层 61,440，30 层共 1,843,200
```

### 第 4 部分：Qwen3.5-0.8B——默认目标只覆盖四分之一的层

第 24 章讲过，Qwen3.5 的语言模型是**线性注意力和全注意力的混合**：24 层里每 4 层有 3 层是线性注意力（Gated DeltaNet），1 层是全注意力。两种层的投影矩阵**名字不同**：

```
语言模型里的线性层（按名字的最后一段）：
  out_proj 18，in_proj_qkv 18，in_proj_z 18，in_proj_b 18，in_proj_a 18，      ← 18 个线性注意力层
  gate_proj 24，up_proj 24，down_proj 24，                                    ← 24 层都有的 MLP
  q_proj 6，k_proj 6，v_proj 6，o_proj 6，                                    ← 6 个全注意力层
  lm_head 1
LoRA 目标 只写 q/k/v/o_proj        24 个层，覆盖  6 / 24 个 decoder 层
LoRA 目标 加上线性注意力的投影      78 个层，覆盖 24 / 24 个 decoder 层
```

照着别的模型的教程写 `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]`，在 Qwen3.5 上只有 6 个全注意力层得到了 LoRA，**18 个线性注意力层完全没有被训练**。训练照样能跑、损失照样下降（6 层的 LoRA 也有一定的能力），不会有任何报错。这就是为什么要在训练前**数一下目标覆盖了多少层**。

我们的选择：线性注意力层的 `in_proj_qkv`（相当于 q、k、v 的投影）、`in_proj_z`（输出门）、`out_proj`（输出投影），加上全注意力层的 q、k、v、o。`in_proj_a`、`in_proj_b`（决定衰减和写入强度的小投影）没有加——它们的输出维度很小，加不加影响不大。你也可以把 MLP 的三个投影加上（很多配方这样做，自己动手第 2 题）。

## 35.4 阻断项检查

在一个新模型上花几个小时训练之前，用几分钟做一次检查。第 3、4 部分就是这样一个检查清单的实现：

| 检查 | 怎样检查 | 第 3、4 部分的结果 |
|---|---|---|
| 能加载 | `from_pretrained` | 两个模型都能（fp32，CPU） |
| 需要的 kernel | 看 `importlib.util.find_spec`；看加载时的警告 | Qwen3.5：flash-linear-attention、causal_conv1d 都没有，退回 PyTorch 的参考实现（"正确但慢"）；全注意力层用 PyTorch 的 SDPA |
| LoRA 的目标 | 数覆盖了多少个 decoder 层 | SmolVLM 30 / 30；Qwen3.5 只写 q/k/v/o 时 6 / 24 |
| B = 0 时是恒等 | 加 LoRA 前后 logits 的最大差 | 两个模型都是 0 |
| 生成没被破坏 | 加 LoRA（未训练）前后贪心解码的结果 | Qwen3.5 读第 2 行：相同（`'spell may'`，读对了） |
| 能学 | 同一小批样本重复训练几步，损失要下降 | SmolVLM：8.93 → 6.37 → 4.36 → 2.56 → 1.77 |
| 能合并 | `merge_and_unload` 前后 logits 的最大差 | SmolVLM：7.68 × 10⁻⁵（fp32 的舍入误差） |
| 速度与内存 | 计时、峰值内存 | SmolVLM：4 条样本一步 1.7 秒，峰值 2.2 GB；Qwen3.5：112 个 token 的前向 + 反向 1.0 秒，峰值 7.9 GB |
| 训练库能不能在你的设备上跑 | 小步数试跑 | trl 1.15 的三个 trainer：不能在 CPU 上跑（下面） |

几点说明：

- **"能学"的检查只是过拟合检查**：4 条样本重复训练 5 步，损失从 8.9 降到 1.8，只说明梯度能流到 LoRA 的参数上、学习率不是太小。它不说明模型学会了 OCR；
- **合并后的差别 7.7 × 10⁻⁵**：合并是把 (α/r)·B·A 加到原来的权重上，数学上完全等价，数值上 W + ΔW 和 W·x + ΔW·x 的舍入方式不同，fp32 下这个量级的差别是正常的。如果差别是 10⁻¹ 量级，说明合并的方式（比如 α/r 的缩放）有错；
- **"和训练前的最大差 14.64"**：训练了 5 步之后，logits 变了很多——在只有 4 条样本上过拟合，这是预期的。

### trl 1.15 的 trainer 不能在 CPU 上跑

第 1 部分的最后一行：

```
trl 的融合 LM head（Triton kernel）支持 CPU 张量吗：False
（SFTTrainer、DPOTrainer、GRPOTrainer 都无条件地装上它，所以这三个 trainer 在这个版本里只能在 GPU 上跑）
```

这是我们在 CPU 上试跑 `GRPOTrainer` 时发现的：训练的第一步就报错 `RuntimeError: 0 active drivers ([]). There should only be one.`。原因是 trl 1.15 为了省内存，把"LM head + log_softmax + 取出目标 token 的对数概率"融合成一个分块计算的 Triton kernel（不需要生成完整的 (B, T, V) logits），三个 trainer 在初始化时都无条件地把模型的前向换成这个版本；而这个 kernel 只支持 CUDA 和 XPU。

这类问题只有试跑才能发现：文档里不一定写，配置里也没有开关。对我们的影响是：

- **trl 的路线只能在 GPU 上验证**。本书的作者环境的 GPU 被别的程序占着，所以本章**没有**实际跑过 trl 的 trainer，下面 35.6 节的 trl 代码是按 trl 1.15 的接口写的参考，没有经过测试；
- **我们自己的训练循环在 CPU 上能跑**：第 6–8 部分用的都是自己的循环（HF 模型 + 本书的损失函数），作者用很小的设置冒烟测试过。

## 35.5 显存估算

第 5 部分用第 28 章的 `static_memory` 估算参数、梯度、优化器状态（不含激活值）：

```
模型            全参数 fp32   全参数 bf16 混合   LoRA（bf16 底座）
SmolVLM-256M       3.82 GB          3.82 GB         517.33 MB
Qwen3.5-0.8B      12.71 GB         12.71 GB           1.66 GB
```

（第 28 章讲过：bf16 混合精度训练每个参数仍然要 16 字节——bf16 的参数和梯度各 2 字节，fp32 的主权重 4 字节，AdamW 的两个状态 8 字节——所以和 fp32 一样。省显存靠的是 LoRA：底座的参数只存 bf16 的 2 字节，没有梯度和优化器状态。）

在一张 16 GB 的显卡上：

- Qwen3.5-0.8B 的全参数训练，静态的部分就要 12.71 GB，再加上激活值（第 28 章：和序列长度、批大小成正比；一张文档页有上千个视觉 token）几乎一定放不下，需要梯度检查点、更小的批或者 8 位优化器；
- LoRA 的静态部分只有 1.66 GB，剩下的显存都可以给激活值；
- **先看显存有没有被别的程序占用**：`nvidia-smi`。本书作者的 GPU 上同时跑着一个推理服务，只剩不到 1 GB——这时任何训练都放不下。

## 35.6 用我们自己的循环训练真实模型

### SFT

第 6 部分的核心循环：

```python
pm = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=lora_target_names(m)))
for 每 accum 条样本:
    for 每条:
        b, k = smol_example(p, img, text)                      # processor + 前缀法定出 mask
        loss = -hf_answer_logp(pm(**b).logits, b["input_ids"], k, "mean").mean() / accum
        loss.backward()
    设学习率（预热 5% + 余弦）；裁剪梯度到 1.0；opt.step()；opt.zero_grad()
```

和第 26 章的区别只有三处：模型换成了 HF 的模型（输入是 processor 的输出），mask 用前缀法定，LoRA 用 peft。损失还是"回答的 token 的平均负对数概率"（第 26 章的按 token 平均）。

为什么一条一条地前向（而不是凑成一个批）？SmolVLM 的 processor 按图的大小切子图，不同的图切出的子图数不同（第 24 章）；Qwen 的视觉 token 数也随图的大小变化。凑批需要补齐、需要处理不同数量的图像 token。一条一条地做加梯度累积最简单、不容易错，代价是慢。真实的训练框架会把同样大小的图分到一个批里，或者做打包（第 26 章）。

### GRPO

第 7 部分的一步：

```python
for 每道题:
    full, mask, outs = smol_rollout(p, pm, img, G)            # 采 G 个回答（温度 1，top_k=0），回答到第一个结束符为止
    r = trl_reward("text")(outs, [ref] * G)                    # 1 − NED，格式闸门（第 31 章）
    adv = group_advantages(r, G)                               # 组内标准化（第 29 章）
    lp = token_logprobs(pm(**full).logits, full["input_ids"])
    loss, st = grpo_loss(lp, lp.detach(), adv, mask[:, 1:], mode="token")   # μ = 1，不加 KL
    (loss / n_prompt).backward()
```

和第 31 章的 `grpo_train` 一样，只是采样和前向都换成了 HF 的模型。`lp.detach()` 当作旧的对数概率：μ = 1（每批只更新一次），比值恒为 1，裁剪不起作用——和 trl 的默认 `num_iterations=1` 相同。

**起点很重要**。第 7 部分默认从第 6 部分 SFT 之后的 LoRA 出发。原始的 SmolVLM 几乎读不出这些文字行：第 24 章测到 CER 超过 100%；作者的冒烟测试里，它对"Read the text in the image."的回答是一段图像描述（"### Image Description\n\nThe image is a scanned page from a book…"）。这时每组 G 个回答的奖励都接近 0，组内的差别很小，GRPO 几乎学不到东西（第 31 章：没有梯度的组）。先 SFT、再 RL，是几乎所有 VLM 后训练配方的顺序（第 34 章的 Xiaomi-OCR-0 也是）。

### 练习 35.3：`trl_reward`

如果你在 GPU 上用 trl 的 `GRPOTrainer`，奖励函数的签名是 `f(completions, **kwargs) -> list[float]`：trl 把数据集里除 `prompt` 以外的列按名字传进来。

```python
def trl_reward(task):
    def text_of(c):
        if isinstance(c, str):
            return c
        content = c[-1]["content"]
        if isinstance(content, str):
            return content
        return "".join(p.get("text", "") for p in content if p.get("type") == "text")

    def f(completions, ref, **kwargs):
        return [task_reward(task, text_of(c).strip(), r) for c, r in zip(completions, ref)]
    f.__name__ = f"reward_{task}"
    return f
```

数据集用对话格式时，`completions` 的每一项是 `[{"role": "assistant", "content": ...}]`，`content` 可能是字符串，也可能是多段内容的列表。`__name__` 会出现在 trl 的日志里（每个奖励函数的平均值）。

### trl 的写法（参考，未在本书的环境里运行）

```python
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
ds = Dataset.from_dict({"prompt": [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]}]] * n,
                        "image": imgs, "ref": texts})
cfg = GRPOConfig(output_dir="out", learning_rate=1e-5, num_generations=8, max_completion_length=24,
                 per_device_train_batch_size=8, beta=0.0, loss_type="dapo", epsilon_high=0.28, logging_steps=10)
trainer = GRPOTrainer(model=model, processing_class=processor, reward_funcs=[trl_reward("text")], args=cfg,
                      train_dataset=ds, peft_config=LoraConfig(r=16, lora_alpha=32, target_modules=names))
trainer.train()
```

再强调一次：这段代码在作者的环境里只跑到了第一步的 Triton 报错（CPU）。在你的 GPU 上跑之前，先用 `max_steps=2` 试跑，并核对 trl 的日志里 `completions/mean_length`、`reward`、`clip_ratio` 这些量和你用自己的循环得到的是否一致。

### 自己的循环 vs trl

| | 自己的循环 | trl 的 trainer |
|---|---|---|
| 透明 | 每一步都看得见，和第 26–31 章的公式一一对应 | 很多细节藏在默认值里（35.1 节） |
| 速度 | 一条一条地做，慢 | 批处理、融合 kernel、vLLM 加速采样 |
| 显存 | 要生成完整的 logits | 融合的 LM head 不生成完整的 logits |
| 分布式、检查点、日志 | 要自己写 | 有 |
| 设备 | CPU、GPU 都行 | 1.15 版：只有 GPU |
| 适合 | 学习、调试、小实验、核对库的行为 | 正式的训练 |

一个好的习惯：用自己的循环在小数据上跑出一个结果，再用 trl 跑同样的设置，两者的损失曲线、奖励曲线应该接近。不接近时，往往是某个默认值和你以为的不一样。

## 35.7 实验

```bash
python -m labs.ch35.real_tour 1 2 5     # 不加载权重：库的对应与默认值、聊天模板、显存估算（约 10 秒）
python -m labs.ch35.real_tour 3 4       # 加载权重，CPU：SmolVLM 与 Qwen3.5 的 LoRA 检查（作者的 CPU 上约 20 秒，峰值内存约 8 GB）
python -m labs.ch35.real_tour 6         # 你自己跑：SmolVLM-256M 的 LoRA SFT
python -m labs.ch35.real_tour 7         # 你自己跑：SmolVLM-256M 的 GRPO
python -m labs.ch35.real_tour 8         # 你自己跑：Qwen3.5-0.8B 的 LoRA SFT
# 有 GPU：在后面加 --gpu（先用 nvidia-smi 确认显存够）
```

需要 `requirements-hf.txt` 和第 24 章下载好的两个模型。

### 第 6 部分：SmolVLM-256M 的 LoRA SFT（你自己跑）

**做什么**：第 13 章的文字行（噪声 0.15、形变 1.0），2,000 行训练、1 轮、每 8 条更新一次（250 步），学习率 10⁻⁴（LoRA 常用的量级，比全参数大），预热 5% + 余弦衰减。图像的长边缩放到 512（第 24 章：64 个视觉 token）。训练前后在 50 行测试图上（第 24 章的同一批）测 CER 和整行全对。

**时间**：第 3 部分测到 4 条样本一步约 1.7 秒（CPU），所以 2,000 条约 15 分钟；加上训练前后的评测（每行几秒的生成）。作者用 8 条训练、2 条测试冒烟测试过。

**运行前先预测**：

1. 训练前的 CER 是多少？（第 24 章：长边 512 时 CER 346%，模型在写图像描述。）
2. 2,000 行、250 步之后，CER 能降到多少？能比第 13 章专门训练的 CRNN（CER 2.28%）好吗？
3. 模型学会的是"读字"还是"不再写描述、改写一个短答案"？怎样区分？（提示：看错误的类型——是整行写错，还是个别字错。）

**跑完后看什么**：

- 训练日志的损失曲线：前几十步是不是下降得特别快？那可能主要是在学**格式**（短答案、结束符），不是读字；
- 训练后的预测：错误是字符级的（形近字、漏字），还是整行的（完全不相关的文字）？
- 和第 24 章 Qwen3.5 的零样本（2× 放大时 CER 12.80%）比，SmolVLM 微调后能追上吗？

### 第 7 部分：SmolVLM-256M 的 GRPO（你自己跑）

**做什么**：从第 6 部分的 LoRA 出发，100 步，每步 4 道题 × 8 个回答，学习率 10⁻⁵，奖励是 1 − NED（第 31 章的 `task_reward("text")`），μ = 1，不加 KL。训练前后测同样的 50 行。

**运行前先预测**：

1. 第 6 部分之后，每组 8 个回答里，"没有梯度的组"（8 个回答的奖励全相同）的比例大概是多少？
2. GRPO 之后 CER 会降吗？会降多少？（第 31 章第 4 部分：Qwen3.5 在 OCR 上，0/1 奖励时一半的组没有梯度，1 − NED 时少一些。）
3. 回答的长度会变吗？（第 31 章：按 token 汇总时，长回答的梯度份额更大。）

**跑完后看什么**：平均奖励和"没有梯度的组"随步数的变化；测试 CER；把第 7 部分改成从原始的 SmolVLM 出发（删掉或改名 `outputs/ch35/smol_lora`）再跑一次，对照"先 SFT 再 RL"的必要性。

### 第 8 部分：Qwen3.5-0.8B 的 LoRA SFT 与 Xiaomi-OCR-0（你自己跑）

**做什么**：Qwen3.5-0.8B，500 行、每 8 条更新一次，LoRA 加在全部 78 个目标上（第 4 部分），图像放大 2 倍（第 24 章的最好设置）。训练前后测 50 行。

**时间**：第 4 部分测到一条样本的前向 + 反向约 1 秒（CPU），500 行约 10 分钟；评测时每行的生成约 5 秒（第 24 章）。作者用 4 条训练、2 条测试冒烟测试过（这个设置下的数字没有意义）。

**Xiaomi-OCR-0 的零样本**：它是从 Qwen3.5-0.8B-Base 继续训练的 OCR 模型（Apache-2.0，第 34 章），结构和 Qwen3.5 相同，所以可以用同一套代码评测：下载 `SeerRay-Lab/Xiaomi-OCR-0`（先读模型卡），把 `labs/ch24/hf_tour.py` 里的 `QWEN` 换成它的名字，用 `qwen_ask` 在同样的 50 行上评测。注意它的提示词：模型卡里给出了它训练时用的提示词，用它的提示词和用我们的提示词，结果可能差别很大（第 25 章：评测设置要写清楚）。

**运行前先预测**：Qwen3.5 在 500 行上微调之后，能超过 Xiaomi-OCR-0 的零样本吗？（Xiaomi-OCR-0 训练了约 1.7 亿个样本，但没有见过我们的渲染器；我们的 500 行是同分布的。）

### 记录表

| 模型 | 设置 | 训练前 CER | 训练后 CER | 训练前整行全对 | 训练后整行全对 | 用时 |
|---|---|---|---|---|---|---|
| SmolVLM-256M | 第 6 部分 LoRA SFT | | | | | |
| SmolVLM-256M | 第 7 部分 GRPO（从第 6 部分出发） | | | | | |
| SmolVLM-256M | 第 7 部分 GRPO（从原始模型出发） | | | | | |
| Qwen3.5-0.8B | 第 8 部分 LoRA SFT | | | | | |
| Xiaomi-OCR-0 | 零样本（模型卡的提示词） | | — | | — | |
| 第 13 章 CRNN | 专门训练 | — | 2.28% | — | | |

## 35.8 自己动手

### 必做

1. **前缀检查的反例**：在 SmolVLM 的分词器上，找一个提示结尾和回答开头会被合并成一个 token 的例子（提示：试试回答以冒号后面常见的字符开头，或者提示不以空格结尾而回答以字母开头），确认 `answer_mask_by_prefix` 报错。
2. **LoRA 目标的消融**：第 8 部分用三种目标各训练一次：只有 q/k/v/o（6 层）、加上线性注意力的投影（24 层）、再加上 MLP。比较训练后的 CER 和可训练参数量。
3. **SFT 默认值的陷阱**：在第 6 部分里把 mask 换成"全部 token 都算损失"（相当于 trl 的 `assistant_only_loss=False`），其他不变。训练后的 CER 和第 6 部分比有什么不同？模型会不会开始"复述"提示？

### 选做

4. **视觉塔也加 LoRA**：在第 6 部分里用 `skip=()`（视觉塔的 q/k/v/o 也加 LoRA），比较 CER。读字的错误主要在视觉还是语言？
5. **trl 对照**（需要 GPU）：用 35.6 节的 trl 写法跑第 7 部分的设置，和自己的循环比较奖励曲线。
6. **批处理**：把第 6 部分改成每批 8 条一起前向（同样大小的图，processor 一次处理 8 张），测速度的提升，并检查损失和一条一条地做是否一致。
7. **用错归一化**（第 2 章）：SmolVLM 的 processor 按 `image_processor.image_mean`、`image_std` 归一化。在 `smol_read` 里把 processor 输出的 `pixel_values` 换成用另一组 mean/std（比如 CLIP 的，或者干脆不归一化、直接 [0, 1]）重新算出来的值，测训练前的 CER。模型会报错吗？CER 变了多少？

## 35.9 常见错误

| 错误 | 症状 | 原因与修正 |
|---|---|---|
| 照搬别的模型的 LoRA 目标 | 训练能跑，但效果差 | Qwen3.5 只写 q/k/v/o 时只覆盖 6 / 24 层；训练前数一下覆盖了多少层 |
| 用后缀选 LoRA 目标 | 视觉塔也被训练了 | 视觉塔的层也叫 q_proj；用完整名字的列表 |
| 提示和回答分别分词再拼接 | 训练和推理的 token 序列不一致 | 用整段文字分词，再用前缀法定 mask |
| 不检查前缀 | 回答的第一个 token 不算损失，或者提示的最后一个算了 | `answer_mask_by_prefix` 不是前缀时报错 |
| 回答前少了空格（或多了） | 和模板的格式不一致 | 用 `apply_chat_template` 生成带回答的完整对话来确认 |
| 不看训练库的默认值 | 以为在复现 GRPO 原文，其实 β = 0、按 token 汇总 | 打印配置，逐项对照 |
| SFT 用了默认的 `assistant_only_loss=False` | 提示也算损失 | 打开 `assistant_only_loss` 或用提示 / 回答两列的格式 |
| 在原始模型上直接做 GRPO | 奖励全接近 0，学不动 | 先 SFT 学会格式，再 RL |
| 不确认显存 | 训练一开始就显存不足 | `nvidia-smi`；第 28 章的估算 |
| 合并后的差别很大 | 合并之后效果变了 | 检查 α/r 的缩放；fp32 下差别应该在 10⁻⁵ 量级 |

## 35.10 真实世界对照

| 本章 | 真实世界 |
|---|---|
| 自己的循环 + HF 模型 | 很多研究代码的做法：用 transformers 加载模型，训练循环自己写 |
| trl 的 trainer | Hugging Face 生态的标准训练库；VLM 的 SFT、DPO、GRPO 都支持 |
| 其他训练框架 | LLaMA-Factory（配置驱动的 SFT / DPO）、verl 与 OpenRLHF（大规模 RL，采样用 vLLM / SGLang）、ms-swift |
| 融合的 LM head | 大词表（Qwen3.5 约 25 万）的模型，完整的 (B, T, V) logits 非常大；Liger Kernel、trl 的分块 kernel 都是为了不生成它 |
| 阻断项检查 | 换模型、换库的版本、换硬件时都要做；kernel 和版本的兼容性是最常见的问题 |
| Xiaomi-OCR-0 | 从 Qwen3.5-0.8B-Base 出发：Q-Mask 锚定 → 持续预训练 → Mix-RL（第 34 章） |

## 35.11 深入一步

### 为什么学习率差这么多

trl 的默认学习率：SFT 2 × 10⁻⁵，DPO 和 GRPO 10⁻⁶；本章的 LoRA SFT 用 10⁻⁴，GRPO 用 10⁻⁵。差别来自两个方面：

- **LoRA vs 全参数**：LoRA 的更新是 (α/r)·B·A，B 从 0 开始，一开始的有效更新很小；而且 LoRA 只改变一个低秩的子空间。所以 LoRA 常用比全参数大 5–10 倍的学习率（第 32 章的 v2 实验也看到了：同样的学习率下，LoRA 和全参数的"有效步长"不同）；
- **SFT vs RL**：RL 的梯度方差大（第 29 章），而且模型在自己的输出上训练，一步走太远容易让输出分布崩溃（第 31 章的熵崩溃），所以学习率小得多。

这些都只是起点：第 32 章的教训是，比较方法时每种方法都要自己扫学习率。

### 为什么一个 0.8B 的模型在 CPU 上训练只要 1 秒一步

第 4 部分的序列只有 112 个 token（文字行放大 2 倍只有 32 个视觉 token）。训练的计算量约是 6 × 参数量 × token 数（第 28 章的估算：前向 2 × 参数量 × token，反向约两倍）= 6 × 0.85 × 10⁹ × 112 ≈ 5.7 × 10¹¹ 次浮点运算；CPU 8 线程约每秒几千亿次，所以约 1 秒。一张 150 dpi 的整页有 2,145 个视觉 token（第 33 章），一步就要约 20 倍的时间，而且线性注意力的参考实现在长序列上更慢。这就是为什么整页的训练需要 GPU。

## 35.12 横向对照

| 做法 | 改动的参数 | 显存 | 效果 | 适合 |
|---|---|---|---|---|
| 全参数 | 全部 | 最多（16 字节 / 参数 + 激活） | 上限最高 | 资源充足、数据多 |
| 冻结视觉塔、全参数训练语言模型 | 语言模型 + 连接器 | 少一些 | 视觉不变 | 视觉已经够好的任务 |
| LoRA（只在语言模型） | 低秩增量 | 很少 | 通常接近全参数 | 资源有限、数据少 |
| LoRA（视觉 + 语言） | 两边的低秩增量 | 很少 | 视觉也能适应 | 新的视觉域（比如手写、特殊字体） |
| QLoRA | 底座 4 位量化 + LoRA | 最少 | 略差于 LoRA | 显存极其有限（需要 bitsandbytes，本书环境没有安装） |

## 35.13 本章小结

1. transformers 管模型，peft 管 LoRA，trl 管训练流程；它们和本书的函数一一对应；
2. 训练库的默认值要打印出来逐项核对：trl 1.15 的 GRPO 默认 β = 0、按 token 汇总、μ = 1；SFT 默认整段对话都算损失；
3. 回答的 mask 用"提示是完整序列的前缀"来定，并且检查前缀；回答前的空格按模板的完整对话来定；
4. LoRA 的目标要数覆盖了多少层：Qwen3.5 的混合注意力结构让 q/k/v/o 只覆盖 6 / 24 层；用后缀选目标会把视觉塔也选上；
5. 阻断项检查：kernel、B = 0 的恒等、生成不变、过拟合检查、合并一致、速度与内存、训练库在你的设备上能不能跑（trl 1.15 的 trainer 只能在 GPU 上跑）；
6. LoRA 让 Qwen3.5-0.8B 的静态显存从 12.71 GB 降到 1.66 GB；
7. 用自己的循环训练真实模型：前向换成 HF 的模型，损失还是本书的损失；先 SFT 再 RL。

下一章是毕业项目：把整本书的方法用在一个完整的研究问题上。

## 术语卡

| 中文 | 英文 | 一句话 |
|---|---|---|
| 聊天模板 | chat template | 把对话变成模型输入文字的规则（Jinja 模板） |
| 生成提示 | generation prompt | 提示末尾的"助手开始说话"的标记 |
| 前缀法 | prefix masking | 提示是完整序列的前缀，前缀之后的是回答 |
| LoRA 目标 | target modules | 加 LoRA 的线性层 |
| 合并 | merge | 把 LoRA 的增量加回原来的权重 |
| 阻断项检查 | blocker check | 训练前确认模型、kernel、库、显存都没有问题 |
| 参考实现 | reference implementation | 没有优化 kernel 时用的纯 PyTorch 实现（正确但慢） |
| 融合的 LM head | fused LM head | 不生成完整 logits、直接算目标 token 对数概率的 kernel |
| 奖励函数 | reward function | trl 里 `f(completions, **kwargs) -> list[float]` |

## 自测题

**概念**

1. transformers、peft、trl 各管什么？
2. 为什么不能只靠"提示单独分词的长度"来定回答从哪里开始？
3. Qwen3.5 的训练模板里有一段空的 `<think>\n\n</think>\n\n`，它应该算损失吗？为什么？
4. 为什么在 Qwen3.5 上写 `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]` 是一个静默错误？
5. "同一批样本重复训练几步，损失下降"说明了什么、不说明什么？
6. trl 1.15 的 `GRPOConfig` 默认值里，哪几项和 GRPO 原文不同？
7. 为什么在原始的 SmolVLM 上直接做 GRPO 很难学到东西？

**计算**

8. 一个线性层 2048 → 512，加 r = 8 的 LoRA，多多少个参数？
9. SmolVLM 的 30 层里，如果 LoRA 只加在 q、v 上（r = 16），可训练参数是多少？
10. 用第 35.11 节的估算：Qwen3.5-0.8B 在一张 2,145 个视觉 token、回答 1,000 个 token 的整页上训练一步，大约多少次浮点运算？

**代码**

11. 下面的 mask 有什么问题？
    ```python
    p_ids = tok(prompt)["input_ids"]
    a_ids = tok(answer + "<|im_end|>")["input_ids"]
    ids = p_ids + a_ids
    mask = [0] * len(p_ids) + [1] * len(a_ids)
    ```
12. 下面的 LoRA 统计有什么问题？
    ```python
    pm = get_peft_model(m, LoraConfig(target_modules=lora_target_names(m)))
    print(len(lora_target_names(pm)))
    ```

## 参考答案

1. transformers：模型的结构、权重、processor、生成；peft：参数高效的训练（LoRA 等），把可训练的增量加到模型上、合并、保存；trl：训练流程（SFT、DPO、GRPO 的数据处理、损失、采样、日志）。
2. BPE 会把提示的最后几个字符和回答的前几个字符合并成新的 token，拼起来分词之后，回答开始的位置可能不是"提示单独分词的长度"。所以要检查提示的 token 序列是完整序列的前缀，不是前缀就报错。
3. 不应该。推理时 `add_generation_prompt=True` 生成的提示已经包括这一段，模型直接从回答开始写，不需要学着生成它。算进损失会让模型把训练的力气花在一段固定的、推理时由提示给出的文字上。（如果你想训练模型"思考"，那是另一回事：思考的内容就是回答的一部分。）
4. Qwen3.5 的 24 层里有 18 层是线性注意力，它们的投影叫 `in_proj_qkv`、`in_proj_z`、`out_proj` 等，不叫 q/k/v/o_proj。只写 q/k/v/o 时 LoRA 只加在 6 个全注意力层上，训练照样能跑、损失照样下降，没有任何报错，但 18 层完全没被训练。
5. 说明梯度能流到可训练的参数、学习率不至于太小、损失的计算没有明显的错误（比如 mask 全为 0）。不说明模型学会了任务：在 4 条样本上过拟合，和在测试集上读对字是两回事。
6. `beta=0.0`（原文 0.04，不加 KL），`loss_type='dapo'`（按 token 汇总；原文每条回答先平均），`num_iterations=1`（μ = 1，裁剪不起作用；原文的设定是多次更新）。`scale_rewards='group'`（除以组内标准差）和原文一致。
7. 原始的 SmolVLM 对这个提示写的是图像描述，几乎每个回答的 1 − NED 都接近 0，组内的奖励几乎相同，组内标准化之后的优势很小或者为 0（第 31 章：没有梯度的组）。RL 只能在模型"有时做对"的题上学习；先 SFT 让模型学会格式和基本的读字，RL 才有信号。
8. r × (d_in + d_out) = 8 × (2048 + 512) = 20,480。
9. q：16 × (576 + 576) = 18,432；v：16 × (576 + 192) = 12,288；每层 30,720，30 层 921,600。
10. 6 × 0.85 × 10⁹ × (2,145 + 1,000 + 提示的几十个 token) ≈ 6 × 0.85 × 10⁹ × 3,200 ≈ 1.6 × 10¹³ 次。是文字行（112 个 token）的约 30 倍。
11. 提示和回答分别分词再拼接：拼出来的序列和"整段文字分词"的结果可能不同（边界上的合并），训练时模型看到的 token 序列和推理时不一样。而且 `tok(...)` 可能会给回答也加上开头的特殊 token（取决于分词器的 `add_special_tokens` 设置），回答前面多出一个 token。应该把整段文字一起分词，再用前缀法定 mask。
12. `get_peft_model` 之后，原来的 `nn.Linear` 被 LoRA 的层包了起来（`lora.Linear`，不再是 `nn.Linear` 的实例），`lora_target_names(pm)` 会返回空列表（第 4 部分的第一版就犯了这个错误，打印出"全部 0 个目标"）。要在 `get_peft_model` 之前把目标列表存下来。
