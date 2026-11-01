# 第 33 章　习题：VLM 的 LoRA 微调

说明：题目里的实测数字都来自正文引用的文件（`course-v4/outputs/book/ch33/gpu/remote_runs/`，下文简称 `runs/`），RTX 4090（租用），bf16，单次运行。标"【离线重算】"的数字是作者从 val JSON 重新统计的，练习 33.4 让你自己重算。

---

## 一、概念题

**1.** `answer_labels` 已经按"位置累加和 ≤ 提示长度"遮掉了整个提示，图片占位符本来就在提示里。为什么还要再按 token id 遮一遍图片 token？去掉这一步，正常数据上会出错吗？什么样的数据上会出错？

**2.** 左 padding 时，答案区的下标会整体右移。为什么用 attention_mask 的累加和来判断"是否进入答案区"，左右 padding 可以共用一条公式？

**3.** tour 部分 3 的实测中，Qwen3.5 language 策略挂了 186 个 LoRA 模块，每个模块有 A、B 两个张量，但第一步只有 186 个参数张量的梯度非零。是哪 186 个？为什么另外 186 个梯度严格为 0？代码为什么不能断言"每个可训练参数的梯度都非零"？

**4.** connector 策略下，`parameters.json` 里 connector 组的总参数是 25,177,088，而 merger 只有 12,588,544 个参数。多出来的一份是什么？它和 `disable_adapter()` 能恢复零样本基线有什么关系？

**5.** Qwen3.5 微调最后一步的训练 loss 是 0.242，比 Qwen3-VL 的 0.258 还低，生成时却有 75/100 条写满上限。用"teacher forcing"和"自由生成"的区别解释：为什么训练 loss 发现不了这个问题？

**6.** 视觉依赖对照里，错图条件为什么要把别人的图 resize 到原图尺寸？如果不 resize，你测到的差异里混进了什么？

**7.** 视觉依赖对照的三行结果能证明"Qwen3-VL 微调后的输出依赖图片"，但不能证明"它的每个符号都是从图上读来的"。说明两者的区别，并设计一个能回答后一个问题的检查（不要求实现）。

**8.** 本章的 test 只保存汇总，不保存逐条预测。这个做法防的是什么？它让本章在 test 上失去了哪种统计方法？

**9.** Qwen3-VL 微调后 parse_rate 从 0.76 降到 0.65。有人说"微调让输出变得不合法了"。结合"val 目标本身只有 53/100 可解析"【离线重算】，评价这个说法。

---

## 二、手算题

**1. label mask。**词表里 id 99 是图片占位符，id 0 兼任 padding 和 EOS。计算下面 batch 的 labels（形状 [2, 7]），并写出错开一位以后哪些 logits 位置有梯度：

```text
input_ids      = [[ 0,  0, 11, 99, 12, 30,  0],
                  [11, 99, 99, 12, 31, 32,  0]]
attention_mask = [[ 0,  0,  1,  1,  1,  1,  1],
                  [ 1,  1,  1,  1,  1,  1,  0]]
prefix_lengths = [3, 4]
```

**2. 可训练参数。**用正文 33.3.4 节的模块表：

(a) Qwen3.5-0.8B 的 language 策略，r = 8，可训练参数是多少？

(b) 只挂 `q_proj` 和 `v_proj`，r = 16，有多少个模块、多少参数？占正文 language 策略（10,822,656）的百分之几？

(c) 只挂 DeltaNet 层的 5 个线性层，r = 16，有多少参数？

**3. 视觉 token 数。**一张公式图高 40、宽 400。像素预算 min = 16384、max = 65536，因子 32。按第 32 章 32.2.3 节的 `smart_resize` 规则（注意 Python 的 `round` 是四舍六入五成双），算出缩放后的尺寸、视觉 token 数和视觉塔里的 patch 数。

**4. 静态显存。**Qwen3.5-0.8B 用 vision 策略、r = 16。基座 bf16（N_base = 852,985,920），LoRA 参数 fp32（16 字节/参数，含梯度和 AdamW 的 m、v），merger 副本 bf16（权重 2 字节 + 梯度 2 字节 + AdamW 两个 bf16 状态 4 字节）。merger 原件已经算在基座里。估算静态显存，单位 GiB。

**5. Wilson 区间。**手算 10/100 的 Wilson 95% 区间（z = 1.96），保留 4 位小数，再和 `runs/qwen3vl_main_val/val_finetuned.json` 里记录的区间比较。

**6. 精确 McNemar。**

(a) 微调多对 5 条、少对 0 条，双侧 p 值是多少？

(b) 如果是多对 8 条、少对 1 条呢？

(c) 解释为什么 (a) 中配对 bootstrap 的 95% 区间 [0.01, 0.10] 不含 0，精确检验却没有达到 0.05。

**7. 步数与 epoch。**lab 的补充实验用 20,000 条训练数据、batch 16 × 累积 1、3000 步。共呈现多少条样本？是多少个 epoch？和主实验的 1.17 个 epoch 比是几倍？

**8. 样本量。**要以 95% 置信度、80% 检验力区分 EM 0.10 和 0.15（独立两组，正态近似），每组大约要多少条？补充实验的 500 条 val 够不够？

---

## 三、编程题

**练习 33.1　答案监督 mask（必做）**

在 `miniocr/data/vlm_sft.py` 中找到 `# >>> 练习 33.1`，实现 `answer_labels(input_ids, attention_mask, prefix_lengths, image_token_ids=())`：

- 支持左右 padding；padding 按 attention_mask 判断，不按 token 值；
- prefix_lengths 是每条样本不含 padding 的提示 token 数；
- 图片 token 无论出现在哪里都设为 −100；
- 输入形状不对、长度为负、或者某条样本一个监督 token 都没有时，抛出 `ValueError`。

验证：

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q book/part4_vlm/ch33_vlm_lora \
  -k "answer_mask or padding_id or missing_answer or shifted_mask"
CUDA_VISIBLE_DEVICES="" python -m book.part4_vlm.ch33_vlm_lora.tour 1
```

tour 1 的 `mask.json` 应和正文 33.2.4 节的玩具 batch 结果一致。

**练习 33.2　按模块树选目标（必做）**

在 `miniocr/train/vlm_lora.py` 中找到 `# >>> 练习 33.2`，实现 `adapter_spec(model, strategy)`：

- language：语言模型里的所有 `nn.Linear`；
- connector：再加上整块连接器（`visual.merger`，以及 Qwen3-VL 的 deepstack mergers）作为 `modules_to_save`；
- vision：再加上视觉塔里除连接器以外的 `nn.Linear`；
- 找不到语言线性层，或非 language 策略找不到连接器时报错。

验证：

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q book/part4_vlm/ch33_vlm_lora -k target_selection
CUDA_VISIBLE_DEVICES="" python -m book.part4_vlm.ch33_vlm_lora.tour 2
```

**练习 33.3　test 锁（必做）**

在 `tour.py` 的 `part5` 中找到 `# >>> 练习 33.3`。实现：`--split test` 时必须同时给出 `--final-test` 和 `--adapter`；锁文件建在**数据目录**（不是输出目录）下，用排他方式创建，已存在就失败；锁里先写入时间和完整配置，再开始评测。

验证：

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q book/part4_vlm/ch33_vlm_lora \
  -k "test_protocol or lock_cannot"
```

思考：为什么锁放在输出目录下就不够？

**练习 33.4　从 val JSON 重算诊断数字（必做）**

写一个脚本（放在你自己的目录，不要改仓库代码），只读 `runs/` 下已有的 val JSON，不加载模型，重算正文里所有标"【离线重算】"的数字：

1. `qwen35_main_val/val_finetuned.json` 的 75 条截断里，含 `\nassistant` 的条数（正文：74）和其余条数（1），以及平均 `generated_tokens`（正文：209.3；Qwen3-VL 微调为 65.6）。
2. 用 `miniocr.eval.postcorrect_eval.em_ignoring_final_punct` 重算 33.7.5 节的三行（3→22、5→23、10→29），以及 val 目标中规范化后以 `.` `,` `;` 结尾的条数（67）。
3. 视觉依赖错图条件下，第 i 条预测与第 i+1 条目标规范化后一致的条数（正文：7）。
4. 把 Qwen3.5 微调的 raw 在第一个 `\nassistant` 处切断，再用 `extract_formula` 和 `score_postcorrect` 评分（正文：EM_norm 5/100，CER 0.174）。
5. 用 `book.part4_vlm.ch33_vlm_lora.summarize_supplement.paired` 比较 Qwen3-VL 零样本与微调的 val（正文：EM +0.05 [0.01, 0.10]，CER −0.060 [−0.087, −0.034]，多对 5 条、少对 0 条）。

验证：你的数字和正文一致。如果不一致，先检查样本 id 的顺序和 `normalize_latex` 的用法，再报告差异。

**练习 33.5　参数公式对拍（选做）**

写一个函数，只读模型的 `config.json`，按 33.3.4 节的模块表算出三种策略的可训练参数，和 `runs/*/parameters.json` 对比。

- Qwen3.5-0.8B：本地已有 config（`~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364…/config.json`），应得到 10,822,656、23,411,200、25,770,496。
- Qwen3-VL-2B：正文的结构数字（hidden 2048、28 层、16/8 头、头宽 128、MLP 6144）标为待验证。在实验机的模型缓存里找到它的 config.json，确认这些数字，并核对 17,432,576；顺便确认它的词表大小是否为 151,936。

**练习 33.6　验证停止 token 假说（选做，需 GPU）**

正文 33.8.4 节的解释是：评测时 `<|im_end|>` 不在 Qwen3.5 的停止集合里。在实验机上验证（只用 val，不碰 test，结果只做诊断）：

1. 加载 Qwen3.5 基座和 `qwen35_main/adapter`，打印 `model.generation_config.eos_token_id`。
2. 取 10 条截断样本，生成后用 `skip_special_tokens=False` 解码，看 `<|im_end|>` 是否出现在公式后面。
3. 在你自己的脚本副本里把 `eos_token_id=[248046, 248044]` 传给 `generate`，重评这 100 条 val，记录截断数、EM_norm、CER。

写下：假说是否成立？如果成立，这是训练的问题还是评测配置的问题？要修正，你会改 label（不监督 `<|im_end|>` 后的换行）、改生成配置，还是两者都改？各自对第 34 章有什么影响？

**练习 33.7　答案比输出上限还长（选做）**

正文 33.2.4 节指出训练集的监督 token 最多 275 个，超过评测上限 256。只用 Qwen3.5 的 tokenizer（本地 CPU，不加载权重），对 `runs/qwen3vl_main_val/val_finetuned.json` 里 100 条 val 的 `target` 分词，统计有几条答案本身就超过 256 个 token。这些样本即使模型完美，也会被判截断。对 val 结论有影响吗？（提示：Qwen3-VL 的分词可能和 Qwen3.5 略有不同，结论只取量级。）

---

## 四、调试题

**1.** 小王写了自己的 collator：

```python
labels = batch['input_ids'].clone()
labels[labels == tokenizer.pad_token_id] = -100
labels[:, :prompt_len] = -100
```

他用的模型里 pad 和 EOS 是同一个 id，batch 内用右 padding，`prompt_len` 取的是 batch 里第一条样本的提示长度。训练 loss 正常下降。请指出至少三个问题，并分别说明它们在评测时会有什么现象。

**2.** 小李在 Qwen3.5-0.8B 上训练，日志显示 `trainable params: 638,976`。训练 300 步后，val 指标几乎没变。他的 `LoraConfig` 最可能怎么写的？怎样一眼从数字上判断出来？

**3.** 小张用 connector 策略训练，保存前在 val 上 CER 0.18。重新加载适配器后再评，CER 变成 0.21，接近只训 LoRA 的水平。最可能是哪一步漏了？用 tour 部分 3 的哪个检查能在训练前就发现？

**4.** 小赵做视觉依赖对照：错图条件直接用别人的原图，不 resize。结果"改变 100 条，CER 1.3"。他据此说"模型非常依赖图片"。结论可能没错，但证据有什么问题？还缺哪一项证据？

**5.** 某次训练后，val 的 CER 是 2.6，截断 70 条，训练 loss 0.24。按正文 33.8.5 节的流程，写出你排查的前五步，每一步要看什么字段、预期看到什么。

**6.** 小陈估算 Qwen3-VL-2B 的 language 策略训练要 4.2 GiB 静态显存，实测"显存占用"却是 6.5 GiB。他看的是 nvidia-smi，像素预算设成了 min 65536、max 262144。说出两个可能的原因，以及怎样分别验证。

---

## 五、闭卷手写

不看书，在纸上完成：

1. 写出 `answer_labels` 的核心实现（不超过 8 行），要求支持左右 padding、遮图片 token、拒绝没有监督 token 的样本。
2. 写出 LoRA 可训练参数的求和公式，以及 language / connector / vision 三种策略下 N_train 的表达式；说明 `modules_to_save` 为什么让总参数多出一份。
3. 写出 VLM LoRA 训练的静态显存估算式（bf16 基座、fp32 LoRA），并列出另外两项不能忽略的显存。
4. 写出 Wilson 95% 区间的公式，以及精确 McNemar 检验的 p 值公式。
5. 列出视觉依赖对照的设计要点（至少四条）和它不能证明的事。
6. 列出发现"停不下来"的五个信号。

---

## 答案与提示

### 一、概念题

**1.** 这是一道保险。正常数据上，图片 token 都在提示里，第二步是多余的，去掉也不出错。它防两种情况：一是数据里有人把图片放进了答案区（测试 `test_answer_mask_padding_and_image_inside_answer` 就构造了这种样本）；二是 prefix 长度算短了，图片 token 落进了"答案区"。没有这一步，模型会被要求预测占位符。

**2.** 累加和 c_t 只数真实 token。不管 padding 在左还是右，第 P 个真实 token 之后才是答案，条件 c_t > P 在两种情况下含义相同；padding 位置再用 a_t = 0 单独遮掉。

**3.** 非零的是 186 个 B 的梯度。B 初始化为 0，而 ∂L/∂A 含有因子 B（第 20 章 20.3.2），所以第一步所有 A 的梯度严格为 0。如果断言"全部非零"，一个正确的 LoRA 也会被判失败；所以代码只要求"梯度存在、有限、至少一个非零"，同时要求冻结参数没有梯度。

**4.** 多出来的是原始 merger。PEFT 的 `modules_to_save` 保留原件、训练一份副本。`disable_adapter()` 时前向走原件，所以基座（包括原始连接器）完全恢复；tour 部分 3 的 `disabled_adapter_max_abs_error = 0.0` 证明了这一点。

**5.** 训练时每一步的输入都是**真实**的前缀（teacher forcing），loss 衡量的是"给定正确前文，猜对下一个 token"。答案后面的 `<|im_end|>` 和换行都被监督，模型学得很好，loss 自然低。但"换行之后做什么"从来没被监督，而自由生成时模型要从自己写出的前缀继续写，停不停取决于生成配置的停止 token。这件事 loss 完全看不到。

**6.** resize 保证视觉 token 数不变，从而序列长度、位置编码、注意力的范围都和正确图片条件一样。不 resize，差异里就混进了"输入长度不同"这个因素，你无法区分是图片内容变了还是序列结构变了。

**7.** "依赖"是整体层面的：换图后输出变了。"每个符号来自图片"是逐 token 的归因：可能前半段读图、后半段靠先验补全。一个检查办法：对正确图片下的输出，逐 token 比较"正确图片"和"白图"条件下该 token 的对数概率（同一前缀，teacher forcing），差值大的 token 依赖图片，差值接近 0 的 token 靠先验。也可以对局部遮挡图片区域做类似比较（第 31 章 saliency 的思路）。

**8.** 防止看了 test 的错例后调整设置（用 test 调参）。代价是 test 上不能做配对比较（配对 bootstrap、McNemar），只能比较各自的区间，检出差异的能力更弱。这是本次产物的限制；第 0 章 §0.10.1 允许封存逐条结果，最终报告定稿后再解封做错例分析。

**9.** 说法不准确。parse_rate 是"能被 mathtext 子集解析"的比例，不等于"合法 LaTeX"。val 目标本身只有 53% 可解析，说明数据里大量使用 mathtext 不支持的写法（例如 `&` 对齐）。微调后输出更像目标，parse_rate 向 0.53 靠拢，这是"学到了数据的写法"的表现之一，不是变差的证据。要判断合法性，应该用完整的 LaTeX 编译器检查。

### 二、手算题

**1.** 第 0 行：累加和 `[0, 0, 1, 2, 3, 4, 5]`，遮 padding（位置 0、1），遮 c ≤ 3（位置 0–4），图片 99 在位置 3 已被遮。labels = `[-100, -100, -100, -100, -100, 30, 0]`。注意最后的 0 是真实的 EOS（attention = 1），要监督。

第 1 行：累加和 `[1, 2, 3, 4, 5, 6, 6]`，遮 c ≤ 4（位置 0–3），遮 padding（位置 6）。labels = `[-100, -100, -100, -100, 31, 32, -100]`。

错开一位后，labels[:, 1:] 为 `[-100, -100, -100, -100, 30, 0]` 和 `[-100, -100, -100, 31, 32, -100]`。有梯度的 logits 位置：第 0 行 4、5；第 1 行 3、4。

**2.** (a) LoRA 参数和 r 成正比：10,822,656 / 2 = 5,411,328。

(b) 只有 6 层普通注意力有 `q_proj`、`v_proj`：12 个模块，(81,920 + 24,576) × 6 = 638,976，约占 5.9%。

(c) 246,272 × 18 = 4,432,896。

**3.** h₀ = round(40/32)·32 = round(1.25)·32 = 32；w₀ = round(400/32)·32 = round(12.5)·32 = 12·32 = 384（五成双，12.5 → 12）。面积 32 × 384 = 12288 < 16384，走放大分支。s = √(16384 / (40·400)) = √1.024 ≈ 1.01193。H′ = ⌈40 × 1.01193 / 32⌉·32 = ⌈1.265⌉·32 = 64；W′ = ⌈400 × 1.01193 / 32⌉·32 = ⌈12.649⌉·32 = 13·32 = 416。面积 64 × 416 = 26624，视觉 token 26624 / 1024 = 26，patch 数 26 × 4 = 104。注意面积 26624 比下限 16384 大不少，这就是"向上取整可能溢出"（第 32 章 32.2.3）。

**4.**

- 基座：852,985,920 × 2 = 1,705,971,840 字节。
- LoRA：(10,822,656 + 2,359,296) × 16 = 13,181,952 × 16 = 210,911,232 字节。
- merger 副本：12,588,544 × 8 = 100,708,352 字节。
- 合计 2,017,591,424 字节 ≈ 1.879 GiB。

（`parameters.json` 的 `trainable_dtypes` 显示 vision 策略下 fp32 13,181,952 个、bf16 12,588,544 个，与题设一致。）

**5.** z²/n = 0.038416，分母 1.038416。中心 (0.1 + 0.019208) / 1.038416 ≈ 0.1148。根号内 0.0009 + 0.00009604 = 0.00099604，开方 ≈ 0.03156。半宽 1.96 × 0.03156 / 1.038416 ≈ 0.0596。区间 ≈ [0.0552, 0.1744]，与 JSON 中的 [0.05523, 0.17437] 一致。

**6.** (a) 2 × (1/2)⁵ = 0.0625。

(b) b + c = 9，p = 2 × [C(9,8) + C(9,9)] / 2⁹ = 2 × 10 / 512 ≈ 0.039。

(c) 只有 5 个不一致对时，bootstrap 重抽样得到的差值分布非常离散：原始差值是 5 个 1 和 95 个 0，重抽样差值只能是 K/100，K 服从 Binomial(100, 0.05)，绝大部分概率挤在 0 到 0.10 这十来个格点上（K > 10 的概率约 1.1%，数学结论），百分位区间的覆盖率不可靠。精确检验直接用二项分布，在这种小计数下更诚实。

**7.** 3000 × 16 = 48,000 条；48,000 / 20,000 = 2.4 个 epoch；约是主实验 1.17 的 2 倍。注意补充实验同时改了数据量、步数、rank、学习率和 batch，不能把差异归因于其中一个（lab"补充实验"一节）。

**8.** n ≈ 7.84 × (0.09 + 0.1275) / 0.05² = 7.84 × 0.2175 / 0.0025 ≈ 682。500 条不够（按独立两组的近似）；同一批 val 上的配对比较需要的样本更少，能否检出取决于不一致对的多少。

### 三、编程题

**33.1 提示。**累加和用 `attention_mask.long().cumsum(-1)`；prefix 长度要 `[:, None]` 广播；"没有监督 token"的检查用 `(labels != -100).sum(-1).min() == 0`。别忘了 `test_padding_id_can_also_be_supervised_eos`：padding 判断只看 attention_mask。

**33.2 提示。**先收集连接器名字，再遍历 `named_modules()`；vision 策略下用 `name.startswith(c + '.')` 排除连接器内部的 Linear。不要用 `'q_proj' in name` 这类名字列表。

**33.3 提示。**`(args.data / 'ch33_final_test.lock').open('x')`。放在输出目录下，换一个 `--out` 就能绕过；放在数据目录下，本章的 test 流程只能启动一轮，其中预先登记的基线和微调各评一次；选型先冻结，不据此改设置（第 0 章 §0.10.1、正文 §33.7.1）。

**33.4 提示。**每个 val JSON 的 `observations` 里有 `id`、`target`、`raw`、`pred`、`generated_tokens`、`truncated`。用 `PostCorrectExample(id, 'val', target, target)` 构造评分样本。错图条件的"下一条"是 `(i + 1) % n`。

**33.5 提示。**Qwen3.5 的 `layer_types` 告诉你哪些层是 `linear_attention`；DeltaNet 的 key 维 = `linear_num_key_heads × linear_key_head_dim`，value 维同理；普通注意力的 `q_proj` 因为 `attn_output_gate` 要乘 2。

**33.6 提示。**本项目的验证结果见正文 33.8.4 节末尾：加上 248046 以后，截断从 75 降到 1，CER 从 2.724 降到 0.174，EM 从 1/100 升到 5/100（本项目观察）。你的复现应当方向一致。如果 `eos_token_id` 只有 248044，而解码后公式后面紧跟 `<|im_end|>\n<|im_start|>assistant`，假说基本成立。修正方案各有代价：只改生成配置，训练出的习惯仍在，换一个推理框架又可能出问题；只改 label，模型可能学不会在 `<|im_end|>` 后接什么。第 34 章的 GRPO 依赖模型自由生成，停止问题会直接影响奖励和长度。

**33.7 提示。**`AutoTokenizer.from_pretrained(本地快照, local_files_only=True)`，只分词 `target`。答案 token 数加上 `<|im_end|>` 才是需要生成的长度。

### 四、调试题

**1.** ① 按值遮 pad，EOS 也被遮掉，模型学不会停：评测时大量写满上限。② 用第一条样本的 `prompt_len` 遮整个 batch：其他样本的提示长度不同，有的答案开头被遮掉，有的模板尾巴被监督；现象是输出开头缺符号或多出模板文字。③ 没有遮图片 token（如果提示长度算错，图片 token 会进入监督）。④ 没有验证提示是完整样本的精确前缀。

**2.** 只挂了 `q_proj` 和 `v_proj`（r = 16）。638,976 正好是 (81,920 + 24,576) × 6：只有 6 层普通注意力被挂上，18 层 DeltaNet 和全部 MLP 都漏了。

**3.** merger 没有放进 `modules_to_save`（或保存时没包含连接器），重新加载后连接器回到原样。tour 部分 3 会保存、从新基座重新加载，比较前后 logits；漏存连接器时 `reload_max_abs_error` 会明显大于 0，断言失败。

**4.** 不 resize 时视觉 token 数随图片改变，差异混进了序列长度的变化。还缺：错图条件下的预测是否与"那张图的目标"一致（正文中 7/100 一致），以及白图对照。

**5.** ① 截断数和零样本比：70 远高于正常水平。② 平均 `generated_tokens`：是否接近上限。③ CER > 1：插入错误主导，说明多写了大量字符。④ 打开几条截断样本的 `raw`：循环的是公式符号（解码退化）还是 `assistant` 一类模板文字（停止问题）。⑤ 用 `skip_special_tokens=False` 重新解码，打印 `generation_config.eos_token_id`，看结束符是否在停止集合里。

**6.** ① nvidia-smi 显示的是进程占用的总量（含缓存池和 CUDA 上下文），比 `max_memory_allocated` 大；用 `torch.cuda.max_memory_allocated()` 重新测。② 像素预算提高 4 倍，视觉 token 数大致从 16–64 变成 64–256（取整可能让个别极扁的图超出），序列变长，激活和 logits 跟着涨；用 `audit_samples` 统计新预算下的序列长度，重算激活和 logits 的估计。

### 五、闭卷手写

1. 见正文 33.2.5 节的代码。
2. N_LoRA = Σ r(in_m + out_m)；N_train = N_LoRA^lang + [connector]·N_merger + [vision]·N_LoRA^vis；`modules_to_save` 保留原件并训练副本，所以总数多出一份 N_merger。
3. M_static ≈ 2·N_base + 16·N_LoRA；另外两项是激活（受序列长度和 gradient checkpointing 影响）和 logits（B × T × V）。
4. 见正文 33.10.3 节。
5. 同一 prompt、像素预算和解码设置；错图 resize、白图保留原尺寸，保证视觉 token 数不变；同时报告相对参考答案的指标和相对正确图片的变化数；检查错图条件下是否读出了那张图的公式；只在 val 上做。它不能证明每个符号都来自图片。
6. 截断数高；平均生成长度接近上限；CER 超过 1；原始输出末尾有重复的符号或模板文字；评测时间明显变长。
