# A4：评测与监督微调——指标、置信区间、打包的 SFT、LoRA 和内存

| | |
|---|---|
| 位置 | 第 28 章之后 |
| 预计用时 | 25–35 小时 |
| 算力 | 写代码、跑测试：CPU。实验：CPU 可以做小配置，GPU 快得多。排行榜：一张租用的 24 GB 显卡，SFT 训练时间不超过 20 分钟（CPU 赛道 90 分钟） |
| 交付 | ① `a4/` 里的代码，`tests/adapters.py` 接好后全部测试通过；② 书面报告 `writeup.md`（从 `writeup_template.md` 复制）；③ 排行榜记录 |
| 满分 | 100 分：代码 58 分，书面 8 分，实验 28 分，排行榜 6 分 |

A3 训练出了一个会看图答题的小 VLM。这个作业回答两个问题：**它到底有多好**（第 25、27 章），以及**怎样把它微调得更好、更省**（第 26、28 章）。

1. **指标**（第 25 章）：带回溯的编辑距离、CER/WER/NED、BLEU、ANLS，表格的 TEDS（Zhang–Shasha 树编辑距离）；
2. **一个数字有多可信**（第 25、27 章）：bootstrap 置信区间、配对 bootstrap、McNemar 检验、校准（ECE）、AUROC；
3. **回答的对数概率**（第 27 章）：错位、两种 mask 约定、熵；
4. **SFT**（第 26 章）：每个 token 的身份、提示权重、两种平均、补齐与打包、打包的注意力 mask；
5. **LoRA 与内存**（第 28 章）：LoRA 层、装进模型与合并、静态内存估算、数出保存下来的激活值、梯度检查点。

最后你会在 A3 的模型上加一个新任务——**读文字行**——用 SFT 和 LoRA 训练它，并且每个结论都带着置信区间。

---

## 1. 规则

### 1.1 能用什么、不能用什么

A1–A3 的规则照样适用（`torch.nn.functional`、现成的层和损失、`torchvision` 都不能用）。这个作业另外：

| 不能用 | 自己写 |
|---|---|
| `scipy.stats`、`sklearn.metrics`（`roc_auc_score`、`calibration_curve`）、`statsmodels` | bootstrap、McNemar、ECE、AUROC |
| `nltk`、`sacrebleu`、`evaluate`、`jiwer`、`editdistance`、`Levenshtein` | 编辑距离、BLEU、ANLS |
| `apted`、`zss` 等树编辑距离的库 | Zhang–Shasha |
| `peft` | LoRA |

**可以用**：Python 标准库（`html.parser`、`math.comb`、`functools.lru_cache`）、numpy、`torch.autograd.graph.saved_tensors_hooks`、`torch.utils.checkpoint`（第 28 章：梯度检查点的"重算"机制本身不是这个作业的重点）。

**可以把你自己 A1–A3 的代码复制进 `a4/`**，不能 import 前面的作业。

`miniocr` 里能用的仍然只有**数据生成器**：A3 用过的那些，加上 `miniocr.data.text`（`sample_line_text`）、`miniocr.data.render`（`render_line`）、`miniocr.data.table`（`sample_table`、`to_html`）。

### 1.2 AI 助手和参考实现

与 A1 相同。

### 1.3 前提

你需要 A3 的检查点（至少一个 align 阶段的 VLM）。如果 A3 没有做完，也可以从随机初始化开始，但实验的结论会弱很多。

---

## 2. 准备

```
assignments/a4_eval_sft/
├── handout.md
├── a4/                   你的代码
├── tests/
│   ├── adapters.py       ← 你要填的接线
│   └── test_a4_*.py
├── leaderboard.py        排行榜（读文字行）
├── data/lines_test.npz   排行榜测试集：500 行
└── writeup_template.md
```

```bash
python -m pytest -q assignments/a4_eval_sft/tests
python -m assignments.a4_eval_sft.leaderboard info
```

第一到三部分互不依赖，也不依赖 A3，可以先做；第四、五部分要用你 A3 的 GPT。

---

## 3. 第一部分：指标

### 3.1 问题（edit_align）：编辑距离与回溯（3 分）

1. `edit_distance(a, b)`：Levenshtein 距离，a、b 可以是字符串或列表。只保留两行。
2. `align_ops(pred, ref)`：回溯出一组最少的操作 `[(op, pred 的字符, ref 的字符)]`，op ∈ `"="`（相同）、`"S"`（替换）、`"D"`（删掉 pred 的字符）、`"I"`（插入 ref 的字符）。要建完整的表。**回溯从右下角开始，优先级：对角线（相同或替换）→ 删除 → 插入**。最少操作通常不止一组，这个优先级让答案唯一，测试按它对照。

有了 `align_ops`，就能统计"哪个字符最常被认成哪个"——实验要用。

### 3.2 问题（cer_wer_ned）：CER、WER、NED（2 分）

- CER = Σ 编辑距离 / Σ 参考长度（语料级，**不是**逐行错误率的平均）；
- WER：按空白切成词再算；
- NED = 编辑距离 / max(两者长度)，两个都空时为 0。

### 3.3 问题（bleu）：BLEU（3 分）

语料级 BLEU，preds、refs 是 token 列表的列表，每条一个参考：

1. 对 n = 1 … max_n：截断精确率 pₙ = Σ min(预测里的次数, 参考里的次数) / Σ 预测里的 n 元组数，**分子分母分别对整个语料累加**；
2. 简短惩罚 BP = 1（预测总长 c ≥ 参考总长 r）或 exp(1 − r/c)；
3. BLEU = BP · exp(平均 log pₙ)。任何一个 pₙ 为 0（或 c = 0）时返回 0。

### 3.4 问题（anls）：ANLS（2 分）

每题可以有多个正确答案。都转小写、去首尾空白；对每个正确答案 s = 1 − NED，NED ≥ τ 时 s = 0；取最大的 s；对题目求平均。

### 3.5 问题（teds）：表格的 TEDS（5 分）

1. **解析**：用 `html.parser` 把 HTML 表格解析成树：table → tr → td。`th` 当作 td；`thead`、`tbody` 展开（它们的子节点直接挂到上一层）；td 记下 `colspan`、`rowspan`（默认 1）和去掉首尾空白的文字；没闭合的标签要容忍（遇到结束标签时一直弹到对应的那一层）；只取第一个 `<table>`，一个都没有时返回空的 table 节点。
2. **树编辑距离**（Zhang–Shasha，第 25 章）：后序编号、最左叶子、关键根；对每一对关键根算森林距离。插入、删除代价 1，改标签的代价由 `rename(x, y)` 给出。**测试会和一个直接按定义递归的版本对照**（随机的小树、两种 rename）。
3. **TEDS** = 1 − 距离 / max(两棵树的节点数)。rename：标签不同 1；都是 td 时 colspan/rowspan 不同记 1，否则记文字的 NED（只比结构时记 0）；其余相同标签记 0。

**交付**：`run_edit_distance`、`run_align_ops`、`run_cer`、`run_wer`、`run_ned`、`run_bleu`、`run_anls`、`run_parse_table`、`run_tree_edit_distance`、`run_teds`；`test_a4_metrics.py` 全部通过。

### 问题（teds_by_hand）：手算 TEDS（3 分，书面）

参考表格是 2×2：第一行 a、b，第二行 c、d。对下面四个预测，写出最少的编辑操作、距离、TEDS 和 TEDS-S（只比结构）：

1. 第一行合并成一个 `colspan="2"` 的格子，内容 "ab"；
2. d 写成了 "dd"；
3. 整个第二行丢了；
4. a 和 b 的内容对调了。

哪一个说明了"只看结构"的 TEDS-S 不够？哪一个说明了 TEDS 对"位置对、内容错"和"结构错"的惩罚差不多？

---

## 4. 第二部分：一个数字有多可信

### 4.1 问题（bootstrap）：bootstrap 置信区间（3 分）

`bootstrap_ci(num, den=None, n_boot=2000, alpha=0.05, seed=0)`：比值型指标 Σnum / Σden（den 为 None 时每个样本的分母是 1，即平均值）。**抽样方式规定死**：`idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))` 一次抽完，每行算一次 Σnum[idx]/Σden[idx]，下界、上界是 `np.quantile(..., [alpha/2, 1 − alpha/2])`（默认的线性插值）。返回 (点估计, 下界, 上界)。

为什么 CER 要按"行"重抽、而不是按"字符"重抽？

### 4.2 问题（paired_bootstrap）：配对 bootstrap（2 分）

两个系统在**同一批**样本上：每次用同一组下标，算差 (Σa − Σb) / Σden。返回 (差, 下界, 上界, 差 ≤ 0 的比例)。抽样方式同上。测试里有一个例子：两个系统只差 15 题，各自的置信区间重叠，配对的区间却完全在 0 的一侧。

### 4.3 问题（mcnemar）：McNemar 精确检验（3 分）

b10 = a 对 b 错的个数，b01 = a 错 b 对。原假设下 a 赢的次数 ~ Binomial(b10 + b01, ½)。p = min(1, 2·P(X ≤ min(b10, b01)))，用 `math.comb` 精确算。b10 + b01 = 0 时 p = 1。返回 (b10, b01, p)。

### 4.4 问题（calibration）：分段与 ECE（2 分）

`reliability(conf, correct, n_bins)`：等宽分成 [0, 1/n)、…、[(n−1)/n, 1]（**1.0 落在最后一段**）；返回有样本的段 [(下界, 上界, 平均置信度, 准确率, 样本数)]。ECE = Σ（段内样本数 / 总数）· |准确率 − 平均置信度|。

### 4.5 问题（auroc）：AUROC（2 分）

用排名算（Mann–Whitney）：所有分数排序，并列的取平均名次（从 1 开始），AUROC = (R₊ − n₊(n₊+1)/2) / (n₊·n₋)。只有一类时返回 nan。测试有大量并列的分数，和 O(n²) 的逐对比较对照。

**交付**：`run_bootstrap_ci`、`run_paired_bootstrap`、`run_mcnemar`、`run_reliability`、`run_ece`、`run_auroc`；`test_a4_stats.py` 全部通过。

### 问题（mcnemar_by_hand）：手算（2 分，书面）

100 道题，两个模型：都对 50、都错 35、A 对 B 错 3、A 错 B 对 12。

1. 两个模型的准确率各是多少？差多少？
2. 手算 McNemar 的 p 值（写出式子）。在 0.05 的水平上能说 B 更好吗？
3. 为什么"都对"和"都错"的 85 道题不影响检验？如果只给你两个准确率，不给逐题的对错，你能做这个检验吗？

---

## 5. 第三部分：回答的对数概率

### 5.1 问题（token_logprobs）：每个 token 的对数概率（2 分）

`token_logprobs(logits (B, T, V), ids (B, T)) → (B, T−1)`：`out[b, t] = log softmax(logits[b, t])[ids[b, t+1]]`。**用 float32 算**（输入是 bf16 时先转）。

### 5.2 问题（masks）：两种约定（2 分）

- 约定一（A3 的 targets）：`targets (B, T)`，−100 表示不算；
- 约定二：`mask (B, T−1)`，第 t 列对应"预测第 t+1 个 token"。

写 `labels_to_mask`、`mask_to_labels(ids, mask)`、`mask_from_lengths(prompt_lens, total_lens, T)`（目标位置 j = t+1 满足 prompt_len ≤ j < total_len 时为 1）。

### 5.3 问题（entropy_seq）：熵与求和（2 分）

`token_entropy(logits) → (B, T−1)`（只取前 T−1 个位置，与上面对齐；完全确定时不能出 nan）；`seq_logprob(lp, mask, reduce)`：`"sum"` 或 `"mean"`（除以 mask 之和，至少 1）。

**交付**：`test_a4_logprobs.py` 全部通过。

---

## 6. 第四部分：SFT

### 6.1 问题（chat_roles）：每个 token 的身份（2 分）

拼法与 A3 的 `chat_template` 相同，但同时返回每个 token 的身份：

| 身份 | 编码 | 哪些 token |
|---|---|---|
| 格式 | 0 | `<bos>`、`"User: "`、用户消息末尾的 `"\n"`、`"Assistant: "` |
| 图像 | 1 | 图像占位符 |
| 用户 | 2 | 用户消息的文字（不含角色前缀和末尾的换行） |
| 回答 | 3 | 助手的文字和它后面的 eos |

`n_img` 可以是整数，也可以是列表（第 k 张图的占位符个数）——这次一批里会有 64×64 的场景图和 32×256 的文字行，视觉 token 数不同。

### 6.2 问题（sft_targets）：标签与权重（2 分）

位置 t 预测 ids[t+1]，权重由 **ids[t+1] 的身份**决定：回答 1，用户 `prompt_weight`，格式 `prompt_weight`，图像**永远 0**。`train_eos=False` 时回答末尾的 eos 权重为 0。权重 0 的位置标签是 −100。最后一位补 (−100, 0)。

### 6.3 问题（sft_loss）：两种平均（3 分）

`sft_loss(logits, targets, weights, mode)`，ce 用你自己的 log_softmax：

- `"token"`：Σ w·ce / Σ w；
- `"sample"`：每条样本 Σₜ w·ce / Σₜ w，再对**有权重的**样本求平均；
- weights 为 None 时是 1；targets 为 −100 的位置权重当作 0；权重全为 0 时返回 0，而且能反向传播（不能出 nan）。

### 6.4 问题（pack）：打包（3 分）

`pack(examples, max_len, pad_id)`：examples 是 [(ids, targets, weights)]。**顺序装箱**：当前箱子装得下就放进去，装不下就开新箱子（不回头找空位）。单条超过 max_len 时抛出 `ValueError`。返回 dict：`ids`、`targets`、`weights`、`seg`（箱内第几条，从 1 开始，pad 为 0）、`pos`（条内位置，pad 为 0），都是 (箱子数, max_len)；`order`：每个箱子装了哪几条（图像要按这个顺序填进占位符）。

### 6.5 问题（packed_mask）：打包的 mask（2 分）

`packed_mask(seg) → (B, 1, T, T)` bool：i 能看 j 当且仅当 j ≤ i 且 seg[i] = seg[j] ≠ 0；pad 位置只看自己（否则整行都是 False，softmax 得到 nan）。

### 6.6 问题（packed_forward）：打包后的前向（3 分）

改你的 GPT：前向接受 `pos (B, T)`（每个 token 自己的位置编号，RoPE 用它查表）和 `mask (B, 1, T, T)`（代替因果 mask）。测试检查两件事：**打包后每条样本的 logits 和单独送进去的完全相同**；打包和补齐算出的 token 平均损失相同。

**交付**：`run_chat_roles`、`run_sft_targets`、`run_sft_loss`、`run_pack`、`run_packed_mask`、`get_lm`、`run_lm_logits`；`test_a4_sft.py` 全部通过。

---

## 7. 第五部分：LoRA 与内存

### 7.1 问题（lora_linear）：LoRA 层（2 分）

包住你自己的 Linear（weight (out, in)）：`y = base(x) + (α/r)·(x Aᵀ) Bᵀ`。A (r, in) ~ U(−1/√in, 1/√in)，B (out, r) = 0（刚装上时输出不变）。base 的参数冻结，只有 A、B 可训练。先乘 A 再乘 B，不要算出 B·A。

### 7.2 问题（apply_merge）：装进模型、合并、计数（3 分）

- `apply_lora(model, targets, r, alpha)`：先冻结全部参数；把**子模块自己的名字**在 targets 里（如 `"qkv"`）的 Linear 换成 LoRA 层；已经是 LoRA 的不再包；返回被替换的完整名字（如 `"blocks.0.attn.qkv"`）。遍历时先复制一份子模块列表，不要边遍历边改。
- `merge_lora(model)`：把每个 LoRA 层换回普通的 Linear，权重 = W + (α/r)·B·A；返回合并的层数。合并后 state_dict 的键和原模型相同。
- `count_params(model) → (总数, 可训练)`：共享的权重只算一次。

### 7.3 问题（static_memory）：静态内存（2 分）

`static_memory(n_params, n_trainable, param_dtype, optimizer, master_fp32)` 返回字节数：params = 全部参数 × 类型字节；grads = 可训练 × 类型字节；optim：AdamW 每个可训练参数 8 字节（fp32 的 m、v），SGD 0；master：param_dtype 是 bf16/fp16 且 master_fp32 时，每个可训练参数 4 字节；total。

### 7.4 问题（saved_tensors）：数出激活值（2 分）

一个上下文管理器：`with SavedTensors(model) as s: 前向` 之后 `s.bytes` 是为反向保存下来的张量的总字节数。用 `torch.autograd.graph.saved_tensors_hooks`：pack 钩子拿到每个被保存的张量，用 `t.untyped_storage()` 的 `data_ptr()` 识别存储、`nbytes()` 得到大小；**同一块存储只算一次**（视图），**模型的参数和缓冲区不算**。

### 7.5 问题（checkpoint）：梯度检查点（1 分）

`checkpoint_blocks(blocks)`：原地把 ModuleList 里的每个块包一层，训练且需要梯度时用 `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`；已经包过的不再包；返回包了几个。测试检查梯度和不包时相同、保存的字节数少于一半。存盘前记得拆掉这层包装（否则 state_dict 的键多了一级）。

**交付**：`get_lora_linear`、`run_apply_lora`、`run_merge_lora`、`run_count_params`、`run_static_memory`、`get_saved_tensors`、`run_checkpoint_blocks`；`test_a4_lora_memory.py` 全部通过。

### 问题（memory_accounting）：算账（3 分，书面）

一个 Llama-2-7B 形状的模型：d = 4096，32 层，SwiGLU 中间维 11008，词表 32000，输出层不共享，没有偏置。

1. 总参数量是多少（逐项写）？
2. 全参微调、AdamW：fp32 训练和"bf16 参数 + fp32 主权重"的混合精度训练，静态内存各是多少 GB？为什么混合精度没有省下静态内存？它省的是什么？
3. 在 q、k、v、o 四个投影上装 r = 16 的 LoRA（基座 bf16，LoRA 参数有 fp32 主权重）：可训练参数多少？静态内存多少 GB？一张 24 GB 的卡还剩多少给激活值？
4. 用你的 `memory` 入口，报告你自己的 VLM 在全参、LoRA r=8 下的静态内存和一个批的激活值（带和不带梯度检查点）。

---

## 8. 训练与评测脚本

写下面几个入口（可以共用代码），要求与 A1 第 7 节相同（可复现、`log.jsonl`、`config.json`、检查点、`--smoke`、`--set`、默认 `device=cpu`）。不单独给分，但实验都要用它们。

**数据**：A3 的九个 ShapesWorld 任务，加一个新任务 **read**：`sample_line_text` + `render_line(text, rng, noise=0.1, distort=0.5)` 得到 32×256 的灰度行图（复制成 3 通道），问题 `"<image>Read the text."`，答案是那行文字。ViT 能处理不同分辨率（A3 的 `patch_tokens`），所以文字行直接送进去：patch 8 时是 4×32 = 128 个 patch，2×2 合并后 32 个视觉 token。

1. `sft`：从 A3 的检查点（或随机初始化）开始。配置至少包括：任务和 read 的权重、`prompt_weight`、`loss_mode`（token/sample）、`train_eos`、`pack` 和 `max_len`、`lora_r`（0 = 全参）和 `lora_targets`、是否训练视觉塔和连接器、梯度检查点。一个批里混合不同任务（图的大小不同：同样大小的图拼在一起过视觉塔，再按顺序填进占位符）。LoRA 训练完先合并再存盘。定期在你自己的验证集上评测。
2. `eval`：每个任务 n 题，贪心生成，**逐样本**写进 `.jsonl`：任务、预测、答案、是否正确、编辑距离、参考长度、回答（含 eos）的对数概率之和、token 数。打印每个任务的准确率和 95% 区间，read 的 CER 和区间，以及用"每 token 平均对数概率"当置信度时的 ECE 和 AUROC。
3. `compare a.jsonl b.jsonl`：同一批样本上两个模型，每个任务和全部的配对 bootstrap 差、区间、McNemar。
4. `memory`：打印静态内存的估算，实测一个批的激活值（带和不带梯度检查点）。
5. `predict`：读排行榜的 500 行，每行一个 JSON 字符串写进 `.jsonl`。

---

## 9. 实验

要求与 A1 第 8 节相同。这次多一条：**所有比较都要附置信区间或检验**——"A 比 B 高 2 个点"后面必须跟着区间或 p 值。

### 问题（exp_protocol）：评测协议（4 分）

用你 A3 最好的模型（还没学过 read）：

**交付**：(a) 按第 25 章 25.10 节的模板写一份评测协议（测试集怎么生成、多少题、指标、怎样报告不确定性）；(b) 每个任务各用 50、200、1000 题评测，区间宽度怎样随题数变化？和 1/√n 比较；(c) 对 A3 的两个检查点（例如 align 之后和 instruct 之后，或者两个种子）做 `compare`：哪些任务的差在区间之外？McNemar 的结论和配对 bootstrap 一致吗？

### 问题（exp_prompt_weight）：提示算不算损失（5 分）

从同一个 A3 检查点出发，提示权重 0、0.1、1，各 3 个种子，其余相同。

**交付**：(a) 每个任务的准确率（3 个种子的平均 ± 标准差）和 read 的 CER；(b) 把同一种子的模型两两 `compare`：哪些差别是真的？(c) 运行前先预测，然后解释：提示权重 1 时，模型多学了什么？梯度被分走了多少（数一数一个批里提示 token 和回答 token 的比例）？

### 问题（exp_sft_details）：平均、打包、eos（6 分）

**交付**：(a) token 平均 vs sample 平均：先算出两种方式下每个任务占梯度的份额（数一数每个任务平均有多少个回答 token），再训练、比较，结果的变化方向和份额一致吗？(b) 补齐 vs 打包：每步的毫秒数、每步实际计算的 token 数中有效 token 的比例、最终结果（应该在噪声范围内相同，用 `compare` 说明）；(c) `train_eos=false`：模型回答之后会写什么？打印 5 个例子，算一算以正确答案**开头**的比例和完全正确的比例。

### 问题（exp_lora）：LoRA 对全参（8 分）

同一个起点、同样的步数和数据，比较：① 全参（视觉塔冻结）；② 只训练连接器；③ LoRA r = 1、4、16、64（LM 的 qkv、o、gate、up、down），连接器全参；④ 选一个 r，只在 qkv、o 上装。

**交付**：(a) 一张表：可训练参数、静态内存（估算）、一个批的激活值（实测）、每步用时、每个任务的准确率和 read 的 CER（带区间）；(b) 每个 LoRA 配置和全参做 `compare`，哪些差别在区间之外？(c) 秩从 1 到 64，收益在哪里饱和？只训练连接器能学会 read 吗？为什么？(d) 合并前后在同一批验证题上的输出是否逐字相同？

### 问题（exp_calibration）：置信度（5 分）

用你最好的 SFT 模型和 `eval` 的逐样本结果：

**交付**：(a) 三种置信度：回答的对数概率之和、每 token 平均、回答里最小的 token 概率（需要你自己加一列）。各自的 reliability 图（10 段）、ECE，以及用它找错例的 AUROC；(b) read 任务上，按置信度从高到低只保留前 x% 的行，画 CER 随 x 的曲线（"选择性预测"）——只交给人看最没把握的 10% 的行，剩下的 CER 是多少？(c) 为什么"对数概率之和"对长答案不公平？在 read 和 count 两个任务上分别看。

---

## 10. 排行榜

### 问题（leaderboard）：读文字行（6 分）

目标：`data/lines_test.npz`（500 行，noise 0.1，distort 0.5）上的 CER 尽量低。

```bash
python -m assignments.a4_eval_sft.leaderboard score 你的预测.jsonl     # 给出 CER、95% 区间、整行全对的比例
```

**规则**

1. **起点**：你自己 A3 的检查点（不管它们花了多久），或随机初始化。不能用下载的权重。
2. **硬件和时间**：一张租用的 24 GB 显卡，A4 的 SFT 训练墙钟时间不超过 **20 分钟**；生成数据的时间不算。CPU 赛道：90 分钟，写明 CPU 型号。
3. **数据**：只能用 `miniocr.data` 的生成器，种子不能用 20261022–20261099；可以只训练 read，也可以混合其他任务；渲染参数（noise、distort、字号）可以自己调。**不能用测试集训练、调参或选检查点**。
4. **代码**：第 1.1 节的限制照样适用。结构、连接器、LoRA 或全参、图像怎样切块，都可以自由改。
5. **报告**：`leaderboard.md` 一行要写 CER **和 95% 区间**。如果你和别人（或你自己的上一次提交）的区间重叠，请在报告里说明这意味着什么。

**交付**：`leaderboard.md` 加一行；报告里一段：试过什么、什么有用，附学习曲线和 `align_ops` 统计出的最常见的 10 种替换（"哪个字符被认成了哪个"）。不按名次给分。

---

## 11. 交付清单与评分

| 部分 | 问题 | 分数 |
|---|---|---|
| 代码：指标 | edit_align 3、cer_wer_ned 2、bleu 3、anls 2、teds 5 | 15 |
| 代码：统计 | bootstrap 3、paired_bootstrap 2、mcnemar 3、calibration 2、auroc 2 | 12 |
| 代码：对数概率 | token_logprobs 2、masks 2、entropy_seq 2 | 6 |
| 代码：SFT | chat_roles 2、sft_targets 2、sft_loss 3、pack 3、packed_mask 2、packed_forward 3 | 15 |
| 代码：LoRA 与内存 | lora_linear 2、apply_merge 3、static_memory 2、saved_tensors 2、checkpoint 1 | 10 |
| 书面 | teds_by_hand 3、mcnemar_by_hand 2、memory_accounting 3 | 8 |
| 实验 | exp_protocol 4、exp_prompt_weight 5、exp_sft_details 6、exp_lora 8、exp_calibration 5 | 28 |
| 排行榜 | leaderboard 6 | 6 |
| **合计** | | **100** |

下一站是第 29 章（强化学习基础），A5 在第 32 章之后：同一个起点，比较 SFT、DPO 和 GRPO。
