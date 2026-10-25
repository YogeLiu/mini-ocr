# A2：序列——从零写分词器、CTC 识别器和 GPT

| | |
|---|---|
| 位置 | 第 18 章之后（里程碑 B） |
| 预计用时 | 25–35 小时 |
| 算力 | 写代码、跑测试：CPU。实验：CPU 可以做小配置，GPU 更快。两个排行榜：各用一张租用的 24 GB 显卡，训练时间不超过 15 分钟 |
| 交付 | ① `a2/` 里的代码，`tests/adapters.py` 接好后全部测试通过；② 书面报告 `writeup.md`（从 `writeup_template.md` 复制）；③ 两个排行榜记录 |
| 满分 | 100 分：代码 64 分，书面 9 分，实验 21 分，排行榜 6 分 |

这个作业有三部分，对应 OCR 模型输出端的两条路线，以及后面 VLM 的语言底座：

1. **分词**（第 14 章）：在混合语料（英文、LaTeX、Markdown 表格）上训练 BPE 分词器，写一个能处理特殊符号、能流式编码的分词器；
2. **CTC 文字行识别**（第 13 章）：对数空间、整批向量化的 CTC 损失，贪心解码和前缀束搜索，自己写的 LSTM，组成 CRNN；
3. **Transformer 语言模型**（第 15–18 章）：RMSNorm、RoPE、注意力、GPT、KV cache、采样、梯度累积、bits per char 评测。

和 A1 一样，你从一个空的包开始，自己决定代码结构，测试只规定行为。

---

## 1. 规则

### 1.1 能用什么、不能用什么

在 A1 的规则（`../a1_basics/handout.md` 第 1.1 节）基础上，`a2/` 里还**不能**用：

| 不能用 | 自己写 |
|---|---|
| `nn.Embedding`、`nn.LSTM`、`nn.RNN`、`nn.GRU`、`nn.LayerNorm`、`nn.RMSNorm`、`nn.MultiheadAttention` | 嵌入就是按下标取矩阵的行；LSTM 按第 13 章的公式逐步算 |
| `F.ctc_loss`、`F.scaled_dot_product_attention`、`F.silu`，以及 `torch.nn.functional` 里的一切 | |
| `torch.ctc_loss`、`torch.lstm`、`torch.embedding` 之类的函数版 | |
| 现成的分词器库（`tokenizers`、`tiktoken`、`sentencepiece`、transformers 的分词器） | |

**可以用**：`torch.sigmoid`、`torch.tanh`、`torch.logsumexp`、`torch.topk`、`torch.sort`、`torch.cumsum`、`torch.multinomial` 这些逐元素运算或通用运算，Python 标准库的 `re`、`heapq`、`collections`。

**可以把你自己 A1 的代码复制进 `a2/`**：Linear、卷积、BN、交叉熵、AdamW、调度、裁剪、检查点。不能 import `assignments.a1_basics`，要复制过来，这样两个作业互不影响。

`miniocr` 里能用的仍然只有**数据生成器**：`miniocr.data.text`（`sample_line_text`、`LINE_CHARSET`、`LineDataset`）、`miniocr.data.render`、`miniocr.data.corpus`（`mixed_corpus`）。

`tests/test_a2_rules.py` 用语法树检查 `a2/`，规则以本节为准。

### 1.2 AI 助手和参考实现

与 A1 相同：可以问概念、问报错、让 AI 检查你的代码，不要让它替你写 `a2/` 里的代码。`assignments/solutions/` 在提交前不要打开。

### 1.3 前提

在 `my-course/` 里做时，第 2 章的练习必须已经完成，因为 `LineDataset` 用到了练习 2.7。`mixed_corpus` 不依赖任何练习。

---

## 2. 准备

```
assignments/a2_sequences/
├── handout.md            本说明
├── a2/                   你的代码
├── tests/
│   ├── adapters.py       ← 你要填的接线
│   ├── fixtures/         测试用的快照（参考实现生成，已与书里的实现交叉核对）
│   └── test_a2_*.py
├── leaderboard.py        两个排行榜
├── data/                 排行榜测试集：ocr_test.npz（1000 行）、lm_test.txt（4.6 万字符）
└── writeup_template.md
```

```bash
python -m pytest -q assignments/a2_sequences/tests                 # 在 course-v3/ 下
python -m pytest -q assignments/a2_sequences/tests -k ctc
```

建议顺序：第一部分（分词，独立）→ 第二部分（CTC）→ 第三部分（GPT）。第二、三部分互不依赖，卡在一边时可以先做另一边。

---

## 3. 第一部分：分词

### 3.1 字符表与预切分

基础字符表固定为 96 个字符：换行符，加上 95 个可打印 ASCII 字符（空格到 `~`），**按码位排序**，即 `"\n" + "".join(chr(c) for c in range(32, 127))`。语料里出现字符表以外的字符时，训练和编码都要抛出 `ValueError`。

预切分用第 14 章的正则（按下面的顺序依次尝试）：

```python
PAT = r" ?[A-Za-z]+|[0-9]| ?\\[A-Za-z]+|\n| +(?=\S)| +| ?[^\sA-Za-z0-9]"
```

即：可选的一个空格加字母串、单个数字、可选空格加 LaTeX 命令、换行、后面跟着非空白的一串空格、其余空格、可选空格加单个其他符号。BPE 只在块内合并，不跨块。

### 问题（pretokenize）：预切分（2 分）

`re.findall(PAT, text)`。所有块首尾相接必须恰好等于原文。

**交付**：`run_pretokenize`；`test_pretokenize` 通过。

### 问题（train_bpe）：训练 BPE（8 分）

`train_bpe(text, vocab_size, special_tokens) → (vocab, merges)`：

1. **特殊符号**：先按特殊符号把文本切开，特殊符号本身不参与统计，也不会被拆开或参与合并。
2. **词表**：`vocab` 是 `{编号: 字符串}`，顺序是：特殊符号（按给定的顺序）→ 96 个基础字符 → 每条合并产生的新符号（按学到的顺序）。合并次数 = `vocab_size − len(special_tokens) − 96`。
3. **合并**：统计每个块出现的次数，块初始表示为字符序列。每一步统计所有相邻符号对的出现次数（乘以块的次数），合并**次数最多**的一对；**次数相同时取 (左, 右) 字典序最小的一对**（Python 的元组比较）。没有相邻对可以合并时提前停止。
4. **效率**：不要每一步都从头重新统计所有相邻对。合并一对以后，只有包含这一对的块会变。维护"相邻对 → 出现在哪些块里"的索引，只更新受影响的块；找最大值可以用带懒删除的堆。

`merges` 是 `[(左, 右), ...]`，按学到的先后排列。

> 次数相同时怎么选，看起来是小事，但它决定了结果能不能复现、能不能和测试对上。真实的分词器（GPT-2、Llama）各有自己的规则，换一个规则，学到的词表就不同。

**交付**：`run_train_bpe`；`test_bpe_toy_example`、`test_bpe_corpus_matches_snapshot`、`test_bpe_specials_are_not_merged` 通过。

### 问题（tokenizer）：编码与解码（6 分）

`Tokenizer(vocab, merges, special_tokens)`，三个方法：

- `encode(text) → list[int]`：先切出特殊符号（**重叠时优先匹配更长的**，例如同时有 `<|end|>` 和 `<|end|><|end|>`）；其余部分预切分，每个块从单个字符开始，反复合并**规则编号最小**的相邻对，直到没有相邻对在规则里。注意：要按训练时的顺序应用规则，不能"能合并就合并"。
- `decode(ids) → str`。
- `encode_iterable(iterable) → 迭代器`：逐段读入字符串（例如一个打开的文件，一次一行），产出与 `encode("".join(iterable))` **完全相同**的编号序列，但不把整个文本读进内存。难点是：从哪里切开才不会改变结果？（提示：看预切分的正则，哪个字符总是单独成块，而且不会出现在特殊符号里？）

**交付**：`get_tokenizer`；`test_encode_decode`、`test_overlapping_specials_prefer_longest`、`test_encode_iterable_matches_encode`、`test_unknown_char_raises` 通过。

---

## 4. 第二部分：CTC 文字行识别

### 4.1 约定

文字行来自 `miniocr.data.text.LineDataset(seed, n, noise=0.15, distort=1.0)`：图像 `(1, 32, 256)`，墨迹 = 1；标签是字符串。类别：第 0 类是**空白**，第 i 类（i ≥ 1）是 `LINE_CHARSET[i − 1]`，共 75 类。

**关于梯度的一个坑**：`F.ctc_loss` 的反向传播假定输入来自 `log_softmax`，它直接返回对 log_softmax **之前**的 logits 的梯度。所以单独比较"对 log_probs 的梯度"，你算的（正确的偏导数）和它给的会不一样；先做 log_softmax 再比较就一致了。测试就是这么比的。想一想为什么这两种梯度在经过 log_softmax 的反向之后会相同。

### 问题（ctc_loss）：CTC 损失（8 分）

`ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction)`，与 `F.ctc_loss(..., blank=0)` 相同：

- `log_probs (T, N, C)`；`targets (N, S)` 补齐（补齐的值任意）；`input_lengths`、`target_lengths (N,)`，每个样本的长度可以不同；
- 在**对数空间**里做前向算法（第 13 章），扩展标签长 2S+1；
- **整批向量化**：只允许对时间步 t 循环，样本和扩展位置都要并行算；
- 样本的 `input_length` 比 T 短时，超过它的时间步不参与；
- 不可能对齐时（例如标签 `aaa` 至少需要 5 步）损失为 `inf`；
- `reduction`：`"none"` 返回 `(N,)`；`"sum"`；`"mean"` 先把每个样本的损失除以它的标签长度，再对批求平均；
- 梯度由 autograd 自动算出，但要注意：如果用 −∞ 表示"不可能"的状态，全是 −∞ 的 logsumexp 在反向时会产生 nan。想办法避开它（提示：一个足够大的有限负数）。

**交付**：`run_ctc_loss`；`test_ctc_loss_matches_torch`（三种 reduction，数值和梯度）、`test_ctc_loss_impossible_is_inf` 通过。

### 问题（ctc_decode）：解码（5 分）

1. 贪心解码：每步取最大，再折叠（合并相邻重复，删掉空白）。
2. 前缀束搜索：每个前缀记两个概率：以空白结尾的 p_b、以非空白结尾的 p_nb（都在对数空间）。每一步扩展所有前缀，只保留 p_b ⊕ p_nb 最大的 `beam` 个。重点是同一个字符连续出现时怎么处理：隔着空白算两个字符，不隔就是同一个。束足够宽时，结果就是概率最大的标签，测试会和穷举对照。

**交付**：`run_ctc_greedy`、`run_ctc_beam`；`test_greedy`、`test_beam_search_is_exact_with_wide_beam`、`test_beam_beats_greedy_on_classic_case` 通过。

### 问题（lstm）：LSTM（5 分）

单层单向 LSTM，参数与 `nn.LSTM` 的一层相同：`weight_ih (4H, d_in)`、`weight_hh (4H, H)`、`bias_ih`、`bias_hh (4H,)`。四个门的顺序是 **i、f、g、o**：

```
[i, f, g, o] = x Wᵢₕᵀ + bᵢₕ + h Wₕₕᵀ + bₕₕ        （按最后一维切成 4 份）
c ← σ(f) ⊙ c + σ(i) ⊙ tanh(g)
h ← σ(o) ⊙ tanh(c)
```

输入部分 `x Wᵢₕᵀ` 可以一次算完所有时间步，只有依赖 h 的部分需要逐步循环。

**交付**：`run_lstm`；`test_lstm_matches_torch` 通过。

### 问题（crnn）：CRNN（3 分）

结构由你设计。要求：输入 `(B, 1, 32, W)`，输出 `(W/4, B, 75)` 的对数概率；包含你自己的卷积、双向 LSTM（正向一个、把序列倒过来再跑一个，输出拼起来）。第 13 章的结构是一个好起点：四个卷积块，前两次池化 2×2，第三次只缩高度（2×1），最后对高度求平均。

**交付**：`get_crnn`；`test_crnn_shapes_and_gradients` 通过（每个参数都要得到非零梯度）。

### 问题（cer）：字符错误率（1 分）

语料级 CER = Σ 编辑距离 / Σ 参考长度（插入、删除、替换各算 1）。

**交付**：`run_cer`；`test_cer` 通过。

### 问题（ctc_by_hand）：手算（3 分，书面）

两个类别：空白 ∅ 和 a。三个时间步的概率：

| t | ∅ | a |
|---|---|---|
| 1 | 0.4 | 0.6 |
| 2 | 0.6 | 0.4 |
| 3 | 0.3 | 0.7 |

1. 列出全部 8 条路径，各自折叠成什么、概率多少；算出 P("")、P("a")、P("aa")，并验证三者之和为 1。
2. 贪心解码得到什么？概率最大的标签是什么？
3. 用前向算法（α 表）重新算 P("a")，写出每一格的值。

---

## 5. 第三部分：Transformer 语言模型

### 5.1 结构

与第 16 章的 GPT 相同：

```
token 编号 (B, T) → 嵌入 tok (V, d)
→ n_layer × Block：x ← x + Attn(RMSNorm₁(x))；x ← x + SwiGLU(RMSNorm₂(x))
→ RMSNorm → head（与嵌入共享权重，tie_weights=True 时）→ logits (B, T, V)
```

- **注意力**：`qkv` 一个线性层（无偏置）输出 3d，按 `view(B, T, 3, h, dh)` 拆成 q、k、v；q、k 加 RoPE，v 不加；因果 mask；合并各头后过 `o`（无偏置）。
- **SwiGLU**：`down(silu(gate(x)) ⊙ up(x))`，三个线性层都没有偏置，中间维 `d_ff`；`d_ff = 0` 时取 `64 · ⌈(8/3)·d / 64⌉`。
- **配置**：`vocab_size、block_size、n_layer、n_head、d_model、d_ff、rope_base、tie_weights`。

### 5.2 问题（rmsnorm）：RMSNorm（2 分）

`y = x / √(mean(x²) + eps) · weight`，对最后一维，weight 初始为 1。提示：在 float32 下算均方，再转回输入的类型（为将来的混合精度做准备）。

**交付**：`run_rmsnorm`；`test_rmsnorm` 通过。

### 5.3 问题（rope）：RoPE（3 分）

θᵢ = base^(−2i/dh)，i = 0 … dh/2 − 1。按 Qwen/Llama 的"前后两半"约定：第 i 维和第 i + dh/2 维组成一个平面，在位置 t 转过角度 t·θᵢ：

```
y[..., :dh/2] = x₁ cos − x₂ sin
y[..., dh/2:] = x₂ cos + x₁ sin
```

`run_rope(x, base, pos0)` 的第一个位置编号是 `pos0`（KV cache 逐个生成时要用）。测试用复数乘法独立地实现了一遍来对照。

**交付**：`run_rope`；`test_rope`、`test_rope_relative` 通过。

### 问题（rope_relative）：为什么是相对位置（2 分，书面）

证明：q 在位置 m、k 在位置 n，加 RoPE 后的点积只取决于 m − n。（提示：一个二维平面里，先把 q 转 mθ、把 k 转 nθ，再求点积，等于把 q 转 (m − n)θ 后与未转的 k 求点积。）

### 5.4 问题（attention）：缩放点积注意力（3 分）

`softmax(q kᵀ / √d) v`，mask 是 bool（True 表示可以看），能广播到 `(…, Tq, Tk)`。softmax 自己写，要数值稳定。

**交付**：`run_attention`；`test_attention` 通过（与 `F.scaled_dot_product_attention` 对照）。

### 5.5 问题（gpt）：GPT（6 分）

`state_dict()` 的键必须**恰好**是：

```
tok.weight
blocks.{i}.n1.weight     blocks.{i}.n2.weight
blocks.{i}.attn.qkv.weight   blocks.{i}.attn.o.weight
blocks.{i}.mlp.gate.weight   blocks.{i}.mlp.up.weight   blocks.{i}.mlp.down.weight
norm.weight
head.weight              （与 tok.weight 是同一个张量）
```

RoPE 的 cos、sin 表注册成 `persistent=False` 的缓冲区，不进 state_dict。前向 `model(idx, targets=None)` 返回 `(logits, loss)`，给了 targets 时 loss 是交叉熵（−100 的位置不计）。初始化：线性层和嵌入用 N(0, 0.02)，`o` 和 `down` 再除以 √(2·n_layer)。

**交付**：`get_gpt`、`run_gpt`；`test_gpt_matches_snapshot`、`test_gpt_structure`（键、参数量、因果性、损失）通过。

### 问题（gpt_accounting）：算账（4 分，书面）

默认配置：vocab 1024、d 256、4 层、4 头、上下文 256。

1. 逐项算参数量（嵌入、每层的注意力、MLP、两个 RMSNorm，最后的 RMSNorm）和总数。共享权重的输出层算不算？
2. 一个 token 的前向需要多少次浮点运算？分成"矩阵乘法里的权重"（每个权重 2 次）和"注意力的 QKᵀ 与 AV"（按满上下文 256 算）两部分。训练大约是前向的几倍？为什么？
3. KV cache：生成到第 256 个 token 时，缓存占多少字节（float32、bfloat16 各算一次）？和参数本身占的字节比一比。

### 5.6 问题（kv_cache）：KV cache（4 分）

注意力层接受一个可选的 cache（例如一个 dict）：有历史时，把新的 k、v 接在历史后面，并写回 cache；新位置能看到全部历史，以及新位置中在自己之前的位置。模型的前向要能接受"第一个位置的编号"`pos0`，RoPE 从这个位置开始。

**交付**：`run_gpt_incremental`；`test_kv_cache_matches_full`（预填充 1、5、16 个 token 之后逐个生成，logits 与一次算完相同）通过。

### 5.7 问题（sampling）：采样前处理 logits（3 分）

`process_logits(logits, temperature, top_k, top_p)`，顺序：除以温度 → top-k（只保留最大的 k 个）→ top-p（按概率从大到小排，保留累计概率**第一次达到** top_p 为止的那些，至少保留 1 个）。被去掉的位置设为 −∞。再写一个 `generate(model, idx, max_new, …)`：先预填充提示，之后每步只送 1 个新 token（用 KV cache）。

**交付**：`run_process_logits`；`test_process_logits` 通过。

### 5.8 问题（lm_utils）：取批、梯度累积、评测（5 分）

1. `get_batch(ids, batch_size, context, generator)`：随机起点 s ∈ [0, len − context)，`x = ids[s : s+context]`，`y = ids[s+1 : s+context+1]`。
2. `accumulate_grads(model, x, y, micro_batches)`：清零梯度，把批均分成几份，逐份前向反向，梯度累加。结果要与一次算整批**相同**。每一份的损失要乘以什么？
3. `nll_sum(model, ids, context)`：整段 token 的总负对数似然（nat）。**切窗方法**：窗口起点 s = 0, context, 2·context, …，每个窗口输入 `ids[s : s+context]`、预测 `ids[s+1 : s+context+1]`（最后一个窗口可能更短）。这样除了第一个 token，每个 token 恰好被预测一次。然后 **bits per char = nll_sum / ln 2 / 字符数**。为什么比较不同分词方式的模型时，必须用 bits per char，而不是每个 token 的损失或困惑度？

**交付**：`run_get_batch`、`run_accumulate_grads`、`run_nll_sum`；`test_get_batch`、`test_accumulate_grads_matches_full_batch`、`test_nll_sum` 通过。

---

## 6. 训练脚本

写两个训练入口（可以共用代码），要求与 A1 第 7 节相同：可复现、日志写进 `log.jsonl`、配置写进 `config.json`、存检查点、`--smoke` 几秒跑完、从命令行覆盖任意配置、**默认 `device=cpu`**。

- **OCR**：在 `LineDataset(seed, n, noise=0.15, distort=1.0)` 上训练 CRNN；定期在验证集上算 CER；能对排行榜测试集做预测，每行一个预测写进文本文件。建议起点：2 万行训练、500 行验证，batch 64，AdamW，max_lr 1e-3，3000 步。
- **LM**：用 `mixed_corpus(0, 20000)` 训练 BPE 和 GPT，`mixed_corpus(1, 400)` 做验证；定期算验证集的 bits per char；能在排行榜测试文本上算 nll_sum；能生成样例。建议起点：词表 1024，第 5.5 节的默认配置，batch 32，max_lr 3e-3，3000 步。

不需要测试，但报告里要写清楚用法，并贴一次 `--smoke` 的输出。

---

## 7. 实验

要求与 A1 第 8 节相同：先写预测，再运行；图从 `log.jsonl` 画；只跑过一次的结论要标明；需要比较大小的结论至少跑 2 个种子。每次运行先 `--smoke`。

### 问题（exp_bpe）：分词器（3 分）

在 `mixed_corpus(0, 20000)` 上训练词表 96（纯字符）、256、512、1024、2048、4096 的分词器。

**交付**：(a) 验证文本上的"字符/token"随词表大小的变化图；(b) 训练用时随词表大小的变化。和第 14 章的朴素实现比一比，你的增量更新快多少？(c) 前 20 条合并是什么？为什么是它们？(d) `\frac`、`|---|`、`3.14`、`$$` 各被切成什么？数字为什么总是单个字符？这对模型学算术或抄写数字有什么好处？

### 问题（exp_ctc）：CTC 识别器（5 分）

**交付**：

(a) 有 LSTM 和没有 LSTM（每列独立分类）的验证 CER 对比，各 2 个种子。运行前先预测：哪些字符最容易错？

(b) 贪心解码和束宽 4、16 的束搜索：CER 和解码用时。差距大吗？为什么 CTC 的束搜索常常只比贪心好一点点？（提示：看下一问。）

(c) 选 3 张图，画出每一帧的 argmax（或者画出 (T, C) 的概率热图）。空白帧占多少？非空白的输出是不是集中在很少几帧（"尖峰"）？

(d) 在 noise = 0、0.15、0.3 的测试集上评测同一个模型（训练用 0.15）。

### 问题（exp_lm_lr）：学习率（3 分）

max_lr 取 3e-4、1e-3、3e-3、1e-2、3e-2，验证 bits/char 的曲线和最终值。最好的学习率和 A1 的 CharCNN 差多少？

### 问题（exp_lm_ablation）：消融（5 分）

以你最好的配置为基准，每次只改一处，**参数量尽量保持一致**（必要时调整中间维）：

1. 去掉 RoPE（不加任何位置信息）；
2. Post-Norm：`x ← Norm(x + Attn(x))`；
3. 去掉所有 RMSNorm；
4. 把 SwiGLU 换成 ReLU MLP（`down(relu(up(x)))`，中间维取 4d，使参数量接近）；
5. 输出层不共享权重。

**交付**：一张表（参数量、验证 bits/char、训练是否稳定），一段分析。哪个改动影响最大？去掉 RoPE 后，模型从哪里还能得到位置信息？

### 问题（exp_tokenization）：分词粒度（3 分）

词表 96、512、2048 各训练一个 GPT。**计算量要相同**：同样的步数、同样的 batch、同样的上下文长度（token 数）。

**交付**：验证 bits/char 对比，一段分析：词表大的模型，一个上下文窗口能看到多少字符？嵌入层的参数占比变成多少？在相同的计算量下，哪种更好？

### 问题（exp_generation）：生成（2 分）

用最好的模型，以 `"# "`、`"$$"`、`"| "` 开头，在温度 0.5、1.0、1.5 和 top-p 0.9 下各生成 20 段。

**交付**：每种设置贴 2 段样例；统计生成的 `$$…$$` 公式里 `{` 和 `}` 配对正确的比例。温度怎么影响"像不像语料"和"多样性"？

---

## 8. 排行榜

### 问题（leaderboard_ocr）：文字行识别（3 分）

目标：`data/ocr_test.npz`（1000 行，noise 0.15，distort 1.0）上的 CER 尽量低。用 `python -m assignments.a2_sequences.leaderboard ocr 你的预测.txt` 打分。

### 问题（leaderboard_lm）：语言模型（3 分）

目标：`data/lm_test.txt` 上的 bits/char 尽量低。用你的 `nll_sum` 在测试文本上算总负对数似然（上下文就用你模型的 block_size，切窗方法按第 5.8 节），再用 `python -m assignments.a2_sequences.leaderboard lm --nll 总nat数` 换算。

**两个排行榜共同的规则**

1. **硬件和时间**：一张租用的 24 GB 显卡（RTX 4090、L4、A10 这一级），训练墙钟时间不超过 **15 分钟**；生成数据、训练分词器的时间不算。CPU 赛道：60 分钟，写明 CPU 型号。
2. **数据**：只能用 `miniocr.data` 的生成器，种子不能用 20261022–20261099；OCR 训练行数不超过 50 万，LM 训练语料不超过 5000 万字符。**不能用测试集训练、调参或选检查点**。
3. **代码**：第 1.1 节的限制照样适用。在这个前提下，结构、分词器、上下文长度、混合精度（`torch.autocast`）、`torch.compile` 都可以自由改。
4. **租用流程**：与 A1 第 9 节相同（先在本机 `--smoke`，开机后先跑 `--smoke --set device=cuda`，结果关机前拷回，开机前写好次数和时长上限，记录费用）。

**交付**：`leaderboard.md` 每个榜加一行；报告里各一段：试过什么、什么有用、为什么，附最终运行的学习曲线。不按名次给分，按要求完成、记录完整、分析清楚即可得满分。

---

## 9. 交付清单与评分

| 部分 | 问题 | 分数 |
|---|---|---|
| 代码：分词 | pretokenize 2、train_bpe 8、tokenizer 6 | 16 |
| 代码：CTC | ctc_loss 8、ctc_decode 5、lstm 5、crnn 3、cer 1 | 22 |
| 代码：GPT | rmsnorm 2、rope 3、attention 3、gpt 6、kv_cache 4、sampling 3、lm_utils 5 | 26 |
| 书面 | ctc_by_hand 3、gpt_accounting 4、rope_relative 2 | 9 |
| 实验 | exp_bpe 3、exp_ctc 5、exp_lm_lr 3、exp_lm_ablation 5、exp_tokenization 3、exp_generation 2 | 21 |
| 排行榜 | leaderboard_ocr 3、leaderboard_lm 3 | 6 |
| **合计** | | **100** |

完成后回到第 0.11 节的里程碑 B：你有了自己写的 CTC 文字行识别器和 GPT，并且 KV cache 与不用 cache 的结果逐位一致。下一站是第 19 章（ViT），A3 在第 24 章之后。
