# A2：从零写 BPE 和 GPT，训练 LaTeX 语言模型

**前置**：第 6–16 章；第 0 章练习 0.1–0.4（2.4 节说明原因）。**算力**：机制题和冒烟只要 CPU；训练层的设备配置待校准（第 9 节）。**总分**：100。

A2 分三层：机制题（70 分）、书面题（10 分）、训练层（20 分）。你从一个空的包开始，自己写字节级 BPE、带 RoPE 的 GPT、KV cache、解码和训练工具，再写训练脚本，训练一个无条件的 LaTeX 语言模型，用统一脚本报告 bits/byte 和可编译率，并做消融。

你要交的东西：

1. `assignments/a2/mechanism/` 里的代码，包括训练脚本 `train.py`；
2. `assignments/a2/tests/adapters.py` 里的接线（测试和评测脚本通过它调用你的代码）；
3. 分词器、检查点、日志和评测结果；
4. 实验报告 `writeup.md`（模板见 `writeup_template.md`）。

**和里程碑 M1 的关系。** M1 和 A2 的训练层共用一次训练，不用训两遍。M1 的五项交付都在 A2 里：

| M1 交付 | 对应题目 |
|---|---|
| 训练曲线 | train_script |
| bits/byte、生成公式的可编译率 | latex_metrics |
| LaTeX 分词分析 | latex_metrics |
| 消融结果 | ablations |

第 13 章的 tour 用的是参考实现。它可以用来对照和理解流程，它产出的日志和检查点不算 A2 的结果。

## 1. 规则

### 1.1 能用什么、不能用什么

`mechanism/` 是**机制实现部分**（`rules.toml` 里列出），受自动检查（`tests/test_a2_rules.py`），子目录也会检查。

- **可以用**：PyTorch 的张量运算（`@`、`exp`、`log`、`sum`、`amax`、`tril`、`where`、`sigmoid`、`multinomial`……）、自动求导、`torch.nn.Module`、`torch.nn.Parameter`、`torch.nn.ModuleList`、`torch.nn.ModuleDict`、`torch.nn.Sequential`、`torch.optim.Optimizer`（作为基类）、`regex`、Python 标准库、NumPy；
- **不能用**：`torch.nn` 里的现成层（`Linear`、`Embedding`、`LayerNorm`、`RMSNorm`……）、`torch.nn.functional` 的任何函数（包括 `cross_entropy`、`softmax`、`silu`、`scaled_dot_product_attention`）、`torch.softmax`、`torch.log_softmax`、张量的 `.softmax()`、`.log_softmax()`、`.silu()` 方法、`torch.optim` 里的现成优化器、`torch.nn.init`、`torch.nn.utils.clip_grad_norm_`；
- **不能 import** `miniocr` 里除 `miniocr.data` 和 `miniocr.device` 之外的任何模块。`miniocr` 的其余部分就是参考答案；
- **不能 import 其他作业**，也不能在 `mechanism/` 里用 `assignments.` 开头的绝对路径。`mechanism/` 内部用相对导入（`from .optim import ...`），训练脚本不能导入 `assignments.a2.tests`。可以复制你自己写过的代码，比如 A1 的 AdamW、调度、裁剪和检查点。

`adapters.py` 只接线，算法都写在 `mechanism/` 里。检查器用语法树查找，抓不住所有绕过方式；规则以本节为准，`adapters.py` 里有没有藏算法由人工审阅。测试文件里用了 `F.scaled_dot_product_attention` 等做对照，那是测试，不受限制。

**章节代码不能照搬。** 第 8–15 章的 `miniocr` 代码用了 `nn.Linear`、`nn.Embedding`、`F.scaled_dot_product_attention`、`F.silu`、`F.cross_entropy`，13.5.3 节的开关用了 `nn.LayerNorm`。这些都不能进 `mechanism/`。

**数据工具有一部分不能用。** `miniocr.data.latex_lm` 能通过 import 检查，但里面的 `train_tokenizer` 和 `load_latex_lm` 调用的是参考 BPE，你不能用。`miniocr.data.latex_corpus` 的 `standard_split`、`corpus_available`、`normalization_key`，以及 `miniocr.data.latex_lm` 的 `FormulaWindows`，都可以直接用。

**分析脚本放在 `mechanism/` 外面。** 比较解码设置要用 `miniocr.eval.render.can_parse`，它不能进 `mechanism/`。这类脚本放到别的目录（例如 `assignments/a2/analysis/`），模型和分词器仍然通过 `tests/adapters.py` 取你自己的实现。

### 1.2 AI 助手

可以问概念、问报错、让 AI 审阅你的代码、要分级提示。**不要让它替你写 `mechanism/` 里的代码**，写的时候建议关闭 AI 自动补全。报告由你自己写。

### 1.3 数字要诚实

报告里的每个数字都要能在日志里找到：运行 id、配置、设备、日志行、代码位置。只跑过一次的结论要标明。没有跑完的栏目写“未运行”，不填推测的结果。CPU 冒烟和章节演示的数字不算训练结果。

### 1.4 验证集和测试集

规则来自第 0 章 0.10.1 节：

- **验证集用来做决定。** 调参、选模型、选解码设置、看错例、提假设，都只用它，可以反复用；
- **测试集是留出集。** 选型冻结以后，预先登记的候选可以各评一次，只报告数字和区间，不根据结果改任何设置；
- **开发期不看测试集的错例。** 最终报告定稿以后才可以解封；
- **已经根据测试集改了设置怎么办？** 在报告里写明，再补一份新的、没看过的测试集。

## 2. 准备

### 2.1 命令

```bash
cd course-v4                                              # 或 make_student 生成的 my-course/
python -m pytest -q assignments/a2/tests                  # 一开始：规则和评测协议的独立测试通过，其余因 NotImplementedError 失败
python -m pytest -q assignments/a2/tests/test_a2_rules.py # 只跑规则检查
```

始终不要设置环境变量 `A_SOLUTIONS`。设了以后，测的就不是你的代码。

### 2.2 代码结构

建议的结构（不强制）：

```
mechanism/
  tokenizer.py    预分词、BPE 训练、编码解码
  nn.py           线性层、嵌入、归一化、softmax、交叉熵
  model.py        RoPE、注意力、多头、SwiGLU、Block、GPT、KV cache
  generate.py     logits 处理、生成循环
  optim.py        AdamW、调度、裁剪（可以从 A1 复制）
  checkpoint.py   保存和恢复
  train.py        训练脚本
```

机制题按行为评测，不要求内部类名和别人一样。`state_dict` 的键名、分词器文件和检查点的格式是公开格式，必须遵守。

### 2.3 统一约定

- 非法形状或参数抛 `ValueError`；
- 张量输入不就地修改；
- 每题的完整契约写在 `tests/adapters.py` 的文档字符串和本讲义里。各题测试是行为契约，不要求你复制参考实现的内部算法。

### 2.4 语料

语料是公开数据集 OleehyO/latex-formulas 的 `cleaned_formulas`（第 6 章本章导读有来源和许可证）。只在本地读取，不要分发。

- 脚本从 Hugging Face 的缓存目录读 `datasets--OleehyO--latex-formulas`，遵守 `HF_HOME` 和 `HF_HUB_CACHE`。评测脚本不会替你下载，先自己把数据集放进缓存；
- 用 `corpus_available()` 检查缓存。没有缓存时 `standard_split` 只发警告，然后降级成几十条内置样例，训练不会报错。**你的 `train.py` 必须先检查，没有缓存就退出。** 评测脚本在这种情况下会拒绝正式评测；
- 如果你在 `make_student` 生成的目录里工作：`standard_split` 依赖第 0 章练习 0.1–0.4 的 `tokenize_latex`、`apply_token_rules`、`parse_latex`、`serialize`。没做完会抛 `NotImplementedError`。先让 `python -m pytest -q book/ch00_intro` 里对应的测试通过。

## 3. 第一部分：字节级 BPE

### 问题（pretokenize）：预分词（2 分）

实现 `run_pretokenize(text)`。第 6 章 6.4 节。

用下面这个正则（需要 `regex` 模块）切词块。拼回去必须等于原文，不丢、不多任何字符。

```python
PAT = r"""'(?:s|t|re|ve|m|ll|d)| ?\p{L}+| ?\p{N}| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

`I'm 12 岁。` 切成 `I`、`'m`、` 1`、`2`、` 岁`、`。`。测试还用手写的中文、多行文本和 LaTeX 词块。

### 问题（train_bpe）：训练 BPE（8 分）

实现 `run_train_bpe(texts, n_merges, special_tokens)`，返回 `(vocab, merges)`。第 6 章 6.5–6.7 节。

- `vocab` 的编号：0–255 是单个字节，256 + i 是第 i 条合并的结果，特殊符号按给出的顺序排在最后；
- 特殊符号不参与训练：先在文本里把它们切掉，切开的两边分别预分词；
- 每轮合并次数最多的相邻对。**平局时取 `(bytes, bytes)` 元组更大的一对**；最高次数小于 2 时提前结束；
- 负的 `n_merges`、空的或重复的特殊符号抛 `ValueError`。

测试包括手算的前三轮、平局字节序，以及 80 条合并的 LaTeX 快照（逐条相同）。正式规模是 2 万条公式学 2000 条合并，建议用 6.5.6 节的增量更新。这个规模的速度测试不检查。

### 问题（tokenizer）：编码与解码（6 分）

实现 `get_tokenizer(vocab, merges, special_tokens)`，返回的对象有 `encode`、`decode`、`encode_iterable`。第 6 章 6.3、6.6–6.10 节。

- `encode(text, allow_special=True)`：特殊符号整段编码成一个编号，多个特殊符号重叠时取最长的；其余部分预分词后，在每个词块内**按合并规则的先后顺序**合并；
- `encode(text, allow_special=False)`：特殊符号的拼写当普通文字。**语料正文必须走这条路径**（6.7.3 节）；
- `decode(ids)`：字节拼起来按 UTF-8 解码，不完整的字节用 U+FFFD。语料里没见过的字退回字节，半个汉字解码成替换字符，不抛异常；
- `encode_iterable(iterable)`：逐个元素编码，依次产出编号。只有元素边界也是预分词边界时，结果才等于整段 `encode`。连续空行就是反例：`\n\n` 整体是一个词块，逐行切开后变成两个。测试用逐行读取的文本。任意切分都一致不是必做题。

### 问题（bpe_by_hand）：手算（2 分，书面）

语料 `["ab ab abc abc abc"]`，学 3 条合并。写出每一轮所有相邻对的次数和选中的一对，再写出 `" abcab"` 的编码结果。第 6 章 6.5 节、6.6 节。

## 4. 第二部分：Transformer 语言模型

### 问题（rmsnorm）：RMSNorm（4 分）

实现 `run_rmsnorm(d, eps, weight, x)`，在最后一维计算。第 9 章 9.3 节，第 16 章 16.2 节。

- y = x / √(mean(x²) + ε) · γ，γ 就是 `weight`。输入和 `weight` 都要保留梯度，所以外部传进来的 `weight` 不能复制后切断；
- bf16、fp16 输入在 fp32 里算，结果转回原 dtype。大数值的 bf16 输入不能溢出；
- 形状不符或 ε < 0 要拒绝。

x = [3, 4]、γ = [2, 3]、ε = 0 时，结果是 [6, 12]/√12.5。

### 问题（rope）：RoPE（5 分）

实现 `run_rope(x, base, pos0)`，x 的形状是 (B, h, T, dh)，第 t 个位置的编号是 pos0 + t。第 10 章 10.3–10.5 节。

- θ_i = base^(−2i/dh)，第 i 维和第 i + dh/2 维配对（“前后两半”约定）；
- 测试用一份独立的复数写法逐元素对拍，并检查保范数、相对位置、位置偏移和输入梯度；
- dh 为奇数、pos0 为负、base ≤ 0 要拒绝；
- 角度不要在 bf16 里算（10.10.2 节）。

### 问题（rope_relative）：为什么是相对位置（2 分，书面）

证明旋转后的点积只取决于相对位置。如果训练用“前后两半”约定，推理误用“相邻两维”约定，相对位置的测试能抓住吗？什么测试能抓住？第 10 章 10.3.3 节、10.5.5 节。

### 问题（attention）：缩放点积注意力（6 分）

实现 `run_attention(q, k, v, mask)`，只返回输出。第 8 章 8.2–8.7 节。

- q、k 的形状是 (..., T, d)，v 是 (..., Tk, dv)；
- `mask` 是 bool，True 表示可以看；`None` 表示全都可以看；
- softmax 自己写，先减去每行最大值（第 3 章 3.3.4 节）；
- 错尺寸、非 bool 的 mask 要拒绝。**有一行全被遮蔽时抛 `ValueError`，不能返回 NaN**（8.7.4 节）。

测试和 `F.scaled_dot_product_attention` 对拍，并手算 Q = K = 0 时因果 mask 下的均值：v = [[2, 4], [6, 8]] 得到 [[2, 4], [4, 6]]。q、k、v 的梯度也要对上。

### 问题（multihead）：多头注意力（6 分）

实现 `run_multihead(x, kv, n_heads, wq, wk, wv, wo, mask=None)`。第 8 章 8.3、8.8 节，第 9 章。

- x 是 (B, Tq, d)，kv 是 (B, Tk, d)，四个权重都是 (d, d)；
- 线性变换写成 `x @ W.T`，没有偏置，也没有位置编码；
- mask 的 True 表示可见，返回 (B, Tq, d)，所有输入和四个权重都保留梯度；
- 头数不能整除 d、形状错误要拒绝。

测试对两个头逐头独立计算再拼接，并覆盖 Tq ≠ Tk 的交叉注意力。

### 问题（swiglu）：SwiGLU（4 分）

实现 `run_swiglu(x, gate, up, down)`。第 9 章 9.5 节。

- x 是 (..., d)，`gate`、`up` 是 (f, d)，`down` 是 (d, f)；
- 输出 down(silu(gate(x)) ⊙ up(x))，没有偏置，线性变换的写法同上题；
- 不能用 `.silu()` 和 `F.silu`，用 `torch.sigmoid` 写。

三个权重都是单位阵、x = [0, 1] 时，输出 [0, 1/(1 + e⁻¹)]。

### 问题（gpt）：GPT（10 分）

实现 `get_gpt(cfg)` 和 `run_gpt(cfg, state_dict, idx)`。第 11 章，第 13 章 13.5 节。

**结构**：

```
tok：Embedding(V, d) → [Block] × L → RMSNorm → head（无偏置，默认与 tok 共享）
Block：x = x + Attn(RMSNorm(x))；x = x + SwiGLU(RMSNorm(x))
Attn：q、k、v、o 四个 d×d 的无偏置线性层；q、k 加 RoPE；因果 mask；普通多头，没有 dropout
```

不要求 GQA 和 Q/K norm。

**`cfg` 的键**：

- 必需：`vocab_size`、`block_size`、`n_layer`、`n_head`、`d_model`、`d_ff`、`rope_base`、`tie_weights`；
- 可选，消融用：`pos_emb`（`rope`、`none`、`learned`，默认 `rope`）、`norm`（`rmsnorm`、`layernorm`，默认 `rmsnorm`）、`resid_scale`（默认 True）、`dropout`（默认 0）；
- `d_ff = 0` 表示 64·⌈(8d/3)/64⌉；
- 维度必须是正数，头数整除 d，头维是偶数，否则抛 `ValueError`。

**`state_dict` 的键名**（RoPE 表不持久化）：

```
tok.weight
blocks.{i}.n1.weight   blocks.{i}.n2.weight
blocks.{i}.attn.q.weight   blocks.{i}.attn.k.weight   blocks.{i}.attn.v.weight   blocks.{i}.attn.o.weight
blocks.{i}.mlp.gate.weight   blocks.{i}.mlp.up.weight   blocks.{i}.mlp.down.weight
norm.weight
head.weight
```

- 权重形状：`tok`、`head` 是 (V, d)；`q`、`k`、`v`、`o` 是 (d, d)；`gate`、`up` 是 (f, d)；`down` 是 (d, f)；归一化的 `weight` 是 (d,)；
- 共享时 `head.weight` 和 `tok.weight` 是同一个 `Parameter`，参数量只算一次；
- `pos_emb="learned"` 多一个 `pos.weight`，(T, d)；
- `norm="layernorm"` 时每个归一化多一个 `bias`，自己实现，不能用 `nn.LayerNorm`。

**`forward(idx, targets=None, caches=None, pos0=0, loss_mask=None)`** 返回 `(logits, loss)`：

- 没有 `targets` 时 `loss` 是 `None`；
- `loss` 按 token 平均，`targets` 里 −100 的位置不计入，`loss_mask` 为 False 的位置不计入；
- 训练时 softmax 不屏蔽 BOS 这一类；
- `caches` 和 `pos0` 见 kv_cache。

**初始化**：线性层和嵌入 std = 0.02，归一化 γ = 1；残差输出（`attn.o`、`mlp.down`）std = 0.02/√(2L)，`resid_scale=False` 时取 0.02。测试不检查分布。把你的统计写进报告，不要把一次随机样本当标准答案。

**测试检查**：一份独立写法的前向（复数 RoPE + SDPA）的 logits 和梯度、参数键和不重复的参数量、−100 目标、因果性。无位置、学习式位置、LayerNorm、不共享、`d_ff=0` 这些变体要能跑，增量前向要和整段前向一致。

测试快照是 V = 64、T = 32、L = 2、h = 2、d = 32、f = 96 的小配置，只用来核对数值。第 11 章 11.1.2 节的本书配置是实际词表、T = 128、L = 4、h = 4、d = 256、f = 704，它和这份快照、第 9 节的起始配置都不相同。

### 问题（gpt_accounting）：算账（4 分，书面）

第 11 章 11.3 节，第 13 章 13.6 节，第 15 章 15.5 节，第 16 章 16.6、16.7、16.10、16.11 节。

对你实际用的配置：

1. 写出共享权重、RMSNorm、普通多头时的参数量公式。不共享、学习式位置、LayerNorm 分别多多少？代入数字，和代码数出的结果对照；
2. 写出 KV cache 的字节数公式，代入你的 B、T 和精度；
3. fp32 AdamW 的静态内存是多少？激活值怎样得到？静态加激活等于峰值吗？
4. 写出 6ND 里 N 的口径，再写出注意力附加的计算量。MFU 用什么峰值？

## 5. 第三部分：KV cache 与解码

### 问题（kv_cache）：KV cache（5 分）

实现 `run_gpt_incremental(cfg, state_dict, idx, n_prefill)`。第 15 章 15.2–15.4 节。

- 先把 `idx[:, :n_prefill]` 一次送进去，之后每次送 1 个 token，把每步的 logits 拼起来，形状 (B, T, V)；
- 每层的 cache 是 `{k, v}`，保存旋转后的 k 和未旋转的 v，形状 (B, h, T, dh)。不重算整个前缀；
- `n_prefill` 不在 [1, T] 内要拒绝。

测试在 n_prefill = 1、5、16 时和整段前向对拍，消融变体也要对上。

### 问题（decoding）：logits 处理与生成（6 分）

第 14 章 14.2–14.5 节，第 15 章 15.3 节。

实现 `run_process_logits(logits, temperature, top_k, top_p)`：

- logits 是 (B, V)，返回处理后的 logits，不许采样的位置是 −∞，不修改输入；
- 顺序是温度 → top-k → top-p；
- top-k：和第 k 大并列的全部保留；
- top-p：按概率从大到小，保留到累计概率**第一次达到** p 为止（含跨过阈值的那一个），至少保留 1 个；
- 温度必须是有限正数，top_k ≥ 0，0 < top_p ≤ 1，否则拒绝。

实现 `run_generate(model, idx, max_new, eos, banned, greedy=True, seed=0, use_cache=True)`：

- 返回含提示的编号。`banned` 里的编号不能生成；
- 遇到 EOS 的行之后一直补 EOS，全部结束就停；
- 贪心，或温度 1 的多项式采样，同一个 `seed` 结果相同；
- `max_new` 为负、提示为空要拒绝；`max_new = 0` 原样返回；
- **调用模型的方式是契约的一部分**，因为测试会传入一个脚本模型：
  - 用 `model(x, caches=caches, pos0=p)` 调用，取返回值的第 0 项作 logits；
  - `use_cache=True` 时先预填充整个提示（`pos0` 为 0），之后每步只传 1 个 token，`pos0` 是它的位置；
  - `caches` 按 `model.blocks` 的长度建，每层一个空 dict。不要依赖 `model` 上的其他方法。

束搜索、重复惩罚等第 14 章的扩展不在 A2 的接口里。

## 6. 第四部分：训练工具

### 问题（lm_utils）：取批、累积、总 NLL（8 分）

第 7 章 7.2 节，第 13 章 13.1.3 节，第 16 章 16.9 节。

实现 `run_get_batch(ids, batch_size, context, generator)`：

- 起点均匀取自 [0, len(ids) − context − 1]，两端都要能取到；x = ids[s:s+context]，y 错开一位；用传入的生成器；
- `ids` 不是一维 long、`batch_size` < 1、`context` 不在 [1, len(ids) − 1] 内要拒绝。

实现 `run_accumulate_grads(model, micro_batches)`：

- 先清零梯度，再对每个小批 (x, y) 前向、反向；y 里 −100 的位置不计；
- 累积后的 `.grad` 必须等于所有小批拼成一个大批、按全部有效 token 平均的梯度；
- 返回大批的平均损失。没有有效目标要拒绝。

实现 `run_nll_sum(model, ids, context)`：

- 从位置 0 起每 context 个 token 切一个窗口，最后一个窗口可以更短，每个窗口独立前向，不跨窗传缓存；
- 返回所有目标的总负对数似然（nat）。长度小于 2 或 context < 1 要拒绝。

`run_nll_sum` 只是训练诊断工具。**它不是正式评测的口径**（第 8 节），两者的数字不要放在一起比。

### 问题（protocol_and_accum）：协议与累积（2 分，书面）

变长样本做梯度累积时，为什么不能把每个微批的平均 loss 再平均？正确的写法是什么？bits/byte 的分子分母分别算什么，EOS、BOS 怎样处理？test 在什么时候、怎样才可以评？第 0 章 0.10.1 节，第 7 章 7.8 节，第 16 章 16.9 节。

## 7. 训练层任务

训练层共 20 分。分工如下：

| 题目 | 内容 | 分 |
|---|---|---|
| train_script | 训练脚本、数据、曲线 | 6 |
| checkpoint | 可中断续训与对照 | 3 |
| latex_metrics | 统一评测、分词分析（第 8 节） | 4 |
| ablations | 组件、词表、解码三类比较（第 10 节） | 5 |
| resources_and_report | 资源核算（第 11 节）、完整记录 | 2 |

### 7.1 数据协议 latex-v1

```python
from miniocr.data.latex_corpus import corpus_available, standard_split

if not corpus_available():
    raise SystemExit("缺本地语料")
split = standard_split(n_train=max(20000, N))   # 协议参数保持默认
bpe_text = split["train"][:20000]              # 只用来学 BPE
train_text, val_text = split["train"][:N], split["val"]
del split["test"]                               # 开发期不用
```

- 清洗后按规范化结果分组，再按哈希分到 val 和 test，各约 5000 条，边界组不拆。**不要先截原语料再划分**。扩大训练集只取 train 更长的前缀（第 13 章 13.1 节）；
- `standard_split` 只能整体生成，开发进程的内存里会短暂有 test。规则管的是你的行为：丢掉它，不对它做预测、统计或查看；
- BPE 只在 `bpe_text` 上学 2000 条合并，特殊符号依序是 `<|bos|>`、`<|eos|>`。这是初始协议，词表规模的比较另记（第 10 节）；
- 正文一律 `allow_special=False`，每条加 BOS、EOS：`[bos] + tok.encode(t, allow_special=False) + [eos]`；
- 训练可以拼成一条流再随机截窗。窗口跨公式时，目标是 BOS 的位置要忽略：把它设成 −100，或用 `loss_mask`（13.1.3 节）；
- 泄漏检查：用 `normalization_key` 算 train 前缀和 val 的键交集，应当是 0（12.3.5 节）。

分词器保存成 JSON，评测脚本按这个格式读，合并的两边都写成十六进制：

```json
{"merges": [["5c", "66"], ["5c66", "72"]], "specials": ["<|bos|>", "<|eos|>"]}
```

**自查账目。** 第 13 章 13.1.3 节列了协议下的账目：词表 2258、训练流 1,189,263 个 token、验证 119,750 个目标、验证原文 239,566 字节（参考实现，本项目观察）。这些数字由数据和 BPE 决定，和模型无关。你的 BPE 合并结果和第 6 章的约定一致时应当对得上；对不上，先查数据和 BPE，再开训练。

### 问题（train_script）：训练脚本（6 分）

在 `mechanism/train.py` 里写训练脚本，要求：

- **用你自己的零件**：你的 BPE、GPT、交叉熵、AdamW、预热 + 余弦、梯度裁剪（4.4–4.7 节，13.2 节），按有效 token 加权的梯度累积（16.9 节）；
- **训练前的检查**：初始 loss ≈ ln V（11.5 节），同一批过拟合（11.7 节）；
- **日志**：逐步记录 train loss，固定步记录验证点，还有 lr、梯度范数、token/秒、累计 token，存成文件；
- **验证**：用第 8 节的口径。周期验证可以只用前若干条公式省时间；周期验证和最终验证样本量不同时，分开画线（13.1.4 节），日志里记下公式数、目标数和字节数；
- **检查点格式**：`torch.save` 的字典里至少有 `cfg` 和 `state_dict`，评测脚本只读这两项。`cfg` 就是传给 `get_gpt` 的 dict，`vocab_size` 必须等于分词器的词表大小。

报告里附训练曲线，并按第 13 章 13.3 节解释它：起点、前期下降、后期的缓慢下降、训练和验证的差距。

**自检。** 机制题接好以后，先用内置样例走一遍流程：

```bash
python -m assignments.a2.evaluate --tokenizer bpe.json --ckpt model.pt --smoke --samples 2 --max-new 32
```

`--smoke` 只用内置的几条自写公式，不读语料，不能登记，也不能进 test。它只检查保存、加载和评测能走通，它的参数不能抄进正式配置，它的数字不能写成训练结果，也不能用来定门槛。

### 问题（checkpoint）：可中断续训（3 分）

第 13 章 13.4 节，第 4 章 4.10 节。

- 保存模型、优化器、步数、Python/NumPy/Torch/设备及取批生成器的随机状态、配置、分词器和数据的摘要、日志位置；
- 原子写入（13.4.5 节）；调度器的进度要恢复（13.4.3 节）；日志要接着写（13.4.4 节）；
- 续训时对比配置，配置变了就拒绝。

**提交对照**：不中断训到 T 步，和中断在第 c 步、恢复后再训到 T 步，比较最终权重和验证数字。同一环境才讨论逐位一致；换设备时比较曲线（4.10.4 节）。

评测协议测试里的 CPU 冒烟只走一次更新、保存、加载和评测，不代替这项验收。

## 8. 统一评测

### 8.1 验证集评测

```bash
python -m assignments.a2.evaluate --tokenizer bpe.json --ckpt model.pt --max-new 128 > runs/baseline-s0.val.json
```

可选参数：`--device`（`cpu`、`mps`、`cuda`，默认 `cpu`）、`--samples`（生成条数，默认 50）、`--max-new`（默认 128）、`--seed`（默认 0）。`--max-new` 不能超过 `block_size`，所以 CPU 配置要显式写 `--max-new 32`。评测脚本目前只在 CPU 上验证过，`mps` 和 `cuda` 没有验证。

### 8.2 bits/byte 的口径

正式评测固定取 val 前 2000 条（`--final-test` 时取 test 前 2000 条）：

- 每个内容 token 和 EOS 恰好作一次目标；
- 逐目标滑窗，窗口长度是 `block_size`，不跨公式，窗口内位置从 0 开始；
- BOS 只作条件，输出时屏蔽 BOS 这一类；
- 除以清洗后原文的 UTF-8 字节和 ln 2。BOS、EOS 不计字节。

用 bits/byte 比，不用 nat/token：分词器不同，nat/token 不能比（6.10 节，7.8 节）。这个口径和 `run_nll_sum` 的不重叠分块不同。随机长流训练时的近似验证也不能和它混用（13.11.4 节）。

### 8.3 可编译率的口径

- 从 BOS 开始，用温度 1 采样，禁止再生成 BOS，遇到 EOS 截止；
- 没遇到 EOS 就被截断的样本**也计入分母**；
- 输出成功数、总数、比例、Wilson 95% 区间和以 EOS 结束的条数（13.9.4、13.9.5 节）；
- 固定 `seed`、样本数和生成上限。生成是无条件的，不使用 test 提示。

**这个可编译率是近似。** 渲染器是 `matplotlib.mathtext`，脚本的 `renderer` 字段写明了。mathtext 不支持矩阵类环境，也不认识一部分合法命令（12.7 节，13.9.2 节）。所以这里报告的严格说是 mathtext 可解析率。真正的 TeX 后端没有验证过，**不能把这个数字写成 TeX 编译结果**。

模型的比例要和真实公式的参考值比。13.9.3 节量过 val 前 2000 条：1199/2000，0.5995，区间 [0.5779, 0.6208]。参考值也不是严格上界：模型爱写短公式时，比例可能更高，它只是更单调。所以要和 bits/byte 一起看。

A2 是无条件语言模型。第 0 章 0.9.3 节的 EM、CER、渲染一致率不适用，不要报告。

### 问题（latex_metrics）：LaTeX 指标（4 分）

报告：

- **bits/byte**：选定模型的 `metrics` 字段（公式数、字节数、目标数、总 NLL、nat/token、bits/byte）；
- **可编译率**：`compilation` 字段的全部内容，并和 13.9.3 节的参考值对照，写明限制；
- **分词分析**：你的分词器对 LaTeX 命令、括号、数字各怎样切（6.8.2、6.8.3、6.11.3 节），UTF-8 压缩率，即每 token 平均字节数（6.8.1 节）。分析只用 train 和 val。

门槛状态按第 9 节写。

## 9. 门槛与起始配置

下表抄自第 13 章的 `PRESETS` 和模型阶梯。**它们是起始建议，不是验证过的配置，表内每个数字都待校准。** 实际 V 由 BPE 决定，`d_ff = 0` 按 gpt 题的公式取整。

| 配置 | L / d / h / d_ff | batch / context / steps / warmup / lr | 精度 | bits/byte 门槛 | 可编译率门槛 |
|---|---|---|---|---|---|
| CUDA | 6 / 192 / 6 / 0（待校准） | 64 / 128 / 5000 / 200 / 0.003（待校准） | bf16（待校准） | 待校准 | 待校准 |
| MPS | 4 / 128 / 4 / 0（待校准） | 32 / 128 / 5000 / 200 / 0.003（待校准） | fp32（待校准） | 待校准 | 待校准 |
| CPU 兜底 | 2 / 64 / 4 / 0（待校准） | 16 / 32 / 60 / 6 / 0.003（待校准） | fp32（待校准） | 待校准（校准时可放宽） | 待校准（校准时可放宽） |

共同起点（全部待校准）：min_lr = 0.0003，weight_decay = 0.1，裁剪上限 1，accum = 1。`block_size` 至少等于 context。

CPU 的 60 步是流程起点，不能保证完整训练，也不能保证合格。第 13 章 13.1.7 节算过 MPS 和 CUDA 配置大约把训练流读多少遍。完整训练和真实 test 评测都还没有跑过，这里没有已验证的速度、显存、效果或设备等价性结论。

门槛目前全部**待校准**，由你第一次完整训练后确定：

- 定门槛**只看验证集结果**，并且**写在 test 评测之前**。看了 test 再定门槛，等于用 test 做决定；
- 冒烟和章节演示的数字不能作为校准依据；
- 还没做完整训练时，报告里就写“待校准”；
- 目前门槛不单独计分。训练层的 20 分看流程和证据。

测试通过不等于达到门槛。

## 10. 消融

每个比较都写：预测 → 配置差异 → 结果 → 解释。只在验证集上比较，选择规则在跑之前写定。

### 问题（ablations）：组件、词表、解码（5 分）

第 13 章 13.5 节，第 6 章 6.8 节，第 14 章 14.9 节。

**组件消融。** baseline 加至少两个组件消融，每个设置至少两个种子。从下面选：

| 变体 | `cfg` 的改动 |
|---|---|
| 无位置 | `pos_emb="none"` |
| 学习式位置 | `pos_emb="learned"` |
| LayerNorm | `norm="layernorm"` |
| 不共享 | `tie_weights=False` |
| 无残差初始化缩放 | `resid_scale=False` |

- 先写预测（13.5.7 节）；
- 一次只改一项（13.5.4 节）。多头、FFN 这些机制不能换成现成层；
- 同数据、同预算、同评测（13.5.6 节）。参数量不同时写明差多少；
- **差别没有明显超过种子波动，就不下结论**（13.5.5 节）；
- 失败和发散的运行也保留。

**词表规模。** 至少比较两种 BPE 合并数。报告压缩率、验证 bits/byte，以及词表变大带来的参数成本（6.8.1 节，11.3 节）。

**解码设置。** 至少比较两种解码设置，比如温度或 top-p，报告可解析率、多样性和区间（14.9.2 节）。评测脚本只用温度 1 采样，额外的设置由你的分析脚本记录（规则见 1.1 节）。比较时用相同的模型、样本数、生成上限和种子。解码设置也是评测设置，选择只看验证集。

**必做之外。** 训练规模的消融（比如改 `n_train`）要在报告里先写计划，再跑。第 13 章 13.7 节的规模阶梯和拟合是拓展，不算 A2 的必做题，自动评分没有覆盖它。

## 11. 资源核算

### 问题（resources_and_report）：资源核算与记录（2 分）

第 16 章 16.11 节。

1. **先估算**：参数量、静态内存、激活、6ND 和注意力附加项；
2. **再实测**：跑一小段，测吞吐和峰值内存。峰值要写明 API 口径（16.11.4 节）。没有核实的设备峰值不填，MFU 写“待验证”；
3. **再对照**：写出差多少，解释差异；
4. **实际运行时间**：完整训练的耗时由你填。

报告还要写全这些记录：运行 id、代码版本或位置、完整配置、设备与库版本、命令、数据来源与使用权（0.10.2 节）、latex-v1 和数据快照、分词器摘要、划分与泄漏检查。

## 12. 冻结与最终测试（可选）

候选身份绑定 `mechanism/`（包括训练脚本 `train.py`）和学习者的 `tests/adapters.py`；参考模式也保留后者的 SHA256。接线变化会在启动前拒绝，不占次数。

A2 不强制评 test，这一步不计分。要评，就按下面做。没有评的话，报告写“未运行”。

**第一步：选型。** 超参、模型、解码设置全部在 val 上选完，门槛写进报告。

**第二步：登记。** 在同一份报告里逐个登记候选，建议把 baseline、选中设置和它的其他种子都登记，不要在 test 上挑种子：

```bash
python -m assignments.a2.evaluate --tokenizer bpe.json --ckpt model.pt --candidate baseline --register baseline_identity.json
```

这一步写两个文件：`--register` 给的审阅文件，和固定的 `assignments/a2/final_test_registry.json`。它记下检查点、分词器、语料快照、评测脚本、适配器和 `mechanism/` 源码的摘要，以及 samples、max-new、seed、device。登记不加载模型，所以先跑一次 8.1 节的命令，确认候选能加载。

**第三步：全部登记完以后，逐个评测。**

```bash
python -m assignments.a2.evaluate --tokenizer bpe.json --ckpt model.pt --candidate baseline --final-test
```

约束：

- 验证、登记、最终评测的 `--samples`、`--max-new`、`--seed`、`--device` 要一致；最终评测和登记不一致时会拒绝；
- 登记之后改动 `mechanism/`、`tests/adapters.py`、`evaluate.py`、检查点、分词器或语料快照，最终评测都会拒绝。这时还没有占用次数，把文件恢复原样才能继续；
- 第一次启动会封存候选表，之后不能再登记；
- 启动记录在加载测试划分之前写好，**失败也占用次数**；
- 同一内容不能改名重复登记，不能重试，也不要删登记表或换文件重来。这是本地防误用的协议，靠你守规矩，它防不住有意篡改。

每个候选的汇总写进 `assignments/a2/final_test_<候选名>.json`。结果只有汇总数字，不保存 test 文本、预测、逐条 loss 或错例。不要用这些结果挑模型，也不要因此改设置（1.4 节）。

## 13. 交付清单与评分

| 项 | 内容 | 分 |
|---|---|---|
| 机制题 | 测试全部通过（含规则测试）：pretokenize 2、train_bpe 8、tokenizer 6、rmsnorm 4、rope 5、attention 6、multihead 6、swiglu 4、gpt 10、kv_cache 5、decoding 6、lm_utils 8 | 70 |
| 书面题 | bpe_by_hand 2、rope_relative 2、gpt_accounting 4、protocol_and_accum 2 | 10 |
| train_script | `train.py`、分词器文件、初始 loss 和单批过拟合、日志、曲线和解释 | 6 |
| checkpoint | 不中断和中断恢复的权重、验证数字对照 | 3 |
| latex_metrics | 统一脚本的 bits/byte、可编译率（含区间、EOS 率、限制）、分词分析 | 4 |
| ablations | 预测、两个种子的组件消融表、词表规模、解码设置、不能确定的结论 | 5 |
| resources_and_report | 估算、实测、差异解释、运行时间、完整记录 | 2 |
| 合计 | | 100 |
| 代码可追溯 | 随机抽报告里 3 个数字，能给出运行 id、配置、设备、日志行、代码位置 | 通过 / 不通过 |

测试通过不等于学术诚信的证明。算法是否放在 `mechanism/`、`adapters.py` 是否只有接线、报告结论是否受 test 影响，都要人工审阅。
