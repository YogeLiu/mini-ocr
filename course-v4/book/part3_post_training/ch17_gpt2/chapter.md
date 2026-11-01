# 第 17 章　解剖 GPT-2

## 第三部分导读

第二部分结束时，你手里有一个自己从零写的 GPT：几百万参数，在 LaTeX 公式上训练。它能跑，但离"真实的大模型"还差得远——真实模型是别人花大量算力预训练好的，你要做的是**读懂它、加载它、再用数据和训练改变它的行为**。

第三部分就做这件事。全程围绕一个任务：**OCR 后纠错**。输入是一个 OCR 系统识别出来的 LaTeX（可能有错），输出是改正后的 LaTeX。比如 OCR 把 `\alpha` 认成了 `\aIpha`，把 `x^{2}` 漏成了 `x^{2`，模型要把它们改回来。本部分的数据用 latex-formulas 里的公式，人为加上 OCR 式的噪声（字符替换、括号丢失、命令拼错等）合成出"错误—正确"对。

路线是这样的：

| 章 | 主题 | 一句话 |
|---|---|---|
| 17 | 解剖 GPT-2 | 读懂并加载一个真实模型，逐个数字对拍 |
| 18 | 提示与上下文学习 | 不训练，只改提示词，建立最便宜的基线 |
| 19 | SFT 的监督信号 | chat template、label mask、EOS |
| 20 | LoRA | 只训练很少的参数 |
| 21 | 用 TRL 做 SFT 实验 | 用框架把后纠错训起来 |
| 22 | 评测与误差分析 | 分数可信吗？错在哪里？ |
| 23 | 偏好优化与 DPO | 从"好坏对比"里学习 |
| 24 | 强化学习基础 | 策略梯度的来龙去脉 |
| 25 | GRPO 与可验证奖励 | 用能自动判对错的奖励来训练 |

从第 18 章起，主角换成 Qwen3-0.6B。第 17 章先拿一个更小、更老、结构更简单的模型练手：GPT-2。

## 本章导读

GPT-2 是 OpenAI 在 2019 年发布的语言模型。最小的版本（下文简称 GPT-2，指 124M 的 small 版）有 124,439,808 个参数（数学结论，17.2 节），一台笔记本的 CPU 就能跑。它和你在第 11 章写的 GPT 是同一个家族：都是解码器式 Transformer，都用 Pre-Norm 残差块，都共享输入嵌入和输出层。

那为什么还要专门花一章？因为"同一个家族"和"能直接加载"之间隔着一堆细节。官方权重是一堆按名字存放的张量：`h.0.attn.c_attn.weight`、`h.0.mlp.c_fc.bias`……每个名字对应你代码里的哪一个模块？形状为什么是 (768, 2304) 而不是 (2304, 768)？一个转置忘了，代码照样能运行，只是输出全是胡话——而且某些情况下连形状检查都不报错。

这一章的核心技能是**对拍**：自己写一个 GPT-2 前向，加载官方权重，和 transformers 的输出逐个数字比较。这个方法可以搬到别的开源模型上。第 32 章解剖真实的 VLM 时会用到其中一部分：打印结构、数参数、把 processor 和官方实现逐位对拍。

本章路线：

1. **先认识它**：加载、打印结构（17.1），手算参数量（17.2），读懂 `config.json`（17.3）。
2. **找差别**：GPT-2 和你的 GPT 哪里一样、哪里不一样（17.4）。
3. **重点：加载并对拍**：从零写前向（17.5），转换权重（17.6），对拍的方法与容差（17.7），故障演练（17.8）。
4. **两个配角**：GPT-2 的字节级 BPE 分词器（17.9），从 GPT-2 到 Qwen3 的结构变化（17.10，概括）。

实验步骤都在 [lab.md](lab.md)，正文只引用部分号。tour 默认用 CPU、2 个线程；本章所有"本项目观察"的数字都来自 AMD Ryzen 9 9950X、CPU、2 线程、fp32、torch 2.14.1、transformers 5.19.0，可以用 `tour.py` 重跑。

> **回顾：本书的 GPT（第 9–11 章）**
>
> - Pre-Norm 块：`x = x + Attn(n1(x))`，`x = x + MLP(n2(x))`；最后一个块后面再接一个归一化。
> - 归一化用 RMSNorm，前馈用 SwiGLU，位置用 RoPE，所有线性层都不带偏置。
> - 输出层和 token 嵌入共享同一个矩阵（权重共享，第 11 章 11.4 节）。
> - 残差支路的输出层按 1/√(2L) 缩小初始化（第 11 章 11.6 节）。
> - 代码在 `miniocr/models/gpt.py`。

## 17.1 先见一面：加载并打印 GPT-2

### 17.1.1 三行代码

用 transformers 加载一个模型只要三行：

```python
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained(path, local_files_only=True, attn_implementation="eager")
print(model)
```

`path` 是本地快照目录（lab 第 0 节说明怎样准备）。`local_files_only=True` 保证不联网。`attn_implementation="eager"` 让注意力用显式的矩阵乘写法，便于对拍（17.7 节解释为什么）。

### 17.1.2 读结构树

`print(model)` 打印出一棵模块树（lab 第 1 部分）。去掉 dropout，骨架是这样的：

```text
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)          ← token 嵌入，V × d
    (wpe): Embedding(1024, 768)           ← 位置嵌入，P × d
    (h): ModuleList(
      (0-11): 12 x GPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05)
        (attn): GPT2Attention(
          (c_attn): Conv1D(nf=2304, nx=768)   ← q、k、v 三个投影拼在一起
          (c_proj): Conv1D(nf=768, nx=768)    ← 注意力输出投影
        )
        (ln_2): LayerNorm((768,), eps=1e-05)
        (mlp): GPT2MLP(
          (c_fc): Conv1D(nf=3072, nx=768)     ← d → 4d
          (c_proj): Conv1D(nf=768, nx=3072)   ← 4d → d
          (act): NewGELUActivation()
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05)  ← 最后的归一化
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)
```

试着把它和第 11 章的 GPT 一一对上。你会发现大部分都能认出来：嵌入、12 个块、每块里一个注意力一个 MLP、最后一个归一化、一个输出层。名字不一样（`h` 是 blocks，`c_attn` 是 q/k/v），但骨架是一样的。

陌生的东西只有几个：位置嵌入 `wpe`、`LayerNorm`、一个叫 `Conv1D` 的层、`NewGELUActivation`。17.4 节逐个解释。先剧透一条：`Conv1D` 不是卷积。它就是一个线性层，只是权重的存放方向和 `nn.Linear` 相反，加载权重时最容易在这里出错。

### 17.1.3 两种数参数的方法

数参数有两种常见写法：

```python
n1 = sum(p.numel() for p in model.parameters())
n2 = sum(t.numel() for t in model.state_dict().values())
```

结果（lab 第 1 部分）：n1 = 124,439,808，n2 = 163,037,184，差 38,597,376（本项目观察：CPU、2 线程，tour 第 1 部分）。

差的这一块恰好是 50257 × 768 = 38,597,376（数学结论），也就是一个嵌入矩阵的大小。原因是权重共享：`lm_head.weight` 和 `transformer.wte.weight` 是同一个张量。`parameters()` 会自动去重，同一个 Parameter 只出现一次；`state_dict()` 按名字列出，同一个张量在两个名字下各出现一次。

**哪个才是"参数量"？** 前一个。模型里真正独立可训练的数是 124,439,808。数参数时要用去重的方法——第 11 章的 `n_params` 也是这么做的。

> **停下来想一想：** 如果把权重共享解开（让 `lm_head` 有自己的矩阵），参数量变成多少？
> （答：多 V·d = 38,597,376，共 163,037,184。数学结论。）

## 17.2 参数量逐项手算

### 17.2.1 公式

记 V = 词表大小，P = 最大位置数，d = 宽度，L = 层数。GPT-2 的参数分四块：

**嵌入：** token 嵌入 V·d，位置嵌入 P·d。

**每个块：**

| 模块 | 权重 | 偏置 | 小计 |
|---|---|---|---|
| ln_1（γ 和 β） | d | d | 2d |
| c_attn（q、k、v） | d · 3d | 3d | 3d² + 3d |
| attn.c_proj | d · d | d | d² + d |
| ln_2 | d | d | 2d |
| c_fc | d · 4d | 4d | 4d² + 4d |
| mlp.c_proj | 4d · d | d | 4d² + d |
| **合计** | | | **12d² + 13d** |

**最后的 ln_f：** 2d。

**输出层：** 和 wte 共享，0。

所以

```text
N = V·d + P·d + L·(12d² + 13d) + 2d          （数学结论）
```

和第 11 章的公式 `N = V·d + L·(4d² + 3df + 2d) + d` 对比一下：注意力都是 4d² 的权重；你的 SwiGLU 有三个矩阵 3df，GPT-2 的 MLP 只有两个，但中间宽是 4d，所以 2 · d · 4d = 8d²；RMSNorm 每个只有 d 个参数，LayerNorm 有 2d。每块的 13d 里，9d 是投影偏置，4d 是两个 LayerNorm 的 γ 和 β；和你的 GPT（两个 RMSNorm，共 2d）相比，每块多出 11d，最后的归一化多出 d，另外还多了位置嵌入 P·d。

### 17.2.2 代入数字

V = 50257，P = 1024，d = 768，L = 12（这些数从 config.json 读出，17.3 节）。

```text
V·d             = 50257 × 768                 = 38,597,376
P·d             = 1024 × 768                  =    786,432
12d²            = 12 × 589,824                =  7,077,888
13d             = 13 × 768                    =      9,984
每块            = 7,077,888 + 9,984           =  7,087,872
L 块            = 12 × 7,087,872              = 85,054,464
ln_f            = 2 × 768                     =      1,536
总计            = 38,597,376 + 786,432 + 85,054,464 + 1,536
                = 124,439,808                                （数学结论）
```

tour 第 1 部分把这个公式、本章代码的统计、transformers 的统计三者对拍，结果一致（本项目观察）。

两个值得记住的比例：

- 嵌入占 38,597,376 / 124,439,808 ≈ 31.0%（数学结论）。小模型的大词表很"贵"。
- 偏置和 LayerNorm 参数一共 L·13d + 2d = 121,344，不到 0.1%（数学结论）。它们对参数量几乎没影响，可是少了任何一个，对拍时输出就对不上。

GPT-2 的技术报告把最小的模型写作 117M，代码统计出来是 124,439,808。差异的来源报告里没有解释（待验证）。所以本章的参数量以代码统计为准。

### 17.2.3 占多少内存

fp32 每个参数 4 字节：124,439,808 × 4 = 497,759,232 字节，约 0.50 GB（数学结论）。

可是本地的 `model.safetensors` 有 548,105,171 字节（本项目观察：本地文件大小）。多出来的 50,345,939 字节是什么？官方文件里每层还存了一个名为 `h.{i}.attn.bias` 的张量，形状 (1, 1, 1024, 1024)——那是旧代码保存的**因果 mask 缓冲区**，不是参数。12 层 × 1024 × 1024 × 4 字节 = 50,331,648 字节，剩下的 14,291 字节是文件头（8 字节的长度加 14,283 字节的 JSON 头）（数学结论 + 本项目观察：读 safetensors 文件头）。转换权重时要跳过它们（17.6 节）。

## 17.3 读懂 config.json

模型目录里的 `config.json` 是模型的"身份证"：权重文件只有一堆张量，形状和名字之外的一切——有几层、几个头、用什么激活函数——都写在这里。tour 第 2 部分逐字段打印本地文件的值和解释。按用途分成四类：

**结构字段**（决定张量形状和计算方式，错一个就加载失败或者结果全错）：

| 字段 | 值 | 含义 |
|---|---|---|
| `vocab_size` | 50257 | 词表大小 V |
| `n_positions` | 1024 | 位置嵌入表的行数 P，也是最长序列 |
| `n_embd` | 768 | 宽度 d |
| `n_layer` | 12 | 块数 L |
| `n_head` | 12 | 注意力头数；head_dim = 768 / 12 = 64 |
| `activation_function` | `"gelu_new"` | MLP 激活：tanh 近似的 GELU（17.4 节） |
| `layer_norm_epsilon` | 1e-05 | LayerNorm 分母里的小常数 |

还有一个 `n_ctx`，值也是 1024。这是历史遗留字段，和 `n_positions` 相同；位置表的大小以 `n_positions` 为准。

**训练字段**（只在训练时起作用）：

- `attn_pdrop`、`embd_pdrop`、`resid_pdrop` = 0.1：三处 dropout 的比率。`model.eval()` 后全部关闭。对拍时一定要先 `eval()`，否则每次输出都不一样。
- `initializer_range` = 0.02：随机初始化的标准差，和第 11 章一样。加载了预训练权重，它就不起作用了。

**生成和特殊 token：**

- `bos_token_id` = `eos_token_id` = 50256：GPT-2 只有一个特殊 token `<|endoftext|>`，同时当开头和结尾用。
- `task_specific_params`：`{"text-generation": {"do_sample": true, "max_length": 50}}`，是给 `pipeline` 的默认建议。它不影响权重，tour 里显式指定贪心解码覆盖它。

**元数据和其他头：**

- `architectures` = `["GPT2LMHeadModel"]`、`model_type` = `"gpt2"`：告诉 `AutoModel` 该用哪个类。
- `summary_*` 五个字段：给序列分类头（`GPT2DoubleHeadsModel` 等）用的，语言模型头不用。

读 config 时，结构字段一个个核对，其他字段知道是干什么的就行。Qwen3 的 config 里有 `num_key_value_heads`、`rope_theta`、`tie_word_embeddings` 这些新字段，取值列在 17.10 节的对照表里；第 20 章 20.7 节算 LoRA 的参数量时，会从 config 读 Qwen3-0.6B 的形状。

## 17.4 GPT-2 和你的 GPT：同与不同

### 17.4.1 一样的地方

先说一样的：

- **解码器式 Transformer**：因果自注意力，只能看前面的 token。
- **Pre-Norm 残差块**：GPT-2 报告第 2.3 节说，LayerNorm 被移到每个子块的输入处，并在最后一个块之后再加一个 LayerNorm。这正是第 9 章的 Pre-Norm 结构。初代 GPT 用的是 Post-Norm，GPT-2 改成了 Pre-Norm，从此几乎所有大模型都沿用。
- **权重共享**：`lm_head` 和 `wte` 是同一个矩阵。
- **残差缩放初始化**：报告第 2.3 节说残差层的权重按 1/√N 缩放，N 是残差层数。第 11 章的 0.02/√(2L) 就是这个想法（每块两个残差支路，N = 2L）。

### 17.4.2 不一样的五处

| | GPT-2 | 你的 GPT（第 11 章） |
|---|---|---|
| 位置信息 | 学习式位置嵌入 `wpe`，加在输入上 | RoPE，转进每层的 q、k |
| 归一化 | LayerNorm（减均值，有 γ 和 β） | RMSNorm（不减均值，只有 γ） |
| 前馈 | `fc → GELU(tanh 近似) → proj`，中间宽 4d | SwiGLU，三个矩阵 |
| 偏置 | 所有投影都有偏置（输出层除外） | 全部没有 |
| 权重存放 | `Conv1D`，形状 (in, out) | `nn.Linear`，形状 (out, in) |

逐个说。

**学习式位置嵌入。** 位置 t 的信息就是 `wpe` 表的第 t 行，一个可训练的 768 维向量，在输入处和 token 嵌入相加：`x = wte[idx] + wpe[0..T-1]`。简单直接，代价是表只有 1024 行：第 1025 个位置没有对应的行，模型根本无法处理（17.8.4 节）。RoPE 没有表，位置是算出来的角度，至少"能算"任意位置（算得好不好是另一回事，第 10 章讨论过）。

**LayerNorm。** 对每个 token 的 768 维向量：

```text
LayerNorm(x) = (x − mean(x)) / √(var(x) + ε) · γ + β
RMSNorm(x)   =  x / √(mean(x²) + ε) · γ
```

LayerNorm 多减一个均值、多加一个偏移 β。PyTorch 有现成的 `nn.LayerNorm`，`eps` 要设成 config 里的 1e-5（PyTorch 默认值也是 1e-5，但不要依赖默认值）。

**GELU 的 tanh 近似。** GELU 的精确定义用到误差函数 erf：

```text
GELU(x)     = x/2 · (1 + erf(x / √2))
gelu_new(x) = x/2 · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
```

第二行是一个近似，计算上曾经比 erf 便宜。GPT-2 训练时用的是这个近似，所以加载它的权重也必须用这个近似。两者差多少？tour 第 10 部分在 [−5, 5] 上取 10001 个点比较，最大差约 4.7 × 10⁻⁴，出现在 x ≈ −2.7 附近（本项目观察：CPU、float64 网格，tour 第 10 部分）。看着很小，17.8.3 节会看到它对输出的影响。transformers 把这个近似叫 `gelu_new`，把精确式叫 `gelu`——名字很容易搞混。

**偏置。** GPT-2 每个投影都是 `y = xW + b`。偏置参数不多（17.2 节），但一个都不能漏。现代模型大多去掉了偏置（17.10 节）。

**Conv1D。** 存放方向的问题单独放到 17.6 节讲。

> **停下来想一想：** GPT-2 的 MLP 里没有门控。它的中间宽是 4d = 3072，参数 8d²。如果你的 SwiGLU 也想要 8d² 的参数，中间宽 f 应该取多少？
> （答：3df = 8d²，f = 8d/3 ≈ 2048。这就是第 9 章"≈ 8/3 · d"的来历。数学结论。）

## 17.5 从零写 GPT-2 前向

### 17.5.1 为什么要自己写

transformers 已经有 GPT-2 了，为什么要重写一遍？因为你以为自己懂了一个模型，写一遍、和官方输出比一下，没懂的细节就会变成数字上的差异暴露出来。

代码在 `miniocr/models/gpt2.py`，不 import transformers 的任何建模代码。我们故意让模块名和 HF 不同（q、k、v 分开，叫 `fc`、`proj`），这样转换函数（17.6 节）就必须把每一个张量都显式对上——不能偷懒靠"名字一样就自动加载"。

### 17.5.2 配置

```python
@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    layer_norm_epsilon: float = 1e-5
    activation: str = "gelu_new"
```

默认值就是 GPT-2 small。`GPT2Config.from_hf_dict(d)` 从 config.json 的字典构造，并且对它不支持的变体（例如 `n_inner` 不等于 4d、`scale_attn_by_inverse_layer_idx=True`）直接报错。宁可报错，也不要静默地按另一种结构计算。

### 17.5.3 注意力

```python
class GPT2Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head, self.head_dim = cfg.n_head, cfg.n_embd // cfg.n_head
        self.q = nn.Linear(cfg.n_embd, cfg.n_embd)        # 默认 bias=True
        self.k = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.v = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x):                                                     # (1)
        B, T, d = x.shape
        q, k, v = [layer(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                   for layer in (self.q, self.k, self.v)]                     # (2)
        scores = (q @ k.transpose(-1, -2)) * (self.head_dim ** -0.5)          # (3)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)     # (4)
        y = scores.softmax(dim=-1) @ v                                        # (5)
        return self.proj(y.transpose(1, 2).reshape(B, T, d))                  # (6)
```

用 GPT-2 的真实尺寸跟一遍形状（B = 1，T = 10）：

- **(1)** x 形状 (1, 10, 768)。
- **(2)** 每个投影输出 (1, 10, 768)，拆成 12 个头 (1, 10, 12, 64)，换轴成 (1, 12, 10, 64)。
- **(3)** 分数 (1, 12, 10, 10)，乘 1/√64 = 1/8。
- **(4)** 因果 mask：上三角（看未来的位置）填成 fp32 能表示的最小数。HF 的 eager 实现也是这样填的，不是填 −∞。两种填法 softmax 后都是 0，但用同一种填法，对拍时少一个差异来源。
- **(5)** softmax 后乘 v，(1, 12, 10, 64)。
- **(6)** 合并头 (1, 10, 768)，过输出投影。

这和第 8 章的多头注意力是同一个东西，只是投影带偏置、不加 RoPE。

### 17.5.4 MLP、块和整个模型

```python
class GPT2MLP(nn.Module):
    def forward(self, x):
        x = self.fc(x)                                         # (B, T, 768) → (B, T, 3072)
        return self.proj(gelu_new(x) if self.activation == "gelu_new" else F.gelu(x))

class GPT2Block(nn.Module):
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))                        # Pre-LN，和第 9 章一样
        return x + self.mlp(self.ln_2(x))
```

整个模型的前向：

```python
def forward(self, idx, return_hidden=False):
    # （省略：检查 idx 是非空的二维张量）
    T = idx.shape[1]
    if T > self.cfg.n_positions:                                              # (1)
        raise ValueError(f"序列长度 {T} 超过最大位置数 n_positions={self.cfg.n_positions}；请缩短输入")
    x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))          # (2)
    hidden = []
    for block in self.blocks:
        if return_hidden:
            hidden.append(x)                                                  # (3)
        x = block(x)
    x = self.ln_f(x)
    logits = self.head(x)                                                     # (4)
    return (logits, tuple(hidden + [x])) if return_hidden else logits
```

- **(1)** 先检查长度。如果不检查，越界的位置会在 `wpe` 查表时报一个 `IndexError: index out of range in self`——这条信息完全不提"位置"二字，很难一眼看懂（17.8.4 节）。
- **(2)** 学习式位置嵌入：位置 0..T−1 各查一行，和 token 嵌入相加。形状 (1, T, 768) + (T, 768)，广播相加。
- **(3)** `return_hidden=True` 时收集每个块的输入，用来逐层定位误差（17.7.4 节）。口径和 HF 的 `output_hidden_states` 对齐：共 L + 1 项，前 L 项是各块的输入，最后一项是**经过 ln_f 之后**的输出。
- **(4)** 输出层。构造函数里 `self.head.weight = self.wte.weight`，两者是同一个 Parameter。

## 17.6 权重转换：Conv1D 转置与 qkv 切分

### 17.6.1 一堆按名字存放的张量

官方 `model.safetensors` 里有 160 个张量（本项目观察：读文件头）：4 个全局的（`wte.weight`、`wpe.weight`、`ln_f.weight`、`ln_f.bias`），加上 12 层 × 每层 13 个。每层的 13 个里有 12 个是参数，1 个是 17.2.3 节说的 mask 缓冲区 `attn.bias`。tour 第 3 部分直接读文件头核对了这些数，并验证了 17.2.3 节的字节账。文件里**没有** `lm_head.weight`（它和 wte 共享，不用存）。

我们的模型有自己的一套名字。转换函数的任务就是：对每个我们需要的张量，找到它在官方文件里的来源，做必要的变形，放到正确的名字下。

### 17.6.2 Conv1D 只是"反着存"的线性层

线性层 y = x W + b 有两种存法，就像一张表格可以按行存也可以按列存：

- `nn.Linear(in, out)` 存 W 的转置，形状 **(out, in)**，计算 `y = x @ W.T + b`。
- HF 的 `Conv1D(nf=out, nx=in)` 直接存 W，形状 **(in, out)**，计算 `y = x @ W + b`。

两者算的是同一个东西，只是矩阵放倒了。名字叫 Conv1D，是因为 OpenAI 最初的 TensorFlow 代码里用了一个叫 `conv1d` 的函数来实现这个线性映射（核宽为 1 的一维卷积在数学上就是对每个位置做同一个线性变换）。HF 为了直接加载 OpenAI 的原始权重，保留了这个存放方向。

所以转换规则很简单：HF 的 Conv1D 权重，转置一下才能放进 nn.Linear。偏置是一维的，不用转置。

### 17.6.3 qkv 拼接沿哪一维？

`c_attn` 把 q、k、v 三个投影合成一个 Conv1D，权重形状 (768, 2304)，偏置 (2304,)。2304 = 3 × 768，是输出维。

HF 的计算是：

```text
qkv = x @ W_c + b_c                      # (B, T, 768) @ (768, 2304) → (B, T, 2304)
q, k, v = qkv.split(768, dim=2)          # 各 (B, T, 768)，顺序 q、k、v
```

矩阵乘法的第 j 个输出只依赖 W_c 的第 j 列。所以"先乘再沿输出切"等于"先把 W_c 沿列切，再分别乘"：

```text
q = x @ W_c[:, 0:768]     + b_c[0:768]
k = x @ W_c[:, 768:1536]  + b_c[768:1536]
v = x @ W_c[:, 1536:2304] + b_c[1536:2304]
```

再套用 17.6.2 的规则：每一片是 (in, out) = (768, 768)，转置后放进 `nn.Linear`。

结论：**沿最后一维（输出维）切成三片，顺序 q、k、v，每片权重转置，偏置只切不转置**。

### 17.6.4 带数字的小例子

取 d = 2，x = [1, 1]：

```text
W_c = [[1, 2,  3,  4,  5,  6],          形状 (in=2, out=6)
       [7, 8,  9, 10, 11, 12]]
b_c = 0
```

HF 的算法：x @ W_c 是 W_c 每列求和 = [8, 10, 12, 14, 16, 18]，切开得 q = [8, 10]，k = [12, 14]，v = [16, 18]。

我们的算法：q 片是 W_c[:, 0:2] = [[1, 2], [7, 8]]，转置成 [[1, 7], [2, 8]] 放进 Linear；Linear 计算 W x = [1·1 + 7·1, 2·1 + 8·1] = [8, 10]。和 HF 一致。

再看两种错误：

- **忘了转置**：Linear 拿到 [[1, 2], [7, 8]]，计算 W x = [1 + 2, 7 + 8] = [3, 15]。错了，**但形状完全正确**——这一片是 2 × 2 的方阵，转不转置形状都一样。
- **切分顺序错**（当成 k、q、v）：q 拿到第二片，算出 [12, 14]。也错了，形状也完全正确。

这两种错误都不会报错，所以必须对拍。

### 17.6.5 代码逐行

```python
def convert_hf_gpt2_state_dict(hf_sd, cfg, bug=None):
    src = {k.removeprefix("transformer."): v for k, v in hf_sd.items()}                  # (1)
    out = {k: src[k] for k in ("wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias")}  # (2)
    if "lm_head.weight" in src and not torch.equal(src["lm_head.weight"], src["wte.weight"]):
        raise ValueError("lm_head.weight 与 wte.weight 不一致，无法使用共享权重")           # (3)
    out["head.weight"] = out["wte.weight"]
    for i in range(cfg.n_layer):
        s, t = f"h.{i}.", f"blocks.{i}."
        for name in ("ln_1", "ln_2"):                                                    # (4)
            for suffix in ("weight", "bias"):
                out[t+name+"."+suffix] = src[s+name+"."+suffix]
        order = ("k", "q", "v") if bug == "qkv_order" else ("q", "k", "v")
        for suffix in ("weight", "bias"):                                                # (5)
            chunks = src[s+"attn.c_attn."+suffix].split(cfg.n_embd, dim=-1)
            for name, chunk in zip(order, chunks):
                out[t+"attn."+name+"."+suffix] = chunk.T if suffix == "weight" else chunk
        for hf_name, own in (("attn.c_proj", "attn.proj"), ("mlp.c_fc", "mlp.fc"),
                             ("mlp.c_proj", "mlp.proj")):                                # (6)
            w = src[s+hf_name+".weight"]
            out[t+own+".weight"] = w if bug == "no_transpose" and w.shape[0] == w.shape[1] else w.T
            out[t+own+".bias"] = src[s+hf_name+".bias"]
    return out
```

（为了阅读，这里省略了参数检查。）

- **(1)** 去掉 `transformer.` 前缀。`GPT2LMHeadModel.state_dict()` 的键带前缀，safetensors 文件里的键不带，两种来源都能接受。
- **(2)** 嵌入和最后的归一化：照搬。
- **(3)** 输出层：如果来源里有 `lm_head.weight`，先检查它确实和 wte 相等，然后直接用 wte。不相等说明这不是一个共享权重的模型，必须报错而不是悄悄丢掉一个矩阵。
- **(4)** 两个 LayerNorm 的 γ（`weight`）和 β（`bias`）：照搬。
- **(5)** c_attn：沿最后一维（`dim=-1`）每 768 切一片，按 q、k、v 的顺序分配；权重转置，偏置不转置。注意权重的最后一维是输出维 2304，偏置只有一维，`dim=-1` 对两者都对。
- **(6)** 其余三个 Conv1D：权重转置，偏置照搬。`bug` 参数只给故障演练用（17.8 节），正常调用时是 `None`。

mask 缓冲区 `h.{i}.attn.bias` 呢？转换函数只取自己需要的键，它自然被跳过了。加载时用 `strict=True`：我们模型的每一个参数都必须在转换结果里有值，多一个少一个都报错。tour 第 3 部分把第 0 层的每个张量列成表：HF 名字和形状 → 我们的名字和形状 → 做了什么操作（照搬 / 切分 / 转置 / 共享）。

### 17.6.6 strict=True 能抓住什么，抓不住什么

`load_state_dict(..., strict=True)` 检查两件事：名字一一对应，形状一一相同。它**不检查数值放得对不对**。

- `c_fc` 忘了转置：(768, 3072) 放进 (3072, 768) 的位置，形状不符，立即报错。好。
- `c_attn` 合在一起忘了转置：(768, 2304) 对 (2304, 768)，报错。好。
- `attn.c_proj` 忘了转置：(768, 768) 对 (768, 768)，**静默通过**。
- q、k、v 顺序错：三片都是 (768, 768)，**静默通过**。

strict 加载只是第一道关，静默通过的错误要靠第二道关：数值对拍。

## 17.7 对拍：方法与容差

### 17.7.1 怎样对拍

对拍就是：同一个输入，两个实现，逐个数字比较输出。要比得公平，先把所有"随机"和"不同"的来源关掉：

1. **两边都 `eval()`**：关掉 dropout。
2. **同一个 dtype**：都用 fp32。bf16 的舍入误差会大到掩盖真正的 bug。
3. **同一台设备**：都在 CPU 上。不同设备的矩阵乘法实现不同，误差的量级也不同。
4. **同一个注意力实现**：HF 用 `attn_implementation="eager"`，和我们一样是显式矩阵乘。用 SDPA 也能对上，但多了一个差异来源。
5. **同一串 token**：用同一个 tokenizer 编码，不加特殊 token。

然后比两样东西：

- **logits 的最大绝对误差** `(ours − hf).abs().max()`。这是最严格的指标。
- **下一 token 的 top-1 / top-k 是否一致，贪心续写是否一致**。这是最终用户看得到的行为。

### 17.7.2 结果

tour 第 4 部分对两段文本做了对拍（本项目观察：CPU、2 线程、fp32、HF eager，tour 第 4 部分）：

| 输入 | logits 最大绝对误差 | 最大 \|logit\| | 误差 / 最大 \|logit\| |
|---|---|---|---|
| `The quick brown fox jumps over the lazy dog.` | 7.63 × 10⁻⁵ | 155.3 | 4.9 × 10⁻⁷ |
| 含 LaTeX 的英文句子 | 0 | 143.7 | 0 |

第二段的误差恰好是 0：在这个输入上，两边的输出逐位相等。第一段不是 0，但很小。不要以为"对拍成功就应该是 0"——换一个长度、换一台机器，矩阵乘法内部的求和顺序就可能不同。

tour 第 5 部分看行为（本项目观察：同上，tour 第 5 部分）：下一 token 的 top-5 编号完全一致，概率最大差 7.7 × 10⁻⁷；贪心续写 20 个 token 逐个一致。

### 17.7.3 容差怎样定

7.63 × 10⁻⁵ 算不算"一致"？要和 fp32 本身的精度比。

fp32 有 23 位尾数，相对精度 2⁻²³ ≈ 1.19 × 10⁻⁷（数学结论，第 16 章）。一个大小在 128 到 256 之间的数，相邻两个 fp32 数的间隔是 2⁷ × 2⁻²³ = 2⁻¹⁶ ≈ 1.53 × 10⁻⁵（数学结论）。最大 |logit| 是 155，落在这个区间；误差 7.63 × 10⁻⁵ 恰好是 5 × 2⁻¹⁶，相当于最大 logit 那个量级上的 5 个最小间隔（数学结论 + 本项目观察）。（最大误差不一定出现在最大的那个 logit 上；在绝对值更小的 logit 处，间隔更小，同样的误差折合的间隔数更多。）一个 12 层、上亿次乘加的计算，误差只有这个量级，和"两边算的是同一个函数、只是求和顺序不同"的解释相符。

在本章的设置下（CPU、fp32、这几段输入），可以这样对照：正确转换的相对量级是 10⁻⁷（17.7.2 节），而 17.8 节的几种故障从 10⁻⁴（GELU 版本）到接近 1（转置、qkv 顺序）。误差大小只是线索，不能单独证明原因——一个极小的实现差异也可能只造成很小的误差，所以出现可疑的误差时，还是要逐层定位。本章的测试对随机初始化的小模型用 atol = 1e-5，对官方权重用 atol = 2e-4（logits 和每层 hidden）；这些阈值只对 CPU fp32 有意义，换到 GPU 或 bf16 要重新标定。

### 17.7.4 逐层定位

logits 对不上时，只知道"错了"，不知道错在哪。办法是**逐层比较残差流**。

我们的模型和 HF 都能返回每一层的 hidden state（17.5.4 节），共 L + 1 = 13 个边界：边界 0 是嵌入相加的结果，边界 1..11 是第 0..10 块的输出，边界 12 是最后一块经过 ln_f 的输出。从边界 0 往后逐个比，找到**第一个超过容差的边界**，就先检查它前面的那个块（边界 12 还包括 ln_f）。注意这是"第一个看得出来的地方"：一个很小的误差可能在更早的块就引入了，累积几层后才越过容差；如果这个块查不出问题，就往前回溯更早的块，或者把容差调小再比一次。

正确转换时，13 个边界的最大误差都在 10⁻⁵ 量级，最大的一个是 9.2 × 10⁻⁵（本项目观察：英文句子，tour 第 4 部分），没有一个超过 tour 用的 10⁻⁴ 容差。17.8 节会看到出错时这张表长什么样。

这个方法和调试任何程序一样：沿着计算顺序逐段比较，找第一个出错的地方。第 32 章不复刻 VLM 的整个前向，所以不做这样的逐层数值比较：它把 processor 和官方实现逐位对拍（32.3 节），模型内部则用 forward hook 逐段记录形状（32.4.4 节）。

## 17.8 故障演练

每个故障都用 `convert_hf_gpt2_state_dict(..., bug=...)` 或改一个开关注入，然后跑 17.7 节的对拍。看现象、找原因，比读一百遍"注意转置"更管用。

### 17.8.1 忘记转置 Conv1D

tour 第 8 部分先演示会报错的情况：`c_attn`、`c_fc`、`mlp.c_proj` 不转置，`load_state_dict` 直接抛出 `size mismatch`。这种错误一眼就能发现。

然后是静默的情况：只让方阵 `attn.c_proj`（768 × 768）不转置，其他都正确。加载成功，没有任何警告。对拍结果（本项目观察：英文句子，tour 第 8 部分）：

- logits 最大误差 137.5，而最大 |logit| 才 155.3——误差和信号本身一样大。
- 下一 token 的 top-1 变了。
- 逐层误差：边界 0 是 0（嵌入没问题），边界 1（第 0 块输出）就到了 160.6，之后迅速增大，在边界 9 达到约 2595，随后略降；经过 ln_f 的边界 12 约为 210。

**怎样发现：** 第一个超差的是边界 1，所以错误在第 0 块里。块里有 6 个模块，逐个检查转换就能找到。如果你只看续写文本，它可能"像英文"但语义混乱；只有对拍能给出确定的结论。

### 17.8.2 qkv 切分顺序错

tour 第 9 部分把切分顺序当成 k、q、v。所有形状都对，strict 加载通过。对拍结果（本项目观察：英文句子，tour 第 9 部分）：logits 最大误差 114.6，top-1 变了；第一个超差的边界同样是 1，这一层误差 5.75，到边界 3 时已经 528。

为什么 q、k 互换不是"无害地换个名字"？注意力分数 q_i · k_j 表示"位置 i 去看位置 j"。q、k 互换后变成 k_i · q_j，相当于把分数矩阵转置了——"i 看 j"变成"j 看 i"。再加上因果 mask 只保留下三角，转置后保留下来的是完全不同的一组数。

**一个陷阱**：如果测试只用 1 个 token，注意力矩阵是 1 × 1，softmax 永远是 1，q、k 怎么换都看不出来（v 换了还是看得出）。所以测试至少要用几个 token。

### 17.8.3 GELU 用了精确式

这个错误最隐蔽。tour 第 10 部分把 MLP 的激活从 tanh 近似换成精确的 erf 版本，其余全对（本项目观察：英文句子，tour 第 10 部分）：

- logits 最大误差 0.075，相对最大 |logit| 是 4.8 × 10⁻⁴。
- 第一个超差的边界是 1，误差 0.011。
- **top-1 没变，20 个 token 的贪心续写也完全一样。**

如果你只看生成的文本，会以为一切正常。但 4.8 × 10⁻⁴ 比正确转换的 4.9 × 10⁻⁷ 大了约一千倍，远在浮点噪声之上。它在这一段输入上没改变贪心结果，换一段输入、换成采样、或者继续微调，影响都可能显现出来。

所以对拍要看 logits 的数值误差，不能只看文本是否一样。文本一致不能证明数值实现正确；反过来，如果前两名候选的 logit 非常接近，正确实现的浮点噪声也可能翻转 argmax、让文本不同，这时同样要回到 logits 误差和候选之间的差距来判断。

### 17.8.4 位置越界

GPT-2 的位置表只有 1024 行（`n_positions` = 1024）。tour 第 11 部分（本项目观察：tour 第 11 部分）：

- 长度 1024：正常，logits 形状 (1, 1024, 50257)，全部有限。
- 长度 1025，我们的模型：抛出 `ValueError: 序列长度 1025 超过最大位置数 n_positions=1024；请缩短输入`。
- 长度 1025，HF 模型：抛出 `IndexError: index out of range in self`（transformers 5.19.0 的行为；不同版本可能不同）。

HF 的错误信息不提位置，第一次遇到很难联想到"序列太长"。所以我们在前向的第一行就检查长度，给出能看懂的信息。

能不能把位置编号取模、或者截断来"绕过"？截断可以：只保留最近的 token，使输入加上要生成的 token 不超过 1024（本章的 `generate_greedy` 不做滑窗，要生成 n_new 个 token，输入最多保留最后 1024 − n_new 个）。代价是模型看不到更早的内容了。取模让位置 1024 重用位置 0 的嵌入，模型训练时从没见过这种情况，输出不可信。学习式位置嵌入的上下文长度是**硬上限**；这是现代模型换成 RoPE 的原因之一（17.10 节）。

## 17.9 GPT-2 的分词器：字节级 BPE

GPT-2 的分词器和第 6 章的 BPE 核心算法相同：从字节出发，反复合并最常见的相邻对。区别在三处：规模、训练数据和预分词规则。GPT-2 在英文网页文本上合并了 50,000 次，词表 50,257 = 256 个字节 + 50,000 个合并 + 1 个 `<|endoftext|>`（数学结论）。预分词方面，第 6 章的规则把数字逐位切开，GPT-2 允许连续数字成为一个词块（例如 `12345` 在第 6 章的预分词下是 5 块；在 GPT-2 里是一整块，BPE 再把它切成 `123`、`45` 两个 token——本项目观察，本地 GPT-2 tokenizer），这会改变允许合并的范围。

tour 第 6 部分把它和第 6 章的 `train_bpe` 现场训练的小分词器对比。小分词器只在 256 条 LaTeX 公式上合并 256 次，词表 512（本项目观察：本地语料的 standard_split 训练集前缀，tour 第 6 部分）。规模差了两个数量级，看几个例子（本项目观察：tour 第 6 部分）：

| 文本 | GPT-2 token 数 | 第 6 章小 BPE token 数 |
|---|---|---|
| `hello world` | 2 | 10 |
| `\frac{a+b}{\sqrt{x^2+1}}` | 17 | 15 |
| `\alpha+\beta=\gamma` | 9 | 6 |
| `\text{面积}=ab` | 12 | 12 |
| `模型读入图像，写出文字。` | 27 | 36 |

几个值得看的细节：

- **`Ġ` 是空格。** GPT-2 的词表显示 `hello world` 切成 `['hello', 'Ġworld']`。字节级 BPE 要把 256 个字节都映射成可打印字符，空格（字节 32）被映射成了 `Ġ`。它不是原文里的字母。
- **LaTeX 命令被切碎。** GPT-2 把 `\sqrt` 切成 `\`、`sq`、`rt`，把 `\gamma` 切成 `\`、`gam`、`ma`——它的训练数据里 LaTeX 不多。只合并了 256 次的小 BPE 反而把 `sqrt`、`gamma` 合成了整块，因为它只见过 LaTeX。
- **中文按字节切。** 常用汉字在 UTF-8 里是 3 个字节（本例的字都是；少数扩展区汉字是 4 个字节）。"模"在 GPT-2 里被切成 3 个 token，每个 token 是一个字节，单独解码会显示成乱码字符。12 个汉字和标点共 36 字节，GPT-2 用了 27 个 token，平均每个 token 1.3 字节（本项目观察 + 数学结论）。小 BPE 没见过中文，36 个字节就是 36 个 token。

这对后纠错任务意味着什么？token 越多，序列越长、算得越慢，一个公式要占的上下文也越多。这是第 18 章起换成 Qwen3 的原因之一：它的词表有约 15 万个 token（17.10 节）。在本书的样本上，Qwen3-0.6B 的 token 数和 GPT-2 相比：本书正文约为 49%，`miniocr` 的 Python 代码约为 51%，验证集前 2000 条 LaTeX 公式约为 88%，书里的 ASCII 英文行约为 101%（本项目观察，`outputs/book/ch17/tokenizer_compare/`）。中文样本是本书正文，去掉了代码块和行内公式，仍含 Markdown、英文和表格，不是通用评测。这些比例只说明这几份样本，不能推广到所有语言。

## 17.10 从 GPT-2 到 Qwen3（概括）

从 2019 年的 GPT-2 到 2025 年的 Qwen3，骨架没变：还是解码器式 Transformer，还是 Pre-Norm 残差块。变的是块里的零件。tour 第 7 部分从本地 config 读出数值，列成下表（本项目观察：tour 第 7 部分）。

关于 Qwen3 一列的来源：tour 第 7 部分直接读 Qwen3-0.6B 的本地 config 和 safetensors 文件头里的键名和形状，不加载张量（本项目观察）。Q/K norm 在 config 里没有字段，是从键名（有没有 `q_norm`、`k_norm`）读出来的。"权重共享"的依据是 config 的 `tie_word_embeddings`：权重文件里同时存了 `lm_head.weight` 和 `model.embed_tokens.weight`，只看键名和形状看不出两者是否共享。键名和形状也不能证明前向的运算顺序、mask 的实现这类运行时行为。这张表只比较配置字段和权重键名，不逐行核对 Qwen 的建模源码，`q_proj` 这类模块名和 mask 的写法以你安装的 transformers 版本为准。

| 项目 | GPT-2 | 你的 GPT（默认配置） | Qwen3-0.6B |
|---|---|---|---|
| 位置编码 | 学习式绝对位置 | RoPE | RoPE，rope_theta = 1,000,000 |
| 归一化 | LayerNorm，eps 1e-5 | RMSNorm，eps 1e-6 | RMSNorm，eps 1e-6 |
| MLP | GELU(tanh)，中间宽 3072 | SwiGLU，中间宽 704 | SwiGLU（SiLU 门控），中间宽 3072 |
| 注意力头 | MHA，12 个头 | MHA，4 个头 | GQA，16 个 q 头、8 个 kv 头 |
| head_dim | 64 | 64 | 128 |
| Q/K norm | 无 | 无（可选） | 有 |
| 线性层偏置 | 有 | 无 | 无 |
| 权重共享 | 是 | 是 | 是 |
| 词表大小 | 50,257 | 2,048 | 151,936 |
| 最大位置 | 1,024 | 256 | 40,960 |
| 层数 / 宽度 | 12 / 768 | 4 / 256 | 28 / 1024 |

你的 GPT 已经有了这张表里的大部分现代零件：第 9–11 章写的就是 RoPE、RMSNorm、SwiGLU、无偏置，`GPTConfig` 里还留了 `n_kv_head`（GQA）和 `qk_norm` 两个开关。所以从你的 GPT 走到 Qwen3，比从 GPT-2 走到 Qwen3 近得多。

每项改动的理由：

- **RoPE 取代学习式位置**（第 10 章）：位置信息以相对的方式进入注意力分数，没有一张固定行数的表。把 rope_theta 调大（Qwen3 是 10⁶），配合长文本训练，就能把上下文拉长到几万。
- **RMSNorm 取代 LayerNorm**（第 9 章）：少算一个均值、少一个 β，更省；实践中效果相当。
- **SwiGLU 取代 GELU MLP**（第 9 章）：门控让 MLP 在相同参数量下效果更好（Shazeer 2020 的实验）。中间宽相应调小，保持参数量不变（17.4 节的"停下来想一想"）。不过各家的取法不同：Qwen3 的 config 用 `intermediate_size` 直接写明中间宽 3072，是宽度 1024 的 3 倍，不等于 8/3 · d（本项目观察：本地 config）。
- **GQA**（第 15 章）：几个 q 头共用一组 k、v。KV cache 按 kv 头数算，Qwen3 的 8 个 kv 头让缓存减半，推理时显存和带宽都省。
- **Q/K norm**：在算注意力分数之前，对每个头的 q、k 做 RMSNorm。Qwen3 技术报告说引入它是为了保证训练稳定。直观的机制是：归一化限制了 q、k 的尺度，注意力分数就不容易变得过大（这是机制解释，不是报告原话）。
- **去掉偏置**：偏置参数少、收益小，去掉后结构更简单。Qwen3 技术报告提到它去掉了 Qwen2 的 QKV 偏置。
- **更大的词表**：在 17.9 节的样本上，中文和代码的 token 数约减半，公式少得多一些，英文基本不变。代价是嵌入矩阵变大：151,936 × 1024 = 155,582,464，约 1.56 亿参数（数学结论），占了 0.6B 模型的很大一块。
- **head_dim 与宽度解耦**：Qwen3 的 head_dim 是 128，16 个头的 q 一共 2048 维，比宽度 1024 还宽。所以它的 config 里显式写了 `head_dim`，不能再用 hidden_size / num_attention_heads 推出来。

这一节只列差异，不展开。第 18 章开始实际使用 Qwen3-0.6B。后面的章节不再逐项读它的 config、手算全部参数，只在用到时引用实测值（第 19 章的词表和总参数量、第 20 章的可训练参数量）；想练手，可以用本章的方法自己读一遍。

## 常见错误

| 错误 | 现象 | 怎样发现 |
|---|---|---|
| Conv1D 权重忘了转置 | 非方阵：加载时报 size mismatch；方阵（attn.c_proj）：静默加载，输出胡话 | strict 加载抓非方阵；logits 对拍 + 逐层比较抓方阵（17.8.1） |
| c_attn 沿错误的维切分 | 切出来的形状不对，或者形状对但数值错 | 记住 Conv1D 的输出维是最后一维；逐层比较边界 1 |
| q、k、v 顺序错 | 静默加载，输出胡话 | 逐层比较，第一个超差的是边界 1；测试用多个 token（17.8.2） |
| 用精确 GELU 代替 tanh 近似 | 文本看起来正常，logits 误差约 0.07（本项目观察，tour 第 10 部分） | 看 logits 最大误差，不只看文本（17.8.3） |
| 忘了偏置或 LayerNorm 的 β | strict 加载报 missing key；关掉 strict 就静默出错 | 永远用 strict=True |
| LayerNorm 的 eps 用错 | 误差很小但明显高于浮点噪声 | 从 config 读 `layer_norm_epsilon`，不依赖默认值 |
| 忘了 `eval()` | 每次输出不一样 | 同一输入跑两次，结果应完全相同 |
| 序列超过 1024 | `IndexError: index out of range in self` | 前向开头检查长度（17.8.4） |
| 数参数用 state_dict | 多算一个嵌入矩阵，得到 163,037,184 | 用 `parameters()`，共享张量只算一次（17.1.3） |
| 把 mask 缓冲区当参数 | 转换时找不到对应的模块，或者文件大小对不上参数量 | 认识 `attn.bias` 这个旧缓冲区（17.2.3） |

## 和真实框架的对应

- **transformers 的 GPT-2**：`transformers/models/gpt2/modeling_gpt2.py`。`GPT2Attention` 里 `self.c_attn(hidden_states).split(self.split_size, dim=2)` 就是 17.6.3 节的切分；`Conv1D` 定义在 `transformers/pytorch_utils.py`，前向是 `x @ weight + bias`（用 `addmm` 实现）。`GPT2LMHeadModel` 用 `_tied_weights_keys` 声明 `lm_head.weight` 与 `transformer.wte.weight` 共享。
- **nanoGPT 的 `from_pretrained`**：做的事和本章的转换函数一样。它列出需要转置的四个权重名（`attn.c_attn.weight`、`attn.c_proj.weight`、`mlp.c_fc.weight`、`mlp.c_proj.weight`），并跳过 `attn.bias`、`attn.masked_bias` 这些缓冲区。不同之处是它保留了合并的 c_attn，没有拆成 q、k、v。
- **`safetensors`**：权重文件格式。文件开头 8 个字节是头部长度，接着是一段 JSON（每个张量的名字、dtype、形状、偏移），然后是原始数据。只读头部就能列出所有张量名和形状，不用把几百 MB 读进内存——17.10 节检查 Qwen3 有没有 `q_norm` 就是这么做的。
- **转换脚本**：大多数开源模型仓库都附带"把原始权重转成 HF 格式"的脚本（比如 transformers 里的 `convert_*_to_hf.py`）。它们干的就是本章 17.6 节的事：改名、转置、切分或拼接。以后遇到新模型，先找这个脚本读。

## 小结

GPT-2 和你的 GPT 骨架相同，差别在零件。加载官方权重靠一张对应表：每个张量照搬、转置、切分还是共享。strict 加载只检查名字和形状，方阵转置和 qkv 顺序错都会静默通过，所以还要做数值对拍，出错时逐层找第一个超差的边界。在本章的输入上（CPU、fp32），正确实现的 logits 误差只有 10⁻⁵ 量级，像 GELU 版本这样的小错误也高出约一千倍，生成的文本却可能一样，所以要看数字。

## 精读

1. **Radford et al. 2019, *Language Models are Unsupervised Multitask Learners*（GPT-2 技术报告），第 2.3 节 Model。** 只有一页多。带着问题读：
   - 报告说 LayerNorm "移到了每个子块的输入处"，并在最后加了一个 LayerNorm。对照 17.1.2 节的结构树，`ln_1`、`ln_2`、`ln_f` 分别是哪一个？
   - 报告说残差层权重按 1/√N 缩放。N 是什么？和第 11 章的 1/√(2L) 有什么关系？
   - 报告表 2 写最小的模型 117M。用 17.2 节的公式，你能找到一种算法得到接近 117M 的数吗？（开放问题，没有标准答案。）
2. **nanoGPT 的 `model.py`，`GPT.from_pretrained` 方法**（github.com/karpathy/nanoGPT）。几十行代码。带着问题读：
   - 它怎样决定哪些权重需要转置？如果 HF 将来加了一个新的 Conv1D 层，这段代码会怎样？
   - 它用 `assert` 检查了哪些东西？和本章的 strict 加载相比，哪个更严格？
3. **（选读）Qwen3 Technical Report（Qwen Team, 2025），第 2 节 Architecture。** 带着问题读：
   - 报告列出了哪些结构选择（GQA、SwiGLU、RoPE、RMSNorm、QK-Norm）？和 17.10 节的表一一对照。
   - 报告怎样解释去掉 QKV 偏置、加入 QK-Norm 的理由？
