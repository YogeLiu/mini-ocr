# 第 17 章　生成与 KV cache

## 本章在全景图中的位置

第 16 章训练出了一个语言模型：给定前面的字符，它输出下一个字符的概率分布。本章解决两个问题：

1. **怎样从分布里选字**：每次取最可能的？按概率随机抽？只在前几名里抽？不同的选法得到的文字差别很大。
2. **怎样生成得快**：第 16 章的 `sample` 每生成一个字符都把整个序列重新算一遍，生成 n 个字符的计算量 ∝ n²。**KV cache** 让每一步只算一个新 token。

这两件事直接决定 VLM 做 OCR 时的表现和速度。VLM 读一页文档、写出几千个字符的 Markdown，用的就是本章的 `generate` 函数（第 22 章起，提示里多了图像特征）。OCR 的一个常见故障——模型陷入循环，一行字反复输出——也要靠本章的工具来理解和缓解。

## 学习目标

1. 理解温度、top-k、top-p 对分布的作用，能手算；
2. 用实验看到"多样性与质量"的权衡；
3. 理解贪心解码为什么会陷入循环，以及重复惩罚的作用和局限；
4. 推导 KV cache 为什么可行（因果 mask 保证过去的 k、v 不变），实现它，并验证与不用 cache 的结果完全一致；
5. 计算 KV cache 的内存，理解 GQA 的意义；
6. 区分预填充和解码两个阶段，理解它们的瓶颈不同；
7. 实现束搜索，理解它适合 OCR 而不适合开放式写作的原因。

## 前置检查

1. 第 6 章：softmax(z / T) 在 T → 0 和 T → ∞ 时分别趋于什么？
2. 第 15 章：因果 mask 下，位置 t 的输出依赖哪些位置的输入？
3. 第 16 章：`GPT.forward` 的 `pos0` 参数是做什么用的？
4. 第 13 章：CTC 的前缀束搜索和普通束搜索有什么不同？

---

## 17.1 自回归生成

语言模型生成文字的过程叫**自回归**（autoregressive）：

```
1. 输入提示 x_1 … x_n
2. 模型输出 p(x_{n+1} | x_1 … x_n)
3. 从这个分布里选一个 token，接到序列后面
4. 重复 2–3，直到生成了足够多的 token，或者生成了结束符 <eos>
```

每一步只决定一个 token，而且一旦写下就不能修改。这和第 13 章的 CTC 很不一样：CTC 一次输出所有时间步的分布，各步独立；自回归模型的每一步都依赖前面所有已生成的 token。这让它能保持括号配对、写出连贯的结构（第 16 章：公式括号不配对率从 56% 降到 0%），代价是只能一个一个地生成，无法并行。

## 17.2 从分布里选字

### 贪心

每步取概率最大的 token。确定性的：同样的输入永远得到同样的输出。

### 采样

按概率随机抽取：概率 0.3 的 token 有 30% 的机会被选中。每次运行结果不同。

### 温度

把 logits 除以温度 T 再做 softmax：

```
p_i(T) = exp(z_i / T) / Σ_j exp(z_j / T)
```

- T = 1：原分布；
- T < 1：差距被放大，大的更大，分布更"尖"。T → 0 时趋于贪心；
- T > 1：差距被缩小，分布更"平"。T → ∞ 时趋于均匀分布。

**手算**：logits [2, 1, 0]。

- T = 1：e² : e¹ : e⁰ = 7.39 : 2.72 : 1，概率 [0.665, 0.245, 0.090]；
- T = 0.5：logits 变成 [4, 2, 0]，e⁴ : e² : 1 = 54.6 : 7.39 : 1，概率 [0.867, 0.117, 0.016]；
- T = 2：logits 变成 [1, 0.5, 0]，2.72 : 1.65 : 1，概率 [0.506, 0.307, 0.186]。

为什么叫"温度"？这来自统计物理的玻尔兹曼分布 p ∝ exp(−E / kT)：温度高时，粒子在各种能量状态之间更均匀地分布。

### top-k

只保留概率最大的 k 个 token，其余的概率设为 0，重新归一化。避免抽到概率极低的"离谱"token。

**手算**：概率 [0.5, 0.3, 0.15, 0.05]，k = 2 → [0.5, 0.3] 归一化为 [0.625, 0.375, 0, 0]。

缺点：k 是固定的。分布很尖时（模型很确定），前 k 个里可能混进几个不该选的；分布很平时（很多合理的选择），k 又可能太小。

### top-p（核采样）

按概率从大到小排序，保留累计概率**刚好达到** p 的最少几个 token。

**手算**：概率 [0.5, 0.3, 0.15, 0.05]：

- p = 0.7：0.5 不够，0.5 + 0.3 = 0.8 ≥ 0.7，保留前 2 个 → [0.625, 0.375, 0, 0]；
- p = 0.4：0.5 ≥ 0.4，只保留第 1 个 → [1, 0, 0, 0]；
- p = 0.9：0.5 + 0.3 + 0.15 = 0.95 ≥ 0.9，保留前 3 个。

top-p 自动适应分布的形状：模型很确定时只留一两个，不确定时留很多。

### 重复惩罚

对已经出现过的 token，把它的 logit 压低：

```
z_i ← z_i / penalty   （如果 z_i > 0）
z_i ← z_i × penalty   （如果 z_i < 0）
```

penalty > 1。正的 logit 除以它变小，负的 logit 乘以它变得更负——两种情况都使概率降低。（如果统一除以 penalty，负的 logit 反而会变大，所以要分开处理。）这是 Hugging Face transformers 的约定，Qwen-VL 等模型的默认生成配置里常设为 1.05 左右。

### 练习 17.1

```python
def process_logits(logits, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0, prev=None):
    logits = logits.clone()
    if repetition_penalty != 1.0 and prev is not None and prev.numel():
        sc = logits.gather(1, prev)
        sc = torch.where(sc > 0, sc / repetition_penalty, sc * repetition_penalty)
        logits.scatter_(1, prev, sc)
    logits = logits / temperature
    if top_k > 0:
        kth = logits.topk(min(top_k, logits.shape[-1]), -1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        srt, idx = logits.sort(-1, descending=True)
        cum = srt.softmax(-1).cumsum(-1)
        drop = cum - srt.softmax(-1) >= top_p
        drop[:, 0] = False
        logits = logits.masked_fill(drop.scatter(1, idx, drop), float("-inf"))
    return logits
```

逐段解释：

- **`clone()`**：不修改调用者的张量；
- **重复惩罚**：`gather(1, prev)` 取出每行中已出现 token 的 logit（prev 里同一个 token 出现多次也没关系，取出的值相同，写回的也相同）；处理后用 `scatter_` 写回原位置；
- **top-k**：`topk(k)` 返回最大的 k 个值，`[:, -1:]` 是第 k 大的值；比它小的都设为 −∞。（如果有并列，可能保留多于 k 个，影响不大。）
- **top-p**：排序后算累计概率。关键是 `cum - p_自己 >= top_p`：**在它之前**的累计概率已经够了，它就是多余的。用 `cum >= top_p` 会把"刚好使累计达到 p 的那个"也去掉。`drop[:, 0] = False` 保证至少保留一个；
- **`drop.scatter(1, idx, drop)`**：drop 是在排序后的顺序里算的，要放回原来的顺序。`scatter(1, idx, drop)` 把 drop[i][j] 放到位置 idx[i][j]；
- 顺序：先重复惩罚，再温度，再截断。这也是 Hugging Face 的顺序。

测试 `test_temperature_topk_topp` 验证了上面的手算。

## 17.3 KV cache

### 浪费在哪里

第 16 章的 `sample`：

```python
for _ in range(n):
    x = torch.tensor([ids[-T:]])
    p = m(x)[0][0, -1].softmax(-1)          # 把整个序列送进模型，只用最后一个位置的输出
    ids.append(...)
```

第 t 步把 t 个 token 全部送进模型，算出 t 个位置的输出，只用最后一个。生成 n 个 token，总共处理 1 + 2 + … + n ≈ n²/2 个位置。

### 为什么可以缓存

看第 ℓ 层的注意力。位置 t 的输出是：

```
out_t = Σ_{j ≤ t} softmax_j(q_t · k_j / √d) v_j
```

它需要：位置 t 自己的 q_t，以及所有 j ≤ t 的 k_j、v_j。

关键问题：生成第 t+1 个 token 时，位置 j ≤ t 的 k_j、v_j 会变吗？

k_j = (第 ℓ 层位置 j 的输入) W_k。第 ℓ 层位置 j 的输入是前 ℓ − 1 层在位置 j 的输出，由于**因果 mask**，它只依赖位置 0…j 的 token，和之后的 token 无关。所以 k_j、v_j 一旦算出就**永远不变**。

（如果没有因果 mask——比如 BERT 这样的双向模型——新 token 会改变之前所有位置的表示，就不能缓存了。）

所以：每一层把算过的 k、v 存起来。生成新 token 时只需：

1. 只算新 token 的 q、k、v（1 个位置）；
2. 把新的 k、v 接到缓存后面；
3. 用新的 q 和**全部**缓存的 k、v 做注意力。

每步的计算量从 O(t) 个位置降到 1 个位置（注意力部分仍要和 t 个键做点积，但这比整个网络重算便宜得多）。

### 实现

第 16 章的 `CausalSelfAttention.forward` 里已经写好了：

```python
past = 0
if cache is not None:
    if "k" in cache:
        past = cache["k"].shape[2]
        k = torch.cat([cache["k"], k], 2)
        v = torch.cat([cache["v"], v], 2)
    cache["k"], cache["v"] = k, v
...
if past == 0:
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop)
elif T == 1:
    y = F.scaled_dot_product_attention(q, k, v, dropout_p=drop)
else:
    mask = torch.ones(T, past + T, dtype=torch.bool, device=x.device).tril(past)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop)
```

- cache 是一个 dict，每层一个。第一次调用时是空的，存入这次的 k、v；之后每次把新的接在后面（第 2 维是时间维：形状 (B, h, T, dh)）；
- **RoPE 要在存入缓存之前加**：缓存里的 k 已经旋转过了，之后不再处理。新 token 的位置是 past、past+1……，所以 `GPT.forward` 用 `pos0` 取对应位置的 cos、sin 表；
- **三种 mask 情况**：
  - 没有历史（past = 0）：普通的因果注意力；
  - 只有 1 个新 token：它可以看所有缓存和自己，不需要 mask；
  - 多个新 token（比如一次送入一段提示接在历史后面）：新 token i 可以看全部历史和新 token 0…i。mask 形状 (T, past + T)，`tril(past)` 把下三角向右平移 past 列。T = 2、past = 3 时：
    ```
    [[T, T, T, T, F],
     [T, T, T, T, T]]
    ```

测试 `test_kv_cache_matches_full_forward`（第 16 章）把 12 个 token 分成"一次 5 个 + 逐个 4 个 + 一次 3 个"送入，拼起来的 logits 与一次处理全部 12 个的结果一致（误差 < 10⁻⁵）。

### 练习 17.2：generate

```python
for step in range(max_new):
    if step == 0 or not use_cache:
        logits = model(out, caches=caches)[0][:, -1]                 # 预填充：整个提示
    else:
        logits = model(out[:, -1:], caches=caches, pos0=prompt_len + step - 1)[0][:, -1]   # 解码：1 个 token
    if banned:
        logits[:, banned] = float("-inf")
    if greedy:
        nxt = process_logits(logits.float(), repetition_penalty=repetition_penalty, prev=out).argmax(-1)
    else:
        lp = process_logits(logits.float(), temperature, top_k, top_p, repetition_penalty, out)
        nxt = torch.multinomial(lp.softmax(-1), 1, generator=generator)[:, 0]
    if eos is not None:
        nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
        done |= nxt == eos
    out = torch.cat([out, nxt[:, None]], 1)
    if eos is not None and done.all():
        break
```

- **第一步**把整个提示送进去（预填充），缓存所有位置的 k、v；之后每步只送最后一个 token；
- **`pos0 = prompt_len + step − 1`**：第 step 步送入的是第 step − 1 个生成的 token，它在整个序列里的位置是 prompt_len + step − 1；
- **`banned`**：不许生成的 token（比如 `<pad>`）设为 −∞；
- **`torch.multinomial`**：按概率抽样；传入 generator 使结果可复现；
- **`eos` 和 `done`**：一批里每行可能在不同时刻结束。结束了的行之后一直补 eos，直到所有行都结束；
- `inputs_embeds` 分支（代码里省略了）供第 22 章使用：第一步送入的是图像特征和文字嵌入的拼接。

测试 `test_generate_cache_matches_no_cache` 检查用 cache 和不用 cache 的生成结果**逐个 token 相同**（贪心和采样都是）。

## 17.4 实验：分布与采样策略

以下实验都用第 16 章预训练好的模型（`outputs/ch16/gpt_char.pt`），在 CPU 上运行。

### 一个位置上的分布

第 1 部分：提示 "The cat "，模型给出的下一个字符：

```
  提示 'The cat '，模型给出的下一个字符（前 8 个）：'|':0.266  's':0.072  't':0.063  'w':0.046  'f':0.042  'm':0.040  'b':0.036  'p':0.036
  熵 3.065 nat（相当于在 e^熵 = 21.4 个字符里均匀地选）
```

第一名竟然是 `|`（26.6%）。原因在训练语料：表格的表头是 `| Cat | Word |` 这样的形式，"一个大写开头的单词 + 空格"之后常常是 `|`；而句子里的 "The cat " 之后是什么单词完全随机（446 个单词均匀选），没有哪个字母特别突出。模型忠实地反映了语料的统计。

不同的设置对这个分布的影响：

```
  设置                         最大概率      熵      可选字符数
  温度 0.5                    0.742  1.262         96
  温度 1.0（原分布）               0.266  3.065         96
  温度 1.5                    0.134  3.622         96
  top-k 5                   0.544  1.311          5
  top-p 0.9                 0.296  2.675         23
  top-p 0.5                 0.503  1.479          6
```

- **温度 0.5**：最大概率从 0.27 升到 0.74，熵减半多。但所有 96 个字符仍然可能被选中（只是概率很小）；
- **温度 1.5**：分布变平，最大概率降到 0.13；
- **top-k 5**：只剩 5 个字符，前 5 名重新归一化后 `|` 占 54%；
- **top-p 0.9**：保留 23 个字符——这个位置模型不太确定，需要很多字符才能凑够 90% 的概率；
- **top-p 0.5**：6 个。

（可选字符数是 96 而不是 100：4 个特殊符号被禁止生成。）

### 多样性与质量

第 2 部分：提示 "The "，每种设置生成 20 次，每次 120 个字符。用三个指标衡量：

- **真实单词比例**：生成的英文单词有多少在语料的单词表里（质量的粗略指标）；
- **不同的 4 字符片段**：一段文字里不重复的 4 字符片段的比例（越低越重复）；
- **20 次里不同的开头**：前 16 个字符有多少种不同（多样性）。

```
  设置                     真实单词比例       不同的 4 字符片段       20 次里不同的开头
  贪心                      87.5%            58.7%                1
  温度 0.5                  96.1%            91.9%               16
  温度 0.8                  96.9%            94.9%               19
  温度 1.0                  95.8%            95.9%               19
  温度 1.5                  88.1%            97.9%               20
  top-k 10                98.1%            89.7%               17
  top-p 0.9               97.8%            94.3%               19
```

几个样例：

```
  贪心：'The | There |\n|---|---|---|\n| 199 | 193 | 933 |\n| 193 | 933 | 133 |\n| 193 | 333 | 133 |\n\n# Them start\n- start '
  温度 0.5：'The |\n|---|---|\n| change | 744 |\n| 378 | 117 |\n| 546 | white |\n\n# Reach\n$$\\frac{5+7}{k-5k}-\\frac{k^{z}+7}{2b-z'
  温度 1.5：'The each?\nClass up world idea.\nknew 78%\n- his eat. pass 6 9%\n\n926 small\nscieNCE 50% best behind\nseveral\nCountr'
```

- **贪心**：20 次完全一样（不同的开头只有 1 种）。更糟的是它的质量反而**不高**（87.5%）：它陷入了重复（数字 193、933 反复出现，"start" 反复出现），不同的 4 字符片段只有 58.7%；
- **温度 1.5**：最多样（20 种开头，97.9% 不重复），但错字明显增多："scieNCE"（大小写乱了）、"Countr"，真实单词比例降到 88.1%；
- **中间的设置**（温度 0.8–1.0、top-k、top-p）在两者之间取得了平衡：真实单词比例 96%–98%，同时保持多样；
- **top-k 10 的质量最高**（98.1%）：它直接排除了概率排在第 11 名之后的所有字符，那些通常是错字的来源。

没有一种设置在所有指标上都最好。**多样性和质量之间存在权衡**：要多样就得允许低概率的选择，而低概率的选择里有更多错误。开放式的写作（聊天、创作）通常用温度 0.7–1.0 加 top-p 0.8–0.95。

## 17.5 贪心的循环与重复惩罚

第 3 部分：

```
  提示 '- '                         贪心                   重复的行   44%  不同的 4 字符片段 13.7%
    'start stood start\n- start stood start start\n- start stood start start\n- start stood start start\n- start stood stood start\n...'
  提示 '- '                         贪心 + 重复惩罚 1.3        重复的行   33%  不同的 4 字符片段 35.0%
    'star\n- $34.65 which mountain\n- (feel) spell start,\n- start stood stood start\n- 10% start stood start\n...'
  提示 '| Name | Price |\n|---|---|\n' 贪心                   重复的行   40%  不同的 4 字符片段 26.4%
    '| 193 | 993 |\n| 19 | 933 |\n| 833 | 833 |\n\n# Them start\n- start stood stood start\n- start stood stood start\n...'
  提示 '| Name | Price |\n|---|---|\n' 贪心 + 重复惩罚 1.3        重复的行    0%  不同的 4 字符片段 63.5%
    '| 193 | 657 |\n| 840 | 233 |\n| stop | 933 |\n\n# Then figure why\n$$\\frac{z}{k_{b}^{x+a}}=\\frac{z}{z}$$\n...'
```

**为什么贪心会循环**：贪心在每一步选最可能的字符。"start stood start" 这种文字一旦生成，下一次遇到 "\n- " 时，上下文和上次很像，最可能的续写又是同样的内容。生成的文字越来越像自己之前写过的，模型就越确信应该继续这样写——一个正反馈循环。这在所有语言模型里都存在，包括大模型。

**重复惩罚的效果**：表格的例子里，重复的行从 40% 降到 0%，模型转而生成了标题和公式；列表的例子里只从 44% 降到 33%，"start stood" 仍然反复出现。

**重复惩罚的局限**：它作用于**单个 token**，对字符级模型很粗暴。空格、`e`、`t` 本来就该反复出现，惩罚它们会让单词也写不对（第一个样例开头的 "star" 就是被截断的 "start"）。它也无法区分"该重复的"（表格的 `|`、公式里重复出现的变量）和"不该重复的"（同一行文字写两遍）。真实模型用 BPE 词表，一个 token 是一个词或词的一部分，惩罚更有针对性；即便如此，取值一般也只有 1.05 左右。

**对 OCR 的意义**：VLM 做 OCR 时，循环是一个真实的故障模式。比如一张表格有很多相似的行，模型可能在某一行之后不断重复它，直到达到最大长度。缓解的办法：轻微的重复惩罚；设置合理的最大生成长度；检测到 n 元组重复时提前停止；更根本的是在训练数据里包含足够多的长文档和重复结构，让模型学会"看图"而不是"看自己写过的"。第 25 章评测时会专门统计这类故障。

## 17.6 KV cache 的实测

第 4 部分：

```
  贪心生成 150 个字符：用 cache 和不用 cache 的结果完全相同
  逐个位置用 cache 算出的 logits 与一次算完整序列的 logits：最大差 1.7e-05（浮点误差）
      生成长度      不用 cache（秒）     用 cache（秒）     加速
       100             0.12           0.05    2.5×
       200             0.37           0.10    3.8×
       400             1.20           0.21    5.6×
```

- **结果完全相同**：生成的 150 个字符逐个一样；logits 的差别 1.7 × 10⁻⁵ 来自浮点运算顺序的不同（一次算 T 个位置的矩阵乘法 vs 逐个算），不影响 argmax；
- **用 cache 的时间与长度成正比**（0.05 → 0.10 → 0.21 秒，每次翻倍），不用 cache 的时间增长得更快（0.12 → 0.37 → 1.20 秒，每次约 3 倍），趋向于平方增长；
- **加速比随长度增长**：理论上生成 n 个 token，不用 cache 的计算量 ∝ n²/2，用 cache ∝ n，加速比 ∝ n/2。实测远小于这个值（400 个 token 只有 5.6 倍而不是 200 倍），因为我们的模型太小：每一步的大部分时间花在 Python 调度、启动几十个小的矩阵运算上，而不是真正的计算。模型越大、序列越长，实测越接近理论。

（生成 400 个字符时，后半段超出了训练长度 256，第 16 章 16.10 节说过质量会变差。这里只测速度。）

### cache 的内存

```
  cache 的大小：每个 token 每层存 k、v 各 d 维 → 2 × 4 层 × 256 × 4 字节 = 8,192 字节/token；256 个 token 共 2048 KB
  Qwen3-0.6B（bf16，GQA 8 组键值头 × 128 维）：2 × 28 × 8 × 128 × 2 = 114,688 字节/token；一个 8000 token 的文档 875 MB；如果不用 GQA（16 组）要翻倍
```

公式：每个 token 的 cache = 2（k 和 v）× 层数 × 键值头数 × 每头维度 × 每个数的字节数。

对 Qwen3-0.6B：一个 8000 token 的文档（一页密集的文档图像加上输出的 Markdown 很容易达到）需要 875 MB 的 cache，而模型本身 bf16 只有 1.2 GB。批大小 8 时 cache 就是 7 GB，超过模型本身好几倍。

**分组查询注意力（GQA）**就是为此设计的：16 个查询头共用 8 组键值头（每两个查询头共用一组 k、v），cache 减半。更激进的做法是所有查询头共用一组（MQA，多查询注意力），cache 降到 1/16，但效果会有所损失。GQA 是折中。

vLLM 等推理框架的核心技术之一（PagedAttention）就是像操作系统管理内存页一样管理 KV cache，减少碎片、在请求之间共享相同前缀的 cache。

## 17.7 预填充与解码

第 5 部分：

```
  一次送入 250 个 token（预填充）：3.4 ms，72,773 token/秒
  一个一个送入同样 250 个 token（解码）：121.1 ms，2,065 token/秒（慢 35 倍）
```

同样是处理 250 个 token、同样的计算量，一次送入比逐个送入快 35 倍。

**预填充**（prefill）：提示的所有 token 一次送入，每一层都是 (250 × 256) @ (256 × 256) 这样的大矩阵乘法，硬件的并行计算能力被充分利用。

**解码**（decode）：每次只有 1 个 token，每一层是 (1 × 256) @ (256 × 256) 的矩阵-向量乘法。为了做这一点点计算，要把整个权重矩阵从内存读到计算单元。瓶颈不是计算，而是**内存带宽**：每生成一个 token，都要把全部参数读一遍。

对 VLM 的意义：

- **处理图像很快，写答案很慢**：一张图的几千个视觉 token 是预填充，一次完成；输出几百个字是解码，要一步一步来。实际使用中，"首个 token 的延迟"（time to first token，主要是预填充）和"每秒生成的 token 数"（解码速度）是两个分开的指标；
- **批处理**：服务端把多个请求拼成一批一起解码。每次读入的权重同时服务批里所有的请求，内存带宽被摊薄。这是 vLLM 这类推理服务器提高吞吐的主要手段。批处理需要处理长度不同的序列（左侧补齐 + 补齐 mask，或者更复杂的连续批处理），我们的 `generate` 为了简单没有实现。

## 17.8 束搜索

### 算法

贪心每步只保留一个候选。**束搜索**（beam search）保留累计 log 概率最高的 k 个候选（k 叫束宽）：

1. 每个候选扩展出 V 个后继（k × V 个）；
2. 按累计 log 概率排序，保留前 k 个；
3. 以 `<eos>` 结尾的候选移入"完成"列表；
4. 重复，直到有 k 个完成的候选或达到最大长度；
5. 按 累计 log 概率 / 长度^α 选最好的（长度惩罚：否则短的序列总是占优，因为每多一个 token 就多乘一个小于 1 的概率）。

### 练习 17.3

```python
beams = [(0.0, idx[0].tolist())]
finished = []
for _ in range(max_new):
    x = torch.tensor([s for _, s in beams], device=idx.device)
    lp = F.log_softmax(model(x)[0][:, -1].float(), -1)
    cand = []
    for (score, seq), row in zip(beams, lp):
        top = row.topk(beam)
        cand += [(score + v.item(), seq + [k.item()]) for v, k in zip(top.values, top.indices)]
    cand.sort(key=lambda c: -c[0])
    beams = []
    for score, seq in cand:
        if eos is not None and seq[-1] == eos:
            finished.append((score, seq))
        else:
            beams.append((score, seq))
        if len(beams) == beam:
            break
    if not beams or len(finished) >= beam:
        break
```

- 每个候选只取前 `beam` 个后继（而不是全部 V 个），因为最终只保留 `beam` 个，每个候选贡献的不会超过 `beam` 个；
- 所有候选一起送进模型（批大小 = 束宽），每步重新计算整个序列（没有 KV cache，为了简单）。带 cache 的束搜索需要在每步按选中的候选重新排列 cache；
- 与第 13 章 CTC 的前缀束搜索不同：这里每个候选就是一个序列，不需要合并路径。

测试 `test_beam_search_at_least_as_good_as_greedy` 检查束搜索的结果概率不低于贪心（在那个小模型上），以及束宽 1 时和贪心相同。

### 实验

第 6 部分：

```
  提示 '$$\\frac{'：
    贪心         累计 log 概率   -45.38  'z}{z}$$\n- start start start start\n- star'
    束搜索 4      累计 log 概率   -34.04  'z}{z}$$\n\n# Thought\n$$\\frac{z}{z}+\\frac{z'
    束搜索 16     累计 log 概率   -33.29  'z}{z}$$\n\n# Thought\n$$\\frac{a}{z}$$\n$$\\fr'
  提示 'The '：
    贪心         累计 log 概率   -27.53  '| There |\n|---|---|---|\n| 199 | 193 | 93'
    束搜索 4      累计 log 概率   -29.89  '|\n|---|---|---|\n| 199 | 199 | 933 |\n| 19'
    束搜索 16     累计 log 概率   -19.68  '|\n|---|---|---|\n| complete | thought | 3'
```

- 束宽 16 找到的序列概率都明显高于贪心（−33.3 vs −45.4，−19.7 vs −27.5）；
- **束宽 4 不一定比贪心好**："The " 的例子里束宽 4 是 −29.89，比贪心的 −27.53 还低。这并不矛盾：束搜索在每一步按累计概率剪枝，贪心路径的某个前缀可能在中途被剪掉（当时有 4 个累计概率更高的前缀），而那些前缀后来的发展不如贪心路径。束搜索只是一种启发式搜索，不保证找到最优解，甚至不保证比贪心好（测试里的小例子恰好满足，这里的不满足）；
- **概率高不等于好**：束宽 16 在 "$$\frac{" 之后写出 `\frac{z}{z}`、`\frac{a}{z}`——最"保险"、最常见的模式。在开放式写作中，人类写的文字并不是概率最高的那个序列；概率最高的序列往往平淡、重复。这是开放式生成通常用采样而不用束搜索的原因。

**OCR 不一样**。OCR 的答案是唯一的（图上写了什么就是什么），模型越好，正确答案的概率越高，概率最高的序列通常就是正确答案。所以 OCR 和机器翻译、语音识别这类"有标准答案"的任务适合贪心或束搜索。实际中 VLM 做 OCR 大多用**贪心**：快、确定、可复现；加一点重复惩罚防止循环。束搜索要多算 k 倍，收益通常很小（第 13 章：CTC 上束搜索只改变了 2000 行测试数据中 26 行的结果）。第 25 章会在我们的 VLM 上比较不同解码方式对 CER 的影响。

## 17.9 自己动手

### 必做

1. **完成练习 17.1–17.3**，使 `python -m pytest labs/ch17` 全部通过（4 项）。
2. **手算**：logits [3, 1, 0, −1]。分别计算温度 1、温度 0.5、top-k 2、top-p 0.9 之后的概率分布。
3. **重复惩罚手算**：logits [2, −1, 0.5]，已出现的 token 是 0 和 1，penalty 2。处理后的 logits 和概率是多少？
4. **KV cache 内存**：一个模型 32 层、32 个查询头、8 组键值头、每头 128 维，bf16。每个 token 的 cache 多少字节？上下文 32K 时一个请求要多少 GB？
5. **复现第 2 部分**：运行 `python -m labs.ch17.gen_tour 2`。加一行"温度 0.8 + top-p 0.9"的组合，结果如何？

### 选做

6. **min-p 采样**：一种较新的方法：只保留概率 ≥ min_p × 最大概率的 token。在 `process_logits` 里实现它，和 top-p 比较。
7. **n 元组重复检测**：在 `generate` 里加一个选项：如果最近生成的 n 个 token 在之前已经出现过 m 次，就停止生成。用它处理第 3 部分的循环。
8. **带 cache 的束搜索**：每步选出新的 k 个候选后，用 `index_select` 按候选的来源重新排列每层的 cache。测量加速比。
9. **批量生成**：让 `generate` 支持长度不同的一批提示（左侧补齐，传入补齐 mask）。需要修改 `CausalSelfAttention` 的 mask 和 RoPE 的位置。

## 17.10 常见错误

| 现象 | 原因 | 怎么查 |
|---|---|---|
| 用 cache 和不用 cache 结果不同 | RoPE 的位置没有用 pos0 偏移；或 cache 在 RoPE 之前存入 | 第 16 章的 `test_kv_cache_matches_full_forward` |
| 生成到一半报"超过 block_size" | 提示 + 生成长度超过 RoPE 表的长度 | 限制最大长度，或截断提示 |
| top-p 总是只留下一个 token | 用 `cum >= p` 而不是 `cum − p_自己 >= p` 判断 | 用 17.2 节的手算例子测试 |
| 重复惩罚让负的 logit 变大 | 对所有 logit 统一除以 penalty | 负数要乘以 penalty |
| 同一个种子两次生成结果不同 | 没有把 generator 传给 `multinomial`；或用了 GPU 的非确定性算子 | 显式传入 `torch.Generator` |
| 贪心生成陷入循环 | 贪心解码的固有问题 | 采样、重复惩罚、重复检测、限制长度 |
| 一批里有的行已经结束，还在生成 | 没有处理 eos | 用 done 标记，结束的行补 eos |

## 17.11 真实世界对照

- **Hugging Face `generate`**：参数 `do_sample`、`temperature`、`top_k`、`top_p`、`repetition_penalty`、`num_beams`、`max_new_tokens`，含义和本章相同。模型目录里的 `generation_config.json` 给出默认值，比如 Qwen2.5 的对话模型默认 temperature 0.7、top_p 0.8、top_k 20、repetition_penalty 1.05；
- **OCR 的生成设置**：OCR 模型的推荐设置通常是贪心（`do_sample=False`）或很低的温度，较大的 `max_new_tokens`（一页文档可能有几千个 token），有时加上 `no_repeat_ngram_size` 防止循环；
- **vLLM**：最常用的开源推理服务器之一（作者测量时 GPU 上同时运行的"另一个服务"就是它，第 9 章）。它的 PagedAttention 管理 KV cache，连续批处理（continuous batching）让新请求随时加入正在解码的批，前缀缓存让相同的系统提示只算一次预填充；
- **投机解码**（speculative decoding）：用一个小模型快速"猜"几个 token，再用大模型一次预填充式地验证。利用了"预填充比解码快得多"这一点，在不改变输出分布的前提下加速 2–3 倍。

## 17.12 深入一步

### 为什么不能直接取"概率最高的序列"

概率最高的序列需要在 V^n 个序列里搜索，n = 100、V = 15 万时是天文数字。贪心和束搜索都是近似。更根本的问题是，即使能找到，对开放式生成来说它也不是我们想要的（17.8 节）。语言模型的分布中，"最可能的单个序列"的概率极低，真实的文字分布在大量概率都很低的序列上；从分布中采样才能得到"典型"的文字。

### 温度与熵

温度 T 下的分布 p(T) 的熵随 T 单调增加。T = 1 时就是模型学到的分布。17.4 节的第 1 部分里，温度 0.5 把熵从 3.07 降到 1.26，温度 1.5 升到 3.62。一些方法（如 entropy-based sampling）根据每一步的熵动态调整温度：模型确定时更贪心，不确定时更随机。

### KV cache 与注意力的对称性

为什么只缓存 k、v 而不缓存 q？因为过去位置的 q 只在计算过去位置的输出时用到，而过去位置的输出已经算完了（因果 mask 保证它们不受新 token 影响）。新 token 只需要自己的 q，去和所有的 k、v 做注意力。

## 17.13 横向对照

| 解码方式 | 确定性 | 多样性 | 速度 | 适用 |
|---|---|---|---|---|
| 贪心 | 是 | 无 | 最快 | OCR、代码补全、有标准答案的任务 |
| 束搜索 | 是 | 无 | 慢 k 倍 | 机器翻译、语音识别 |
| 温度采样 | 否 | 可调 | 快 | 开放式写作 |
| top-k / top-p | 否 | 可调，截断低概率尾部 | 快 | 聊天、创作 |

| | 预填充 | 解码 |
|---|---|---|
| 每次处理 | 整个提示 | 1 个 token |
| 瓶颈 | 计算 | 内存带宽 |
| 本章实测 | 72,773 token/秒 | 2,065 token/秒 |
| VLM 中对应 | 读图像 + 提示 | 写答案 |

## 17.14 本章小结

- 自回归生成：每步从模型的分布中选一个 token，接到序列后面；
- 温度调节分布的尖锐程度；top-k 保留固定个数；top-p 保留累计概率达到 p 的最少几个；
- 多样性与质量之间存在权衡；贪心最不多样，而且会陷入循环；
- 重复惩罚压低已出现 token 的 logit，能缓解循环，但对字符级模型很粗暴；
- KV cache 利用因果 mask 下过去的 k、v 不变，每步只算新 token；结果与不用 cache 完全相同；
- cache 的内存 = 2 × 层数 × 键值头数 × 每头维度 × 字节数 × 长度；GQA 减少键值头数；
- 预填充是计算密集的，解码是内存带宽密集的，同样的 token 数相差 35 倍；
- 束搜索找概率更高的序列，但不保证比贪心好；适合 OCR 这类有标准答案的任务，不适合开放式写作；VLM 做 OCR 大多用贪心。

## 术语卡

| 中文 | 英文 | 一句话 |
|---|---|---|
| 自回归 | autoregressive | 每步依赖之前生成的所有 token |
| 贪心解码 | greedy decoding | 每步取概率最大的 token |
| 温度 | temperature | logits 除以 T 再做 softmax |
| top-k 采样 | top-k sampling | 只在概率最大的 k 个里采样 |
| 核采样 | nucleus (top-p) sampling | 只在累计概率达到 p 的最少几个里采样 |
| 重复惩罚 | repetition penalty | 压低已出现 token 的 logit |
| 键值缓存 | KV cache | 保存过去位置的 k、v，避免重算 |
| 预填充 | prefill | 一次处理整个提示 |
| 解码 | decode | 逐个生成 token 的阶段 |
| 束搜索 | beam search | 保留累计概率最高的 k 个候选 |
| 分组查询注意力 | grouped-query attention, GQA | 多个查询头共用一组键值头 |
| 投机解码 | speculative decoding | 小模型猜、大模型验证 |

## 自测题

**概念**

1. 温度 T → 0 和 T → ∞ 时，采样分别变成什么？
2. top-p 相比 top-k 的优点是什么？
3. 为什么贪心解码会陷入循环？
4. 为什么 KV cache 只在因果模型里可行？
5. 预填充和解码的瓶颈分别是什么？为什么 VLM 读图比写答案快？
6. 为什么 OCR 适合贪心解码，而聊天不适合？
7. 第 17.8 节中束宽 4 的结果比贪心还差，为什么这可能发生？

**计算**

8. logits [2, 0, −2]，温度 0.5 后的概率是多少？
9. 概率 [0.4, 0.3, 0.2, 0.1]，top-p = 0.75 保留哪几个？重新归一化后是多少？
10. 本章 GPT 生成 400 个 token，不用 cache 时总共处理多少个位置？用 cache 时呢？
11. Qwen3-0.6B，bf16，批大小 4、每个请求 16000 个 token，KV cache 一共多少 GB？

**代码**

12. 下面的 top-k 实现有什么问题？
    ```python
    v, i = logits.topk(k)
    logits[:] = float("-inf")
    logits[i] = v
    ```
13. 写出 `generate` 中，第 step 步（step ≥ 1）用 KV cache 时传给模型的输入和 `pos0`。
14. 怎样验证一个 KV cache 实现是正确的？写出测试思路。

## 参考答案

1. T → 0：分布集中到最大的 logit 上，变成贪心；T → ∞：所有 logit 除以无穷大都趋于 0，变成均匀分布。
2. top-p 根据分布的形状自动调整保留的个数：模型确定时只保留一两个，不确定时保留很多；top-k 的 k 固定，在两种情况下都可能不合适。
3. 一旦生成了一段常见的文字，之后遇到相似的上下文时，最可能的续写又是同样的内容；生成的内容越重复，上下文越相似，模型越确信要重复，形成正反馈。
4. 因果 mask 保证位置 j 的 k、v 只依赖位置 0…j 的 token，新 token 不会改变它们；双向模型里新 token 会改变之前所有位置的表示，缓存的 k、v 就失效了。
5. 预填充一次处理很多 token，大矩阵乘法，瓶颈是计算；解码每次 1 个 token，要为很少的计算读入全部权重，瓶颈是内存带宽。VLM 的视觉 token 是预填充，一次完成；答案是逐个解码。
6. OCR 有唯一的正确答案，模型好时概率最高的序列就是正确答案，贪心快且确定；聊天是开放式的，概率最高的序列往往平淡重复，需要采样带来多样性。
7. 束搜索按累计概率剪枝，贪心路径的某个前缀可能在中途被 4 个累计概率更高的前缀挤掉，而那些前缀后来的发展不如贪心路径。束搜索是启发式的，不保证最优，也不保证优于贪心。
8. logits 变成 [4, 0, −4]；e⁴ = 54.6，e⁰ = 1，e⁻⁴ = 0.018；概率 [0.982, 0.018, 0.0003]。
9. 0.4 不够，0.4 + 0.3 = 0.7 不够，0.7 + 0.2 = 0.9 ≥ 0.75，保留前 3 个；归一化：[0.444, 0.333, 0.222, 0]。
10. 不用 cache：1 + 2 + … + 400 ≈ 80,200 个位置（加上提示的长度会更多）；用 cache：提示一次 + 400 次各 1 个位置。
11. 114,688 字节/token × 16000 × 4 = 7.34 × 10⁹ 字节 ≈ 6.8 GB。
12. 这段代码没有处理批的维度：logits 是 (B, V) 时，`logits[i] = v` 会按第 0 维（批）索引，结果完全错误。应该用 `logits.masked_fill(logits < 第 k 大的值, -inf)` 或者 `scatter`。另外它直接修改了输入张量。
13. 输入 `out[:, -1:]`（上一步生成的那个 token），`pos0 = prompt_len + step − 1`。
14. 用同一个模型和输入，(a) 一次前向整个序列得到 logits；(b) 用 cache 把序列分成几段（一段多个 token、之后逐个）依次送入，拼接 logits；比较两者是否在浮点误差内相等。再比较用和不用 cache 的贪心生成结果是否逐个 token 相同。
