# 第 32 章 解剖真实 VLM

## 本章导读

上一章你亲手搭了一个小公式 OCR：ViT 看图，两层 MLP 转接，GPT 写 LaTeX，视觉 token 用**前缀拼接**塞到文字前面。全部约 587 万参数，每个零件都是你写的。

这一章换成别人造的大家伙。我们把三个真实 VLM 拆开看：

- **Qwen3.5-0.8B**：本章的主角。数据流、形状和参数都用它来讲。
- **SmolVLM-256M-Instruct**：最小的那个，用来对照另一种 processor 和连接器。
- **Qwen3-VL-2B-Instruct**：同一家族里更大的一个，主要出现在零样本实测里。

方法沿用第 17 章拆 GPT-2 的做法：先打印形状，再逐项数参数（对拍只做 processor，范围见 32.3.5 节）。VLM 比 GPT-2 多出一整段把图片变成 token 的流水线，它藏在 processor 和 chat template 里，容易出错，也少有人讲清楚。本章一半的篇幅花在这里。

先说结论：真实 VLM 和你的小 OCR 是同一种结构，都是"视觉编码器 → 连接器 → 语言模型"，视觉 token 都排在问题文字前面。大模型多做的事情主要有四件：

1. **动态分辨率**：不把所有图压成同一个尺寸，而是按面积预算缩放，视觉 token 数跟着图变。
2. **token 合并**：相邻 2×2 个 patch 并成一个 token，进语言模型的序列短了 4 倍。
3. **占位符机制**：先在文字里放一排占位 token，再把视觉向量填进这些位置，不直接拼在最前面。
4. **预训练**：视觉塔、语言模型都是预训练过的，开箱就能做零样本 OCR，只是格式不一定合你的意。

实测的结论是：不训练时三个模型的 EM_norm 都很低，最高的 Qwen3-VL-2B 也只有 13/180（本项目观察，【G-val】）。但低分不全是没看清图。Qwen3.5 有 48 条只差句末标点，其中 46 条模型其实抄对了，是提取规则把标点删了（32.9 节）。

**学完能做到：**

1. 画出 Qwen3.5-0.8B 的完整数据流：图片 → processor → 视觉塔 → merger → 语言模型，写出每一步的形状。
2. 用 `smart_resize` 手算任意一张图的缩放尺寸和视觉 token 数，说清三条分支和它们的取整方向。
3. 解释 `pixel_values` 为什么是 `[N, 1536]`，为什么 patch 要按"2×2 邻域优先"排列；复刻 processor 并和 transformers 逐位对上。
4. 逐项数出视觉塔、连接器、语言模型的参数量，和实测总数对上。
5. 比较三种连接器：LLaVA 的 MLP、Qwen 的 2×2 合并 + MLP、BLIP-2 的 Q-Former，说出它们分别压缩了什么。
6. 读懂 chat template，算出占位符展开以后的输入长度；说清视觉向量怎样填进语言模型的输入。
7. 读懂零样本实测表，从原始输出里找出失败模式，区分"模型没看清"和"评分口径不合"。

**重点：** processor（32.2、32.3）、三段结构与参数量（32.4）、连接器比较（32.5）、chat template 与占位符（32.6）、视觉 token 进入语言模型（32.7）、零样本实测（32.8）、失败模式与输出格式诊断（32.9）。**概括：** OCR 专用 VLM 的模块说明（32.10）、各模型的训练流程（第 36 章读报告时再看）、Qwen3.5 语言模型里的线性注意力层（只数参数，不讲原理）。

**本章代码：** 复刻与连接器在 `miniocr/models/vlm_anatomy.py`（下文简称 `vlm_anatomy.py`），实验脚本是本章目录下的 `tour.py`，汇总脚本是 `summarize_runs.py`，测试是 `test_ch32_anatomy.py`，操作步骤见 [lab.md](lab.md)。正文说"tour 部分 k"时，指 `python -m book.part4_vlm.ch32_real_vlm.tour k`，对应 lab 的第 k 节。本章不训练任何模型。

**数字的来源标记。** 实测数字都来自 `course-v4/outputs/book/ch32/gpu/`（下文简称 `gpu/`）。先约定几个记号：

| 记号 | 文件 | 设备与配置 |
|---|---|---|
| 【G-val】 | `gpu/remote_runs/main/*_val_original.json` | RTX 4090（租用），模型 bf16、processor fp32；val 原图 180 条；batch 1；贪心，max_new_tokens=256；训练步数 0 |
| 【G-ren】 | `gpu/remote_runs/rendered/*_val_rendered.json` | 同上；val 中可渲染的前 80 条，用 mathtext 重新渲染的图 |
| 【G-pair】 | `gpu/remote_runs/paired.json` | 同上；同一批 80 条目标，原图和渲染图配对比较 |
| 【G-test】 | `gpu/remote_runs/final/qwen3vl_test_original.json` | 同上；test 100 条，只评一次 |
| 【G-flow】 | `gpu/remote_runs/main/dataflow.json`、`processor.json` | 同上；forward hook 记录的形状，processor 对拍记录 |
| 【权重头】 | 本地 Hugging Face 缓存里 safetensors 文件的头部 | 只读文件头里的张量名和形状，不加载权重；快照和 GPU 运行一致（Qwen3.5 `2fc06364…`、SmolVLM `7e3e67ed…`） |

【G-…】都是"本项目观察"，单次运行、单一数据前缀。【权重头】给的是形状，由形状相乘得到的参数量算"数学结论"。没有标记的推导结果是"数学结论"，没实测过的写"待验证"。

**和项目的关系。** 第 31 章是你从零写的小模型（M2）。这一章换成真实 VLM，按预先定好的规则在 val 上选出一个模型，交给第 33 章做 LoRA 微调（R3 的零样本基线）。相关作业的门槛**待校准**。

> **先修回顾。**
> - 第 17 章：拆一个真实模型的四步是打印结构、读 config、逐项数参数、自己写前向和官方对拍；对拍要关掉一切随机和差异来源。本章前三步照做；第四步缩小范围，第 17 章自己写了完整前向，这一章只复刻 processor 和官方逐位对拍，模型内部用 forward hook 记录形状，不复刻前向。
> - 第 26 章：两类多模态接口（前缀拼接、交叉注意力）；Qwen 的 processor 把高宽调到 32 的倍数，切 16×16 patch，2×2 合并。
> - 第 29 章：patch 嵌入就是核与步长都为 p 的卷积；视觉 token 数 (H/p)(W/p)；分辨率变了要插值位置表。
> - 第 30 章：CLIP/SigLIP 用图文对比预训练视觉编码器，SmolVLM 的视觉塔就是 SigLIP 系列。
> - 第 31 章：前缀拼接 `torch.cat((visual, tok(inputs)), dim=1)`；EM_norm、CER、parse_rate 的口径；val 选模型，选型冻结后预先登记的候选各评一次 test（0.10.1 节）。

---

## 32.1 全景：一张公式图在 Qwen3.5 里的旅程

### 32.1.1 动机

第 31 章的小 OCR 只认一种输入：32×160 的灰度图，320 个视觉 token，一个字节一个 token。真实 VLM 面对的是什么尺寸都有的彩色图，还要和聊天格式、子词词表、上百个特殊 token 打交道。

所以第一步是跟着一张真图走一遍，把每一站的形状记下来。tour 部分 5 用 forward hook 做了这件事（lab 第 5 节）。

### 32.1.2 直觉：前台、眼睛、转接头、笔

把 VLM 想成一家翻译社：

- **processor 是前台。**它不懂内容，只负责排版：把图片裁成规定的块，把你的问题套进固定的表格，再在表格里留出一排"图片位"。
- **视觉塔是眼睛。**一个 ViT，把每个图块变成一个向量。
- **连接器是转接头。**眼睛说的是"视觉方言"，宽度 768；笔只认宽度 1024 的"文字方言"。连接器负责换插头，顺便把相邻四块并成一块。
- **语言模型是笔。**它看见的是一串向量，前面一段来自图，后面一段来自文字，然后一个 token 一个 token 往下写。

第 31 章里这四个角色全都有，只是前台很简陋（固定画布），转接头不压缩（320 进 320 出）。

### 32.1.3 全景表：每一站的形状

拿 val 的第一张图做例子。它原本高 32、宽 298 像素，是一行很扁的公式。【G-flow】记录的形状如下：

| 站 | 张量 | 形状 | 怎样来的 |
|---|---|---|---|
| 原图 | RGB 图片 | 32 × 298 × 3 | |
| `smart_resize` | 缩放后的图 | 96 × 800 × 3 | 面积太小，放大到至少 65536 像素（32.2 节） |
| 切 patch | `image_grid_thw` | [[1, 6, 50]] | 96/16 = 6 行，800/16 = 50 列 |
| 排列 | `pixel_values` | [300, 1536] | 300 = 6 × 50 个 patch；1536 = 3 × 2 × 16 × 16（32.3 节） |
| 套模板 | `input_ids` | [1, 111] | 36 个文字 token + 75 个图片占位（32.6 节） |
| patch 嵌入 | `visual.patch_embed` 输出 | [300, 768] | 一个三维卷积，等价于 1536 → 768 的线性层 |
| 12 个视觉 block | 每个 block 输出 | [300, 768] | 个数、宽度都不变 |
| merger | `visual.merger` 输出 | [75, 1024] | 4 个相邻 patch 拼成 3072 维，再 MLP 到 1024（32.5 节） |
| 填入占位 | `language_model` 的 `inputs_embeds` | [1, 111, 1024] | 75 个占位位置换成视觉向量（32.7 节） |
| 位置编号 | `position_ids` | [3, 1, 111] | 每个 token 三个坐标（32.7 节） |
| 输出 | `logits` | [1, 111, 248320] | 词表 248320 |

和第 31 章对照着看（第 31 章数字取自它的 31.1.3 节）：

| | 第 31 章小 OCR | Qwen3.5-0.8B |
|---|---|---|
| 输入图 | 固定 32×160 灰度 | 任意尺寸 RGB，按面积缩放 |
| patch | 4×4 | 16×16，静态图复制成 2 帧 |
| 视觉编码器 | 3 层 ViT，宽 256，随机初始化 | 12 层 ViT，宽 768，预训练 |
| 进语言模型的视觉 token | 320 | 这张图 75，val 上 65 到 96 |
| 连接器 | 两层 MLP，不压缩 | 2×2 合并 + 两层 MLP，压缩 4 倍 |
| 接口 | `torch.cat` 前缀拼接 | 占位符 + 填入 |
| 词表 | 259（字节） | 248320（子词） |
| 参数 | 约 587 万 | 852,985,920 |

最后一行的 852,985,920 是 tour 加载模型时打印的参数总数（本项目观察，`gpu/remote_logs/ch32_all_remote_logs.log` 的 `parameters=` 字段）。32.4 节会把它一项一项拆出来。

再按代码里的名字把零件对一遍。中间一列是第 31 章 `formula_ocr.py` 里 `FormulaOCR` 的属性，右边一列是 Qwen3.5 的模块路径，`print(model)` 和 safetensors 文件头（【权重头】）里都是这些名字：

| 角色 | 第 31 章 `FormulaOCR` | Qwen3.5-0.8B | 本章哪一节 |
|---|---|---|---|
| 前台：图像 | 无，数据端直接给 32×160 灰度画布 | image processor：`smart_resize` 缩放、切 patch、排列 | 32.2、32.3 |
| 前台：文字 | `targets_for`：字节值加 1，前面放 BOS | tokenizer 加 chat template，在图片的位置留一个占位符 | 32.6 |
| patch 嵌入 | `patch_embed(images, self.patch_weight)` | `model.visual.patch_embed`（三维卷积） | 32.3.3、32.4.3 |
| 视觉位置 | `self.pos`，固定 320 行 | `model.visual.pos_embed`，48×48 的表按实际网格插值；注意力里另有二维 RoPE | 32.4.3 |
| 视觉编码器 | `self.vision`，3 个 `VisionBlock` | `model.visual.blocks`，12 个 block | 32.4.3 |
| 连接器 | `self.connector`：LayerNorm 加两层 MLP，320 进 320 出 | `model.visual.merger`：LayerNorm、2×2 拼接、两层 MLP，300 进 75 出 | 32.5 |
| 接口 | `torch.cat((visual, self.gpt.tok(inputs)), dim=1)` | 先查嵌入表，再用 `masked_scatter` 把视觉向量填进占位位置 | 32.7 |
| 语言侧位置 | GPT 里的一维 RoPE | M-RoPE，`position_ids` 形状 [3, 1, L] | 32.7 |
| 语言模型 | `self.gpt`，4 层 | `model.language_model`，24 层 | 32.4.3 |
| 输出层 | 和字节嵌入共享权重 | 和词嵌入共享权重（`tie_word_embeddings`） | 32.4.3 |

每一行左右两边做的是同一件事。差别最大的是两行前台、接口和两行位置，本章大半篇幅在讲它们。

### 32.1.4 三个模型一览

同一张 32×298 的图，三个模型的输入长度差得很远【G-val，第一条样本的 `observations[0]`】：

| 模型 | 参数 | 视觉 token | 输入 token 总数 | `pixel_values` 形状 |
|---|---|---|---|---|
| Qwen3.5-0.8B | 852,985,920 | 75 | 111 | [300, 1536] |
| Qwen3-VL-2B | 2,127,532,032 | 75 | 107 | [300, 1536] |
| SmolVLM-256M | 256,484,928 | 320 | 364 | [1, 5, 3, 512, 512] |

参数总数来自同一份远程日志。两个 Qwen 用同一套图像处理规则，所以视觉 token 一样多；Qwen3-VL 少的 4 个 token 来自 chat template（32.6 节）。SmolVLM 最小，输入却最长：它把图切成 4 块再加一张缩略图，每块 64 个 token。参数少不等于序列短。

---

## 32.2 processor（一）：缩放与视觉 token 数

### 32.2.1 动机

视觉 token 数决定了三件事：语言模型的序列有多长、注意力要算多少、显存要占多少。第 29 章说过，ViT 的 token 数是 (H/p)(W/p)。问题是：真实 VLM 的 H 和 W 是多少？

processor 会先把图缩放一次，所以 H 和 W 不是你给的那个尺寸。缩放规则叫 `smart_resize`，下游的形状都由它决定。不会手算它，你就没法预测显存、没法解释"为什么我的图变糊了"，也没法在第 33 章调图像预算。

### 32.2.2 直觉：按面积缩放的复印机

想象一台复印机，规定"复印件面积必须在 A 到 B 之间，边长必须是 32 的倍数，尽量保持长宽比"。

- 原件太小（一张便利贴）：放大到面积刚好超过 A。
- 原件太大（一张海报）：缩小到面积不超过 B。
- 大小合适：只把边长凑成 32 的倍数。

公式图几乎全是便利贴：高几十像素、宽几百像素。所以对本章的数据，`smart_resize` 基本上只走放大这一支。

为什么是 32？因为一个 patch 是 16×16，合并又是 2×2，一个视觉 token 管 32×32 像素。边长是 32 的倍数，才能不多不少地切完、并完。

### 32.2.3 推导：三条分支

记原图高 h、宽 w，因子 f = patch × merge = 16 × 2 = 32，面积下限 m（`min_pixels`），上限 M（`max_pixels`）。本章实验设置 m = 65536 = 256²，M = 262144 = 512²（lab 第 0 节；这是实验设置，不是模型的原生最大分辨率）。

**第一步，就近取整。**

$$
h_0=\operatorname{round}(h/f)f,\qquad w_0=\operatorname{round}(w/f)f
$$

注意 Python 的 `round` 是"四舍六入五成双"：`round(2.5) = 2`，`round(3.5) = 4`。

**第二步，看面积落在哪里。**

- **缩小分支**，h₀·w₀ > M（太大）：令 s = √(h·w / M)。s 是边长要缩的倍数：面积是两条边相乘，要把面积缩到原来的 1/s²，两条边各除以 s 就够了，所以面积比要开平方。

$$
H'=\max\left(f,\left\lfloor\frac{h}{sf}\right\rfloor f\right),\qquad W'=\max\left(f,\left\lfloor\frac{w}{sf}\right\rfloor f\right)
$$

  向下取整。两条边都没有被 `max(f, …)` 钉住时，面积不超过 M；短边被钉在 f 时可能超过。例如 m = 16384、M = 65536 时，32×3200 缩成 32×2560，面积 81920 > 65536（数学结论，第 33 章的预算就是这一组）。
- **放大分支**，h₀·w₀ < m（太小）：令 s = √(m / (h·w))，两条边各乘以 s，

$$
H'=\left\lceil\frac{hs}{f}\right\rceil f,\qquad W'=\left\lceil\frac{ws}{f}\right\rceil f
$$

  向上取整，保证不低于 m。
- 否则，面积在 [m, M] 之间：H' = h₀，W' = w₀。

**第三步，数 token。**

$$
\begin{aligned}
\text{patch 数：}\quad N_{\mathrm{patch}} &= \frac{H'}{16}\cdot\frac{W'}{16}=\frac{H'W'}{256} \\
\text{视觉 token：}\quad N_{\mathrm{vis}} &= \frac{N_{\mathrm{patch}}}{4}=\frac{H'W'}{1024}
\end{aligned}
$$

也就是说（数学结论）：**静态图片的视觉 token 数等于缩放后面积除以 1024。**1024 = 32²，正好是一个视觉 token 管的 32×32 像素；H'、W' 都是 32 的倍数，所以这个除法总能除尽。视觉编码器里跑的是 N_patch，语言模型里跑的是 N_vis，前者是后者的 4 倍。注意这是整幅缩放后画布的 token 总数，不是“有内容的 token”数：白边、空白区域照样占 token，墨迹落在多少块上要按网格逐块统计（见第 29 章 29.4.4 节）。

**两个取整方向带来的意外（数学结论）：**

1. 放大分支用向上取整，两条边各自向上凑，结果面积**可能超过 M**。尤其当 m = M 时，几乎一定超过。`smart_resize` 不是一个严格的面积夹子。
2. 缩小分支用向下取整，两条边各自向下凑，结果面积**可能低于 m**。

所以 [m/1024, M/1024] 不是无条件的硬保证，越不越界取决于预算和图的形状。m = M 时两端都容易越界；M 较小时，极扁的图可能超过上限（上面 32×3200 的例子）。本章的 m = 65536、M = 262144 这一组，对长宽比不超过 200、短边 1 到 1499 的整数尺寸逐一穷举，没有越界（数学结论，离线穷举，只覆盖这个尺寸范围）。

**动态分辨率在解决什么问题？**固定尺寸（例如 SmolVLM 关掉分块时统一成 512×512，第 27 章）有两个毛病：扁长的公式被压扁或者四周填满白边；大图被缩小，小符号糊掉。按面积缩放保留长宽比，token 数跟着内容走。代价是 batch 里每张图的 token 数不同，要靠占位符机制把它们安排进序列（32.7 节）。

### 32.2.4 数字例子：32×298 怎样变成 75 个 token

还是那张 val 第一张图，h = 32，w = 298。

1. 就近取整：h₀ = round(1.0)·32 = 32，w₀ = round(9.3125)·32 = 9·32 = 288。面积 9216 < 65536，走放大分支。
2. s = √(65536 / (32·298)) = √(65536 / 9536) ≈ √6.8725 ≈ 2.6215。
3. H' = ⌈32·2.6215/32⌉·32 = ⌈2.62⌉·32 = 3·32 = 96。
4. W' = ⌈298·2.6215/32⌉·32 = ⌈781.2/32⌉·32 = ⌈24.41⌉·32 = 25·32 = 800。
5. 面积 96·800 = 76800，比 65536 多了 17%，这就是向上取整的溢出。
6. N_patch = 6·50 = 300，N_vis = 76800/1024 = 75。

和【G-flow】的 `resized_hw = [96, 800]`、`grid = [[1, 6, 50]]`、`merged_tokens = 75` 完全一致。原图被放大了约 2.6 倍：原来 32 像素高的公式，现在占 6 行 patch、3 行视觉 token。

tour 部分 2 把这张图按 0.5、1、2、4 倍缩放，再配三个面积上限，算出 12 个组合（`gpu/remote_runs/main/resolution.json`，数学结论，只做尺寸计算）：

| 倍率 | 输入 h × w | 上限 65536 | 上限 262144 | 上限 1048576 |
|---|---|---|---|---|
| 0.5 | 16 × 149 | 96×800，75 | 96×800，75 | 96×800，75 |
| 1 | 32 × 298 | 96×800，75 | 96×800，75 | 96×800，75 |
| 2 | 64 × 596 | 96×800，75 | 96×800，75 | 96×800，75 |
| 4 | 128 × 1192 | 64×768，48 | 128×1184，148 | 128×1184，148 |

（下限都是 65536；单元格写"缩放后尺寸，视觉 token 数"。）

读这张表：

- **平台。**前三行完全相同。原图不管是 16 高还是 64 高，面积都不到 65536，都被放大到同一个尺寸。对小公式来说，动态分辨率其实就是统一放大到下限面积，视觉 token 被下限钉死在 75 附近。
- **下溢。**最后一行第一格：下限和上限都是 65536，图太大走缩小分支，结果 64×768 = 49152，比下限还小，只有 48 个 token。这就是上面说的第二个意外。
- **上限够大就不动。**128×1192 就近取整成 128×1184 = 151552，在 [65536, 262144] 之间，不缩放。

整个 val 上，Qwen3.5 的视觉 token 最少 65、最多 96（本项目观察，`gpu/remote_runs/main/processor.json` 180 行的 `merged_tokens`），平均 75.4（【G-val】的 `mean_visual_tokens`）。

> **停下来想一想：** 公式图被放大 2.6 倍，像素是插值出来的，没有新信息。那为什么不干脆把下限调小，让小图保持原样、少用 token？第 27 章把下限改成 1024 做过类似的对照。提示：想想一个 32 像素高的公式在 16×16 的 patch 网格上只占几行，以及预训练时模型习惯看多大的字。

### 32.2.5 对照代码：`smart_resize`

`vlm_anatomy.py` 第 42 到 55 行（`# >>> 练习 32.1`）：

```python
def smart_resize(height, width, factor=32, min_pixels=65536, max_pixels=262144):
    if min(height, width, factor, min_pixels) <= 0 or max_pixels < min_pixels:
        raise ValueError(...)
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be <= 200")
    h, w = round(height/factor)*factor, round(width/factor)*factor
    if h*w > max_pixels:
        scale = math.sqrt(height*width/max_pixels)
        h, w = [max(factor, math.floor(x/scale/factor)*factor) for x in (height, width)]
    elif h*w < min_pixels:
        scale = math.sqrt(min_pixels/(height*width))
        h, w = [math.ceil(x*scale/factor)*factor for x in (height, width)]
    return h, w
```

先对一下名字。代码里的 `height`、`width` 是推导里的原图 h、w；代码里的 `h`、`w` 却是推导里的 h₀、w₀，经过某个分支以后又被改写成 H'、W'。同一个字母在推导和代码里指的不是同一个量，读的时候别混。

逐行对照 32.2.3 节：

- 第 44 到 47 行：参数检查。长宽比超过 200 直接报错，这条规则来自官方实现（例如 1×201 的图会被拒绝，测试 `test_resize_rejects_invalid`）。
- 第 48 行：第一步，就近取整。`round` 是 Python 内置的，自带五成双。
- 第 49 到 51 行：缩小分支。注意 `scale` 用的是**原图**面积 `height*width`，不是取整后的 `h*w`；`floor` 之外还有 `max(factor, …)`，保证至少一个 32。
- 第 52 到 54 行：放大分支，`ceil`。
- 判断条件用的是取整后的 `h*w`，缩放比例用的是原图的 `height*width`。两者混用是官方规则，复刻时必须照抄。

`factor` 由调用方传入 `patch_size*merge_size`（第 68 行），不是写死的 32。

测试 `test_resize_rounding_and_budget` 钉住了三件事：`smart_resize(80, 80, min_pixels=1, max_pixels=100000)` 等于 (64, 64)，因为 80/32 = 2.5，五成双取 2；1000×2000 的图缩完以后边长是 32 的倍数且面积不超过上限；37×239 在 m = M = 65536 时面积超过 65536。

### 32.2.6 常见错误

| 错误 | 现象 | 怎样发现 |
|---|---|---|
| 把 `min_pixels`/`max_pixels` 当边长 | 设 `max_pixels=512` 以为"最长边 512"，结果本章复刻因上下限倒置（默认 `min_pixels=65536` > 512）直接报 `ValueError`；官方 `smart_resize` 不查这一条，32×298 的示例图会被缩成 32×64，只剩 2 个视觉 token（CPU 复算） | 打印 `image_grid_thw`；transformers 里它们对应 `size={"shortest_edge", "longest_edge"}`，名字有误导性，但含义是**面积** |
| 自己写成"四舍五入" | 80×80 算成 96×96，和官方差一格 | `test_resize_rounding_and_budget` |
| 以为面积在任何预算下都落在 [m, M] 里 | 预算较紧时按 M/1024 估算显存上限，实际多出几个 token | 用 m = M 的组合试一下，看 37×239 的结果 |
| 用 `patch_size` 当 factor | 边长只凑到 16 的倍数，合并时 grid 不能被 2 整除 | `expand_image_placeholder` 会报"不能按 merge_size 平方合并" |
| 缩放比例用了取整后的面积 | 个别尺寸差 32 像素 | `test_processor_elementwise` 里的 37×239、16×3000 两个尺寸 |

---

## 32.3 processor（二）：像素怎样排成 `pixel_values`

### 32.3.1 动机

第 29 章的 ViT 输入是 `[B, C, H, W]`，在模型里面切 patch。Qwen 的 `pixel_values` 却是一个二维张量 `[300, 1536]`：没有 batch 维，没有高宽维，patch 已经在 processor 里切好、展平、排好队了。

这带来两个问题：

1. 1536 是怎么来的？为什么里面有个 2？
2. 300 行按什么顺序排？顺序错了会怎样？

第二个问题更危险：顺序错了，形状完全正确，程序不报错，模型照样输出，只是变笨了。这类 bug 要靠逐位对拍才抓得到。

### 32.3.2 直觉：先按 2×2 小方阵分组，再排队

想象一块瓷砖墙，6 行 50 列。工人要把瓷砖一块块搬上传送带，下游的机器（merger）每次从传送带上拿**连续的 4 块**，拼成一块大砖。

如果工人一行一行地搬（普通的行优先），机器拿到的 4 块是同一行里相邻的 4 块，拼出来是一条 1×4 的长条。我们要的是 2×2 的小方块。

所以工人得换个搬法：先把墙划成 3×25 个 2×2 的小方阵，一个方阵一个方阵地搬，方阵内部按"左上、右上、左下、右下"的顺序。这样机器每次拿连续 4 块，正好是一个 2×2 方阵。

这叫"merge 邻域优先"排列。它是 processor 和 merger 之间的约定，两边谁改了都会出错。

### 32.3.3 推导：1536 和排列

**每一行的 1536。** Qwen 的视觉塔原本是为视频设计的：patch 不仅是 16×16 的空间块，还在时间上跨 2 帧（`temporal_patch_size = 2`）。一张静态图怎么办？**复制一份当第二帧。**于是每个 patch 有

$$
C\times T\times P\times P=3\times2\times16\times16=1536
$$

个数（数学结论）。复制之后时间维有 2 帧，除以 2 得 `grid_t = 1`，所以 `image_grid_thw` 的第一个数是 1，不是 2。

因为两帧完全相同，patch 嵌入那个核为 (2, 16, 16) 的三维卷积，效果等于一个二维卷积，核是两个时间切片的权重之和（数学结论）。静态图白白多花了一倍的乘法，换来和视频共用一套权重。

**归一化。** 像素先乘 1/255 变到 [0, 1]，再减均值 0.5、除标准差 0.5，变到 [−1, 1]。均值和标准差来自 processor 配置（`image_mean = image_std = (0.5, 0.5, 0.5)`）。

**排列。** 设缩放后图是 `(3, H', W')`，网格 gh = H'/16，gw = W'/16，合并尺寸 m = 2。注意这里的 m 是 `merge_size`，和 32.2 节的面积下限 m 不是同一个量，本节和 32.3.4 节的 m 都指合并尺寸。把高拆成 (gh/m, m, 16)，宽拆成 (gw/m, m, 16)：

```text
(3, gh/m, m, 16, gw/m, m, 16)
```

七个轴依次叫 C、Bh、mh、ph、Bw、mw、pw（B 是"第几个方阵"，m 是"方阵内第几个"，p 是"patch 内第几个像素"）。我们要的顺序是：先按方阵（Bh、Bw），再按方阵内位置（mh、mw），最后是一个 patch 的内容（C、T、ph、pw）：

```text
transpose → (Bh, Bw, mh, mw, C, ph, pw)
插入 T=2  → (Bh, Bw, mh, mw, C, T, ph, pw)
reshape   → (gh·gw, C·T·16·16)
```

### 32.3.4 数字例子：4×4 的编号墙

用一个能手算的小例子：4×4 个 patch，每个 patch 只有 1 个像素（p = 1），m = 2，像素值就是自己的行优先编号 0 到 15：

```text
 0  1  2  3
 4  5  6  7
 8  9 10 11
12 13 14 15
```

划成 2×2 个方阵：左上 {0, 1, 4, 5}，右上 {2, 3, 6, 7}，左下 {8, 9, 12, 13}，右下 {10, 11, 14, 15}。方阵按行优先、方阵内也按行优先，排出来是：

```text
0 1 4 5 | 2 3 6 7 | 8 9 12 13 | 10 11 14 15
```

merger 拿连续 4 个，正好是一个方阵。如果按普通行优先排成 0, 1, 2, …, 15，merger 拿到的第一组是 {0, 1, 2, 3}，是第一行的一条横带。

再看形状。真实例子里 gh = 6、gw = 50：

```text
(3, 96, 800)
→ reshape   (3, 3, 2, 16, 25, 2, 16)
→ transpose (3, 25, 2, 2, 3, 16, 16)
→ 插 T      (3, 25, 2, 2, 3, 2, 16, 16)
→ reshape   (300, 1536)
```

第一维 3·25·2·2 = 300，第二维 3·2·16·16 = 1536，和【G-flow】的 `pixel_values: [300, 1536]` 一致。

### 32.3.5 对照代码：`replica_processor`

`vlm_anatomy.py` 第 59 到 78 行（`# >>> 练习 32.2`）：

```python
image = image.convert("RGB")
h, w = smart_resize(image.height, image.width, patch_size*merge_size, min_pixels, max_pixels)
image = image.resize((w, h), resample=Image.Resampling.BICUBIC)
pixels = (np.asarray(image).astype(np.float64)*(1/255)).astype(np.float32)
pixels = ((pixels-np.asarray(mean, dtype=np.float32))/np.asarray(std, dtype=np.float32)).transpose(2, 0, 1)
p, m, t = patch_size, merge_size, temporal_patch_size
gh, gw = h//p, w//p
blocks = pixels.reshape(3, gh//m, m, p, gw//m, m, p).transpose(1, 4, 2, 5, 0, 3, 6)
blocks = np.repeat(blocks[..., None, :, :], t, axis=-3)
return {"pixel_values": blocks.reshape(gh*gw, 3*t*p*p),
        "image_grid_thw": np.array([[1, gh, gw]], dtype=np.int64), "resized_hw": [h, w]}
```

- 第 69 行：PIL 的 `resize` 参数顺序是 **(宽, 高)**，和 numpy 的 (高, 宽) 相反。插值用双三次（BICUBIC），和官方 PIL 后端一致。插值本身借用 PIL，不自己写。
- 第 71 行：先转 float64 乘 `1/255`，再转 float32。看起来多此一举，但官方 PIL 后端就是这个顺序。如果改成在 float32 里乘 `np.float32(1/255)`，256 个像素值里有 126 个最后一位不同，归一化后最大差 1.19e-07，逐位相等就过不了；float32 里直接除以 255 倒是和官方逐位相同（CPU 穷举 0 到 255 复算）。
- 第 72 行：减均值、除标准差，再把 HWC 转成 CHW。
- 第 75 行：`reshape` 成七个轴，`transpose(1, 4, 2, 5, 0, 3, 6)` 就是 32.3.3 节的 (Bh, Bw, mh, mw, C, ph, pw)。
- 第 76 行：在倒数第三个位置插一个长度为 1 的轴，再沿它重复 t = 2 次，得到 (…, C, T, ph, pw)。静态图复制成两帧就是在这一步做的。
- 第 77 到 78 行：展平，并返回 `image_grid_thw = [[1, gh, gw]]`。

**对拍的结果。** tour 部分 1 把复刻和 transformers 的 `Qwen2VLImageProcessorPil` 跑在同一张图上，比较 `pixel_values` 有几个元素不相等、最大绝对误差多少、grid 是否相同；有 Qwen3.5 缓存时，还比较套完模板以后的 `input_ids` 和 `attention_mask`。在 val 的 180 张原图上：不相等的元素 0 个，最大绝对误差 0.0，grid 和 `input_ids` 全部相同（本项目观察，`gpu/remote_runs/main/processor.json`，processor 在 CPU 上 fp32 运行）。这里的"逐位"指 float32 严格相等，不是 `allclose`；它只对这套 Pillow 版本和 RGB 静态图路径成立（lab 第 1 节）。

测试 `test_patch_order_independent_anchor` 就是 32.3.4 节的 4×4 编号墙：它不和 transformers 比，而是直接断言顺序是 `[0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]`，并检查每个通道的两帧完全相同。这叫**独立锚点**：万一官方实现某天改了顺序，和官方对拍的测试会跟着"错得一致"，独立锚点不会。

### 32.3.6 常见错误

| 错误 | 现象 | 怎样发现 |
|---|---|---|
| 按普通行优先排 patch | 形状对、不报错，模型输出变差但不是乱码 | `test_patch_order_independent_anchor`；和官方对拍时 `unequal_elements` 很大 |
| 忘了复制第二帧 | 每行 768 个数，patch 嵌入的 `view` 报形状错 | 报错信息里的 768 和 1536 |
| 以为 `grid_t = 2` | 占位符多算一倍，150 个对 75 个特征 | 语言模型报 "Image features and image tokens do not match" |
| PIL `resize((h, w))` 写反 | 图被转置拉伸，grid 的行列对调 | 打印 `resized_hw` 和 grid |
| float32 里乘 `1/255` | `max_abs_error` 是 1e-7 量级而不是 0 | 逐位对拍 |
| 用 torchvision 后端和 PIL 后端对拍 | 插值实现不同，误差更大 | 先确认 `type(processor.image_processor).__name__` |

---

## 32.4 三段结构：形状与参数量

### 32.4.1 动机

"0.8B"这个名字告诉你大概多大。但做 LoRA（第 33 章）时你要决定冻结哪一段，算显存时要知道每段多重，读技术报告时要知道"视觉塔 600M"是什么分量。第 17 章逐项手算过 GPT-2 的参数，这里对 VLM 做同样的事。

一个 VLM 的参数分三段：视觉塔、连接器、语言模型。在 Qwen3.5 里，单块最大的是语言模型的词嵌入表：词表有 24 万多个 token，每个 1024 维，光这一块就比整个视觉塔重。

### 32.4.3 推导：逐项数

下面的形状全部来自【权重头】：读 safetensors 文件开头的 JSON 头，里面写着每个张量的名字和形状，不需要加载权重。参数量是形状相乘（数学结论）。

**视觉塔（`model.visual`）**，配置：宽 768，12 层，12 头，MLP 中间层 3072，patch 16，时间 patch 2（【G-flow】的 `config.vision_config`）。

| 部件 | 形状 | 参数 |
|---|---|---|
| patch 嵌入（Conv3d） | 权重 [768, 3, 2, 16, 16] + 偏置 [768] | 768·1536 + 768 = 1,180,416 |
| 学习式位置表 | [2304, 768] | 1,769,472 |
| 每个 block：qkv | [2304, 768] + [2304] | 1,771,776 |
| 每个 block：输出投影 | [768, 768] + [768] | 590,592 |
| 每个 block：MLP | [3072, 768] + [3072]，[768, 3072] + [768] | 4,722,432 |
| 每个 block：两个 LayerNorm | 4 × [768] | 3,072 |
| 每个 block 合计 | | 7,087,872 |
| 12 个 block | | 85,054,464 |
| merger | 见下 | 12,588,544 |
| **视觉塔合计** | | **100,592,896** |

位置表有 2304 = 48² 行，对应一个 48×48 的网格。实际 grid 是多少就双线性插值到多少（transformers 5.19.0 `Qwen3_5VisionModel` 里 `interpolation_mode = "bilinear"`），也就是第 29 章 29.5 节的"插值位置表"。此外注意力里还有二维旋转位置编码，不占参数。

**连接器（`model.visual.merger`）**：

| 部件 | 形状 | 参数 |
|---|---|---|
| LayerNorm（作用在每个 patch 的 768 维上） | [768] × 2 | 1,536 |
| `linear_fc1` | [3072, 3072] + [3072] | 9,440,256 |
| `linear_fc2` | [1024, 3072] + [1024] | 3,146,752 |
| 合计 | | 12,588,544 |

3072 = 4 × 768：四个 patch 拼起来的宽度。1024 = 语言模型的宽度。

**语言模型（`model.language_model`）**，配置：宽 1024，24 层，词表 248320，词嵌入和输出层共享（`tie_word_embeddings = true`）。

| 部件 | 参数 |
|---|---|
| 词嵌入 [248320, 1024] | 254,279,680 |
| 24 层 | 498,112,320 |
| 最后的 norm | 1,024 |
| **语言模型合计** | **752,393,024** |

24 层并不一样。config 的 `layer_types` 是"三层线性注意力、一层全注意力"循环六次（`full_attention_interval = 4`）。线性注意力层（transformers 里叫 gated delta rule）每层 21,555,360 个参数，全注意力层每层 18,352,640 个参数，两者的 MLP 都是 3 × 1024 × 3584 = 11,010,048。于是 18 × 21,555,360 + 6 × 18,352,640 = 498,112,320（数学结论）。线性注意力的原理不在本书范围，这里只数参数。

**合计：**

$$
\underbrace{100{,}592{,}896}_{\text{视觉塔，含 merger}}+\underbrace{752{,}393{,}024}_{\text{语言模型}}=852{,}985{,}920
$$

和 tour 打印的 `parameters=852985920` 完全一致（本项目观察，远程日志）。

几个比例（数学结论）：

- 词嵌入占全部参数的 29.8%，约是整个视觉塔（11.8%）的 2.5 倍。
- merger 只有 1259 万，占 1.5%。
- safetensors 文件里其实有 873,438,784 个参数。多出的 20,452,864 个属于 `mtp.*`（一个额外的多 token 预测头），`AutoModelForImageTextToText` 加载时没有用到它，所以不计入。"文件里有多少"和"模型用了多少"可以不同。

**SmolVLM-256M**（【权重头】）：

| 段 | 参数 |
|---|---|
| 视觉塔（SigLIP 结构：patch 嵌入 + 位置表 1,377,024；12 层编码器 85,054,464；最后的 LayerNorm 1,536） | 86,433,024 |
| 连接器（一个无偏置线性层 [576, 12288]） | 7,077,888 |
| 语言模型（词嵌入 28,385,280；30 层 106,202,880；norm 576；输出层 28,385,280，不共享） | 162,974,016 |
| 合计 | 256,484,928 |

和日志里的 `parameters=256484928` 一致。一个有趣的巧合：SmolVLM 的 12 层视觉编码器和 Qwen3.5 的 12 个视觉 block 参数量都是 85,054,464，因为它们是同一个形状的 ViT（宽 768、12 层、MLP 3072）。形状相同不代表权重相同。SmolVLM 的视觉塔占全部参数的三分之一，比例远高于 Qwen3.5。

Qwen3-VL-2B 本地没有缓存，这里只给日志里的总数 2,127,532,032，不拆分。

### 32.4.4 数字例子：hook 看到的形状

【G-flow】在五个模块上挂了 forward hook：

| 模块 | 输入 | 输出 |
|---|---|---|
| `model.visual.patch_embed` | [300, 1536] | [300, 768] |
| `model.visual.blocks.0` | [300, 768]，关键字参数 `cu_seqlens` 形状 [2] | [300, 768] |
| `model.visual.merger` | [300, 768] | [75, 1024] |
| `model.visual`（整个视觉塔） | [300, 1536]，`grid_thw` [1, 3] | `last_hidden_state` [300, 768]，`pooler_output` [75, 1024] |
| `model.language_model` | `inputs_embeds` [1, 111, 1024]，`position_ids` [3, 1, 111] | [1, 111, 1024] |

`cu_seqlens` 形状 [2]，按它的定义内容应是 [0, 300]（推断，数值没有落盘）：一张图一段，视觉注意力只在这一段里做。batch 里有多张图时，它们被首尾相接成一个长序列，`cu_seqlens` 记录每张图的边界，防止图和图之间互相注意。这和第 29 章的 `[B, N, D]` 不同：没有 batch 维，靠边界表分段。

整个视觉塔的输出有两个：`last_hidden_state` 是 merger 之前的 300 个 patch，`pooler_output` 是 merger 之后的 75 个 token。送进语言模型的是后者。

### 32.4.5 对照代码：tour 部分 5

`tour.py` 第 261 到 281 行：

```python
for name in ("model.visual.patch_embed","model.visual.blocks.0","model.visual.merger","model.visual","model.language_model"):
    module=model.get_submodule(name)
    def hook(mod,pos,kwargs,output,name=name):
        traces[name]=dict(input=shape_tree(pos),kwargs=shape_tree({k:v for k,v in kwargs.items() if isinstance(v,torch.Tensor)}),output=shape_tree(output))
    handles.append(module.register_forward_hook(hook,with_kwargs=True))
with torch.inference_mode(): output=model(**inputs,use_cache=False)
for h in handles: h.remove()
```

- `get_submodule` 按点号路径取模块。路径怎么来？`print(model)` 一次就知道（第 17 章的第一步）。
- `with_kwargs=True` 让 hook 也能看到关键字参数。语言模型是用 `inputs_embeds=...` 关键字调用的，位置参数为空，不开这个选项就看不到 [1, 111, 1024]。
- `name=name` 是 Python 闭包的老坑：不写的话，五个 hook 都会记到最后一个名字上。
- `shape_tree` 只记形状，不留张量，所以 hook 不会额外占显存。
- 用完 `h.remove()`，否则后面每次前向都会触发。

### 32.4.6 常见错误

| 错误 | 现象 | 怎样发现 |
|---|---|---|
| 把 lm_head 再算一遍 | Qwen3.5 多出 2.5 亿 | 看 `tie_word_embeddings`；safetensors 里没有 `lm_head.weight` |
| 把 `mtp.*` 算进模型 | 总数比加载后多 2045 万 | 和 `sum(p.numel() for p in model.parameters())` 对 |
| 以为 24 层都一样 | 用一层乘 24，差几千万 | 读 `layer_types` |
| 以为视觉塔输出 [1, 75, 1024] | 自己写拼接时多一个 batch 维 | hook 打印；视觉侧没有 batch 维 |
| 闭包没绑定 `name` | 所有 hook 记录挤在一个键下 | `traces` 只有一个模块名 |

---

## 32.5 连接器：MLP、2×2 合并与 Q-Former

### 32.5.1 动机

连接器是 VLM 里最小的一段（Qwen3.5 里 1.5%），却决定了两件大事：

1. **宽度对齐**：视觉向量多宽，语言模型就得收多宽。
2. **长度**：多少个视觉 token 进语言模型。语言模型每层都要处理这些 token，序列长度直接决定算力和显存。

第 31 章 31.3.3 节算过：前缀拼接的注意力代价随视觉 token 数平方增长。这个平方代价只落在全注意力层上（Qwen3.5 语言侧 24 层里有 6 层），线性注意力层随长度线性增长。砍长度的手段不止一处：processor 调小面积预算（32.2 节）能减少 token，视觉塔内部也可以合并或剪枝。但在本章的三段式流水线里，视觉塔输出固定 N 个 token 之后，承担长度压缩的正是连接器。

### 32.5.2 直觉：三种转接头

- **LLaVA 式 MLP**：一对一翻译。每个视觉 token 单独过一个线性层（最早的 LLaVA）或两层 MLP（LLaVA-1.5），宽度变了，个数不变。像同声传译，一句对一句。
- **Qwen 式 2×2 合并 + MLP**：先把相邻 4 个 patch 拼成一条长向量，再翻译。个数变成四分之一。像把四张小照片拼成一张拼图再寄出去。SmolVLM 的"像素重排"（pixel shuffle）是同一个想法，一次并 4×4 = 16 个。
- **BLIP-2 式 Q-Former**：派固定数量的"记者"（可学习的 query）去采访全部视觉 token，每个记者交回一篇稿子。不管图多大，交回的稿子数固定。像开新闻发布会，记者人数事先定好。

前两种是位置对位置的局部操作，保留了空间结构；Q-Former 是按内容挑选的全局操作，输出个数和图的大小无关。

### 32.5.3 推导：长度、宽度和参数

记视觉塔输出 N 个 token、宽 d_v，语言模型宽 d_ℓ。

| 连接器 | 输出长度 | 参数量（主要项） |
|---|---|---|
| 线性投影（LLaVA） | N | d_v·d_ℓ |
| 两层 MLP（LLaVA-1.5） | N | d_v·d_ℓ + d_ℓ² |
| k×k 合并 + 两层 MLP（Qwen） | N/k² | (k²d_v)² + k²d_v·d_ℓ |
| k×k 像素重排 + 线性（SmolVLM） | N/k² | k²d_v·d_ℓ |
| Q 个 query 的交叉注意力（Q-Former） | Q | 与 N 无关；含注意力和 FFN |

代入 Qwen3.5：k = 2，d_v = 768，d_ℓ = 1024。fc1 是 3072×3072，fc2 是 3072×1024，和 32.4.3 节的表一致。合并本身不需要参数，只是 `view`：因为 processor 已经把每个 2×2 方阵排成连续 4 行（32.3 节），`view(-1, 3072)` 就把它们拼起来了。

代入 SmolVLM：一个 512×512 的图块切成 32×32 = 1024 个 patch，k = 4（config 的 `scale_factor = 4`），合并后 1024/16 = 64 个 token，宽 16 × 768 = 12288，再线性投影到 576。processor 配置里的 `image_seq_len = 64` 就是这样来的（数学结论）。

合并改长度，MLP 改宽度。一个只有 MLP 的连接器，不管多深，都不会让序列变短。

**真实的 Q-Former** 比本章的缩小版复杂得多：BLIP-2 用 32 个 query，Q-Former 是一个用 BERT 初始化的多层 Transformer，query 之间有自注意力，和视觉特征有交叉注意力，还要用三个目标预训练（论文报告，待验证细节）。本章的 `QueryConnector` 只保留了"固定 query + 一次交叉注意力 + FFN"这个骨架。

### 32.5.4 数字例子：两个缩小版连接器

tour 部分 3 用 d_v = 32、d_ℓ = 48 的缩小版，输入 16、64、256 个 token，batch 2（本项目观察，随机初始化的结构探针，训练步数 0，`gpu/remote_runs/main/connectors.json`）：

| 连接器 | 输入 | 输出 | 参数 |
|---|---|---|---|
| MLPConnector | [2, 16, 32] | [2, 16, 48] | 3,936 |
| QueryConnector | [2, 16, 32] | [2, 8, 48] | 20,832 |
| MLPConnector | [2, 64, 32] | [2, 64, 48] | 3,936 |
| QueryConnector | [2, 64, 32] | [2, 8, 48] | 20,832 |
| MLPConnector | [2, 256, 32] | [2, 256, 48] | 3,936 |
| QueryConnector | [2, 256, 32] | [2, 8, 48] | 20,832 |

参数可以手算（数学结论）：

- MLP：32·48 + 48 = 1,584；48·48 + 48 = 2,352；合计 3,936。
- Query：query 8·48 = 384；视觉投影 32·48 + 48 = 1,584；多头注意力的 qkv 3·48·48 + 3·48 = 7,056、输出投影 48·48 + 48 = 2,352；LayerNorm 96；FFN 48·96 + 96 = 4,704 和 96·48 + 48 = 4,656；合计 20,832。

MLP 的输出长度跟着输入走，Query 的输出永远是 8。两者参数量都和输入长度无关。

### 32.5.5 对照代码

`vlm_anatomy.py` 第 82 到 107 行（`# >>> 练习 32.3`）：

```python
class MLPConnector(nn.Module):
    def __init__(self, vision_dim=32, language_dim=48):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(vision_dim, language_dim), nn.GELU(), nn.Linear(language_dim, language_dim))

class QueryConnector(nn.Module):
    def __init__(self, vision_dim=32, language_dim=48, queries=8, heads=4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, queries, language_dim)*.02)
        self.kv = nn.Linear(vision_dim, language_dim)
        self.attn = nn.MultiheadAttention(language_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(language_dim)
        self.ffn = nn.Sequential(...)

    def forward(self, x):
        q = self.queries.expand(x.shape[0], -1, -1)
        kv = self.kv(x)
        z = self.norm(q + self.attn(q, kv, kv, need_weights=False)[0])
        return z + self.ffn(z)
```

- `MLPConnector`：`nn.Linear` 只作用在最后一维，所以序列长度原样保留。
- `self.queries` 的形状是 `[1, 8, 48]`，`expand` 到 batch 大小，不复制内存。
- `self.kv` 先把视觉向量投到 48 维，注意力的查询来自 query，键和值来自视觉。输出长度跟查询走，所以是 8。
- 残差 + LayerNorm + FFN：一个最小的交叉注意力块。

再对照真实的 Qwen merger（transformers 5.19.0 `Qwen3_5VisionPatchMerger.forward`）：

```python
x = self.norm(x).view(-1, self.hidden_size)     # [300, 768] → [75, 3072]
x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))  # → [75, 1024]
```

先对每个 patch 的 768 维做 LayerNorm，再 `view` 成 3072 维（这一步就是 2×2 合并），然后两层 MLP。三行代码，没有任何挑选。

测试 `test_connectors_use_image_and_backprop` 检查三件事：输出形状对、梯度能传回输入且有限非零、把输入换成全零以后输出会变。最后一条的用处见下面的常见错误。

### 32.5.6 常见错误

| 错误 | 现象 | 怎样发现 |
|---|---|---|
| 以为 MLP 能压缩序列 | 加深连接器以后显存不降 | 打印输出形状 |
| 形状对就以为"用到了图" | 连接器其实没接上视觉输入，输出全靠 query 或偏置 | 输入换成全零，输出不变就是没接上（测试的最后一个断言） |
| 把 merger 输入的 300 当成语言侧的视觉 token 数 | 占位符放了 300 个，特征只有 75 个 | 语言模型报 token 数和特征数不匹配 |
| 用缩小版 Q-Former 的结果评价 BLIP-2 | 随机结构的梯度范数被当成"质量" | 训练步数为 0，只能说明结构能反传 |

---

## 32.6 chat template 与图像占位符

### 32.6.1 动机

你写的提示词是一句英文。模型实际读到的是 111 个 token。中间多出来的那些是谁加的、加在哪里？

它们来自 chat template。这是一段存在模型目录里的 Jinja 模板（Qwen3.5 是 `chat_template.jinja`），把"角色 + 内容"的消息列表渲染成模型训练时见过的那种格式。图片在模板里只是一个占位符，processor 再按 grid 把它展开成一排。

模板错一个字符，模型就像在读一份格式陌生的试卷，输出会变得奇怪。第 18 章说过提示格式对 LLM 的影响，这里又多了图片占位符这一层。

### 32.6.3 推导：占位符展开成多少个

**Qwen 系列。**模板里图片写成

```text
<|vision_start|><|image_pad|><|vision_end|>
```

processor 把唯一的 `<|image_pad|>` 换成 N_vis 个：

$$
N_{\mathrm{vis}}=\frac{\mathrm{grid}_t\cdot\mathrm{grid}_h\cdot\mathrm{grid}_w}{\mathrm{merge}^2}=\frac{\operatorname{prod}(\mathrm{image\_grid\_thw})}{4}
$$

`<|vision_start|>` 和 `<|vision_end|>` 是普通的特殊 token，各占 1 个，不算视觉 token。

**SmolVLM。**模板里写的是 `<image>`。processor 把图切成若干个 512×512 的块，再加一张缩到 512 的全图，每块展开成

```text
<fake_token_around_image><row_i_col_j><image>×64
```

全图用 `<global-img>` 代替行列标记。视觉 token 数 = 64 × (块数 + 1)。

### 32.6.4 数字例子：111 和 364

**Qwen3.5。**tour 用的消息是一条 user 消息，内容先图后文，`add_generation_prompt=True`，`enable_thinking=False`。渲染出的字符串是（【G-val】JSON 里的 `formatted_prompt`）：

```text
<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Transcribe the mathematical formula in this image into LaTeX. Output only the LaTeX formula, without explanation or markdown.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
```

（`\n` 表示换行。）分词以后逐段数（本项目观察，作者在本地 CPU 上只加载 processor 和词表、不加载权重，对一张 32×298 的白图复现，长度与【G-val】第一条的 111 一致）：

| 段 | token | 个数 |
|---|---|---|
| 开头 | `<\|im_start\|>` `user` `\n` `<\|vision_start\|>` | 4 |
| 图片 | `<\|image_pad\|>` × 75 | 75 |
| 图片结束 | `<\|vision_end\|>` | 1 |
| 提示词 | `Trans` `cribe` `Ġthe` … `Ġmarkdown` `.`（`Ġ` 是字节级 BPE 里的空格，第 17 章） | 22 |
| 轮次切换 | `<\|im_end\|>` `\n` `<\|im_start\|>` `assistant` `\n` | 5 |
| 空思考块 | `<think>` `\n\n` `</think>` `\n\n` | 4 |
| 合计 | | 111 |

36 个非图片 token 加 75 个图片 token。图片占位排在下标 4 到 78。

**Qwen3-VL。**它的 `formatted_prompt` 在 `assistant\n` 处就结束了，没有空思考块（【G-val】的 `qwen3vl_val_original.json`）。少 4 个 token，所以同一张图是 107。这就是 32.1.4 节那 4 个 token 的来历。

把 32.2 节和这里串起来，任何一张图进 Qwen 的输入长度都能在跑之前算出来：先用 `smart_resize` 得到 H'×W'，再算 N_vis = H'W'/1024，最后加上模板里固定的文字 token。用本章的提示词，Qwen3.5 的输入长度是 N_vis + 36，Qwen3-VL 是 N_vis + 32（数学结论，前提是提示词和模板不变，单图）。比如 val 上视觉 token 最多的那张图有 96 个，进 Qwen3.5 就是 132 个 token。

**为什么 Qwen3.5 要一个空的思考块？**Qwen3.5 默认会先思考再作答。`enable_thinking=False` 的做法是让模板替模型写好一个空的 `<think></think>`，模型就直接进入作答。忘了这个开关，模型会先输出一大段推理，256 个 token 的上限很快被用完。

**SmolVLM。**同一张 32×298 的图，`pixel_values` 是 [1, 5, 3, 512, 512]：4 个图块加 1 张全图（【G-val】第一条的 `pixel_shape`）。展开以后的 token 序列（本地 CPU 复现，只加载 processor）：

```text
<|im_start|> User : <fake_token_around_image> <row_1_col_1> <image>×64
                    <fake_token_around_image> <row_1_col_2> <image>×64
                    <fake_token_around_image> <row_1_col_3> <image>×64
                    <fake_token_around_image> <row_1_col_4> <image>×64 \n\n
                    <fake_token_around_image> <global-img> <image>×64 <fake_token_around_image>
Trans c ribe … <end_of_utterance> \n Ass istant :
```

视觉 token 5 × 64 = 320，输入总长 364，和【G-val】一致。扁长的公式被横着切成 1 行 4 列。整个 val 上 SmolVLM 的视觉 token 在 320 到 832 之间（5 到 13 块），平均 425.2（【G-val】的逐条 `visual_tokens` 和 `mean_visual_tokens`）。图块怎样决定行列数，细节以 transformers 的 `Idefics3ImageProcessor` 源码为准（待验证）；这里只引用实测的块数。

注意两个模型告诉语言模型图像位置的办法不同：SmolVLM 用文字 token `<row_1_col_2>` 告诉语言模型"这一块在第 1 行第 2 列"，Qwen 用三维位置编号（32.7 节）。

### 32.6.5 对照代码

模板渲染在 `tour.py` 第 158 到 164 行的 `inputs_for`：

```python
messages=[{"role":"user","content":[{"type":"image"},{"type":"text","text":PROMPT}]}]
prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
inputs=processor(text=prompt,images=[image],return_tensors="pt")
```

- `{"type": "image"}` 不带图片本身，只是告诉模板"这里有一张图"。图片本身在下一行作为 `images=[image]` 传给 processor。
- `tokenize=False` 先拿到字符串，方便打印和存档；然后由 processor 一次性完成展开占位符、分词和处理图像。
- `enable_thinking=False` 对 SmolVLM 的模板没有作用，多传不会出错。

占位符展开的复刻在 `vlm_anatomy.py` 第 132 到 140 行（`# >>> 练习 32.4`）：

```python
def expand_image_placeholder(prompt, image_token, grid, merge_size=2):
    if prompt.count(image_token) != 1:
        raise ValueError("静态单图实验要求恰好一个 image placeholder")
    count = int(np.prod(grid))
    if count <= 0 or count % (merge_size**2):
        raise ValueError("grid 的 patch 数必须能按 merge_size 平方合并")
    return prompt.replace(image_token, image_token*(count//merge_size**2))
```

- `np.prod(grid)` 是 patch 数（300），除以 `merge_size**2` 才是视觉 token 数（75）。练习 32.4 的标题就是这个提醒：**不要把 patch 数当 token 数。**
- 展开在**字符串**上做，然后交给官方 tokenizer 分词。tour 部分 1 比较这样得到的 `input_ids` 和官方 processor 的输出，180 张图全部相同（【G-flow】的 `input_ids_equal`）。
- 测试 `test_placeholder_uses_merged_count`：grid [1, 4, 6] 展开成 6 个占位符；没有占位符、或 grid [1, 3, 3] 不能被 4 整除，都报错。

### 32.6.6 常见错误

| 错误 | 现象 | 怎样发现 |
|---|---|---|
| 忘了 `add_generation_prompt=True` | 序列停在 `<\|im_end\|>\n`，模型可能续写一轮 user | 打印 `prompt` 结尾 |
| Qwen3.5 忘了 `enable_thinking=False` | 先输出一段推理，频繁达到长度上限 | `truncated` 计数暴涨；看原始输出开头 |
| 自己在提示词里手写 `<\|image_pad\|>`，又让 processor 展开 | 两个占位符，数量对不上 | 复刻函数的"恰好一个"检查；官方报不匹配 |
| 把 `apply_chat_template(tokenize=True)` 返回的 token ID 再当 `text` 交给 processor | `tokenize=True` 时框架已经调用过一次 processor（消息带图片数据时会处理图像、展开占位符、分词），默认只返回 ID；把 ID 再当文本处理是输入类型错误 | 二选一：`tokenize=False` 拿字符串再交给 processor，或 `tokenize=True, return_dict=True` 一步拿到全部张量 |
| 把 `<\|vision_start\|>` 也算进视觉 token | 统计的视觉 token 比实际多 2 | 只数 `input_ids == image_token_id` 的位置（tour 第 206 行） |

---

## 32.7 视觉 token 怎样进入语言模型

### 32.7.1 动机

第 31 章的做法简单粗暴：`torch.cat((visual, tok(inputs)), dim=1)`，视觉向量排在最前面。真实 VLM 要处理多张图、图文交错、每张图 token 数不同的情况，硬拼接会让下标管理变成噩梦。

Qwen 和 SmolVLM 的做法是：先把占位符当普通 token 查嵌入表，再把这些位置的嵌入整行替换成视觉向量。这一节把这一步拆开。

### 32.7.2 直觉：先排座位，再请人入座

前缀拼接像临时加椅子：视觉 token 来了，在队伍最前面加一排。占位符机制像提前排好座位表：processor 已经在第 4 到 78 号座位贴了"留座"的条子，语言模型只要把 75 位视觉嘉宾按顺序请到这些座位上。座位表上还写了人数，人数不对当场就会发现。

### 32.7.3 推导：替换与检查

记文字嵌入表 E，输入 `input_ids` 长 L，视觉向量 V 形状 [N_vis, d]。

1. 所有 token 查表：X = E[input_ids]，形状 [1, L, d]。占位符也查到一个向量，但马上会被覆盖。
2. 找出占位符位置：mask = (input_ids == image_token_id)。
3. **检查**：mask 里 True 的个数 × d 必须等于 V 的元素数。不等就报错。
4. 按顺序替换：X[mask] = V。第 k 个占位符拿到 V 的第 k 行。

结果和第 31 章的拼接是同构的：视觉向量在语言模型眼里就是一段普通的输入嵌入，之后一起走因果 Transformer。差别只在于视觉段前面多了 4 个文字 token（`<|im_start|>user\n<|vision_start|>`），后面跟着 32 个文字 token。因果 mask 下，后面的文字仍然能看到全部 75 个视觉 token。

**位置编号。**第 31 章进了 GPT 就按 0, 1, …, S−1 加一维 RoPE。Qwen 的 `position_ids` 形状是 [3, 1, L]：每个 token 有三个坐标（时间、行、列），叫 M-RoPE（多模态旋转位置编码，Qwen2-VL 技术报告提出）。规则（transformers 5.19.0 `Qwen3_5Model.get_rope_index`）：

- 文字 token：三个坐标相同，都是"当前位置"，逐个加 1。
- 图片 token：时间坐标 = 起点；行坐标 = 起点 + 合并后网格的行号；列坐标 = 起点 + 列号。
- 图片之后，"当前位置"跳到 起点 + max(合并后行数, 列数)。

### 32.7.4 数字例子：111 个位置

用 32.6.4 节的序列。合并后网格是 3 行 25 列（6/2，50/2）。

| 下标 | 内容 | 时间 | 行 | 列 |
|---|---|---|---|---|
| 0 到 3 | `<\|im_start\|>` `user` `\n` `<\|vision_start\|>` | 0 到 3 | 0 到 3 | 0 到 3 |
| 4 | 第 1 个视觉 token（第 0 行第 0 列） | 4 | 4 | 4 |
| 5 | 第 2 个视觉 token（第 0 行第 1 列） | 4 | 4 | 5 |
| … | | | | |
| 28 | 第 25 个（第 0 行第 24 列） | 4 | 4 | 28 |
| 29 | 第 26 个（第 1 行第 0 列） | 4 | 5 | 4 |
| … | | | | |
| 78 | 第 75 个（第 2 行第 24 列） | 4 | 6 | 28 |
| 79 到 110 | `<\|vision_end\|>` 和后面 31 个文字 token | 29 到 60 | 29 到 60 | 29 到 60 |

图片之后的起点是 4 + max(3, 25) = 29。整个序列 111 个 token，最大位置编号只有 60（数学结论，按上述规则手算；`position_ids` 的形状 [3, 1, 111] 是【G-flow】实测，具体数值本章没有落盘，练习里请你验证）。

这里也能看出视觉 token 内部的顺序：合并后网格按行优先排，第 0 行的 25 个，再第 1 行。这和 32.3 节的方阵按行优先是一回事：processor 排的是 patch，merger 之后每个方阵变成一个 token，token 自然按方阵的行优先顺序排列。

### 32.7.5 对照代码

transformers 5.19.0 的 `Qwen3_5Model.forward` 里，相关的几行是：

```python
if inputs_embeds is None:
    inputs_embeds = self.get_input_embeddings()(input_ids)
...
image_embeds = torch.cat(mm_encoder_outputs["image"].pooler_output, dim=0)
image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
```

`get_placeholder_mask` 里的检查：

```python
special_image_mask = input_ids == self.config.image_token_id
n_image_tokens = special_image_mask.sum()
special_image_mask = special_image_mask.unsqueeze(-1)
torch_compilable_check(
    n_image_tokens * inputs_embeds.shape[-1] == image_features.numel(),
    f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {image_features.shape[0]}")
```

逐行对照 32.7.3 节：

- 第一段：步骤 1，所有 token 查表。
- `pooler_output` 是 merger 之后的 [75, 1024]（32.4.4 节）。多张图时是一个列表，`cat` 成一长条。
- `image_token_id` 是 248056（【G-flow】的 config），也就是 `<|image_pad|>`。
- `unsqueeze(-1)` 把 mask 变成 [1, 111, 1]，`masked_scatter` 会把它广播到 [1, 111, 1024]，然后按顺序用 `image_embeds` 的元素填满所有 True 的位置。
- 检查比较的是**元素数**：75 × 1024 = 76800。

对照第 31 章 `formula_ocr.py` 第 63 行：

```python
embeds = torch.cat((visual, self.gpt.tok(inputs)), dim=1)
```

一行拼接 vs. 查表 + 找位置 + 检查 + 填入。后者多了几步，换来三样东西：图片可以出现在任意位置；一条序列里可以有多张图；数量不对时有明确的报错。

### 32.7.6 常见错误

| 错误 | 现象 | 怎样发现 |
|---|---|---|
| 绕过完整模型，把查表得到的 `inputs_embeds` 直接交给 `language_model` | 模型照样输出，但完全不看图（占位位置里只有 `<\|image_pad\|>` 的嵌入） | 换一张图，输出不变（第 33 章的视觉依赖检查） |
| 占位符数和特征数对不上 | "Image features and image tokens do not match" | 报错信息里的两个数：常见的是 300 对 75（忘了除以 4） |
| 自己算 `position_ids` 时按 0…110 编号 | 不报错，输出变差 | 和 `get_rope_index` 的结果比；最大值应为 60 |
| batch 里左右 padding 不一致 | 个别样本位置错位 | `get_rope_index` 会用 `attention_mask` 跳过 padding；本章 batch 固定为 1 避开这个问题 |
| 以为视觉 token 在语言模型里也是双向注意 | 写 prefix-LM mask 却不知道官方是因果的 | 读 `attention_mask` 的构造；第 31 章练习 31.8 |

---

## 32.8 零样本公式 OCR：实测

### 32.8.1 动机

拆完结构，要回答项目关心的问题：不训练，这三个模型能把公式读对多少？这是第 33 章 LoRA 的起点，也是 R3 报告里的零样本基线。

公平比较要满足第 22 章和第 31 章反复强调的几条：同一批题目、同一个提示词、同一种解码、同一个评分脚本。三个模型的 tokenizer、模板、切图方式各不相同，这些属于考生自带的文具，不强求统一（lab 第 0 节：比较的是完整部署配置，不是等视觉 token 预算实验）。

### 32.8.3 实验设置

- **数据**：OleehyO/latex-formulas 的 cleaned_formulas（许可 openrail），按第 6 章定义、第 12 章 12.3.5 节核验过的 latex-v1 规则先在全语料上分组划分，再取 val/test 的前缀并去重，得到 val 189 条、test 188 条（`gpu/data/metadata.json`）；val 和 test 交集为 0（`gpu/remote_runs/data_audit_copy.json`）。主评测用 val 前 180 条；test 只用 100 条，只评一次。
- **提示词**：`Transcribe the mathematical formula in this image into LaTeX. Output only the LaTeX formula, without explanation or markdown.`
- **解码**：贪心，max_new_tokens = 256，batch 1，bf16；Qwen3.5 关闭思考。
- **图像预算**：两个 Qwen 用 min = 65536、max = 262144；SmolVLM 用它自带的切图配置。
- **评分**：复用第 18 章的 `extract_formula` 先从原始输出里提取公式，再用 `score_postcorrect` 算 EM_raw、EM_norm、CER、parse_rate（和第 31 章同一套口径）。EM 给 Wilson 95% 区间，CER 给 1000 次 bootstrap 区间。parse_rate 只是 mathtext 可解析率，不是 LaTeX 编译率。
- **面板**：第 0 章 0.9.3 节 OCR 主任务面板的精简版，报告 EM_raw、EM_norm、CER、parse_rate 和截断条数。面板里的渲染一致率和 CER_token 本章没有报告，因为 `score_postcorrect` 只按字符算 CER、只调用 `can_parse`、不比较渲染图。
- **选择规则**（事先写定）：val 原图 EM_norm 最高者胜出，平局看 CER；胜出者冻结后评 test 一次。本章预先登记的 test 候选只有这一个，符合 0.10.1 节的测试集规则。

### 32.8.4 结果：val 原图 180 条

【G-val】：

| 模型 | EM_raw | EM_norm [95% 区间] | CER [95% 区间] | parse_rate | 截断 | 峰值显存 |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | 0/180 | 0.061（11/180）[0.034, 0.106] | 0.185 [0.145, 0.235] | 0.733 | 8 | 1.68 GiB |
| Qwen3-VL-2B | 1/180 | 0.072（13/180）[0.043, 0.120] | 0.135 [0.112, 0.161] | 0.806 | 1 | 4.02 GiB |
| SmolVLM-256M | 0/180 | 0.006（1/180）[0.001, 0.031] | 1.459 [1.253, 1.682] | 0.539 | 62 | 0.81 GiB |

（"截断"指生成到 256 个 token 还没有结束符的条数；峰值显存是本进程 PyTorch 的 `max_memory_allocated`，不是整卡。）

怎样读这张表：

1. **都很低。**最好的也只有 13/180 能规范化后一字不差。和第 31 章的小 OCR 在自己的分布上 0.472 比，这看起来很惨。但两者的数据、提示、评分条件都不同，不能直接比。32.9 节会看到，这个低分有一大块不是"没看清"。
2. **两个 Qwen 拉不开。**11 对 13，区间大幅重叠。按事先的规则 Qwen3-VL 胜出，但这只说明"按规则选了它"，不说明它在统计上显著更好。CER 0.135 对 0.185 的差距更明显一些。
3. **SmolVLM 基本不会。**CER 大于 1 意味着编辑距离比参考答案还长，主要来自复读：62 条写满 256 个 token。

速度只作参考：生成速度 Qwen3.5 每秒 28.1 个 token、Qwen3-VL 14.3、SmolVLM 32.8（本项目观察，`gpu/remote_runs/metrics.csv` 的 `tokens_per_second`）。这台租用卡上有其他任务并行，Qwen3.5 的线性注意力还在用 PyTorch 回退实现（日志里有 `falling back to its reference PyTorch implementation` 的提示），这些数字不能当作架构的速度排序。

### 32.8.5 结果：渲染图对照

原图来自各种 PDF，字体、粗细、噪声不一。如果把同一个公式用干净的渲染器重画一遍，模型会不会读得更好？

val 原图里有 103 条能被 mathtext 渲染（`data_audit_copy.json` 的 `renderable_val`），取前 80 条，同一个公式各有一张原图和一张渲染图（dpi 160，白底黑字，四周留白 12 像素，lab 第 0 节）。【G-pair】：

| 模型 | 原图 EM_norm | 渲染图 EM_norm | 原图 CER | 渲染图 CER |
|---|---|---|---|---|
| Qwen3.5-0.8B | 7/80 | 9/80 | 0.128 | 0.139 |
| Qwen3-VL-2B | 9/80 | 9/80 | 0.106 | 0.138 |
| SmolVLM-256M | 1/80 | 1/80 | 1.634 | 0.521 |

这里要用配对表。如果拿渲染图的 80 条（【G-ren】，Qwen3.5 和 Qwen3-VL 都是 0.113）去和原图的 180 条（0.061、0.072）比，会以为"渲染图普遍好很多"。可这 80 条是能被 mathtext 渲染的子集，本身就偏简单。同一批 80 条配对比，两个 Qwen 几乎没变，Qwen3-VL 的 CER 甚至变差了；只有 SmolVLM 的 CER 大幅下降，主要因为渲染图上它不再复读（截断 0 条，【G-ren】）。

结论：在这批数据上，换成干净的图没有给 Qwen 带来可见的提升。80 条的区间很宽，这个结论只是"没看到差别"，不是"证明没有差别"。

### 32.8.6 结果：test

按规则冻结 Qwen3-VL-2B（`gpu/remote_runs/val_selection.json`，冻结时间 2026-10-29T11:56:38Z，早于 test），然后评 test 100 条一次。【G-test】：EM_norm 0.110（11/100）[0.063, 0.186]，CER 0.150 [0.117, 0.184]，parse_rate 0.81，截断 1 条，峰值显存 4.01 GiB。

test 比 val 高（0.110 对 0.072），但两个区间重叠，样本也不同。按协议（0.10.1 节），test 只报告数字，不据此改任何设置，开发期不看错例，最终报告定稿后才可以解封做错例分析；也不能因为它"比预期好"或"比预期差"去换模型重测（lab 第 0 节）。

> **停下来想一想：** 如果当初的规则是"CER 最低者胜出"，选出来的还是 Qwen3-VL 吗？如果规则是"EM_norm 最高，平局看显存"呢？规则为什么必须在看结果之前写下来？

### 32.8.7 对照代码

`tour.py` 部分 4 的核心循环（第 178 行起）：

```python
image_id=getattr(processor,"image_token_id",model.config.image_token_id)
image_tokens=int((inputs["input_ids"]==image_id).sum())
with torch.inference_mode():
    output=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False,use_cache=True)
tokens=output[0,inputs["input_ids"].shape[1]:]
raw=processor.decode(tokens,skip_special_tokens=True)
pred=extract_formula(raw); preds.append(pred)
...
truncated=len(tokens)==args.max_new_tokens and int(tokens[-1]) not in eos
```

- 视觉 token 数是**从 `input_ids` 里数出来的**，不是用公式算的。这样算法写错时，记录不会跟着错。
- `generate` 的输出包含输入，要切掉前 `input_ids.shape[1]` 个才是新生成的部分。
- `raw` 和 `pred` 都存进 val 的 JSON。先保存后评分（lab 第 0 节），评分脚本出问题时可以只重算不重跑。
- 截断的判定：长度到顶**且**最后一个 token 不是结束符。

test 的保护在 `selected_rows`（第 167 行）：不带 `--final-test`、不是单一模型、不是原图，一律拒绝；`part4` 在加载模型前先独占创建 `.lock` 文件，第二次运行会失败。测试 `test_test_guard_precedes_data_access` 检查这道门在读数据之前就关上。

---

## 32.9 失败模式与输出格式诊断

### 32.9.1 动机

aggregate 指标只告诉你"错了多少"。要决定第 33 章该怎么做，得知道"错在哪里"：是模型没看清图，还是看清了但写法不合，还是写着写着停不下来？lab 第 4 节的要求是：**先读原始输出，再解释指标。**

### 32.9.2 直觉：阅卷老师的三种扣分

想象一个阅卷老师批 LaTeX 听写：

- **真错**：漏了一个上标，把 ν 写成 v。
- **格式扣分**：答案对，但写成了 `\frac { a } { b }` 而不是 `\frac{a}{b}`；或者多套了一层 `$$`。
- **评分规则误伤**：答案对，标点也对，但阅卷规则先把学生答案末尾的逗号擦掉，再和带逗号的标准答案比。

第三种听起来荒唐，但第 18、19 章已经见过，下面在图像任务上又碰到了。

### 32.9.3 推导：从 raw 到分数要过几道关

一条输出要变成分数，经过三步：

1. **提取**：`extract_formula(raw)` 去掉思考块、代码围栏，找出 `$$…$$` 等定界符里的公式；最后**去掉一个句末的 `.` `,` `;`**（第 18 章 18.6.1 节抽取器的规则）。
2. **规范化**：`normalize_latex` 统一空格、花括号等写法。所以 `I _ { \Sigma }` 和 `I_\Sigma` 在 EM_norm 下可以相等，在 EM_raw 下不等。EM_raw 几乎全是 0，原因就在这里：Qwen 习惯输出 `\frac { a } { b }` 这种每个符号之间都有空格的写法。
3. **比较**：规范化后的 token 序列逐个比。

问题出在第 1 步的最后一条。如果**目标**以标点结尾（arXiv 公式里很常见，比如 `z=\varepsilon x,`），而模型也老老实实抄了这个逗号，提取会把模型的逗号删掉，目标的逗号却还在，于是判错。

`val_format_diagnostic.json` 用 `em_ignoring_final_punct`（两边都去掉末尾的标点再比）做了一个只用于诊断的对照（本项目观察，`gpu/val_format_diagnostic.json`，n = 180，val 原图）：

| 模型 | 官方 EM_norm 命中 | 忽略句末标点后命中 |
|---|---|---|
| Qwen3.5-0.8B | 11 | 59 |
| SmolVLM-256M | 1 | 2 |

Qwen3.5 从 11 条跳到 59 条。也就是说，有 48 条**只差句末标点**。（这个 JSON 里没有 Qwen3-VL，你可以在练习 32.6 里自己算出来。）

### 32.9.4 数字例子：逐条看

下面的例子都来自【G-val】的 `main/qwen35_val_original.json`（val，可以看错例）。

**例 1：评分误伤。**

```text
目标  z=\varepsilon x,
raw   $$z = \varepsilon x ,$$
pred  z = \varepsilon x
```

模型连逗号都抄对了，`extract_formula` 把它删了。逐条检查那 48 条"只差句末标点"的样本：其中 46 条，raw 里 `$$…$$` 内部的内容和目标在规范化后完全相同（本项目观察，由 `main/qwen35_val_original.json` 的 `raw`、`target` 字段逐条比对，本章写作时离线计数，未另存文件；练习 32.6 请你自己重算）。这 46 条不是模型的错，是评分口径和这批数据不合。180 条 val 目标中，规范化后以 `.` `,` `;` 结尾的有 123 条（同样由 JSON 逐条统计），所以这个口径问题影响面很大。

**例 2：复读到上限。**

```text
目标  \rho (C)=\{(\rho (c_{1}), …, \rho (c_{n}))\\vert\ (c_{1}, …) \in C\}.
raw   $$\rho ( C ) = \{ ( \rho ( c _ { 1 } ) , … )$$ $$\vert \vert \vert \vert …
```

前半段读对了，遇到目标里一个奇怪的 `\\vert`（换行后紧跟 `vert`），模型进入 `\vert` 的无限循环，写满 256 个 token。Qwen3.5 有 8 条截断，SmolVLM 有 62 条（【G-val】）。第 31 章 31.8 节也见过这种复读，第 14 章讲过它是贪心解码的老毛病。

**例 3：结构写法不同。**

目标用 `&` 和 `\\` 的对齐写法，Qwen3.5 写成 `\begin{array} { r l } { … } \\ { … } \end{array}`，Qwen3-VL 写成 `\begin{aligned} … \end{aligned}`。渲染出来可能差不多，规范化后完全不同。这类差异 EM_norm 判错、CER 也会计较很多字符。

**例 4：漏掉内容。**

```text
目标  \beta_{U}(H) = \sum_{\substack{i < j \\ i,j \in U}} \beta_{ij}(H)
Qwen3.5  \beta _ { U } ( H ) = \sum _ { i < j } \beta _ { i j } ( H )
Qwen3-VL \beta _{U} ( H ) = \sum _ { \substack { i < j \\ i , j \in U } } \beta _ { i j } ( H )
```

求和号下面的第二行 `i,j \in U` 很小，Qwen3.5 漏了，Qwen3-VL 读到了。这才是"没看清"。

**例 5：胡编。**同一张图，SmolVLM 输出 `\Theta(A) = \{x\in\Sigma_{1}\backslash A|\{A\U\{x\}\}\}`，结构像，符号对不上（目标是 `I_\Sigma(A)=…`）。SmolVLM 的大部分输出属于这一类，加上复读。

**输出包装。**Qwen3.5 的 180 条原始输出全部用 `$` 定界符包着（由 raw 字段逐条统计；`format_issues` 的 `dollar_wrapped`），尽管提示词说了"without markdown"。这不影响分数，因为 `extract_formula` 会剥掉；但它说明零样本模型有自己的习惯格式，提示词管不住。

### 32.9.5 对照代码

`miniocr/eval/postcorrect_eval.py` 里 `extract_formula` 的最后几行：

```python
tail = _LATEX_TOKEN.findall(formula)
if tail and tail[-1] in (".", ",", ";", "。"):
    invisible_delimiter = (tail[-1] == "." and len(tail) > 1
                           and tail[-2] in (r"\left", r"\right", r"\middle"))
    if not invisible_delimiter:
        formula = formula[:-1].rstrip()
return formula
```

- 只删**模型输出**的末尾标点，不动目标。第 18 章 18.6.1、18.6.2 节已经发现它的副作用：模型输出本来规范化后和目标相同，抽取器删掉句末标点，判对变成判错；第 19 章 19.8.3 节 SFT 的干净输出上，抽取后口径同样更差。这批图像数据的目标也常带句末标点，同样的误伤又出现了。
- `\right.` 这种不可见定界符被特意保留，说明规则考虑过标点的数学含义，但没有处理目标本身带标点的情况。

诊断函数 `em_ignoring_final_punct`（同一文件）对**两边**都去掉末尾所有 `.` `,` `;` token 再比。诊断脚本 `gpu/diagnose_val.py` 只读已有的 val 输出，不读 test，结果文件里写明了"exploratory val-only diagnostic; not used to select models"。

**我们该怎么办？**

- **不改本章已冻结的官方结果。**选择规则和指标是事先定的，看过结果再改这份结果的口径，就是在用结果调评分。本章的 EM_norm 照原样报告，诊断数字单独列出。
- **把它记成已知的指标盲区。**写进 R3 报告：在这批数据上，官方 EM_norm 对"目标带句末标点"的样本系统性偏低。
- **改口径要走新版协议。**根据 val 错例开发新版协议是可以的（0.10.5 节）：规则升级版本号，所有候选用新版重算，最终测试另行安排，不和旧协议下的数字混比。决定要在看新结果之前做，比如第 33 章开跑之前，就写明是否换一个只对模型输出剥壳、不删标点的提取规则。
- **对第 33 章的启示。**零样本模型的失分里，有一大块是格式（标点、空格风格、array 还是 aligned）。LoRA 微调最容易学会的也是格式。所以微调后 EM_norm 大涨时，要分清是"读得更准"还是"写得更像标注"。CER 和逐条错例能帮你分开这两件事。

### 32.9.6 常见错误

| 错误 | 现象 | 怎样发现 |
|---|---|---|
| 只看 aggregate 就下结论 | "Qwen 零样本只有 6%，不能用" | 读 raw，按失败类型数一数 |
| 看完诊断就改官方指标 | 分数突然翻几倍，报告里说不清为什么 | 协议要求口径事先定；要改就升版本、重算全部候选；诊断单列 |
| 拿不同队列直接相减 | "渲染图提高了 5 个点" | 用 `paired.json` 这种同 ID 配对表 |
| 把复读导致的 CER > 1 当成"模型完全不懂" | SmolVLM 在原图上 CER 1.46，渲染图上 0.52 | 看截断计数，和 CER 一起读 |
| 开发期看 test 的错例 | 不知不觉按 test 调了提示词 | test 不落盘生成文本（tour 只给 val 存 `raw`） |

---

## 32.10 读 OCR 专用 VLM 的模块说明（概括）

最后读一读 GLM-OCR、PaddleOCR-VL 的模块说明，和 Qwen3.5 比较。本章只读官方模型卡，不下载、不推理（lab 第 6 节，tour 部分 6 只保存阅读索引 `reading.json`）。以下全部是文档说明，**未经本项目验证**：

| 模型 | 视觉与连接 | 语言侧/任务 | 和 Qwen3.5 比较时看什么 |
|---|---|---|---|
| Qwen3.5-0.8B | 动态网格 ViT，2×2 空间合并 merger | 混合注意力解码器，通用多模态 | 本章已经实测了 grid 和 merger 形状 |
| GLM-OCR | CogViT 视觉编码器，轻量连接器并做 token 下采样 | GLM 解码器，专做文档 OCR | 图像预处理、下采样发生在哪里、OCR 指令格式 |
| PaddleOCR-VL | NaViT 风格的动态分辨率视觉编码器 | ERNIE-4.5 语言模型，文档元素识别 | "动态分辨率"不等于 patch 和合并方式相同 |

读这类说明时，用本章的四个问题去问它：processor 怎样缩放和切块？视觉 token 数怎么算？连接器压缩了什么？视觉 token 怎样进入语言模型？说明里回答不了的，就是需要读源码的地方。另外注意，GLM-OCR 的完整应用流水线里还有版面分析，那是模型外面的独立模块，不是语言模型里的层。

---

## 32.11 和项目的关系

本章在 TexPixel 路线上做了三件事：

1. **选定第 33 章的模型。**按事先写定的规则，Qwen3-VL-2B 以 val EM_norm 13/180 对 11/180 胜出（`val_selection.json`）。两者区间重叠，差异不显著；第 33 章 lab 对 Qwen3.5 和 Qwen3-VL 都做了实验，再在 val 的前 100 条上选择（与本章那 180 条的前 100 条相同，第 33 章 33.11 节）；那是一份新的报告协议，不能拿本章的 test 调参。
2. **零样本基线和协议。**同一个 val、同一个 test、同一个评测脚本；val 存原始输出，test 只存数字。R3 报告里的"零样本基线"一栏就来自本章。
3. **图像预算的出发点。**本章的 min = 65536 让小公式统一放大到约 75 个视觉 token。第 33 章 lab 改用 min = 16384、max = 65536（见其 lab 第 0 节），视觉 token 更少、训练更省显存。用 32.2 节的公式，你能在跑之前就算出它的 token 数。

和第 31 章对照：小 OCR 的 0.472 是在自己的训练分布上、用字节词表、从零训练出来的；本章的数字是预训练 VLM 的零样本表现，数据不同，评分口径里还有 32.9 节的盲区。两者不能直接比较，但同一套评测脚本和协议把它们串在一起。

---

## 32.12 常见错误

各节末尾的错误表按现象索引：

- "Image features and image tokens do not match"：32.3.6、32.6.6
- 形状全对，模型输出明显变差：32.3.6、32.7.6
- 视觉 token 数和估算不符：32.2.6
- 逐位对拍差 1e-7：32.3.5、32.3.6
- 参数总数多出 2.5 亿或 2000 万：32.4.6
- Qwen3.5 输出一大段推理、频繁截断：32.6.6
- EM_norm 很低，CER 却不高：32.9
- 渲染图"明显更好"：32.8.5
- test 分数和 val 差很多就想换模型：32.8.6

## 32.13 和真实框架的对应

| 本章 | transformers |
|---|---|
| `load_image_only_processor` | `AutoProcessor.from_pretrained(path)`：按模型目录里的配置组装 image processor、tokenizer 和 chat template。transformers 5.19 的 Qwen processor 还会构造视频组件，本章为了只用 PIL 后端，写了一个只含图像和 tokenizer 的子类（`vlm_anatomy.py` 第 111 到 129 行），模板和占位符展开仍是官方逻辑 |
| `smart_resize`、`replica_processor` | `Qwen2VLImageProcessor`（PIL 版 `Qwen2VLImageProcessorPil`）；`size={"shortest_edge", "longest_edge"}` 就是 `min_pixels`、`max_pixels`。SmolVLM 对应 `Idefics3ImageProcessor` |
| `expand_image_placeholder` | processor 的 `__call__(text=..., images=...)`：先算 `image_grid_thw`，再把每个 `<\|image_pad\|>` 换成 prod(grid)/merge² 个，最后分词 |
| 模板渲染 | `processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)`；消息里用 `{"type": "image"}` 标记图片位置。也可以 `tokenize=True, return_dict=True` 一步得到张量 |
| 整个模型 | `AutoModelForImageTextToText.from_pretrained(path, dtype=..., attn_implementation="sdpa")`：Qwen3.5 解析为 `Qwen3_5ForConditionalGeneration`，SmolVLM 为 `Idefics3ForConditionalGeneration` |
| 视觉向量填入 | `get_placeholder_mask` + `inputs_embeds.masked_scatter(...)`；M-RoPE 位置由 `get_rope_index` 计算 |
| 生成 | `model.generate(**inputs, max_new_tokens=256, do_sample=False)`；输出包含输入，要切掉前缀再 `processor.decode(..., skip_special_tokens=True)` |
| `MLPConnector` | LLaVA 的 `multi_modal_projector`（transformers 里 `llava` 模型的两层 MLP）；Qwen 的 `Qwen3_5VisionPatchMerger`；SmolVLM 的 `Idefics3Connector`（像素重排 + 线性） |
| `QueryConnector` | BLIP-2 的 `Blip2QFormerModel` 加 `query_tokens`（完整版有多层、自注意力和预训练目标） |

transformers 的类名和函数名以 5.19.0 为准，版本升级后可能变化。读源码时用"找 `masked_scatter`""找 `smart_resize`"这样的关键词定位，比记行号可靠。

## 32.14 小结

真实 VLM 和第 31 章的小 OCR 是同一种结构：processor 把图按面积缩放成 32 的倍数，切成 16×16、复制成两帧的 patch，按 2×2 邻域优先排成 `[N, 1536]`；12 层 ViT 处理 N 个 patch，merger 把每 4 个拼成一个、投到语言模型的宽度；视觉 token 数等于缩放后面积除以 1024。chat template 留出一个占位符，processor 把它展开成 prod(grid)/4 个，语言模型查完嵌入表后用 `masked_scatter` 把视觉向量填进这些位置，再配上三维的 M-RoPE 位置。Qwen3.5-0.8B 的 852,985,920 个参数里，视觉塔约 1 亿，词嵌入约 2.5 亿，merger 只有 1259 万。零样本实测里按规则选中的 Qwen3-VL-2B 在 val 原图上 EM_norm 也只有 13/180（本项目观察，RTX 4090（租用），bf16），和 Qwen3.5 的区间重叠。逐条看原始输出，相当一部分失分来自格式和删掉模型句末标点的提取规则，不是没看清图；第 33 章做 LoRA 之前要先分清这两类失分。

复习时抓住下面几条：

- **视觉 token 数**：`smart_resize` 先把高宽就近取整到 32 的倍数；取整后的面积小于 m 时，原图两边各乘 s = √(m/hw) 再向上取整，大于 M 时各除以 s = √(hw/M) 再向下取整（每条边至少 32）。然后 N_patch = H'W'/256，N_vis = H'W'/1024。两个取整方向让面积可能越出 [m, M]，越不越界看预算和图的形状。例子：32×298 → 96×800 → 6×50 = 300 个 patch → 75 个视觉 token。
- **`pixel_values`**：每行 3 × 2 × 16 × 16 = 1536 个数（静态图复制成两帧），行按 2×2 邻域优先排列。patch 嵌入和视觉 block 把每行变成 768 维、不改行的顺序，所以 merger 能用一次 `view` 把连续 4 行拼成 3072 维。顺序错了形状照样对，只能靠逐位对拍或独立锚点发现。
- **模块对应**：Qwen3.5 的 `model.visual.patch_embed`、`pos_embed`、`blocks`、`merger` 和 `model.language_model`，对应第 31 章的 `patch_embed`、`self.pos`、`self.vision`、`self.connector` 和 `self.gpt`。视觉塔（含 merger）加语言模型正好是 852,985,920；输出层和词嵌入共享，文件里的 `mtp.*` 加载时不用，都不能再算一遍。
- **连接器**：合并改长度，MLP 改宽度；Q-Former 的输出个数等于 query 数，和图多大无关。
- **占位符与位置**：processor 把唯一的 `<|image_pad|>` 展开成 prod(grid)/4 个，输入长度等于 N_vis 加模板里固定的文字 token（本章提示词下 Qwen3.5 加 36，Qwen3-VL 加 32）。模型查表后用 `masked_scatter` 填入视觉向量，先检查元素数相等。M-RoPE 给每个图片 token 时间、行、列三个坐标，图片后的文字从图片起点 + max(合并后行数, 列数) 接着编号。
- **读零样本结果**：两个 Qwen 的 val 区间重叠，按规则选中不等于显著更好。先读原始输出再解释指标；诊断口径单独列出，不改已经冻结的官方结果。
- **最常见的错误**：把 patch 数（300）当视觉 token 数（75）；把 `min_pixels`、`max_pixels` 当边长；Qwen3.5 忘了 `enable_thinking=False`；拿渲染图子集和全量 val 直接比。

## 32.15 精读

1. **Liu, Li, Wu, Lee. *Visual Instruction Tuning.* NeurIPS 2023.**（LLaVA）读模型结构和训练两节。
   - LLaVA 的连接器是一个线性投影，后来的 LLaVA-1.5 换成两层 MLP。对照 32.5.3 节的表：它们的输出长度是多少？为什么 LLaVA 不需要 token 合并（提示：它用的图像分辨率和视觉 token 数是多少）？
   - LLaVA 分两阶段训练，每个阶段分别冻结哪些模块、训练哪些？这和第 33 章的冻结策略（只训语言模型 LoRA、加训连接器、再加视觉塔 LoRA）有什么对应？
   - 它的指令数据是怎样生成的？这种数据能教会模型"只输出 LaTeX、不加 `$$`"吗？

2. **Wang et al. *Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution.* 2024.**（Qwen2-VL 技术报告）读"动态分辨率"和"M-RoPE"两部分。
   - 报告怎样描述"任意分辨率"和相邻 2×2 token 的合并？和本章 32.2、32.5 节复刻的 `smart_resize`、merger 对照，有哪些细节报告里没写、只能从源码里看到（例如取整方向）？
   - M-RoPE 把位置拆成时间、高、宽三个分量。用 32.7.4 节的例子说明：为什么图片之后的文字位置从 29 开始，而不是从 79 开始？这样做有什么好处？
   - 报告说静态图片怎样处理时间维？和 32.3.3 节"复制成两帧"一致吗？

3. **Marafioti et al. *SmolVLM: Redefining Small and Efficient Multimodal Models.* 2025.**
   - SmolVLM 为什么选择激进的像素重排（本章看到的 4×4，一次压 16 倍）？报告里怎样权衡视觉 token 数和小语言模型的上下文长度？
   - 它用文字 token（如 `<row_1_col_2>`）标记图块位置，而不是像 Qwen 那样用三维位置编码。报告对这类设计做了什么消融？
   - 本章实测里 SmolVLM 在公式 OCR 上几乎全错且大量复读。从报告的训练数据构成看，你能找到可能的原因吗？哪些只是猜测、需要实验验证？
