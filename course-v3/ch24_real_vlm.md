# 第 24 章　解剖真实的 VLM

## 本章在全景图中的位置

第 19–23 章，我们从零搭了一个 VLM：ViT 视觉塔（第 19 章），用 CLIP 预训练（第 20 章），通过连接器接到 GPT 上（第 22 章），再用指令数据教它做各种任务（第 23 章）。它只有几百万参数，只认识 ShapesWorld 里的几何图形和渲染出来的文字行。

本章打开四个真实的开源 VLM：

| 模型 | 发布 | 为什么选它 |
|---|---|---|
| **Qwen3.5-0.8B** | 2026 | 本课程后半部分的主角：第 28、35 章在它上面做微调 |
| Qwen2.5-VL-3B | 2025 | Qwen 系列上一代的代表，很多 OCR 模型以它为底座 |
| **SmolVLM-256M** | 2025 | 最小的实用 VLM 之一，结构和 Qwen 不同（切子图 + pixel shuffle） |
| LLaVA-1.5-7B | 2023 | 最经典的"投影式"结构，几乎所有 VLM 教程的起点 |

加粗的两个模型下载了权重（共约 2.2 GB），在 CPU 上运行；另外两个只读配置文件。

本章要回答一个问题：**一张图片从文件到语言模型的输入，中间到底发生了什么？** 你会自己实现 Qwen 的 processor 的每一步（缩放、归一化、切块、展开占位符、三维位置编号），并逐项与 transformers 库的实现对照，结果完全一致。然后用这两个真实模型做我们的玩具任务，看看几亿参数、在海量数据上训练过的模型，在 ShapesWorld 和文字行上表现如何。

读完本章，你会发现真实模型和第 22 章的 VLM 是**同一个结构**：视觉塔、连接器、语言模型、占位符、只在答案上计算损失。区别在于每个部件的细节，以及这些细节背后的工程考虑。

## 学习目标

1. 读懂一个 VLM 的 `config.json`，说出它的视觉塔、连接器、语言模型的规模，以及参数在三者之间的分布；
2. 实现 Qwen 的 `smart_resize`，解释 `min_pixels`、`max_pixels` 怎样控制视觉 token 数；
3. 实现 Qwen 的切块：理解 `temporal_patch_size`（把图片当成两帧的视频）和"按 2×2 窗口分组"的 patch 顺序；
4. 实现占位符的展开，说清 `input_ids`、`pixel_values`、`image_grid_thw`、`mm_token_type_ids` 各是什么；
5. 实现 M-RoPE 的三维位置编号，理解分段式和交错式两种频率分配；
6. 说出 Qwen3.5 语言模型的混合注意力结构（线性注意力 + 全注意力）和它对 KV cache 的影响；
7. 说出 SmolVLM 的切子图方案，比较不同模型同一张图的视觉 token 数；
8. 用真实模型做零样本测试，分析它们在玩具任务和 OCR 上的表现。

## 前置检查

1. 第 19 章：patchify 的输出形状是什么？patch 是按什么顺序排的？一个 `Conv2d(kernel=p, stride=p)` 和"切块 + 线性层"是什么关系？
2. 第 19 章：二维 RoPE 怎样把行号和列号编码进旋转角？
3. 第 22 章：`VLM.embed` 做了什么？占位符 `<img>` 的个数必须等于什么？
4. 第 22 章：2×2 合并把 (B, h·w, d) 变成什么形状？
5. 第 16 章：RoPE 为什么能让注意力分数只依赖相对位置？
6. 第 17 章：KV cache 存的是什么？它的大小与什么成正比？

### 准备环境

本章需要 `requirements-hf.txt` 里的库（主要是 `transformers==5.19.0`），以及两个模型的权重：

```bash
pip install -r requirements-hf.txt
huggingface-cli download Qwen/Qwen3.5-0.8B
huggingface-cli download HuggingFaceTB/SmolVLM-256M-Instruct
# 第 1 部分还要读另外两个模型的配置（不下载权重）
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --include "*.json"
huggingface-cli download llava-hf/llava-1.5-7b-hf --include "*.json"
```

下载的文件存在 `~/.cache/huggingface/hub/` 下。之后运行时设置 `HF_HUB_OFFLINE=1`，transformers 就只读本地缓存，不再联网（本章的脚本默认这样设置）。

两个模型在 CPU 上用 float32 运行：Qwen3.5-0.8B 有 8.5 亿参数，占 3.4 GB 内存；SmolVLM-256M 占 1 GB。生成一句话要几秒，做本章的实验足够了。

**一个小坑**：transformers 的 `AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B")` 会同时创建图像和视频的处理器，视频处理器依赖 `torchvision`。没装 torchvision 就会报错。本章不用它：分词器用 `AutoTokenizer`，图像预处理用我们自己写的函数（24.3–24.5 节），这正好也是本章的练习。SmolVLM 的 processor 不依赖 torchvision，直接用 `AutoProcessor`。

---

## 24.1 四个模型，同一个骨架

### 读配置文件

每个 Hugging Face 上的模型都有一个 `config.json`，描述模型的结构。Qwen3.5-0.8B 的配置（节选）：

```json
{
  "architectures": ["Qwen3_5ForConditionalGeneration"],
  "image_token_id": 248056,
  "vision_start_token_id": 248053,
  "vision_end_token_id": 248054,
  "vision_config": {
    "depth": 12, "hidden_size": 768, "num_heads": 12, "intermediate_size": 3072,
    "patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2,
    "num_position_embeddings": 2304, "out_hidden_size": 1024
  },
  "text_config": {
    "num_hidden_layers": 24, "hidden_size": 1024, "intermediate_size": 3584,
    "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 256,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention", ...],
    "vocab_size": 248320,
    "rope_parameters": {"mrope_section": [11, 11, 10], "mrope_interleaved": true,
                        "rope_theta": 10000000, "partial_rotary_factor": 0.25}
  }
}
```

逐项对应第 19–22 章的概念：

- `vision_config`：视觉塔是一个 12 层、d = 768 的 ViT（和 ViT-B 一样大），patch 16；
- `spatial_merge_size: 2`：连接器先做 2×2 合并（第 22 章的 `merge_2x2`），再用 MLP 映射到 `out_hidden_size: 1024`，也就是语言模型的维度；
- `text_config`：语言模型 24 层、d = 1024。`num_key_value_heads: 2` 是第 16 章的 GQA（8 个查询头共享 2 组键值）；
- `image_token_id: 248056`：占位符 `<|image_pad|>` 的编号，相当于我们的 `<img>`；
- `layer_types` 和 `rope_parameters` 是两个新东西：24.7 节讲 M-RoPE，24.8 节讲线性注意力。

### 四个模型的对照

本章第 1 部分读取四个模型的配置，在 **meta 设备**上建模型、数参数：

```python
with torch.device("meta"):
    m = AutoModelForImageTextToText.from_config(cfg)
```

meta 设备上的张量只有形状、没有数据，不分配内存，所以 70 亿参数的 LLaVA 也能瞬间"建"出来数参数。结果（24.10 节有完整输出）：

| 模型 | 视觉塔 | 连接器 | 语言模型 | 总参数 |
|---|---|---|---|---|
| Qwen3.5-0.8B | 12 层 d=768 p=16，88M（10%） | 2×2 合并 + MLP，12.6M（1.5%） | 24 层 d=1024，752M | 853M |
| Qwen2.5-VL-3B | 32 层 d=1280 p=14，632M（17%） | 2×2 合并 + MLP，36.7M（1.0%） | 36 层 d=2048，3086M | 3755M |
| SmolVLM-256M | 12 层 d=768 p=16，86M（34%） | pixel shuffle 4×4 + 线性，7.1M（2.8%） | 30 层 d=576，163M | 256M |
| LLaVA-1.5-7B | 24 层 d=1024 p=14，304M（4%） | 两层 MLP，21.0M（0.3%） | 32 层 d=4096，6739M | 7063M |

几个规律：

1. **连接器都很小**：0.3%–2.8%。和第 22 章一样（我们的 MLP 连接器约占 2%），连接器只负责"翻译"，真正的能力在两头；
2. **语言模型占大头**：除了 SmolVLM，语言模型都占 80% 以上。VLM 的"智力"主要来自语言模型；
3. **小模型的视觉塔占比高**：SmolVLM 的视觉塔（8600 万）占了 34%。视觉塔不能太小，否则看不清；语言模型可以小，代价是推理能力弱；
4. **词嵌入很大**：Qwen3.5 的词表有 248,320 个 token（支持上百种语言），嵌入矩阵 248,320 × 1024 = 2.54 亿参数，占语言模型的 1/3。它和输出层共享权重（`tie_word_embeddings: true`），否则还要再多 2.54 亿。

### 骨架都一样

四个模型的推理流程，和第 22 章的 `VLM.forward` 完全一致：

```
图片 ──processor──▶ pixel_values ──视觉塔──▶ patch 特征 ──连接器──▶ 视觉 token
                                                                   │
文字 ──聊天模板──▶ 带占位符的 token 序列 ──词嵌入──▶ 文字向量 ──────┤ 把视觉 token 填进占位符的位置
                                                                   ▼
                                                     语言模型 ──▶ 逐个生成答案
```

区别都在细节：图片怎样缩放、怎样切块、占位符有几个、位置怎样编号、语言模型用什么注意力。接下来逐一打开。

---

## 24.2 processor 是什么

在 Hugging Face 的约定里，**processor** 负责把原始输入（图片、文字）变成模型能吃的张量，包括两部分：

- **图像处理器**（image processor）：缩放、归一化、切块，输出 `pixel_values`；
- **分词器**（tokenizer）：套聊天模板、分词，输出 `input_ids`。

两部分要配合：分词器放进去的占位符个数，必须等于图像处理器切出来的视觉 token 数。

Qwen 的 processor 一共四步，对应本章的四个练习：

| 步骤 | 输入 → 输出 | 本章的函数 | 练习 |
|---|---|---|---|
| 1. 缩放 | 任意 h×w → 32 的倍数 | `smart_resize` | 24.1 |
| 2. 归一化 + 切块 | (3, H, W) → `pixel_values` (N, 1536) 和 `image_grid_thw` | `qwen_patchify` | 24.2 |
| 3. 展开占位符 | 一个 `<\|image_pad\|>` → N/4 个 | `expand_image_tokens` | 24.3 |
| 4. 位置编号（在模型里） | `input_ids` → (3, L) 的位置 | `mrope_positions` | 24.4 |

所有练习在 `miniocr/hf_vlm.py` 里，测试在 `labs/ch24/test_hf_vlm.py`。测试不需要下载模型；装了 transformers 时，最后一个测试还会把你的实现和 transformers 的结果逐个元素比较。

---

## 24.3 第一步：缩放（smart_resize）

### 为什么不缩放到固定大小

第 19 章的 ViT 只处理 64×64 的图。CLIP、SigLIP 也是固定分辨率（224、336、384）。LLaVA-1.5 的做法是：把短边缩放到 336，再从中间裁出 336×336。

这对 OCR 是灾难性的：

- 一张 A4 文档（宽高比 1:1.41）被裁成正方形，上下两边的文字直接没了；
- 一行文字（比如 32×256，宽高比 1:8）先放大到 336×2688，再裁出中间的 336×336，**只剩中间 1/8**；
- 一张 3000×4000 的照片缩小到 336×336，每个字只剩一两个像素。

Qwen2-VL 起引入了**原生分辨率**（native resolution）：图片保持原来的宽高比，只把长宽各自调整到 patch 的整数倍，大图多出 token、小图少出 token。这要求位置编码能处理任意大小的网格，所以视觉塔用二维 RoPE（第 19 章第 6 部分的实验证明它换分辨率几乎不掉点）。

### 规则

`smart_resize(h, w, factor, min_pixels, max_pixels)` 要满足三个条件：

1. 新的高和宽都是 `factor` 的整数倍。Qwen3.5 的 factor = patch 16 × merge 2 = 32，这样切块后网格的长宽都是偶数，2×2 合并正好不剩；
2. 总像素数在 `[min_pixels, max_pixels]` 之内；
3. 宽高比尽量不变。

算法：

```python
h_bar = round(h / factor) * factor          # 先各自四舍五入到 factor 的倍数
w_bar = round(w / factor) * factor
if h_bar * w_bar > max_pixels:              # 太大：按面积比例缩小
    beta = sqrt(h * w / max_pixels)
    h_bar = max(factor, floor(h / beta / factor) * factor)    # 向下取整，保证不超
    w_bar = max(factor, floor(w / beta / factor) * factor)
elif h_bar * w_bar < min_pixels:            # 太小：按面积比例放大
    beta = sqrt(min_pixels / (h * w))
    h_bar = ceil(h * beta / factor) * factor                  # 向上取整，保证不少
    w_bar = ceil(w * beta / factor) * factor
```

注意 beta 的定义：缩小时 beta = √(原面积 / 上限) > 1，长宽**除以** beta；放大时 beta = √(下限 / 原面积) > 1，长宽**乘以** beta。两种情况都按面积缩放，所以长宽按同样的比例变化，宽高比基本保持。

### 手算一个例子

一张 192×256 的图（高 192，宽 256），Qwen3.5 的默认参数 min_pixels = 65,536（256²），max_pixels = 16,777,216（4096²）：

1. 四舍五入：192 / 32 = 6 → 192；256 / 32 = 8 → 256。面积 192 × 256 = 49,152；
2. 49,152 < 65,536，太小，要放大：beta = √(65,536 / 49,152) = √1.333 = 1.1547；
3. 高：192 × 1.1547 = 221.7，除以 32 是 6.93，向上取整是 7，乘回 32 得 224；
4. 宽：256 × 1.1547 = 295.6，除以 32 是 9.24，向上取整是 10，乘回 32 得 320；
5. 结果 224×320，面积 71,680 ≥ 65,536。宽高比从 1.33 变成了 1.43（向上取整带来的误差）。

切块后网格是 224/16 × 320/16 = 14 × 20 = 280 个 patch，2×2 合并后 **70 个视觉 token**。

### 练习 24.1：`smart_resize`

在 `miniocr/hf_vlm.py` 里实现 `smart_resize`。测试 `test_smart_resize` 检查上面的例子，以及大图、文字行的情况。

### 视觉 token 数的控制

视觉 token 数 ≈ 面积 / (32 × 32)，每个视觉 token 对应 32×32 = 1024 个像素。`min_pixels` 和 `max_pixels` 就是 token 数的下限和上限：

- Qwen3.5 默认：下限 256² → 64 个 token，上限 4096² → 16,384 个 token；
- Qwen2.5-VL 默认：patch 14，每个 token 28×28 像素；下限 3136（4 个 token），上限 12,845,056（16,384 个 token）。

实际使用时常常手动调这两个值。显存不够就降低 `max_pixels`；文字很小就提高 `min_pixels`，让小图放大一些。比如 Qwen 官方给的 OCR 示例里常把 `max_pixels` 设成 1280 × 28 × 28（最多 1280 个 token），在精度和速度之间折中。第 6 部分的实验会看到：**同一行字，视觉 token 越多读得越准，但推理越慢**。

---

## 24.4 第二步：归一化与切块

### 归一化

把 0–255 的像素值变成模型训练时见过的范围：

```
x = (像素 / 255 − mean) / std
```

Qwen3.5 和 SmolVLM 用 mean = std = 0.5，结果在 [−1, 1] 之间；Qwen2.5-VL 和 LLaVA 用 CLIP 的统计量 mean = (0.481, 0.458, 0.408)、std = (0.269, 0.261, 0.276)。**用错了 mean/std，模型不会报错，只会变差**，所以一定要用模型自己的 `preprocessor_config.json` 里的值。

### temporal_patch_size：把图片当成两帧的视频

Qwen2-VL 系列要同时处理图片和视频。视频的 patch 是一个"时空小块"：2 帧 × 16 × 16 像素。patch 嵌入因此用的是**三维卷积** `Conv3d(3, 768, kernel=(2, 16, 16), stride=(2, 16, 16))`，而不是第 19 章的 `Conv2d`。

为了让图片也能走同一个 Conv3d，图片被**复制成两帧**（`temporal_patch_size: 2`），相当于一段静止了两帧的视频。每个 patch 拉平后是 3 通道 × 2 帧 × 16 × 16 = **1536** 维。这有一点浪费：图片的 patch 向量里一半的数是重复的，但换来了图片和视频的统一处理。

### 按 2×2 窗口分组的顺序

第 19 章的 patchify 按"逐行扫描"排列 patch：第 0 行从左到右，然后第 1 行……Qwen 的顺序不同：先按 2×2 的**窗口**分组，窗口逐行扫描，窗口内部再逐行。以 4×4 的网格为例，patch 编号 = 行 × 4 + 列：

```
网格中的位置           逐行扫描的顺序（第 19 章）        Qwen 的顺序
 0  1 |  2  3
 4  5 |  6  7         0 1 2 3 4 5 6 7 8 ...            0 1 4 5 | 2 3 6 7 | 8 9 12 13 | 10 11 14 15
──────┼──────
 8  9 | 10 11
12 13 | 14 15
```

为什么这样排？因为后面的连接器要做 2×2 合并。按窗口分组之后，**每 4 行相邻的 patch 正好是一个 2×2 窗口**，合并只需要 `x.view(-1, 4 * d)`：把相邻的 4 行拼成一行，不需要再做第 22 章 `merge_2x2` 里那些 reshape 和 permute。顺序在预处理时就排好了，模型里的操作最简单。

用 reshape 和 permute 实现（C 是通道，gh、gw 是网格的行数和列数，p = 16，m = 2）：

```python
t = x.reshape(C, gh // m, m, p, gw // m, m, p)   # 把 H 拆成 (窗口行, 窗口内的行, 像素行)，W 同理
t = t.permute(1, 4, 2, 5, 0, 3, 6)               # (窗口行, 窗口列, 窗口内的行, 窗口内的列, C, p, p)
t = t.unsqueeze(5).expand(..., temporal, ...)    # 在 C 后面插入时间维，复制 2 份
pv = t.reshape(gh * gw, C * temporal * p * p)    # 每个 patch 一行
```

`expand` 不复制数据，只是让同一块内存"看起来"有两份，最后的 `reshape` 才真正复制。每一行元素的顺序是 (C, T, p, p)，和 Conv3d 的权重 (768, C, T, p, p) 拉平的顺序一致，所以"每行乘以 Conv3d 的权重"等于"对这个 patch 做 Conv3d"（测试 `test_qwen_patchify_conv3d` 验证了这一点，和第 19 章"切块 + 线性层 = 卷积"是同一个道理）。

### 输出

处理器的输出是两个张量：

- `pixel_values`：形状 (N, 1536)，N 是**这一批所有图片的 patch 数之和**。注意没有批维度：不同大小的图片切出来的 patch 数不同，没法堆成 (B, N, 1536) 的规则张量，于是全部拼成一长串；
- `image_grid_thw`：形状 (图片数, 3)，每张图的 (时间, 行数, 列数) 网格大小，图片的时间是 1。模型靠它把一长串 patch 切回每张图（视觉塔内部用 `cu_seqlens` 让每张图只在自己的 patch 之间做注意力）。

一张 64×64 的玩具场景图：smart_resize 放大到 256×256，网格 16×16，`pixel_values` 是 (256, 1536)，`image_grid_thw` 是 `[[1, 16, 16]]`，进入语言模型的视觉 token 是 16 × 16 / 4 = 64 个。

### 练习 24.2：`qwen_patchify`

实现 `qwen_patchify(x, patch, merge, temporal)`。测试 `test_qwen_patchify_order` 用"值 = 位置编号"的小图检查顺序；`test_against_transformers` 把整条预处理（`qwen_preprocess` = 缩放 + 归一化 + 切块）与 transformers 的 `Qwen2VLImageProcessorPil` 比较，结果应该**完全相同**（最大差 0）。

---

## 24.5 第三步：聊天模板与占位符

### 聊天模板

Qwen3.5 的聊天模板（`chat_template.jinja`）把一条带图片的用户消息变成这样的字符串：

```
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|>
<|im_start|>assistant
<think>

</think>

```

逐个解释：

- `<|im_start|>`、`<|im_end|>`：一条消息的开始和结束（"im" 是 instant message）。第 23 章说过，真实模型用特殊 token 当角色标记，不会和普通文字混淆；
- `<|vision_start|>`、`<|vision_end|>`：图片的边界。它们是普通的（可学习的）token，告诉模型"图片从这里开始、到这里结束"；
- `<|image_pad|>`：占位符，目前只有**一个**；
- `<think>\n\n</think>\n\n`：Qwen3.5 是一个"会思考"的模型，可以先在 `<think>` 里写推理过程再回答。模板默认放一个**空的**思考块，意思是"不思考，直接回答"（`enable_thinking=False`）。第 31 章讲推理模型时再回来。

用分词器的 `apply_chat_template` 生成这个字符串，分词后是 19 个 token。

### 展开占位符

模板里每张图只有一个 `<|image_pad|>`，但这张图有 64 个视觉 token。processor 要把它**复制成 64 个**，序列从 19 个 token 变成 82 个。

transformers 的做法是在**字符串**上替换（把 `<|image_pad|>` 换成 64 个 `<|image_pad|>` 再分词），我们的练习在 **token 编号**上做，效果相同（第 3 部分验证了两者逐个相等）：

```python
def expand_image_tokens(ids, grids, image_id, merge=2):
    out, k = [], 0
    for t in ids:
        if t == image_id:
            n = t·h·w / merge²           # 第 k 张图的视觉 token 数
            out += [image_id] * n
            k += 1
        else:
            out.append(t)
    return out
```

多张图时，第 k 个占位符对应第 k 张图（`image_grid_thw` 的第 k 行）。占位符和图片的个数不一致，要报错。第 22 章的 `VLM.embed` 里也有同样的断言。

### 练习 24.3：`expand_image_tokens`

实现它，并且在占位符和图片数量不一致时抛出 `ValueError`。

### 模型的四个输入

最后送进 Qwen3.5 的是：

| 输入 | 形状 | 含义 |
|---|---|---|
| `input_ids` | (1, 82) | 展开后的 token 编号 |
| `attention_mask` | (1, 82) | 全是 1（批里只有一条，没有补齐） |
| `pixel_values` | (256, 1536) | 所有图片的 patch，拼成一长串 |
| `image_grid_thw` | (1, 3) | 每张图的网格 [[1, 16, 16]] |
| `mm_token_type_ids` | (1, 82) | 每个位置是文字（0）、图片（1）还是视频（2） |

`mm_token_type_ids` 是 transformers 5 新加的输入，模型用它找出图片的范围来计算三维位置（24.7 节）。缺了它，模型会报错提醒你。

### 在模型里：填进占位符

模型的 `forward` 做的事，和第 22 章的 `VLM.embed` 一模一样：

```python
inputs_embeds = embed_tokens(input_ids)                    # 所有位置先查词嵌入
image_embeds = visual(pixel_values, grid_thw)              # (64, 1024)
mask = (input_ids == image_token_id)                       # 占位符的位置
inputs_embeds = inputs_embeds.masked_scatter(mask, image_embeds)   # 按顺序填进去
```

`masked_scatter` 把 `image_embeds` 按顺序填进 mask 为真的位置，等价于我们的 `emb[mask] = v.reshape(-1, d)`。第 3 部分用钩子（hook）截获语言模型收到的 `inputs_embeds`，验证占位符位置上正好是视觉塔的输出（最大差 0），文字位置上正好是词嵌入（最大差 0）。

---

## 24.6 视觉塔与连接器

Qwen3.5 的视觉塔（`model.visual`）：

```
pixel_values (256, 1536)
  │ Conv3d，核 (2, 16, 16)                                    ← patch 嵌入
  ▼
(256, 768)
  │ + 可学习的位置表（48 × 48 = 2304 格，双线性插值到 16 × 16） ← 绝对位置
  ▼
12 个 Transformer 块，每块的注意力里用二维 RoPE                ← 相对位置
  ▼
(256, 768)
  │ merger：LayerNorm → 每 4 行拼成 3072 维 → Linear(3072, 3072) → GELU → Linear(3072, 1024)
  ▼
视觉 token (64, 1024)
```

和第 19、22 章对照：

- **两种位置编码同时用**：可学习的表（第 19 章方法一）在输入处加一次，二维 RoPE（方法三）在每层的注意力里用。表的大小固定是 48×48，遇到其他大小的网格就插值（思路同第 19 章的 `interpolate_pos`；第 19 章用双三次插值，Qwen 用双线性插值、`align_corners=True`），这和第 19 章第 6 部分"可学习的表 + 插值换分辨率几乎不掉点"的实验一致；
- **merger 就是第 22 章的 `MergeProjector`**：LayerNorm → 拼接 4 个 patch → 两层 MLP。我们的版本在拼接之后做 LayerNorm，Qwen 在拼接之前做，差别不大；
- **注意力是非因果的**，而且每张图只在自己的 patch 之间做注意力：多张图拼成一长串后，用 `cu_seqlens`（每张图的起止位置）把它们隔开。

### 视觉 token 的"音量"

第 3 部分还测了一个有意思的量：语言模型输入里，文字嵌入的平均长度（L2 范数）是 0.65，视觉 token 是 4.53，**大了 7 倍**。视觉 token 的分布和文字嵌入很不一样，语言模型在训练中学会了处理这两种"音量"不同的输入。如果你自己拼一个 VLM，连接器的输出尺度和词嵌入差得太远，训练初期可能不稳定。这也是为什么 merger 里有 LayerNorm。

---

## 24.7 M-RoPE：三维的位置编号

### 问题

第 16 章的 RoPE 给每个 token 一个整数位置 0, 1, 2, ……，按位置旋转查询和键。第 19 章的二维 RoPE 给每个 patch 一对（行，列）。语言模型里，序列中既有文字（一维）又有图片（二维），视频还有时间维（三维）。怎样给它们统一编号？

Qwen2-VL 提出 **M-RoPE**（Multimodal RoPE）：每个 token 都有**三个**位置编号（时间, 行, 列），分别控制旋转角的不同部分。

### 编号规则

- **文字 token**：三个编号相同，都等于当前位置 pos，然后 pos 加 1。纯文字时，M-RoPE 就退化成普通的一维 RoPE；
- **一张图**（合并后 h × w 个 token）：第 (r, c) 个 token 的编号是 (pos, pos + r, pos + c)；
- 整张图结束后，pos 加 **max(h, w)**，而不是加 h × w。

第 4 部分打印了一个例子：一张 64×128 的图（合并后 2 行 × 4 列 = 8 个视觉 token）：

```
  位置    token              时间  行  列
     2    '\n'                 2   2   2
     3    '<|vision_start|>'   3   3   3
     4    '<|image_pad|>'      4   4   4
     5    '<|image_pad|>'      4   4   5
     6    '<|image_pad|>'      4   4   6
     7    '<|image_pad|>'      4   4   7
     8    '<|image_pad|>'      4   5   4
     9    '<|image_pad|>'      4   5   5
    10    '<|image_pad|>'      4   5   6
    11    '<|image_pad|>'      4   5   7
    12    '<|vision_end|>'     8   8   8
    13    'Describe'           9   9   9
```

- 图片前的 `<|vision_start|>` 在位置 3，图片从 pos = 4 开始；
- 第 0 行的 4 个 token：行号都是 4 + 0 = 4，列号 4、5、6、7；第 1 行：行号 5，列号 4–7；
- 时间编号都是 4（静态图只有一帧）；
- 图片结束，pos = 4 + max(2, 4) = 8，`<|vision_end|>` 的编号是 8。

**为什么加 max(h, w)**：一张 1000 个 token 的图，如果按序列长度编号，图后面的文字会被"推远" 1000 个位置。用 max(h, w)，一张 32×32 个 token 的图只占 32 个位置。好处有两个：图后面的文字离图前面的文字不远；位置编号的最大值远小于序列长度，长上下文的外推压力小。

序列有 24 个 token，最大编号却只有 19，差值 −4 记在 `rope_deltas` 里。生成新 token 时（KV cache 模式），模型用"序列位置 + rope_deltas"作为新 token 的三个编号。

### 练习 24.4：`mrope_positions`

实现 `mrope_positions(ids, grids, image_id, merge)`，返回 (3, L) 的编号。测试 `test_mrope_positions` 检查上面的规则；第 4 部分把你的实现和模型自己的 `get_rope_index` 比较，结果应该完全相同。

### 频率怎样分给三个轴

第 16 章的 RoPE 把头维度分成 d/2 对，每对用一个频率 ω_i 旋转，角度 = 位置 × ω_i。ω_i 从高到低：前面的频率转得快（分辨近距离），后面的转得慢（分辨远距离）。

M-RoPE 要把这些频率**分给三个轴**：分到时间轴的频率用时间编号算角度，分到行轴的用行号，分到列轴的用列号。

Qwen3.5 的头维度是 256，但 `partial_rotary_factor: 0.25` 表示只有前 64 维旋转（剩下的 192 维不加位置信息），也就是 32 个频率。`mrope_section: [11, 11, 10]` 表示时间、行、列分别分到 11、11、10 个。

怎么分有两种方案：

```
分段式（Qwen2-VL、Qwen2.5-VL，mrope_section [16, 24, 24]，64 个频率）：
TTTTTTTTTTTTTTTTHHHHHHHHHHHHHHHHHHHHHHHHWWWWWWWWWWWWWWWWWWWWWWWW
高频 ──────────────────────────────────────────────────────▶ 低频

交错式（Qwen3-VL、Qwen3.5，mrope_interleaved: true，32 个频率）：
THWTHWTHWTHWTHWTHWTHWTHWTHWTHWTH
```

分段式的问题：时间轴只分到最高频的 16 个，列轴只分到最低频的 24 个。低频转得极慢，相邻两列的角度几乎一样，**列方向的近距离分辨率很差**。交错式让每个轴都分到从高频到低频的整个频段。

第 4 部分用一个"探针"实测了 Qwen3.5 的频率分配：构造三个位置，分别只在时间、行、列轴上偏移，看每个频率的旋转角跟着哪个轴变。实测结果和 `mrope_axes` 的预测完全一致。

---

## 24.8 语言模型：线性注意力与全注意力的混合

### 结构

Qwen3.5 的 `layer_types` 是 `[linear, linear, linear, full] × 6`：24 层里 18 层是**线性注意力**（Gated DeltaNet），每 4 层有 1 层是普通的**全注意力**（第 15 章的 softmax 注意力，加了 GQA 和一个输出门）。

### 线性注意力是什么

第 15 章的注意力：每个新 token 的查询 q 和**所有**之前的键 k 算分数，softmax 后对值 v 加权求和。要存下所有的 k、v（KV cache），计算量和存储都随序列长度增长。

线性注意力换一种思路：用一个**固定大小的矩阵 S**（"状态"，128 × 128）概括之前所有的信息，每来一个 token 就更新一次：

```
S ← S · α_t                         # 衰减：α_t ∈ (0, 1)，由当前 token 决定，"忘掉"一部分旧信息
S ← S + β_t · k_t (v_t − Sᵀk_t)ᵀ     # delta 规则：先用 S 预测 k_t 对应的值 Sᵀk_t，再按误差修正
o_t = Sᵀ q_t                         # 输出：用查询从状态里"读"出值
```

- **delta 规则**的直觉：S 是一个"键 → 值"的联想记忆。新来一对 (k_t, v_t) 时，先查一下 S 里 k_t 现在对应什么（Sᵀk_t），只把差值 v_t − Sᵀk_t 写进去，写入的强度是 β_t ∈ (0, 1)。这和第 7 章的梯度下降很像：β_t 是学习率，误差是 v_t − Sᵀk_t；
- **门控**（gated）：α_t 让模型可以有选择地遗忘，比如读到新的段落时清空旧的记忆；
- 每层还有一个核大小为 4 的**因果卷积**（`linear_conv_kernel_dim: 4`），让每个 token 先和前 3 个 token 混合一下，再计算 q、k、v。

它是一个**循环**（recurrent）结构：按顺序一个一个 token 地更新状态。训练时有分块并行的算法（`chunk_gated_delta_rule`），推理时每步的计算量和存储都是常数，与序列长度无关。代价是固定大小的状态记不住所有细节，"精确地回看第 3,000 个 token"这类任务不如全注意力。所以 Qwen3.5 每 4 层保留一层全注意力，负责精确的检索。

没装 `flash-linear-attention` 和 `causal-conv1d` 这两个库时，transformers 会退回到纯 PyTorch 的参考实现（会打印一行提示），结果正确，只是慢一些。

### KV cache 的节省

第 3 部分算了一笔账（bf16，每个数 2 字节）：

- 如果 24 层都是全注意力：每个 token 要存 24 层 × 2（K 和 V）× 2 个键值头 × 256 维 × 2 字节 = 48 KB；
- 实际只有 6 层存 KV：每个 token 12 KB；18 层线性注意力各存一个 16 头 × 128 × 128 的状态（float32），共 18 MB，**与长度无关**；
- 序列 10 万个 token 时：全注意力版 4.58 GB，混合版 1.16 GB。

对 OCR 来说这很重要：一页高分辨率的文档有几千个视觉 token，一份多页文档有几万个。

### 位置信息从哪来

线性注意力是循环的，**天然知道顺序**：S 是按顺序一个一个更新的，先来的信息被衰减得更多。因果卷积也是有方向的。所以 18 层线性注意力不需要任何位置编码，只有 6 层全注意力用 RoPE，而且只用 1/4 的维度。

这带来一个推论：Qwen3.5 的语言模型对位置编号应该**不太敏感**。第 7 部分的实验验证了这一点，结果出人意料。

---

## 24.9 SmolVLM：另一条路

SmolVLM 基于 Idefics3 的结构，视觉塔是 SigLIP（patch 16，固定 512×512 的输入），语言模型是 SmolLM2（Llama 结构，普通的一维 RoPE）。它处理任意大小图片的方法，和 Qwen 的"原生分辨率"完全不同：**切子图**。

### 切子图

1. 把图片的**长边缩放到 2048**（小图会被放大），短边按比例；
2. 两边各自向上取整到 512 的倍数；
3. 切成若干块 512×512 的子图（tile），再加一张把整张图压缩到 512×512 的**全局图**；
4. 每块子图都单独过 SigLIP：512 / 16 = 32，得到 32 × 32 = 1024 个 patch；
5. **pixel shuffle 4×4**：把相邻的 4×4 = 16 个 patch 拼成一个，1024 → **64 个 token**。这是第 22 章 2×2 合并的加强版：每个 token 的维度变成 768 × 16 = 12,288，再用一个线性层映射到语言模型的 576 维。

一张 192×256 的小图：长边放大到 2048 → 1536×2048 → 3 行 × 4 列 = 12 块子图 + 1 张全局图 = 13 块，13 × 64 = **832 个视觉 token**。同一张图在 Qwen3.5 里只有 70 个。

可以通过 processor 的 `size={"longest_edge": 512}` 把长边设成 512：只有一块，64 个 token。本章的零样本实验用这个设置（玩具图本来就小）。

### 提示里的子图标记

SmolVLM 展开后的提示长这样（节选）：

```
<|im_start|>User:<fake_token_around_image><row_1_col_1><image><image>…（64 个）
<fake_token_around_image><row_1_col_2><image>…
…
<fake_token_around_image><global-img><image>…（64 个）<fake_token_around_image>Describe the image.<end_of_utterance>
Assistant:
```

每块子图前面有一个**文字标记** `<row_r_col_c>` 告诉模型这一块在哪；`<fake_token_around_image>` 是子图之间的分隔符；`<global-img>` 标记全局图。位置信息不是靠 M-RoPE，而是靠这些"写出来"的标记：语言模型用普通的一维位置，子图内部的空间结构由 SigLIP 负责。

### 两种方案的比较

| | Qwen：原生分辨率 | SmolVLM：切子图 |
|---|---|---|
| 视觉塔的输入 | 任意大小（32 的倍数） | 固定 512×512 |
| 视觉塔的位置编码 | 可学习表（插值）+ 二维 RoPE | 可学习表（固定大小） |
| 大图的处理 | 一整张图做注意力，token 多时计算量是平方增长 | 每块子图独立做注意力，计算量是线性增长 |
| 跨子图的信息 | 天然有 | 子图边界上的字被切断；靠全局图和语言模型补 |
| 语言模型里的位置 | M-RoPE（三维） | 一维 RoPE + 文字标记 |
| 视觉 token 数 | ≈ 面积 / 1024，平滑变化 | 64 × (子图数 + 1)，按块跳变 |

切子图的方案在 LLaVA-NeXT（"AnyRes"）、InternVL、MiniCPM-V 里也很常见。它的好处是可以直接复用固定分辨率预训练的视觉塔（SigLIP、CLIP），不必重新训练。对 OCR 来说，子图边界上的字会被切开，是一个已知的弱点。第 33 章 33.1 节会算原生分辨率下一整页要多少视觉 token，那是这两种方案取舍的出发点。

---

## 24.10 实验

本章的实验全部是**推理**，不训练任何参数，所以下面引用的数字都是在作者机器上真实跑出来的（CPU，8 线程；日志在 `outputs/ch24/run.log`）。你自己跑一遍，零样本的准确率会因为 transformers 版本、线程数带来的浮点差异而有一两个百分点的出入，但结论应该相同。

```bash
python -m labs.ch24.hf_tour 1 2          # 配置和 processor：不加载权重，几秒
python -m labs.ch24.hf_tour 3 4          # 模型内部与 M-RoPE：加载 Qwen3.5，约 1 分钟
python -m labs.ch24.hf_tour 5 6 7        # 零样本测试：CPU 上约 40 分钟
N_TEST=10 python -m labs.ch24.hf_tour 5  # 先用 10 个样本试跑
```

**运行之前，先写下你对这几个问题的预测**（写在纸上，跑完再对照）：

1. 一张 3000×4000 的照片，Qwen3.5 和 SmolVLM 谁的视觉 token 多？多几倍？
2. Qwen3.5-0.8B 从没见过 ShapesWorld，"数物体"能对多少？"写出完整的描述"呢？
3. 把一行字放大 2 倍、3 倍、4 倍再给 Qwen3.5 读，CER 会怎样变化？
4. 把语言模型的位置编号全换成 0，Qwen3.5 和 SmolVLM 的损失各会涨多少？

### 第 1 部分：配置对照

| 模型 | 总参数 | 视觉塔 | 连接器 | 语言模型 |
|---|---|---|---|---|
| Qwen3.5-0.8B | 853M | 88.0M（10%） | 12.6M（1.5%） | 752.4M（其中嵌入 254M） |
| Qwen2.5-VL-3B | 3755M | 632.0M（17%） | 36.7M（1.0%） | 3085.9M |
| SmolVLM-256M | 256M | 86.4M（34%） | 7.1M（2.8%） | 163.0M |
| LLaVA-1.5-7B | 7063M | 303.5M（4%） | 21.0M（0.3%） | 6738.9M |

三个观察：

- **连接器永远很小**（0.3%–2.8%）。第 22 章说"连接器只是一座桥"，在真实模型里也成立；
- **小模型里视觉塔的比例高**：SmolVLM 的视觉塔（SigLIP-base）和 Qwen3.5 的差不多大，但它的语言模型小得多，视觉塔就占了 1/3；
- **Qwen3.5-0.8B 的 0.75B 语言模型里，1/3 是词嵌入**（词表 24.8 万 × 1024 维）。词表大是为了多语言，本课程只用英文，这部分参数大多用不上。

第 1 部分还打印了同一张图在四个模型里的视觉 token 数（SmolVLM 一列是默认的长边 2048，括号里是长边设成 512）：

| 原图（高×宽） | Qwen3.5 | Qwen2.5-VL | SmolVLM | LLaVA-1.5 |
|---|---|---|---|---|
| 64×64 | 64 | 4 | 1,088（64） | 576 |
| 32×256 | 69 | 9 | 320（64） | 576 |
| 448×448 | 196 | 256 | 1,088（64） | 576 |
| 384×576 | 216 | 294 | 832（64） | 576 |
| 1024×1024 | 1,024 | 1,369 | 1,088（64） | 576 |
| 3000×4000 | 11,750 | 15,301 | 832（64） | 576 |

小图上 Qwen3.5 比 Qwen2.5-VL 多（`min_pixels` 不同：256² 对 56²），大图上反过来（Qwen2.5-VL 每个 token 只管 28×28 像素）。这张表最值得看的是最后一行：3000×4000 的照片，Qwen3.5 是 11,750 个，Qwen2.5-VL 是 15,301 个，SmolVLM 是 832 个，LLaVA 永远是 576 个。**原生分辨率的模型，token 数随面积增长；切子图的模型，有上限；固定分辨率的模型，是常数**。哪个更好取决于任务：读一张照片里的大字，576 个就够；读一页小字，可能几千个都不够。

### 第 2 部分：processor 与 transformers 逐元素对照

两张图（64×64 的场景、32×256 的文字行）经过我们的 `qwen_preprocess`，与 transformers 的 `Qwen2VLImageProcessorPil` 比较，`pixel_values` 最大差 **0.0**、`image_grid_thw` 相同。这说明 24.3、24.4 节描述的每一步（缩放规则、插值方法、归一化、2 帧复制、2×2 窗口顺序）都没有遗漏。

如果你的实现差了一点点（比如最大差 0.01），最常见的原因是插值：Qwen 用 PIL 的 BICUBIC；用 `torch.nn.functional.interpolate` 的 bicubic 结果会稍有不同（边界处理和抗锯齿不一样）。

### 第 3、4 部分：模型内部

- 19 个 token 的提示，展开占位符后变成 82 个（64 个 `<|image_pad|>`），与"先复制占位符再分词"的结果相同；
- 语言模型输入里，占位符位置的向量**正好等于** merger 的输出（最大差 0），文字位置**正好等于**词嵌入。`masked_scatter` 就是这么简单；
- 视觉 token 的平均长度是文字嵌入的 7 倍（4.53 vs 0.65），见 24.6 节；
- 我们的 `mrope_positions` 与 `model.get_rope_index` 完全相同；64×128 的图（合并后 2×4）让 rope_deltas = −4；
- 频率探针实测的轴分配 `THWTHW…TH` 与 `mrope_axes` 的预测相同。

这几项是"对照检查"：它们不告诉你新东西，但保证你对模型的理解和真实代码一致。**读真实模型代码时，养成"自己实现一遍，再与原版逐元素比较"的习惯**，比读十遍源码更可靠。

### 第 5 部分：零样本做玩具任务

| 任务 | Qwen3.5-0.8B | SmolVLM-256M |
|---|---|---|
| 写出完整描述（caption） | 整句正确 36%，CER 20.5% | 0%，CER 141.7% |
| 数物体（count） | 98% | 74% |
| 最左边物体的颜色（left） | 86% | 52% |
| 某物体是否存在（exists） | 98% | 86% |

（每个任务 50 个样本。CPU 上 Qwen3.5 每个任务约 90–200 秒，SmolVLM 约 30–70 秒。）

怎么读这张表：

- **简单的问答，Qwen3.5 不用训练就几乎全对**。数物体 98%、判断存在 98%：这两个任务只需要"认出形状和颜色"，真实模型在海量图片上早就学会了；
- **caption 难在格式，不难在看**。Qwen3.5 的典型错误是：`a large blue triangle, a large green circle, a small green square` → `a large blue triangle, a small green circle, a small green s…`（大小判断错了一个，加上回答被截断）；另一个例子里三个物体都认对了，但**顺序**不对——我们的描述是按从左到右排列的，模型不知道这个约定。CER 20.5% 里有相当一部分是格式和顺序，不是视觉错误；
- **SmolVLM 的 caption 几乎完全失败**：回答是 `A large red circle.`、`A.` 这样的短句。256M 的模型不太会遵循"按某种格式列出所有物体"这类指令。它在问答上的 52%–86% 也说明，**模型的大小首先影响的是理解指令的能力**；
- **"最左边"比"数数"难**：两个模型在 left 上都是最差的问答任务（86%、52%）。空间关系需要把颜色和位置绑定在一起，这正是第 22、23 章里我们的小 VLM 也最难学的部分。

第 26 章在我们自己的小 VLM 上做监督微调，第 35 章再把同样的流程搬到 Qwen3.5、SmolVLM 上，你可以亲手看几百步 LoRA 训练能把文字行的 CER 降到多少。

### 第 6 部分：零样本 OCR

测试集是第 13 章的前 50 行文字（32×256 的灰度图）。

| 设置 | 视觉 token | CER | 整行正确 |
|---|---|---|---|
| Qwen3.5，原图（1×） | 8 | 27.85% | 46% |
| Qwen3.5，放大 2× | 32 | **12.80%** | 60% |
| Qwen3.5，放大 3× | 72 | 20.11% | 56% |
| Qwen3.5，放大 4× | 128 | 22.08% | 58% |
| Qwen3.5，默认（smart_resize 到 96×736） | 69 | 20.25% | 60% |
| SmolVLM，长边 512 | 64 | 345.99% | 2% |
| SmolVLM，长边 1024 | 192 | 146.27% | 8% |
| 对照：第 13 章 CRNN（0.8M 参数，专门训练） | — | 2.28% | 80.1% |

这张表里有三个出人意料的地方。

**一、专门训练的 0.8M 小模型，比 0.8B 的通用 VLM 好 5 倍以上。** 第 13 章的 CRNN 只见过这一种字体、这一种行高，它在自己的分布上是专家。通用 VLM 什么都见过，但没见过**这个**分布。这是整个课程反复出现的主题：**数据分布对上了，小模型也很强；对不上，大模型也会错**。第 26 章开始的监督微调、第 33–35 章的文档 OCR 数据工程和真实模型微调，就是要把这两者的优点结合起来。

**二、CER 不随视觉 token 单调下降。** 从 1× 到 2×，token 从 8 个变成 32 个，CER 从 27.85% 降到 12.80%，符合直觉；但 3×、4× 反而又升到 20% 以上。整行正确率却一直稳定在 56%–60%。两个指标的走势不一样，说明 CER 的变化不是"每行都读得更差了"。

为了弄清楚原因，作者把 2× 和 3× 的逐行结果按编辑距离排序（只是推理，几分钟）：

| | 50 行的总编辑距离 | 总字符数 | 最差 5 行占总编辑距离的比例 |
|---|---|---|---|
| 2× | 91 | 711 | 67% |
| 3× | 143 | 711 | 59% |

最差的几行长这样：

| 放大 | 真值 | 模型读出 |
|---|---|---|
| 2× | `listen money minute` | `after recession methods` |
| 2× | `$225.24 story was` | `3205.16500000` |
| 2× | `even 649 20 keep top` | `ени 040 20 кгр тп.` |
| 3× | `listen money minute` | `дефолт незначительных ошибок` |
| 3× | `us an 09-07-2006 there` | `淋 雨10917-22美 10917` |
| 3× | `first $198.50` | `测试150002` |

这不是"看不清"——看不清的错误是 `it` 读成 `ii`、`ON` 读成 `on` 这种一两个字符的错。这是**幻觉**（hallucination）：模型输出了一段和图片无关、甚至换了语言和文字系统（俄文、中文）的文本，或者一串停不下来的数字。一行幻觉贡献的编辑距离可能是正常错误行的十几倍。

所以第 6 部分的 CER 实际上由**两种完全不同的错误**组成：大多数行的小错误（随分辨率改善），加上少数几行的灾难性幻觉（随机出现，与分辨率关系不大）。50 行的测试集里，多一两行幻觉，CER 就能差 7 个百分点。"3× 比 2× 差"这个结论，**在 50 行上是不可靠的**。

这正是第 25 章要系统讨论的问题：

- 测试集太小，指标的置信区间很宽（第 25 章会算出：n = 100 时，CRNN 的 CER 置信区间宽约 2 个百分点；这里的幻觉让方差更大）；
- 语料级 CER 对离群值非常敏感，要同时看整行正确率、NED（每行先归一化再平均）和"最差 k 行"；
- 按错误类型分桶，比只看一个总数有用得多。

**三、SmolVLM 的 CER 超过 100%。** CER = 编辑距离 / 真值长度，模型输出比真值长得多时，CER 可以远超 100%。长边 512 时，它把一行 `every FAST: first face` 读成了 `### Image Description\n\nThe image is a scanned page from a book, specifically a book titled "The Great Gatsby"…`——它没把任务理解成"读出文字"，而是"描述图片"。256M 的模型对指令的服从度不够，这和第 5 部分 caption 的失败是同一个原因。

### 第 7 部分：位置编号有多重要

用 teacher forcing（把正确答案喂给模型，只看每个答案 token 的损失，不生成、不训练）测量换掉位置编号的影响。"三行文字"是把三行文字竖着拼成一张图，让模型读出全部三行；"最左边的颜色"用第 5 部分的场景图。

| 位置编号 | 三行文字：损失 / 逐 token 正确 | 最左边颜色：损失 / 逐 token 正确 |
|---|---|---|
| M-RoPE（原样） | 0.549 / 86.4% | 0.803 / 76.7% |
| 一维位置（图片 token 也按顺序编号） | 0.558 / 86.5% | 0.825 / 75.0% |
| 图内全同位置 | 0.552 / 86.3% | 0.776 / 76.7% |
| 图内行列互换 | 0.551 / 86.5% | 0.815 / 76.7% |
| 全部位置 = 0 | 0.538 / 86.4% | 0.874 / 73.3% |

再对同一段纯英文文本，比较两个语言模型的每 token 损失：

| | 正常位置 | 全部 = 0 | 随机打乱 |
|---|---|---|---|
| Qwen3.5-0.8B | 1.96 | 1.95 | 1.96 |
| SmolVLM-256M | 2.21 | 5.11 | 5.98 |

这是本章最出人意料的结果：

- **Qwen3.5 对位置编号几乎完全不敏感**。三行文字的损失变化不到 0.02，连"全部 = 0"都没有变差；最左边颜色的变化最大也只有 0.07（全部 = 0 时，逐 token 正确率从 76.7% 降到 73.3%，即 30 个样本里差 1 个）。纯文字的损失在三种设置下都是 1.95–1.96；
- **SmolVLM 正好相反**：位置全换成 0，损失从 2.21 涨到 5.11（困惑度从约 9 涨到约 166），随机打乱更糟。它是普通的 Llama 结构，所有层都是全注意力，顺序信息**只能**从 RoPE 来；
- 差别来自 24.8 节讲的结构：Qwen3.5 有 18 层线性注意力和因果卷积，它们是循环的，**顺序已经写在计算过程里**；视觉 token 进入语言模型时，又已经带着视觉塔里可学习位置表和二维 RoPE 编码的空间信息。

24.14 节会讨论这个结果能说明什么、不能说明什么。

### 自己做的扩展

1. **把第 6 部分的测试集扩大到 500 行**（`N_TEST=500`，CPU 上几个小时，可以只跑 2× 和 3× 两种设置）。2× 和 3× 的差距还在吗？用第 25 章的 `paired_bootstrap` 算差距的置信区间；
2. **统计幻觉**：定义"幻觉行"为编辑距离大于真值长度一半的行（NED > 0.5），统计每种放大倍数下幻觉行的个数；去掉幻觉行后，CER 是否随分辨率单调下降？
3. **压制幻觉**：在提示里加一句 "Output only the text in the image, in English."，或者把 `max_new_tokens` 设成真值长度的 2 倍。幻觉行变少了吗？哪种方法更有效？
4. **第 7 部分的长序列版本**：把三行文字换成十行、二十行（图片更高，视觉 token 更多）。位置编号的影响会不会随序列变长而变大？

## 24.11 自己动手

### 必做

1. **完成练习 24.1–24.4**，使 `python -m pytest labs/ch24` 全部通过（8 项；最后一项需要装 transformers）。
2. **手算 smart_resize**：一张宽 1000、高 750 的照片，Qwen3.5 的默认参数，缩放后是多大？网格多少？视觉 token 多少个？把 `max_pixels` 设成 512×512 呢？用 `smart_resize` 核对。
3. **手算 Qwen 的 patch 顺序**：一个 4×6 的网格（4 行、6 列），按 Qwen 的顺序，第 6、7、8、9 个 patch 分别在第几行第几列？
4. **手算 M-RoPE**：序列是 "文字 × 3、一张合并后 3×2 的图、文字 × 2"，写出所有 11 个 token 的 (时间, 行, 列) 编号。最大编号是多少？rope_deltas 是多少？
5. **数 KV cache**：Qwen2.5-VL-3B 的语言模型 36 层、2 个键值头、头维度 128，bf16。一张 3000×4000 的照片（第 1 部分的表）加 200 个文字 token，KV cache 有多大？换成 Qwen3.5-0.8B 呢？

### 选做

6. **SmolVLM 的切图**：仿照 `qwen_preprocess`，实现 SmolVLM 的完整预处理（长边缩放 → 512 的倍数 → 切块 + 全局图），与 `AutoProcessor` 的 `pixel_values` 比较（提示：它用 LANCZOS 插值、mean = std = 0.5）。
7. **SmolVLM 的提示**：自己写一个函数生成展开后的提示字符串（`<fake_token_around_image><row_1_col_1><image>×64 …`），与 processor 的结果逐个 token 比较。
8. **max_pixels 与 OCR**：对第 33 章会用到的整页文档（`miniocr.data.page.sample_page`），把 `max_pixels` 从 256² 调到 1024²，观察 Qwen3.5 读出的文字的 CER 怎样变化、每页要多少时间。
9. **思考模式**：把聊天模板的 `enable_thinking` 设成 True，让 Qwen3.5 先思考再回答"最左边的物体是什么颜色"。准确率变了吗？生成的 token 数变了多少？
10. **两个模型的错误对比**：在第 5 部分的 `count` 任务上，按物体数统计两个模型的错误。它们在哪种图上最容易错？

## 24.12 常见错误

| 现象 | 原因 | 怎么查 |
|---|---|---|
| `AutoProcessor` 报"requires the Torchvision library" | Qwen3-VL 系列的视频处理器依赖 torchvision | 装 torchvision，或者像本章一样分开用分词器和 PIL 版的图像处理器 |
| 报"Image features and image tokens do not match" | 占位符没有展开，或者展开的个数与 `image_grid_thw` 不符 | 数一数 `input_ids` 里 image_token_id 的个数，与 `grid.prod() // 4` 比较 |
| 报"mm_token_type_ids is missing" | 自己组装输入时漏了这一项 | `mm_token_type_ids = (input_ids == image_token_id).int()` |
| 模型能跑，但回答和图无关 | mean/std 用错，或者图片通道顺序不对（BGR） | 打印 `pixel_values` 的范围和均值；用 `preprocessor_config.json` 的参数 |
| 小字读错，大字读对 | 视觉 token 太少，图被缩得太小 | 提高 `min_pixels` / `max_pixels`，或先把图放大 |
| 显存 / 内存爆掉 | 大图的视觉 token 太多（3000×4000 有 11,750 个） | 设 `max_pixels`；先算 token 数再推理 |
| 输出里有 `<think>` 一大段 | 打开了思考模式 | 聊天模板里 `enable_thinking=False`（默认） |
| 生成停不下来，或者在奇怪的地方停 | `eos_token_id` 不对；SmolVLM 的结束标记是 `<end_of_utterance>` | 检查 `generation_config.json`；解码时用 `skip_special_tokens` |
| 打印一行"falling back to its reference PyTorch implementation" | 没装 flash-linear-attention / causal-conv1d | 不是错误，结果正确，只是慢；需要速度时再装 |
| LLaVA-1.5 读长条形的图很差 | 它把图裁成中间的 336×336 | 换用原生分辨率或切子图的模型，或者先把图补成正方形（padding） |

## 24.13 真实世界对照

- **更多的原生分辨率模型**：Qwen2-VL / 2.5-VL / 3-VL / 3.5、GLM-4.1V、Kimi-VL 都用"任意分辨率的 ViT + 二维 RoPE"。DeepSeek-OCR 则走另一条路：用一个专门设计的编码器把整页文档**压缩**到很少的视觉 token（几十到几百个），第 33 章 33.1 节会简单介绍这种"光学压缩"；
- **切子图的模型**：LLaVA-NeXT（AnyRes）、InternVL 2/2.5（动态切图，448×448 的块，pixel shuffle 2×2 后每块 256 个 token）、MiniCPM-V、SmolVLM、Idefics3；
- **位置编码**：Qwen2-VL 提出 M-RoPE（分段式）；Qwen2.5-VL 把时间编号和真实时间（秒）对齐；Qwen3-VL 改成交错式（interleaved MRoPE），并且视频帧之间插入文字时间戳；
- **混合注意力**：Qwen3-Next、Qwen3.5 用 Gated DeltaNet 和全注意力 3:1 混合；同类的还有 Mamba 系列（Jamba、Nemotron-H）、MiniMax-01 的 lightning attention。它们都是为了降低长上下文的 KV cache 和计算量；
- **vLLM 等推理引擎**：真实部署时，processor 的这些步骤由推理引擎（vLLM、SGLang）调用 transformers 的 processor 完成，视觉塔和语言模型分开调度。本课程不展开部署，第 35 章搬到真实模型时会再提到它们和 transformers 的对应关系。

## 24.14 深入一步

### 为什么 factor 是 patch × merge

如果网格的长或宽是奇数，2×2 合并会剩下一行或一列。有三种处理方法：补齐（pad）、丢掉、要求网格必须是偶数。Qwen 选了第三种：在缩放时就让长宽都是 32 的倍数，模型里的合并永远不需要特殊处理。代价是缩放会轻微改变宽高比（192×256 → 224×320，宽高比 1.33 → 1.43）。

### 视觉 token 和文字 token 的"汇率"

一个视觉 token 对应 32×32 像素。一行 19 像素高的英文字，每个字符约 10 像素宽，一个 32×32 的区域里大约有 3 个字符。对应到文字 token：英文的 BPE 平均一个 token 约 4 个字符。所以在"刚好看得清"的分辨率下，**一个视觉 token 携带的文字信息和一个文字 token 差不多**。这是一个粗略但有用的估算：一页 2000 个字符的文档，至少需要几百个视觉 token。DeepSeek-OCR 研究的就是这个汇率能压缩到多少（第 33 章 33.1 节）。

### 位置编号的"空洞"

M-RoPE 里，图片结束后 pos 加 max(h, w)，导致图片的 h × w 个 token 共享 max(h, w) 个编号范围。一维的位置编号上出现了"空洞"：位置 4–7 被 8 个 token 共用。这对模型没有问题，因为 RoPE 只关心编号的差，不要求编号连续。但它意味着 **"序列长度"和"最大位置编号"不再相等**，长上下文的长度限制（`max_position_embeddings: 262144`）指的是位置编号，不是 token 数。

### 第 7 部分的结果说明了什么

第 7 部分发现，把 Qwen3.5 语言模型的位置编号全部换成 0，困惑度几乎不变；而 SmolVLM（普通的 Llama 结构）的损失从 2.21 涨到 5.11，困惑度从约 9 涨到约 166。这告诉我们两件事：

1. **混合结构里，顺序信息主要来自线性注意力**（循环结构天然有方向），RoPE 的作用很小。一些研究甚至发现，去掉位置编码（NoPE）的因果 Transformer 也能从因果掩码里"数出"位置；
2. **视觉塔已经编码了空间结构**：视觉 token 在进入语言模型之前，已经带着可学习位置表和二维 RoPE 的信息。语言模型里的 M-RoPE 是"锦上添花"。

但这不意味着 M-RoPE 没用。我们只测了很短的序列（不到 200 个 token）、不重新训练、只看困惑度。在几万个 token 的长文档里、在需要精确定位（输出坐标）的任务里，位置编号的作用可能大得多。**一个实验只能回答它测量的那个问题**。

## 24.15 横向对照

| | 第 22 章的 VLM | Qwen3.5-0.8B | SmolVLM-256M | LLaVA-1.5-7B |
|---|---|---|---|---|
| 视觉塔 | ViT，4 层 d=128，64×64 | ViT，12 层 d=768，任意分辨率 | SigLIP，12 层 d=768，512×512 | CLIP ViT-L，24 层 d=1024，336×336 |
| patch | 8 | 16（Conv3d，2 帧） | 16 | 14 |
| 视觉塔的位置编码 | 可学习 / RoPE | 可学习表（插值）+ 二维 RoPE | 可学习表 | 可学习表 |
| 连接器 | MLP / 2×2 合并 / Q-Former | 2×2 合并 + MLP | pixel shuffle 4×4 + 线性 | 两层 MLP |
| 每张图的 token | 16–64 | 面积 / 1024 | 64 × (子图数 + 1) | 576 |
| 语言模型 | GPT，4 层 d=256 | 24 层 d=1024，线性 + 全注意力 | Llama，30 层 d=576 | Llama，32 层 d=4096 |
| 语言模型的位置 | 一维 RoPE | M-RoPE（交错式） | 一维 RoPE + 子图标记 | 一维 RoPE |
| 占位符 | `<img>` | `<\|image_pad\|>` | `<image>` | `<image>` |

| processor 的输出 | 形状 | 谁用它 |
|---|---|---|
| `input_ids` | (B, L) | 词嵌入；找占位符 |
| `attention_mask` | (B, L) | 补齐位置不参与注意力 |
| `pixel_values`（Qwen） | (所有图的 patch 总数, 1536) | 视觉塔 |
| `image_grid_thw`（Qwen） | (图片数, 3) | 视觉塔切分图片；M-RoPE |
| `mm_token_type_ids`（Qwen） | (B, L) | M-RoPE 找图片的范围 |
| `pixel_values`（SmolVLM） | (B, 子图数, 3, 512, 512) | 视觉塔 |
| `pixel_attention_mask`（SmolVLM） | (B, 子图数, 512, 512) | 标出补齐的子图 |

## 24.16 本章小结

- 真实 VLM 和第 22 章的 VLM 是同一个骨架：视觉塔、连接器、语言模型、占位符；连接器只占 0.3%–3% 的参数，语言模型占大头；
- Qwen 的 processor 四步：`smart_resize`（长宽变成 32 的倍数，像素数在上下限之间）→ 归一化 → 切块（复制成 2 帧给 Conv3d，按 2×2 窗口排序）→ 展开占位符；我们的实现与 transformers 逐元素相同；
- 视觉 token 数 ≈ 面积 / (32×32)，用 `min_pixels`、`max_pixels` 控制；同一张图在不同模型里的 token 数可以差十几倍；
- M-RoPE 给每个 token 三个编号（时间、行、列），文字三个相同，图片按网格编号，图片结束后位置只前进 max(h, w)；Qwen3.5 用交错式分配频率，每个轴都覆盖高中低频；
- Qwen3.5 的语言模型 3/4 的层是线性注意力（Gated DeltaNet，固定大小的状态，delta 规则更新），KV cache 只有全注意力版的 1/4，长文档的内存压力小得多；它对位置编号也不敏感；
- SmolVLM 用切子图 + pixel shuffle，靠文字标记 `<row_r_col_c>` 告诉模型子图的位置；
- 零样本测试里，Qwen3.5 在简单问答（数物体、判断存在）上几乎全对，难的是遵循我们的描述格式（caption 整句正确 36%）；SmolVLM 连指令都不太会遵循。OCR 上，0.8B 的通用模型（CER 12.8%–27.9%）远不如专门训练的 0.8M CRNN（2.28%），而且 CER 被少数几行幻觉主导，50 行的测试集不足以比较分辨率设置——这是第 25 章的出发点。第 26 章起学习监督微调，第 35 章把它用到这两个真实模型上。

**大作业 A3**：第四部分到这里结束（里程碑 C）。接下来做 [A3：从零写 ViT、对比学习、连接器和一个会看图答题的小 VLM](assignments/a3_vlm/handout.md)：三种位置编码与二维 RoPE、CLIP/SigLIP、IoU/NMS/AP、三种连接器与聊天模板、带 KV cache 的 VLM 生成，再把本章的 Qwen 预处理自己写一遍、与 transformers 逐位对照；6 组实验，最后参加 ShapesWorld 九任务问答排行榜。预计 30–40 小时。

## 术语卡

| 中文 | 英文 | 一句话 |
|---|---|---|
| 处理器 | processor | 把原始图片和文字变成模型输入张量的组件 |
| 原生分辨率 | native resolution | 保持宽高比、按需调整 token 数的图像输入方式 |
| 智能缩放 | smart_resize | 把长宽变成 factor 的倍数、像素数落在上下限之间 |
| 时间 patch | temporal patch | 视频的"时空小块"；静态图复制成 2 帧 |
| 网格大小 | image_grid_thw | 每张图切块后的 (时间, 行, 列) |
| 多模态旋转位置编码 | M-RoPE | 每个 token 有时间、行、列三个位置编号 |
| 交错式 | interleaved | 频率按 T H W T H W … 轮流分给三个轴 |
| 线性注意力 | linear attention | 用固定大小的状态代替 KV cache 的注意力 |
| delta 规则 | delta rule | 只把"预测误差"写进联想记忆的更新方式 |
| 切子图 | image splitting / tiling | 把大图切成固定大小的块分别编码 |
| 像素重排 | pixel shuffle | 把相邻的 k×k 个 patch 拼成一个 token |
| meta 设备 | meta device | 只有形状、没有数据的"假"设备，用于数参数 |

## 自测题

**概念**

1. 为什么 LLaVA-1.5 的预处理（短边 336 + 中心裁剪）对 OCR 不友好？举两个例子。
2. `smart_resize` 的 factor 为什么是 patch × merge，而不是 patch？
3. Qwen 为什么把图片复制成两帧？这样做有什么代价？
4. Qwen 的 patch 顺序为什么按 2×2 窗口分组？如果按逐行扫描的顺序，merger 要怎么改？
5. `pixel_values` 为什么没有批维度？模型怎样知道哪些 patch 属于哪张图？
6. M-RoPE 里，图片结束后 pos 为什么加 max(h, w) 而不是 h × w？
7. 分段式和交错式的频率分配有什么区别？分段式有什么缺点？
8. 线性注意力的状态为什么是固定大小的？它和全注意力相比，擅长什么、不擅长什么？
9. SmolVLM 怎样让语言模型知道每块子图在原图的什么位置？
10. 把 Qwen3.5 语言模型的位置编号全部设成 0，困惑度几乎不变。为什么？这能说明 M-RoPE 没用吗？

**计算**

11. 一张宽 2480、高 3508 的 A4 扫描件（300 dpi），Qwen3.5 默认参数下有多少个视觉 token？Qwen2.5-VL 默认参数呢？SmolVLM 默认参数呢？
12. 序列 "文字 × 5、一张合并后 4×6 的图、文字 × 3"：序列长度、最大位置编号、rope_deltas 各是多少？
13. Qwen3.5 的 32 个旋转频率中，行轴分到了第几个频率（0 开始编号）？
14. SmolVLM 的 pixel shuffle 把 32×32 个 768 维的 patch 变成 64 个多少维的向量？连接器的线性层有多少参数（不含偏置）？

**代码**

15. 下面的展开占位符代码有什么问题？
    ```python
    n = grid.prod() // 4
    text = text.replace("<|image_pad|>", "<|image_pad|>" * n)
    ```
    （提示：一条消息里有两张不同大小的图。）
16. 怎样不加载权重就知道一个模型有多少参数？

## 参考答案

1. 中心裁剪会丢掉非正方形图片的两边：A4 文档的上下边、一行文字的左右两端（32×256 的行只剩中间 1/8）；缩放到 336 会让大图上的小字只剩一两个像素。
2. 切块后网格的长宽要能被 merge 整除，2×2 合并才不剩；像素长宽是 patch × 网格，所以要是 patch × merge 的倍数。
3. 为了让图片和视频共用一个 Conv3d 的 patch 嵌入（核 2×16×16）。代价是每个图片 patch 向量有一半的数是重复的，patch 嵌入的计算量翻倍。
4. 分组后相邻的 4 行就是一个 2×2 窗口，merger 只要 `view(-1, 4d)`。按逐行扫描的顺序，merger 要先 reshape 成网格、permute、再拼接（第 22 章的 `merge_2x2`），还需要知道每张图的网格大小。
5. 不同大小的图片 patch 数不同，堆不成规则的张量，所以全部拼成一长串。模型用 `image_grid_thw` 知道每张图有几个 patch，视觉塔里用 `cu_seqlens` 让每张图只在自己内部做注意力。
6. 图片的行号、列号最大只到 pos + h − 1、pos + w − 1，加 max(h, w) 正好让后面的文字编号比图片里的所有编号都大；加 h × w 会让后面的文字被"推远"很多，浪费位置编号的范围，增加长上下文外推的难度。
7. 分段式把连续的一段频率分给一个轴（时间拿最高频，列拿最低频）；交错式按 T H W 轮流分配，每个轴都有高中低频。分段式的列轴只有低频，相邻列的旋转角几乎相同，近距离的列分辨能力差。
8. 状态是一个 d_k × d_v 的矩阵，每个 token 用 delta 规则更新它，不需要存下所有的 k、v。擅长长序列（存储和每步计算都是常数）、概括性的信息；不擅长精确地检索很久以前的某个具体 token，所以要和全注意力层混合。
9. 每块子图前面插入文字标记 `<row_r_col_c>`，全局图前面插入 `<global-img>`，子图之间用 `<fake_token_around_image>` 分隔。
10. 24 层里 18 层是线性注意力，它是循环结构，天然知道顺序；只有 6 层用 RoPE，而且只有 1/4 的维度旋转；视觉 token 本身已经带着视觉塔编码的空间信息。不能说明 M-RoPE 没用：只测了短序列、没有重新训练、只看了困惑度。
11. Qwen3.5：smart_resize(3508, 2480)（参数顺序是高、宽）：四舍五入到 32 的倍数是 3520×2496，面积 879 万 < 4096²，不再缩放；网格 220×156，视觉 token = 220 × 156 / 4 = 8,580 个。Qwen2.5-VL：factor 28，3500×2492，面积 872 万 < 1284 万，网格 250×178，token = 11,125 个。SmolVLM：长边（高）缩放到 2048，宽 = int(2048 × 2480 / 3508) = 1447，取偶数 1448；取整到 512 的倍数：高 2048、宽 1536，4 行 × 3 列 = 12 块 + 1 张全局图 = 13 块，832 个 token。（用 `n_qwen_tokens`、`idefics3_layout` 核对。）
12. 序列长度 5 + 24 + 3 = 32；文字占 0–4，图片从 5 开始，行号 5–8，列号 5–10，图片之后 pos = 5 + max(4, 6) = 11，后面三个文字是 11、12、13；最大编号 13，rope_deltas = 13 + 1 − 32 = −18。
13. 第 1、4、7、……、31 个（i % 3 == 1，共 11 个）。
14. 每个 token 是 4×4 = 16 个 patch 拼起来，768 × 16 = 12,288 维；线性层 12,288 × 576 = 7,077,888 个参数（正是第 1 部分数出的连接器参数量）。
15. 两张图大小不同时，n 只按一张图算，而 `replace` 会把**所有**占位符都换成同样的个数。应该逐个替换：第 k 个占位符换成第 k 张图的个数（transformers 的实现就是按顺序逐个替换）。另外 `grid.prod()` 对 (图片数, 3) 的张量会把所有图的网格乘在一起，要用 `grid[k].prod()`。
16. 用 `AutoConfig.from_pretrained` 读配置，在 `torch.device("meta")` 下用 `from_config` 建模型，再数 `p.numel()`；meta 张量只有形状，不分配内存。
