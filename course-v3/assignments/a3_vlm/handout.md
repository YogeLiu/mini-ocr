# A3：视觉语言模型——从零写 ViT、对比学习、连接器和一个会看图答题的小 VLM

| | |
|---|---|
| 位置 | 第 24 章之后（里程碑 C） |
| 预计用时 | 30–40 小时 |
| 算力 | 写代码、跑测试：CPU。实验：CPU 可以做小配置，GPU 快得多。排行榜：一张租用的 24 GB 显卡，训练时间合计不超过 30 分钟（CPU 赛道 120 分钟） |
| 交付 | ① `a3/` 里的代码，`tests/adapters.py` 接好后全部测试通过；② 书面报告 `writeup.md`（从 `writeup_template.md` 复制）；③ 排行榜记录 |
| 满分 | 100 分：代码 56 分，书面 10 分，实验 28 分，排行榜 6 分 |

这个作业把第 19–24 章串成一条线：

1. **视觉编码器**（第 19 章）：切块、三种位置编码、换分辨率时的插值、二维 RoPE、ViT；
2. **对比学习**（第 20 章）：CLIP 和 SigLIP 的损失、检索召回率、零样本分类；
3. **框**（第 21 章）：IoU、NMS、AP——定位任务的答案是一个框，评测要用到它们；
4. **VLM**（第 22–23 章）：2×2 合并、三种连接器（MLP、PatchMerger、Q-Former）、聊天模板、把视觉 token 填进占位符、带 KV cache 的生成；
5. **真实模型的预处理**（第 24 章）：Qwen 系列的 smart_resize、切块顺序、占位符展开、M-RoPE 位置编号，和 transformers 逐位对照。

最后你会把自己的 ViT 和 A2 的 GPT 接起来，在 ShapesWorld 的九个任务（描述、数数、按颜色数、最左边、存在、定位、指代、两图比较、读字）上训练，并参加排行榜。

和 A1、A2 一样，你从一个空的包开始，自己决定代码结构，测试只规定行为。

---

## 1. 规则

### 1.1 能用什么、不能用什么

在 A1、A2 的规则基础上，`a3/` 里还**不能**用：

| 不能用 | 自己写 |
|---|---|
| `nn.LayerNorm`、`nn.MultiheadAttention`、`nn.TransformerEncoderLayer` 之类 | LayerNorm 就是减均值、除标准差、乘加；注意力在 A2 写过，这次加上非因果、填充 mask 和交叉注意力 |
| `F.gelu`、`F.interpolate`、`F.layer_norm`，以及 `torch.nn.functional` 里的一切 | GELU 用 `torch.erf`；双线性插值自己按坐标公式算 |
| `torchvision`（包括 `torchvision.ops.nms`、`box_iou`）、`timm`、`open_clip` | |
| transformers 的图像处理器和模型 | 测试会用它来对照你的结果，但 `a3/` 里不能 import |

**可以用**：`torch.erf`、`torch.meshgrid`、`torch.polar`、`torch.topk`、`torch.sort`、`torch.cummax` 这类通用运算；**PIL 的 `Image.resize`**（第五部分要和 transformers 逐位一致，它的缩放就是 PIL 的双三次插值，自己写双三次不是这个作业的重点）。

**可以把你自己 A1、A2 的代码复制进 `a3/`**：Linear、嵌入、RMSNorm、softmax、交叉熵、RoPE、注意力、GPT、AdamW、调度、裁剪。不能 import `assignments.a1_basics` 或 `a2_sequences`。

`miniocr` 里能用的仍然只有**数据生成器**：`miniocr.data.shapes`（`sample_scene`、`draw_scene`、`describe`）、`miniocr.data.instruct`（`TASKS`、`make_task`、`draw_text_scene`、`parse_box`、`box_text`；**不包括** `apply_chat_template`，那是练习 23.1，这次自己写）、`miniocr.data.corpus`。

`tests/test_a3_rules.py` 用语法树检查 `a3/`，规则以本节为准。

### 1.2 AI 助手和参考实现

与 A1 相同：可以问概念、问报错、让 AI 检查你的代码，不要让它替你写 `a3/` 里的代码。`assignments/solutions/` 在提交前不要打开。

### 1.3 前提

ShapesWorld 和指令数据的生成器不依赖任何练习。第五部分的对照测试需要 transformers（第 24 章已经装过）；没装时那一个测试会被跳过，但评分时会跑。

---

## 2. 准备

```
assignments/a3_vlm/
├── handout.md            本说明
├── a3/                   你的代码
├── tests/
│   ├── adapters.py       ← 你要填的接线
│   └── test_a3_*.py
├── leaderboard.py        排行榜
├── data/test.npz         排行榜测试集：9 个任务各 60 题，共 540 题、600 张 64×64 的图
└── writeup_template.md
```

```bash
python -m pytest -q assignments/a3_vlm/tests                 # 在 course-v3/ 下
python -m pytest -q assignments/a3_vlm/tests -k qwen
python -m assignments.a3_vlm.leaderboard info                # 看看测试集长什么样
```

建议顺序：第一部分（视觉）→ 第三部分（框，独立，很快）→ 第二部分（对比学习）→ 第四部分（VLM）。第五部分只依赖 PIL 和张量操作，随时可以做。

---

## 3. 第一部分：视觉编码器

### 3.1 问题（patchify）：切块（2 分）

`patchify(x, p)`：`(B, C, H, W) → (B, h·w, C·p·p)`，h = H/p、w = W/p。

- patch **按行排列**：第 i 行第 j 列的 patch 是第 i·w + j 个；
- 每个 patch 内部按 **(C, p, p)** 拉平——这个顺序让"切块 + 线性层"恰好等于核大小 = 步长 = p 的卷积：线性层的权重就是卷积权重 `reshape(d, C·p·p)`。

不能用 `unfold`。用 `reshape` 和 `permute` 一步做完。

**交付**：`run_patchify`；`test_patchify_matches_conv` 通过。

### 3.2 问题（pos_embed）：位置编码与插值（3 分）

1. **二维正弦**：`sincos_2d(h, w, d)` 返回 `(h·w, d)`。前 d/2 维编码**行号**，后 d/2 维编码**列号**；每一半都是一维正弦编码：第 2i 维 sin(t·ωᵢ)，第 2i+1 维 cos(t·ωᵢ)，ωᵢ = 10000^(−2i/(d/2))。
2. **插值**：可学习的位置表是按训练分辨率的网格学的，换分辨率时要把它当成一张 d 通道的图缩放。`interpolate_pos(pos, old_hw, new_hw)` 做双线性插值，坐标约定与 `F.interpolate(mode="bilinear", align_corners=False)` 相同：输出第 k 个位置对应输入坐标 `(k + 0.5)·(n_in / n_out) − 0.5`，小于 0 的截到 0，再在相邻两点之间线性插值（超出末尾的取最后一点）。先沿行插，再沿列插。

**交付**：`run_sincos_2d`、`run_interpolate_pos`；`test_sincos_2d`、`test_interpolate_pos` 通过。

### 3.3 问题（rope_2d）：二维 RoPE（3 分）

把每个头的 dh 维分成两半，前一半编码行号、后一半编码列号。具体约定（Qwen2-VL 的视觉塔用的就是这一种）：

```
q = dh / 4,   θ_k = base^(−k/q),   k = 0 … q−1
第 i·w + j 个 patch：ang = cat([i·θ₀ … i·θ_{q−1},  j·θ₀ … j·θ_{q−1}])        （长 dh/2）
cos = cos(cat([ang, ang]))，sin 同理                                           （长 dh，"前后两半"的约定）
```

应用时与 A2 的 RoPE 相同：`x·cos + rotate_half(x)·sin`。这次 `base` 默认取 100——图的网格很小，用 10000 的话大部分频率几乎不转。

**交付**：`run_rope_2d`；`test_rope_2d_matches_complex`（用复数乘法独立实现对照）、`test_rope_2d_relative`（内积只取决于行差和列差）通过。

### 3.4 问题（vit_layers）：GELU、LayerNorm（2 分）

- 精确版 GELU：x · ½(1 + erf(x/√2))；
- LayerNorm：对最后一维，有偏方差，eps = 1e-5，参数 `weight`（初始 1）、`bias`（初始 0）。

**交付**：`run_gelu`、`run_layer_norm`；`test_gelu_and_layer_norm` 通过。

### 3.5 问题（vit_block）：ViT 块（3 分）

Pre-LN，非因果：

```
x ← x + Attn(LN₁(x))
x ← x + MLP(LN₂(x))          MLP = fc2(GELU(fc1(x)))，中间维 4d
```

注意力：`q` 一个线性层（有偏置）；`kv` 一个线性层（有偏置）输出 2d，按 `view(B, T, 2, h, dh)` 拆成 k、v；合并各头后过 `o`（有偏置）。可选的 `key_mask (B, T)`：True 是真实 token，False 是填充（任何位置都不能看它）。可选的 cos/sin：给出时对 q、k 加 RoPE。

把 q 和 kv 分开，是为了同一个注意力层可以直接用作**交叉注意力**：`attn(x, kv=别的序列)`，q 来自 x，k、v 来自另一个序列。第四部分的 Q-Former 要用。

state_dict 的键必须是：

```
n1.weight  n1.bias  n2.weight  n2.bias
attn.q.weight  attn.q.bias  attn.kv.weight  attn.kv.bias  attn.o.weight  attn.o.bias
mlp.fc1.weight  mlp.fc1.bias  mlp.fc2.weight  mlp.fc2.bias
```

**交付**：`run_vit_block`；`test_vit_block`、`test_vit_block_mask_and_rope` 通过（与用 `F.layer_norm`、`F.scaled_dot_product_attention` 搭的版本对照）。

### 3.6 问题（vit）：ViT（2 分）

配置：`image_size、patch、in_ch、d、n_layer、n_head、pos`，`pos ∈ {"learned", "sincos", "rope2d"}`。

```
图 (B, 3, H, W) → 切块 + 线性层 → 加位置信息（learned：插值后的表；sincos：按当前网格算；rope2d：每层注意力里加）
→ n_layer × ViT 块 → LayerNorm → patch_tokens (B, h·w, d)
```

`patch_tokens(x)` 返回 `(tokens, (h, w))`，**输入分辨率可以和 image_size 不同**（是 patch 的倍数即可）。`forward(x)` 对 patch 求平均，得到整张图的向量。

不加 [CLS] token：VLM 要的是每个 patch 的向量，对比学习用平均池化就够了。

**交付**：`get_vit`；`test_vit_shapes_and_resolution`（三种位置编码，两种分辨率）通过。

### 问题（vit_accounting）：算账（4 分，书面）

默认配置：64×64 的图，patch 8，d 192，4 层，4 头，mlp_ratio 4，rope2d。

1. 逐项算参数量（切块的线性层、每层的两个 LN、q、kv、o、fc1、fc2，最后的 LN）和总数。为什么 rope2d 不增加参数？
2. 一张图有多少个 patch token？经过 MLP 连接器、2×2 合并连接器、16 个查询的 Q-Former 之后，各变成多少个视觉 token？
3. Qwen 系列（patch 16、2×2 合并、`factor = 32`、`min_pixels = 256²`、`max_pixels = 4096²`）处理三张图：1000×750 的照片、A4 纸 300 dpi 的扫描件（3508×2480）、32×2000 的一行字。各缩放成多大、多少个 patch、多少个视觉 token？如果 `max_pixels` 设成 1280·28·28，A4 那张变成多少个 token？这对 OCR 意味着什么？

---

## 4. 第二部分：对比学习

### 4.1 问题（clip_loss）：CLIP（3 分）

`clip_loss(zi, zt, logit_scale)`：zi、zt `(B, D)` 已归一化，第 i 张图配第 i 句话。`logits = s · zi ztᵀ`，损失 = (按行的交叉熵 + 按列的交叉熵) / 2，正确答案都在对角线上。用你自己的 log_softmax。

### 4.2 问题（siglip_loss）：SigLIP（3 分）

每一对 (i, j) 是一个独立的二分类：`logit = s · zi_i · zt_j + b`，标签 y = +1（i = j）或 −1。

```
loss = −Σ_ij log σ(y_ij · logit_ij) / B
```

log σ(x) 要数值稳定：x 是 −1000 时不能得到 −inf 或 nan。（提示：log σ(x) = −softplus(−x)，按 x 的正负分开写。）

**交付**：`run_clip_loss`、`run_siglip_loss`；`test_clip_loss_loop`、`test_siglip_loss_loop`、`test_siglip_stable_and_grads` 通过。

### 4.3 问题（retrieval）：召回率与零样本（2 分）

1. `recall_at_k(sim, k, same=None)`：sim `(N, M)`，第 i 个查询的前 k 名里至少有一个"算对"的候选，就算命中；返回命中的比例。`same` 默认是单位矩阵。为什么需要 `same`？（ShapesWorld 里两张图的描述可能一字不差。）
2. `zero_shot(zi, class_zt)`：每张图选相似度最大的类别。

**交付**：`run_recall_at_k`、`run_zero_shot`；`test_recall_at_k`、`test_zero_shot` 通过。

### 问题（siglip_init）：初始化（3 分，书面）

1. 刚初始化时所有余弦相似度约为 0。取 s = 10，CLIP 的损失约是多少（写成 B 的函数）？B = 32、128、512 时各是多少？
2. SigLIP 取 s = 10、b = 0 时损失约是多少？取 b = −10 时呢？论文为什么把 b 初始化成 −10？（提示：一个批里正样本和负样本各有多少对？）
3. CLIP 的损失需要整个批的 softmax，SigLIP 不需要。多卡训练时这意味着什么？

---

## 5. 第三部分：框

框的约定是 `(x0, y0, x1, y1)`，x1、y1 不包含，面积 = (x1 − x0)·(y1 − y0)。

### 5.1 问题（box_iou）：IoU（2 分）

`box_iou(a (N, 4), b (M, 4)) → (N, M)`，用广播一次算完，不许写 Python 循环。

### 5.2 问题（nms）：NMS（3 分）

贪心：按分数从高到低，保留当前最高的框，删掉剩下的框里与它 IoU **大于** `iou_thr` 的，重复。返回保留的下标（long），按分数从高到低。

### 5.3 问题（ap）：AP（3 分）

`average_precision(preds, gts, iou_thr)`：`preds` 是每张图的 `(boxes (P, 4), scores (P,))`，`gts` 是每张图的 `boxes (G, 4)`（可以是 0 个）。

1. 所有图的所有预测合在一起，按分数从高到低排；
2. 每个预测找同一张图里**还没被匹配**的真实框中 IoU 最大的一个，≥ iou_thr 就是对的（并把那个真实框标成已匹配），否则是错的；
3. 算出每个位置的精确率和召回率；精确率改成"从这里往后的最大值"（插值）；
4. AP = Σ（召回率的增量 × 插值后的精确率）。

**交付**：`run_box_iou`、`run_nms`、`run_average_precision`；`test_box_iou_*`、`test_nms_bruteforce`（与逐个算的版本对照）、`test_ap_cases`、`test_ap_multi_image` 通过。

---

## 6. 第四部分：VLM

### 6.1 问题（merge_2x2）：2×2 合并（2 分）

`merge_2x2(x (B, N, d), (h, w)) → (B, N/4, 4d)`：h、w 都是偶数，x 按行排列。新 token 是一个 2×2 小方块的 4 个 patch 按**左上、右上、左下、右下**拼接；新网格 (h/2, w/2) 也按行排列。不许写循环。

### 6.2 问题（connectors）：三种连接器（4 分）

都是 `forward(x (B, N, d_v), hw) → (B, N', d)`：

| 名字 | 结构 | N' |
|---|---|---|
| `mlp`（LLaVA） | Linear(d_v, d) → GELU → Linear(d, d) | N |
| `merge`（Qwen2-VL 的 PatchMerger） | merge_2x2 → LN(4d_v) → Linear(4d_v, 4d_v) → GELU → Linear(4d_v, d) | N/4 |
| `qformer`（BLIP-2，简化版） | 16 个可学习查询 (16, d_v)；2 层，每层：自注意力 → 对 patch 的交叉注意力 → MLP（都是 Pre-LN、残差）；最后 LN → Linear(d_v, d) | 16 |

Q-Former 不给 patch 加位置信息，所以它的输出与 patch 的顺序无关、与 patch 的个数无关——测试会检查这两点。想一想：这对"定位"这种需要知道位置的任务意味着什么？（ViT 输出的 patch 向量本身带着位置信息吗？）

**交付**：`get_connector`；`test_connector_shapes`、`test_qformer_ignores_patch_order_and_count` 通过。

### 6.3 问题（chat_template）：聊天模板（3 分）

`chat_template(messages, n_img, encode, bos, eos, img) → (ids, targets)`。messages 是用户和助手轮流的发言（从用户开始），每条 `{"role", "content"}`。拼法：

```
[bos]
用户：  encode("User: ") + 内容（每个 "<image>" 换成 n_img 个 img）+ encode("\n")
助手：  encode("Assistant: ") + encode(内容) + [eos]
```

`targets[t] = ids[t+1]`，但只保留"下一个 token 是助手的内容或它后面的 eos"的位置，其余（包括用户的话、`"Assistant: "` 本身、图像占位符）都是 −100；最后一位补 −100。长度与 ids 相同。

**交付**：`run_chat_template`；`test_chat_template` 通过。

### 6.4 问题（scatter）：把视觉 token 填进去（2 分）

`scatter_visual(emb (B, T, d), visual (B, n, d), mask (B, T))`：mask 每行恰好 n 个 True，第 b 行第 k 个 True 的位置换成 `visual[b, k]`，其余不变。返回新张量（不改输入），梯度要能流回 visual。个数不对时抛出 `ValueError`。

### 6.5 问题（vlm）：VLM 与生成（2 分）

`VLM(vision, connector, lm, img_id)`：

```
embed(images, ids)：lm 的嵌入 → scatter_visual(嵌入, connector(vision.patch_tokens(images)), ids == img_id)
forward(images, ids, targets=None) = lm(inputs_embeds=embed(images, ids), targets)
```

你 A2 的 GPT 要改一处：前向接受 `inputs_embeds` 代替 `idx`。images 可以是 `(B, C, H, W)`，也可以是 `(B, k, C, H, W)`（每条 k 张图，compare 任务 k = 2，视觉 token 按图的顺序排）。

`generate(images, prompt, max_new)`：贪心。先把整个提示（含视觉 token）预填充进 KV cache，之后每步只送 1 个新 token。注意视觉 token 也占位置编号。

**交付**：`run_scatter_visual`、`get_vlm`、`run_vlm_logits`、`run_vlm_generate`；`test_scatter_visual`、`test_vlm_forward`（换图之后，图之前的 logits 不变、图之后的变）、`test_vlm_generate_matches_no_cache` 通过。

---

## 7. 第五部分：Qwen 的图像预处理

这一部分不训练，目标是**和真实模型的预处理逐位一致**。读第 24 章的 `labs/ch24/hf_tour.py` 可以看到这些函数在真模型上的输入输出，但不要复制 `miniocr.hf_vlm` 的代码。

### 7.1 问题（smart_resize）：缩放到哪里（3 分）

`smart_resize(h, w, factor, min_pixels, max_pixels)`：

1. 宽高比超过 200 时抛出 `ValueError`；
2. h、w 各自四舍五入到 factor 的倍数；
3. 面积超过 max_pixels：β = √(h·w / max_pixels)，h、w 各自除以 β 后**向下**取到 factor 的倍数（至少 factor）；
4. 面积小于 min_pixels：β = √(min_pixels / (h·w))，h、w 各自乘以 β 后**向上**取到 factor 的倍数。

注意第 3、4 步用的是**原始**的 h、w，不是第 2 步的结果。

### 7.2 问题（qwen_patchify）：切块的顺序（3 分）

`qwen_patchify(x (C, H, W), patch, merge, temporal) → (pixel_values (N, C·temporal·patch·patch), grid [[1, H/patch, W/patch]])`：

- patch 先按 merge×merge 的**窗口**分组：窗口按行排，窗口内部的 patch 也按行排（这样后面的 2×2 合并只要把相邻的 4 行拼起来）；
- 单张图在时间维上复制 temporal 份（视频的两帧共用一个 Conv3d，图片就复制成两帧）；
- 每一行按 **(C, T, p, p)** 拉平，与 Conv3d 的权重 reshape 后对应。

`qwen_preprocess(img)`：PIL 图 → smart_resize（factor = patch·merge）→ PIL 双三次缩放 → 除以 255，减 0.5 除以 0.5 → qwen_patchify。

### 7.3 问题（tokens_and_mrope）：占位符和位置编号（4 分）

1. `expand_image_tokens(ids, grids, image_id, merge)`：第 k 个 image_id 展开成 t·h·w / merge² 个；占位符比图多或少都抛出 `ValueError`。
2. `mrope_positions(ids, grids, image_id, merge) → (3, L)`，三行分别是时间、行、列：
   - 文字：三行相同，从当前编号 pos 开始逐个加 1；
   - 一张图（合并后网格 h' × w'，按行排列）：第 (ti, r, c) 个 token 的编号是 `(pos + ti, pos + r, pos + c)`；整张图结束后 `pos += max(h', w')`（Qwen2-VL 的约定，不是加 token 数）。

**交付**：`run_smart_resize`、`run_qwen_patchify`、`run_qwen_preprocess`、`run_expand_image_tokens`、`run_mrope_positions`；`test_a3_qwen.py` 全部通过，包括与 `Qwen2VLImageProcessorPil` 对照的 `test_against_transformers`（三种尺寸的随机图，pixel_values 误差 < 1e-4）。

### 问题（mrope_by_hand）：手算（3 分，书面）

序列：3 个文字 token，一张 `grid = [1, 6, 4]` 的图（merge = 2），再 2 个文字 token。

1. 展开后序列多长？
2. 写出 (3, L) 的位置编号表。
3. 图之后的文字从几开始编号？如果改成"加上图的 token 数"，长文档（很多张大图）会出现什么问题？

---

## 8. 训练脚本

写四个入口（可以共用代码），要求与 A1 第 7 节相同：可复现、日志写进 `log.jsonl`、配置写进 `config.json`、存检查点、`--smoke` 几秒跑完、从命令行覆盖任意配置、**默认 `device=cpu`**。

**数据**。字符级分词器：4 个特殊符号 `<pad> <bos> <eos> <img>`（编号 0–3）加 A2 的 96 个字符，共 100 个（不做合并）。图像 64×64，uint8 转成 [−1, 1]。

- 对比学习的数据：`sample_scene` + `draw_scene` 生成图，`describe(objs)` 作为配对的文字；
- 指令数据：`make_task(任务, objs, rng, ...)` 给出 (问题, 答案)；`text` 任务用 `draw_text_scene` 在图上写一个词，`compare` 任务要两张图。可以以一定概率在同一张图上再问一轮（多轮对话），`chat_template` 负责拼接；
- 一批里所有样本用同一个任务（长度相近、图的张数相同），任务按权重抽。

**入口**：

1. `clip`：ViT + 字符级文字编码器（小 Transformer，填充位置用 key_mask 屏蔽，对真实字符求平均），CLIP 或 SigLIP 损失，s 和 b 可学习；存 ViT 的权重。
2. `lm`：A2 的 GPT（词表 100）在 `mixed_corpus` 上预训练；存权重。
3. `vlm`：`stage=align`（冻结 ViT 和 LM，只训连接器，只用 caption 数据）或 `stage=instruct`（九个任务一起，可选冻结哪些部分）；可以从 `clip` 和 `lm` 的检查点、或者从一个 align 的检查点开始，也可以全部从零开始。定期在你自己的验证集（每个任务 50 题，种子自选）上算每个任务的准确率。
4. `predict`：对排行榜测试集生成答案，每行一个 JSON 字符串写进 `.jsonl`。

判分规则（你的验证集和排行榜相同）：`ground` 任务答案是 `"none"` 时要完全相同，否则用 `parse_box` 解析出的框与真值 IoU ≥ 0.5 算对；其余任务去掉首尾空白后完全相同。

**建议起点**（只是起点，不是答案）：ViT d 192、4 层、patch 8、rope2d；GPT d 256、4 层、block_size 320；连接器 merge；batch 32，AdamW，max_lr 3e-4，instruct 3000 步。

不需要测试，但报告里要写清楚用法，并贴一次每个入口的 `--smoke` 输出。

---

## 9. 实验

要求与 A1 第 8 节相同：先写预测，再运行；图从 `log.jsonl` 画；只跑过一次的结论要标明；需要比较大小的结论至少跑 2 个种子。每次运行先 `--smoke`。

### 问题（exp_pos）：位置编码与分辨率（4 分）

用 `clip` 入口训练三个视觉塔（learned、sincos、rope2d），其余相同。

**交付**：(a) 64×64 测试图上的图文检索 R@1、R@5（用 `same` 处理描述相同的情况）；(b) 同样的场景画成 96×96（`sample_scene` 和 `draw_scene` 都接受大小参数）再评测，三种位置编码各掉多少？运行前先预测谁最稳。(c) learned 位置表做 PCA，画出前两个主成分在网格上的样子，它学到了什么？

### 问题（exp_clip）：CLIP 对 SigLIP（5 分）

**交付**：(a) batch 32、128、512 下两种损失的 R@1（相同的训练样本总数）；(b) 训练中 s（和 b）怎么变化，画曲线；(c) 零样本分类：用 "a red …"、"a blue …" 这样的提示词给每张图的颜色（或形状）分类，和随机猜比；(d) 一段分析：小批量时哪种更好？为什么？

### 问题（exp_connector）：三种连接器（6 分）

从同一组 `clip` 和 `lm` 检查点出发，align 再 instruct，只换连接器。

**交付**：(a) 一张表：视觉 token 数、连接器参数量、每步用时、每个任务的准确率；(b) 哪个任务对连接器最敏感？运行前先预测 `ground` 和 `text` 会怎样；(c) Q-Former 换成 4 个、64 个查询各跑一次，准确率和速度怎么变？

### 问题（exp_stages）：训练流程（6 分）

固定连接器，比较：① 从零训练全部；② 预训练 ViT + LM，直接 instruct；③ 预训练，先 align 再 instruct；④ 同 ③ 但 instruct 时冻结 ViT。四者训练的**总步数相同**。

**交付**：(a) 每个任务的准确率曲线（横轴是步数）；(b) 哪个阶段对哪个任务最有用？LM 预训练帮了什么，没帮什么？(c) 只看 align 之后的模型：它能描述图片吗？能回答问题吗？为什么？

### 问题（exp_errors）：错误分析（4 分）

用你最好的模型，在你的验证集上：

**交付**：(a) `count` 的混淆矩阵（真值 × 预测）；(b) `ground` 的 IoU 分布直方图，以及 "none" 问题的准确率；(c) 挑 20 个错例，按原因分类（看错颜色、数错、框偏、编造等），各举一例并附图；(d) 多轮对话的第二轮比单轮准还是不准？

### 问题（exp_qwen）：真实模型的视觉 token 预算（3 分）

不训练，只用你第五部分的代码：

**交付**：(a) 一张图：横轴是图的长边（256 到 4096，宽高比 1:1、4:3、1:√2、1:20 各一条线），纵轴是视觉 token 数，`max_pixels` 取默认值和 1280·28·28；(b) 同一页 A4 文档按 150、200、300 dpi 扫描，token 数各是多少？用书里第 24 章的模型，30 行字的一页，大约每个字符分到多少个视觉 token？(c) 一段分析：OCR 模型为什么常常要把 max_pixels 设得比通用 VLM 大？代价是什么？

---

## 10. 排行榜

### 问题（leaderboard）：ShapesWorld 问答（6 分）

目标：`data/test.npz`（9 个任务各 60 题，种子 20261025）上**九个任务准确率的平均**尽量高。

```bash
python -m assignments.a3_vlm.leaderboard score 你的预测.jsonl
```

**规则**

1. **硬件和时间**：一张租用的 24 GB 显卡（RTX 4090、L4、A10 这一级），**所有训练阶段**（clip、lm、align、instruct）的墙钟时间合计不超过 **30 分钟**；生成数据的时间不算。CPU 赛道：120 分钟，写明 CPU 型号。
2. **数据**：只能用 `miniocr.data` 的生成器，种子不能用 20261022–20261099。生成多少不限。**不能用测试集训练、调参或选检查点**。
3. **模型**：从随机初始化开始（可以用你自己在预算内训练的 clip、lm 检查点，不能用下载的权重）；图像输入就是 64×64。
4. **代码**：第 1.1 节的限制照样适用。在这个前提下，结构、分词器、连接器、任务权重、课程式训练、混合精度、`torch.compile` 都可以自由改。
5. **租用流程**：与 A1 第 9 节相同（先在本机 `--smoke`，开机后先跑 `--smoke --set device=cuda`，结果关机前拷回，开机前写好次数和时长上限，记录费用）。

**交付**：`leaderboard.md` 加一行；报告里一段：试过什么、什么有用、为什么，附最终运行的学习曲线和每个任务的准确率。不按名次给分，按要求完成、记录完整、分析清楚即可得满分。

---

## 11. 交付清单与评分

| 部分 | 问题 | 分数 |
|---|---|---|
| 代码：视觉 | patchify 2、pos_embed 3、rope_2d 3、vit_layers 2、vit_block 3、vit 2 | 15 |
| 代码：对比学习 | clip_loss 3、siglip_loss 3、retrieval 2 | 8 |
| 代码：框 | box_iou 2、nms 3、ap 3 | 8 |
| 代码：VLM | merge_2x2 2、connectors 4、chat_template 3、scatter 2、vlm 2 | 13 |
| 代码：Qwen | smart_resize 3、qwen_patchify 3、tokens_and_mrope 4、（preprocess 含在 qwen_patchify 里） | 10 |
| 代码：训练脚本 | 四个入口能跑、`--smoke` 输出 | 2 |
| 书面 | vit_accounting 4、siglip_init 3、mrope_by_hand 3 | 10 |
| 实验 | exp_pos 4、exp_clip 5、exp_connector 6、exp_stages 6、exp_errors 4、exp_qwen 3 | 28 |
| 排行榜 | leaderboard 6 | 6 |
| **合计** | | **100** |

完成后回到第 0.11 节的里程碑 C：你有了自己从零写的 VLM，能看 64×64 的图回答九种问题；你的预处理和 Qwen 的官方实现逐位一致。下一站是第 25 章（评测），A4 在第 28 章之后。
