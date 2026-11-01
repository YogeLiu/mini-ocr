# 第 32 章　习题：解剖真实 VLM

题目里的实测数字都来自 `course-v4/outputs/book/ch32/gpu/`（下文简称 `gpu/`），记号同正文：【G-val】【G-pair】【G-flow】等。手算题默认 Qwen 的设置：patch 16、merge 2、factor 32、`min_pixels = 65536`、`max_pixels = 262144`，除非题目另说。

---

## 一、概念题

**32.1** 用一句话说出下面四个东西各自的职责，并写出它们在 Qwen3.5-0.8B 里的输入输出形状（以 val 第一张 32×298 的图为例）：processor、视觉塔、merger、语言模型。

**32.2** "Qwen 支持动态分辨率，所以图越大，视觉 token 越多。"对本章的公式图来说，这句话哪里不准确？用 `resolution.json` 里倍率 0.5、1、2 三行的结果说明。

**32.3** merger 的 `view(-1, 3072)` 不需要任何参数就完成了 2×2 合并。它能这样做的前提是什么？如果 processor 换成普通的行优先排列，模型会报错吗？会发生什么？

**32.4** 比较 LLaVA 的 MLP、Qwen 的 2×2 合并 + MLP、BLIP-2 的 Q-Former：各自改变了视觉 token 的什么（个数、宽度）？哪一种的输出个数和图片大小无关？哪一种保留了空间位置的一一对应？

**32.5** 为什么静态图片的 `image_grid_thw` 第一个数是 1，而 `temporal_patch_size` 是 2？patch 嵌入那个核为 (2, 16, 16) 的三维卷积，在两帧完全相同时等价于什么？

**32.6** Qwen 把视觉向量"填进占位符位置"，第 31 章用 `torch.cat` 把视觉向量"拼在最前面"。说出前者多出的三个好处。两者在语言模型眼里有本质区别吗？

**32.7** 同一张 32×298 的图，SmolVLM-256M 的输入有 364 个 token，Qwen3.5-0.8B 只有 111 个，而前者参数少得多。解释原因。"输入 token 少"能直接推出"更快"或"更准"吗？

**32.8** 【G-ren】里两个 Qwen 的渲染图 EM_norm 都是 0.113，【G-val】里原图只有 0.061 和 0.072。有人据此说"把图重新渲染能把准确率翻倍"。指出这个推理的问题，并说出应该看哪个文件。

**32.9** Qwen3.5 在 val 上官方 EM_norm 命中 11 条，忽略句末标点后命中 59 条（`gpu/val_format_diagnostic.json`）。为什么本章不把 59 当作正式结果？这个诊断对第 33 章有什么启示？

---

## 二、手算题

**32.10（smart_resize）** 对下面每张图，写出就近取整的结果、走哪条分支、缩放后的尺寸、patch 数和视觉 token 数：

1. 64×64
2. 333×177
3. 48×960（注意 48/32 = 1.5）
4. 1000×2000
5. 16×3000（注意 16/32 = 0.5）

**32.11（第 33 章的预算）** 第 33 章 lab 用 `min_pixels = 16384`、`max_pixels = 65536`。val 第一张图 32×298 在这个预算下缩放成多大？有多少个视觉 token？和本章的 75 相比，少了多少比例？

**32.12（取整溢出）** 37×239 的图，取 `min_pixels = max_pixels = 65536`。算出缩放后的面积，说明它为什么超过了 `max_pixels`。

**32.13（patch 排列）** 一张 2×4 个 patch 的图，每个 patch 只有 1 个像素（p = 1），像素值是行优先编号：

```text
0 1 2 3
4 5 6 7
```

merge = 2。写出 processor 排出来的顺序，以及 merger 拿到的两组。如果误用行优先排列，merger 拿到的两组又是什么？

**32.14（形状链）** 一张缩放后 256×256 的图，写出 Qwen3.5-0.8B 里每一站的形状：`image_grid_thw`、`pixel_values`、patch 嵌入输出、merger 输出。再写出 Qwen3.5 和 Qwen3-VL 的输入 token 总数（提示词与本章相同）。

**32.15（参数量）**

1. 不看正文，用视觉配置（宽 768、MLP 中间层 3072、12 层、patch 16、时间 patch 2、位置表 2304 行）算出 Qwen3.5 视觉塔 12 个 block 加 patch 嵌入加位置表的参数量（每个 block 有 qkv、输出投影、两层 MLP、两个带偏置的 LayerNorm，线性层都带偏置）。
2. 假设有一个语言模型宽 2048 的变体，视觉部分不变。merger（LayerNorm 在 768 维上，fc1 是 3072→3072，fc2 是 3072→2048，都带偏置）有多少参数？

**32.16（M-RoPE）** 接 32.14，一张缩放后 256×256 的图套 Qwen3.5 的模板。写出：图片 token 在序列里的下标范围；第一个和最后一个图片 token 的三个位置坐标；图片之后第一个 token（`<|vision_end|>`）的位置；整个序列的最大位置编号。

**32.17（SmolVLM 的 token）** SmolVLM 每个 512×512 的图块切成 16×16 的 patch，像素重排的 `scale_factor = 4`。

1. 每个图块进语言模型的视觉 token 数是多少？每个 token 的宽度（重排后、投影前）是多少？
2. 【G-val】里 SmolVLM 的视觉 token 最多 832 个。这对应几个图块（含全图）？

**32.18（区间）** 【G-val】里 Qwen3.5 是 11/180，区间 [0.034, 0.106]；Qwen3-VL 是 13/180，区间 [0.043, 0.120]。能说 Qwen3-VL 显著更好吗？按事先的规则选它有没有问题？

---

## 三、编程题

练习 32.1–32.4 的位置已经在 `miniocr/models/vlm_anatomy.py` 里用 `# >>> 练习 32.x` 标出。做的时候先把函数体删掉，自己写，再跑测试。所有命令在 `course-v4/` 下运行：

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q book/part4_vlm/ch32_real_vlm
```

**练习 32.1　smart_resize（必做）**

实现 `smart_resize(height, width, factor=32, min_pixels=65536, max_pixels=262144)`，规则见正文 32.2.3 节。要求：长宽比超过 200 报错；参数非正或 `max_pixels < min_pixels` 报错；用 Python 的 `round`（五成双）。

验证：`test_resize_rejects_invalid`（4 组非法参数）、`test_resize_rounding_and_budget`；再用手算题 32.10 的五张图自查。

**练习 32.2　replica_processor（必做）**

实现静态 RGB 图的 processor：`smart_resize` → PIL 双三次缩放 → float64 乘 1/255 再转 float32 → 减均值除标准差 → merge 邻域优先排列 → 复制成两帧 → 返回 `pixel_values`、`image_grid_thw`、`resized_hw`。PIL 的插值可以直接用，尺寸规则、归一化和排列要自己写。

验证：`test_processor_elementwise`（5 种尺寸，和 transformers 的 `Qwen2VLImageProcessorPil` 逐位相等）、`test_patch_order_independent_anchor`（4×4 编号墙）。再跑 tour 部分 1（lab 第 1 节），确认输出里 `unequal_elements=0`。

**练习 32.3　两种连接器（必做）**

实现 `MLPConnector`（两层 MLP，输出个数等于输入个数）和 `QueryConnector`（8 个可学习 query、一次交叉注意力、残差 + LayerNorm、FFN，输出个数固定为 8）。

验证：`test_connectors_use_image_and_backprop`（形状、梯度有限非零、输入换成全零时输出会变）；跑 tour 部分 3（lab 第 3 节），参数量应为 3,936 和 20,832。

**练习 32.4　占位符展开（必做）**

实现 `expand_image_placeholder(prompt, image_token, grid, merge_size=2)`：恰好一个占位符，否则报错；patch 数必须能被 merge² 整除；把占位符换成 prod(grid)/merge² 个。

验证：`test_placeholder_uses_merged_count`。本地有 Qwen3.5 缓存时，tour 部分 1 还会把你展开的字符串交给官方 tokenizer，和官方 processor 的 `input_ids`、`attention_mask` 比较（`input_ids_equal=true`），测试 `test_cached_real_processor` 不再跳过。

**练习 32.5　只读文件头数参数（选做）**

safetensors 文件的前 8 个字节是一个小端 64 位整数 n，接着 n 个字节是 JSON 头，里面有每个张量的名字和形状。写一个脚本，只读文件头（不加载权重），按名字前缀（`model.visual.patch_embed`、`model.visual.blocks`、`model.visual.merger`、`model.language_model.embed_tokens`、`model.language_model.layers`、`mtp` 等）分组求和。

验证：Qwen3.5-0.8B 去掉 `mtp.*` 以后应为 852,985,920，SmolVLM-256M 应为 256,484,928，和 `gpu/remote_logs/ch32_all_remote_logs.log` 里的 `parameters=` 一致；各组数字和正文 32.4.3 节的表一致。

**练习 32.6　输出格式诊断（选做）**

用 `miniocr.eval.postcorrect_eval` 里的 `em_ignoring_final_punct`、`normalize_latex`，对 `gpu/remote_runs/main/` 下三个模型的 val JSON 做诊断：

1. 统计"官方 EM_norm 错、忽略句末标点后对"的条数。
2. 在这些条目里，统计 raw 中 `$$…$$` 内部规范化后和目标完全相同的条数。
3. 统计目标规范化后以 `.` `,` `;` 结尾的条数。

验证：Qwen3.5 和 SmolVLM 的"忽略句末标点后命中"应和 `gpu/val_format_diagnostic.json` 一致：文件里忽略句末标点后的总命中是 59、2，减去原本就对的 11、1，答案是 48、1（本项目观察，按 `observations` 逐条 CPU 复算）；第 2、3 问的 Qwen3.5 结果应和正文 32.9.4 节一致（46、123）。Qwen3-VL 的结果文件里没有，自己算出来写进你的 R3 草稿。不要对 test 做同样的事（test 本来也没有保存生成文本）。

**练习 32.7　在自己的机器上跑一个小模型（选做）**

按 lab 第 0 节准备数据，在 CPU（或 MPS）上跑 `tour 4 --models smol --n 20`，把输入 token、视觉 token、EM_norm、CER、截断数和耗时填进 lab 的记录表。

验证：数据准备步骤和 lab 相同时，前 20 条的视觉 token 应和【G-val】里 SmolVLM 前 20 条 `observations` 的 `visual_tokens` 逐条相同（processor 是确定的）；生成文本在 fp32 下可能和 GPU 的 bf16 不同，记录差异，不要求一致。

**练习 32.8　自己写 M-RoPE 位置（选做）**

写 `mrope_positions(mm_token_type_ids, grid_thw, merge=2)`，按正文 32.7.3 节的规则返回 [3, L] 的位置。用手算题 32.16 和正文 32.7.4 节的两个例子自查（最大位置分别是 43 和 60）。有 Qwen3.5 权重时，可以和 `model.model.get_rope_index(...)` 的返回值逐位比较。

---

## 四、调试题

**32.19** 某同学的 `replica_processor` 和官方对拍时 `pixel_values` 形状相同，`unequal_elements` 却很大；而 `smart_resize` 的测试全部通过。他的排列代码是：

```python
blocks = pixels.reshape(3, gh, p, gw, p).transpose(1, 3, 0, 2, 4)   # (gh, gw, 3, p, p)
blocks = np.repeat(blocks[:, :, :, None], 2, axis=3)                # (gh, gw, 3, 2, p, p)
pixel_values = blocks.reshape(gh*gw, 3*2*p*p)
```

指出问题。如果不和官方对拍，只看模型输出，他会看到什么？

**32.20** 运行模型时报错：

```text
Image features and image tokens do not match, tokens: 300, features: 75
```

最可能是哪个函数写错了？错在哪一行？

**32.21** 用 Qwen3.5 跑零样本，`truncated` 从 8 条变成了 150 多条，原始输出都以一大段英文推理开头。代码哪里最可能被改动了？

**32.22** 某同学想"省掉视觉塔重复计算"，自己写了前向：`x = model.get_input_embeddings()(input_ids)`，然后直接调用 `model.model.language_model(inputs_embeds=x, ...)` 再接 `lm_head`。程序不报错，但换任何图片输出都一样。原因是什么？怎样用一个实验确认？

**32.23** 某同学报告：

> 原图 val 180 条 EM_norm 0.061，渲染图 val 80 条 EM_norm 0.113，渲染让 Qwen3.5 提高了 5.2 个百分点。

至少指出两处问题，并用【G-pair】的数字重写这句话。

**32.24** 某同学的 `replica_processor` 和官方的 `max_abs_error` 是 1.19e-07，不是 0。最可能差在哪一步？

---

## 五、闭卷手写

**32.25** 不看任何资料，写出 `smart_resize` 的完整代码（含两个报错检查），并用 32×298 验证结果是 (96, 800)。

**32.26** 写出把 `(3, H', W')` 的归一化图排成 `[gh·gw, 3·2·16·16]` 的三行 numpy 代码（reshape、transpose、复制两帧、reshape）。

**32.27** 写出视觉向量填入语言模型输入的四步（查表、找位置、检查、填入），用 PyTorch 写出核心两行。

**32.28** 写出 Qwen3.5-0.8B 参数总数的分解式：视觉塔（patch 嵌入、位置表、12 个 block、merger）+ 语言模型（词嵌入、24 层、norm），每项给出数值，合计 852,985,920。

---

## 答案与提示

**32.1** processor：缩放、切块、排列图片，套模板并展开占位符；32×298 图 → `pixel_values` [300, 1536]、`input_ids` [1, 111]。视觉塔：把 patch 变成向量；[300, 1536] → [300, 768]。merger：2×2 合并并投到语言模型宽度；[300, 768] → [75, 1024]。语言模型：在填好视觉向量的输入上做因果生成；`inputs_embeds` [1, 111, 1024] → logits [1, 111, 248320]。

**32.2** 小公式图面积不到 `min_pixels`，全部走放大分支，被放大到同一个尺寸。倍率 0.5、1、2 的输入是 16×149、32×298、64×596，缩放后都是 96×800、75 个 token。对小图而言，token 数由下限决定，不随原图大小变化。

**32.3** 前提是 processor 已经按"2×2 邻域优先"排列，每个方阵的 4 个 patch 在 `pixel_values` 里是连续 4 行（视觉 block 不改变顺序）。换成行优先不会报错，形状完全一样；merger 会把同一行相邻的 4 个 patch（1×4 横条）拼成一个 token，空间结构被打乱，模型输出变差但不是乱码。

**32.4** MLP：只改宽度，个数不变，保留一一对应。2×2 合并 + MLP：个数变成四分之一、宽度变成语言模型宽度，保留"一个 token 对应一个 2×2 方阵"的局部对应。Q-Former：个数固定为 query 数（BLIP-2 是 32），和图片大小无关，不保留一一对应。

**32.5** 视觉塔为视频设计，时间上一个 patch 跨 2 帧。静态图复制成两帧，2/2 = 1，所以 `grid_t = 1`。两帧相同时，三维卷积等于一个二维卷积，核是两个时间切片的权重之和。

**32.6** 图片可以放在序列任意位置；一条序列可以有多张图；占位符数和特征数不一致时有明确报错。在语言模型眼里没有本质区别：视觉向量都只是一段输入嵌入，后面的文字通过因果注意力看到它们。

**32.7** SmolVLM 把扁长的图切成 4 个 512×512 的块再加一张全图，每块 64 个 token，共 320；Qwen 把整图放大到约 76800 像素，按 1024 像素一个 token 只有 75 个。参数量和输入长度无关。输入 token 少不能直接推出更快或更准：速度还取决于模型大小、注意力实现和生成长度（SmolVLM 每秒生成的 token 反而最多），准确率取决于模型能力。

**32.8** 两组不是同一批样本：渲染图那 80 条是能被 mathtext 渲染的子集，本身偏简单。应该看 `gpu/remote_runs/paired.json`：同一批 80 条上，Qwen3.5 从 7/80 到 9/80，Qwen3-VL 都是 9/80，看不出"翻倍"。

**32.9** 选择规则和指标是事先定的；看过结果再改口径，就是用结果调评分，会让选择和报告都失去可信度。诊断数字单独列出，作为已知的指标盲区写进报告。启示：零样本失分里有一大块是格式（句末标点、写法风格），LoRA 最容易学会格式，所以第 33 章要把"读得更准"和"写得更像标注"分开看；如果要换提取规则，要在看第 33 章结果之前决定。

**32.10**

| 图 | 就近取整 | 分支 | 缩放后 | patch | 视觉 token |
|---|---|---|---|---|---|
| 64×64 | 64×64，面积 4096 | 放大，s = 4 | 256×256 | 256 | 64 |
| 333×177 | 320×192，面积 61440 | 放大，s ≈ 1.0545 | 352×192 | 264 | 66 |
| 48×960 | round(1.5) = 2 → 64×960，面积 61440 | 放大，s ≈ 1.1926 | 64×1152 | 288 | 72 |
| 1000×2000 | round(31.25) = 31、round(62.5) = 62 → 992×1984 | 缩小，s ≈ 2.7621 | 352×704 | 968 | 242 |
| 16×3000 | round(0.5) = 0 → 0×3008，面积 0 | 放大，s ≈ 1.1685 | 32×3520 | 440 | 110 |

（数学结论。前三张和 `gpu/../cpu_smoke/processor.json`（即 `outputs/book/ch32/cpu_smoke/processor.json`） 里同尺寸随机图的 `resized_hw`、`merged_tokens` 一致。）第 5 张：就近取整把高变成 0，是放大分支的 `ceil` 把它救回了 32。第 4 张：62.5 五成双取 62。

**32.11** 就近取整 32×288 = 9216 < 16384，放大；s = √(16384/9536) ≈ 1.3108；H' = ⌈1.31⌉·32 = 64，W' = ⌈298·1.3108/32⌉·32 = ⌈12.21⌉·32 = 416；面积 26624，视觉 token 26。比 75 少约 65%（数学结论）。

**32.12** 就近取整 32×224 = 7168，放大；s = √(65536/8843) ≈ 2.7223；H' = ⌈37·2.7223/32⌉·32 = ⌈3.15⌉·32 = 128，W' = ⌈239·2.7223/32⌉·32 = ⌈20.33⌉·32 = 672；面积 86016 > 65536。两条边各自向上取整，面积必然不小于下限，但没有任何机制把它压回上限以下。

**32.13** 只有一个方阵行，两个方阵：{0, 1, 4, 5}、{2, 3, 6, 7}，顺序是 `0 1 4 5 2 3 6 7`。行优先时 merger 拿到 {0, 1, 2, 3} 和 {4, 5, 6, 7}，是两条横带。

**32.14** `image_grid_thw` = [[1, 16, 16]]；`pixel_values` [256, 1536]；patch 嵌入 [256, 768]；merger [64, 1024]。Qwen3.5 输入 36 + 64 = 100 个 token，Qwen3-VL 32 + 64 = 96 个（少了空思考块的 4 个）。

**32.15**

1. patch 嵌入 768·(3·2·16·16) + 768 = 1,180,416；位置表 2304·768 = 1,769,472；每个 block：qkv 768·2304 + 2304 = 1,771,776，输出投影 768·768 + 768 = 590,592，MLP 768·3072 + 3072 + 3072·768 + 768 = 4,722,432，LayerNorm 4·768 = 3,072，合计 7,087,872；12 个 block 85,054,464。三项合计 88,004,352（数学结论；加上 merger 12,588,544 就是正文的视觉塔 100,592,896）。
2. LayerNorm 1,536；fc1 3072·3072 + 3072 = 9,440,256；fc2 3072·2048 + 2048 = 6,293,504；合计 15,735,296（数学结论，假设的配置）。

**32.16** 合并后网格 8×8。图片 token 下标 4–67（共 64 个）。第一个：(4, 4, 4)；最后一个（第 7 行第 7 列）：(4, 11, 11)。图片之后起点 4 + max(8, 8) = 12，所以 `<|vision_end|>` 的位置是 12。后面共 32 个 token（下标 68–99），位置 12–43，最大位置 43（数学结论）。

**32.17** 1. 512/16 = 32，每块 32×32 = 1024 个 patch；重排把每 4×4 = 16 个并成一个，得 64 个 token；宽度 16·768 = 12288，再线性投影到 576。2. 832/64 = 13 块，即 12 个图块加 1 张全图。图块怎样排成行列以 `Idefics3ImageProcessor` 源码为准。

**32.18** 不能。两个区间大幅重叠，11 和 13 只差 2 条，180 条样本区分不了。按事先的规则选它没有问题，规则的意义就在于不用事后争论；但报告里要写清"按规则胜出，差异不显著"，后续实验也不宜只押一个模型（第 33 章 lab 对两个模型都做了实验）。

**练习 32.1–32.4** 参考实现就是 `vlm_anatomy.py` 里对应的函数，做完后可以对照。常见卡点：32.1 的判断条件用取整后的面积、缩放比例用原图面积；32.2 的 float64 → float32 顺序和 PIL `resize((w, h))`；32.3 别忘了 `batch_first=True`；32.4 除以 `merge_size**2`。

**练习 32.5** 读头的核心：

```python
import json, struct, math
with open(path, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
header.pop("__metadata__", None)
total = sum(math.prod(v["shape"]) for v in header.values())
```

Qwen3.5 文件里合计 873,438,784，`mtp.*` 合计 20,452,864，相减得 852,985,920。Qwen3.5 的权重文件里没有 `lm_head.weight`（共享词嵌入）；SmolVLM 有，且和词嵌入同形状 [49280, 576]。

**练习 32.6** Qwen3.5：只差句末标点 48 条，其中 46 条的 `$$…$$` 内部和目标规范化后相同；目标以标点结尾 123 条（与正文一致）。提取 `$$…$$` 内部可以用 `re.search(r"\$\$(.*?)\$\$", raw, re.S)`，没有匹配就用整个 raw。比较用 `normalized_tokens`。

**练习 32.8** 提示：按 `mm_token_type_ids` 把序列分成连续的"文字段"和"图片段"；文字段的三个坐标都是 `arange(len) + 当前位置`；图片段用 `meshgrid` 生成 (t, h, w)，三者都加上起点；图片段之后当前位置加 max(合并后行数, 列数)。

**32.19** 他用的是普通行优先：`(3, gh, p, gw, p) → (gh, gw, 3, p, p)`，复制两帧后形状和官方一样，但没有拆出 merge 维，patch 的顺序是逐行的。正确做法是拆成 `(3, gh/m, m, p, gw/m, m, p)` 再 `transpose(1, 4, 2, 5, 0, 3, 6)`。只看模型输出时，不报错，merger 把横条当方块拼，输出变差但看上去"还像回事"，很难发现。

**32.20** `expand_image_placeholder`（或等价的展开逻辑）：`count // merge_size**2` 写成了 `count`，把 300 个 patch 当成了 300 个视觉 token。

**32.21** `apply_chat_template` 调用里的 `enable_thinking=False` 被删了（或者写错了参数名）。打印 `prompt` 结尾，看有没有空的 `<think>\n\n</think>\n\n`。

**32.22** 他跳过了外层 `Qwen3_5Model.forward` 里"算视觉特征 → `get_placeholder_mask` → `masked_scatter`"这一段。75 个占位位置里放的都是 `<|image_pad|>` 这一个 token 的嵌入，和图片无关，语言模型根本没见到视觉向量；位置编号也退回了默认值，不是 M-RoPE。确认方法是视觉依赖对照：固定文字，换两张差别很大的图（或一张白图），比较 logits；完全相同就说明视觉向量没有进去。正常用法是把 `input_ids`、`pixel_values`、`image_grid_thw` 一起交给完整模型。

**32.23** 问题一：两个数来自不同队列（180 条全体 vs 80 条可渲染子集），不能相减。问题二：80 条子集本身偏简单，差异混入了样本难度。问题三：80 条的区间很宽。重写："在同一批 80 条可渲染公式上，Qwen3.5 的 EM_norm 从原图的 7/80 变为渲染图的 9/80，CER 从 0.128 变为 0.139（本项目观察，RTX 4090（租用），bf16，`paired.json`）；样本量小，看不出可靠差异。"

**32.24** 归一化前的 rescale：在 float32 里乘 `1/255`。官方 PIL 后端先在 float64 里乘 `1/255` 再转 float32。rescale 这一步最大差 5.96e-08 ≈ 2⁻²⁴，正是 float32 在 [0.5, 1) 里相邻两个数的间距：差了最后一位；再除以标准差 0.5，放大成 1.19e-07。注意 float32 里直接除以 255 和官方逐位相同，不是元凶（CPU 穷举 0–255 复算：乘法 126 个值不同，除法 0 个）。

**32.25–32.28** 对照正文 32.2.5、32.3.5、32.7.3 与 32.7.5、32.4.3 节。32.27 的核心两行：

```python
mask = (input_ids == image_token_id).unsqueeze(-1)
inputs_embeds = inputs_embeds.masked_scatter(mask, image_embeds)
```

写之前要先检查 `mask.sum() * hidden == image_embeds.numel()`。
