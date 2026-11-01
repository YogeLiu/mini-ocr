# 第 32 章实验手册：解剖真实 VLM

## 0. 准备

代码：[tour](tour.py)、[CPU 测试](test_ch32_anatomy.py)、[processor/连接器](../../../miniocr/models/vlm_anatomy.py)、[开源图像抽取](../../../miniocr/data/latex_images_ch32.py)。下文 `gpu/` 统一指 `course-v4/outputs/book/ch32/gpu/`，实测结果见[本章正文](chapter.md) §32.8、§32.9。

在 `course-v4/` 运行。本章不训练；默认只运行轻量机制部分 1、2、3、6。位置参数是部分号，按给定顺序执行；每部分重置 seed。真实推理默认 `--n 20`，设备默认 CPU、精度默认 fp32，模型仅从本地缓存加载。

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q book/part4_vlm/ch32_real_vlm
CUDA_VISIBLE_DEVICES="" python -m book.part4_vlm.ch32_real_vlm.tour
CUDA_VISIBLE_DEVICES="" python -m book.part4_vlm.ch32_real_vlm.tour 1 2 3 --out outputs/book/ch32/local
# 读取现有开源 parquet；不是下载命令。先全语料划分，再取每个 split 前缀。
python -m book.part4_vlm.ch32_real_vlm.tour --prepare-data --n 200 --data /tmp/ch32/data
# 需已有模型权重；学习者可先只选小模型。
python -m book.part4_vlm.ch32_real_vlm.tour 4 --models smol --n 20 --data /tmp/ch32/data --out outputs/book/ch32/local_inference
```

CLI 支持 `--device cpu|mps|cuda`、`--dtype fp32|bf16`、`--n`、`--seed`、`--model-cache`、`--data`、`--out`、`--models qwen35 smol qwen3vl`、`--min-pixels`、`--max-pixels`、`--max-new-tokens`。图像预算默认 min=65536、max=262144；这是**实验设置**，不是模型原生最大分辨率。Qwen 使用此面积预算，Smol 保持其原生切图配置；比较的是完整部署配置，不是等视觉 token 预算实验。batch 固定为 1，以免不同长度 padding 干扰逐样本计时。

GPU 推荐命令（远程现有 Python）：先设置 `$CODE_DIR` 为代码根目录、`$RUN_DIR` 为本章运行产物目录、`$LOG_DIR` 为日志目录、`$HF_CACHE` 为已有 Hugging Face 缓存目录；后三者放在代码目录外。

```bash
nvidia-smi
cd "$CODE_DIR"
mkdir -p "$RUN_DIR" "$LOG_DIR"
HF_HUB_CACHE="$HF_CACHE" MPLCONFIGDIR="$RUN_DIR"/mpl PYTHONDONTWRITEBYTECODE=1 \
  python -m book.part4_vlm.ch32_real_vlm.tour --prepare-data --n 200 --data "$RUN_DIR"/data
export HF_HOME="$RUN_DIR"/hf
export MPLCONFIGDIR="$RUN_DIR"/mpl
export PYTHONDONTWRITEBYTECODE=1
# 数据已由准备步骤抽取在该目录。用已存在的原始缓存抽取时，不要将 HF_HOME 指向空缓存。
nohup python -u -m book.part4_vlm.ch32_real_vlm.tour 1 2 3 5 6 4 \
  --device cuda --dtype bf16 --n 180 --data "$RUN_DIR"/data \
  --out "$RUN_DIR"/main > "$LOG_DIR"/ch32_main.log 2>&1 &
# 上一任务结束后再跑同一 val 队列中前 80 个可渲染目标。
nohup python -u -m book.part4_vlm.ch32_real_vlm.tour 4 \
  --device cuda --dtype bf16 --n 80 --source rendered --data "$RUN_DIR"/data \
  --out "$RUN_DIR"/rendered > "$LOG_DIR"/ch32_rendered.log 2>&1 &
```

MPS 可将小规模命令改成 `--device mps --dtype fp32 --n 20`；5080 可用上述 CUDA 参数，从小规模检查显存开始。这里不填这些设备的性能数字。所有远程写入均在 `$RUN_DIR/` 或 `$LOG_DIR/ch32_*.log`；原始数据缓存只读，模型分开串行加载；加载前要求空闲显存至少 Smol 2 GiB、Qwen3.5 3 GiB、Qwen3-VL 6 GiB（调度阈值，非实测峰值），每 15 秒检查，等待超过 20 分钟则明确退出。同步脚本带 `--delete`，因此不能把运行产物放进代码目录。

**环境实测：** Python 3.11、torch 2.14.1+cu130、transformers 5.19.0；未安装任何新包。静态图使用 `Qwen2VLImageProcessorPil` / `Idefics3ImageProcessorPil`。transformers 5.19 的 Qwen AutoProcessor 会同时构造视频组件，视频没有 PIL 后端；`load_image_only_processor` 缩窄构造签名，只加载图像和 tokenizer，模板和视觉 token 展开仍复用官方逻辑。不支持视频。Qwen3.5 缺少 causal-conv1d / flash-linear-attention，使用正确但较慢的 PyTorch 回退；速度结论仅限这套环境。远程并行任务可能竞争 GPU，不能把这些延迟当作独占机器基准。

**模型来源：** 下载 [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)、[SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)、[Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)，下载时设置 `HF_HUB_DISABLE_XET=1 HF_ENDPOINT=https://hf-mirror.com`。完整 revision 见结果 JSON 的 snapshot，不依赖以后变化的 main。独立下载日志已于 2026-10-30 拉取至 `outputs/remote_4090/logs/ch32_download.log`（相对 course-v4/），其中三模型 snapshot revision 分别为 `2fc06364715b967f1860aea9cf38778875588b17`、`7e3e67edbbed1bf9888184d9df282b700a323964`、`89644892e4d85e24eaac8bacfd4f463576704203`，与结果 JSON 一致；日志未打印上述环境变量。`outputs/remote_4090/logs/ch32_main.log` 的 RecursionError 是 SmolVLM val180 生成后评分时主进程崩溃，后续 `ch32_recovery.log` 补完 SmolVLM/Qwen3-VL，`ch32_final.log` 正常完成，详见已有的[恢复日志](../../../outputs/remote_4090/logs/ch32_recovery.log)和[最终评测日志](../../../outputs/remote_4090/logs/ch32_final.log)；旧运行合并日志 `gpu/remote_logs/ch32_all_remote_logs.log` 保留。

```bash
HF_HUB_DISABLE_XET=1 HF_ENDPOINT=https://hf-mirror.com HF_HOME="$RUN_DIR"/hf python -c 'from huggingface_hub import snapshot_download; [snapshot_download(m, allow_patterns=["*.json","*.safetensors","*.txt","*.model","*.jinja","README.md"], max_workers=4) for m in ["Qwen/Qwen3.5-0.8B","HuggingFaceTB/SmolVLM-256M-Instruct","Qwen/Qwen3-VL-2B-Instruct"]]'
```

**数据口径：** [OleehyO/latex-formulas](https://huggingface.co/datasets/OleehyO/latex-formulas)，cleaned_formulas，数据卡许可 openrail。复用现有六个 parquet（本次读取 metadata 实测共 552340 行），不下载额外数据，不分发原图。按 `latex_corpus.standard_split` 的完整 latex-v1、seed=0、val/test 各约 5000 条分组规则划分；再取前 200 条并按清洗后的目标去重，各取原始 parquet 中第一张图，得到 val 189、test 188 条。主 val 使用前 180 个不同目标。不是随机代表性样本，也没有从易样本中挑主评测。规范化键相同的公式仍只能属于同一划分。

原始 val 队列中有 103 条能由现有 mathtext 渲染器处理，配对对照取前 80 条（都落在主 val 前 180 条中）。渲染是本章控制图像来源的实验：相同目标、同一 renderer、dpi=160、白底黑字、四周 12 像素留白；不能渲染的条目不替换目标，失败原因在远程 manifest。mathtext 不支持全部 LaTeX，也没有完整中文字体，不能把这个子集代表整个公式分布。

**输出口径：** 日志为 `key=value`，JSON 保留快照、模板、精度、seed、逐样本输入/视觉/输出 token、截断状态、计时、显存和 val 原始输出。val 每 10 条另存 `predictions/` 检查点，先保存后评分；test 不保存生成文本。视觉 token 指真正插入语言序列的 image placeholder 数，不包含视觉边界、crop 坐标或普通文字。输入 token 则包含全部模板和图像 token。先做 2-token warmup，再计时；每次 CUDA 同步，生成时间包括 prefill 和 decode，另记 CPU processor 时间，不包含模型加载和评分。峰值显存使用本进程 PyTorch allocated/reserved，非整卡 nvidia-smi；不包含其他进程；峰值计数在 warmup 后重置，reserved 可能包含 warmup 留下的缓存。

**评测口径：** 使用第 0 章 §0.9.3 的 OCR 面板精简版；本章未报告 CER_token 和渲染一致率。复用第 18 章 `extract_formula` 和 `score_postcorrect`，不改接口。先按固定规则提取输出公式再计算原串 EM、规范化 EM、语料级 CER（插入过多时可超过 1）；提取会移除外壳和某些尾部标点。EM 给 Wilson 95% 区间，CER 给固定 seed 的 1000 次样本 bootstrap。parse_rate 只是 mathtext 可解析率，不是真 LaTeX 编译率。此次 SmolVLM 输出暴露了 mathtext 深嵌套时的 RecursionError；`render.can_parse` 现在把它与 ValueError 一样记为不可解析，并有 80 层嵌套的回归测试，不更改既有接口或 EM/CER 口径。固定 greedy、max_new_tokens=256、Qwen3.5 禁用 thinking；达到上限又没 EOS 的输出仍进入评分并单列截断数。不使用后纠错专用的 fix/broken 指标。

所有选择只看 val，预先约定按原图 val EM_norm 最高选模型，同分取 CER 更低者。冻结后只对胜出模型评 test 一次；`--split test --final-test --models 单模型` 必须搭配 `--out`，先写独占 lock，test 不落盘答案或生成文本；开发期不看 test 错例，最终报告定稿后才可解封（第 0 章 §0.10.1）。后续章节重新训练属于新的报告协议，不能拿本章 test 调参。

## 1. Processor 逐位对照

运行 `tour 1`。CPU 无权重时使用四种尺寸的固定随机 RGB 图片；有 `--data` 时使用开源 val 原图，有 Qwen3.5 缓存时同时对比实际 tokenizer 输入。

练习入口在 `miniocr/models/vlm_anatomy.py` 的 `练习 32.1/32.2/32.4`：smart_resize → PIL bicubic → float64 rescale 再转 float32 → normalize → merge 邻域优先排列 → 静态时间复制 → `image_grid_thw` → 占位符展开。底层 PIL 插值和官方词表可复用，尺寸规则、布局和占位符计数应自己写。

比较 `unequal_elements`、`max_abs_error`、grid、input_ids 和 attention_mask。这里的“逐位”指 float32 元素严格相等，不只是 allclose；不是跨后端、跨 Pillow 版本的位级保证。RGB 静态图路径之外不作通过声明。

## 2. 动态分辨率与 token 合并

运行 `tour 2`，输出 `resolution.json/png`。对同一图的尺寸按四个倍率和三个面积预算计算 grid。数学结论：patch=16、merge=2 时，静态图片的视觉 token 为 `H'W'/1024`；先进入视觉编码器的 patch 数是它的四倍。MLP 改宽度并不自动缩短序列。

观察取整平台、面积上限分支和最小面积分支。最小面积分支使用 ceil，可能超过名义 max_pixels，尤其当 min=max；不要错误地把 smart_resize 理解成严格面积裁剪器。这个部分只做尺寸机制分析，不声称改变预算一定提高识别率。

## 3. MLP 与 query 连接器

运行 `tour 3`。`练习 32.3` 提供缩小结构：两层 LLaVA 风格 MLP 保持 token 数；可学习 query 的 cross-attention 连接器输出固定数量 token。每个结构检查一次反传，确认输入梯度有限且非零。训练步数为 0；随机结构探针不代表模型质量，不是完整 BLIP-2 Q-Former 或其预训练权重。

结构出处：[LLaVA 代码](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/multimodal_projector/builder.py)、[BLIP-2 论文](https://arxiv.org/abs/2301.12597)。读取它们时标出 query self-attention、与视觉的 cross-attention、FFN 和到 LLM 的投影，明确本实验省略了哪些堆叠与训练目标。

## 4. 同批公式的真实模型比较

运行 `tour 4`。主评测 `--source original`；渲染对照 `--source rendered`；若需要重新生成配对原图结果，可运行相同 n 的 `--source paired-original`。本次直接按 ID 从主评测取出对应原图预测重评，避免重复推理。

```bash
python -m book.part4_vlm.ch32_real_vlm.summarize_runs "$RUN_DIR"
```

主表 original n=180 与 rendered n=80 是不同队列，不应直接相减；`paired.json` 才是相同 80 条目标的原图/渲染图对比。保存逐条 raw 和 pred，先检查是否返回解释、漏掉括号、复制幻觉或达到长度上限，再解释 aggregate 指标。

## 5. 真实前向数据流

运行 `tour 5 --models qwen35`（本部分固定 Qwen3.5）。`dataflow.json` 记录 patch embedding、第一个视觉 block、merger、完整视觉输出、语言模型 inputs_embeds 和 logits。forward hook 只记录 shape，不留激活副本；用实际 checkpoint config 验证每个宽度。

本次第一张开源 val 图观察：processor `[300,1536]`，patch embedding/视觉 block `[300,768]`，merger `[75,1024]`，语言输入 `[1,111,1024]`，logits `[1,111,248320]`。均为本项目观察，RTX 4090（租用），模型 bf16，processor fp32，n=1，训练步数=0；旧独立日志 `gpu/logs/ch32_main.log` 本地不存在，现存合并日志为 `gpu/remote_logs/ch32_all_remote_logs.log`，详细配置在 `gpu/main/dataflow.json`。

## 6. 只读 OCR 模块说明

运行 `tour 6` 保存阅读索引。只访问说明，不下载 GLM-OCR 或 PaddleOCR-VL。

| 模型 | 视觉与连接 | 语言侧/任务 | 与本次 Qwen3.5 对照时检查什么 |
|---|---|---|---|
| Qwen3.5 | 动态网格视觉 Transformer、空间合并器 | 混合注意力语言解码器、通用多模态 | 先核对实测 grid 和 merger shape |
| GLM-OCR | CogViT、轻量跨模态连接器及 token 下采样 | GLM 解码器、专门的文档 OCR | 图像前处理、下采样位置、OCR 指令格式 |
| PaddleOCR-VL | NaViT 风格动态分辨率视觉编码器 | ERNIE-4.5 语言模型、文档元素识别 | 动态分辨率并不等于采用相同 patch/merge 布局 |

后两行来自官方模型卡：[GLM-OCR](https://huggingface.co/zai-org/GLM-OCR)、[PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)。这些是文档说明，未经本项目推理验证，不引用它们的宣传分数作为本项目观察。GLM 的完整应用流水线还包含版面分析，不要把外部版面模块误当成语言模型内部的层。

## 运行前的预测题

1. 同一张图片尺寸翻倍后，视觉 token 一定变成四倍吗？最小面积分支会发生什么？
2. 若把 patch 按普通行优先排列而不是 merge 邻域优先排列，shape 还会正确吗？哪个独立锚点测试会失败？
3. 静态图片为什么仍有 temporal_patch_size=2？这意味着出现两个时间 grid 吗？
4. 将 merger 输入的 300 个 patch 误当成语言侧 300 个视觉 token，会在哪里报错？
5. MLP 与固定 query 连接器分别压缩了什么？输出 shape 相同能否证明图像被利用？
6. 三模型用了同一公式和提示，但采用不同 tokenizer、模板、切图策略，输入 token 少能否直接说明更快、更准确？
7. 渲染子集分数更高时，怎样区分图片变清晰与公式子集更简单？
8. 若 val 选出的模型在 test 上更差，为什么不能据此换另一个模型重测？

## 怎样读结果与记录表

下方仅填本项目实测。所有 GPU 行来源均为 **本项目观察，RTX 4090（租用），bf16（processor fp32），batch=1，训练步数=0**；n 和生成设置见各表。CPU/MPS/5080 留白供学习者填写。速度包含共享 GPU 干扰及 Qwen3.5 PyTorch 回退，不能推断架构的绝对速度排序。正式结果填写于运行完成后。

<!-- CH32_RESULTS -->

局限：latex-v1 防止本项目划分泄漏，不证明这些开源样本从未出现在厂商预训练中；小规模前缀不代表完整数据集。规范化 EM 不等于数学等价，mathtext 只覆盖 LaTeX 子集。这里只提供第 33 章选型的零样本证据，不能推出微调后的排序。
