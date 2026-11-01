# 第 31 章实验：从零搭一个小公式 OCR

## 0. 准备

从 `course-v4/` 运行。先完成不需要数据、预训练权重或 GPU 的结构检查：

```bash
CUDA_VISIBLE_DEVICES="" python -m book.part4_vlm.ch31_mini_ocr.tour 1 --device cpu
CUDA_VISIBLE_DEVICES="" python -m pytest -q book/part4_vlm/ch31_mini_ocr
```

默认只运行部分 1；`--n 20` 是验证推理/视觉对照样本数。各部分可以单独指定。
本章训练的是随机初始化的小模型，没有真实大模型推理、LoRA 或权重下载。
CPU/MPS 的小规模流程检查可用固定字体渲染对照（仅验证训练工程，不作为开源数据成绩）：

```bash
python -m book.part4_vlm.ch31_mini_ocr.tour 2 3 4 --device cpu --source rendered \
  --train-n 128 --val-n 20 --test-n 20 --n 20 --steps 40 --out outputs/book/ch31/cpu_smoke
# MPS：同一命令把 --device cpu 改成 --device mps，另设新的 --out。
```

正式数据默认 `--source open`：只读本地已缓存的 `OleehyO/latex-formulas` 的
`cleaned_formulas` PNG 字节和公式文本，缺缓存就报错，不自动下载或换数据。
来源与许可见 [数据记录](../../../data/LICENSES.md)：openrail，来源包含 arXiv 和
im2latex-100K；仅本地学习，不把原始/转换后的数据或训练权重随教材分发。
本章正式训练使用开源原图，渲染数据只用于流程检查。

远程 Python 3.11、torch 2.14.1+cu130；复用已有 numpy、Pillow、pyarrow、matplotlib。
本章没有安装新包。以下用 `$COURSE_DIR` 表示远程教材副本目录，`$RUN_DIR` 表示本章
数据张量、模型和 matplotlib 缓存目录，`$LOG_DIR` 表示日志目录。运行前将这三个变量
设为自己的绝对路径，并创建运行和日志目录。下载缓存只读。
训练使用 bf16 autocast，参数、优化器和交叉熵为 fp32；TF32 关闭；saliency 用 fp32。
单进程 CUDA 分配上限为设备容量的 35%；开跑前仍须查看 `nvidia-smi`。

GPU 推荐配置（本次正式运行，先完成 30 步冒烟）：

```bash
# 先同步代码；远程执行下面的命令。
cd "$COURSE_DIR"
nvidia-smi
nohup env MPLCONFIGDIR="$RUN_DIR/mpl" PYTHONDONTWRITEBYTECODE=1 \
  python -u -m book.part4_vlm.ch31_mini_ocr.tour 3 \
  --device cuda --precision bf16 --source open \
  --train-n 20000 --val-n 500 --test-n 500 --n 100 \
  --width 256 --vision-depth 3 --decoder-depth 4 --patch 4 --max-length 64 \
  --batch-size 64 --steps 10000 --eval-every 500 --log-every 100 \
  --data-dir "$RUN_DIR/data" --out "$RUN_DIR/train" \
  > "$LOG_DIR/ch31_train.log" 2>&1 < /dev/null &
# 先用部分 1 2 3、--patch 8、--steps 30、--eval-every 30、--log-every 10
# 和新的 --out .../smoke 建缓存并检查；随后 patch=4 的部分 3 用 .../smoke_p4。
# 正式训练完成以后才运行分析；最后才单独打开最终 test。
nohup env MPLCONFIGDIR="$RUN_DIR/mpl" PYTHONDONTWRITEBYTECODE=1 \
  python -u -m book.part4_vlm.ch31_mini_ocr.tour 4 \
  --device cuda --precision bf16 --source open \
  --train-n 20000 --val-n 500 --test-n 500 --n 500 --batch-size 64 \
  --data-dir "$RUN_DIR/data" --run-dir "$RUN_DIR/train" \
  --out "$RUN_DIR/analysis" > "$LOG_DIR/ch31_analysis.log" 2>&1 < /dev/null &
# 等分析结束后：上述命令部分号改 5，加 --final-test，
# --out 改 .../test，日志改 "$LOG_DIR/ch31_test.log"。
```

以上训练规模是配置，不是效果数字。重复实验需更换输出路径。按第 0 章 §0.10.1，
选型冻结后，预先登记的候选可各评一次 test，不据此改任何设置。
本次运行详见正文 §31.7.4。
5080 可沿用较小 batch 的 CUDA 配置，但显存与耗时均待实测；不得把 4090 的数值抄入该列。

口径：完整语料先按 `latex-v1` seed=0 规范化键分组；之后筛选非空、UTF-8 长度不超过
64 字节的公式，每个键只留 parquet 顺序的首张非空图片。样本数参数不改变原始划分。
裁白边后等比双线性缩放、居中补零墨迹为 `1×32×160`。这是短公式、固定顺序子集，
不代表完整数据集，也不代表 TexPixel。manifest 保存快照、parquet、行号及键哈希。

## 1. 结构与监督信号

ViT 的 patch 线性投影和双向块复用第 29 章，MLP 连接器把每个视觉 token 映射到
GPT 维度；拼接 `[visual tokens; BOS; UTF-8 bytes]`，复用第 11 章带 RoPE 的 GPT。
采用前缀拼接，便于直接验证既有 GPT 的因果 mask 和 KV cache；ViT 在进入 GPT 前已
让视觉 token 双向交流。训练全参数，初始化随机，无预训练先验。

```mermaid
flowchart LR
    A[开源 PNG] --> B[裁白边 / 等比缩放 / 32×160 墨迹]
    B --> C[4×4 patch / 320 个视觉 token]
    C --> D[3 层 ViT / 宽度 256]
    D --> E[MLP 连接器]
    E --> F[视觉前缀 + BOS + 字节]
    F --> G[4 层 GPT / 因果注意力]
    G --> H[逐字节 greedy 生成 / EOS 停止]
```

图中为正式实验配置；320 是 `(32/4)×(160/4)` 的数学结果。

`# >>> 练习 31.1` 是连接器，31.2 是拼接/标签 mask，31.3 是 saliency。
机制实现集中在 `miniocr/models/formula_ocr.py`。这是本章的参考实现；A5 训练层用你自己的实现，只拿它对照。
PAD=0，字节=1..256，BOS=257，EOS=258。BOS 预测第一字节，最后字节预测 EOS；
视觉位置、padding 的 labels 为 -100。没有截断后伪造 EOS 的数据。
`part1.json` 保存 shape、参数量、有效标签数和所有参数梯度有限的断言。

## 2. 数据与划分

部分 2 写 `manifest.json`、`pairs.png` 并断言 train/val 规范化键无交集。
不读取 test 图片。开源路径复用 `miniocr.data.contrastive_pairs.prepare_pairs`，
由其调用 `latex_corpus.standard_split`。不能先抽出 20000 行再划分。
图像缓存是 caller-owned 的 `data/train.pt` 和 `data/val.pt`，更改样本数、来源或长度需
使用新的 `--data-dir`。渲染对照使用独立哈希分组，不能与 latex-v1 结果混报。

## 3. 训练与验证选择

AdamW、梯度裁剪 1、100 步 warmup、余弦学习率衰减。正式训练前先做 30 步冒烟，
检查有限 loss、每个可训练参数梯度，以及视觉/padding mask；冒烟不触碰 test。
模型按固定 val 前缀的最低规范化 CER 选取，平局取较早 checkpoint。
`curves.json` 保留逐段 loss、学习率、梯度范数、耗时和验证指标，
`learning_curves.png` 显示训练曲线。训练结束对整个选定 val 子集评测，并写
`frozen.json` 和 checkpoint SHA-256。重新训练须新 `--run-dir`，不会覆盖冻结实验。

本章使用第 0 章 §0.9.3 的 OCR 面板精简版，复用第 18 章 `score_postcorrect` 的 EM_norm、CER、mathtext parse_rate；
适配为 OCR 时只报告识别指标，不把无噪声文本适配器的 broken/fix rate 当成 OCR 指标。
CER 是规范 token 拼接字符的总编辑距离/总参考字符数，可超过 1；
parse_rate 只是 mathtext 子集可解析率，不是完整 LaTeX 编译率，也不意味着答案正确。
EM/parse 给 Wilson 95% 区间，CER 用 1000 次样本 bootstrap，seed=0。
区间只描述当前子集抽样波动，不涵盖训练种子、数据选择和设备差异。

## 4. 视觉依赖、错例与 saliency

只使用 val，同一 checkpoint、同一 greedy 解码：原图、中央 40 像素宽遮挡、
图片循环错配（无固定点）、全白图。`controls.json` 保存指标和逐条输出，
`error_cases.json` 并排显示四种输出供人工分析。

从 val 第 0 张取第 0、第 3（不足则末尾）、EOS 位置，画正确目标 token 的
log probability 对输入像素的绝对梯度。`saliency.npz` 保存原始数值，
`saliency.png` 显示叠图。这里的 token 是 UTF-8 字节，不一定是完整 LaTeX 命令。
梯度有 teacher-forced 正确前缀，和自由生成的注意模式不同；亮区不构成因果解释。

原图正确、遮挡后错误提示视觉证据重要；错配是同一批图片的重排，解析率不变不能证明语言先验
足以产出形式正确的文本。原图也错、遮挡后还错不能直接归因为视觉编码器；
需要人工回看预处理图片。分布外干预不是“视觉没看清/语言脑补”的精确分类器。

## 5. 最终 test（M2）

部分 5 必须显式 `--final-test`，否则完全跳过。冻结配置和 checkpoint 哈希校验通过后
用独占创建的 `TEST_STARTED.json` 防止重复评测；不导出 test 错例，不按 test 调参。
`final_test.json` 只记录汇总、区间、checkpoint 与 test 键哈希。
开发期不看 test 错例，最终报告定稿后才可解封。
M2 交付包括结构图说明、训练曲线、val/test 指标、val saliency 与错例。
A5 门槛待校准：由你第一次完整训练后只依据 val 结果确定，写在 test 评测之前；这次开源短公式实验的数字不是门槛，不能宣称已达到未定义的门槛。

## 运行前的预测题

1. 为什么第一输出字节的 label 对应 BOS 位置，而不是最后一个视觉位置？
2. 把 padding label 改成 0 会如何改变 loss 和生成的停止行为？
3. 如果遮挡后 parse_rate 维持很高而 EM 大降，能说明什么，不能说明什么？
4. 固定验证前缀上 CER 最低的 checkpoint，是否一定有最高 EM？
5. teacher-forced EOS 的 saliency 与第一个字节的 saliency 应该有什么差别？

## 怎样读结果

先看 mask/梯度断言，再看训练 loss 与 val CER 是否一起改善；接着对照 EM 与解析率，
最后用干预和 saliency 提出可检验的错因假设。只凭 loss 下降不能称为学会看图。
所有计时均为共享 GPU 的任务墙钟，包含同一任务的 CPU 评测/准备，并非独占 CUDA 内核时间。

## 6. 4090 实测记录（本项目观察）

以下数字来自 `outputs/book/ch31/gpu/`。RTX 4090（租用），bf16 autocast、参数与优化器 fp32、TF32 关闭；开源 latex-v1 子集（val=500、test=500）；单训练种子 31；贪心解码，最长 64 字节。CPU/MPS/5080 列留空。

### 6.1 训练运行与 val 选择（完整 val 500）

三次训练都是同一模型结构（宽度 256、ViT 3 层、GPT 4 层、batch 64、10000 步），只改变训练数据量与 patch 大小。随机初始化基线仅在 val 前 100 张（n=100）上测得 EM 0、CER 1.580，不能与完整 val500 混为同一评测集合；来源 `outputs/book/ch31/gpu/train/frozen.json` 的 baseline 和 `train/started.json` 的 n。

| 运行 | train_n | patch | 参数 | 最佳步 | val EM | val CER | val 可解析率 | 训练耗时 s |
|---|---|---|---|---|---|---|---|---|
| train（20k，patch 4） | 20000 | 4 | 5867264 | 10000 | 0.012 | 0.604 | 0.746 | 2682 |
| train_p8（20k，patch 8） | 20000 | 8 | 5818112 | 9500 | 0.004 | 0.648 | 0.784 | 1148 |
| **train100k（100k，patch 4）** | 100000 | 4 | 5867264 | 9000 | **0.454** | **0.124** | 0.902 | 1841 |

选择规则：只看 val，按完整 val 的 CER 最低取胜，选中 **train100k**。训练集扩大到 5 倍后，同一模型结构的 val CER 从 0.604 降到 0.124；20k 运行的训练 loss 在 9000 步附近降至约 0.01，而 val 停在 0.6 左右，属于记忆化。train100k 的训练 loss 为 0.158（第 10000 步），val CER 在第 9000 步最低（0.119，前 100 张子集上的选择点）。峰值显存 2681 MiB（进程 allocated）。

### 6.2 视觉依赖对照（val 500，同一 checkpoint）

| 条件 | EM | CER | 可解析率 |
|---|---|---|---|
| 原图 | 0.454 | 0.124 | 0.902 |
| 中央遮挡（40 像素宽） | 0.000 | 0.454 | 0.700 |
| 图片循环错配 | 0.000 | 0.969 | 0.902 |
| 全白图 | 0.000 | 0.887 | 0.000 |

遮挡和错配都使 EM 降为 0，说明模型确实读取了图像。错配输出 500/500 是原图输出的循环重排（CPU 核验 `outputs/book/ch31/gpu/analysis100k/controls.json`），解析率不变不能作为语言先验证据；全白图则几乎全部无法解析。这些是分布外干预，只作为图像依赖的证据，不做逐个错例的因果归因。错例与并排输出见 `outputs/book/ch31/gpu/analysis100k/error_cases.json`。

### 6.3 saliency

`analysis100k/saliency.png` 与 `saliency.npz` 给出 val 第 0 张图的三个 teacher-forced 目标位置的绝对梯度图。位置为 UTF-8 字节，不是完整 LaTeX 命令；梯度是 teacher-forced 的局部敏感度，不是自由生成的注意力。

### 6.4 最终 test（只评一次）

`train100k` 在 test 500 上评测一次（`test_evaluated_once=true`，test 键与 train/val 键无交集的断言通过）。

| 指标 | 值 | 95% 区间 |
|---|---|---|
| EM（规范化） | 0.472 | [0.429, 0.516] |
| 规范化前 EM | 0.100 | — |
| CER | 0.109 | [0.095, 0.123] |
| 可解析率 | 0.892 | [0.862, 0.916] |

记录：`outputs/book/ch31/gpu/train100k/final_test.json`。test 结果与 val（0.454 / 0.124）接近，说明选择没有明显过拟合 val；仍是单训练种子、单数据子集。

### 6.5 设备对照（待填）

| 设备 | 配置 | val EM | val CER | 训练耗时 s | 备注 |
|---|---|---|---|---|---|
| CPU | | | | | |
| MPS | | | | | |
| RTX 5080 | | | | | |

注：本章的 GPU 推荐命令默认 `--train-n 20000`，这是先跑的两个配置。最终冻结并测试的是 `--train-n 100000` 的运行，命令见下方。

```bash
# 100k 数据准备（CPU，约 2 分钟）：部分 2
python -m book.part4_vlm.ch31_mini_ocr.tour 2 --device cpu --source open --train-n 100000 --val-n 500 --test-n 500 --max-length 64 --data-dir "$RUN_DIR/data100k" --out "$RUN_DIR/prep100k"
# 100k 训练：部分 3（patch 4，其他与上面的推荐命令相同）
python -u -m book.part4_vlm.ch31_mini_ocr.tour 3 --device cuda --precision bf16 --source open --train-n 100000 --val-n 500 --test-n 500 --n 100 --width 256 --vision-depth 3 --decoder-depth 4 --patch 4 --max-length 64 --batch-size 64 --steps 10000 --eval-every 500 --log-every 100 --data-dir "$RUN_DIR/data100k" --out "$RUN_DIR/train100k"
# 分析：部分 4；随后单独运行部分 5 加 --final-test（已完成，TEST_STARTED.json 存在，不能重跑）
```

日志：`outputs/book/ch31/gpu/logs/`。

数据统计范围：`outputs/book/ch31/gpu/data_statistics.json` 只含 train=20000、val=500；100k 运行的规模和配置见 `train100k/started.json`，不要把20k字节长度分布当作100k统计。test CER 区间来源 `train100k/final_test.json`。
