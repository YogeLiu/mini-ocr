# 第 27 章实验手册：图像作为张量

## 0. 准备

在 `course-v4/` 下运行。沿用第 26 章的位置参数部分号、`--device`、`--n`、`--out` 和逐行 JSON 日志约定。默认仅运行部分 1、2，CPU、fp32、两线程、seed=0；真实推理默认 `--n 20`。本章无训练，步数始终为 0，不需要训练 loss、标签 mask 或 LoRA；真实模型被冻结，日志核验可训练参数为 0，并检查图像占位符数量。

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q book/part4_vlm/ch27_image_tensors
CUDA_VISIBLE_DEVICES="" python -m book.part4_vlm.ch27_image_tensors.tour
CUDA_VISIBLE_DEVICES="" python -m book.part4_vlm.ch27_image_tensors.tour 2 --sides 24 48 96 --out outputs/book/ch27/my-resize
# 已有本地权重和数据才运行；不会自动下载。缺权重明确 skip，缺语料报错。
CUDA_VISIBLE_DEVICES="" python -m book.part4_vlm.ch27_image_tensors.tour --prepare-data --n 20 --test-n 10 --data outputs/book/ch27/my-data --out outputs/book/ch27/prepare
CUDA_VISIBLE_DEVICES="" python -m book.part4_vlm.ch27_image_tensors.tour 3 --n 20 --models smol --configs letterbox384 --data outputs/book/ch27/my-data --out outputs/book/ch27/local-inference
```

MPS 使用 `--device mps --dtype fp32`；5080 使用 `--device cuda --dtype bf16`。这两种设备没有本章实测结果，不给出速度外推。`--height/--width` 放大部分 1 的张量；`--sides/--dpi` 放大部分 2；`--n/--max-new-tokens` 放大推理。数据准备不覆盖已有 manifest，最终 test 有独占锁；其余同名输出会覆盖，比较配置请使用不同 `--out`。

### GPU 准备与实际启动方式

远程 Python 3.11、torch 2.14.1+cu130、transformers 5.19.0、pyarrow 26.0.0、matplotlib 3.11.2、Pillow 10.2.0。**没有 pip 安装新包**。直接复用既有批准模型：以硬链接复制到本章 `hf/hub`，只读快照；已有数据仓库在本章 `hf/hub` 建只读使用的符号链接。下文的 `$PROJECT_DIR`、`$RUN_DIR`、`$LOG_DIR` 分别表示远程教材副本、本章运行产物和日志目录的绝对路径，运行前设为自己的路径。所有新缓存、图、manifest、锁和运行输出在 `$RUN_DIR/`，日志在 `$LOG_DIR/ch27_*.log`。远程源码目录不保存运行产物。配套 GPU 脚本读取环境变量 `COURSE_WORK_ROOT`（未设置时默认 `/hy-tmp`）；可用 `export COURSE_WORK_ROOT=/自己的工作根目录` 设置，教材目录、产物和日志分别由 `$COURSE_WORK_ROOT/course-v4`、`$COURSE_WORK_ROOT/runs/...`、`$COURSE_WORK_ROOT/logs` 派生。

模型快照：

| 模型 | revision | 许可来源 |
|---|---|---|
| Qwen/Qwen3.5-0.8B | `2fc06364715b967f1860aea9cf38778875588b17` | [模型卡，Apache-2.0](https://huggingface.co/Qwen/Qwen3.5-0.8B) |
| HuggingFaceTB/SmolVLM-256M-Instruct | `7e3e67edbbed1bf9888184d9df282b700a323964` | [模型卡，Apache-2.0](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) |

本次没有下载新模型或数据。若复现环境没有这两个批准模型，应在远程用 `HF_HUB_DISABLE_XET=1 HF_ENDPOINT=https://hf-mirror.com HF_HOME=$RUN_DIR/hf` 下载上述固定 revision；不要用推理脚本隐式下载。processor 使用现有的 PIL 静态图适配器，避免构造未使用的视频后端；没有安装 torchvision。Qwen 提示缺少 causal_conv1d/flash-linear-attention 并回退 PyTorch 实现，保持现有环境，速度记录包含这一限制。Smol 加载时有默认配置 pad_token 警告；实加载 tokenizer/generation 的 pad ID 均为 2，嵌入表有 49280 行，运行时逐批检查 token 未越界及 attention_mask 为左 padding。相关警告保留在原日志，没有修改系统包。

先同步代码到远程教材副本，再执行以下内容。每次开跑前检查 `nvidia-smi`。脚本按模型顺序运行；默认 batch=1，4090 正式配置放大到 `--batch-size 8`，先做 20 张批推理冒烟并检查显存。左侧 padding 配合 attention_mask，生成后裁掉首个 EOS 之后的 pad；逐图时间是批耗时均分，汇总是整个配置墙钟时间。

```bash
cd "$PROJECT_DIR"
export PYTHONDONTWRITEBYTECODE=1 HF_HOME="$RUN_DIR/hf"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="$RUN_DIR/mpl" XDG_CACHE_HOME="$RUN_DIR/cache" TMPDIR="$RUN_DIR/tmp"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
# nohup 外层保证 SSH 断开后继续。目录和只读缓存须先准备好。
nohup python -u -m book.part4_vlm.ch27_image_tensors.tour 1 2 --device cuda \
  --prepare-data --n 200 --test-n 100 --data "$RUN_DIR/data" \
  --data-cache "$RUN_DIR/hf/hub" --out "$RUN_DIR/mechanisms" \
  > "$LOG_DIR/ch27_prepare.log" 2>&1 < /dev/null &
# manifest 完成后：每模型、每配置 20 张，检查图像 token、截断数和耗时。
nohup python -u -m book.part4_vlm.ch27_image_tensors.tour 3 --device cuda --dtype bf16 --n 20 \
  --configs letterbox128 letterbox384 --data "$RUN_DIR/data" \
  --model-cache "$RUN_DIR/hf/hub" --out "$RUN_DIR/smoke" \
  > "$LOG_DIR/ch27_smoke.log" 2>&1 < /dev/null &
# 再以 --batch-size 8、--out "$RUN_DIR/smoke_batch" 重跑同一 20 张冒烟。
# 原图 native 也做 --batch-size 8 的 20 张冒烟，以验证变长图像的左 padding。
# 冒烟完成并估算正式时长后：受控 val -> 原图 val -> 冻结 -> test 一次 -> CSV/图。
nohup sh book/part4_vlm/ch27_image_tensors/run_gpu.sh \
  > "$LOG_DIR/ch27_driver.log" 2>&1 < /dev/null &
```

### 数据与评测口径

来源为 [OleehyO/latex-formulas，cleaned_formulas](https://huggingface.co/datasets/OleehyO/latex-formulas)，数据卡标记 openrail。本次读取已缓存的六个 parquet，快照 `436791f14f92cbb7fdc1b27f2c316cf1f4e65445`；不下载其他数据，不分发原始语料。实读 552340 行（本项目观察，来自六个 parquet 的 metadata，详见归档 `environment.json`）。

先调用 `standard_split(n_train=0)` 对**完整语料**按规范化键执行 latex-v1（seed=0、val/test 默认各约 5000 条），再取不同规范键。主验证集是标准 val 顺序的前 200 张开源原始 PNG，不按长度或渲染能力筛选。受控验证集独立取同一 val 顺序中前 200 条 mathtext 可渲染公式，在 DPI=200、白边 12 像素下重渲染，用于隔离字体与图像处理变量。共尝试 346 个规范键、排除 146 个解析失败项（本项目观察）；manifest 完整记录失败原因、源 parquet 行号和 PNG 哈希。这两个 val 子集重叠 108 个规范键（本项目观察），但不等同，禁止把二者成绩差异全部归因于字体。

受控比较中两模型、六种预处理（含 native）使用完全相同的 200 张图，汇总带 `sample_ids_sha256`。原图比较中两模型、六种预处理使用完全相同的另一组 200 张图。test 取独立标准 test 前 100 个规范键，只在原图 val 选定一个模型和配置后评一次；选择规则依次为 EM 高、CER 低、视觉 token 少。冻结文件包含模型 revision、提示词、精度、批量、生成上限、val 结果哈希和 test 数量；test 锁写在数据目录，切换 `--out` 不能绕过。测试产物只保存总分、置信区间和运行统计，不输出预测文本或逐样本正确性，开发期间不做 test 错例分析，最终报告定稿后才可解封（第 0 章 §0.10.1）。

本章使用 OCR 主任务面板的一部分（第 0 章 §0.9.3）：主要报告 EM_norm、字符 CER 和达到生成上限的条数；CER_token 和 `well_formed` 保存在结果 JSON 中，可编译率和渲染一致率没有计算。像素指标不属于渲染一致率。

复用 `miniocr/eval/postcorrect_eval.extract_formula` 提取答案，复用 `score_pair/summarize` 计算规范化匹配率和**语料级** CER。CER 为总编辑数除以总参考字符数，可能大于 1；不是逐图 CER 的平均。原始回答指标也保留，便于看格式提取带来的变化。正式 batch=8（最后一批可较小），bf16；批量矩阵计算的舍入可能改变贪心解码近似平票时的结果，所以不把 batch=1 冒烟成绩混入正式汇总。贪心解码、禁用 thinking、`max_new_tokens=128`，达到上限且没有 EOS 的数量单列。所有推理都是零样本，不把参考公式放入提示词。

## 1. HWC、CHW 与归一化

`tour 1` 输出形状、stride、范围和逆变换误差。图像先把透明背景合成白色，再 `uint8 -> float32 /255 -> permute(2,0,1)`。逐通道 `(x-mean)/std` 默认 mean/std 都为 0.5；逆变换可检查轴与广播。`reshape` 不等于换轴。练习标记 27.1、27.2 在 `miniocr/data/image_tensors.py`，不修改后纠错接口。

## 2. 缩放、letterbox 与细线

`tour 2` 对三个已知单分式做 3 个边长 × 4 种处理，共 36 组比较。最近邻采用像素中心的 nearest-exact；双线性采用 `align_corners=False`，不额外抗锯齿；面积缩放按源与目标像素覆盖长度积分，非整数比例也保持权重含义。`square` 拉伸到正方形；`letterbox` 保持比例并补白，奇数 padding 明确留在右/下侧。练习标记 27.3、27.4 对应面积权重和几何坐标。

每张图保存原图和分式线掩码。分式线掩码只针对已知单分式：中央 60% 高度中最长连续深色线段，至少占图宽三分之一。应先看掩码是否确实覆盖分式线，不能把这个启发式当成开源图片的人工标注。

处理图先去 padding、再双线性回到原坐标，然后计算像素 MAE、墨迹 IoU/召回。分式线保留率是原掩码像素中仍小于 0.5 的比例；连续墨迹量另存 `fraction_line_soft`，避免只看二值阈值。这里的“恢复”只统一坐标，不能恢复丢失信息。渲染器会随字体与版本变化，本地 CPU 与远程的字体并不相同，跨设备应复用归档 PNG 才能做纯设备比较。

## 3. 同批受控 val 的双模型识别

`tour 3` 默认五个处理配置；正式运行另加 native 基线，共六个：

| 配置 | 外部像素处理 | 要隔离的因素 |
|---|---|---|
| native | 无本章外部缩放，直接送入 processor | 原分辨率基线 |
| nearest128 | 最近邻拉伸为 128×128 | 与 bilinear128 比插值 |
| bilinear128 | 双线性拉伸为 128×128 | 小图基线 |
| area128 | 面积拉伸为 128×128 | 与双线性比面积抗锯齿 |
| letterbox128 | 双线性、等比例、128×128 白底 | 与 bilinear128 比几何形变 |
| letterbox384 | 双线性、等比例、384×384 白底 | 与 letterbox128 比分辨率 |

Qwen processor 的最小/最大面积为 1024/262144，使这些预处理尺寸不会被统一放大成同一种输入；grid 合并数与实际图像占位 token 数逐条核对。SmolVLM 使用预训练 512 像素单裁剪、关闭图像分块，保持原来的图像 token 序列长度。不能只修改 max_image_size 而不检查模型与占位 token 是否一致。记录的是 **processor 后**的 token 和 `pixel_values` 形状，外部尺寸不等于模型实际尺寸。Qwen 的分辨率对照同时改变了像素信息与视觉 token 预算，不能把收益单独归因于其中一个因素；Smol 的固定单裁剪让本章外部尺寸对照保持相同 token 数，但不能推广到开启图像分块的配置。

每个配置输出完整 val 预测 JSONL、字符串指标、输入图哈希、像素差异、视觉 token、耗时、长度截断和进程峰值 allocated/reserved 显存。显存是本进程测量，不是整张卡所有并行任务的总量。

## 4. 原始开源 PNG 复核与 val 选择

`tour 4` 支持同样所有配置。批推理冒烟确认时长允许后，本次正式放大为原图的全部五种处理加 native；native 表示不做本章外部缩放，仍经过各模型的官方 processor。六个配置、两模型的设置选择只看这组 200 张原图 val。不要把受控集的高分直接当作原图效果。

```bash
python -m book.part4_vlm.ch27_image_tensors.tour --select-from "$RUN_DIR/original" \
  --test-n 100 --data "$RUN_DIR/data" --out "$RUN_DIR/freeze"
```

## 5. 最终 test 一次

只有 `tour 5 --final-test` 可读取 test；必须先有 `frozen.json`。模型、预处理、精度、批量和解码配置自动读取冻结设置，样本数必须等于冻结数量。报告 EM 与 CER 以及按公式 bootstrap 的 95% percentile 区间（2000 次重采样、seed=0）；这不是对其他数据域的保证。

## 运行前的预测题

1. 把 HWC 直接 reshape 为 CHW，为何形状对了，颜色仍会错？怎样用一个彩色像素验证？
2. 一条细线缩小时，最近邻是否总能保留它？线条相对采样网格的位置会怎样影响结果？
3. 面积缩放保留平均墨迹量，为什么二值分式线召回仍可能很低？
4. 等比例缩放避免形变，但长公式放进正方形时，为什么小字反而可能更小？
5. 输入边长增大后，两个模型的视觉 token 是否都会增加？先看什么日志再回答？
6. 预测更长导致 CER 大于 1，能否把它裁成 1？为什么不能按 test 成绩重新选择分辨率？

## 怎样读结果

先核对 `sample_ids_sha256` 和样本数，再比较相同来源中的 EM/CER；先看分式线掩码与 `resizing.png`，再把二值召回、连续墨迹、MAE 和识别指标对照。若更大图片未带来收益，检查 processor 后尺寸、视觉 token、截断率和模型输出内容。像素保留多不自动等于 OCR 好；两模型的绝对成绩也受预训练和结构差异影响。

下方数字仅填已完成的 GPU 实测。CPU/MPS/5080 保留空白供学习者记录；本地 CPU 单元测试不是这张表的跨设备成绩。

<!-- RESULTS_BEGIN -->
以下数字来自 `outputs/book/ch27/gpu/` 的归档 JSON（本项目观察，RTX 4090（租用），bf16 推理，batch=8，贪心解码，`max_new_tokens=128`，零训练步）。EM 为规范化后的完全匹配率，CER 为语料级字符编辑距离除以参考字符数（可大于 1）。`视觉 token` 是 processor 后的平均图像占位数。CPU/MPS/5080 列留空。

### 1 部分：张量形状与归一化

| 项目 | 实测 |
|---|---|
| HWC 形状 | (48, 96, 3) |
| CHW 形状与 stride | (3, 48, 96)，stride (4608, 96, 1) |
| 归一化后范围 | [-1, 1] |
| 逆变换最大绝对误差 | 1.49e-8 |
| `reshape` 是否等于换轴 | 否 |

### 2 部分：缩放与分式线（3 个已知单分式 × 3 个边长）

每格为 3 个公式的均值。分式线二值保留是原掩码像素中仍小于 0.5 的比例；连续墨迹量保留软值。

| 边长 | 方法 | 几何 | MAE | 墨迹 IoU | 墨迹召回 | 分式线二值保留 | 分式线连续墨迹 |
|---|---|---|---|---|---|---|---|
| 32 | bilinear | letterbox | 0.0523 | 0.282 | 0.317 | 0.000 | 0.096 |
| 32 | area | square | 0.0547 | 0.225 | 0.234 | 0.000 | 0.346 |
| 32 | bilinear | square | 0.0505 | 0.340 | 0.410 | 0.333 | 0.448 |
| 32 | nearest | square | 0.0481 | 0.363 | 0.458 | 0.333 | 0.325 |
| 64 | bilinear | letterbox | 0.0326 | 0.698 | 0.715 | 0.333 | 0.489 |
| 64 | area | square | 0.0296 | 0.739 | 0.756 | 0.667 | 0.674 |
| 64 | bilinear | square | 0.0275 | 0.761 | 0.777 | 0.667 | 0.721 |
| 64 | nearest | square | 0.0250 | 0.721 | 0.793 | 0.992 | 0.807 |
| 128 | bilinear | letterbox | 0.0176 | 0.958 | 0.967 | 1.000 | 0.687 |
| 128 | area | square | 0.0087 | 0.968 | 0.976 | 1.000 | 0.889 |
| 128 | bilinear | square | 0.0151 | 0.969 | 0.975 | 1.000 | 0.731 |
| 128 | nearest | square | 0.0047 | 0.956 | 0.971 | 1.000 | 0.999 |

只有 3 个公式，每格样本极少，只用于说明机制，不作为缩放方法的排名。

### 3 部分：受控 val（mathtext 渲染，200 张，两模型同批）

| 模型 | 配置 | EM | CER | 视觉 token | 被截断数 | 耗时 s |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | native | 0.105 | 0.160 | 36.3 | 20 | 257 |
| Qwen3.5-0.8B | letterbox384 | 0.095 | 0.225 | 144 | 28 | 222 |
| Qwen3.5-0.8B | area128 | 0.025 | 0.754 | 16 | 56 | 256 |
| Qwen3.5-0.8B | bilinear128 | 0.030 | 0.890 | 16 | 77 | 248 |
| Qwen3.5-0.8B | letterbox128 | 0.015 | 0.930 | 16 | 74 | 235 |
| Qwen3.5-0.8B | nearest128 | 0.015 | 0.948 | 16 | 85 | 228 |
| SmolVLM-256M | native | 0.010 | 0.919 | 64 | 49 | 119 |
| SmolVLM-256M | area128 | 0.000 | 1.245 | 64 | 106 | 145 |
| SmolVLM-256M | bilinear128 | 0.005 | 1.270 | 64 | 115 | 153 |
| SmolVLM-256M | letterbox128 | 0.000 | 1.702 | 64 | 151 | 127 |
| SmolVLM-256M | letterbox384 | 0.000 | 1.294 | 64 | 99 | 126 |
| SmolVLM-256M | nearest128 | 0.000 | 1.483 | 64 | 132 | 158 |

注意 Qwen 的 128 边长配置只有 16 个视觉 token，letterbox384 有 144 个，因此 384 的收益同时混合了分辨率和 token 预算。SmolVLM 的视觉 token 固定为 64。

### 4 部分：原始开源 PNG val（200 张，选择只看此表）

| 模型 | 配置 | EM | CER | 视觉 token | 被截断数 | 耗时 s |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | letterbox384 | 0.060 | 0.292 | 144 | 54 | 188 |
| Qwen3.5-0.8B | native | 0.035 | 0.306 | 23.1 | 57 | 224 |
| Qwen3.5-0.8B | area128 | 0.010 | 0.766 | 16 | 88 | 194 |
| Qwen3.5-0.8B | bilinear128 | 0.010 | 0.800 | 16 | 94 | 184 |
| Qwen3.5-0.8B | letterbox128 | 0.015 | 0.844 | 16 | 102 | 167 |
| Qwen3.5-0.8B | nearest128 | 0.000 | 1.045 | 16 | 127 | 224 |
| SmolVLM-256M | native | 0.005 | 1.046 | 64 | 87 | 102 |
| SmolVLM-256M | letterbox384 | 0.000 | 1.113 | 64 | 122 | 120 |
| SmolVLM-256M | area128 | 0.000 | 1.190 | 64 | 125 | 117 |
| SmolVLM-256M | bilinear128 | 0.005 | 1.200 | 64 | 133 | 126 |
| SmolVLM-256M | letterbox128 | 0.000 | 1.381 | 64 | 162 | 115 |
| SmolVLM-256M | nearest128 | 0.005 | 1.415 | 64 | 163 | 116 |

冻结规则（EM 高、CER 低、视觉 token 少、字典序）选出 **Qwen3.5-0.8B + letterbox384**。冻结记录见 `outputs/book/ch27/gpu/freeze/frozen.json`。

### 5 部分：最终 test 一次（原始 PNG，100 张，冻结配置）

| 指标 | 值 | 95% 区间（公式级 bootstrap，2000 次，seed=0） |
|---|---|---|
| EM（规范化） | 0.050 | [0.010, 0.100] |
| CER（字符） | 0.353 | [0.284, 0.412] |
| 规范化前 EM | 0.000 | — |
| well_formed | 0.840 | — |
| 视觉 token | 144 | — |
| 被截断数（128 上限） | 30 | — |

test 只读取一次（锁文件 `FINAL_TEST.lock` 存在）。记录：`outputs/book/ch27/gpu/final/qwen35_test_original_letterbox384.json`。
<!-- RESULTS_END -->

### 颜色空间与增强（仅列概念，本章不做效果实验）

| 项目 | 操作 | 本章需要记住的边界 |
|---|---|---|
| RGB / BGR | 通道交换 | 顺序必须符合 processor |
| 灰度 | 通道合并 | 公式黑白图可用，彩色信息会丢失 |
| sRGB / 线性 RGB | 非线性编码转换 | 本章直接在编码像素值上插值，未实验线性光域 |
| 亮度、对比度 | 光度增强 | 可能改变细线的二值阈值结果 |
| 噪声、模糊 | 采集扰动 | 不得把“更难”直接当成“更真实” |
| 裁剪、旋转、透视 | 几何增强 | 可能裁掉上下标、改变题意 |
| 颜色抖动 | 通道扰动 | 不保证适合单色公式 OCR |

`summarize_runs.py` 会复算保存预测的 val 指标，检查同配置两模型的图片哈希一致、样本数及截断数一致，并检查 test 产物没有逐条预测。输出 `audit.json`、`metrics.csv`、`recognition.png` 和 `timing.json`。

实验结论见本章正文 §27.5–§27.6，冻结记录见 `outputs/book/ch27/gpu/freeze/frozen.json`，测试结果见 `outputs/book/ch27/gpu/final/qwen35_test_original_letterbox384.json`。

### 缩放与环境补充

`outputs/book/ch27/gpu/mechanisms/pixels.json` 中三张分式原图 H×W 为 79×91、74×92、69×111；边长 128 的 square/letterbox 都是放大，不能据此评价缩小的抗混叠效果。远程 Pillow=10.2.0（`environment.json`），本地 CPU 复核环境为 12.3.0；跨机器应复用归档 PNG，记录版本，不能要求重渲染逐像素一致。
