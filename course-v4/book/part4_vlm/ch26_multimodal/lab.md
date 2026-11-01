# 第 26 章实验手册：多模态的核心问题

## 0. 准备

在 `course-v4/` 下执行，沿用位置参数部分号、`--device`、`--n`、`--out` 约定。本章不训练。

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q book/part4_vlm/ch26_multimodal
CUDA_VISIBLE_DEVICES="" python -m book.part4_vlm.ch26_multimodal.tour
# 位置参数可单独选择，默认只运行轻量的 1、2：
CUDA_VISIBLE_DEVICES="" python -m book.part4_vlm.ch26_multimodal.tour 2 --device cpu
# 有本地权重才进行推理；不自动下载。3 最多三张，4 默认 val n=20。
CUDA_VISIBLE_DEVICES="" python -m book.part4_vlm.ch26_multimodal.tour 3 4 --n 20 --out outputs/book/ch26/local-inference
```

默认 CPU、fp32、两线程、seed=0。部分号 **1–4**，每部分重置随机种子。默认输出 `outputs/book/ch26/local/`，日志是逐行 JSON，可重定向保存。相同输出目录会覆盖同名产物，比较配置应指定不同 `--out`。部分 3/4 的权重缺失会明确 skip；部分 4 的本地语料缺失也会 skip，不以合成数据冒充开源验证。部分 3/4 的 `--n` 默认 20；部分 3 取 `min(n,3)`，部分 4 必须有 n 个不同规范键。

`--batch-size` **只控制部分 2 的随机张量批大小**，真实模型始终 batch=1。`--visual-tokens`、`--text-tokens`、`--vision-dim`、`--text-dim` 可放大形状。`--dpi` 控制受控图渲染，默认 160；模型 processor 的面积配置为 `--min-pixels 65536 --max-pixels 262144`，这是处理参数，不是实测图片面积的承诺。`--max-new-tokens 256` 是生成上限，达到上限且无 EOS 会单列记录。

MPS 使用 `--device mps --dtype fp32`；5080 使用 `--device cuda --dtype bf16`。本章未测这两种设备，不给速度估算。

### 租用 GPU 的实际命令

先将代码同步到远程机器，以下命令在远程执行。`$SOURCE_DIR` 表示源码目录，`$RUN_DIR` 表示本章运行产物目录，`$LOG_DIR` 表示日志目录；运行前将这三个变量设为实际绝对路径，后两者放在源码目录之外。权重、临时目录、图、日志均在源码目录之外；不要把任何运行产物放进 `$SOURCE_DIR`。每次启动前执行 `nvidia-smi`，检查 GPU 占用。

本章使用 [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)（Apache-2.0）。下载命令在远程执行：

```bash
mkdir -p "${RUN_DIR}/hf" "${RUN_DIR}/mpl" "${RUN_DIR}/cache" "${RUN_DIR}/tmp" "${LOG_DIR}"
HF_HUB_DISABLE_XET=1 HF_ENDPOINT=https://hf-mirror.com HF_HOME="${RUN_DIR}/hf" HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Qwen/Qwen3.5-0.8B", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "README.md"]))'
```

本次固定快照 `2fc06364715b967f1860aea9cf38778875588b17`；复现时给 `snapshot_download` 加同一 `revision`。推理完全离线。

```bash
cd "$SOURCE_DIR"
export PYTHONDONTWRITEBYTECODE=1 HF_HOME="${RUN_DIR}/hf"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="${RUN_DIR}/mpl" XDG_CACHE_HOME="${RUN_DIR}/cache" TMPDIR="${RUN_DIR}/tmp"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
# 冒烟：先检查结构梯度、遮罩、模型参数和三图输出。
nohup python -u -m book.part4_vlm.ch26_multimodal.tour 1 2 3 --device cuda --dtype bf16 --n 3 --model-cache "${RUN_DIR}/hf/hub" --out "${RUN_DIR}/smoke" > "${LOG_DIR}/ch26_smoke.log" 2>&1 &
# 冒烟完成后再运行：放大结构张量和真实 val 样本，不改变提示或解码设置。
nohup python -u -m book.part4_vlm.ch26_multimodal.tour 1 2 3 4 --device cuda --dtype bf16 --n 100 --batch-size 8 --visual-tokens 64 --text-tokens 32 --vision-dim 32 --text-dim 48 --model-cache "${RUN_DIR}/hf/hub" --data-cache /root/.cache/huggingface/hub --out "${RUN_DIR}/main" > "${LOG_DIR}/ch26_main.log" 2>&1 &
tail -f "${LOG_DIR}/ch26_main.log"
```

**环境实测：** Python 3.11.10、torch 2.14.1+cu130、transformers 5.19.0。没有安装新包、修改系统设置或关闭实例。复用 `vlm_anatomy.load_image_only_processor` 的 PIL 静态图路径，不构造视频组件；仍使用官方模板和视觉 token 展开逻辑。缺少 `causal-conv1d`、`flash-linear-attention` 时采用现有 PyTorch 回退。共享 GPU 上的时间包含资源竞争，不能当成独占性能基准。

### 数据、计数和评测口径

- 部分 1/3 是固定三张受控教学图，隔离上下标、分式/根号、求和结构；使用既有 `miniocr/eval/render.py`。`render` 返回 0 白、255 墨迹，保存时反转并加白边。mathtext 仅支持部分 LaTeX；这三张图不能代表项目泛化成绩。
- 部分 4 优先用已缓存 [OleehyO/latex-formulas](https://huggingface.co/datasets/OleehyO/latex-formulas) 的 `cleaned_formulas` 原始 PNG（许可 openrail）。复用 `formula_images.load_formula_images`：全量文本先依 `latex_corpus.py` 的 `latex-v1`、seed=0、val/test 各约 5000 条划分，再按 parquet 顺序取 val 中前 n 个不同规范键。不截断语料后划分；不按公式难度、长度、可渲染性筛样本。只做白底透明合成、灰度再转 RGB，不重渲染开源图。快照/原始行 ID、规范键 SHA 和本次保存图片 SHA 在 `val_manifest.json`。
- 所有观察和设置选择只使用教学图与 val；没有训练、调参搜索或读取 test 图片/预测。本章不是 R3 定稿报告，**test 未评测**；为后续报告保留测试集。读取全量文本确定分组是划分步骤，不是 test 评测。
- 复用第 18 章 `extract_formula` 和既有 `score_pair/summarize`，不改接口。分别报告原输出和抽取输出的原串 EM、规范化 EM、语料级字符 CER；CER 是总编辑距离/总参考字符数。规范化不证明数学等价，抽取可能丢失信息，始终保留 `raw`。不报告后纠错专有的修好率/改坏率。
- `visual_tokens` 是网格经过空间合并后的数量，并断言它等于实际 `input_ids` 中 image token 的计数；不是原始 patch 数。`input_tokens` 包含这些视觉占位 token 和模板/指令；`non_image_input_tokens` 含特殊 token，不能当成自然语言词数。`generated_tokens` 包含实际生成的结束/控制 token；可见文本长度不等于该值。
- 每图 `seconds` 含 processor、传输、generate、decode，计时两端同步设备；不含加载、评分、原图提取。`run.json` 为整个 tour 墙钟，含数据准备与绘图，CUDA 峰值是本进程 PyTorch 分配器统计。零训练步，不把结构探针的 diagnostic loss 称为训练效果。
- 远程 parquet 元数据实查为 **552340 行、6 个文件**，快照 `436791f14f92cbb7fdc1b27f2c316cf1f4e65445`；这是本项目缓存观察，见 `environment.json`，不是其他快照规模的承诺。

## 1. 三张公式图

运行 `tour 1`，检查 `images/formula_1.png` 至 `formula_3.png` 和 `rendered.json`。公式内容固定，重新运行可核对图片 SHA；字体环境改变可能影响像素。

为每张图建立六行观察表，先写预测，运行真实识别后再补证据：

| 问题 | 本次要记录的证据 |
|---|---|
| 表示 | 原图尺寸、processor 网格、视觉 token 数；上标是否仍可见 |
| 对齐 | 指出某个符号对应的图像区域与输出 LaTeX 片段；不能用随机注意力图证明对齐 |
| 推理 | 区分转写、补全与求值；检查模型是否擅自改写公式 |
| 生成 | 原输出是否带围栏/定界符；抽取改变了什么；是否截断 |
| 迁移 | 三张渲染图与开源原图有什么分布差别；本次没有训练迁移实验 |
| 量化评估 | 原串 EM、规范化 EM、CER、像素比较是否给出相同判断 |

这些记录可作为 R3 错误分析素材，不在这里替学习者写六段正文。

## 2. 两类结构的最小形状演示

运行 `tour 2`，查看 `shapes.json`、`shapes.png`。练习入口位于 `miniocr/models/multimodal.py` 的 `# >>> 练习 26.1` 与 `26.2`。

数学结论：`[B,V,Dv]` 经线性连接器变成 `[B,V,Dt]`，与 `[B,T,Dt]` 沿序列轴拼接为 `[B,V+T,Dt]`。示意遮罩的视觉区双向可见；视觉不能读文字，文字不能偷看未来文字。这是一个明确的教学前缀遮罩选择，不声称所有 VLM 采用同一遮罩。

交叉注意力的 Q 为 `[B,T,Dt]`，K/V 为 `[B,V,Dt]`，权重 `[B,T,V]`，输出 `[B,T,Dt]`。实现显式矩阵乘法、缩放与 softmax；权重应按视觉轴求和为一。两类连接器均做一次反传，检查输入和全部参数梯度有限且非零，不进行 optimizer 更新。图中权重来自随机张量，**没有学到语义对齐**；此处也没有完整视觉编码器或语言解码器。

## 3. Qwen3.5 三图零样本

运行 `tour 3 --device cuda --dtype bf16`。只给图片和固定转写指令，不给参考答案；禁用 thinking、greedy 解码。`controlled.jsonl` 保存完整 prompt、生成 ID、原输出和抽取输出；`controlled.json` 汇总成绩和显存。

参考答案仅在生成结束后进入评分。观察 raw 与 extracted 两列，不能只贴较好的一列。部分 3 独立运行会自己生成图片，不依赖之前执行部分 1。

## 4. 开源原图 val 验证

运行 `tour 4 --n 20`，正式命令使用 n=100。图片和元数据保存到本次 `--out`，数据缓存只读。比较这部分与三张教学图时，要同时指出采样、公式复杂度和图像风格的差别；不是严格配对的因果消融。

记录 `val.jsonl` 中输出长度、截断标志、错误结构和 token 数。整个 100 张样本都计入汇总，不删错例。后续第 27–35 章再做预处理、训练和视觉依赖的配对实验，本章不据此宣称模型确实依靠图像完成了推理。

## 运行前的预测题

1. 视觉 token 翻倍时，前缀长度和交叉注意力矩阵的哪一个轴变化？输出文字长度是否必须变化？
2. 把前缀遮罩改为全可见，会通过什么路径泄漏未来答案？如何用扰动测试发现？
3. 三张公式中哪张生成 token 最多？视觉 token 的排序是否一定相同？
4. 模型输出 `$$...$$` 而参考没有定界符，原串与规范化评分会怎样？抽取能否被视为无损？
5. 三张渲染图全对，是否足以预测开源原图的成绩？随机交叉注意力热图能否解释真实模型的识别？

## 怎样读结果

先核对 `run.json` 的设备、精度、范围与训练步数，再看形状和 token 计数，最后看逐样本输出与汇总。对齐、推理、迁移都需要进一步实验，不能从三张正确输出直接下结论。像素不一致也可能来自渲染器对空格、分组和字体的处理，应回看规范化结果与图，而不是立即判断识别内容错误。

拉回 `smoke/` 和 `main/` 后，可在本地重建 `summary.json`、`summary.csv`、逐图 `tokens.csv` 与 `tokens.png`，不再调用模型：

```bash
python -m book.part4_vlm.ch26_multimodal.summarize_runs outputs/book/ch26/gpu
```

CSV 中的 EM Wilson 95% 区间复用现有统计函数，仅作小样本不确定性的参考；本次是固定 parquet 前缀、受控图也是人工选择，不能把独立同分布假设下的区间解释成整个真实 OCR 分布的保证。

以下记录表只填写已经完成的 GPU 实测；CPU/MPS/5080 列故意留空。所有 GPU 行均为 **本项目观察，RTX 4090（租用），bf16（processor 输出 fp32、送入模型前转 bf16；图片 uint8），训练步数=0**。真实模型 batch=1。实测产物位于 `outputs/book/ch26/gpu/`，具体结果与时间见该目录的 `summary.json`、`summary.csv`。

| 实验与规模 | RTX 4090（租用）实测 | CPU | MPS | 5080 | 本地证据（相对 `outputs/book/ch26/gpu/`） |
|---|---|---|---|---|---|
| 1：三图，DPI=160，uint8 | 尺寸 W×H：160×63、97×73、163×95 | | | | `main/rendered.json`、`main/images/` |
| 2：B=8、V=64、T=32、Dv=32、Dt=48，bf16 | 前缀 `[8,96,48]`；交叉权重 `[8,32,64]`；输出 `[8,32,48]` | | | | `main/shapes.json`、`main/shapes.png` |
| 2：单次反传、零更新 | 梯度全有限非零；权重行和最大误差 0.000610352；前缀/交叉参数 1584/5376 | | | | `main/shapes.json` |
| 3：三图零样本，bf16，batch=1，0 步 | 抽取后 EM_norm=3/3；CER=0；生成 89 token，4.320885 秒；截断 0 | | | | `main/controlled.json`、`ch26_main.log` |
| 3：原输出直接评分 | 原串 EM=0/3、EM_norm=0/3、CER=0.16；三条都带 `$$` 外壳 | | | | `main/controlled.jsonl` |
| 4：开源 val n=100，bf16，batch=1，0 步 | 抽取后 EM_norm=13/100；字符 CER=0.121768782；生成 8366 token，309.598576 秒；截断 1/100 | | | | `main/val.json`、`ch26_main.log` |
| 4：原输出直接评分 | 原串 EM=0/100、EM_norm=0/100、CER=0.157848422；全部保留在日志中 | | | | `main/val.jsonl` |
| 4：输入和显存 | 视觉 token 65–90，均值 75.65；非图像输入 36 token/图；峰值分配 1.673938 GiB，预留 1.746094 GiB | | | | `tokens.csv`、`main/val.json` |
| 完整成本：冒烟+正式 | tour 墙钟 9.656845+363.029859=372.686703 秒，即 0.103524 GPU 小时 | | | | `smoke/run.json`、`main/run.json`、`summary.json` |

| 三张受控图（顺序同部分 1） | 视觉 token | 输入总 token | 生成 token | 抽取后 EM_norm | 原串 mathtext 像素 diff |
|---|---|---|---|---|---|
| 上下标 | 78 | 114 | 25 | 1 | 0.951140065 |
| 分式与根号 | 70 | 106 | 27 | 1 | 0.590717300 |
| 求和与分式 | 77 | 113 | 37 | 1 | 0.957066190 |

上述像素比较对参考与抽取结果直接渲染，未先规范化；三条内容规范化匹配，但空格/分组的 mathtext 呈现不同，因此像素一致率为 0/3。这是本次观测到的指标分歧，不证明真实 LaTeX 渲染也会有同样差异。开源 val 的 13% 也不等同于数学语义正确率，需按本章问题框架看错例。

本地最终全量验收：`CUDA_VISIBLE_DEVICES="" python -m pytest -q`，**772 passed、1 skipped、2 warnings，10.70 秒**。警告来自第 29 章的 Matplotlib/NumPy 数组转换；没有失败项。日志：`outputs/book/ch26/gpu/local_full_pytest.log`。

### 冻结口径与渲染环境

`outputs/book/ch26/gpu/main/val.jsonl` 的CPU诊断：只去两端 `$$`、保留标点为43/100，冻结抽取口径仍是13/100；不替换R3成绩，抽取规则的盲区见正文 §26.7.3。同样DPI的 mathtext 图可能因字体/渲染环境改变尺寸，视觉token也随之改变；跨机器复现应复用 `main/images/` 与 `main/rendered.json`，不把归档token数当作DPI必然值。
