# 第 30 章实验手册：CLIP 与 SigLIP

## 0. 准备

从 `course-v4/` 运行。部分号 **1–5** 对应 `tour.PARTS`，按位置参数顺序执行，每部分重置种子。本文件是实验操作与记录，不替代正文和习题。机制入口保留 `# >>> 练习 30.1–30.3`，供章节练习从零实现与检查；A5 只要求练习 30.1 对应的对称 InfoNCE，写在 `assignments/a5/mechanism/` 里。

```bash
# 离线 CPU 测试，不需要 GPU 或预训练权重
CUDA_VISIBLE_DEVICES="" python -m pytest -q book/part4_vlm/ch30_clip
# 默认：受控 mathtext 图文对，train=128、val=32，宽度32，CLIP/SigLIP 各80步
# part5 默认跳过，不读 test；换一个 --out 可重复实验
CUDA_VISIBLE_DEVICES="" python -m book.part4_vlm.ch30_clip.tour --out outputs/book/ch30/first
# 单独运行损失检查
python -m book.part4_vlm.ch30_clip.tour 1 --device cpu --out outputs/book/ch30/loss
# MPS 建议起点（待验证，不填性能数字）
python -m book.part4_vlm.ch30_clip.tour 1 2 3 4 --device mps --train-n 1024 --val-n 128 --test-n 128 --width 64 --batches 64 --steps 1000 --eval-every 100 --out outputs/book/ch30/mps
# 5080 建议先冒烟（待验证）
python -m book.part4_vlm.ch30_clip.tour 1 2 3 4 --device cuda --batches 32 --steps 30 --out outputs/book/ch30/5080_smoke
```

环境：租用 RTX 4090 48GB；Python 3.11.10、PyTorch 2.14.1+cu130、NumPy 2.1.2、Pillow 10.2.0、PyArrow 26.0.0、Matplotlib 3.11.2。**没有安装新包，没有下载任何模型或数据。** 完整环境和代码 SHA-256 在 `outputs/book/ch30/gpu/environment.json`。本章双塔全部随机初始化，不需要 CLIP/SigLIP 大权重。`--n` 默认 20，仅限制已训练模型的 val 推理/近重复展示数量；训练规模另由 `--train-n` 等控制。

远程在开跑前检查 `nvidia-smi`。程序将 PyTorch 分配上限设为整卡的 20%；CUDA 上下文、其他进程不计入该上限。先设置 `$RUN_DIR`、`$LOG_DIR`、`$COURSE_DIR`，分别指向远程本章运行目录、日志目录和教材副本目录，均使用绝对路径。运行产物、缓存、检查点都在 `$RUN_DIR`；现有开源 parquet 仅从共享缓存只读，不往该缓存写文件。`PYTHONDONTWRITEBYTECODE=1` 避免在代码同步目录生成字节码。所有训练使用 **fp32，关闭 TF32**。

```bash
# 先将 course-v4 代码同步到 "$COURSE_DIR"
mkdir -p "$RUN_DIR" "$LOG_DIR"
cd "$COURSE_DIR"
nvidia-smi
# 30 步冒烟；数据准备仍先划分完整语料；不读 test 图片
nohup env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR="$RUN_DIR/mpl" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m book.part4_vlm.ch30_clip.tour 1 2 3 4 --device cuda --source open \
  --train-n 8192 --val-n 512 --test-n 512 --width 96 --batches 128 \
  --temperatures .07 --steps 30 --eval-every 30 \
  --out "$RUN_DIR/smoke" --data-dir "$RUN_DIR/data_open" \
  > "$LOG_DIR/ch30_smoke.log" 2>&1 < /dev/null &
# 正式 8 配置：同样的训练图文对数量，统一每 16384 对验证一次
nohup env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR="$RUN_DIR/mpl" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m book.part4_vlm.ch30_clip.tour 1 2 3 4 --device cuda --source open \
  --train-n 8192 --val-n 512 --test-n 512 --width 96 --batches 32 128 \
  --temperatures .07 .2 --budget-examples 262144 --eval-examples 16384 --n 100 \
  --out "$RUN_DIR/final" --data-dir "$RUN_DIR/data_open" \
  > "$LOG_DIR/ch30_final.log" 2>&1 < /dev/null &
# 小规模固定字体/布局的受控渲染实验；两种损失各 2000 步
nohup env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR="$RUN_DIR/mpl" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m book.part4_vlm.ch30_clip.tour 1 2 3 4 --device cuda --source rendered \
  --train-n 1024 --val-n 128 --test-n 128 --width 64 --batches 64 \
  --temperatures .07 --steps 2000 --eval-every 200 --n 100 \
  --out "$RUN_DIR/rendered" --data-dir "$RUN_DIR/data_rendered" \
  > "$LOG_DIR/ch30_rendered.log" 2>&1 < /dev/null &
```

单独运行部分 3 时，数据不存在则创建；部分 4/5 必须通过 `--run-dir` 指向部分 3 的目录，并提供一致的 `--source`、`--data-dir`、训练/验证/测试数量及 `--max-length`。`--out` 必须是新目录，防止覆盖日志。部分 5 的最终命令见下文。`started.json` 保存完整参数与环境，`run.json` 保存起止时间，`curves.csv` 保存训练过程，`frozen.json` 保存所有验证结果、最终胜出配置及数据 SHA-256。

### 数据与评测口径

- 主实验使用 [OleehyO/latex-formulas](https://huggingface.co/datasets/OleehyO/latex-formulas) 的 `cleaned_formulas` 原始 PNG。许可 **openrail**，沿用仓库 `data/LICENSES.md` 的记录；只在本地学习使用，不再分发原数据或训练权重。
- 共用 `miniocr/data/latex_corpus.py::standard_split(n_train=None)` 在**完整缓存语料**上按规范化键分组，保持 **latex-v1 / seed=0**。随后过滤 UTF-8 长度不超过 96 字节、非空图，每键保留第一条；按 parquet 顺序取 train=8192、val=512、最终 test=512。没有先截语料再划分，没有偷偷重渲染开源 PNG。这个固定前缀是短公式子集，不能代表全语料。
- 图片裁白边、保比例缩放、居中补白到 `1×32×160`；墨迹为 1，背景为 0。文本使用完整 UTF-8 字节，ID 1–256，padding=0；过长样本排除，禁止截断。文本 Transformer 双向注意力屏蔽 padding，池化也排除 padding。
- 渲染对照用项目 `miniocr/eval/render.py` 的 mathtext，固定字体和布局，公式为线性式、分式、幂式三种模板；唯一公式键在选择规模前用哈希分配到 train/val/test。它独立于开源 latex-v1，不与开源实验合并统计，也不能视作真实 LaTeX 排版性能。
- 训练批内不放重复规范化键。两种损失共享随机初始化和种子；相同批量时训练抽样序列相同。温度固定用于控制变量；SigLIP 偏置初始化为 `-log(B-1)` 后学习，CLIP 不训练偏置。优化器 AdamW，lr=0.001、weight decay=0.001，梯度裁剪 1。
- 每个配置用 **val 双向 R@1 的均值最高**的 checkpoint，同分保留最早者；配置间同分按预先声明顺序。所有调参与选择只看 val。主实验测试只评这个胜出模型一次，不将 8 组配置都放到 test 比较。
- R@K 在同一完整候选图库上计算，图→文/文→图各只有一个规范化键正例。图库大小和查询数相等；K 大于图库时取图库大小。并列分数按图库索引稳定排序。95% 区间为固定图库上的 1000 次查询 bootstrap，不包含训练种子方差。
- 固定已见样本预算不等于固定 FLOPs：B=32 为 8192 步，B=128 为 2048 步（数学结论）；负样本对数量和优化更新次数仍不同。只做单训练种子，不把小差异解释成普遍规律。

## 1. 损失、温度与偏置

运行部分 1，查看 `loss_checks.csv`。`miniocr/nn/contrastive.py` 中的 InfoNCE 用 `logsumexp` 从零实现行/列交叉熵平均；SigLIP 用 `logaddexp(0,z)` 从零实现逐对 logistic 损失。没有在机制函数中调用 CE/BCE。测试才用 PyTorch 参考值对拍前向、反向和 double gradcheck。

数学口径：`S = normalize(image) @ normalize(text).T / temperature`，对角线为正例。CLIP 两方向各按 B 平均，再取均值。SigLIP 对 B² 对求和，**除以 B，不是 B²**。零 logits 时分别为 `log(B)` 和 `B*log(2)`。给所有 logits 加同一常量不改变 CLIP，却改变 SigLIP。零相似度下 `-log(B-1)` 的偏置使总偏置梯度为零。共同温度/偏置不会改变固定 embedding 的检索排序，但会改变训练梯度。

SigLIP 的逐对形式不需要全局 softmax 归一化，便于分块或分布式组织；本实验仍显式建立 B×B 矩阵，**没有实现通信优化，也没有证明大批量吞吐优势**。

## 2. 图文对与泄漏检查

运行部分 2，检查 `pairs.png` 与 `data_manifest.json`。ID 包含数据快照、parquet 文件和行号；规范化键 SHA-256 用于去重与划分交集检查。图片张量和数据缓存留在 `--data-dir`。这一步不解码 test 图片。图标题为排版而截短，完整文本保存在 manifest；模型输入没有截断。

先读原图，再看缩放后的分数/下标是否仍可辨认。归一化键不相同的公式仍可能视觉相似，单正例损失仍可能把它们当负例；这个限制正是部分 4 的人工审阅动机。

## 3. 双塔检索与批量/温度消融

运行部分 3。图像塔为三层小 CNN，保留空间网格后投影；文本塔为两层小 Transformer、四头注意力、padding-safe 平均池化。两塔输出 L2 归一化向量，训练后用余弦相似度检索。它不会自回归生成 LaTeX；不能把 R@1 当成 OCR 的字符串准确率。

先查日志中的 `smoke`：可训练参数、正/负对数、首步 loss、梯度和 mask 检查。再读 `curves.csv` 的 loss、val R@1/5/10、SigLIP 偏置。检查点只按 val 选择，不按训练 loss，也不跨损失比较 loss 的绝对值。`train_subset` 仅为前 512 个训练样本组成的诊断图库，与 val 使用独立图库。

## 4. 找近重复样本

运行部分 4，使用 val 胜出模型。`near_duplicates.json/png` 为排除自身后的图像近邻，供人工审阅；`retrieval_examples.json` 保存默认 20 个图→文查询的 top-5 候选 ID。自然近邻没有人工重复标注，因此不能报告“去重准确率”。

另外对前 n 张 val 图施加亮度乘 0.85 和右移 1 像素（不环绕），再到原 val 图库找来源。这个注入对照有已知答案，报告来源恢复 R@1；它验证轻微扰动下的检索行为，不等同于自然近重复召回率。它完全不改训练数据和 test。

## 5. 最终测试与归档

只在 val 选择结束、配置冻结后运行。程序写 `TEST_STARTED.json`，随后一次性编码 test；所有 R@K 与区间复用这次结果。存在该标记会拒绝再次评测，故不能用同一个 `--run-dir` 重跑 test。`frozen.json` 保留测试前的快照（其中 `test_evaluated=false`），最终是否完成以 `final_test.json` 为准。

```bash
cd "$COURSE_DIR"
nohup env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR="$RUN_DIR/mpl" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m book.part4_vlm.ch30_clip.tour 5 --device cuda --source open \
  --train-n 8192 --val-n 512 --test-n 512 --n 100 --final-test \
  --data-dir "$RUN_DIR/data_open" --run-dir "$RUN_DIR/final" \
  --out "$RUN_DIR/test" \
  > "$LOG_DIR/ch30_test.log" 2>&1 < /dev/null &
```

实验结果只从实测日志填入下面的记录；CPU/MPS/5080 性能列留空。本章完整报告、日志、JSON/CSV 和图位于 `outputs/book/ch30/gpu/`。训练权重和缓存留在远程本章运行目录，不复制到代码同步目录。

## 运行前的预测题

1. 同一相似度矩阵的温度从 0.2 降到 0.07，排序、loss 和梯度分别怎样变化？错配的强负例会怎样？
2. SigLIP 若误除以 B²，批量变大时梯度尺度如何变化？为何偏置不能照搬 CLIP？
3. 相同已见图文对预算下，B=128 一定胜过 B=32 吗？哪些变量还没有控制？
4. 同一个公式的两张图进了同批，单对角正例标签会怎样？跨 train/val 则会怎样？
5. 图库从 32 扩到 512，随机 R@1 的数学期望如何变化？能直接比较两种规模的 R@1 吗？
6. 图像近邻看起来很像，是否足以删除其中一条？把 `x^2` 和 `x_2` 当重复会带来什么后果？

## 怎样读结果

先确认图库规模、划分、精度和训练预算，再与随机初始化基线比较。优先看双向 R@1，再看 R@5/10 能否容纳正确答案。若训练改善而 val 不改善，应怀疑泛化；若两者都接近随机，应查梯度、掩码、配对和模型容量。最终 test 的作用是报告冻结方案，不用来决定继续训练哪个配置。不要把短公式、单种子、固定图库的结果外推为 CLIP/SigLIP 的通用胜负。

## 6. 4090 实测记录

以下数字全部来自 `outputs/book/ch30/gpu/` 中的 JSON/CSV（本项目观察，RTX 4090（租用），fp32、TF32 关闭；开源 latex-v1 固定前缀子集 train=8192、val=512、test=512；单训练种子；图库大小 512，128 为渲染对照）。R@K 是同一固定图库上的检索率，不是 OCR 字符串准确率。

### 6.1 部分 1：损失检查（B=16，归档 GPU 参考误差为 0；CPU 有浮点舍入差）

| 损失 | 温度 | 数值 | 梯度范数 | 偏置梯度 |
|---|---|---|---|---|
| CLIP | 0.03 | 47.691 | 10.831 | — |
| SigLIP | 0.03 | 176.375 | 23.192 | 7.053 |
| CLIP | 0.07 | 20.580 | 4.515 | — |
| SigLIP | 0.07 | 66.111 | 9.069 | 6.131 |
| CLIP | 0.2 | 7.563 | 1.496 | — |
| SigLIP | 0.2 | 16.547 | 2.465 | 3.802 |
| CLIP | 1.0 | 2.990 | 0.248 | — |
| SigLIP | 1.0 | 4.027 | 0.252 | 0.388 |

与 PyTorch 参考实现的绝对误差均为 0.0（见 `outputs/book/ch30/gpu/final/loss_checks.csv`）。这只对应归档 GPU 运算；CPU 不能要求逐 bit 相同。SigLIP 与 CLIP 的目标函数不同（逐对 logistic 与双向 softmax CE），SigLIP 还有偏置；除以 B 的归一化也影响量级，不能仅凭归一化解释二者差值或比较优劣。

### 6.2 部分 3：批量与温度消融（开源 val，最终选择只看 val）

图库 512，随机初始化基线的 val i2t R@1 为 0.002。

| 配置 | 损失 | B | 温度 | 步数 | 最佳步 | val i2t R@1 | val t2i R@1 | val i2t R@5 | val t2i R@5 | val i2t R@10 | val t2i R@10 | 训练耗时 s | 峰值显存 MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clip_b32_t0.07 | clip | 32 | 0.07 | 8192 | 6656 | 0.568 | 0.553 | 0.807 | 0.809 | 0.896 | 0.893 | 328 | 317 |
| clip_b32_t0.2 | clip | 32 | 0.2 | 8192 | 8192 | 0.586 | 0.562 | 0.824 | 0.820 | 0.889 | 0.881 | 342 | 317 |
| clip_b128_t0.07 | clip | 128 | 0.07 | 2048 | 2048 | 0.482 | 0.484 | 0.758 | 0.732 | 0.828 | 0.830 | 147 | 637 |
| clip_b128_t0.2 | clip | 128 | 0.2 | 2048 | 2048 | 0.434 | 0.420 | 0.732 | 0.713 | 0.811 | 0.811 | 171 | 638 |
| siglip_b32_t0.07 | siglip | 32 | 0.07 | 8192 | 8192 | 0.506 | 0.518 | 0.779 | 0.781 | 0.855 | 0.859 | 359 | 317 |
| siglip_b32_t0.2 | siglip | 32 | 0.2 | 8192 | 7168 | 0.453 | 0.465 | 0.754 | 0.738 | 0.830 | 0.816 | 345 | 317 |
| siglip_b128_t0.07 | siglip | 128 | 0.07 | 2048 | 2048 | 0.416 | 0.381 | 0.666 | 0.656 | 0.762 | 0.742 | 182 | 637 |
| siglip_b128_t0.2 | siglip | 128 | 0.2 | 2048 | 2048 | 0.357 | 0.336 | 0.637 | 0.645 | 0.756 | 0.758 | 179 | 637 |

按 val 双向 R@1 均值选出 **clip_b32_t0.2**（均值 0.574）。预算口径：B=32 训练 8192 步，B=128 训练 2048 步，两者都看到 262144 个图文对，但负样本数和更新次数不同。单种子下，B=32 的 CLIP 高于 B=128，CLIP 在 B=32 时温度 0.2 与 0.07 相差约 2 个百分点，在 B=128 时温度 0.07 比 0.2 高约 5 个百分点（i2t R@1）；SigLIP 在两种批量下都是 0.07 更好。这些都只来自单种子，不宜外推。

### 6.3 部分 3：渲染对照（mathtext，train=1024、val=128、图库 128）

| 配置 | 损失 | B | 温度 | 步数 | 最佳步 | val i2t R@1 | val t2i R@1 | val i2t R@5 | val i2t R@10 | 训练耗时 s | 峰值显存 MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clip_b64_t0.07 | clip | 64 | 0.07 | 2000 | 800 | 1.000 | 1.000 | 1.000 | 1.000 | 114 | 239 |
| siglip_b64_t0.07 | siglip | 64 | 0.07 | 2000 | 1800 | 1.000 | 0.992 | 1.000 | 1.000 | 120 | 239 |

渲染数据已饱和（R@1 接近 1），**不能用来比较批量或温度**；它只证明管线在简单受控数据上能学会对齐。胜出配置为 clip_b64_t0.07（`outputs/book/ch30/gpu/rendered/frozen.json`）。

### 6.4 最终测试（只评一次）

胜出配置 clip_b32_t0.2 在开源 test（512 图库，1000 次查询 bootstrap，95% 区间）：

| 指标 | 值 | 95% 区间 |
|---|---|---|
| i2t R@1 | 0.578 | [0.531, 0.619] |
| t2i R@1 | 0.553 | [0.508, 0.598] |
| i2t R@5 | 0.826 | [0.791, 0.857] |
| t2i R@5 | 0.818 | [0.783, 0.850] |
| i2t R@10 | 0.879 | [0.850, 0.908] |
| t2i R@10 | 0.875 | [0.844, 0.902] |

test 结果与 val（0.586 / 0.562）接近，说明选择没有明显过拟合 val；但仍是单训练种子。输出为 `outputs/book/ch30/gpu/final/final_test.json`，`TEST_STARTED.json` 存在，不能重跑。

### 6.5 部分 4：近重复演示

- 注入对照（val 前 100 张，亮度 ×0.85 且右移 1 像素）：开源图库 512 中来源恢复 R@1 = 0.990（99/100）；渲染图库 128 中为 0.980（98/100）。注入标签已知，这验证轻微扰动下的检索，不是自然近重复召回率。
- 自然近邻：`outputs/book/ch30/gpu/final/near_duplicates.json` 与 `near_duplicates.png` 给出 100 张查询的图像近邻，供人工审阅；没有重复标注，不报告“去重准确率”。

### 6.6 设备对照（待填）

| 设备 | 配置 | val i2t R@1 | 训练耗时 s | 备注 |
|---|---|---|---|---|
| CPU | | | | |
| MPS | | | | |
| RTX 5080 | | | | |

CPU补核：`tour 1 --device cpu --seed 30 --out /tmp/ch30-review` 的最大参考误差为1.9073486328125e-6（PyTorch2.14.1+cu130、fp32），来源为本章 `tour.py::part1` 的CPU复算；归档GPU `final/loss_checks.csv` 的0.0不能推广到CPU。随机初始化实测t2i R@1=0.00390625，来源 `outputs/book/ch30/gpu/final/frozen.json` 的baseline；不是随机排序数学期望。
