# 第 28 章实验手册：卷积与 CNN

## 0. 准备

从 `course-v4/` 运行。部分号 **1–5** 与 `tour.PARTS` 一致，位置参数保留执行顺序；每部分重置种子。采用与第 18、27 章一致的部分号、`--device`、`--n` 和结构化日志约定。

```bash
# 快速正确性检查，无 GPU、无权重、无网络依赖
CUDA_VISIBLE_DEVICES="" python -m pytest -q book/part4_vlm/ch28_cnn
# 本地小规模：默认 CPU、fp32、2 线程、384 张符号图、100 步
CUDA_VISIBLE_DEVICES="" python -m book.part4_vlm.ch28_cnn.tour 1 2 4 5 --out outputs/book/ch28/first
# 真实 PNG 下采样：只读已有本地 Hugging Face 缓存，默认 n=20
CUDA_VISIBLE_DEVICES="" python -m book.part4_vlm.ch28_cnn.tour 3 --n 20 --out outputs/book/ch28/images
# MPS 推荐起步配置；数字待实测
python -m book.part4_vlm.ch28_cnn.tour 4 5 --device mps --sources 40 --per-source 4 --steps 500 --batch-size 64 --out outputs/book/ch28/mps
```

GPU 命令（在租用机器上运行）。先 `nvidia-smi` 确认空闲显存；运行文件、临时缓存、检查点放在 `$RUN_DIR`，日志放在 `$LOG_DIR`。`$RUN_DIR` 是本章运行目录，`$LOG_DIR` 是日志目录，`$COURSE_DIR` 是教材副本根目录；运行前将这三个变量设为机器上的绝对路径。已有 HF 缓存只读，未下载数据/权重、未 pip 安装、未修改系统设置。`MPLCONFIGDIR` 和禁止写 pyc 的设置避免把运行产物留在同步目录。

```bash
mkdir -p "${RUN_DIR}" "${LOG_DIR}"
cd "${COURSE_DIR}"
# 40 步冒烟，不评 test
nohup env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR="${RUN_DIR}/mpl" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m book.part4_vlm.ch28_cnn.tour 1 2 4 5 --device cuda \
  --sources 30 --per-source 4 --steps 40 --batch-size 256 --eval-every 20 \
  --out "${RUN_DIR}/smoke" > "${LOG_DIR}/ch28_smoke.log" 2>&1 < /dev/null &
# 正式配置：独立数据 seed=28；6 个训练（2 模型 × 3 种子）各 5000 步
nohup env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR="${RUN_DIR}/mpl" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m book.part4_vlm.ch28_cnn.tour 1 2 4 5 3 --device cuda \
  --seed 28 --seeds 0 1 2 --sources 400 --per-source 4 --steps 5000 \
  --batch-size 256 --eval-every 250 --n 256 --require-data --final-test \
  --out "${RUN_DIR}/final" > "${LOG_DIR}/ch28_final.log" 2>&1 < /dev/null &
tail -f "${LOG_DIR}/ch28_final.log"
```

重跑请换一个新的 `--out`；训练遇到已有数据缓存/训练结果会拒绝覆盖。`--steps`、`--sources`、`--per-source`、`--batch-size`、`--width`、`--seeds` 可放大。CPU 默认只跑一个种子；本章不加载预训练模型，因此没有大权重用例。`--n` 默认 20，仅控制真实公式 val 图数，不截断划分语料。默认缺 PNG 时明确跳过真实图部分，仍做受控字号实验；正式运行加 `--require-data`，缺数据直接失败。

**环境与口径：** 租用 RTX 4090（本章 `outputs/book/ch28/gpu/final/run.json` 只记录 GPU 名称，未记录显存容量），Python 3.11、torch 2.14.1+cu130；确切版本见 `gpu/final/run.json`。第 1 部分即使指定 CUDA 仍强制 CPU/NumPy float64；其余张量机制及训练为指定设备 float32，TF32 关闭，不使用 AMP。CPU/MPS/5080 未作为性能基准实测，记录表留空。GPU 时长用同步后的墙钟时间，包含评估和检查点 I/O；共享机器负载会影响时间，不将它称为纯 GPU 内核时间。

**数据口径：**

- 主视觉损伤实验用已下载的 [OleehyO/latex-formulas](https://huggingface.co/datasets/OleehyO/latex-formulas) `cleaned_formulas` PNG，许可沿用仓库 `data/LICENSES.md` 的 openrail 记录，来源 arXiv / im2latex-100K，仅在本地学习使用。快照 `436791f14f92cbb7fdc1b27f2c316cf1f4e65445`，6 个 parquet。完整文本先走 `latex_corpus.standard_split(n_train=None)`，保持 latex-v1 seed=0、val/test 各约 5000 条的规范键分组；再按 parquet 顺序选择前 256 个不同 val 规范键，不触碰 test 图。CSV 保存快照/文件/行号及规范键 SHA-256。不是先截取原始语料再划分，也不是 standard_split 列表的前缀。
- 公式 PNG 没有单符号框标注，不能直接用作第 5 章的 24 类单符号监督任务。为复做该任务、固定字号/姿态/噪声及来源组，分类对照特意复用 `miniocr.data.symbols` 的受控生成器。共 24 类 × 400 来源/类 × 4 张/来源 = 38,400 张；按来源组 80/10/10，数据 seed=28、划分 seed=0，来源交集必须为零。这是受控合成符号准确率，不是开源公式 OCR 准确率。第 5 章现有脚本实际比较 CNN 与线性分类器；本章另建两隐藏层 MLP 作要求的对照。
- 两种模型用相同样本、取批种子、batch=256、AdamW（lr=0.001，weight_decay=0.0001）、5000 步、fp32；无额外在线增广。MLP 隐藏宽 72/72，CNN 通道 24/48；参数量接近但不完全相等，不能声称是严格等参数或等 FLOPs 对比。分类每张图一个标签，无 padding，无须语言模型的 label mask。原有后纠错与字符串评测接口未改；像素损伤不能借用 EM/CER 冒充识别指标。

**评测协议：** 冒烟只检查训练/val，不评 test；正式数据与冒烟种子不同。每个模型/种子只按 val CE 最低选检查点，所有配置写入 `frozen_validation.json` 后冻结。只有显式 `--final-test` 才对这套预声明的六个模型各做一次 test 前向，结果单列于 `final_test.json`；不查看 test 错例，不按它挑模型或改设置。区间按来源组做 1000 次 bootstrap，不能把四张近重复图当作独立观测。下采样分类对照只看 val。

## 1. 手写卷积与数值梯度（部分 1）

`miniocr/nn/cnn.py` 中的 `conv_forward` / `conv_backward` 使用 NCHW 和 OIHW，做互相关，支持矩形核、padding 和 stride。机制层不调用现成卷积、unfold 或自动微分。反向将重叠窗口的输入梯度累加；padding=0 不可用 `p:-p` 切片。

三组几何 `(stride,padding)=(1,0),(1,1),(2,1)`，输入 2×2×4×5、核 3×2×2×3，所有 x/w/b 坐标都用中心差分（epsilon=1e-5）检查，再与 PyTorch 前后向对拍。报告绝对误差和相对误差；近零梯度的相对误差可能偏大。`gradients.csv` 保留逐组结果。练习标记 28.2/28.3 对应 A5 的 M1、M2（A5 在 `assignments/a5/mechanism/` 里实现）；训练为了速度使用优化的 torch 卷积。

## 2. 输出尺寸与感受野（部分 2）

数学结论：`floor((H+2p-d(k-1)-1)/s)+1`；有效核 `d(k-1)+1`。每层更新 `r += (k_eff-1)*jump`，然后 `jump *= stride`。工具还记录首个输出的输入坐标中心，区分 padding 对中心和感受野的影响。

本章 CNN 的 conv3/pool2/conv3/pool2 使 32→32→16→16→8；末端特征局部理论感受野 10×10、跳距 4，后续全连接分类器聚合所有位置。不要把这个局部数字说成整网分类输出的感受野。练习位置 28.1。

## 3. 连续下采样与小字（部分 3；分类影响见部分 5）

真实 PNG 不先缩放，墨迹=1、白纸=0；透明处先合成白纸。连续 2×2 average pooling 0–4 次，再用最近邻放回原尺寸；奇数边在右/下补白纸至 2 的相应幂倍数，统计时裁回原范围。记录各图原始 H/W，放回大图仅供逐坐标比较，不恢复丢失的信息。

`real_downsampling.csv` 记录整图 MSE、原图墨迹>0.5 像素位置上的恢复对比度与阈值召回。均值是逐图平均，空墨迹图不进入前景均值。白色背景容易掩盖损伤，因此不能只看整图 MSE。`has_script` 只是 LaTeX 含上下标的粗标签，不是小字定位。

受控对照固定符号 x，只改字号 14/40，再同样下采样；这些字号是实验配置，不是实测结论。`controlled_downsampling.csv`、`downsampling.png` 分别保存数字与图。部分 5 用相同的最佳分类器在 val 上比较 2/4/8 倍降采样后放回 32×32 的识别率；不重新训练，也不借用 test 调整下采样方案。

## 4. BatchNorm 与 LayerNorm（部分 4）

练习 28.4 实现按 N/H/W 统计每个通道的 BatchNorm，手写均值、方差、仿射变换和 running 状态；这一部分反向由基础张量运算的 autograd 完成。训练输出用有偏方差，running variance 更新用无偏估计；momentum 表示新观测的权重。测试覆盖多批训练、梯度、eval 不更新状态和退化单值批。

对比 LayerNorm 的轴为每个样本的 C/H/W。固定第一张 anchor，只把其他样本整体加 5：比较 anchor 输出变化；再切到 eval，使用固定 running 统计重做。归一化函数与 PyTorch 的前向、dx、dweight、running variance 最大误差写入 `part4.json`。这是统计依赖演示，没有宣称 BN 比 LN 的分类准确率更高。

## 5. CNN 与 MLP（部分 5）

查看 `learning_curves.png` 和 `learning_curves.csv`：虚线为训练 CE，实线为验证 CE；另画验证准确率。曲线每 250 步评完整 train/val，并额外评第 1 步及末步。参数、最佳步数、训练墙钟秒数、PyTorch 峰值 allocated 显存、干净 val 和降采样 val 准确率都在 `training.json`。峰值不含别的进程、驱动或所有 reserved 缓存，不能与 nvidia-smi 的整卡占用混为一谈。

CPU/MPS 先用小配置检查机制；表中 CUDA 结果不能当作它们的速度承诺。单种子不是普遍结论；三个种子共享同一数据，因此种子标准差不等于数据抽样不确定性。

## 运行前的预测题

1. stride 从 1 变 2，输出宽高怎样变化？参数量是否变化？手算一个矩形核例子。
2. 反向的窗口重叠处如果用赋值代替累加，哪个梯度先失败？padding=0 能否暴露切片错误？
3. 两次 3×3 卷积之间加一次 stride=2 池化，感受野与 jump 各怎么变？
4. 小字号放回大图为什么仍模糊？整图 MSE 与前景对比度谁更容易掩盖细线消失？
5. 固定 anchor、改变同批其他图，BN train、BN eval、LN 各是否变化？
6. 参数量接近时 CNN 一定胜过 MLP 吗？降采样的损伤会线性增长吗？先写下预测，再看验证结果。

## 怎样读结果

先检查机制误差，再确认训练 loss、参数梯度和数据划分；随后看完整 val 曲线、最佳检查点及退化图。最后单独读取已冻结的 test 报告。准确率来自受控符号任务，真实公式部分只有像素损伤；本章没有训练公式序列解码器，不能声称测得真实公式 OCR 的 EM/CER。

下列记录为**本项目观察，RTX 4090（租用）**；精度、样本数、步数随表标明。梯度检查实际运行在该租用主机的 CPU float64，不冒称 GPU 数值差分。CPU/MPS/5080 栏仅留待同配置复跑填写。

| 实验与口径 | 本项目观察，RTX 4090（租用） | CPU | MPS | 5080 |
|---|---|---|---|---|
| 手写卷积中心差分；租用主机 CPU float64，3 组几何，357 个坐标，0 训练步 | 最大绝对误差 6.879e-10；最大相对误差 5.351e-09 | | | |
| 与 PyTorch 对拍；同上 | 前向最大误差 1.776e-15；反向最大误差 2.220e-15 | | | |
| 手写 BN 对拍；CUDA fp32，N=8/C=3/H=6/W=5，0 训练步 | 前向 4.768e-07；dx 4.768e-07；dweight 5.722e-06；running var 5.960e-08 | | | |
| 替换 anchor 的批同伴；同上 | BN train 平均绝对变化 2.366321；BN eval=0；LN train/eval=0 | | | |


训练：CUDA fp32，train/val/test=30,720/3,840/3,840，batch=256，每个模型 5,000 步，种子 0/1/2；只按 val CE 选检查点。

| 模型/种子 | 参数量 | 最佳步 | 4090 val 准确率 | val CE | 训练含评估秒 | 峰值 allocated MiB | CPU | MPS | 5080 |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| mlp/0 | 80,808 | 2750 | 80.0260% | 0.630751 | 11.285 | 22.569 | | | |
| cnn/0 | 84,408 | 2000 | 93.4896% | 0.213258 | 18.582 | 273.128 | | | |
| mlp/1 | 80,808 | 2500 | 80.1302% | 0.603541 | 10.151 | 22.892 | | | |
| cnn/1 | 84,408 | 1750 | 93.9583% | 0.195261 | 13.643 | 273.128 | | | |
| mlp/2 | 80,808 | 3000 | 80.3385% | 0.618525 | 8.581 | 22.892 | | | |
| cnn/2 | 84,408 | 2000 | 93.6198% | 0.206660 | 14.226 | 273.128 | | | |

MLP val 三种子均值 80.1649%，样本标准差 0.1591 个百分点。CNN val 三种子均值 93.6892%，样本标准差 0.2420 个百分点。


**独立最终 test：** 所有配置冻结后每个模型只前向评测一次，CUDA fp32，n=3,840（960 个来源组）；没有用 test 选择设置。区间为固定种子 1000 次来源组 bootstrap。

| 模型/种子 | 4090 test 正确数/样本数 | 准确率 | 来源组 95% 区间 | CPU | MPS | 5080 |
|---|---:|---:|---|---|---|---|
| mlp/0 | 3122/3840 | 81.3021% | [79.3223%, 83.2819%] | | | |
| cnn/0 | 3621/3840 | 94.2969% | [93.1510%, 95.4948%] | | | |
| mlp/1 | 3120/3840 | 81.2500% | [79.0625%, 83.2552%] | | | |
| cnn/1 | 3630/3840 | 94.5312% | [93.3854%, 95.5475%] | | | |
| mlp/2 | 3158/3840 | 82.2396% | [80.3639%, 84.1927%] | | | |
| cnn/2 | 3648/3840 | 95.0000% | [93.8802%, 96.1198%] | | | |


真实公式：CUDA fp32，latex-v1 val 的 256 个不同规范键，原生 PNG 分辨率，0 训练步；下表为逐图平均，前景阈值 0.5。

| 降采样倍数 | 4090 前景恢复对比度 | 前景阈值召回 | 整图 MSE | CPU | MPS | 5080 |
|---|---:|---:|---:|---|---|---|
| 1× | 0.698095 | 100.0000% | 0.000000 | | | |
| 2× | 0.411325 | 25.8789% | 0.007914 | | | |
| 4× | 0.231434 | 1.2683% | 0.013426 | | | |
| 8× | 0.141872 | 0.0000% | 0.016279 | | | |
| 16× | 0.086751 | 0.0000% | 0.018156 | | | |

控制字号 x（CUDA fp32，每个字号一张、0 训练步）：4× 下采样后，字号 14 的前景阈值召回为 0%，字号 40 为 68.4211%。这是受控字形结果，不能泛化到所有字体。


固定最佳模型的下采样识别对照：CUDA fp32，val n=3,840，三个种子准确率均值，不再训练。

| 模型 | 4090 原图 | 2× | 4× | 8× | CPU | MPS | 5080 |
|---|---:|---:|---:|---:|---|---|---|
| MLP | 80.1649% | 79.5312% | 53.3941% | 17.6389% | | | |
| CNN | 93.6892% | 89.4965% | 49.1319% | 17.1701% | | | |

CNN 在原图上的优势不等于抗任意下采样：4×/8× 时本次 CNN 均值反而低于 MLP。像素恢复后的外观相近不能保证模型仍能识别。


正式实验 2026-10-29T10:42:13.157385+00:00 至 2026-10-29T10:44:42.644948+00:00，墙钟 149.488 秒；冒烟 4.808 秒，合计 **154.295 秒（2.572 分钟）**。六个正式训练循环（含评估/检查点）合计 76.469 秒；含冒烟为 77.299 秒。完整墙钟还包含 CPU 数据准备、float64 梯度检查及 PNG 读取，不能当作纯 CUDA 内核时长。正式部分 3 用时 45.288 秒。

证据：`outputs/book/ch28/gpu/ch28_final.log`、`final/part1.json`、`final/part4.json`、`final/real_downsampling.csv`、`final/training.json`、`final/final_test.json`；解读见[本章正文](chapter.md)。原始语料 552,340 行（parquet 元数据实测），实际图像实验只读 val 256 张。

