# 第 29 章实验手册：ViT

## 0. 准备

从 `course-v4/` 运行。部分号 **1–5** 对应 `tour.PARTS`，位置参数保留顺序，每部分重置种子。机制代码保留 `# >>> 练习 29.1–29.3`，作为章节练习的实现与检查入口；A5 在 `assignments/a5/mechanism/` 里实现同样的机制，规则更严（见 A5 讲义）。

```bash
# CPU 快速检查，不依赖 GPU、网络或大权重
CUDA_VISIBLE_DEVICES="" python -m pytest -q book/part4_vlm/ch29_vit
# 默认 CPU/fp32，384 张受控符号、每配置 80 步、2 个数据量、1 个种子
CUDA_VISIBLE_DEVICES="" python -m book.part4_vlm.ch29_vit.tour 1 2 3 5 --out outputs/book/ch29/first
# 原始开源 PNG；n 默认 20；缺缓存时显式 skip，正式实验用 --require-data 禁止跳过
CUDA_VISIBLE_DEVICES="" python -m book.part4_vlm.ch29_vit.tour 4 --n 20 --out outputs/book/ch29/images
# MPS 建议起点；未在 MPS 测量，不承诺耗时和准确率
python -m book.part4_vlm.ch29_vit.tour 5 --device mps --sources 40 --per-source 4 --steps 500 --batch-size 64 --out outputs/book/ch29/mps
# 5080 建议先用同一冒烟配置，再按显存放大；未实测
python -m book.part4_vlm.ch29_vit.tour 1 2 3 5 --device cuda --sources 30 --per-source 4 --steps 40 --batch-size 128 --width 64 --cnn-width 48 --eval-every 20 --fractions .125 1 --out outputs/book/ch29/5080_smoke
```

GPU 复现命令（租用 4090）：先查看 `nvidia-smi`，与其他章节共享实例。进程将 torch 显存分配限制为设备容量的 20%；这不是整机限制，CUDA 上下文与其他进程仍有额外开销。所有运行文件、数据缓存、检查点和 matplotlib 缓存都写在本章运行目录。`PYTHONDONTWRITEBYTECODE=1` 避免写入同步目录。完整命令参数也记录于 `started.json` / `run.json`。下面的 `$PROJECT_DIR` 是教材副本的绝对路径，`$RUN_DIR` 是本章运行文件目录，`$LOG_DIR` 是日志目录；运行前先设置这三个变量。

```bash
mkdir -p "$RUN_DIR" "$LOG_DIR"
cd "$PROJECT_DIR"
# 冒烟：不读 test；确认梯度、有效标签、checkpoint 保存/加载
nohup env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR="$RUN_DIR/mpl" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m book.part4_vlm.ch29_vit.tour 1 2 3 5 --device cuda \
  --sources 30 --per-source 4 --steps 40 --batch-size 128 --width 64 --cnn-width 48 \
  --eval-every 20 --fractions .125 1 --out "$RUN_DIR/smoke" \
  > "$LOG_DIR/ch29_smoke.log" 2>&1 < /dev/null &
# 正式：先固定全部 12 个配置；每个配置只按 val CE 选择 checkpoint，全部冻结后各评一次 test，不据此改设置
nohup env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR="$RUN_DIR/mpl" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m book.part4_vlm.ch29_vit.tour 1 2 3 4 5 --device cuda \
  --seed 29 --seeds 0 1 2 --sources 200 --per-source 4 --steps 5000 \
  --batch-size 128 --width 64 --cnn-width 30 --eval-every 250 \
  --fractions .125 1 --n 256 --require-data --final-test \
  --out "$RUN_DIR/final" > "$LOG_DIR/ch29_final.log" 2>&1 < /dev/null &
tail -f "$LOG_DIR/ch29_final.log"
```

重跑必须换一个新的 `--out`，拒绝覆盖已有部分结果或训练缓存。可放大 `--sources`、`--per-source`、`--steps`、`--batch-size`、`--width`、`--cnn-width`、`--depth`、`--seeds`；`--patch` 改训练模型，`--patches` 改 token 预算和图片网格。修改超参数的试验不要加 `--final-test`。没有真实预训练模型推理，本章全部模型从零初始化；`--n 20` 控制真实 PNG 分析数量。

环境：远程 Python 3.11.10、torch 2.14.1+cu130、NumPy 2.1.2，使用已有 Pillow 10.2.0、matplotlib 3.11.2、pyarrow 26.0.0；未安装新包、未下载新数据或权重、未改系统设置。原始 Hugging Face parquet 只读；本章没有向共享 HF 缓存写入数据。日志来自租用的 NVIDIA GeForce RTX 4090（实例提供 48GB 显存），共享负载下的耗时不能作为独占 GPU 基准。部分 1/3 的张量机制和部分 5 模型使用 CUDA；部分 4 的 PNG 解码、网格占用统计和绘图在实例 CPU 上完成；部分 2 是公式计算。

数据口径：真实图片来自 [OleehyO/latex-formulas](https://huggingface.co/datasets/OleehyO/latex-formulas) 的 `cleaned_formulas`，许可依仓库 `data/LICENSES.md` 记录为 **openrail**，来源含 arXiv/im2latex-100K；只在本地教学分析，不再分发原始数据。使用现有 `load_formula_images`，先在完整文本语料上建立 `latex-v1` 默认规范化键分组，再按 parquet 顺序取 val 中不同键的首张 PNG；不先截断语料、不改变第 18 章后纠错或评测接口。固定快照、行号、规范键 SHA-256 见 `final/part4.json`、`final/image_manifest.json`。真实图分析只读 val。

符号分类是**控制变量实验**：原始公式图没有单符号框标签，所以不把整幅公式伪称为单符号。复用第 5/28 章的 24 类符号生成器，固定字体、姿态、纸色、污点、噪声和数据 seed=29，只改变架构、训练来源数，并在同一 val 上施加固定 4 像素平移。这项控制专门检查数据量与空间归纳偏置，不宣称达到真实公式 OCR 效果。每类 200 个来源、每来源 4 张；按来源组 80/10/10 划分，seed=0；低数据组按类别取 train 来源的固定随机前缀，与全量嵌套。`.125` 对每类来源数向下取整，实际是 1888 张，不是直接对全部 train 张数乘比例。所有模型共享完全相同的 val/test；同一 seed 的批次索引配对。生成数据 SHA-256 和划分计数见 `final/training.json`。

## 1. Patch 嵌入：reshape 与 conv 对拍

入口 `tour 1`；实现 `miniocr/nn/vit.py` 的 `patchify` / `patch_embed`（练习 29.1）。把同一 `D,C,p,p` 权重分别用于 `reshape + linear` 和步长为 p 的卷积，比较输出以及 x、weight、bias 的梯度。包含多通道、非方形图像和三个 patch 尺寸。patch 内是 `C,pH,pW` 顺序，网格是行优先；改变展平顺序而不调整权重会破坏等价性。

不能整除时，机制函数明确报错，不能默默丢边缘。部分 2/4 采用显式右侧/底部补白的另一口径。CPU 单测另以 float64 数值梯度、独立切片参考、torch SDPA 参考验证。训练使用手写双向注意力，不调用现成 ViT/Transformer。

## 2. 视觉 token 数与注意力预算

入口 `tour 2`，输出 `tokens.csv`。**数学结论**：严格可整除时 N=(H/p)(W/p)；补白时 N=ceil(H/p)ceil(W/p)；CLS 额外增加 1。score 张量元素数为 B×heads×(N+1)²，fp32 每元素 4 字节。表中的 MiB 仅包含这一个张量，不包含 Q/K/V、MLP、梯度和优化器，不能当作峰值显存。

例如 64×256、p=16 得到 64 个空间 token、65 个含 CLS token；同图 p=8 为 256/257。高度和宽度都加倍时空间 token 数变为四倍。上述数字均是数学结论。

## 3. 位置表与输入分辨率改变

入口 `tour 3`；`interpolate_positions`（练习 29.2）显式接收旧/新二维网格，用 bicubic、`align_corners=False` 插值空间位置表，CLS 原样保留。同网格直接返回原张量。不能把长方形网格用 sqrt(token 数) 猜成正方形。

`position_interpolation.png` 展示随机位置表的一个通道，从 4×6 到 7×10；这只是机制可视化，不是假装学到的特征。小 ViT 的 32×32 和 48×64 前向都运行。部分 5 再对低/全量 × 三种子共 6 个 ViT 配置各自按 val CE 选中的 checkpoint 做 32→48 双线性图像缩放、位置表插值的 val 实验，检验“能运行”和“准确率能保持”的差别。

## 4. 真实公式的 patch 网格

入口 `tour 4`，输出 `formula_patches.png`、`patch_occupancy.csv`、`image_manifest.json`。在原始 PNG 像素上叠加 p=4/8/16/32 网格，不先缩小图片。展示样本按 LaTeX 含分式、上标、下标的标签选择；所有 n 张图都纳入统计。标签不等于符号框标注。

沿一条分式线观察它跨过多少列 patch、同 patch 是否还混入分子或分母；沿上/下标观察它与主符号共享哪些格子。网格边界穿过笔画不自动意味着笔画丢失：patch 嵌入是线性投影，**不是对 patch 求平均**。表中“占用”仅表示该 patch 至少有一个墨迹值>0.5 的像素；占用比例既不是信息保留率，也不是 OCR 准确率。要判断识别损伤还需训练/评测，不能从格子数量直接推断。

## 5. 小 ViT 与 CNN：数据量、平移和分辨率

入口 `tour 5`，机制是 `VisionBlock` / `TinyViT`（练习 29.3）。ViT：32×32 灰度、p=4、CLS、可学习二维位置表、pre-LN、双向 4 头注意力、两层 block、MLP 扩展比 4。CNN 复用第 28 章的两层 3×3 卷积、ReLU、池化和线性分类头。模型参数量由代码实计；正式配置尽量接近，但计算量与优化难度并不相同。

读归纳偏置时，先对应到代码：CNN 用共享的局部卷积核和池化强制逐层组合邻近笔画；ViT 先做局部 patch 投影，再让所有 token 通过注意力交互，邻近关系更多依赖位置表和训练数据。ViT 仍有 patch 切分这一局部先验，不能说完全没有归纳偏置；本实验也不预设 CNN 必胜。比较早期学习速度、低/全量差距和干预后的变化，而不是只看最终一列。

共同使用 fp32、关闭 TF32、AdamW(lr=.001, weight_decay=.0001)、梯度裁剪 1、batch=128、每配置 5000 步、seed=0/1/2。每 250 步及首/末步记录 val，最低 val CE 保存 checkpoint，再加载校验。每图一个有效类别，标签范围 0–23；无 padding 标签，无 causal mask。前向/反向有限性与每个可训练参数的梯度在第一步检查。低数据量与全量**相同步数而非相同 epoch 数**，过拟合也是观察内容。

`learning_curves.csv/.png`：val CE/准确率轨迹与 batch loss。`progress.json`：完成配置的中间记录。`frozen_validation.json`：在 test 前写出的全部配置结果。`final_test.json`：最终留出 test 各配置只跑一次前向，同时计算准确率、CE、按来源组 bootstrap 的 95% 区间（1000 次，固定 seed=0）。不看 test 错例、不根据 test 修改模型。种子均值与标准差描述初始化波动，组 bootstrap 描述该留出集的抽样波动，两者不可混为一个置信区间。

平移用补白而非环绕，可能裁掉边缘；该干预并不纯粹测量平移不变性。CNN 的 flatten 分类头也不严格平移不变。来源组互斥不等于字体家族互斥；本实验不能证明未见字体泛化。更大分辨率的 val 变化同时包含像素插值和位置插值，不能归因给位置插值单独一项。

## 运行前的预测题

1. 按 patch 顺序展平和按通道顺序展平有什么差别？错误顺序能否仍得到相同张量形状？
2. 65×257 的图用 p=16 补白时有多少空间 token？去掉 CLS 会怎样改变 attention score 大小？
3. 将 CLS 和空间 token 一起插值会发生什么？长方形位置表为何不能用平方根恢复形状？
4. 小上标完全落在一个 patch 中是否必然消失？分式线跨多个 patch 时模型还需什么机制整合它？
5. 相同步数下，低数据量的模型会经历更多次重复样本；预测 train/val 曲线会怎样分离。
6. 参数量接近时，哪个架构更快学会局部笔画？4 像素平移与 32→48 分辨率改变是否会保留准确率？先写预测再看记录。

## 怎样读结果

按 `part1.json` 的数值误差 → `tokens.csv` 的数学预算 → `position_interpolation.png` 的形状兼容 → 真实网格的结构位置 → 训练曲线的泛化差距顺序读。对比模型时同时读参数量、训练/val 准确率、最优步数、三种子波动、显存与时间。不能把单个种子、某次最优 checkpoint 或受控符号结论推广到所有 ViT/CNN，更不能替代 TexPixel 的真实公式 OCR 评测。

以下数字只填写本次实际运行所得；CPU/MPS/5080 空白表示未做对应设备的正式测量。实测产物统一位于 `outputs/book/ch29/gpu/`，远程日志为 `$LOG_DIR/ch29_*.log`。

## 实测记录

下表均为**本项目观察，RTX 4090（租用）**。机制/训练 fp32、TF32 关闭；真实图统计为原始 PNG 灰度 float32，在该实例 CPU 完成；CUDA用于部分1/3张量及部分5模型。训练每配置 5000 步、batch=128，三个种子 0/1/2；train/val/test 总划分为 15360/1920/1920，低数据量实际使用 1888 张 train。CPU/MPS/5080 正式测量留空。

| 实验与口径 | RTX 4090（租用） | CPU | MPS | 5080 |
|---|---:|---|---|---|
| patch 前向最大绝对误差；3 组形状 | 4.76837158e-06 | | | |
| patch x/w/b 梯度最大绝对误差 | 1.90734863e-06 | | | |
| 位置表 4×6→7×10，CLS 误差；梯度有限 | 0；是 | | | |
| 训练循环峰值 torch allocated（MiB） | 177.717 | | | |
| 冒烟任务墙钟（秒），2880 张符号、4×40 步 | 12.313 | | | |
| 正式任务墙钟（秒），12×5000 步及 256 张真实 PNG | 2055.743 | | | |

真实 val PNG 数 n=256，各 patch 大小仅统计网格，不送入预训练模型：

| patch | 平均空间 token 数 | 平均非空 patch 比例 | CPU | MPS | 5080 |
|---|---:|---:|---|---|---|
| 4 | 1228.210938 | 17.44% | | | |
| 8 | 318.621094 | 31.44% | | | |
| 16 | 85.882812 | 48.63% | | | |
| 32 | 24.589844 | 67.49% | | | |

准确率单位为百分比，均值 ± 三种子样本标准差（百分点）。test 是冻结设置后的独立最终评测，未用于选配置；表中保留全部预先声明的模型与数据量，不按 test 排名挑选。

| 模型 / train 张数 / 参数量 | val | val 平移 +4px | val 32→48 插值 | 最终 test（一次/配置） | CPU | MPS | 5080 |
|---|---:|---:|---:|---:|---|---|---|
| cnn / 1888 / 108744 | 32.26 ± 24.41 | 9.95 ± 5.28 | — | 32.29 ± 23.95 | | | |
| vit / 1888 / 106968 | 36.98 ± 1.07 | 20.49 ± 1.35 | 32.17 ± 1.98 | 35.80 ± 0.33 | | | |
| cnn / 15360 / 108744 | 88.77 ± 1.49 | 29.50 ± 0.68 | — | 90.83 ± 0.57 | | | |
| vit / 15360 / 106968 | 91.27 ± 0.58 | 41.04 ± 1.51 | 69.20 ± 5.44 | 91.35 ± 1.08 | | | |

单配置完整结果（同为本项目观察，RTX 4090 租用，fp32、每配置 5000 步）：

| 配置 | 最优步数（val CE） | train 准确率 | val 准确率 | test 准确率 [组 bootstrap 95%] | 训练循环秒 | 峰值 allocated MiB |
|---|---:|---:|---:|---:|---:|---:|
| cnn_f0.125_s0 | 250 | 99.84% | 47.76% | 48.96% [44.89%, 53.23%] | 101.560 | 177.304 |
| vit_f0.125_s0 | 250 | 93.11% | 36.93% | 35.42% [32.13%, 38.86%] | 111.819 | 113.126 |
| cnn_f0.125_s1 | 1 | 6.09% | 4.11% | 4.84% [3.18%, 6.72%] | 20.017 | 177.717 |
| vit_f0.125_s1 | 250 | 84.69% | 35.94% | 35.94% [32.40%, 39.53%] | 43.941 | 113.126 |
| cnn_f0.125_s2 | 250 | 99.58% | 44.90% | 43.07% [39.01%, 47.03%] | 133.256 | 177.717 |
| vit_f0.125_s2 | 250 | 87.92% | 38.07% | 36.04% [32.29%, 39.38%] | 153.871 | 113.126 |
| cnn_f1_s0 | 1000 | 99.71% | 87.19% | 90.26% [87.86%, 92.45%] | 107.563 | 177.717 |
| vit_f1_s0 | 5000 | 99.37% | 91.87% | 92.03% [90.21%, 93.75%] | 165.228 | 113.126 |
| cnn_f1_s1 | 2000 | 99.97% | 90.16% | 91.41% [89.27%, 93.49%] | 114.088 | 177.717 |
| vit_f1_s1 | 4750 | 98.74% | 90.73% | 90.10% [87.97%, 92.08%] | 302.375 | 113.126 |
| cnn_f1_s2 | 1250 | 99.95% | 88.96% | 90.83% [88.75%, 92.92%] | 347.152 | 177.717 |
| vit_f1_s2 | 4750 | 99.44% | 91.20% | 91.93% [89.90%, 93.80%] | 373.982 | 113.126 |

证据路径：`outputs/book/ch29/gpu/ch29_smoke.log`、`ch29_final.log`；`final/part1.json` 至 `part5.json`；`final/learning_curves.csv`、`final/frozen_validation.json`、`final/final_test.json`。`aggregate.json/.csv` 由每配置结果汇总，标准差用 ddof=1。图见 `final/formula_patches.png`、`final/position_interpolation.png`、`final/learning_curves.png`。


耗时与显存口径：`configs[].seconds` 是每配置训练循环墙钟（含周期 val 与 checkpoint 写盘），不含数据生成、加载后 train/平移/分辨率评估或最终 test；`peak_allocated_mib` 是该循环的 torch 峰值分配，不含 CUDA 上下文/其他进程，也不是整个程序的 NVML 峰值。总任务墙钟取 `run.json` 起止时间，另在台账列出，不能解释为独占设备内核占用时间。

检查点选择的一个实测陷阱：低数据量 CNN 的 seed=1 按最低 val CE 选中了第 1 步，尽管后续 val 准确率更高。这不是读取错检查点，而是置信度失准使 CE 和准确率排序不同。表中如实保留这个结果；因此不能把该组均值单独当作架构能力的证据。查看同种子的完整曲线、选择目标与最优步数，再讨论低数据泛化。若后续改以 val 准确率选 checkpoint，应作为新的评测协议重新冻结，并遵守测试集使用规则。
