# 第 20 章实验：LoRA

## 0. 准备

在仓库根目录执行：

```bash
cd course-v4
CUDA_VISIBLE_DEVICES="" python -m pytest -q book/part3_post_training/ch20_lora
python -m book.part3_post_training.ch20_lora.tour --device cpu
# 有本地 Qwen3 权重时，小规模完整流程（真实推理默认 n=20）
python -m book.part3_post_training.ch20_lora.tour 4 5 6 --device cpu --n 20 --train-n 32 --steps 2
```

默认只跑 1/2/3/7，CPU 可完成，不下载权重。MPS 将 `--device cpu` 改为 `--device mps`，使用 fp32。没有 PEFT 时第 3 部分明确跳过；没有权重时真实模型部分明确跳过。CPU 的小规模训练只证明流程能运行，不能证明效果。

租用 GPU 推荐命令（在远程仓库根目录执行）：`$RUN_DIR` 是本章输出、缓存和 checkpoint 的目录，`$LOG_DIR` 是日志目录；先把下面的占位路径换成实际路径。

```bash
# 先看其他进程的显存占用；余量不足时等待，不终止他人的进程
nvidia-smi
# 所有输出、缓存、checkpoint 写到 ch20 专属路径
export RUN_DIR="/path/to/runs/ch20"
export LOG_DIR="/path/to/logs"
mkdir -p "$RUN_DIR" "$LOG_DIR"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$RUN_DIR/mpl"
export XDG_CACHE_HOME="$RUN_DIR/cache"
# 先做 20 步冒烟
nohup python -u -m book.part3_post_training.ch20_lora.tour 1 2 3 4 5 7 6 \
  --device cuda --dtype bf16 --n 20 --train-n 256 --steps 20 \
  --batch-size 2 --accumulation 2 --ranks 4 \
  --out "$RUN_DIR/smoke" > "$LOG_DIR/ch20_smoke.log" 2>&1 &
# 正式参数与结果见下方实测记录；必须等待冒烟完成
```

环境：使用实例已有 Python 3.11、torch 2.14.1+cu130、transformers 5.19.0、PEFT 0.21.2。未安装新包。PEFT 通过公开 `LoraConfig/get_peft_model` 创建对照，不依赖其内部 Linear 构造接口。bitsandbytes 不可用，所以 QLoRA 仅演示思路。加载既有 Hub 缓存只读，不向其中下载或写新数据。

精度口径：基座、LoRA 参数与 AdamW 状态均为 FP32；`--dtype bf16` 表示训练和生成用 BF16 autocast，交叉熵用 FP32。机制和合并对拍用 FP32。此配置不是“全模型权重都存 BF16”。显存单位 GiB=2^30 字节，记录 PyTorch 本进程 allocated/reserved 峰值；它们不等于 `nvidia-smi` 的整卡占用。吞吐只除训练步耗时，另记生成、数据准备与总墙钟时间。并行任务会影响时间，不能据此给其他 GPU 排名。

数据：复用 `miniocr/data/postcorrect.py` 的 postcorrect-v1，与 `latex_corpus.py` 的 latex-v1 完整语料分组划分（seed=0，val/test 约各 5000 条）一致。只从已有 OleehyO/latex-formulas cleaned_formulas 读取公式文本，不读取图片、不渲染图片、不下载其他数据。按规范化键分组以后才取前缀，噪声种子固定为 0。原数据许可 openrail，来源 arXiv/im2latex；实验只使用本地缓存，不重新分发语料。合成噪声不能代表真实 OCR 错误分布。

训练使用第 19 章的 `encode_sft/collate_sft/train_sft`，提示和 padding label=-100，答案末尾监督 EOS，causal shift 仅一次。长度超过上限的训练样本整条拒收，报告实际接受数。生成评测不按训练长度过滤，保留相同 val 前缀；NLL 单独标注通过长度过滤的样本数。贪心生成、非思考模板、无采样；按第 0 章 §0.9.3 的文本后纠错面板报告 EM_norm、CER、改坏率和 fix_rate，保留 raw 和沿第 18 章抽取后的两套指标。

## 1. 零初始化与 LoRA 公式

运行部分 `1`。`miniocr/nn/lora.py` 的练习 20.1 保留实现入口：A 为 `[r,in]`，B 为 `[out,r]`，增量为 `(alpha/r) BA`。验证初始输出等于基座、第一步 A 梯度为零且 B 梯度非零；随后用非零 B 检查合并。不是用 B=0 的平凡情况证明合并正确。

## 2. 参数量对拍

运行 `2 --ranks 4 16 64 --targets 'q_proj,v_proj;q_proj,k_proj,v_proj,o_proj'`。这一部分使用随机初始化的小型 Qwen 架构，不装载大权重；实际 Qwen3-0.6B 的统计在部分 4 和 6。数学结论：单层新增可训练参数 `r(in+out)`，多层求和；alpha 不改变参数数目，rank 线性改变它；冻结 bias 不计入。比较目标模块时要考虑 GQA 的 k/v 输出维度。

## 3. 与 PEFT 对拍

运行 `3`。两边采用相同 rank/alpha、无 dropout、冻结 bias，复制相同 A/B；B 特意设为非零。对拍前向、A 梯度、B 梯度和输入梯度，用 CPU/GPU FP32 容差检查。PEFT 只作为验证参考；训练始终使用自写 LoRA。

## 4. 挂载真实 Qwen 与合并

运行 `4`。挂载 Qwen3-0.6B 的 q_proj/v_proj，将 B 设为小的非零随机值，比较完整 logits；练习 20.2 将每个层替换成独立冻结 Linear，第二次合并不会重复加入增量。FP32 运算顺序不同会产生舍入误差，不能要求逐 bit 一致。此处验证随机非零 adapter 的机制；训练 checkpoint 仍保留为 adapter 便于重载。

## 5. 显存先估后测

运行 `5`。每次先计算并保存权重、梯度、Adam m/v、激活代理和 logits 的估算，再执行两步训练。固定 rank=16、q/v，对比 batch/length=(1,128)、(2,256)、(2,512)。真实参数字节可精确计算；激活代理 `12×层数×batch×length×hidden×计算字节数` 仅是教学估计，不是训练峰值上界。日志保留前向前、前向后、反向后、optimizer 后的 allocated；这些差值还包含类型转换/工作区，不能全叫“激活大小”。优化器状态直接按实际 tensor 统计（含 step 标量）。

LoRA 降低梯度与优化器状态，但前向仍需完整基座；反向传播还要经过冻结层。长序列、大 batch、大词表 logits、注意力与交叉熵工作区都占显存，所以少量可训练参数仍可能 OOM。不要用本章估算替代运行前的余量检查。

## 6. 后纠错 SFT 扫描

运行 `6`。训练前保存 mask/EOS 检查与样本接受数；每组重载同一基座、重设随机种子与数据顺序。正式配置每组实际处理 400×2×4=3200 次样本呈现（数学结论），少于合格训练池 4049 条，即未完整遍历一个 epoch；训练池大小不能当作实际见过的样本数。固定 alpha/r=2，扫描 rank 与目标模块，避免把缩放变化混入 rank 对比。保存逐步 loss、参数量、显存、吞吐、adapter、val NLL 和生成指标。相同 rank 的 q/v 与 q/k/v/o 参数预算不同，这不是等参数量消融。

正式扫描前固定选型规则：val 原始 EM_norm 越高越好，同分按 CER 越低、rank 越低、名字字典序选择。不能根据 test 改步数、模块、提示或抽取规则。20 步 val 冒烟发现抽取器会删除公式的合法尾标点，因此正式扫描采用共用评测的默认原串口径；原始输出与抽取后的结果均保留，不修改第 18 章的评测接口。复制输入基线也保留，若模型低于复制基线需如实报告。

## 7. QLoRA 思路

运行 `7`。4bit 保存冻结基座，计算时反量化，adapter 及状态用较高精度。当前没有 bitsandbytes，未声称做过 QLoRA 训练、速度或显存测量。LoRA 本身不量化基座。

## 8. 最终单次 test

按第 0 章 §0.10.1，选型冻结后，预先登记的候选可各评一次，不据此改任何设置。本章只评由事先固定的选型规则选出的一个配置（正文 §20.11.5）。

先完成 val 选型，再运行 `8 --final-test`，`--out` 指向同一正式实验目录，n、dtype、seed、生成预算、评测批量与长度参数须保持一致。读取 `selection.json` 与选定 adapter；目录内建立 `FINAL_TEST_STARTED.json`，再次调用将拒绝。test 只保存汇总与置信区间，不保存逐条预测。开发期不看 test 错例，最终报告定稿后才可解封做错例分析。该守卫是本目录的防误操作设施，不是跨目录的全局访问控制。

## 运行前的预测题

1. B=0 时，为什么第一步 B 有梯度而 A 没有？把 A/B 都设为 0 会怎样？
2. rank 翻四倍、alpha 翻四倍，各会改变什么？保持 alpha/r 固定消除了哪项混杂因素？
3. q/v 与 q/k/v/o 的可训练参数为何不能简单按目标模块个数推算？
4. batch×length 翻四倍时，权重、Adam 状态、激活与 logits 哪些会增长？
5. train loss 更低、val NLL 更低是否保证贪心生成 EM 更高？
6. 若抽取后 EM 上升但原始 EM 不变，下一步该查看哪类 val 输出？

## 怎样读结果与实测记录

所有表中设备实测标记为“本项目观察，RTX 4090（租用）”；估算不混入实测列。CPU/MPS/RTX 5080 尚未测量的格子留空，不由 4090 外推。完整日志、JSON/CSV、图及报告位于 `outputs/book/ch20/gpu/`。正式运行完成后本节填写实测表。

正式扫描命令（本项目配置）：

```bash
nohup python -u -m book.part3_post_training.ch20_lora.tour 1 2 3 4 5 7 6 \
  --device cuda --dtype bf16 --n 200 --train-n 4096 --steps 400 \
  --batch-size 2 --accumulation 4 --eval-batch-size 8 --ranks 4 16 64 \
  --targets 'q_proj,v_proj;q_proj,k_proj,v_proj,o_proj' \
  --out "$RUN_DIR/full" > "$LOG_DIR/ch20_full.log" 2>&1 &
# 扫描完成、selection.json 冻结以后；相同的生成与评测参数
nohup python -u -m book.part3_post_training.ch20_lora.tour 8 --final-test \
  --device cuda --dtype bf16 --n 200 --eval-batch-size 8 \
  --out "$RUN_DIR/full" > "$LOG_DIR/ch20_final_test.log" 2>&1 &
```

400 步正式训练的预计训练时间由冒烟实测估计（**待验证**），不是已完成训练时间。最终表只引用实际日志；共享实例中另一个任务一度占约 32 GB，因此正式训练从 microbatch 4/累积 2 改为 microbatch 2/累积 4，有效 batch=8 不变；旧进程在训练开始前终止，未产生可选模型。正式扫描仍可用不同 `--out`、规模参数重跑，但不要用反复 test 来挑选结果。

还可用选出的训练 adapter 再验证合并（不接触 test）：

```bash
python -u -m book.part3_post_training.ch20_lora.tour 4 --device cuda \
  --checkpoint-dir "$RUN_DIR/full" --out "$RUN_DIR/trained_merge"
```

下载实测文件后，在本地重画 CSV/图：

```bash
python -m book.part3_post_training.ch20_lora.summarize \
  --root outputs/book/ch20/gpu
```

## 实测记录（RTX 4090 租用，bf16 autocast，val n=200，贪心解码）

本项目观察，RTX 4090（租用），精度 FP32 参数 + bf16 autocast，训练 400 步，有效 batch 8（microbatch 2 × 累积 4），训练 4096 条（合格 4049 条）。完整 JSON、日志与图见 `outputs/book/ch20/gpu/`。

| 配置 | rank / alpha | 目标模块 | 可训练参数 | val EM_norm（95% 区间） | val CER | val NLL | 峰值 allocated（GiB） | 训练秒 |
|---|---|---|---|---|---|---|---|---|
| r4 qv | 4 / 8 | q,v | 573,440 | 0.605 [0.536, 0.670] | 0.0166 | 0.0336 | 5.37 | 243.8 |
| r16 qv | 16 / 32 | q,v | 2,293,760 | 0.630 [0.561, 0.694] | 0.0162 | 0.0308 | 5.40 | 284.2 |
| r64 qv | 64 / 128 | q,v | 9,175,040 | 0.640 [0.571, 0.703] | 0.0141 | 0.0276 | 5.52 | 558.8 |
| r4 qkvo | 4 / 8 | q,k,v,o | 1,146,880 | 0.625 [0.556, 0.689] | 0.0212 | 0.0314 | 7.71 | 590.1 |
| r16 qkvo | 16 / 32 | q,k,v,o | 4,587,520 | 0.645 [0.577, 0.708] | 0.0190 | 0.0283 | 7.71 | 533.9 |
| **r64 qkvo（选中）** | 64 / 128 | q,k,v,o | 18,350,080 | **0.670** [0.602, 0.731] | 0.0136 | 0.0292 | 5.72 | 445.9 |

- 选型规则（运行前固定）：val 原始 EM_norm 降序，CER 升序，rank 升序，名字字典序。选中 `r64-q_proj-k_proj-v_proj-o_proj`。
- 对照：复制输入 val EM_norm 0.365；未微调基座（同一后纠错提示）EM_norm 0.000，parse_rate 0.005，broken 1.0，几乎不能给出可解析公式。
- **冻结后单次 test**（`8 --final-test`，n=200）：EM_norm 0.620 [0.551, 0.684]，CER 0.0275 [0.0163, 0.0414]。val 与 test 的差距在区间内，但 val 选型本身有乐观偏差。
- 抽取后的 test EM_norm 只有 0.195，远低于原始输出的 0.620：沿第 18 章的抽取器会误删多数正确输出，这是评测问题，不是模型能力的结论。
- 峰值显存不随 rank 单调：qkvo 的 r4/r16 为 7.71 GiB，r64 为 5.72 GiB。这是 PyTorch 本进程 allocated 峰值，测量时其他进程也在用显存；未进一步拆分原因，不作结论。

GPU 时长：扫描（部分 6）墙钟 4709.97 秒（约 1.31 GPU 小时，单卡，含评测与 NLL 检查）；冻结 test 约 1 分钟。
