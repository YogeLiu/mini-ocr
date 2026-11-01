# 第 21 章实验：用 TRL 做 SFT

## 0. 准备

在仓库根目录执行：

```bash
cd course-v4
CUDA_VISIBLE_DEVICES="" python -m pytest -q book/part3_post_training/ch21_trl_sft
CUDA_VISIBLE_DEVICES="" python -m book.part3_post_training.ch21_trl_sft.tour --device cpu
# 已有本地 Qwen3 权重时的小规模完整流程；真实推理默认 n=20
CUDA_VISIBLE_DEVICES="" python -m book.part3_post_training.ch21_trl_sft.tour 2 3 4 6 \
  --device cpu --n 20 --train-n 32 --steps 2 --out outputs/book/ch21/cpu
# MPS 使用相同命令，改为 --device mps；精度保持 fp32
```

默认只跑第 1 部分，使用随机初始化的小 Qwen，不下载模型。真实模型部分缺权重时明确报错；对应大权重测试在缺权重时 skip。规模、部分号、输出路径与第 20 章一致。CPU 小规模结果只验证流程，不能作为效果结论。

远程 GPU 推荐先执行以下准备和冒烟命令，在远程仓库根目录运行。先设置 `$RUN_DIR` 为本章运行目录、`$LOG_DIR` 为日志目录、`$HUB_CACHE` 为已有 Hub 权重缓存目录，均使用绝对路径。运行结果和新缓存写入本章运行目录，日志写入日志目录，既有 Hub 权重/数据缓存只读。

```bash
nvidia-smi
mkdir -p "$RUN_DIR" "$LOG_DIR"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_CACHE="$HUB_CACHE"
export HF_HOME="$RUN_DIR/cache/hf"
export HF_DATASETS_CACHE="$RUN_DIR/cache/datasets"
export XDG_CACHE_HOME="$RUN_DIR/cache"
export TRITON_CACHE_DIR="$RUN_DIR/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/cache/inductor"
export MPLCONFIGDIR="$RUN_DIR/cache/mpl"
nohup python -u -m book.part3_post_training.ch21_trl_sft.tour 1 2 3 4 \
  --device cuda --dtype bf16 --n 20 --train-n 256 --steps 20 --eval-every 20 \
  --batch-size 2 --accumulation 2 --rank 16 --checkpointing \
  --data-dir "$RUN_DIR/data" --out "$RUN_DIR/smoke" \
  > "$LOG_DIR/ch21_smoke.log" 2>&1 < /dev/null &
```

精度口径：基座、LoRA 参数及 AdamW 状态为 FP32；`--dtype bf16` 表示训练/推理用 BF16 混合精度。机制与 loss/梯度对拍固定 FP32。显存用 PyTorch 本进程 allocated/reserved 峰值，单位 GiB；不是整卡占用。训练墙钟含各阶段保存、val 生成和 NLL 检查，不冒充纯 GPU 内核时间。同时运行的其他实验会影响耗时。

环境：实例已有 Python 3.11、torch 2.14.1+cu130、transformers 5.19.0、trl 1.15.0、peft 0.21.2、accelerate 1.15.0、datasets 5.1.0；没有安装新包。完整版本与代码 SHA-256 写在 `environment-*.json`。这版 TRL 的 fused LM head 使用 Triton；CPU/MPS 使用 SFTTrainer 子类，仅将 loss 改为 Transformers 原生 causal LM loss，数据、优化器及训练循环仍由 TRL/Trainer 管理。CUDA 使用未经覆盖的 TRL `compute_loss`。本地 CPU 对拍不能代替 GPU fused loss 的对拍。

数据：只读取已下载的 [OleehyO/latex-formulas cleaned_formulas](https://huggingface.co/datasets/OleehyO/latex-formulas) 的 `latex_formula` 列，不读取图片、不渲染、不下载新数据。课程登记许可 openrail，来源 arXiv/im2latex。复用 `postcorrect-v1` 默认噪声、`latex-v1 seed=0` 全语料规范化分组划分，再取 train/val/test 的前缀；噪声种子默认 0。训练超过 `--max-length` 的样本整条拒收，val 生成不按这个阈值过滤；NLL 独立报告实际接受数。合成 OCR 噪声不能代表真实 OCR 错误分布。

## 1. 框架与自写机制对拍

```bash
python -m book.part3_post_training.ch21_trl_sft.tour 1 --device cpu
```

使用两条不同长度的微型样本。先把 TRL 的实际 batch 与第 19 章 `collate_sft` 对拍，再比较 loss、每个 LoRA 参数的梯度、一次 AdamW 更新。两个 microbatch 共用有效答案 token 总数作为分母。对拍后恢复参数，避免污染训练初始化。

`miniocr/train/trl_sft.py` 的 `# >>> 练习 21.1` 是框架数据合同检查；`21.2` 是 loss/更新审计。第 19 章的模板/mask、变长整理和第 20 章的 LoRA 机制仍是 A3 要从零实现的部分。调用 TRL 不算完成这些机制题。

## 2. 真实模型检查清单

```bash
python -m book.part3_post_training.ch21_trl_sft.tour 2 --device cuda --rank 16 \
  --train-n 256 --out "$RUN_DIR/audit"
```

读取 Qwen3-0.6B 官方本地快照，非思考模板沿用第 18 章。逐 token 打印并保存第一批 `id/token/label/attention`，提示与 padding 的 label 应为 −100，答案最后的 EOS 应受监督。`completion_only_loss=True` 不足以证明 mask 正确：本实验传入第 19 章生成的完整 labels，显式 `skip_prepare_dataset=True`，禁止框架再次套模板或截断，并对拍整理结果。此接口依据 [TRL 1.15 SFT 文档](https://huggingface.co/docs/trl/v1.15.0/en/sft_trainer) 和实例安装源码核实。

LoRA 只挂 `q_proj/v_proj`，alpha=2×rank，dropout=0；检查全部可训练参数名字包含 `lora_`。真实模型 FP32 loss 对拍选两条最短训练样本，控制显存。对拍 AdamW 两边显式 eps=1e-6：fused 分块与完整词表矩阵乘的舍入差会被接近零的 Adam 分母放大。正式训练使用配置中记录的默认 eps=1e-8；两者口径不可混淆。

## 3. 完整 SFT 与多阶段 val 对照

第 3 部分先评固定提示基线与 copy 基线，然后训练。每 `--eval-every` 步保存 adapter、tokenizer，测同一 val，并对固定 train 前缀（最多 128 条）和 val 计算按 token 加权的 NLL。不同阶段的随机训练 batch loss 不能代替固定探针比较。

输出包括 `sft-config.json`、`trainable-parameters.json`、`data-manifest.json`、`training-first-batch.json`、`trainer-history.json`、`stages.json`、各 checkpoint 的预测与指标。最大化 val **原始输出 EM_norm**，同分最小化 CER，再选较早步数；规则预先固定。raw 与第 18 章抽取后的指标同时保存，不能事后换更好看的口径。

## 4. 保存与重载

第 4 部分加载冻结基座与选定 PEFT adapter，使用选择时完全一致的精度、种子、n、长度和生成 batch。比较 val 的逐条输出，不只比较汇总分数；通过后生成 `reload-verified.json`。移动到另一台机器时保留整个实验目录中的配置/数据身份文件，以及对应基座快照和选定 `step-*` 目录；不要只复制 adapter 权重而丢失模板/生成设置。

## 5. 最终 test 一次

只有已完成重载验证且显式加 `--final-test`，第 5 部分才运行。独占创建 `FINAL_TEST_STARTED.json` 后评一次；失败也不自动重试。只保存汇总指标与置信区间，不保存 test 逐条预测、开发期不查看错例、不依据 test 修改配置。最终报告定稿后才可解封做错例分析（第 0 章 §0.10.1）。test 输入缓存保留用于复现，报告只读取汇总文件。

## 6. 结果表与曲线

```bash
python -m book.part3_post_training.ch21_trl_sft.tour 6 --out "$RUN_DIR/main"
```

只读已有指标，生成 `results.csv` 和 `learning-curves.png`；不触发模型推理或再次评 test。曲线把固定 train/val NLL 与 val EM 分开，防止把 token-level loss 和序列 exact match 当成同一个量。

## 运行前的预测题

1. 如果只设置 completion-only 选项却没有正确 labels，怎样从第一批发现问题？
2. 两个 microbatch 分别监督 2 个和 4 个 token，能否直接平均两个 mean loss？
3. 梯度几乎一样，为什么 AdamW 的个别参数更新仍可能出现更大的相对误差？
4. 训练 NLL 继续下降时，val NLL、EM 与 CER 是否一定同步改善？
5. adapter 重载后的 EM 相同，是否足以证明逐条输出复现？
6. test 比 val 好或差，是否允许重新挑 checkpoint？

## 怎样读结果

先看 mask、EOS、可训练参数和 loss 对拍，再看效果。训练 loss 下降只说明当前监督目标更容易拟合；固定 train 探针下降而 val 指标不升，是本章需要解释的现象，不能拿 test 找理由。EM 的 Wilson 区间、CER 的按样本 bootstrap 区间已由原评测器写入 JSON；小样本点估计波动不等于稳定提升。本章使用第 0 章 §0.9.3 的文本后纠错面板：EM_norm、CER、改坏、fix_rate；完整结果见正文 §21.8.2、§21.8.4。clean/noisy 分桶揭示“修错”与“改坏干净输入”的取舍。

所有下表实测值均标为“本项目观察，RTX 4090（租用）”；CPU/MPS/5080 没有实测同规模实验，留空。冒烟和正式实验的样本数不同，不横向比较它们的百分比。最终数据、日志和报告位于 `outputs/book/ch21/gpu/`。

本项目观察，RTX 4090（租用），bf16 autocast，LoRA rank 32（q/v），800 步，有效 batch 8（microbatch 2 × 累积 4），训练 2048 条（合格 2023），val 200。

| 阶段 | val EM_norm（95% 区间） | val CER | val NLL | 固定 train 探针 NLL |
|---|---|---|---|---|
| 零样本（第 18 章提示，未微调） | 0.000 | 1.062 | — | — |
| 严格 3-shot（第 18 章） | 0.040 | 0.144 | — | — |
| copy（原样输出） | 0.365 | 0.026 | — | — |
| step 200 | 0.640 [0.571, 0.703] | 0.0140 | 0.0327 | 0.0208 |
| **step 400（选定）** | **0.655 [0.587, 0.717]** | 0.0155 | **0.0319** | 0.0091 |
| step 600 | 0.635 [0.566, 0.699] | 0.0155 | 0.0364 | 0.0044 |
| step 800 | 0.640 [0.571, 0.703] | 0.0195 | 0.0349 | 0.0025 |

- 选型规则（运行前固定）：val 原始 EM_norm 降序，CER 升序，步数升序。选中 step 400。
- 重载验证：`reload-verified.json`，逐条输出与 step 400 完全一致（`predictions_identical=true`）。
- 冻结单次 test（`5 --final-test`，n=200）：EM_norm **0.615 [0.546, 0.680]**，CER 0.0265 [0.0167, 0.0376]。
- 训练 loss 从 0.496 降到 0.009，而 val 在 step 400 后不再改善、NLL 反而回升；这就是本章要观察的"训练下降、验证不一定变好"。

CPU/MPS/5080 没有同规模实测，列留空。


正式运行命令（先确认冒烟结束，再使用新的输出目录）：

```bash
nohup python -u -m book.part3_post_training.ch21_trl_sft.tour 2 3 4 \
  --device cuda --dtype bf16 --n 200 --train-n 2048 --steps 800 --eval-every 200 \
  --batch-size 2 --accumulation 4 --eval-batch-size 8 --rank 32 \
  --checkpointing --strong-baseline --data-dir "$RUN_DIR/data" \
  --out "$RUN_DIR/main" > "$LOG_DIR/ch21_main.log" 2>&1 < /dev/null &
# 等待训练与重载验证完成、冻结选择后才运行；只执行一次
nohup python -u -m book.part3_post_training.ch21_trl_sft.tour 4 5 6 \
  --device cuda --dtype bf16 --n 200 --train-n 2048 --steps 2 --eval-every 100 \
  --batch-size 1 --accumulation 1 --eval-batch-size 8 --rank 32 \
  --final-test --data-dir "$RUN_DIR/data" \
  --out "$RUN_DIR/main" > "$LOG_DIR/ch21_final.log" 2>&1 < /dev/null &
```

`--strong-baseline` 额外运行第 18 章严格格式 3-shot 基线，示例从 train 的固定种子池取出，保存其 ID。零样本与 3-shot 都只看 val；最终 test 只运行所选 SFT 设置。

实测发现的评测细节：已有 `extract_formula` 会删除一个句末标点，而语料中的部分公式保留了句末 `.`。因此，某些原样输出正确的公式经抽取后反而不再 exact match。这是既有启发式的限制；本章保留接口和两套结果，主指标预先固定为原始输出的 EM_norm。不能看到这个现象后只报告更有利的一列。

第二条正式命令按 `outputs/book/ch21/gpu/environment-4-5-6.json` 还原；其中 steps/batch 是该次 CLI 的默认训练参数，此调用没有部分 3，不会重新训练。既有 test 不得再次执行。训练与重载的代码哈希不同，当前源码从头训练能否得到同一 adapter 未验证，见正文 §21.7.4。
