# 第 33 章实验手册：VLM 的 LoRA 微调

## 0. 准备

在 `course-v4/` 运行。默认只运行无权重的部分 1、2；位置参数是部分号，每部分重置 seed。默认 CPU、fp32、真实推理 `--n 20`、训练 16 样本/5 步/rank 8。真实模型只读取本地缓存，脚本不下载权重。

```bash
CUDA_VISIBLE_DEVICES="" python -m book.part4_vlm.ch33_vlm_lora.tour
CUDA_VISIBLE_DEVICES="" python -m pytest -q book/part4_vlm/ch33_vlm_lora
# 从已经缓存的完整开源 parquet 提取原始 PNG；不要先截语料再划分。
python -m book.part4_vlm.ch33_vlm_lora.tour \
  --prepare-data --train-n 2048 --n 100 --data /tmp/ch33/data
# 需要已有模型。先用单条真实样本验证前向、反向、保存和重新加载。
python -m book.part4_vlm.ch33_vlm_lora.tour 3 \
  --data /tmp/ch33/data --out outputs/book/ch33/smoke
```

远程推荐命令；运行前检查其他任务的显存占用。单模型串行，正式训练 batch 4 × 梯度累积 2（初期单张冒烟与正式设置区分记录）；像素面积预算 min=16384、max=65536 是本实验设置，不是模型最大分辨率。真实图片处理沿用第 32 章的 PIL processor。下文 `$RUN_DIR` 表示本章产物目录，`$LOG_DIR` 表示日志目录，`$COURSE_DIR` 表示教材副本目录，`$MODEL_CACHE` 表示已有模型的 hub 缓存目录；运行前设置这些变量。所有产物写入 `$RUN_DIR/`，日志写入 `$LOG_DIR/ch33_*.log`。已有数据/权重缓存只读。不得把产物放在会被 `rsync --delete` 清理的代码目录。配套 GPU 脚本读取环境变量 `COURSE_WORK_ROOT`（未设置时默认 `/hy-tmp`）；可用 `export COURSE_WORK_ROOT=/自己的工作根目录` 设置，教材目录、产物和日志分别由 `$COURSE_WORK_ROOT/course-v4`、`$COURSE_WORK_ROOT/runs/...`、`$COURSE_WORK_ROOT/logs` 派生。

```bash
nvidia-smi
cd "$COURSE_DIR"
mkdir -p "$RUN_DIR" "$LOG_DIR"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$RUN_DIR/hf"
export HF_DATASETS_CACHE="$RUN_DIR/datasets"
export MPLCONFIGDIR="$RUN_DIR/mpl"
# --model-cache 指向完整的现有只读模型 hub；数据准备时需另指向原始 parquet 所在 hub。
nohup python -u -m book.part4_vlm.ch33_vlm_lora.tour 3 4 \
  --device cuda --dtype bf16 --model qwen35 --strategy language \
  --model-cache "$MODEL_CACHE" --data "$RUN_DIR/data" \
  --train-n 128 --steps 20 --rank 16 --batch-size 4 --accum 2 \
  --out "$RUN_DIR/qwen35_smoke" > "$LOG_DIR/ch33_qwen35_smoke.log" 2>&1 &
```

MPS 从 `--device mps --dtype fp32 --steps 5 --train-n 16` 开始；5080 从 CUDA 冒烟命令开始再增规模。这些设备没有实测性能数字。CPU 机制部分不需要 transformers/peft/trl；真实部分需要对应依赖。不会为没有跑过的设备填写结果。

**环境：** 远程 Python 3.11、torch 2.14.1+cu130、transformers 5.19.0、peft 0.21.2、trl 1.15.0、accelerate 1.15.0。复用已安装环境，不安装新包。Qwen3.5 的 DeltaNet 使用环境中的 PyTorch 回退，速度受并行任务影响。TRL 的版本接口以实际安装源码为准；[官方 SFT 文档](https://huggingface.co/docs/trl/sft_trainer)说明视觉训练不应截断图片 token，本实现 `max_length=None`，自己的 collator 超出长度直接报错。

**数据与口径：** [OleehyO/latex-formulas](https://huggingface.co/datasets/OleehyO/latex-formulas) 的 cleaned_formulas，数据卡许可 openrail。原图只在实验机缓存，不随日志分发。`miniocr/data/latex_corpus.py` 的完整 latex-v1，seed=0，规范化键整组进入 train/val/test；之后取唯一公式前缀，原始 parquet 首次出现的图片。无渲染替代，不按 val 可识别性过滤。采用第 0 章 §0.9.3 的 OCR 面板精简版，复用第 18 章 `PostCorrectExample` 和 `score_postcorrect`，仅报告 EM_raw、EM_norm、CER、parse_rate；文本后纠错的 broken/fix 指标不适用；本章未报告 CER_token 和渲染一致率。输出先经过 `extract_formula`，val 同时保存 raw。EM_norm 是规范化字符串一致，不是数学等价；CER 可以超过 1；parse_rate 是 mathtext 子集可解析率。比例有 Wilson 95% 区间，CER 有按样本 bootstrap 区间。

只用 val 比较策略、选择配置；按第 0 章 §0.10.1，test 在最终设置冻结后，预先登记的基线和微调在同一轮各评一次。`data/ch33_final_test.lock` 使用排他创建，换输出目录也不能重评。test 只保存汇总和长度/截断统计，不保存预测或错例；不根据 test 再调设置。

## 1. 图文 SFT 与 label mask

`tour 1` 无权重验证左右 padding、图片占位 token、提示和答案。损失内部 shift 一位，验证被遮蔽位置的 logits 梯度为零。真实 collator 使用官方 chat template，分别展开完整对话和 generation prompt，断言提示 token 是完整样本的精确前缀。图片、提示、padding 设为 −100，只监督答案和对话结束符。padding 按 attention_mask 判断，不能把兼任 pad 的 EOS 全部遮蔽。机制位置 `# >>> 练习 33.1`。

## 2. 冻结策略和目标模块

`tour 2` 用缩小模块树检查选择逻辑；真实参数名和数量写入每次运行的 `parameters.json`。`language` 仅语言 Linear 的 LoRA；`connector` 再训练完整视觉 merger（Qwen3-VL 包括 deepstack mergers）；`vision` 再加入视觉塔 Linear 的 LoRA，视觉塔其余参数仍冻结。三者都不是全参数微调。连接器使用 `modules_to_save` 随适配器保存。按实际模块树匹配，覆盖 Qwen3.5 的 `in_proj_qkv` 等 DeltaNet 层，不硬套 q_proj/v_proj 列表。机制位置 `# >>> 练习 33.2`。

## 3. 真实样本闭环与显存

`tour 3` 执行一个真实 batch 的前向、反向、优化、保存、新基座加载。检查每个可训练参数都有有限梯度、冻结参数没有梯度、至少有非零梯度；初始 LoRA A 的梯度可能为零，不能误判。比较保存前后 logits。`smoke.json` 保存 loss、监督 token 数、梯度、重载误差、峰值 allocated/reserved 显存。随后 `tour 4 --steps 20` 测实际 TRL 吞吐，再估算放大训练。

## 4. TRL LoRA 训练

显式选择 `tour 4`，用 `--train-n --steps --batch-size --accum --rank --lr` 放大。有效 batch 是 batch-size × accum（单卡）；步数是优化器步，不是前向次数。使用 AdamW、cosine、warmup、gradient checkpointing；无量化。`training.json` 记录完整配置、loss 曲线、耗时、峰值显存；`adapter/` 是可供第 34 章复用的模型产物。基座冻结不等于前向无需显存，答案 logits 与反向激活也有成本。

## 5. 基线、微调与最终 test

`tour 5 --split val` 是零样本基线；加 `--adapter .../adapter` 评微调。相同数据、prompt、像素预算、greedy、输出上限。默认输出上限 256 token；明确统计截断，不把它解释成模型完整答案的性能。

最终命令需 `tour 5 --split test --final-test --adapter ...`；基线通过 PEFT 的 `disable_adapter()` 同时恢复原始连接器，然后启用适配器评微调。最终锁先记录配置。不能删除锁后偷偷重跑。

## 6. 视觉依赖

`tour 6 --adapter ...` 仅用于 val。对同一队列依次给正确图片、循环错配图片、全白图片。错图 resize 到原图尺寸，空白图保留原图尺寸，使每条样本的视觉 token 预算相同；错图的重采样仍是对照的局限。统计对参考答案的指标和相对正确图片的预测变化数。输出变化不等于图片被正确理解，必须同时看 EM/CER。

## 运行前的预测题

- 只看 loss 下降，能否区分学会 LaTeX 格式与学会读图？
- 为什么不能直接把输入 token 全部复制成 labels？EOS 与 padding 共用 ID 时会发生什么？
- 哪些 Qwen3.5 层没有 q_proj？少选 DeltaNet 目标会遗漏什么？
- 加连接器/视觉塔时，哪个部分的参数、激活和优化器状态会增加？
- 错图与白图的 CER 都变差，是否足以证明每个符号都来自视觉？
- 训练前写下预期策略排序、显存和运行时间；完成后标清哪些预期被实测推翻。

## 怎样读结果

先检查 mask、冻结和保存加载，再读训练曲线；最后对照相同 val 的 EM、CER、截断数与视觉依赖。参数和像素预算是实验设置；结果表仅填写实测，不能由模型大小或其他设备外推。共享 GPU 的任务墙钟包含竞争、CPU 预处理和评测，不是独占 CUDA 内核时间。

实测来源均为 `outputs/book/ch33/gpu/remote_runs/`，原始 JSON 保留冻结评分。

| 设置 | val n | EM_norm | CER | parse | 截断 |
|---|---:|---:|---:|---:|---:|
| qwen35_baseline/val_baseline.json | 100 | 0.030 | 0.207289 | 0.720 | 1 |
| qwen35_main_val/val_finetuned.json | 100 | 0.010 | 2.724192 | 0.700 | 75 |
| qwen3vl_baseline/val_baseline.json | 100 | 0.050 | 0.196735 | 0.760 | 0 |
| qwen3vl_main_val/val_finetuned.json | 100 | 0.100 | 0.136873 | 0.650 | 0 |

正式训练配置来自 `{qwen35,qwen3vl}_main/config.json`：2048 图、300 步、batch 4×2、rank16。Qwen3.5 EOS 修复及已有 val-only 验证见正文 §33.8.4；此表保留原冻结数字。新代码默认逐张评测，训练指定 im_end，生成双 EOS；不改变评分默认行为或已有 test。

## 完整租卡编排与恢复

`gpu_suite.py` 按阶段串行运行子进程，把每个子任务日志分开保存。Qwen3.5 单样本任务要求至少 5000 MiB 空闲，2B 或放大 batch 要求至少 9000 MiB；不足时每 30 秒检查一次并等待；不杀其他任务、不修改系统。子任务成功后才写 `timing.json`。再次运行相同阶段会校验部分号和显式参数，跳过已经完成的任务；参数不同会报错，避免把旧结果当新结果。

```bash
# 第一轮 language 冒烟完成后运行三策略筛选/零样本/预处理对照/2B 冒烟。
nohup env PYTHONDONTWRITEBYTECODE=1 python -u -m book.part4_vlm.ch33_vlm_lora.gpu_suite screen \
  > "$LOG_DIR/ch33_screen.log" 2>&1 &
# screen 完成、根据冒烟估算确认后：先测 batch=4/accum=2 的 20 步，再分别正式训练。
nohup env PYTHONDONTWRITEBYTECODE=1 python -u -m book.part4_vlm.ch33_vlm_lora.gpu_suite main --steps 300 \
  > "$LOG_DIR/ch33_main.log" 2>&1 &
# main 完成，final_selection.json 已冻结；进行视觉依赖，再对预先登记的基线和微调各评一次 test。
nohup env PYTHONDONTWRITEBYTECODE=1 python -u -m book.part4_vlm.ch33_vlm_lora.gpu_suite final \
  > "$LOG_DIR/ch33_final.log" 2>&1 &
```

`screen_selection.json` 只依据相同 40 条 val 的 EM_norm（平局看 CER）选择 Qwen3.5 冻结策略；Qwen3-VL 固定 language 策略。正式训练从原基座重新开始，不接着 20 步模型训练。`final_selection.json` 只依据两个正式模型在相同 100 条 val 的结果选模型。三策略筛选只是小规模配置选择，不能把 20 步的排序宣称为所有数据规模下的结论。

`--audit-data --train-n 2048` 可在 CPU 上对全部训练样本预检，读取真实 processor 而不载模型权重。`summarize.py --root "$RUN_DIR" --out "$RUN_DIR/summary" --logs "$LOG_DIR"` 从落盘 JSON 生成 CSV、loss 曲线、基线对照和视觉依赖图。汇总不自动填任何缺失结果。

## 补充实验：扩大训练，仅使用 val

本轮保留上面的冻结实验及其 test 结论。补充数据与适配器独立放在 `$RUN_DIR/supp/`，日志为 `$LOG_DIR/ch33_supp_*.log`；不导出补充 test，不改旧 selection 或锁文件。`supplement.py` 记录旧冻结文件的 SHA-256，在每个任务后检查一致性，并将 GPU 子任务墙钟合计限制在 9000 秒。PyTorch 分配器上限设为 19 GiB，仍需在启动前看其他任务显存。

数据准备复用 `prepare_images(..., include_test=False)`：完整 latex-v1 划分后取 20,000 条唯一 train 公式和 500 条 val，使用已有开源原图。测试集只参与标准划分排除，不提取或评估。先做 20 步 rank 32 / batch 16 冒烟，再预检全部训练样本，最后从原始 Qwen3-VL-2B 基座重新训练。正式设置为 language LoRA r32、alpha 64、lr 1e-4、cosine、20 步 warmup、bf16、batch 16 × accum 1、3000 步（48,000 次样本呈现，2.4 epoch），像素预算及输出上限沿用原实验。

```bash
# 先检查显存；以下长任务均应使用 nohup，并把日志写到 "$LOG_DIR"/。
python -m book.part4_vlm.ch33_vlm_lora.supplement prepare-smoke
python -m book.part4_vlm.ch33_vlm_lora.tour --audit-data --model qwen3vl \
  --model-cache "$MODEL_CACHE" --train-n 20000 \
  --data "$RUN_DIR/supp/data" --out "$RUN_DIR/supp/audit"
python -m book.part4_vlm.ch33_vlm_lora.supplement main --steps 3000
```

`tour 5` 新增 `--eval-batch-size`（默认 1；补充实验显式设为 4），采用左 padding，逐条在首个 EOS 处截断计数。基线和候选必须使用相同评估 batch；bf16 批量运算可能改变边界 token 的选择，不能假定与旧单张输出逐字一致。CPU 小规模学习者可显式选 1。训练仍由独立的 `--batch-size` 控制。

补充评估保留每条 val 的原始输出、样本 ID 哈希、截断数及指标区间。`summarize_supplement.py --root .../supp` 导出训练 loss / 学习率曲线 CSV、PNG 和汇总；比例使用 Wilson 95% 区间，CER 使用按样本 bootstrap（1000 次），模型差值使用同一 val 样本配对 bootstrap（10000 次，seed=33）。此轮同时改变了数据量、步数、rank、学习率和 batch，不能把提升单独归因于某一个因素；仅有一个训练 seed，也不包含训练随机性的区间。


### 补充实测记录

本项目观察，RTX 4090（租用），bf16；以下仅 val，test 未再使用。CPU/MPS/5080 未实测。

| 记录 | 配置 / 样本数 | RTX 4090（租用）实测 | CPU | MPS | 5080 |
|---|---|---|---|---|---|
| 补充：正式训练 | bf16；20k 图，3000 步，language r32，batch 16 | train_loss 0.272600；训练 5487.965s；峰值 allocated / reserved 6.012 / 6.975 GiB | — | — | — |
| 补充：零样本 | bf16；val 500，eval batch 4 | EM_norm 0.0640；CER 0.1902；parse_rate 0.8440；截断 7 | — | — | — |
| 补充：原方案：2048 图 / 300 步 / r16 / lr 2e-4 / 有效 batch 8 | bf16；val 500，eval batch 4 | EM_norm 0.0960；CER 0.1519；parse_rate 0.7200；截断 3 | — | — | — |
| 补充：扩大方案：20k 图 / 3000 步 / r32 / lr 1e-4 / 有效 batch 16 | bf16；val 500，eval batch 4 | EM_norm 0.1000；CER 0.1237；parse_rate 0.6740；截断 3 | — | — | — |

扩大方案相对原 r16/300 步：EM_norm 的配对区间跨零，未显示确定的改善；CER 的配对区间支持改善。完整 95% 区间、配对差值和局限见 正文 §33.7.8，曲线见 [PNG](../../../outputs/book/ch33/gpu/supp/summary/training_curves.png)。本轮 GPU 子任务墙钟合计 7977.540s（2.2160h）；本地全量测试：876 passed, 1 skipped, 14 warnings in 12.96s。
