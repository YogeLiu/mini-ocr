# 第 35 章实验手册：蒸馏与合成数据

## 0. 准备

在 `course-v4/` 运行。位置参数为部分号，可单独运行；默认只执行不需要权重的第 1 部分，CPU、fp32。真实推理默认 `--n 20`；训练默认 5 步、rank 8、梯度累积 1。日志为逐行 JSON，结果写入 `--out`。实现参照第 32–34 章的 VLM 加载与训练接口。

```bash
CUDA_VISIBLE_DEVICES="" python -m book.part4_vlm.ch35_distill.tour
CUDA_VISIBLE_DEVICES="" python -m pytest -q book/part4_vlm/ch35_distill
# 已有开源 parquet 时准备小规模数据；不下载，不回退到渲染数据。
python -m book.part4_vlm.ch35_distill.tour \
  --prepare-data --data /tmp/ch35/data --out /tmp/ch35/prep
# 已有模型缓存时：教师生成与规则质检。
python -m book.part4_vlm.ch35_distill.tour 2 3 \
  --n 20 --data /tmp/ch35/data --out /tmp/ch35/teacher --teacher-dir /tmp/ch35/teacher
# 原始数据学生；随后改 arm=distill、out=/tmp/ch35/distill 跑另一组。
python -m book.part4_vlm.ch35_distill.tour 4 \
  --arm original --n 20 --data /tmp/ch35/data --teacher-dir /tmp/ch35/teacher --out /tmp/ch35/original
```

MPS 将命令改为 `--device mps --dtype fp32`。5080 用 `--device cuda --dtype bf16`，先跑 20 步；这里只记录租用 4090 的数字。CLI 提供 `--n-train`、`--n-teacher`、`--steps`、`--accum`、`--rank`、`--lr`、`--max-new-tokens`、`--max-pixels`、`--scale`、`--render-tolerance`、`--temperature`、`--kl-steps`、`--audit-n`、`--seed` 放大或控制实验。

远程命令如下。先设置路径变量：`$CODE_DIR` 是教材副本目录，`$RUN_DIR` 是本章产物目录，`$MODEL_CACHE` 是已缓存模型的目录，`$LOG_DIR` 是日志目录，`$TEACHER_ADAPTER` 是第 33 章教师适配器目录。模型缓存和原始 parquet 只读使用。所有新产物、派生图片、matplotlib 缓存和检查点写在 `$RUN_DIR`；不要把产物放进会被 `--delete` 同步的代码目录。配套 GPU 脚本读取环境变量 `COURSE_WORK_ROOT`（未设置时默认 `/hy-tmp`）；可用 `export COURSE_WORK_ROOT=/自己的工作根目录` 设置，教材目录、产物和日志分别由 `$COURSE_WORK_ROOT/course-v4`、`$COURSE_WORK_ROOT/runs/...`、`$COURSE_WORK_ROOT/logs` 派生。`CH35_MODEL_CACHE` 仍可覆盖 `finish_gpu.sh` 的默认模型缓存。

```bash
nvidia-smi
cd "$CODE_DIR"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export MPLCONFIGDIR="$RUN_DIR/mpl"
python -u -m book.part4_vlm.ch35_distill.tour --prepare-data \
  --n-train 256 --n-teacher 192 --n 64 --data "$RUN_DIR/data" --out "$RUN_DIR/prep"
# 先完成这个 20 步阻断检查，再开始主实验。
nohup python -u -m book.part4_vlm.ch35_distill.tour 1 4 \
  --device cuda --dtype bf16 --n 4 --n-train 256 --steps 20 --accum 4 --rank 16 \
  --max-new-tokens 64 --model-cache "$MODEL_CACHE" --data "$RUN_DIR/data" \
  --out "$RUN_DIR/smoke" > "$LOG_DIR/ch35_smoke.log" 2>&1 &
# 确认 smoke/run_end.json 后再运行：
nohup python -u -m book.part4_vlm.ch35_distill.tour 2 3 \
  --device cuda --dtype bf16 --n 192 --max-new-tokens 192 \
  --model-cache "$MODEL_CACHE" --data "$RUN_DIR/data" \
  --out "$RUN_DIR/teacher" --teacher-dir "$RUN_DIR/teacher" \
  > "$LOG_DIR/ch35_teacher.log" 2>&1 &
# 确认 teacher/run_end.json 后执行；顺序跑两组 600 步、val 选择、一次最终 test。
nohup bash book/part4_vlm/ch35_distill/finish_gpu.sh > "$LOG_DIR/ch35_pipeline.log" 2>&1 &
```

环境：Python 3.11、torch 2.14.1+cu130、transformers 5.19.0、peft 0.21.2。未安装新包、未修改系统设置。教师为 [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)，学生为 [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)，模型卡均标 Apache-2.0。只读离线权重；脚本不自动下载。Qwen 静态图 processor 复用第 32 章适配器，避免创建未用的视频后端。Smol 加载时有上游配置 pad_token_id 警告，生成显式使用 tokenizer.pad_token_id；实际冒烟已通过。

数据来自 [OleehyO/latex-formulas](https://huggingface.co/datasets/OleehyO/latex-formulas) 的 `cleaned_formulas`，许可 openrail；本次实际读取 552,340 条语料。先在完整语料上执行 `latex-v1` 标准分组划分，再按规范化键去重取前缀。训练限制参考公式不超过 256 字符，base 256、teacher 192，键互不重叠；val/test 各 64，不按长度、能否解析或教师表现筛选。manifest 保存 parquet 行号、规范键 SHA256、原图 SHA256，参考文本与原始 PNG 一一对应。数据仅用于本地实验，不另行分发原始数据集。

口径：模型前向 bf16，LoRA 参数由 PEFT 管理；KL 机制用 fp32。时间是共享 GPU 的进程墙钟，包含 CPU 整理/生成，不是独占 GPU 内核时长。峰值是 PyTorch allocated，不能等同于 nvidia-smi 的进程显存。EM 使用现有规范化；CER 为全语料编辑数/参考字符数。复用第 18 章 `PostCorrectExample` 与 `score_postcorrect`，使用第 0 章 §0.9.3 的 OCR 面板精简版，报告 EM_norm、CER、parse_rate 和截断条数，评测 JSON 另有 EM_raw；不报告 CER_token 和渲染一致率。训练伪标签的渲染质检不属于评测面板。EM/parse 用 Wilson 区间，CER 用固定 seed 的 500 次 bootstrap。

## 1. 分布蒸馏机制（部分 1）

`miniocr/train/distill.py` 保留练习 35.1：计算 `T² × mean_token KL(teacher || student)`，教师 detach，prompt/pad 不计入分母。演示用相同词表的随机 logits，8×6 个位置、11 个类别、前两位置 mask，50 步优化。这是概率机制验证，不声称真实 VLM 学会 OCR。Qwen 与 Smol 的词表不同，直接逐坐标 KL 没有意义；真实模型对照使用序列蒸馏。

## 2. 教师造标签（部分 2）

教师只看训练图片与固定 OCR 提示，不看参考标签。开源 PNG 下采样到 0.75 倍后恢复尺寸，模拟轻度分辨率损失；不自己渲染训练图。教师像素预算 65,536–131,072，greedy、batch 1、最多 192 新 token。原始输出、抽取公式、截断标记、模型快照、耗时逐条保存。截断预测保留在整体评测分母，不能悄悄删掉。

## 3. 规则质检与人工抽样（部分 3）

练习 35.3 依次检查非空/长度 ≤384、未截断、mathtext 可解析。参考与预测都可渲染时比较同一个渲染器的二值墨迹，异或/并集 ≤0.65 通过；参考不受 mathtext 支持时记 unknown。`quality.json` 保留每条理由和 pixel_diff；`audit.png` / `audit_samples.json` 按固定前缀展示前 6 条通过和前 6 条拒绝。

这个质检借助**训练参考标签**做渲染一致性检查，是有参考的标签增强实验，不能冒充完全无标注蒸馏。比较的是参考/预测的同渲染器输出，并非原 PNG 的像素真值。宽松阈值会放过错误，mathtext 也会拒绝合法 LaTeX；unknown 不是正确。未对 val/test 使用参考质检，也未把拒绝样本从评测中删掉。

## 4. 学生对照与错误传递（部分 4）

两组均从同一个 Smol 基座重新初始化 LoRA，seed 35、rank 16、alpha 32、dropout 0、lr 2e-4、AdamW weight_decay 0.01、梯度裁剪 1、microbatch 1、梯度累积 4、600 优化步。不训练视觉塔，只对文字模型 q/k/v/o 投影加 LoRA。单图最长边 512、关闭切图。练习 35.2 用完整模板和生成前缀做 token 级一致性校验，只训练答案及 EOS，模型内部负责 causal shift。

`original` 只用 base 原始标签；`distill` 加入质检通过的独立训练图及教师标签。两组总优化步数/有效 batch 相同；混合组的原始数据曝光会减少，`exposures` 如实记录。这个对照同时改变数据量、标签来源和图片退化，不能单独归因为教师知识；若后续追究机制，可追加同图人工标签/原图伪标签对照。本轮不依据 test 追加实验。

`error_transfer.json` 固定抽取至多 12 个“质检通过但与参考不匹配”的训练样本，记录两组学生是否完全复制教师错误。它是训练诊断，不是留出泛化指标；复制率为零也不表示教师没有造成较细粒度的错误。规范化字符串不匹配仍可能是等价写法，不能把全部 mismatch 都当作已确认的语义错误，须结合抽样图检查。

## 5. 冻结与最终 test（部分 5）

`summarize --select` 只读取 val，选择 CER 最低者；同分看 EM，再同分选 original。写入选中 adapter SHA256 和选择时间后，`--final-test --adapter ...` 才能评 test；part5 从 adapter 的实验根目录读取 `selection.json`，验证所选臂路径与权重 SHA256。test 使用原图，与 val 完全相同生成配置；在同一 `--out` 落盘一次性锁（换 out 仍能绕过锁），不输出 test 错例、不据此改设置。报告单独标记最终 test，不能把它与 val 合并。

## 运行前的预测题

1. 温度升高却忘记乘 T²，会如何改变学生梯度规模？把 KL 方向调换会得到同一个损失吗？
2. 可解析但把 `x^2` 识别成 `x_2` 的教师输出，会通过哪些规则？为什么参考不可渲染要记 unknown？
3. 固定 2,400 次样本曝光，加入伪标签后 base 样本的曝光减少：若 CER 变差，有哪些解释？
4. 教师输出被长度上限截断后，只评未截断样本会怎样偏移结论？
5. 为什么 val 可以看错例、决定最终模型，而 test 不可以参与这一轮选择？

## 怎样读结果

先读 mask、trainable、finite gradient 与 20 步冒烟，再看教师整体/通过子集的 EM、质检保留率与 unknown 数。对照 `audit.png` 查看错误是否被解析率掩盖。训练 loss 只说明拟合监督目标，不代替 val；用 `comparison.png` / `metrics.csv` 比较相同步数下 CER、EM 与截断数。最后读训练错误复制诊断和独立的最终 test 区间。单种子、小规模结果不支持“蒸馏必然提升”的结论。

实测记录见下表（来源：`outputs/book/ch35/gpu/runs/`）；CPU/MPS/5080 空白表示未测，不填估算值。

| 主实验 | n | EM_norm | CER | parse | 截断 | 来源（runs/ 下） |
|---|---:|---:|---:|---:|---:|---|
| 零样本 Qwen3-VL 教师（未加载 ch33 LoRA） | 192 | 0.09375 | 0.138381 | 0.848958 | — | teacher/teacher.json |
| original val | 64 | 0 | 0.926403 | 0.421875 | 11 | original/original_val.json |
| distill val | 64 | 0 | 1.034649 | 0.59375 | 16 | distill/distill_val.json |

`teacher/quality.json`：109/192 通过，总拒绝83；73条unknown中43条通过。教师整个子任务892.077s（`teacher/run_end.json`），生成阶段879.440s（`teacher/teacher.json`）；两臂训练731.9/712.0s（各臂`training.json`）。主实验生成上限192（各臂`run_start.json`、`final/frozen.json`），test冻结结果见 `outputs/book/ch35/gpu/runs/final/final_test.json`，不重算。

## 补充实验（val-only，test 未再使用）

本次补充沿用 latex-v1 的 train/val 划分，`--val-only --prepare-data` 不提取 test 图片或写出 test manifest；`--val-only` 拒绝第 5 部分及 `--final-test`。旧 selection、test 与锁文件保持冻结。本项目观察，RTX 4090（租用），bf16：教师使用 ch33 的 `$TEACHER_ADAPTER`，`--teacher-adapter` 加载后合并到基座再生成。新参数 `--batch-size` 只控制推理批量，采用左 padding，不改变训练 microbatch。

补充规模为 base 5000、teacher 5000（两者规范键不重叠）、val 500（无长度筛选）；训练图仍限制参考文本 ≤256 字符。教师 scale=1、像素预算 65536、batch 16、最多 192 新 token。两臂均从 SmolVLM 原基座初始化，语言注意力 LoRA rank 64，lr=2e-4、AdamW、梯度裁剪 1、microbatch 1 × accum 2、3000 优化步；seed=35、36。每臂评相同的 500 条 val，batch 16，greedy、192 新 token，截断保留在分母。它与旧实验的步数、rank、曝光、推理上限均不同，不能将跨轮差异解释成单一因素的效果。

质检复用当前 `quality_check`：长度、预测可解析、未截断，且参考可渲染时要求渲染匹配；参考无法渲染记 unknown 并保留。这是用已有训练参考做筛选的模拟伪标签实验，不等同于完全无标签的数据获取。parse_rate 是 mathtext 子集支持率，不是通用 LaTeX 编译率。原始臂与蒸馏臂的总曝光均为 6000，但后者部分曝光分配给伪标签，须结合 `exposures` 阅读。

```bash
# 先检查 nvidia-smi；以下均 nohup 后台运行，冒烟完成后再启动正式流水线。
nohup bash book/part4_vlm/ch35_distill/supp_smoke.sh > "$LOG_DIR/ch35_supp_smoke.log" 2>&1 &
nohup python -u -m book.part4_vlm.ch35_distill.supp_suite > "$LOG_DIR/ch35_supp_pipeline.log" 2>&1 &
# 正式实验完成后，在相同远端环境用 CPU 生成配对区间；再拉回产物。
CUDA_VISIBLE_DEVICES="" python -m book.part4_vlm.ch35_distill.supp_summary "$RUN_DIR/supp"
```

正式流水线串行执行，检查可用显存，限时 8700 秒（另为已完成冒烟留预算），超时仅终止其直接启动的子进程。无需安装包或下载模型。所有远端新产物在 `$RUN_DIR/supp/`，日志 `$LOG_DIR/ch35_supp_*.log`。EM/parse 使用 95% Wilson 区间，CER 使用固定 seed=0 的 500 次样本 bootstrap；每个训练 seed 的两臂差值用同样本配对 bootstrap（2000 次，seed=35）。这些区间不包含训练随机性的总体不确定性，两个 seed 只作重复性检查。

| 记录 | RTX 4090（租用），本项目观察 | CPU | MPS | 5080 |
|---|---|---|---|---|
| 补充：教师冒烟 | bf16、48 张、batch 16，生成 11.456 s；进程峰值 allocated 4.454 GiB | | | |
| 补充：学生冒烟 | bf16、rank 64、30 步、accum 1，训练 5.494 s；32 条 val 31.569 s；进程峰值 allocated 0.867 GiB | | | |
| 补充：教师正式 | bf16、5000 张、batch 16；3115 条通过，EM_norm 0.1302、CER 0.0925、parse_rate 0.7412；含加载/质检墙钟 1930.0s | | | |
| 补充：共享负载学生冒烟 | bf16、rank 64、30 步、accum 2；训练 14.247s、val 8 条 8.854s；进程 27.985s | | | |
| 补充：original / seed 35 | bf16、rank 64、3000 步、accum 2、val 500；EM_norm 0.0020、CER 0.8408、parse_rate 0.6580；训练 1173.7s、val 270.8s | | | |
| 补充：distill / seed 35 | bf16、rank 64、3000 步、accum 2、val 500；EM_norm 0.0000、CER 1.0063、parse_rate 0.6540；训练 1214.5s、val 284.0s | | | |
| 补充：original / seed 36 | bf16、rank 64、3000 步、accum 2、val 500；EM_norm 0.0020、CER 0.8644、parse_rate 0.6140；训练 1152.4s、val 272.5s | | | |
| 补充：distill / seed 36 | bf16、rank 64、3000 步、accum 2、val 500；EM_norm 0.0020、CER 0.8614、parse_rate 0.7480；训练 1106.2s、val 273.9s | | | |

补充共 7904.628s（2.196 GPU 小时，共享卡子任务墙钟，含冒烟与评估）。完整区间、配对差值、质检抽样和局限见正文 §35.8.8、`outputs/book/ch35/gpu/supp/summary.json` 和 `outputs/book/ch35/gpu/supp/teacher/audit_review.json`。本轮未追加全参数或 rank 消融；未再评 test。

补充观察：未获得稳定 OCR 收益。seed 35 的蒸馏 ΔCER=+0.1655（配对 95% CI 0.1079–0.2245），seed 36 为 −0.0030（−0.0585–0.0553）；两种子 EM 均接近零。seed 36 的解析率提升不能替代 OCR 准确率证据。
