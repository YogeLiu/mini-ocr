# 第 25 章实验：GRPO 与奖励投机

## 0. 准备

在仓库根目录执行：

```bash
cd course-v4
CUDA_VISIBLE_DEVICES="" python -m pytest -q book/part3_post_training/ch25_grpo
# 部分 1：优势、GRPO 损失的裁剪与 KL、奖励探针（CPU，秒级，不需要权重）
CUDA_VISIBLE_DEVICES="" python -m book.part3_post_training.ch25_grpo.tour 1 --out outputs/book/ch25/cpu
```

GPU 部分的起点是第 23 章合并的 SFT 模型（`sft-merged`）。它由第 21 章选定 step 的 adapter 合并得到，`--base` 指向该目录。下文 `$RUN_DIR` 表示你的运行产物根目录，包含第 22、23、25 章的产物；`$LOG_DIR` 表示你的日志目录，执行前请设置这两个变量。远程产物写到 `$RUN_DIR/ch25/`，日志写到 `$LOG_DIR/ch25_*.log`。

```bash
BASE=$RUN_DIR/ch23/main/sft-merged
OUT=$RUN_DIR/ch25/main3
# 阻断检查：20 步冒烟，确认奖励、有效组比例、KL 与显存在可接受范围
python -m book.part3_post_training.ch25_grpo.tour 2 --device cuda --dtype bf16 --base $BASE \
  --reward layered --steps 20 --prompts 8 --group 8 --batch-size 16 --accumulation 4 --out $OUT-smoke
# 主运行：分层奖励
python -m book.part3_post_training.ch25_grpo.tour 2 --device cuda --dtype bf16 --base $BASE \
  --reward layered --steps 100 --prompts 8 --group 8 --batch-size 16 --accumulation 4 --out $OUT
# 有缺陷对照：只看能否解析
python -m book.part3_post_training.ch25_grpo.tour 2 --device cuda --dtype bf16 --base $BASE \
  --reward parse_only --steps 100 --prompts 8 --group 8 --batch-size 16 --accumulation 4 --out $OUT
# 继续 SFT 对照：同样的提示与步数，监督换成标准答案
python -m book.part3_post_training.ch25_grpo.tour 3 --device cuda --dtype bf16 --base $BASE \
  --steps 100 --batch-size 8 --accumulation 1 --out $OUT
# val 生成（n=500）与汇总
python -m book.part3_post_training.ch25_grpo.tour 4 --device cuda --dtype bf16 --base $BASE --run grpo-layered --n 500 --out $OUT
python -m book.part3_post_training.ch25_grpo.tour 4 --device cuda --dtype bf16 --base $BASE --run grpo-parse_only --n 500 --out $OUT
python -m book.part3_post_training.ch25_grpo.tour 4 --device cuda --dtype bf16 --base $BASE --run sft-continue --n 500 --out $OUT
python -m book.part3_post_training.ch25_grpo.tour 5 --n 500 --sft-preds $RUN_DIR/ch22/main/preds-sft.jsonl --out $OUT
```

上述命令展示各步骤参数；共享远程实例执行长任务时必须用 `nohup`，将日志重定向到 `$LOG_DIR/ch25_*.log`，并在启动前检查 `nvidia-smi`。

精度口径：合并 SFT 基座 FP32 参数，LoRA 为 FP32，训练与生成用 bf16 autocast。reference 为关闭 LoRA 的合并 SFT 模型（TRL 的 `beta>0` 时加载）。GRPO 的 KL 系数为 0.04。

## 1. 机制检查

部分 1 检查：

- 组内优势：同组奖励相同则优势为 0；标准化后组内均值为 0。
- 裁剪：优势为正且 ratio 超过 1+ε 时梯度为 0；优势为负且 ratio 低于 1−ε 时梯度为 0。
- KL：策略与参考相同时 k3 估计为 0。

`miniocr/train/grpo.py` 的 `# >>> 练习 25.1` 与 `25.2` 标记优势与裁剪损失。TRL 的 `GRPOTrainer` 使用 `loss_type='grpo'`（按序列 token 数平均），与本章手写损失的序列平均口径一致。**TRL 1.15.0 默认是 `loss_type='dapo'`；本次远程代码显式指定 `grpo`，没有使用默认值。** 历史运行未保存完整 Trainer config，依据为恢复时的远程代码；当前脚本会保存 `train-config.json` 和摘要中的 `loss_type`。手写优势用总体标准差，不能据此声称与 TRL 优势的全部数值细节完全对拍。

## 2. 分层奖励与奖励投机

分层奖励（`layered_reward`，保留正文 §25.4.5 所述的验证器缺陷）：

- 无法解析或为空：0；
- 可解析但有格式问题（解释、代码围栏、美元符、think 块）：0.1；
- 规范化完全一致：1.0；
- 其余：0.1 + 0.4·gain，gain 为相对带噪输入的编辑率改善比例。

有缺陷对照（`parse_only_reward`）：可解析即 1.0。部分 1 的探针展示：单字符 `x`、复制输入、加解释后复制，都能拿到只看解析的满分；分层奖励给它们的分数低于 0.5。

训练时，每步 8 个提示，每个提示采样 8 个输出（温度 1.0），组内标准化优势，共 64 条序列。`frac_reward_zero_std` 是全部奖励相同的组比例（无效组）；有效组比例 = 1 − 该值。

## 3. 继续 SFT 对照

从相同的 800 条候选提示（train offset=4096）出发，均训练 100 步，监督换成标准答案，不使用任何奖励。本次 SFT 编码过滤后保留 795 条，GRPO 使用 800 条；两者不是严格相同的有效样本集，也不是等生成 token / 等算力预算。这回答"奖励训练是否比继续监督更好"，而不是"奖励是否能用"。

## 4. 读结果的顺序

1. 训练日志：有效组比例是否下降（组内全对或全错越多，学习信号越少）；`reward` 是否上升。
2. KL 与 `clip_ratio`：KL 是否持续增大，裁剪是否频繁。
3. val：EM_norm 与配对差的区间（与第 22 章 SFT 预测比较）。
4. 投机统计：复制输入、过短或空输出、能解析但不对的输出数量。有缺陷对照的 reward 可能升高，但 val EM 不一定同步。
5. 长度：GRPO 的 `loss_type='grpo'` 对长短序列的权重不同，要看 completion_length 是否变化。

## 5. 运行前的预测题

- 组内全部答对时，优势为什么是 0？这一步对策略有没有梯度？
- 正优势、ratio 已超过 1.2 时，裁剪为什么切断梯度？它阻止的是哪种更新？
- 有缺陷奖励下，哪一种输出最容易被强化？为什么它的 reward 会高于正确但略有差异的输出？
- 继续 SFT 与 GRPO 用同样的提示、同样的步数，哪一个更依赖标准答案？

## 6. 记录表

本章使用第 0 章 §0.9.3 的文本后纠错面板。下表只列出 EM_norm 和 CER，改坏和 fix_rate 见正文 §25.8.3 的主表。

本项目观察，RTX 4090（租用，48GB），FP32 基座与 LoRA 参数、bf16 autocast；Qwen3-0.6B，第 21 章 step-400 SFT 经第 23 章合并。LoRA r=16（q/v），学习率 5e-6，seed=0，100 步。GRPO loss_type=grpo，KL β=0.04，ε=0.2，800 条候选提示，每步 8 提示 × 8 采样（batch 16 × 累积 4），温度 1.0，输出上限 160 token；继续 SFT batch 8 × 累积 1，编码上限 384，实际保留 795 条。val n=500，贪心生成 batch 16，上限 192 token。

下表原始单种子行的 EM 和配对差均为 1000 次样本级 bootstrap 的 95% 区间、seed=0；配对差为该运行减第 22 章 SFT。reward 和 KL 的“末段”指第 91–100 步均值，有效组取全部 100 步均值。投机统计依次为“复制输入 / 少于 4 字符 / 可解析但不精确匹配”，不互斥，复制干净输入本身不表示投机。

| 运行 | 有效组比例 | reward（末段） | KL（末段） | val EM_norm（95% 区间） | 与 SFT 配对差（百分点） | 投机探针计数 |
|---|---:|---:|---:|---|---|---|
| SFT 起点（第 22 章预测） | — | — | — | 67.8% [63.8, 71.8] | — | 复制 176，其余未汇总 |
| GRPO 分层奖励 | 33.75% | 0.358881 | 0.0000607720 | 68.8% [64.8, 72.8] | +1.0 [0.2, 2.0] | 173 / 0 / 54 |
| GRPO parse_only | 14.875% | 0.548438 | 0.000320395 | 68.6% [64.4, 72.4] | +0.8 [-0.2, 2.0] | 167 / 0 / 61 |
| 继续 SFT | — | — | — | 69.2% [65.0, 73.0] | +1.4 [0.6, 2.4] | 174 / 0 / 53 |
| 补充（3种子）GRPO layered | — | — | — | 68.400% ± 0.400个百分点（SD） | 见§7（对继续SFT） | — |
| 补充（3种子）GRPO parse_only | — | — | — | 68.400% ± 0.200个百分点（SD） | 见§7（对继续SFT） | — |
| 补充（3种子）继续 SFT | — | — | — | 69.133% ± 0.115个百分点（SD） | 见§7（对继续SFT） | — |

补充行是均值 ± 样本标准差，不是原始单种子行的95%区间。

| 运行 | val CER | 训练循环秒数 | val 生成秒数 | 训练 peak allocated（GiB） | CPU | MPS | RTX 5080 |
|---|---:|---:|---:|---:|---|---|---|
| GRPO 分层奖励 | 0.0249764 | 1499.578 | 189.099 | 13.368 | | | |
| GRPO parse_only | 0.0247809 | 1495.130 | 188.309 | 13.379 | | | |
| 继续 SFT | 0.0297940 | 42.083 | 310.687 | 未记录 | | | |

GRPO 分层/parse_only 的平均输出长度为 58.993/58.988 token，达到训练长度上限的比例为 3.047%/3.063%；策略 ratio 裁剪比例均为 0。首 10 步到末 10 步 reward 均值分别为 0.301649→0.358881、0.462500→0.548438；每步提示变化，不能将这条曲线解释为固定题集上的净改善。奖励定义不同，也不能直接比较两个 reward 的大小。parse_only 的静态探针确实能被单字符输出骗过，但本次 val 没有观察到短输出崩塌，不能声称已训练出大规模投机。

本次按最高 val EM_norm 选中继续 SFT（相比两个 GRPO 臂仅多答对 2/3 条），CER 则是 parse_only 最低；没有把这种小差异宣称为方法普遍优越。冻结记录见 `outputs/book/ch25/gpu/runs/ch25/main3/frozen-selection.json`。本章只登记了选中的继续 SFT 一个候选，对它执行一次 test（正文 §25.8.5）：

```bash
python -m book.part3_post_training.ch25_grpo.tour 4 --final-test --device cuda --dtype bf16 \
  --base $RUN_DIR/ch23/main/sft-merged --run sft-continue --n 500 --out $RUN_DIR/ch25/main3
```

最终 test（继续 SFT，n=500，唯一一次）：EM_norm **69.8%（349/500）**，Wilson 95% 区间 **[65.637%, 73.661%]**；CER **0.0280480**，bootstrap 95% 区间 **[0.0172289, 0.0434295]**；改坏 **6.4%（10/156）**，fix_rate **59.0%（203/344）**。生成 206.672 秒，任务墙钟 260.782 秒。按第 0 章 §0.10.1，选型冻结后，预先登记的候选可各评一次，不据此改任何设置；开发期不看 test 错例，最终报告定稿后才可解封。`--final-test` 单独输出 `preds-test.jsonl` 与 `test-metrics.json`，已有启动记录时拒绝重复执行。

数据沿用 OleehyO/latex-formulas cleaned_formulas（项目记录许可 openrail）和 latex-v1 规范化键分组划分，postcorrect-v1 合成 OCR 噪声；它不是实际 OCR 系统输出。解析检查使用 mathtext 子集，不等同于完整 TeX 编译；中文字体缺失警告保存在日志，不能据中文渲染比较下结论。本次没有安装依赖或下载新模型。远程 torch 2.14.1、transformers 5.19.0、trl 1.15.0、peft 0.21.2。

20 步冒烟只验证执行与资源（8 条重复提示，不能作泛化证据）；早期中断和旧数据缺陷运行全部排除。最终三臂 main3 已完整保留，未重训。训练曲线与逐步 CSV、汇总 JSON、原日志保存在 `outputs/book/ch25/gpu/`，完整解释见本章正文 §25.8。本地全量测试：868 passed，1 skipped，14 warnings（13.15 秒）。

耗时口径：正式三臂训练与 val 加最终 test，任务墙钟合计 4294.592 秒（1.192942 小时）；含旧冒烟为 1.332532 小时。并发重叠未扣除、包含 CPU 准备，非独占 CUDA 内核时长。计入已完成废弃任务，已知历史总量至少 2.518602 任务小时；更早中断耗时不完整，不能声称全流程低于两小时。

## 7. 补充实验：固定 val 的训练种子复现

补充运行与冻结选型独立：不更新 selection/锁文件，不再使用 test。`--seed` 控制训练随机性和数据顺序；新增 `--data-seed` 独立控制 postcorrect-v1 合成噪声，省略时保持原先沿用 `--seed` 的行为。为只测训练种子波动，训练与评测均指定 `--data-seed 0`，训练种子为 1、2；seed=0 复用 main3，不重训。val 仍为原有 latex-v1 的 500 条，按稳定 ID 对齐。

复现命令沿用第 0 节配置，将 `--out` 分别改为 `$RUN_DIR/ch25/supp/seed1` 和 `seed2`，训练加 `--seed 1 --data-seed 0` 或 `--seed 2 --data-seed 0`；部分 4 统一 `--seed 0 --data-seed 0 --n 500`。先在独立 `supp/smoke` 跑 20 步。完整命令与预算队列见 `outputs/book/ch25/gpu/supp/run_supp.py`，只调用部分 2、3、4。每条 100 步 GRPO 仍为 800 条候选、G=8、microbatch16×累积4；继续 SFT batch8×累积1。

均值 ± 标准差使用三种子的样本标准差（ddof=1）。方法间先在每个种子计算同批 val 的 EM 差，再以三个差值给出 95% Student t 区间（df=2）；另报固定三种子下、共享 val 样本重采样的配对 bootstrap，以及同时重采样种子和样本的探索性区间。三种子仍很少，不能把 3×500 个预测当作 1500 个独立样本；t 区间依赖种子差值近似正态假设。

本项目观察，RTX 4090（租用），FP32 参数 + bf16 autocast，100步/臂，val n=500/臂/种子。三种子的 EM_norm：layered 为68.8%、68.0%、68.4%；parse_only 为68.6%、68.4%、68.2%；继续SFT为69.2%、69.0%、69.2%。均值±标准差见§6新增“补充”行。

| 补充配对方向 | 平均差（百分点） | 种子配对 t 95%区间 | 种子×样本 bootstrap 95%区间 |
|---|---:|---|---|
| GRPO layered − GRPO parse_only | +0.000 | [-0.861, +0.861] | [-1.000, +1.000] |
| GRPO layered − 继续 SFT | -0.733 | [-1.492, +0.026] | [-1.667, +0.067] |
| GRPO parse_only − 继续 SFT | -0.733 | [-1.307, -0.160] | [-1.733, +0.133] |

**判断：layered 未拉开差距。** 它与 parse_only 的三种子均值相同，种子配对区间为 [-0.861, +0.861] 个百分点。继续 SFT 平均领先两臂 0.733 个百分点：在固定这500条val、未校正的种子配对t区间口径下，对 parse_only 的优势区间为 [+0.160, +1.307]，提示超过本次观测的种子波动；对 layered 则为 [-0.026, +1.492]，仍含零。固定三种子的样本区间对 layered 排除零、对 parse_only 触及零；同时重采样种子和样本时，两组区间均含零。因此有继续 SFT 小幅 EM 优势的探索性证据，但尚不足确立跨种子与样本的稳定方法优势。均值相同也不是两种奖励等效的证明。 三组区间未经多重比较校正；仅3个种子，正态假设与bootstrap均有明显局限。

20步冒烟训练387.481秒、任务436.593秒。新增全部任务墙钟8619.064秒（2.394185小时，含加载/CPU评分、共享GPU，非纯CUDA内核时间）；各新GRPO训练allocated峰值13.371–13.402 GiB。200步layered预计另需3871.786秒，超出剩余380.936秒预算，未启动。未再使用test，原冻结记录不变。没有安装依赖或下载模型。

本地最终全量测试：876 passed、1 skipped、14 warnings（13.20秒）。补充实验的解释见本章正文 §25.8.6，配对区间见 `outputs/book/ch25/gpu/supp/seed-summary.json`；配置、各运行耗时/CER、日志、CSV和PNG见 `outputs/book/ch25/gpu/supp/`；测试日志为 `outputs/book/ch25/gpu/logs/pytest-supp-full.log`。

### 验证器覆盖限制

CPU 复核实际训练切片 `build_split('train',4896,seed=0)[4096:]`：800条中310条（38.75%）标准答案无法通过 mathtext 子集，正确提交也得0。来源为本地 cleaned_formulas 快照 `436791f14f92cbb7fdc1b27f2c316cf1f4e65445`，参数见 `outputs/book/ch25/gpu/runs/ch25/main3/environment*.json`。`$x$` 对 target=`x` 的两种奖励均为0（`miniocr/eval/rewards.py` CPU复算）。机制探针金标准得1不能外推到全部语料。保持冻结奖励规则；今后的奖励版本应另行验证覆盖并与本次结果区分。
