# 第 19 章实验手册：SFT 的监督信号

## 0. 准备

在 `course-v4/` 下运行。默认使用微型随机 GPT 和 byte tokenizer，不下载、不读取大权重；默认执行部分 **1、2、3、4、8**。部分号范围 **1–8**，位置参数保留顺序，每部分重置 Python、NumPy、torch 种子。

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q book/part3_post_training/ch19_sft
CUDA_VISIBLE_DEVICES="" python -m book.part3_post_training.ch19_sft.tour
CUDA_VISIBLE_DEVICES="" python -m book.part3_post_training.ch19_sft.tour 1 2 3 --out outputs/ch19/local
# 有本地 Qwen3 权重时，查看真正的模板；这条不加载模型权重。
CUDA_VISIBLE_DEVICES="" python -m book.part3_post_training.ch19_sft.tour 1 2 8 --real
```

`--device cpu|mps|cuda` 默认 cpu，`--dtype fp32|bf16` 默认 fp32，`--n` 默认 **20**，控制真实生成的 val/test 前缀条数；`--train-n 16 --steps 2 --batch-size 2` 是真实训练的小规模默认。`--real` 显式启用 Qwen3；真实训练必须指定 `--out`。输出为逐行 JSON 键值记录，包含来源、设备、精度、样本数、部分号和耗时。使用两条 CPU 线程，HF 离线模式；没有权重时真实 tokenizer 测试 skip，真实实验命令报清楚的缺缓存错误。默认 tour 不需要任何权重。

本机可运行的小规模真实训练命令（需要已有模型和语料缓存）：

```bash
python -m book.part3_post_training.ch19_sft.tour 5 --real --device cpu --train-n 16 --steps 2 --n 20 --out outputs/ch19/cpu-smoke
# MPS 待验证；此循环在 MPS 上使用 fp32。
python -m book.part3_post_training.ch19_sft.tour 5 --real --device mps --train-n 32 --steps 5 --batch-size 1 --out outputs/ch19/mps-smoke
```

4090 推荐命令，均从 `course-v4/` 执行。`$LOG_DIR` 表示保存总日志的目录，运行前设为自己的可写目录并创建。配套 GPU 脚本读取环境变量 `COURSE_WORK_ROOT`（未设置时默认 `/hy-tmp`）；可用 `export COURSE_WORK_ROOT=/自己的工作根目录` 设置，教材目录和日志分别由 `$COURSE_WORK_ROOT/course-v4`、`$COURSE_WORK_ROOT/logs` 派生。第 19 章产物仍位于教材目录下的 `outputs/book/ch19/gpu`。首次必须先运行冒烟，检查 mask、loss、可训练参数及显存，再放大：

```bash
python -u -m book.part3_post_training.ch19_sft.tour 1 2 3 8 --real --device cuda --dtype bf16 --out outputs/book/ch19/gpu/mechanisms
python -u -m book.part3_post_training.ch19_sft.tour 4 --device cuda --dtype bf16 --out outputs/book/ch19/gpu/mechanisms
python -u -m book.part3_post_training.ch19_sft.tour 5 --real --device cuda --dtype bf16 --train-n 512 --steps 30 --batch-size 8 --accumulation 2 --n 32 --eval-batch-size 16 --out outputs/book/ch19/gpu/smoke
# 远程脚本包括主训练、三种提示基线、最终 val、四组真实模型故障对照；不含 test。
nohup bash book/part3_post_training/ch19_sft/run_gpu.sh > "$LOG_DIR/ch19_suite.log" 2>&1 < /dev/null &
```

脚本在主训练前等待至少 32 GiB、故障对照前至少 20 GiB、仅推理前至少 10 GiB 可用显存，不停止其他进程；`monitor_gpu.py` 记录共享实例的占用，耗时不当作独占 GPU 性能。脚本默认 `all` 跑全套；主训练和三基线已完成时可传 `remaining`，最终 val 也已完成时可传 `faults`。这是续跑实验阶段，不是恢复 optimizer 状态。脚本是租用实例的可重跑入口，`$LOG_DIR` 只指定总日志的位置。长任务用 nohup，输出目录须换成新的目录后再重跑，以免覆盖既有观察。主配置为 `--train-n 4096 --steps 512 --batch-size 8 --accumulation 2 --lr 2e-5 --max-length 512 --n 128 --eval-every 128 --save-checkpoint`；生成批大小为 16。5080 的同配置显存可行性**待验证**，可先将 batch-size 降至 1 并启用 `--gradient-checkpointing`，不得把 4090 数字填到 5080 栏。

**精度口径：** 本章 bf16 是混合精度：参数和 AdamW 状态为 fp32，前向用 bf16 autocast，交叉熵用 fp32；不使用 GradScaler、量化、LoRA、Trainer 或 TRL。故障组与主实验均为全参更新。基线也用相同的 fp32 权重 + bf16 autocast，不能把它与第 18 章直接加载 bf16 权重的历史速度混为一谈。本章重新运行第 18 章的三种提示，复用同一生成与评分函数。

**数据口径：** 只读取已下载的 `OleehyO/latex-formulas` / `cleaned_formulas` 的 `latex_formula` 列；不自行渲染训练图片，不重新下载数据。本章是文字后纠错，错误来自第 18 章 `postcorrect-v1` 的合成扰动，不是实际 OCR 错误分布。按 `latex-v1` 的规范化键、固定划分 seed=0 先切分完整语料，再取 train/val/test 前缀；噪声 seed=0。数据来源和许可：arXiv/im2latex-100K，openrail，仅本地使用、不分发原始语料。确切数据快照、所读 parquet 文件、被接受训练样本的身份摘要保存在 `train_config.json`。本章不新增下载和 pip 安装。远程实测环境：Python 3.11.10、torch 2.14.1+cu130、transformers 5.19.0、pyarrow 26.0.0、matplotlib 3.11.2；已安装但本章未使用的 peft/trl/accelerate 版本为 0.21.2/1.15.0/1.15.0。parquet 元数据实测共 552340 行、6 个分片（本项目观察，见 `remote_environment.json`）。

**评测口径：** 使用第 0 章 §0.9.3 的文本后纠错面板，必报 EM_norm、CER、改坏率、fix_rate。主模型按 val 前 128 条的原始输出 EM_norm 选择 checkpoint，同分时选 CER 较低者，完全同分保留更早者。选完后用 val 前 500 条与三种提示基线比较；两个范围不可混用。所有生成均贪心、左 padding、非思考模板，每批预算 `min(256, 2*最长源token数+32)`，与第 18 章相同。训练采用右 padding。生成评测不按训练长度过滤样本。比例报告分子、分母及 Wilson 95% 区间，CER 是规范化字符的语料级比值、1000 次 bootstrap 区间。`parse_rate` 仅是 mathtext 子集可解析率，不能称作完整 LaTeX 编译率。

原始输出和 `extract_formula` 后的分数都保留，不能择优混报。抽取器会去句末标点，而 target 可能保留标点，因此抽取后 EM 可能明显更低。`broken_rate` 的分母是原串无错的 source，`fix_rate` 的分母是原串有错的 source；规范化可使某些原串噪声消失。

按第 0 章 §0.10.1，选型冻结后，预先登记的候选可各评一次，不据此改任何设置。下面的命令只评选中的 checkpoint 一次，保存汇总和置信区间，不输出 test 错例；开发期不看 test 错例，最终报告定稿后才可解封：

```bash
python -u -m book.part3_post_training.ch19_sft.tour 7 --real --device cuda --dtype bf16 --n 500 --eval-batch-size 16 --checkpoint outputs/book/ch19/gpu/main/best --split test --final --out outputs/book/ch19/gpu/final
```

`test_once.json` 在读 test 前独占写入，含选中 checkpoint 的 val 记录、选择记录 SHA256、评测配置和冻结时间；同一输出目录拒绝重复 test。它是防误操作的本地守卫，不是跨目录防作弊系统。模型权重留在租用实例，拉回的仅为日志、指标、CSV、图和报告。

## 1. 模板到 token（部分 1）

查看 `token_trace.json` 与逐 token 日志：模板含 system/user、assistant 前缀与空 think 块；答案接在其后，最后追加 EOS。先联合编码 prompt+answer，再核对 prompt 的 token 前缀。不能把字符数当 token 数，也不能盲目假设分别编码后拼接必然等价。

每个位置打印 token、ID、label、是否参与 loss、对应预测位置。`labels` 和 `input_ids` 等长，提示 label 为 −100，答案及末尾 EOS label 为原 ID。练习入口：`miniocr/data/sft.py` 的 `# >>> 练习 19.1`。

## 2. 变长 batch（部分 2）

用两个不同长度样本观察三张量。右 padding 的 input_id 使用 pad ID，attention_mask 为 0，labels 为 −100；真实 EOS 即使与 pad ID 相同也必须保留监督，因此按**位置**屏蔽，不能把所有等于 pad ID 的 label 清掉。`multiple=8` 演示批宽对齐；真实训练使用动态批宽，不额外补齐。

逐渐缩小长度预算：刚好容纳完整答案和 EOS 时接受；少一个 token 就拒收整条。统计 accepted/rejected，不能丢掉答案后继续计算空 loss，更不能给半截答案强加 EOS。此策略牺牲超长样本覆盖，正式报告必须列出拒收数。练习入口：19.2。

## 3. 因果对齐与 loss（部分 3）

逐 token 的 `-log p(target)` 均值与公共 `causal_loss` 对拍。数学结论：位置 t 的 logit 预测位置 t+1 的 token，数据整理不预先 shift。提示没有直接 CE 项，但答案通过注意力依赖提示，因此提示表示仍可收到梯度。练习入口：`miniocr/train/sft.py` 的 19.3。

## 4. 四组故障（部分 4；真实对照使用部分 5）

微型模型从相同初始化训练 120 步、4 条固定教学样本；GPU 训练/NLL 为混合 bf16，微型生成使用 fp32。这是机制对照，不是后纠错泛化成绩。真实 Qwen3 每组从同一原始 checkpoint 重新开始，固定 train 前 1024 条、80 步、microbatch=2×梯度累积4（有效 batch=8）、启用梯度检查点、val 前 80 条。

| variant | 改变 | 观察重点 |
|---|---|---|
| correct | 提示 mask、答案+EOS 监督 | 正确目标 NLL、生成、停止 |
| shift | labels 提前左移一次；循环仍做 causal shift | 学成跨两个位置预测，低训练 loss 不保证正常生成 |
| prompt | 提示也参与 CE | 分母和目标都变了，训练 loss 不可跨组直接排名 |
| no_eos | 删除训练样本结尾 EOS，input 和 label 都少一个 token（正文 §19.9.1） | 看生成长度及达到预算比例，不预设预训练模型一定不停止 |

四组均额外在**正确 labels** 上测 val NLL，所以能比较模型实际学到了什么。三种错误都不是“loss 必然升高”；必须结合 generation。故障组不选主模型、不触碰 test。

## 5. 自写训练循环（部分 5）

检查 trainable_params == total_params，正确组提示监督 token=0、受监督末尾 EOS 数=accepted 样本数。AdamW 使用 lr=2e-5、weight_decay=0.01、梯度裁剪上限 1；固定学习率，无调度、无断点续训。每次更新把所有 microbatch 的 loss sum 除以同一有效 token 总数。不同长度 microbatch 的 mean 再平均一般不等价。练习入口：19.4。

`history.jsonl` 保存每步 loss、有效 token 数、裁剪前梯度范数和实际训练秒数；每次 val 保存预测 JSONL、原始/抽取两套指标 JSON、正确标签 NLL。`selected.json` 记录所选步，`best/` 存权重；`train_summary.json` 含训练内时间、整段时间和峰值 allocated 显存。分清 optimizer-step 时间与数据准备、评测、写 checkpoint 的时间。

## 6. 提示基线与冻结评测（部分 6、7）

部分 6 复用第 18 章的 zero-shot、few-shot、few-shot-strict 与 train 示例池，额外保留 copy 基线。部分 7 从本地已训练 checkpoint 加载；默认只评 val。比较时核对 `ids_sha256`、n、种子、生成批大小、精度、模板和预算一致，不拿 smoke 的小样本结果与最终大样本结果作提升百分比。

## 7. 多轮概览（部分 8）

打印含两轮 user 的模板，说明上一轮 assistant 是当前轮上下文。这里不实现多轮全 assistant 监督；不同模板的分隔符、思考段和 assistant mask 需要单独处理，不能直接推广单轮的一个前缀长度。

## 运行前的预测题

1. 如果 pad ID 恰好等于 EOS ID，按 ID 统一设 −100 会漏掉哪一种监督？
2. labels 提前 shift 后，某位置实际在预测哪一个 token？训练 loss 能否继续下降？
3. 为什么监督整个 prompt 的 loss 可能更小，却不能证明后纠错更好？
4. 漏 EOS 时，从零训练的小模型和已经学过 EOS 的 Qwen3 会不会出现同样程度的不停止？
5. 把两个不同有效 token 数的 microbatch mean 平均，会给短答案多大权重？
6. 训练 loss 很低但 val 的 EM 没提高时，应该先检查模板、mask、EOS，还是直接增加步数？列出能区分这些假设的日志字段。

## 怎样读结果

先看机制对拍和 train_config 的监督计数，再看固定正确 labels 的 val NLL，最后看生成 EM/CER、改坏率、达到长度预算的数量。NLL 是 teacher forcing，EM 是自回归生成；二者不会一一对应。低 CER 也可能对应大量非精确匹配。只把实际记录到的现象写成“本项目观察”，不能把某个故障在某次实验中的现象写成数学必然。

实测记录见下方及正文 §19.7–§19.9，主模型指标和用时分别见 `outputs/book/ch19/gpu/final/final-val.json`、`outputs/book/ch19/gpu/main/train_summary.json`；CPU/MPS/5080 的记录列留给学习者。置信区间完整保留在 JSON 和 CSV 中，图只作读数辅助。

## 实测记录（本地归档）

本项目观察，RTX 4090（租用），FP32 参数 + bf16 autocast；以下均为 val，test 未评。来源以 `outputs/book/ch19/gpu/` 为根。

| 实验 | n / 步数 | EM_norm | CER | 来源 |
|---|---|---|---|---|
| 主训练选中 checkpoint | 128 / 512 | 0.750 | 0.0133 | `main/val-step512.json` |
| 冻结 checkpoint | 500 / 512 | 0.730 | 0.0188 | `final/final-val.json` |
| correct | 80 / 80 | 0.588 | 0.019 | `fault_correct/val-step80.json` |
| prompt | 80 / 80 | 0.613 | 0.021 | `fault_prompt/val-step80.json` |
| no_eos | 80 / 80 | 0.512 | 0.025 | `fault_no_eos/val-step80.json` |
| shift | 80 / 80 | 0.000 | 2.102 | `fault_shift/val-step80.json` |

shift 达到预算 66/80。主训练 `train_summary.json` 记录训练循环 468.475 秒、总任务 766.936 秒；总任务包含准备/评测，不作为独占内核时长。CPU/MPS/5080 未测。
