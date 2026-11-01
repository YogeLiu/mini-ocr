# 第 18 章实验手册：提示与上下文学习

## 0. 准备

在 `course-v4/` 下运行，先激活已安装依赖的 Python 环境：

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q book/part3_post_training/ch18_prompting
CUDA_VISIBLE_DEVICES="" python -m book.part3_post_training.ch18_prompting.tour
```

部分号范围 **1–10**，与下文、`tour.PARTS` 一致。位置参数保留顺序，可单独运行；每部分重置 Python、NumPy、torch 随机种子。重复的生成会使用进程内缓存，部分耗时与首次生成耗时应分别记录。

```bash
CUDA_VISIBLE_DEVICES="" python -m book.part3_post_training.ch18_prompting.tour 1 2 3 7 10
CUDA_VISIBLE_DEVICES="" python -m book.part3_post_training.ch18_prompting.tour 5 6 --n 100 --seed 0 --out outputs/ch18
# 已有文件时只重抽取/评分；不存在的配置仍会生成。
CUDA_VISIBLE_DEVICES="" python -m book.part3_post_training.ch18_prompting.tour 5 6 --n 100 --seed 0 --out outputs/ch18 --reuse
```

CLI：`--device cpu|mps|cuda` 默认 cpu，`--dtype fp32|bf16` 默认 fp32，`--n` 控制 val 前缀大小（默认 20，完整默认 tour 在本机 CPU 两线程 fp32 下约需二三十分钟，以实际日志为准；运行头部也打印 `默认n`），`--seed` 默认 0，`--out` 可选预测 JSONL 目录。默认不写文件；需要留存时显式指定 `outputs/ch18` 或仓库外 scratch。`--n` 不改变第 1、2 部分的 val 全集统计。默认前缀很小，只供检查流程与观察格式，区间很宽；正式验证比较请显式增大 `--n`。`--batch-size` 默认 4；`--sub-n` 控制第 4、8、9 部分的子集大小（不超过 `--n`）。不传 `--sub-n` 时，第 4、9 部分仍最多取前 30 条，第 8 部分仍最多取前 50 条；没有随机抽换验证集。

第 4/5/8/9 部分均通过 `--out` 保存每个实验配置的逐样本 JSONL：`id`、`source`、`target`、`error_types`、`raw`（完整生成输出）、`pred`（实际评分输入；completion 已按换行截断）、`extracted`（对 pred 抽取后的输出）。另保留完整 `example`、抽取后 `score`、提示及运行元数据；复用相同配置时共用文件。

`--reuse` 必须搭配 `--out`。已有文件须条数相同，且每条的 `id`、完整 `prompt`、`model_path`、`dtype`、`device`、`batch_size` 按顺序与本次一致（旧记录未写批大小时按原固定值 4 校验）；字段缺失、损坏或不一致会指出路径、条号与字段并报错，不会悄悄重新生成。验证提示仍需本地 tokenizer，复用成功无需加载模型。读取原 `raw`/`pred`、token 数及长度上限标记，忽略旧 `extracted`/`score` 后重新评分；原文件保留不覆盖。新生成在每条记录中重复保存该配置的 `generation_total_seconds`，读时只取一次而非累加。旧文件缺少总秒数仍可复用，显示“原生成用时=未记录”；有秒数时标注“原记录”，不能把本次重评时间当成生成时间。

验收以 **CPU fp32 两线程** 为准；程序设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，OMP/MKL/OPENBLAS 线程均为 2，调用 `torch.set_num_threads(2)`。设备不存在时清楚报错；CUDA 路径把模型与输入放在 GPU，并在生成计时前后同步。只有显式 `--dtype bf16` 才启用 bf16；不自动降精度。不训练，不联网下载。

### GPU 上的推荐命令

以下为本次在 RTX 4090 48GB 上运行的完整 tour 命令，运行目录为教材根目录。`$RUN_DIR` 表示你保存本章运行结果和日志的目录，运行前请设置这个环境变量。远程 `python` 为 Python 3.11，torch 2.14.1+cu130、transformers 5.19.0；权重和语料均预先存在远程默认 HF 缓存。

```bash
python -u -m book.part3_post_training.ch18_prompting.tour --device cuda --dtype bf16 --n 2000 --sub-n 500 --batch-size 32 --seed 0 --out "$RUN_DIR/full-bf16-b32" --reuse
```

本次通过 `nohup` 后台执行，日志保存到 `$RUN_DIR/full-tour.log`。主实验取 val 前 2000 条（全集 5001 条），第 4、8、9 部分取相同的前 500 条；批大小 32、bf16。CPU 默认路径仍为 n=20、batch_size=4、fp32，未指定子集时使用 min(n,30)/min(n,50)/min(n,30)。两套结果的子集大小、批大小和精度不同，不能当成严格的设备对拍。左 padding 下改变批大小可能使个别输出不同；每批最长源公式也决定生成预算，改变分批可能改变截断位置。

参数依据是 64 条吞吐探测：batch=32 时 fp32 三种提示合计约 51.30 秒，仅放大到 5001 条主实验生成就约 67 分钟，尚未计入子实验和评分，因此显式选择 bf16（同规模约 36.24 秒）。额外探测 batch=64 的 bf16 合计约 40.69 秒，少样本更慢且显存占用更高，故保留 batch=32。完整运行时同卡出现多个其他章节任务，500 条 few_shot 实测增至 206.65 秒（0.413 秒/条，约为探测的 2.26 倍），全量会明显超过约一小时目标，故从 n=5001 降到 n=2000，并用 `--reuse` 保留已完成的两份 500 条生成。首次尝试日志为 `$RUN_DIR/initial-n5001.log`。样本规模、精度与批大小选择只依据耗时，不依据分数。

只读本地 HF 缓存，支持 `HF_HUB_CACHE` / `HF_HOME`。只寻找 `models--Qwen--Qwen3-0.6B/snapshots/` 中同时有 config 和 safetensors 的目录；模型与 tokenizer 均用 `local_files_only=True`。缺权重时模型相关部分打印跳过信息，数据、指标锚点、协议演示仍可运行。本实验加载的是 **Qwen3-0.6B chat 权重**；模型卡中的 base_model 元数据不表示我们加载或比较过 Base。语料缺缓存时沿用 latex_corpus 的内置样例降级警告，这种运行只能验证流程。

**输出约定：** 头部打印 GPU 名称、batch_size、sub_n，CUDA 来源标注同时包含设备名称。每行 `key=value`，注明“数学结论”“本项目观察（设备、dtype、线程、n）”或“待验证”。报告 torch/transformers 版本和确切快照。指标 JSON 中比例统一包含 `value`、`successes`、`count`、Wilson 95% `interval`；零分母 `value=null`。CER 是规范化字符的语料级比值，区间为固定种子、1000 次样本 bootstrap；不是逐样本 CER 的平均。parse_rate 是 mathtext 子集可解析率，不能称作真正的 LaTeX 编译成功率。规范化匹配不证明数学等价。

生成使用贪心、左 padding，模板字符串编码不重复添加特殊 token。每批上限为 `min(256, 2×最长源公式token数+32)`，达到上限且未遇 EOS 的条数单独打印。生成时间包含编解码、不包含模型加载与评分；部分耗时包含该部分所有工作。completion 在首个含换行的生成 token 停止，评分文本截去换行及后缀；完整已生成文本保存在可选 JSONL 的 raw 字段，生成计数包含停止 token，不含之后的批填充。

**评测协议：** 只在 val 上选提示和做决定；少样本示例只来自 train，默认确定性选有错样本。选型冻结后，预先登记的候选显式使用 `allow_test=True` 在 test 上各评一次，只报告数字和区间，不据此改任何设置；开发期间不看 test 错例，最终报告定稿后才可解封（0.10.1、5.1.3 节）。tour 不评 test。第 1 部分仅比较目标规范键是否跨划分泄漏。噪声 seed 不改变 latex-v1 划分；固定子集必须保留原划分序号。

对照代码：[tour.py](tour.py)、[test_ch18_prompting.py](test_ch18_prompting.py)、[数据](../../../miniocr/data/postcorrect.py)、[评测](../../../miniocr/eval/postcorrect_eval.py)、[加载器](../../../miniocr/models/qwen3.py)、[提示与生成](../../../miniocr/prompting.py)。以下简写 `python` 均指准备节解释器，并带 `CUDA_VISIBLE_DEVICES=""`。

## 1. 合成数据与泄漏检查

**先预测：** 固定 seed 后，截取前缀会改变已有样本的噪声吗？预留无错比例是否等于最终无错比例？

**运行命令：** `python -m book.part3_post_training.ch18_prompting.tour 1`。

**观察什么：** latex-v1 全部划分规模；六种错误的 source/target/errors；val 全集类型数、无错比例、平均错误数、目标公式字符长度均值和中位数（本项目观察）；三个划分目标规范键的交集。字符长度按目标原串的 Python `len`，含空白和 LaTeX 命令字符，未规范化，也不是 token 数。每类教学例子用单一错误配置，不代表该类发生频率。错误位置是每次编辑之前的字符下标，依次重放能恢复 source。

**记录：** 数据版本、噪声 seed、实际统计；不可施加的错误会跳过，可能使无错比例高于配置。extra_space 虽属于原串噪声，规范化通常会消除它。

同时观察 val 全集目标以 `. / , / ;` 结尾的条数和比例：按目标原串 `rstrip()` 后的最后字符统计，分母不受 `--n` 影响。

## 2. 指标锚点

**先预测：** copy 会有哪些非零分数？oracle 的可解析率是否一定满分？

**运行命令：** `python -m book.part3_post_training.ch18_prompting.tour 2`。

**观察什么：** val 全集及前 n 条的 copy/oracle 全指标、计数与区间。copy 的无错改坏率应为零；oracle 的 EM 应为一、CER 应为零。由于 mathtext 只支持子集，oracle 的可解析率也可能不满。

**记录：** 同一子集上的锚点，再比较提示方法。不能因为提示更复杂就认定优于照抄；copy 的零改坏率只能保持或变差，不能要求所有指标严格改善。

## 3. chat template

**先预测：** enable_thinking=False 是否意味着提示中没有 think 字符串？BOS 为 None 时，怎样判断模板有无重复特殊 token？

**运行命令：** `python -m book.part3_post_training.ch18_prompting.tour 3`。

**观察什么：** 两种模板的 repr、token id、token 字符串和特殊 token 映射。False 在 assistant 起始处预填空 think 块；True 不预填。检查实际缓存 tokenizer 的 bos/eos/pad。

**记录：** 两份完整模板、首尾 token 与差异；不把模板中预填的空 think 块误记成模型生成了推理内容。

## 4. 直接续写与 chat

**先预测：** 同样的三条示例和任务指令，换输入包装是否影响模型遵循格式？

**运行命令：** `python -m book.part3_post_training.ch18_prompting.tour 4 --n 30`。

**观察什么：** Noisy:/Corrected: 直接续写与多轮 chat 的原始/抽取后指标、token 和时间。completion 先按首个换行截断，再做可选抽取；两种方式都加载 chat 权重。

**记录：** 截断规则、样本数、生成预算命中数、成绩和格式差异。这个实验不支持“Base 与 chat 哪个更好”的结论。

## 5. 三种提示基线

**先预测：** 三条示例会减少哪类错误？严格格式约束会不会改变本来正确的公式？

**运行命令：** `python -m book.part3_post_training.ch18_prompting.tour 5 --out outputs/ch18`。

**观察什么：** 零样本、少样本 k=3、少样本+格式约束；每种同时输出原始评分和 extract_formula 后评分。记录 EM_raw、EM_norm/Wilson、CER/bootstrap、改坏率及无错分母、fix_rate、可解析率、总/平均时间、提示/生成 token。

**记录：** 填末尾基线表；注明子集、种子、抽取策略。小样本的无错分母可能为零，不能把 null 填成零。长度上限截断也算模型输出的一部分，不能删掉困难样本。

## 6. 输出格式问题

**先预测：** 格式约束对解释、美元符号和围栏是否一样有效？抽取器会不会改坏正确输出？

**运行命令：** `python -m book.part3_post_training.ch18_prompting.tour 5 6`。

**观察什么：** 各提示的 empty、has_explanation、dollar_wrapped、code_fence、think_block 计数、典型原文，以及抽取前后 parse_rate、em_norm 和差值（后减前）。抽取修好指原始输出 em_norm 错、抽取后对；抽取改坏指原始输出 em_norm 对、抽取后错，均统计同一批全部样本，不是按 source 是否有错分桶的 fix_rate/broken_rate。与第 5 部分合跑复用生成结果；单独运行第 6 部分会自行生成。

**记录：** 格式问题表和人工核对例子。has_explanation 的多行/语言启发式会误报。抽取器去 think/围栏后按出现顺序找 `$$…$$`、`\[…\]`、`\(…\)`、`$…$` 完整片段；`\$` 不算定界符。块外只有空白与 `. , ; 。` 时，用一个空格连接所有片段，否则取首片段；内部换行也折叠为空格。没有片段时逐行剥标签，丢弃像自然语言的行，连接剩余非空行，不再截断多行公式。自然语言判据是没有反斜杠、`^ _ { }`，且含连续三个英文单词、连续三个汉字加中文标点，或以 Explanation/Because 开头。末尾去一个未转义的句末 `. , ; 。`，保留 LaTeX 命令及不可见定界符（如 `\right.`）。

这些规则不理解语义、嵌套、注释或 verbatim。块外裸等式也会阻止拼接并被丢弃；说明里有 LaTeX 时可能保留，纯字母公式可能被误删；拼接不能恢复块间的逗号或 LaTeX 换行，裸句末标点也可能属于答案。即使不再截断 array，个别样本仍可能变差。必须同时报告原始与抽取后评分，parse_rate 仍只是 mathtext 子集可解析率。修改抽取规则后可用 `--reuse` 重评已保存生成；这仍属于 val 上的开发，最终规则须在评测 test 前固定。

**句末标点诊断的观察与记录：** 对三种提示的抽取后输出及照抄基线，记录“忽略句末标点的 EM_norm”的成功数、分母和 Wilson 95% 区间，以及“改坏样本中仅差句末标点的条数”，填写下方空表。比较时 target 和 pred 均先规范化，再去掉 token 序列末尾全部独立的 `. , ;`；不删中间标点。“改坏”指 source 原本无错且本次评分 EM_norm 错，与 broken_rate 同口径，不是上文的“抽取改坏”。照抄直接用 source、不经过抽取器，核对其诊断与官方 EM_norm 是否一致、改坏是否仍为零。**诊断="非官方指标，只用于错例分析；官方指标仍是 EM_norm"**；标点可能有数学含义，不能据此追改事先固定的官方指标。

## 7. 上下文占用

**先预测：** k 增长时 token 数是否严格按示例条数等比例增长？

**运行命令：** `python -m book.part3_post_training.ch18_prompting.tour 7`。

**观察什么：** k=0,1,3,5,8 的提示平均/最大 token，相对 k=0 的每例平均增量，以及本地 config.max_position_embeddings。只需 tokenizer 和 config，不生成。

**记录：** 最长提示与生成预算是否仍在配置上限内；平均 token 占用不能替代最坏情况检查。

## 8. 措辞敏感性

**先预测：** 三个语义接近的指令会得到同一输出吗？一次点估计的高低足以选出胜者吗？

**运行命令：** `python -m book.part3_post_training.ch18_prompting.tour 8 --n 50`。

**观察什么：** 三个 PARAPHRASES 在相同 val 前缀上的抽取后 EM_norm、CER、改坏率及区间，最大最小差。

**记录：** 填措辞表，保留每个样本对应关系。区间重叠/不重叠不是这里的显著性检验；同一子集是配对比较，第 22 章用配对 bootstrap。

## 9. 示例选择与顺序

**先预测：** 只倒转示例顺序和换另一组三条示例，哪个变化更大？

**运行命令：** `python -m book.part3_post_training.ch18_prompting.tour 9 --n 30`。

**观察什么：** 原顺序、逆序、seed+1 新池的示例 ID 和全部指标；例子均来自 train，不用 val 错例当提示示例。

**记录：** 三组比较、具体错误变化；该结果仅是同一验证前缀上的观察，不推广成固定排序定律。

## 10. 评测协议

**先预测：** 默认请求 test 时，拒绝会发生在模型调用前还是调用后？

**运行命令：** `python -m book.part3_post_training.ch18_prompting.tour 10`。

**观察什么：** evaluate(split="test") 明确拒绝，协议见第 0 章 0.10.1 节；不读测试样本、不运行测试预测。

**记录：** 决策只能基于 val；选型冻结后，由报告流程显式使用 allow_test=True，让预先登记的候选各评一次，不据此改设置；开发期间不看 test 错例，最终报告定稿后才可解封。单元测试的 allow_test 分支只用注入小语料，不是对真实测试集评测。

## 自己填写的实测记录

环境：设备 ______；系统 ______；torch ______；transformers ______；dtype ______；线程 ______；seed ______；实际 n ______；快照 ______；数据版本 ______；完整用时 ______。

三种提示基线表（每个比例同时记成功数、分母及区间；CER 记 bootstrap 区间）：

| 提示 | 评分口径 | EM_raw | EM_norm / CI | CER / CI | 改坏率 / 分母 / CI | fix_rate / CI | 可解析率 / CI | 总秒 / 每条秒 | 平均提示 / 生成 token |
|---|---|---|---|---|---|---|---|---|---|
| 零样本 | 原始 | | | | | | | | |
| 零样本 | 抽取后 | | | | | | | | |
| 少样本 | 原始 | | | | | | | | |
| 少样本 | 抽取后 | | | | | | | | |
| 少样本+格式约束 | 原始 | | | | | | | | |
| 少样本+格式约束 | 抽取后 | | | | | | | | |

措辞敏感性表：

| 指令 | n | EM_norm / CI | CER / CI | 改坏率 / 分母 / CI | 配对观察 |
|---|---|---|---|---|---|
| Correct OCR errors… | | | | | |
| Repair OCR mistakes… | | | | | |
| Recover the correct… | | | | | |
| 最大最小差 | | | | | |

格式问题表（同一输出可计入多列）：

| 提示 | 空输出 | 解释/多行 | 美元符号 | 围栏 | think 块 | 抽取修好 | 抽取改坏 |
|---|---|---|---|---|---|---|---|
| 零样本 | | | | | | | |
| 少样本 | | | | | | | |
| 少样本+格式约束 | | | | | | | |

句末标点诊断记录表（非官方指标，只用于错例分析；官方指标仍是 EM_norm）：

| 方法 | 评分口径 | 官方 EM_norm | 忽略句末标点的 EM_norm 成功数 / 分母 / Wilson 95% CI | 改坏样本数 | 改坏样本中仅差句末标点的条数 |
|---|---|---|---|---|---|
| 照抄 | source，不抽取 | | | | |
| 零样本 | 抽取后 | | | | |
| 少样本 | 抽取后 | | | | |
| 少样本+格式约束 | 抽取后 | | | | |

项目 R2 第一行（验证集提示基线，测试集留待选型冻结后按 0.10.1 节评测）：

| 项目 | 数据/噪声版本 | 方法及模板 | val n | EM_norm / CI | CER / CI | 改坏率 / 分母 | fix_rate / 分母 / CI | 可解析率 | 时间 | 尚待验证 |
|---|---|---|---|---|---|---|---|---|---|---|
| R2：提示后纠错基线 | | | | | | | | | | |
