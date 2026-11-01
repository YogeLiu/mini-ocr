# 第 22 章实验：评测与误差分析

## 0. 准备

在仓库根目录执行：

```bash
cd course-v4
CUDA_VISIBLE_DEVICES="" python -m pytest -q book/part3_post_training/ch22_eval
# 部分 1：统计工具与机制检查，不需要模型
CUDA_VISIBLE_DEVICES="" python -m book.part3_post_training.ch22_eval.tour 1 --out outputs/book/ch22/cpu
# 部分 3：只读已有预测做分析（需要先有 preds-*.jsonl）
CUDA_VISIBLE_DEVICES="" python -m book.part3_post_training.ch22_eval.tour 3 --n 20 --out outputs/book/ch22/cpu
```

本章默认 `--n 20`，只为让学习者在 CPU 上跑通流程。CPU 的 20 条结果不能作为效果结论。部分 2 需要本地 Qwen3-0.6B 权重；SFT 生成还需要第 21 章选定 step 的 adapter。下面的 `$RUN_DIR` 是本次运行的产物目录，`$LOG_DIR` 是日志目录，运行前请设置为自己的路径：

```bash
python -m book.part3_post_training.ch22_eval.tour 2 --device cuda --dtype bf16 --n 500 \
  --batch-size 16 --adapter /path/to/ch21/step-XXX --out "$RUN_DIR"
python -m book.part3_post_training.ch22_eval.tour 3 --n 500 --out "$RUN_DIR"
```

远程运行的产物写到 `$RUN_DIR`；如需保存终端日志，可重定向到 `$LOG_DIR/ch22_*.log`。val 规模 500，与第 19 章的 val 评测一致；test 本章不评。

精度口径：生成与第 21 章 `evaluate` 相同，基座 FP32 加载，`autocast` 为 bf16，贪心解码，`max_new_tokens=192`。环境版本与 val 的 ID 哈希写在 `environment.json`。

## 1. 统计工具与机制检查

```bash
python -m book.part3_post_training.ch22_eval.tour 1 --out outputs/book/ch22/cpu
```

这一部分检查四件事：

- 比例的 bootstrap 区间：同一批样本上的抽样波动，不是模型之间的显著性。
- 配对差：两个模型在同一批题目上的逐题差值。题目难度相同的部分被抵消，所以配对区间通常比两组独立区间窄。
- 分桶：公式类型（矩阵/数组、分式、上下标、其他）按优先级互斥分配；长度按字符数分三档。
- 错例标签：`format`（输出带解释或包装）、`unparsable`、`unchanged`、`over-correction`、`missing`、`extra`、`substitution`、`mixed`。标签优先级固定，是启发式，不是人工金标准。

## 2. 对 val 重新生成

部分 2 对同一批 500 条 val 生成三种输出：第 18 章零样本提示、第 18 章严格 3-shot、第 21 章 SFT adapter。输出保存为未经抽取的原始文本，部分 3 按原始输出评分，不做抽取（正文 22.6.3 节）。诊断时可以对已有输出重算，但要与冻结的评分结果分开报告。

## 3. 配对比较、分桶与泄漏近似

部分 3 输出：

- 每种方法的 EM_norm、CER、broken/fix 率及其区间；
- SFT 与两个提示基线的配对差（bootstrap，1000 次）；
- 公式类型与长度分桶；
- 错例标签计数表；
- 泄漏近似检查：val 与 train 参考（前 20000 条干净公式）的规范键精确重叠，以及 MinHash/LSH 近重复（Jaccard ≥ 0.8，LSH 只能给出下界）。

泄漏检查的边界要说清：它只检查本课程语料内部的 train 与 val。基座模型 Qwen3 的预训练数据是否包含这些公式，我们无法检查。

## 4. 反复看 test 的窥视模拟

```bash
python -m book.part3_post_training.ch22_eval.tour 1 --out outputs/book/ch22/cpu
```

部分 1 与部分 3 都包含模拟。假设 k 个候选的真实准确率完全相同（0.7），仅用 val 选择再报告其 test 准确率，结果无偏；若对 k 个候选都看 test 并报告最高值，结果随 k 增大而偏乐观。模拟用于理解机制，不代表任何真实模型。本章正式实验没有评 test。

## 5. 读结果的顺序

1. 先看 EM_norm 与 CER 的区间是否重叠；区间重叠不等于没有差别，要看配对差。
2. 再看配对差的区间是否包含 0。
3. 再看分桶：总体提升是否集中在某一类公式或某一类长度。
4. 最后看错例标签：提升主要减少的是哪类错误（漏写、多写、替换、格式）。

注意 `unparsable` 这一类：它依赖的是 matplotlib mathtext 子集的可解析判断，不是 LaTeX 编译。本次测量中，干净参考答案本身只有 58.6% 能被这个子集解析；SFT 输出的 parse_rate 为 0.524。解析失败输出集合不等于互斥错误标签 unparsable：前者 238 条中 133 条 EM 正确，后者 95 条均不匹配（先判 correct/format）。因此解析失败中有相当一部分是解析器覆盖不足（如矩阵、`\\`、`\\text`、`\\left`），不是模型错误。读误差时要同时看 EM_norm，不要把 parse_rate 当作正确率。

## 6. 运行前的预测题

- 如果 SFT 相对 3-shot 的配对差区间包含 0，你期望 EM_norm 的 bootstrap 区间宽度大约是多少？
- 长公式桶的样本数少时，区间会变宽还是变窄？为什么？
- 在窥视模拟中，k=1 时报告值为什么接近真值？k=20 时平均偏差大约是多少？

## 7. 记录表

本项目观察，RTX 4090（租用），基座 FP32 + bf16 autocast，val n=500，贪心解码，`max_new_tokens=192`。原始输出（raw）的 EM_norm 与 CER，区间为样本 bootstrap 95%（CER）与 Wilson 95%（EM）。完整的文本后纠错面板还包括改坏和 fix_rate，见正文 22.2.6 节。

| 方法 | EM_norm（95% 区间） | CER（95% 区间） | 与 SFT 的配对差（SFT − 方法） | 生成墙钟 |
|---|---|---|---|---|
| 零样本（第 18 章提示，未微调） | 0.000 [0.000, 0.008] | 1.162 [1.097, 1.244] | +0.678 [0.638, 0.718] | 515 秒 |
| 严格 3-shot（第 18 章） | 0.048 [0.032, 0.070] | 0.148 [0.125, 0.171] | +0.630 [0.584, 0.672] | 841 秒 |
| SFT（第 21 章 step 400） | **0.678 [0.636, 0.717]** | **0.026 [0.017, 0.040]** | — | 426 秒 |

错例标签（同一 500 条，原始输出）：

| 方法 | correct | format | unparsable | unchanged | over-correction | missing | extra | substitution | mixed |
|---|---|---|---|---|---|---|---|---|---|
| 零样本 | 0 | 499 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| 严格 3-shot | 24 | 347 | 71 | 3 | 20 | 6 | 2 | 8 | 19 |
| SFT | 339 | 13 | 95 | 6 | 5 | 2 | 5 | 26 | 9 |

分桶（SFT，EM_norm）：上下标 0.677（266 条）、分式 0.668（187）、矩阵/数组 0.703（37）、其他 0.800（10）；长度 <30 为 0.727（22），30–59 为 0.683（82），≥60 为 0.674（396）。

泄漏近似：val 与 train 参考（前 20000 条干净公式）规范键精确重叠 0 条；MinHash/LSH 近重复（Jaccard ≥ 0.8）命中 2 条 val 目标（占 500 条的 0.4%），两者各与训练集中的近似公式只差局部下标（Jaccard 0.80）。仅覆盖本语料，不能排除基座模型的预训练数据。

窥视模拟（真值 0.7、n_val=n_test=500、2000 次）：k=1 报告值 0.700（偏差 0.000）；k=2 偏差 +0.011；k=5 +0.023；k=10 +0.031；k=20 +0.038。

CPU/MPS/5080 的对应列暂留空白。

### 全量复核与冻结口径

CPU 全量复核（2026-10-30）：`target_splits()['train']` 共 542339 条，对 `build_split('val',500,seed=0)` 检查，规范键精确重叠 0，MinHash/LSH 近重复命中 **25/500**。沿用 `miniocr/eval/analysis.py::near_overlap` 的规范键字符3-gram、128 permutations、16 bands×8 rows、seed=0、Jaccard≥0.8；仅做 val–train 的 band join，省去无关的 train–train 候选，逐对核验 Jaccard。LSH 结果仍是下界，不能推断预训练泄漏，也不能从命中数直接量化其对 EM 的贡献。数据来源为本地 `OleehyO/latex-formulas/cleaned_formulas` 快照 `436791f14f92cbb7fdc1b27f2c316cf1f4e65445` 的六个 parquet，经 `miniocr/data/latex_corpus.py` / `postcorrect.target_splits` 划分；旧 `outputs/book/ch22/gpu/analysis.json` 的 train_reference=20000、near_hits=2 保留，不覆写历史结果。
