# 第 34 章实验手册：VLM 的可验证奖励（GRPO）

## 0. 准备

在 `course-v4/` 运行。默认只运行无权重的部分 1（奖励分层与投机探针），CPU 即可：

```bash
CUDA_VISIBLE_DEVICES="" python -m book.part4_vlm.ch34_vlm_grpo.tour
CUDA_VISIBLE_DEVICES="" python -m pytest -q book/part4_vlm/ch34_vlm_grpo
```

真实部分需要第 33 章的 SFT/LoRA 适配器（`--adapter`）和第 33 章落盘的数据目录（`--data`，内含 `train.json`、`val.json`、`test.json` 与 PNG）。模型只读本地缓存，不下载。

远程推荐命令（RTX 4090 租用；运行前检查其他任务的显存，本章单个任务控制在约 20 GB 内）：

`$PROJECT_DIR` 是教材副本的根目录，`$RUN_DIR` 是各章运行文件的根目录（含 `ch32`、`ch33`、`ch34`），`$LOG_DIR` 是日志目录；运行前将这三个变量设为自己的路径，并创建日志目录。

```bash
cd "$PROJECT_DIR"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export HF_HOME="$RUN_DIR/ch34/hf" MPLCONFIGDIR="$RUN_DIR/ch34/mpl"
# 20–50 步阻断检查（奖励、优势、KL、有效组比例）
nohup python -u -m book.part4_vlm.ch34_vlm_grpo.tour 3 --device cuda --dtype bf16 --arm grpo \
  --model qwen3vl --model-cache "$RUN_DIR/ch32/hf/hub" --data "$RUN_DIR/ch33/data" \
  --adapter "$RUN_DIR/ch33/qwen3vl_main/adapter" --train-n 2048 --steps 20 --prompts 2 --group 8 \
  --train-max-new-tokens 128 --lr 1e-5 --out "$RUN_DIR/ch34/smoke" > "$LOG_DIR/ch34_smoke.log" 2>&1 &
```

本条命令的运行输出与缓存位于 `$RUN_DIR/ch34/`，日志位于 `$LOG_DIR/ch34_smoke.log`。

**口径：** 样本来自第 33 章的 latex-v1 训练划分（OleehyO/latex-formulas，openrail）；原图只在实验机缓存。奖励只依据 train 的标准答案；调参与选择只看 val；最终设置冻结后，预先登记的起点与所选模型在 test 上同一轮各评一次（`data/ch34_final_test.lock` 排他创建），不据此改任何设置；开发期不看 test 错例，最终报告定稿后才可解封（第 0 章 §0.10.1）。

## 1. 分层奖励与投机探针（无权重）

`tour 1`。奖励先用 `extract_formula` 从原始输出中取出公式，再与标准答案比较：

| 层级 | 条件 | 奖励 |
|---|---|---|
| 0 | 空输出或不可解析 | 0 |
| 1 | 可解析（mathtext 子集） | 0.2 |
| 2 | 编辑距离 ≤ 标准答案长度的 20% | 0.5 |
| 3 | 规范化后完全一致（EM_norm） | 1.0 |

投机探针：对固定的退化输出（空、`x`、`\frac{a}{b}`、`\alpha`、`a+b=c`）打分。结果显示：只要可解析，即使与标准答案无关的字符串也能拿到 0.2。这是奖励本身的漏洞，训练中用退化输出比例（`degenerate`）记录它是否被利用。

## 2. 显存估算（只读 config）

`tour 2` 按词表大小和 `P×G×T` 估算 logits 与 log-prob 的显存，输出 `memory_estimate.json`。这只是上界式估算；真实峰值以训练 JSON 为准。

## 3. GRPO 训练

`tour 3 --arm grpo`。每步取 `P` 张训练图，每张采样 `G` 个候选（纯采样，`top_k=0`、`top_p=1`，使 log-prob 与采样分布一致）。

- 优势：组内标准化 `(r - mean)/std`；标准差为 0 的组（全对或全错）无信号，优势置 0 并计入“无信息组比例”。
- 损失：token 级策略梯度 + β·k3 KL。参考模型是 SFT 起点（单独加载的一份冻结副本，不是未微调的基座）。损失按整步的有效 token 数归一化。
- 每步写入 `train_log.jsonl`：奖励均值、各层比例、无信息组比例、KL、退化比例、平均 token 数。
- 训练前后各做一次投机探针（`probe_before` / `probe_after`）。

机制层位置：`# >>> 练习 34.1`（组内优势）、`# >>> 练习 34.2`（GRPO 损失与 KL 方向）、`# >>> 练习 34.3`（投机探针）。

## 4. 同起点 SFT 对照

`tour 4 --arm sft`。从同一适配器出发，用同样的提示顺序只做交叉熵继续训练，学习率与步数相同。这回答的问题是：奖励信号本身是否比继续模仿标准答案更有用。

## 5. val 评测（只用于选择）

`tour 5 --adapter <某个适配器> --name <名称>`。与第 33 章相同的贪心解码、像素预算和输出上限（256），val 100 条。比较对象：起点 SFT、GRPO、继续 SFT。选择规则：val EM_norm 最高，平局取更低 CER（`summarize.py` 写入 `selection.json`）。

## 6. test 一次性评测

`tour 6 --split test --final-test --adapter <起点> --compare-adapter <所选>`。起点与所选模型在同一轮各评一次；锁文件排他创建，不能删除后重跑。

## 运行前的预测题

- 奖励 0.2 的“可解析”层会不会鼓励模型输出短的、无关的公式？投机探针在训练前已经给出什么信号？
- 为什么同一图片的 G 个候选要一起算优势，而不是每个样本单独比较？
- 无信息组（全对或全错）增多时，有效梯度会怎样变化？
- GRPO 的 KL 参考应该是基座还是 SFT 起点？选错会怎样？
- 若 GRPO 的 val EM 高于继续 SFT，能否断定 RL 更好？还需要看哪些指标（退化比例、截断数、人工抽查）？

## 怎样读结果

先看探针和退化比例，再看 KL 与无信息组比例，最后看 val 的 EM_norm、CER、parse_rate 与截断数（第 0 章 §0.9.3 的 OCR 面板精简版），并对照人工抽查的样例。奖励上升而 EM 不升，或退化比例上升，都是奖励投机的迹象。

## 实测记录

本项目观察，RTX 4090（租用），bf16；Qwen3-VL-2B-Instruct，从 ch33 `qwen3vl_main/adapter` 起步。训练池 2048 图，150 步 × P=2 实际使用 300 张图，GRPO 每图 G=8、输出上限 128；继续 SFT 使用相同图片顺序，lr=1e-5，GRPO β=0.04。val/test 各 100 图，贪心、输出上限 256、像素预算 16384–65536。CPU/MPS/5080 没有性能实测。

| 运行 | RTX 4090（租用）实测 | CPU | MPS | 5080 |
|---|---|---|---|---|
| GRPO，150 步 | 训练 1691.023 s；峰值 allocated/reserved 10.549/14.006 GiB | — | — | — |
| 继续 SFT，150 步 | 训练 95.784 s；峰值 allocated/reserved 5.269/5.979 GiB | — | — | — |
| 起点 val，100 图 | EM_norm 0.10；CER 0.136873；parse 0.65 | — | — | — |
| GRPO val，100 图 | EM_norm 0.10；CER 0.153611；parse 0.69 | — | — | — |
| 继续 SFT val，100 图 | EM_norm 0.09；CER 0.143635；parse 0.67 | — | — | — |
| 补充：起点 SFT，val500 | EM_norm 0.094；CER 0.151200；parse 0.724 | — | — | — |
| 补充：原奖励，150 步快照，val500 | EM_norm 0.096；CER 0.144922；parse 0.744 | — | — | — |
| 补充：原奖励，600 步，val500 | EM_norm 0.096；CER 0.143131；parse 0.750；训练 2926.476s，峰值 allocated/reserved 9.366/10.723 GiB | — | — | — |
| 补充：去 tier-1 奖励，600 步，val500 | EM_norm 0.094；CER 0.142407；parse 0.742；训练 2795.971s，峰值 allocated/reserved 9.366/10.707 GiB | — | — | — |

三组 val 均无截断，同一 sample_ids_sha256。按 EM_norm 最大、CER 最小选择起点 SFT；不能宣称 GRPO 胜出。完整置信区间保存在结果 JSON。

训练诊断：150 步中 139 步更新，有效组均值 0.7000，KL 均值 0.002462（仅对非空记录求均值），退化候选比例 0.002917。前/后 20 步奖励均值 0.284375/0.276250，无法据此断言奖励持续上升。固定探针不调用模型，训练前后相同只表明奖励规则不变；50 个 train 答案上所有非空探针均可拿到至少 tier-1，并不是“模型通过投机测试”。任意可解析但不相干的字符串可获 0.2，存在奖励投机风险。

val 人工文本抽查：样本 `84293739ab0dccc7a2f3` 仅下标括号形式改变，前后均匹配；`2f7d91966a957875b95e` 两者都将 e^t 读成 e^{it}，GRPO 又遗漏 b_k 后的 k；`b03f693458987315291e` 的 vert/集合结构仍有错误，继续 SFT 的抽取结果还丢失大段公式。此处只核对保存的标准答案与输出，不冒称复看了原图。详见 `val_changed_samples.json`。

环境复用 Python 3.11、torch 2.14.1+cu130、transformers 5.19.0、peft 0.21.2、trl 1.15.0、accelerate 1.15.0；接续未安装包或下载模型。日志有 processor 参数弃用与中文字体警告。奖励使用 mathtext 可解析性及规范化字符串/编辑距离，并未实现渲染图像相似度，也不代表数学等价。

流程局限：已有独立冒烟为 Qwen3.5 的 2/6 步、Qwen3-VL 的 4 步；未达到原要求“主训练前 20–50 步阻断检查”。主运行前 20 步只能作为事后诊断，不能替代事前检查。单种子、小样本结果不支持普遍优劣结论。

日志与结果：[训练汇总](../../../outputs/book/ch34/gpu/training_summary.json)、`outputs/book/ch34/gpu/runs/ch34/`、`outputs/book/ch34/gpu/logs/`。

最终 test（同一轮，起点与所选均为同一适配器）：em_norm=0.050000（95% CI 0.021544–0.111750）；cer=0.140137（95% CI 0.110431–0.170293）；parse_rate=0.690000（95% CI 0.593739–0.772203）；两路指标一致，截断 0。全部 CUDA 任务墙钟合计 3905.385 s，各任务墙钟记录见 [runtime_audit.json](../../../outputs/book/ch34/gpu/runtime_audit.json)。全量测试 866 passed、1 skipped。


## 补充实验（val-only，test 未再使用）

上表“补充”行均为本项目观察，RTX 4090（租用），bf16，seed=0。两臂从同一起点独立训练600步，P=1/G=4，训练池2048、同序用600图；lr=1e-5、β=0.04、训练生成上限128。val为latex-v1独立缓存的相同500图、batch=16贪心、上限256。原奖励150步行是600步计划中的快照。与旧P=2/G=8、batch=1实验不作直接等价比较。先完成G=8两种奖励各20步冒烟，再做G=4去奖励20步冒烟，按预算缩小组规模后才启动主训练；原冻结结果保持不变。

新参数：`--reward-mode layered|no_parse_bonus`（默认layered；后者只将tier-1奖励置零）、`--checkpoint-steps 150`、`--eval-batch-size 16`（默认1）。复现完整正式命令见 `outputs/book/ch34/gpu/supp/main.sh`，配置、原始日志与数据划分核验均在supp中。

150→600步的ΔCER=-0.001791，配对95%区间[-0.007074,0.003037]；去奖励600−原奖励600的ΔCER=-0.000724，区间[-0.004905,0.003862]，两者均无明确额外识别收益。原奖励/去奖励的有效更新为310/257步，KL均值0.004816/0.003508。固定无关字符串探针奖励从0.2变为0，但局部“parse变真、距离变大”仍有3/1条，不能说已消除所有投机风险。全局parse提高、CER点估计下降，没有全局退化证据；两臂相对起点的CER配对区间也包含0。

补充CUDA任务墙钟总计7603.295秒（2.112026小时），含所有冒烟和推理检查，非纯内核时间。所有任务reserved峰值13.576 GiB。原test、selection与锁文件哈希不变，未再评test，未下载或安装新依赖。补充全量测试876 passed、1 skipped、14 warnings（13.18s）。完整95%区间见 [补充结果](../../../outputs/book/ch34/gpu/supp/summary.json)，训练曲线见 [training_curves.png](../../../outputs/book/ch34/gpu/supp/training_curves.png)，样例见 [val_changed_samples.json](../../../outputs/book/ch34/gpu/supp/val_changed_samples.json)，限制见正文 §34.7.7。
