# 第 23 章实验：DPO 与偏好对

## 0. 准备

在仓库根目录执行：

```bash
cd course-v4
CUDA_VISIBLE_DEVICES="" python -m pytest -q book/part3_post_training/ch23_dpo
# 部分 1：BT/DPO 手算值、梯度方向、一步自写 DPO（tiny 随机模型，秒级，不需要权重）
CUDA_VISIBLE_DEVICES="" python -m book.part3_post_training.ch23_dpo.tour 1 --out outputs/book/ch23/cpu
```

GPU 部分的依赖关系：部分 2 需要第 21 章选定 step 的 SFT adapter；部分 3 需要部分 2 的候选；部分 4–6 需要部分 3 的偏好对。以下用 `$RUN_DIR` 表示运行产物的根目录，`$LOG_DIR` 表示日志目录，运行前请设为自己的路径。第 23 章产物写到 `$RUN_DIR/ch23/`，日志可保存到 `$LOG_DIR/ch23_*.log`。

```bash
ADAPTER=$RUN_DIR/ch21/main/step-XXX   # 以 ch21 selection.json 为准
OUT=$RUN_DIR/ch23/main
python -m book.part3_post_training.ch23_dpo.tour 2 --device cuda --dtype bf16 \
  --adapter $ADAPTER --prompts 2048 --train-offset 2048 --k 4 --sample-batch 64 --out $OUT
python -m book.part3_post_training.ch23_dpo.tour 3 --min-gap 0.05 --out $OUT
python -m book.part3_post_training.ch23_dpo.tour 4 --device cuda --dtype bf16 --beta 0.1 \
  --steps 75 --batch-size 8 --accumulation 2 --lr 5e-5 --max-length 384 --out $OUT
python -m book.part3_post_training.ch23_dpo.tour 5 --device cuda --dtype bf16 --beta 0.1 --n 500 --gen-batch 16 --out $OUT
python -m book.part3_post_training.ch23_dpo.tour 6 --n 500 --sft-preds $RUN_DIR/ch22/main/preds-sft.jsonl --out $OUT
```

精度口径：SFT 合并后以 bf16 保存作为策略与 reference 的起点；LoRA 参数为 FP32，训练用 bf16 autocast。reference 为关闭 LoRA 的 SFT 合并模型。

## 1. 机制检查

部分 1 用手算值检查 Bradley-Terry 与 DPO 的损失：

- 两个分数相等时，BT 损失为 log 2；
- policy 与 reference 相同时，DPO 第一步损失为 log 2，隐式奖励为 0；
- chosen 的 log-prob 增大时损失减小，梯度为负。

一步自写 DPO 在 tiny 随机模型上运行三步，检查第一步损失为 log 2、之后下降。这只验证机制，不代表效果。

`miniocr/train/preference.py` 的 `# >>> 练习 23.1` 标记的是 BT 与 DPO 损失，要求学习者从零写出。部分 4 的 TRL 日志只记录批平均：令 `r_c`、`r_r` 为日志中的 `rewards/chosen`、`rewards/rejected`，只能检查凸性下界 `loss >= -logsigmoid(r_c - r_r)`，不能要求两者相等（正文 §23.6.5）。同一批偏好对上的逐对损失比较，只在把 TRL 的 Triton 融合内核换成 CPU 实现后做过（tiny 随机模型，四对的差都是 0，本项目观察）；原版内核的数值路径和训练更新没有验证，范围和结果文件见正文 §23.6.5。

## 2. 候选采样

部分 2 合并第 21 章的 SFT adapter，对 train 中第 2048 条之后的 2048 个新提示各采样 4 个候选（温度 1.0，无 top-p/top-k）。这些提示没有进入第 21 章的 SFT 训练，因此偏好对不是对训练样本的记忆。

## 3. 偏好对构造

部分 3 按候选与标准答案的规范化字符编辑率构造偏好对：

- chosen 是编辑率最低的候选（并列时取较短者），rejected 是编辑率最高的候选；
- 两者编辑率之差小于 0.05 时丢弃该提示；
- 分词边界检查：提示 token 必须是 chosen/rejected 完整序列的前缀，否则丢弃。

统计项包括 chosen 的精确匹配率、chosen 比 rejected 长的比例、平均长度与 rejected 的格式问题率。这些统计用来观察长度偏好，长度只用于编辑率并列时的排序，格式不参与配对。

## 4. TRL DPO 训练与 β 扫描

偏好对只有 387 对（见第 3 节的统计），每步最多 16 对（batch 8 × 累积 2），训练 75 步，约 3 轮。未 drop_last：每轮 48 个满 microbatch 加末批 3 对，第 25 个 optimizer step 只有 3 对。

部分 4 使用 TRL `DPOTrainer`，LoRA 只训练 `q_proj`/`v_proj`（r=16）。β 扫描取 0.05、0.1、0.3，每个 β 单独运行一次。训练记录 `loss`、`rewards/chosen`、`rewards/rejected`、`rewards/margins`、`rewards/accuracies`、`logps/*`。

## 5. val 评测与选择

部分 5 在 val 上生成（同第 22 章协议）。部分 6 按固定规则选 β：val EM_norm 最高者，并列取较小 β。选定后与第 22 章的 SFT 预测做配对比较。本章没有评 test。评测使用第 0 章 §0.9.3 的文本后纠错面板：EM_norm、CER、改坏、fix_rate；完整结果见正文 §23.8.3。

## 6. 读结果的顺序

1. 训练日志：margin 是否上升，`rewards/accuracies` 是否超过 0.5。
2. val：EM_norm 是否超过 SFT，配对差的区间是否包含 0。
3. β：较大的 β 是否让模型偏离 SFT 更少（看 CER 与输出长度）。
4. 偏好对的长度统计：如果 chosen 普遍更长，DPO 可能学到的是长度而不是正确性。

## 7. 记录表

本项目观察，RTX 4090（租用），bf16 autocast，LoRA r=16（q/v），学习率 5e-5 常数，偏好对 387 对（来自 2048 个新提示，每个 4 个候选，min_gap 0.05），75 步（batch 8 × 累积 2，约 3 轮），val n=500，贪心解码。SFT 对照使用第 22 章的同一批 val 预测。

| 设置 | 训练末段 loss | 末段 margin | 末段 logps/chosen | val EM_norm（95% 区间） | val CER | 与 SFT 配对差 |
|---|---|---|---|---|---|---|
| SFT（第 21 章 step 400，第 22 章预测） | — | — | — | 0.678 [0.638, 0.718] | 0.0257 | — |
| DPO β=0.05 | 0.286 | 1.87 | −11.43 | 0.414 [0.372, 0.458] | 0.0505 | −0.264 [−0.308, −0.216] |
| DPO β=0.1 | 0.210 | 2.54 | −5.62 | 0.590 [0.546, 0.632] | 0.0234 | −0.088 [−0.122, −0.050] |
| DPO β=0.3 | 0.101 | 4.49 | −2.35 | 0.688 [0.646, 0.727] | 0.0164 | +0.010 [−0.018, 0.038] |
| **DPO β=1.0（val 选中）** | 0.038 | 7.52 | −1.28 | **0.706 [0.665, 0.744]** | 0.0228 | **+0.028 [0.004, 0.052]** |

- 选择规则（运行前固定）：val EM_norm 降序，并列取较小 β。扫描最初为 {0.05, 0.1, 0.3}，最优在边界（0.3，0.688）；看到 val 结果后把 β=1.0 加入扫描，并以同一规则选中它。新增的 β 只用 val 决定，test 未使用。
- 配对差为 DPO − SFT（同一批 val 500）。日志首末 chosen log-prob 来自不同训练批次（末步还可能只有 3 对），不能解释为同一批 chosen 概率下降；这需要固定偏好对探针。本表只支持 β=0.05 的 val EM 明显低于 SFT。β=1.0 的区间不包含 0，但它是在 val 上选 β 得到的数字，存在选择偏差；本章没有在 test 上确认。β=1.0 位于扫描上界，更大的 β 未测。
- 偏好对的统计（用于判断长度与格式偏好）：chosen 的精确匹配率 0.581，rejected 为 0；chosen 比 rejected 长的比例 0.388；平均长度 108.2 与 107.7，基本没有长度偏好；rejected 的格式问题率 0.044。
- 偏好对的筛选：2048 个提示里 941 个的 4 个候选完全相同（采样多样性低），1661 个因编辑率差小于 0.05 被丢弃。

CPU/MPS/5080 的对应列暂留空白。
