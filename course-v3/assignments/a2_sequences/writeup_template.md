# A2 书面报告

> 复制成 `writeup.md` 再写。每个实验：**预测（运行前写）→ 结果（图、表）→ 观察 → 解释**。
> 数字都要能在你的 `log.jsonl` 里找到；只跑过一次的结论标明"单次运行"。

姓名 / 日期：

## 0. 代码说明

- `a2/` 的文件结构：
- 两个训练入口的用法（训练、覆盖配置、`--smoke`、预测、评测、生成）：
- `--smoke` 的输出：
- 测试结果（`python -m pytest -q assignments/a2_sequences/tests` 的最后一行）：

## ctc_by_hand

1. 8 条路径：

| 路径 | 折叠后 | 概率 |
|---|---|---|

P("") = ，P("a") = ，P("aa") = ，合计 =

2. 贪心解码： ；概率最大的标签：
3. α 表：

## gpt_accounting

1.
2.
3.

## rope_relative

## exp_bpe

预测：

| 词表大小 | 字符/token（验证） | 训练用时 |
|---|---|---|

(c) (d)

## exp_ctc

(a) (b) (c) (d)

## exp_lm_lr

## exp_lm_ablation

| 配置 | 参数量 | 验证 bits/char | 稳定吗 |
|---|---|---|---|
| 基准 | | | |
| 去掉 RoPE | | | |
| Post-Norm | | | |
| 去掉 RMSNorm | | | |
| ReLU MLP | | | |
| 不共享输出层 | | | |

分析：

## exp_tokenization

## exp_generation

## leaderboard_ocr

## leaderboard_lm
