# A4 书面报告

> 复制成 `writeup.md` 再写。每个实验：**预测（运行前写）→ 结果（图、表）→ 观察 → 解释**。
> 数字都要能在你的 `log.jsonl` 或 `eval` 输出里找到；每个比较都要附区间或 p 值；只跑过一次的结论标明"单次运行"。

姓名 / 日期：

## 0. 代码说明

- `a4/` 的文件结构：
- 入口的用法（sft、eval、compare、memory、predict）：
- 每个入口的 `--smoke` 输出：
- 测试结果（`python -m pytest -q assignments/a4_eval_sft/tests` 的最后一行）：

## teds_by_hand

| 预测 | 最少的编辑 | 距离 | TEDS | TEDS-S |
|---|---|---|---|---|
| 1. 合并第一行 | | | | |
| 2. d → dd | | | | |
| 3. 丢了第二行 | | | | |
| 4. a、b 对调 | | | | |

分析：

## mcnemar_by_hand

1. 2. 3.

## memory_accounting

1.

| 部分 | 参数量 |
|---|---|
| 嵌入 | |
| 每层：注意力 / MLP / 两个 RMSNorm | |
| 32 层 | |
| 最后的 RMSNorm、输出层 | |
| **合计** | |

2.

| 方式 | params | grads | optim | master | 合计 |
|---|---|---|---|---|---|
| 全参 fp32 | | | | | |
| 全参 bf16 + fp32 主权重 | | | | | |
| LoRA r=16（q、k、v、o） | | | | | |

3. 4.

## exp_protocol

(a) 评测协议：

(b)

| 任务 | 50 题：准确率 [区间] | 200 题 | 1000 题 |
|---|---|---|---|

(c)

## exp_prompt_weight

预测：

| 提示权重 | 九任务平均（± 标准差） | read CER | 变化最大的任务 |
|---|---|---|---|
| 0 | | | |
| 0.1 | | | |
| 1 | | | |

(b) (c)

## exp_sft_details

(a) (b) (c)

## exp_lora

预测：

| 配置 | 可训练参数 | 静态内存 | 激活值 | 毫秒/步 | 九任务平均 [区间] | read CER [区间] |
|---|---|---|---|---|---|---|
| 全参 | | | | | | |
| 只训练连接器 | | | | | | |
| LoRA r=1 | | | | | | |
| LoRA r=4 | | | | | | |
| LoRA r=16 | | | | | | |
| LoRA r=64 | | | | | | |
| LoRA r=?（只 qkv、o） | | | | | | |

(b) (c) (d)

## exp_calibration

| 置信度 | ECE | 找错例的 AUROC |
|---|---|---|
| 对数概率之和 | | |
| 每 token 平均 | | |
| 最小 token 概率 | | |

(b) (c)

## leaderboard
