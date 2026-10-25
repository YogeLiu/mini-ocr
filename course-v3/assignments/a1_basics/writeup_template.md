# A1 书面报告

> 复制成 `writeup.md` 再写。每个实验：**预测（运行前写）→ 结果（图、表）→ 观察 → 解释**。
> 数字都要能在你的 `log.jsonl` 里找到；只跑过一次的结论标明"单次运行"。

姓名 / 日期：

## 0. 代码说明

- `a1/` 的文件结构，每个文件放了什么：
- 训练命令行的用法（训练、覆盖配置、`--smoke`、续训、预测）：
- 测试结果（粘贴 `python -m pytest -q assignments/a1_basics/tests` 的最后一行）：

## data_dup

| | noise=0 | noise=0.05 |
|---|---|---|
| distort=0 | 训练集互不相同 __ 张；验证集与训练集重复 __ 张 | |
| distort=1 | | |

(a)

(b)

## conv_accounting

1. 参数量（逐层）：
2. 乘加次数（逐层）：
3. 感受野：
4. 第一个卷积层输出的字节数：

## adamw_accounting

1.
2.
3.

## exp_debug

预测：

| 故障 | 症状 | 从曲线上怎么认出来 | 原因 |
|---|---|---|---|
| 忘了 zero_grad | | | |
| 标签打乱 | | | |
| 学习率 ×100 | | | |
| 图和标签错位 | | | |
| 双重 softmax | | | |

图：

## exp_lr

预测：

图（训练损失；最终验证准确率对 max_lr）：

(c)

(d)

## exp_batch

预测：

| batch | 学习率规则 | 达到 __% 验证准确率需要的样本数 | 每秒样本数 | 设备 |
|---|---|---|---|---|

分析：

## exp_data

预测：

图（双对数）：

斜率估计：

(c)

## exp_ablation

| 配置 | n_per_class=20 | n_per_class=200 |
|---|---|---|
| 基准 | | |
| 去掉 BN | | |
| label_smoothing=0.1 | | |
| 随机平移 ±2 | | |
| weight_decay=0 | | |
| weight_decay=0.1 | | |

分析：

## exp_bn_eval

| 评测方式 | 验证准确率 |
|---|---|
| eval() | |
| train()，batch 1024 | |
| train()，batch 8 | |
| train()，batch 1 | |

解释：

## leaderboard

- 最终准确率：
- 硬件（显卡或 CPU 型号）、训练耗时、费用：
- 试过的改动和结果：
- 学习曲线：
