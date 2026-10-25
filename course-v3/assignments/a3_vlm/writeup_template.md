# A3 书面报告

> 复制成 `writeup.md` 再写。每个实验：**预测（运行前写）→ 结果（图、表）→ 观察 → 解释**。
> 数字都要能在你的 `log.jsonl` 里找到；只跑过一次的结论标明"单次运行"。

姓名 / 日期：

## 0. 代码说明

- `a3/` 的文件结构：
- 四个训练入口的用法（clip、lm、vlm 的两个阶段、predict；覆盖配置、`--smoke`）：
- 每个入口的 `--smoke` 输出：
- 测试结果（`python -m pytest -q assignments/a3_vlm/tests` 的最后一行，注明 transformers 的版本）：

## vit_accounting

1.

| 部分 | 参数量 |
|---|---|
| 切块的线性层 | |
| 每层：LN×2 | |
| 每层：q / kv / o | |
| 每层：fc1 / fc2 | |
| 最后的 LN | |
| **合计** | |

2.
3.

| 图 | 缩放后 | patch 数 | 视觉 token 数 |
|---|---|---|---|
| 1000×750 照片 | | | |
| A4 300 dpi | | | |
| 32×2000 文字行 | | | |
| A4，max_pixels = 1280·28·28 | | | |

## siglip_init

1. 2. 3.

## mrope_by_hand

1. 2. 3.

## exp_pos

预测：

| 位置编码 | R@1 / R@5（64×64） | R@1 / R@5（96×96） |
|---|---|---|
| learned | | |
| sincos | | |
| rope2d | | |

(c)

## exp_clip

预测：

| batch | CLIP R@1 | SigLIP R@1 |
|---|---|---|
| 32 | | |
| 128 | | |
| 512 | | |

(b) (c) (d)

## exp_connector

预测：

| 连接器 | 视觉 token | 参数量 | 每步用时 | caption | count | count_color | left | exists | ground | refer | compare | text | 平均 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp | | | | | | | | | | | | | |
| merge | | | | | | | | | | | | | |
| qformer | | | | | | | | | | | | | |

(b) (c)

## exp_stages

预测：

(a) (b) (c)

## exp_errors

(a) (b) (c) (d)

## exp_qwen

(a) (b) (c)

## leaderboard
