# 作业

作业分两层（`COURSE_V5_DESIGN.md` 第 9 节）：

- **机制层**：从空包开始写，受禁用检查（不能用现成的层、损失、优化器、注意力）。哪些目录受检查，写在每个作业的 `rules.toml` 里；
- **训练层**：可以用 transformers、PEFT、TRL、vLLM、accelerate，但必须能逐行解释。门槛都是“待校准”，由你第一次完整训练后确定，并且只依据验证集结果。

| 作业 | 位置 | 对应部分 |
|---|---|---|
| A1 自动求导与训练 | [a1/handout.md](a1/handout.md) | 第一部分 |
| A2 语言模型 | [a2/handout.md](a2/handout.md) | 第二部分 |
| A3 SFT 与 LoRA | [a3/handout.md](a3/handout.md) | 第三部分 |
| A4 DPO 与 GRPO | [a4/handout.md](a4/handout.md) | 第三部分 |
| A5 视觉与小 VLM | [a5/handout.md](a5/handout.md) | 第四部分 |

## 运行

在 `course-v4/`（或 `make_student` 生成的目录）下：

```bash
python -m pytest -q assignments/a1/tests       # 换成你在做的作业；不在默认的 python -m pytest 里
```

参考实现在 `assignments/solutions/`，`make_student` 不会复制它。维护者检查参考实现：

```bash
A_SOLUTIONS=1 python -m pytest -q assignments/a1/tests
python -m assignments.solutions.a2.make_fixtures      # 重新生成 A2 的快照（含交叉核对）
```

## 共同规则

1. 机制实现部分只能 import `miniocr.data` 与 `miniocr.device`；`miniocr` 的其余部分是参考答案。
2. 测试通过 `tests/adapters.py` 调用你的代码；`tests/test_*.py` 不要改。
3. 可以复制自己前面作业的代码，不能 import 前面的作业。
4. 超参只在开发集上选。测试集规则见第 0 章 §0.10.1：选型冻结后，预先登记的候选可各评一次，不据此改任何设置；开发期不看 test 错例。评测脚本用显式开关，并在读数据前写启动记录。
5. 报告里的数字都要能追溯到运行 id、配置、设备、日志行和代码位置。
6. Claude 是助教：讲概念、审代码、给提示；不替你写机制实现部分。
