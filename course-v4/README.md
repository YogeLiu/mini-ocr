# Mini OCR：从零到公式 OCR

一本中文教材：从自动求导开始，一步步训练语言模型、做后训练，最后微调视觉语言模型，完成公式图片转 LaTeX。全书第 0 章导论、四个部分、五篇附录，共 37 章。按每周约 12 小时算，约 40.5 周。

## 从哪里开始

先读 [第 0 章](book/ch00_intro/chapter.md)，再按顺序往下学。每学完一个部分，做这部分的作业和自测，再交项目报告。

| 部分 | 目录 | 作业 | 自测 | 项目交付 |
|---|---|---|---|---|
| 导论 | [book/ch00_intro](book/ch00_intro/chapter.md) | | | M0 |
| 第一部分：基础 | [book/part1_foundations](book/part1_foundations) | [A1](assignments/a1/handout.md) | [自测](book/part1_foundations/self_test.md) | R1 |
| 第二部分：语言模型 | [book/part2_lm](book/part2_lm) | [A2](assignments/a2/handout.md) | [自测](book/part2_lm/self_test.md) | M1 |
| 第三部分：后训练 | [book/part3_post_training](book/part3_post_training) | [A3](assignments/a3/handout.md)、[A4](assignments/a4/handout.md) | [自测](book/part3_post_training/self_test.md) | R2 |
| 第四部分：视觉语言模型 | [book/part4_vlm](book/part4_vlm) | [A5](assignments/a5/handout.md) | [自测](book/part4_vlm/self_test.md) | M2、R3、R4 |
| 附录（了解即可） | [book/appendix](book/appendix/README.md) | | | |

项目的时间表、模板和实验规则见 [project/README.md](project/README.md)，作业规则见 [assignments/README.md](assignments/README.md)。

## 目录

```text
book/          正文。每章有 chapter.md、lab.md、exercises.md、tour.py
assignments/   大作业 A1–A5
project/       TexPixel 项目的说明和报告模板
miniocr/       全书共用的代码库
tools/         生成学习者版本等工具
```

## 环境

本书的实验用 Python 3.12、PyTorch 2.14、transformers 5.19、TRL 1.15、PEFT 0.21、datasets 5.1 跑过。其他版本可能要自己调整。

在本目录下运行：

```bash
python -m pytest -q                          # 全部章节测试，不含大作业
python -m pytest -q assignments/a1/tests     # 某个大作业的测试
python -m tools.make_student ../my-course    # 生成挖空的学习者版本，做练习用这一份
```

GPU 训练、下载模型和最终 test 评测都不在默认测试里。怎样运行，见各章 `lab.md`。
