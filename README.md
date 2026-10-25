# Mini OCR：从像素到视觉语言模型

一套中文自学课程，从 Python、NumPy 和基础数学开始，逐步实现神经网络、Transformer、视觉语言模型与 OCR，最后学习 Xiaomi-OCR-0 的训练方法。

**[进入课程目录](course-v3/README.md)** · [第 0 章：全景与环境准备](course-v3/ch00_overview.md) · [大作业](course-v3/assignments/README.md)

## 内容

- 37 章正文与 6 个附录；
- `miniocr` 参考实现、逐章实验及自动检查；
- 6 份大作业、测试、报告模板与参考解答；
- 课程所需的固定测试数据和小型测试夹具。

## 开始学习

先按照[第 0 章](course-v3/ch00_overview.md)安装 Python 环境与依赖，再在 `course-v3/` 下运行：

```bash
cd course-v3
python -m labs.ch00.check_env
python -m pytest
python -m tools.make_student ../my-course
```

`my-course/` 是生成的个人练习副本。它会挖空练习函数，并排除 `assignments/solutions/`；建议独立完成练习后再查看参考实现。

## 仓库收录范围

仓库只同步当前课程。旧版本备份、个人练习副本、内部研究与课程规划、虚拟环境、缓存、训练输出和模型检查点不随仓库分发。实验输出可按照教材命令重新生成。

作业的 `.npz` 测试数据、A2 的 `bpe.pt` / `gpt.pt` 测试夹具，以及 `labs/ch34/data/` 的固定对照数据属于教学资源，保留在仓库中。第 0 章引用的静态插图保存在 `course-v3/assets/`。
