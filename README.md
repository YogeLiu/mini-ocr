# Mini OCR：从零到公式 OCR

一本中文教材：从自动求导开始，一步步训练语言模型、做后训练，最后微调视觉语言模型，完成公式图片转 LaTeX。

**[进入课程目录](course-v4/README.md)** · [第 0 章：导论](course-v4/book/ch00_intro/chapter.md) · [大作业](course-v4/assignments/README.md) · [项目](course-v4/project/README.md)

## 内容

- 第 0 章导论、四个部分，共 37 章，另有五篇附录；
- `miniocr` 共用代码库、逐章实验与测试；
- 5 份大作业 A1–A5、每部分的闭卷自测；
- 贯穿全书的 TexPixel 公式识别项目和报告模板。

## 开始学习

环境要求见[课程目录](course-v4/README.md)。在 `course-v4/` 下运行：

```bash
cd course-v4
python -m pytest -q
python -m tools.make_student ../my-course
```

`my-course/` 是生成的个人练习副本，练习函数会被挖空。建议独立完成练习后再看参考实现。

## 仓库收录范围

仓库只同步当前课程。旧版本、个人练习副本、内部规划、虚拟环境、缓存、训练输出和模型检查点不随仓库分发。

正文引用的 `outputs/` 文件是实验的输出，不在仓库里，可以按教材命令重新生成。不生成它们时，依赖这些文件的测试会跳过或失败，以实际运行结果为准。
