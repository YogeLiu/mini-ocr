# 从像素到 Xiaomi-OCR-0：系统学习视觉语言模型与 OCR

这是一本从零开始的自学教程。目标是：**不读其他材料，也能系统地学会视觉语言模型（VLM）与 OCR 的原理和训练**，最后读懂一个真实的 OCR 模型（Xiaomi-OCR-0）的完整训练配方，并在小规模上复现它的关键做法。

- 先学**通用**的 VLM 与 OCR 方法，Xiaomi-OCR-0 作为最后的案例；
- 两个合成的"玩具世界"贯穿全书：**ShapesWorld**（几何场景，学通用 VLM）和 **MiniDoc**（字符、文字行、公式、表格、页面，学 OCR）；
- 一个逐章生长的代码库 `miniocr`：从图像数组、自动求导，一直写到 VLM、DPO、GRPO；
- 正文引用的数字都是实测的，从不编造。第 0–19 章的训练实验由作者跑过，日志在 `outputs/`；从第 20 章起，**需要训练的实验由你自己跑**：正文给出命令、每一部分在做什么、运行前要回答的预测问题、跑完后看什么，以及记录结果的表格。不需要训练的部分（真实模型的推理、指标的计算）仍然列出实测结果。

不涉及：视频、语音、多语言文字。

## 从哪里开始

1. 读 [第 0 章](ch00_overview.md)，按 0.9 节搭好环境；
2. 在本目录下运行：
   ```bash
   python -m labs.ch00.check_env      # 环境检查
   python -m pytest                   # 全部自动检查，应该全部通过
   ```
3. 生成你自己的练习副本：
   ```bash
   python -m tools.make_student ../my-course    # 所有练习的函数体被挖空
   python -m tools.make_student --list          # 查看练习清单
   ```
4. 每章的节奏：**读 → 预测 → 运行 → 对照 → 自己写（在 my-course 里）→ 自测**。

**规则**：永远在本目录（或 `my-course`）下，用 `python -m 模块名` 运行脚本，例如 `python -m labs.ch02.pixels`。

## 目录

★ 为 VLM/OCR 主线的重点章节。

| 部分 | 章 |
|---|---|
| 第 0 部分　准备 | [0 ★ 全景：VLM 与 OCR 是什么](ch00_overview.md) · [1 Python 与 NumPy 从零](ch01_python_numpy.md) · [2 ★ 图像就是数组](ch02_images_as_arrays.md) · [3 ★ 文本、字符集与结构化输出](ch03_text_and_structured_output.md) |
| 第一部分　数学 | [4 向量与矩阵](ch04_vectors_matrices.md) · [5 导数与梯度](ch05_derivatives_gradients.md) · [6 概率与交叉熵](ch06_probability_cross_entropy.md) |
| 第二部分　神经网络 | [7 梯度下降](ch07_gradient_descent.md) · [8 自动求导](ch08_autograd.md) · [9 PyTorch](ch09_pytorch.md) · [10 训练循环](ch10_training_loop.md) · [11 优化与训练技巧](ch11_optimization.md) · [12 卷积神经网络](ch12_cnn.md) |
| 第三部分　序列与语言模型 | [13 ★ 文字行识别（CTC）](ch13_line_recognition_ctc.md) · [14 分词与语言建模](ch14_tokenization_lm.md) · [15 注意力](ch15_attention.md) · [16 ★ GPT](ch16_gpt.md) · [17 生成与 KV cache](ch17_generation.md) · [18 训练工程](ch18_training_engineering.md) |
| 第四部分　视觉编码器 | [19 ★ ViT](ch19_vit.md) · [20 ★ 对比学习（CLIP/SigLIP）](ch20_contrastive.md) · [21 ★ 检测与版面分析](ch21_detection_layout.md) |
| 第五部分　视觉语言模型 | [22 ★ 搭一个 VLM](ch22_build_vlm.md) · [23 ★ VLM 的任务与指令数据](ch23_vlm_tasks.md) · [24 ★ 真实 VLM 解剖](ch24_real_vlm.md) |
| 第六部分　评测与监督微调 | [25 ★ 评测](ch25_evaluation.md) · [26 ★ VLM 的监督微调](ch26_sft.md) · [27 回答的对数概率](ch27_logprobs.md) · [28 ★ 显存与 LoRA](ch28_memory_lora.md) |
| 第七部分　偏好与强化学习 | [29 强化学习基础](ch29_rl_basics.md) · [30 ★ DPO](ch30_dpo.md) · [31 ★ GRPO 与可验证奖励](ch31_grpo.md) · [32 ★ 遗忘与能力配比](ch32_forgetting.md) |
| 第八部分　文档 OCR 与真实世界 | [33 ★ 文档解析与数据工程](ch33_doc_engine.md) · [34 ★ 案例精读：Xiaomi-OCR-0](ch34_xiaomi_ocr.md) · [35 ★ 搬到真实模型](ch35_real_models.md) · [36 ★ 毕业项目](ch36_capstone.md) |
| 大作业 | [总览](assignments/README.md) · [A1 基础（第 12 章后）](assignments/a1_basics/handout.md) · [A2 序列（第 18 章后）](assignments/a2_sequences/handout.md) · [A3 VLM（第 24 章后）](assignments/a3_vlm/handout.md) · [A4 评测与 SFT（第 28 章后）](assignments/a4_eval_sft/handout.md) · [A5 对齐（第 32 章后）](assignments/a5_align/handout.md) · [A6 数据引擎（第 34 章后）](assignments/a6_data/handout.md) |
| 附录 | [A 公式速查](appendix_a_formulas.md) · [B 静默错误与检查清单](appendix_b_silent_errors.md) · [C 术语表](appendix_c_glossary.md) · [D 环境与排错](appendix_d_troubleshooting.md) · [E 各章实测结果](appendix_e_results.md) · [F 数学推导](appendix_f_math.md) |

本仓库包含当前版本的完整课程；内部课程规划与个人研究计划不随仓库分发。

## 代码结构

```
course-v3/
├── chNN_*.md              各章正文
├── miniocr/               逐章生长的代码库（参考实现；练习用 "# >>> 练习 N.M" 标出）
│   ├── data/              渲染器、图像操作、公式语法、ShapesWorld、文字行、页面、表格、指令与偏好数据（第 2–3、13、21–23、30 章）
│   ├── nn/                micrograd、卷积、归一化层、注意力（第 8、11–12、15 章）
│   ├── models/            MLP、CNN、CRNN、GPT、ViT、CLIP、CenterNet、VLM、LoRA（第 10–28 章）
│   ├── train/             训练循环、优化器与学习率、语言模型、SFT 的标签与打包、显存估算、配比与重放（第 10–32 章）
│   ├── rl/                对数概率、策略梯度、rollout、DPO、奖励、GRPO（第 27–31 章）
│   ├── eval/              区间与配对比较、校准、CDM、TEDS、整页评分（第 25、33 章）
│   ├── engine/            数据引擎：共识、渲染校验、挖掘、清洗与去重（第 33 章）
│   ├── xiaomi/            Xiaomi-OCR-0 的简化版：文字锚定、MTP、Mix-RL（第 34 章）
│   ├── hf_vlm.py、hf_train.py   真实模型的读取与训练辅助（第 24、35 章）
│   └── research.py        毕业项目的工具：运行身份证、配对比较、种子表、分桶（第 36 章）
├── labs/chNN/             每章的演示脚本和自动检查（test_*.py）
├── tools/                make_student.py 生成挖空版练习代码；appendix_tables.py 生成附录 B、C 的表格
├── requirements.txt       全书的基础库
└── requirements-hf.txt    第 24、35 章（以及第 33、36 章读真实数据的部分）需要的 Hugging Face 库
```

演示脚本的输出（图片、日志、模型文件）保存在 `outputs/chNN/`。这个目录已加入 `.gitignore`，不随教程分发，可随时重新生成。正文里"日志在 `outputs/…`"指的是作者机器上的日志，关键输出都已摘录在正文和附录 E 里。真实模型的读法等需要随教程分发的数据放在 `labs/chNN/data/`。
