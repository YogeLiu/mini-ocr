# 大作业

各章的练习是"补完一个函数"。大作业仿照 Stanford CS336 的作业：**从一个空的包开始**，一个作业覆盖 3–7 章，自己设计代码结构，用测试检查行为，做实验、写报告，最后参加排行榜。

| 作业 | 位置 | 从零实现 | 实验与排行榜 | 状态 |
|---|---|---|---|---|
| [A1 基础](a1_basics/handout.md) | 第 12 章后（里程碑 A） | 数据管线、Linear、im2col 卷积、池化、BN、交叉熵、AdamW、调度、裁剪、检查点、可续训的训练脚本 | 5 种故障、学习率、batch、数据量、消融、BN 模式；排行榜：10 分钟单字符识别 | ✅ |
| [A2 序列](a2_sequences/handout.md) | 第 18 章后（里程碑 B） | BPE 分词器、CTC 前向后向（对数空间）、束搜索、CRNN；带 RoPE 的 GPT、KV cache | 分词器、CTC 识别器、学习率、消融、分词粒度、生成；排行榜：文字行 CER、语言模型 bits/char（各 15 分钟） | ✅ |
| [A3 VLM](a3_vlm/handout.md) | 第 24 章后（里程碑 C） | ViT（三种位置编码、插值、二维 RoPE）、CLIP/SigLIP 损失、IoU/NMS/AP、三种连接器、聊天模板、视觉 token 填充与生成；复刻 Qwen 的 processor 与 M-RoPE，与 transformers 逐位对照 | 位置编码与分辨率、CLIP 对 SigLIP、连接器、训练流程、错误分析、视觉 token 预算；排行榜：ShapesWorld 九任务问答（30 分钟） | ✅ |
| [A4 评测与 SFT](a4_eval_sft/handout.md) | 第 28 章后 | 编辑距离回溯、CER/WER/NED/BLEU/ANLS、TEDS（Zhang–Shasha）、bootstrap 与配对 bootstrap、McNemar、ECE/AUROC、对数概率与两种 mask、带身份和权重的 SFT、打包与打包的 mask、手写 LoRA、内存估算与激活值测量、梯度检查点 | 评测协议、提示权重、两种平均/打包/eos、LoRA 的秩与全参（带置信区间）、置信度与选择性预测；排行榜：VLM 读文字行 CER（20 分钟） | ✅ |
| [A5 对齐](a5_align/handout.md) | 第 32 章后（里程碑 D） | 组内标准化与留一的优势、KL 的三个估计量、裁剪目标、最优策略与 DPO（梯度系数、成对数据、制造幻觉）、三种汇总方式与 GRPO 损失、DAPO 的改动、带格式闸门的奖励、带温度和 KV cache 的采样、帕累托前沿、重放、RFT、探测 KL | KL 估计量、DPO 的 β 与似然位移、GRPO 的组大小/汇总/DAPO/KL、奖励投机、同一起点同一预算比较 SFT / RFT / DPO / GRPO；排行榜：学了新的不忘旧的（20 分钟） | ✅ |
| [A6 数据引擎](a6_data/handout.md) | 第 34 章后 | LaTeX 规范化、MinHash 与 LSH、近似重复的簇、按组划分、去污染、三元组共识、加权投票与单参数 EM、去噪的渲染校验与重选、自洽度、k-means、按簇配额、簇内抽样权重 | 数据池体检、去重与泄漏、共识的覆盖率–精度、渲染校验在哪里失效、固定配方下标签来源与数量–质量的对照；排行榜：固定配方，比谁的数据好（GPU / CPU 两个赛道） | ✅ |

## 共同的规则

1. **能 import 的只有数据生成器**：`miniocr.data` 里的渲染器、ShapesWorld、MiniDoc。模型、训练、评测相关的代码都要自己写，`miniocr` 里的这些部分就是参考答案。
2. **测试通过适配器调用你的代码**：每个作业的 `tests/adapters.py` 是你要填的"接线"，`tests/test_*.py` 不要改。
3. **可以复制自己前面作业的代码**（不能 import 前面的作业）：比如 A2 可以把你 A1 的 Linear、卷积、AdamW 复制进 `a2/`。
4. **算力以租用的 GPU 为主**：写代码、跑测试、冒烟都在本机 CPU 上；正式实验和排行榜按每个作业说明的 GPU 时间上限来跑（一张 24 GB 显卡，租用流程见各作业的说明，例如 A1 第 9 节）。每个排行榜另设一个时间更长的 CPU 赛道。
5. **数字要诚实**：报告里的每个数字都能在你的日志里找到；只跑过一次的结论要标明；测试集只用来打最终的分。
6. **AI 助手**：可以问概念、问报错、让它检查你的代码；不要让它替你写作业代码。

## 运行

在 `course-v3/`（或 `make_student` 生成的 `my-course/`）下：

```bash
python -m pytest -q assignments/a1_basics/tests     # 作业的测试；不在默认的 python -m pytest 里
python -m pytest -q assignments/a2_sequences/tests
python -m pytest -q assignments/a3_vlm/tests
python -m pytest -q assignments/a4_eval_sft/tests
python -m pytest -q assignments/a5_align/tests
python -m pytest -q assignments/a6_data/tests
```

参考实现在 `assignments/solutions/`，`make_student` 不会复制它。维护者检查参考实现：`A_SOLUTIONS=1 python -m pytest -q assignments/a1_basics/tests`。
