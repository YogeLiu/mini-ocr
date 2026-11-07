# 逐章问题索引

本索引汇集逐章报告的编号项，便于定位原文、证据和修改建议。同一根因可能在多章出现；[跨章节总报告](cross-chapter-review.md)已合并这些项。P3和主审核降级记录不等同核心算法错误。尚待查证的线索不列入此表。

| 编号 | 逐章分级 | 问题 | 审核报告 |
|---|---|---|---|
| CH00-01 | P1 | 字符 CER 拼接重新抹掉命令边界 | [ch00_intro](chapters/ch00_intro.md) |
| CH00-02 | P2 | 空白规则的执行层次与适用范围没有在首次说明时说清 | [ch00_intro](chapters/ch00_intro.md) |
| CH00-03 | P2 | 单元素分组剥除会改变控制声明作用域 | [ch00_intro](chapters/ch00_intro.md) |
| CH01-01 | P3 | 同设备要求漏掉 CPU 零维标量例外 | [ch01_tensors](chapters/ch01_tensors.md) |
| CH01-02 | P3 | expand 不是所有原地写入都被禁止 | [ch01_tensors](chapters/ch01_tensors.md) |
| CH02-01 | P2 | 将当前 ReLU 无梯度扩大为永远无法恢复 | [ch02_backprop](chapters/ch02_backprop.md) |
| CH02-02 | P3 | 反向两次必报错的表述过于绝对 | [ch02_backprop](chapters/ch02_backprop.md) |
| CH02-03 | P3 | 梯度误差助手对缺失分量可能报告零误差 | [ch02_backprop](chapters/ch02_backprop.md) |
| CH03-01 | P2 | 从一个 loss 标量反推“只学会类别频率”不成立 | [ch03_softmax_ce](chapters/ch03_softmax_ce.md) |
| CH04-01 | P2 | 原子重命名不足以保证“断电后总有完整新/旧检查点” | [ch04_training](chapters/ch04_training.md) |
| CH04-02 | P3 | RNG状态不clone就会随随机抽样变化的理由不准确 | [ch04_training](chapters/ch04_training.md) |
| CH05-01 | P3 | 实验计划应把正向效应的区间条件写完整 | [ch05_generalization](chapters/ch05_generalization.md) |
| CH08-01 | P2 | 用−1e9修复全遮罩行的建议漏掉fp16不可表示条件 | [ch08_attention](chapters/ch08_attention.md) |
| CH09-01 | P2 | loss接近unigram不足以证明没利用上文 | [ch09_block](chapters/ch09_block.md) |
| CH09-02 | P3 | 将TrOCR直接称作公式到LaTeX模型不准确 | [ch09_block](chapters/ch09_block.md) |
| CH10-01 | P3 | 用按公式平均长度解释按token指标近乎不变，遗漏了实际20.8%的受影响目标 | [ch10_rope](chapters/ch10_rope.md) |
| CH12-01 | P2 | 把规范 token 的混合份额称为实际模型 token 份额 | [ch12_pretrain_data](chapters/ch12_pretrain_data.md) |
| CH13-01 | P2 | 对较高可解析率的原因作了过强断言 | [ch13_pretraining](chapters/ch13_pretraining.md) |
| CH13-02 | P2 | 用Kaplan因果半矩阵口径估算本实现的注意力额外开销时没有就地说明 | [ch13_pretraining](chapters/ch13_pretraining.md) |
| CH14-01 | P2 | 将语料级distinct-n低值直接解释为单条输出复读 | [ch14_decoding](chapters/ch14_decoding.md) |
| CH14-02 | P2 | GNMT 的推荐 alpha 与所用打分公式没有对齐 | [ch14_decoding](chapters/ch14_decoding.md) |
| CH16-01 | P2 | DDP“显存不省”缺少固定批大小口径 | [ch16_training_engineering](chapters/ch16_training_engineering.md) |
| CH16-02 | P2 | checkpoint的4/3被提升成普遍数学上限 | [ch16_training_engineering](chapters/ch16_training_engineering.md) |
| CH18-01 | P2 | R2必填清单漏掉本章规定的fix_rate | [ch18_prompting](chapters/ch18_prompting.md) |
| CH18-02 | P3 | 框架对应表把后处理抽取器对应成生成约束 | [ch18_prompting](chapters/ch18_prompting.md) |
| CH20-01 | P2 | 从一次逐位相等反推运算顺序完全相同 | [ch20_lora](chapters/ch20_lora.md) |
| CH21-01 | P3 | 对比表把可由已有计数确定的fix_rate列为“未记录” | [ch21_trl_sft](chapters/ch21_trl_sft.md) |
| CH22-01 | P2 | 参考也解析失败不能证明输出失败“是解析器的问题” | [ch22_eval](chapters/ch22_eval.md) |
| CH22-02 | P2 | 把mathtext支持的命令列为“不认识” | [ch22_eval](chapters/ch22_eval.md) |
| CH23-01 | P3 | 自动评分不会自动排除长度进入标签 | [ch23_dpo](chapters/ch23_dpo.md) |
| CH24-01 | P2 | 由奖励不可导推出“只能强化学习” | [ch24_rl_basics](chapters/ch24_rl_basics.md) |
| CH25-01 | P2 | 将本项目不使用奖励模型写成GRPO本身的性质 | [ch25_grpo](chapters/ch25_grpo.md) |
| CH27-01 | P2 | 将等效面积推成实际空白token数量 | [ch27_image_tensors](chapters/ch27_image_tensors.md) |
| CH27-02 | P2 | 用平均token数量同量级排除其主要作用 | [ch27_image_tensors](chapters/ch27_image_tensors.md) |
| CH29-01 | P2 | 无位置编码造成的不辨顺序被扩大到patch内部 | [ch29_vit](chapters/ch29_vit.md) |
| CH29-02 | P2 | 将任意行优先编号说成二维空间先验 | [ch29_vit](chapters/ch29_vit.md) |
| CH30-01 | P2 | sigmoid输出缺少采样先验与校准条件 | [ch30_clip](chapters/ch30_clip.md) |
| CH31-01 | P2 | 拆解混杂因素的建议重复了已做设定 | [ch31_mini_ocr](chapters/ch31_mini_ocr.md) |
| CH32-01 | P3 | “连接器唯一能缩短长度”范围过宽 | [ch32_real_vlm](chapters/ch32_real_vlm.md) |
| CH33-01 | P2 | 未检出显著差异不能推出没有变差 | [ch33_vlm_lora](chapters/ch33_vlm_lora.md) |
| CH33-02 | P2 | 配对样本量总是更少的断言不成立 | [ch33_vlm_lora](chapters/ch33_vlm_lora.md) |
| CH34-01 | P2 | 把完美转写得分叫作奖励上限，和本章数字自相矛盾 | [ch34_vlm_grpo](chapters/ch34_vlm_grpo.md) |
| CH34-02 | P2 | parse比例靠近参考不等于忠实 | [ch34_vlm_grpo](chapters/ch34_vlm_grpo.md) |
| CH34-03 | P2 | tier2距离丢token边界，错误命令可得零距离部分分 | [ch34_vlm_grpo](chapters/ch34_vlm_grpo.md) |
| CH35-01 | P2 | 把基线低EM当作数据比较无从显现，和自身CER结果冲突 | [ch35_distill](chapters/ch35_distill.md) |
| APPA-01 | P2 | 声称完整的等价条件漏掉逐样本随机计算一致性 | [a_distributed](chapters/a_distributed.md) |

逐章编号项统计：P1 1项，P2 33项，P3 12项。这是逐章统计，不是去重后的跨章问题数量。

主审核的采纳范围与降级说明见 [章节地图](chapter-map.md) 各章的主审核记录。
