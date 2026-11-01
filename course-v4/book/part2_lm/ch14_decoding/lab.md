# 第 14 章实验手册：解码策略

## 0. 准备

在 `course-v4/` 下运行：

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q book/part2_lm/ch14_decoding
CUDA_VISIBLE_DEVICES="" python -m book.part2_lm.ch14_decoding.tour
```

模块名后可指定部分号，范围为 **1–7**，可独立运行或按给定顺序组合：

```bash
CUDA_VISIBLE_DEVICES="" python -m book.part2_lm.ch14_decoding.tour 4
CUDA_VISIBLE_DEVICES="" python -m book.part2_lm.ch14_decoding.tour 1 6
CUDA_VISIBLE_DEVICES="" python -m book.part2_lm.ch14_decoding.tour 5 --ckpt outputs/book/ch13/runs/<你的run>/checkpoint.pt
```

将 `<你的run>` 替换为实际目录名再执行。默认 CPU、两线程、HF 离线，只读本地语料，不下载。`--device cpu|mps|cuda` 显式选设备；本实验验收使用 CPU。默认 `--seed 0 --samples 50 --smoke-steps 150 --prompt-tokens 3`。`--max-new` 默认取上下文剩余空间与生成上限的较小值，具体见 `tour --help`；显式指定后超出上下文会报错。

**没有 `--ckpt` 时，第 5 部分现场训练 tiny 冒烟模型，只证明流程跑通，数字不代表质量。** 没有本地语料时，会显示降级说明并使用内置样例；实际提示条数可能少于请求条数。提供检查点后，根据 `identity["model"]` 恢复结构，根据 `data_version` 中的训练前缀大小加载分词器；完整分词器哈希不一致会拒绝运行。第 5 部分读自己的第 13 章检查点，不续训该检查点。

本章所有产物写入 `outputs/book/ch14/`：冒烟训练目录为 `smoke-cpu-seed{seed}/`（其他显式设备使用相应设备名），含日志与 `checkpoint.pt`；比较报告为 `decoding-seed{seed}.json`，含模型来源、步数、策略指标和全部样例。同种子重复运行会覆盖相应产物，比较不同检查点前先归档报告。没有 KV cache：每步重算整段上下文。tour 保证提示长度加生成上限不超过 `block_size`；独立调用模块时超长会裁剪窗口，RoPE 位置从零重新编号。

## 1. 五 token 例子：处理流水线（14.3–14.5）

**做什么：** 固定 `x、y、^、+、}` 的五 token logits 和已出现集合，依次运行重复惩罚、温度、top-k、top-p、min-p；逐阶段观察 logits 与重新归一化的概率。然后取消重复惩罚，改变温度并比较熵；比较平坦与尖锐分布的截断集合。

**预测题：** 先把每一步的保留集合写在纸上。温度改变最大值的位置吗？能否改变 top-p 的集合大小？平坦分布中 top-k 是否一定只留 k 个？已经被设为负无穷的位置能否在后续过滤中恢复？

**读哪些输出行：** `例子`、`阶段`、`token`、`logit`、`概率`、`保留集合`、`被屏蔽`、`最终采样分布`、`无重复惩罚`、`熵_nat`、`top_p_08保留数`、`top_k_2保留数`、`选做=min-p`。

**对照代码：** [tour.py](tour.py) 的 `part1`、`show_distribution`；[decoding.py](../../../miniocr/decoding.py) 的 `PROCESS_ORDER`、`process_logits` 和五个独立变换函数；[test_ch14_decoding.py](test_ch14_decoding.py) 的手算与顺序检查。

**怎样读结果：** 每阶段的概率来自该阶段 logits 的 softmax，不能直接沿用前一阶段的概率。top-k 保留阈值并列项；top-p 使用“当前 token 之前的累计概率小于 p”，至少留一个，并把掩码映射回原编号。浮点舍入可能影响恰好相等的边界，显示到四位的小数也不等于内部精度。min-p 的演示输入是重复惩罚、温度之后而尚未经过 top-k/top-p 的分布。

## 2. 重复惩罚的正负号（14.6）

**做什么：** 对相同已出现 token 比较正除负乘的实现和“所有 logit 都除以 penalty”的故意错误实现。

**预测题：** 负数除以大于一的数，是变大还是变小？重复出现同一编号应该受到几次惩罚？如果 penalty 小于一，还是在抑制重复吗？

**读哪些输出行：** `重复惩罚`、`token=}`、`logit`、`概率`、`错误说明`。

**对照代码：** [tour.py](tour.py) 的 `part2`、`wrong_negative_penalty`；[decoding.py](../../../miniocr/decoding.py) 的 `apply_repetition_penalty`；测试中的正负号与重复索引断言。

**怎样读结果：** 惩罚系数大于一时，正确变换使已出现 logit 不增；概率还取决于整行重新归一化，不能仅凭一个 logit 推断全部概率的变化。重复编号按集合处理，gather 读取原值后再 scatter，相同索引只写同一个变换值。小于一的系数鼓励重复。生成时已出现集合包括提示，也包括已离开上下文窗口的历史。

## 3. 贪心与温度采样（14.2、14.3）

**做什么：** 不加重复惩罚，对固定 logits 使用不同温度重复抽样，把经验频率与理论 softmax 对照；同时查看贪心选择。

**预测题：** 温度升高时你预期熵怎样变化？概率最大的 token 是否必然在每次采样中胜出？换种子后哪些输出应保持，哪些会改变？

**读哪些输出行：** `温度`、`抽样数`、`token`、`理论概率`、`经验频率`、`贪心永远选择`。

**对照代码：** [tour.py](tour.py) 的 `part3`；[decoding.py](../../../miniocr/decoding.py) 的 `apply_temperature`、`select_next`。

**怎样读结果：** 用频率误差评估抽样是否符合分布，不要求有限次抽样精确等于理论值。贪心走 argmax，温度必须为正；不要用零温度表示贪心。贪心仍可先应用重复惩罚；并列最大值取原词表中的第一个编号。

## 4. 束搜索与偏短（14.7、14.8）

**做什么：** 在固定 bigram 玩具模型上枚举全部正概率路径，区分 EOS 完成与到上限截断；比较贪心、不同束宽和不同长度惩罚，打印长度惩罚表。

**预测题：** 先乘转移概率，再把乘积转为 log 概率，预测每种排序的第一名。束宽扩大时，为什么答案可能比贪心更短？长度惩罚会在什么条件下改变最终答案？束内剪枝丢失的路径能被最终长度惩罚救回吗？

**读哪些输出行：** `长度口径`、`EOS约定`、`剪枝`、`枚举序列`、`概率`、`log概率`、`长度`、各 `alpha` 得分、`finished`、`truncated`、`贪心结果`、`beam`、`暴力最优`、`生成长度`。

**对照代码：** [tour.py](tour.py) 的 `part4`；[decoding.py](../../../miniocr/decoding.py) 的 `SHORT_BIAS_TRANSITIONS`、`short_bias_toy`、`Hypothesis`、`beam_search`、`brute_force_search`、`gnmt_length_penalty`。

**怎样读结果：** 长度为生成 token 数，含 EOS、不含提示。先按累计 log 概率选本步前 beam 个，EOS 占本步名额，随后移入 finished，之后绝不扩展。生成上限处的存活候选也参与最终比较，标记为截断。长度惩罚仅用于最终排序，这是本章简化，并非 GNMT/HF 实现的完整复刻。枚举表也列出截断路径，不能把每行都当成完整公式。空公式在模型概率排序中合法，却不代表任务质量。

## 5. 在 LaTeX GPT 上比较解码策略（14.9）

**做什么：** 使用自己第 13 章训练好的检查点，在相同验证集前缀上运行各策略，记录多样性、mathtext 可解析率及其区间。按以下步骤动手：

1. 确认自己的 run 目录与数据版本，替换命令中的目录占位符。
2. 运行前写下预测：温度升高时，可解析率与 distinct-2 分别可能往哪个方向变；重复惩罚可能影响哪些结构；长度惩罚可能怎样改变结束行为。写出预测理由和不确定之处。
3. 运行并把输出填入空白表，保留 JSON 报告及模型来源。先检查实际样本数和长度口径，再看指标。
4. 改 `--seed` 重复，分别记录各轮；需要更多样本时增加 `--samples`。比较模型或种子前归档已有报告。

```bash
CUDA_VISIBLE_DEVICES="" python -m book.part2_lm.ch14_decoding.tour 5 --ckpt outputs/book/ch13/runs/<你的run>/checkpoint.pt --samples 50 --seed 0
CUDA_VISIBLE_DEVICES="" python -m book.part2_lm.ch14_decoding.tour 5 --ckpt outputs/book/ch13/runs/<你的run>/checkpoint.pt --samples 50 --seed 1
```

**预测题：** 相同提示与相同检查点下，哪些策略会随种子改变？不同提示为何也能让贪心产生不同输出？长度更长是否一定提高 distinct-n？出现不可解析字符串时，是语法结构、截断，还是 mathtext 不支持该命令？

**读哪些输出行：** `数据来源`、`模型说明`或`检查点路径`、`训练步数`、`指标`、`实际提示条数`、`max_new`、`内容口径`、`截断说明`；每策略的 `条数`、`mathtext可解析数`、`率`、`Wilson区间`、`distinct_1`、`distinct_2`、`unique_ratio`、`平均生成长度`、`ended_with_eos`、`用时秒`、两条 `样例`；最后的 `报告文件` 和 `比较口径`。

| 策略 | 条数 | 可解析率 | Wilson 区间 | distinct-1 | distinct-2 | unique_ratio | 平均生成长度 | EOS 结束数 | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| greedy | | | | | | | | | |
| temperature=0.7 | | | | | | | | | |
| temperature=1.0 | | | | | | | | | |
| temperature=1.3 | | | | | | | | | |
| top-k=20，T=1 | | | | | | | | | |
| top-p=0.9，T=1 | | | | | | | | | |
| greedy，repetition_penalty=1.2 | | | | | | | | | |
| beam=4，alpha=0 | | | | | | | | | |
| beam=4，alpha=0.6 | | | | | | | | | |

**对照代码：** [tour.py](tour.py) 的 `Experiment.prepare`、`part5`、`_sample_prompts`、`_content_report`；[decoding.py](../../../miniocr/decoding.py) 的 `sample_sequences`、`gpt_step_fn`、`decoding_report`、`distinct_n`、`unique_ratio`；[pretrain.py](../../../miniocr/train/pretrain.py) 的 `parse_report`、`wilson_interval`。

**怎样读结果：** 每条提示为 BOS 加验证公式开头若干内容 token；公式不足时取全部内容，不把参考 EOS 放进提示。mathtext 可解析率、unique_ratio 与输出样例使用完整文本（提示内容 + 续写，去掉 BOS/EOS）；distinct-1/2 与平均生成长度只统计生成的续写部分，去掉提示内容、BOS/EOS。按每条提示的实际长度切分，遇到首个 EOS 即停止统计，后面的 EOS 填充不计入；立即结束的续写长度为零。JSON 的 `content_scope` 说明这两种口径，`mean_length` 对应平均生成长度。不同长度提示分别成批，不用填充符冒充上下文。禁止再生成 BOS。没有遇 EOS 的截断样本也计入分母。

束搜索只使用最前面的少量提示，输出会明确实际条数；其余策略用全部所选提示，所以直接比较时既有样本量差异，也有提示子集差异。需要公平对照时把 `--samples` 调小到束搜索使用的同一前缀数量，并保持生成上限相同。**样本量不足或 Wilson 区间重叠时，不下“某策略更好”的结论；换种子重复，并检查样例与结束数。** 区间不重叠也不能单凭这一项推断公式语义正确。mathtext 只支持 LaTeX 子集，可解析率只是可编译率的近似。distinct-n 是跨样本汇总的不同 n-gram 数除以总数，不连接样本边界；`None` 表示无可统计的 n-gram，不是零多样性。

## 6. 投机解码的接受-拒绝（14.10）

**做什么：** 对打印出的目标分布 p 与草稿分布 q，解析计算接受部分与拒绝后残差重抽的总分布，再做固定种子的蒙特卡罗抽样。

**预测题：** 草稿对某 token 赋零概率、目标却赋正概率时，最终还能输出它吗？如果 p=q，残差归一化是否还会被用到？接受率较低会破坏目标分布吗？

**读哪些输出行：** `目标p`、`草稿q`、`解析输出分布`、`与p最大差`、`接受率β`、`蒙特卡罗次数`、`经验分布`、`最大误差`、`经验接受率`、`演示范围`。

**对照代码：** [tour.py](tour.py) 的 `TARGET`、`DRAFT`、`part6`；[decoding.py](../../../miniocr/decoding.py) 的 `speculative_accept_reject`、`speculative_output_distribution`。

**怎样读结果：** 接受质量为逐位置 min(p,q)，拒绝时按正残差重新归一化抽样。解析分布应在浮点误差内等于 p；经验分布有抽样误差。这里只演示单 token 的分布正确性，没有实现多 token 草稿验证，也没有测量加速比，不能从接受率直接推出系统吞吐量。

## 7. 常见错误重现（14.11）

**做什么：** 对比温度顺序错、top-p 忘记 scatter、EOS 后继续扩展、负 logit 也除以 penalty 四种故意错误与正确实现。错误函数只放在 tour 中。

**预测题：** 哪些错误会改变保留集合？哪些会生成不合法的结束序列？如果 EOS 自循环概率为一，累计 log 概率没有变，是否就可以继续扩展？用最大值不在第一个位置的例子，能抓住哪种被有序输入掩盖的错误？

**读哪些输出行：** `阶段=正确`与`阶段=错误`、`保留集合`、`被屏蔽`、每个 token 的 `logit`/`概率`、`EOS处理`、`alpha`、`结果`、`log概率`、`长度`、`长度惩罚`、`score`、`说明`、`重复惩罚`、`错误说明`。

**对照代码：** [tour.py](tour.py) 的 `part7`、`wrong_temperature_order`、`wrong_missing_scatter`、`wrong_expand_eos`、`wrong_negative_penalty`；[test_ch14_decoding.py](test_ch14_decoding.py) 的顺序、乱序索引、EOS 与正负号测试。

**怎样读结果：** 把错误定位到具体步骤，不只比较最终字符串。最终分布需在各自保留集合上重新归一化；忘记 scatter 会把排序位置误当成词表编号。对照 `EOS处理` 的正确/错误行：`alpha=0` 时，正确结果为单个 `<eos>`，错误结果为四个 `<eos>`，log 概率与 score 都约为 −1.2040；虽然分数未变，序列已经错误。再看 `alpha=1.0`：正确结果为 `x ^ 2 <eos>`，log 概率约 −1.4837、长度 4、长度惩罚 1.5、score 约 −0.9891；错误结果为 `<eos> <eos> <eos> <eos>`，log 概率约 −1.2040、长度被虚增至 4、长度惩罚 1.5、score 约 −0.8026，比正确结果更高，从而错误地胜出。EOS 后的自循环即使不改变概率，也不应增加生成长度，更不应参与长度惩罚。用这些最小反例解释自己的实现，并保持公共解码模块只有正确算法。
