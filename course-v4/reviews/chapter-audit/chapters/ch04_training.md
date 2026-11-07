# ch04 训练循环与优化：逐章审查

## 阅读与版本

- 源文件：`course-v4/book/part1_foundations/ch04_training/chapter.md`
- SHA256：`94f73e91657f823579c8eb50c4d5019cae55ff9f0e3a0429a5375d0f5a0b4d60`
- 完整逐行阅读范围：1–240、241–480、481–720、721–960、961–1222。
- 验证：更新公式代数审查、PyTorch 2.14.1 CPU 小案例、实际打开 Python/PyTorch 官方文档与原始论文页面。历史训练、GPU一致性、物理断电场景均未重新执行。

## 章节主线与关键协议

章节用螺旋 MLP 将前向、清梯度、反向、全局裁剪、设置当前步学习率、更新及记录串成循环。SGD 使用 `v=μv+g;θ-=ηv`，并区分把学习率放进速度的另一种约定。二次正定例子的稳定界是 `0<η<2/λmax`，heavy ball 对应 `η<2(1+μ)/λmax`，没有把这些作为所有非凸网络的全局定理。

Adam 维护一阶/二阶 EMA，按参数分别计步，None 梯度跳过更新；有条件地推导偏差修正、首步近似sign及小梯度受epsilon影响。AdamW把衰减放在Adam更新前。无动量SGD的L2与本章按学习率缩放的衰减等价，Adam则不等价。调度使用0基步数，warmup含s+1，cosine在total_steps以后固定min_lr。裁剪返回裁剪前全局范数。初始化明确区分二阶矩与方差、Xavier论文公式与本章等宽特例。

Dropout采用inverted dropout，train/eval与梯度开关分开。检查点保存模型、优化器配置与状态、已完成步数和全局/显式Generator状态；数据使用有放回随机采样，因此不另存DataLoader游标。逐位续训承诺限定完全相同运算序列；真实DataLoader不展开。小批量过拟合仅检查可学习性，正文已展示随机错标签也能背下来，不能据此证明标签正确。

## 已确认问题

### CH04-01 · P2 · 原子重命名不足以保证“断电后总有完整新/旧检查点”

**位置：**923–925行只有 `torch.save(ckpt,tmp);os.replace(tmp,path)`；939行“写到一半断电……os.replace……要么完全是旧文件，要么完全是新文件”。

**理由与证据：**同一文件系统的原子rename解决的是名字切换的原子可见性，不等于文件数据和目录项已经持久写入存储。save返回和rename成功以后，尚在缓存中的数据/目录更新仍可能在断电时丢失。Linux fsync文档明确区分文件同步与包含它的目录同步；Python os文档也分别提供fsync和replace。正文以断电场景解释代码，却把两种保证混在一起。

**建议：**教材保持简单时改成“临时文件加原子替换可防止写入中的进程崩溃直接覆盖旧检查点；这不是断电持久性保证”。如确需断电保证，应说明文件flush/fsync、原子替换、目录fsync及文件系统/硬件条件，并考虑保留上一版。本次仅做接口与协议核查，没有实际断电复现。

### CH04-02 · P3 · RNG状态不clone就会随随机抽样变化的理由不准确

**位置：**909行“存的时候clone()一份，否则之后再抽随机数，记录下来的状态可能跟着变”。903–905行对各get_rng_state返回值clone。

**理由与小验证：**`torch.get_rng_state()`和Generator状态接口返回状态快照，不是后续随机抽样会自动原地更新的活视图。本次CPU：`s=torch.get_rng_state();old=s.clone();torch.rand(10);torch.equal(s,old)`为True。额外clone无害，但不能用并不存在的别名关系解释必要性。需要深拷贝的model.state_dict/优化器状态属于另一个问题。

**建议：**改为“这些接口取得独立的状态快照；这里额外clone作为防御性拷贝”，或者去掉这句必要性论断。该项不影响现有检查点实现的正确性。

## 待验证线索与已有局限

- 729、1193行将“大部分步都裁剪”作为异常信号，过于绝对。是否常裁剪取决于任务、损失尺度和阈值策略，建议作为排查提示，不能单凭频率证明超参有误。未把未指定模型的经验习惯单列为已确认算法错误。
- 857行BatchNorm叙述适用于默认`track_running_stats=True`；关闭该参数时eval也用当前batch统计。若面向接口大全应加默认条件；本章实际实验主要是Dropout，不影响实验结论。
- 1156、1172、1191行`torch.optim.Adam(weight_decay=…)`默认是L2这一结论与2.14一致；但现在有`decoupled_weight_decay=True`选项，不宜进一步写成“Adam类不支持解耦”。本文没有明确否认该选项，记为接口补充。
- 1046行“ReLU永远不激活”须承接第2章的条件限定：前层表示或其他参数更新可能使它恢复，不能把一次全负batch等同永久死亡。
- 1181行Muon与loss Newton法的区分方向正确；没有为验证该介绍去运行Muon训练。1179行把Shampoo等统称利用曲率的方法是宽泛分类，不能理解成它们都显式计算loss Hessian。
- 619行已经明确Adam归一化更新没有普遍上界1；451行已经限定SGD不带动量；761行已经承认Xavier原文和等宽实验差别；963–968行已经限定可复现条件；805行把小初始化解冻机制标成待验证。没有重复指控这些简化。

## 跨章关联表

| 目标章节 | 本章输出/依赖 | 待统一核对的命题 |
|---|---|---|
| 02、03 | 梯度、CE、参数更新 | logits梯度有界不妨碍参数梯度爆炸；ReLU恢复条件一致 |
| 05 | L2、Dropout、小批量过拟合 | 参数大与过拟合只是动机，不能作为充分诊断；训练低验证差不能一概称数据泄漏 |
| 08、09、11 | 方差初始化、Pre/Post-LN、残差初始化 | 二阶矩和方差勿混同；Pre-LN不需warmup不是普适承诺 |
| 13 | 矩阵参数衰减、bias/norm不衰减、warmup_cosine | 后文param groups是否确实按约定；step从0、total边界一致 |
| 16 | Adam状态两份参数张量、累积后更新与清零 | 混合精度master weights会改变显存倍数；梯度累积何时归一化/裁剪 |
| 19、21 | 微调不warmup、Trainer负责循环 | 固定学习率是特定实验配置；不要外推成微调永远不预热 |
| 后续所有训练章 | 已完成step、RNG、优化器状态/配置 | checkpoint继续训练所需采样/数据游标/AMP等状态是否充分；原子替换不可当断电持久性 |

## 实际核查来源与方式

- [Python os文档](https://docs.python.org/3/library/os.html)：实际打开replace与fsync条目，区分原子替换、用户缓冲flush及强制写入。
- [Linux fsync(2)](https://man7.org/linux/man-pages/man2/fsync.2.html)：实际打开，核查文件fsync不保证目录项也同步，目录需要独立fsync。CH04-01依赖此协议事实，不声称进行故障注入。
- [PyTorch 2.14 Adam](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html)：实际打开并查decoupled_weight_decay，默认False，与书中默认L2映射一致。
- [PyTorch get_rng_state](https://docs.pytorch.org/docs/2.14/generated/torch.get_rng_state.html)：实际打开并结合CPU快照试验；没有据单个CPU试验声称所有用户提供随机数容器都无别名问题。
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)：实际打开原作者arXiv页面，确认2017首稿、ICLR2019发表与SGD/Adam不等价主张；正文“2019”按发表年可接受。
- [On Layer Normalization in the Transformer Architecture](https://proceedings.mlr.press/v119/xiong20b.html)：实际打开会议论文页面，核查Post-LN初始化梯度与warmup分析、Pre-LN的实验结论；不借论文摘要背书本项目三种种子的数值。
