# 第16章逐章审核

## 阅读记录

- 源文件：`course-v4/book/part2_lm/ch16_training_engineering/chapter.md`
- SHA256：`107960984ccaceb62e3dd360c42ed44b828683813eb08cefc9634d69691411cc`
- 完整通读：1–230、231–460、461–690、691–920、921–1135；共1135行。未用关键词扫描替代通读。
- 只审核，不修改正文。历史GPU实验未复现；做了CPU FLOPs反例。

## 章节主线与约定

本章从浮点表示说明动态范围与精度不同：fp16易溢出、bf16尾数较短，RMSNorm等归约须考虑fp32；loss scaling解决小梯度的范围问题，不增加尾数位。再区分显式master weights与autocast的参数/梯度保存方式，提出静态状态、激活保存、分配峰值是不同口径。

手工MHA的保存激活按aT+bT²估计，假设fp32、无dropout、无checkpoint。块checkpoint保存边界并反向重算；非重入实现可提前停止。梯度累积要按有效token数加权，不能简单平均长度不同的microbatch均值。训练FLOPs采用6ND加完整矩阵注意力项12LBT²d；MFU的有用模型计算与重算实际计算不同。最后把静态状态、激活、实测CUDA/MPS指标和硬件校准区分开，概述DDP、ZeRO/FSDP、TP、PP。

## 已确认问题

### CH16-01（P2）：DDP“显存不省”缺少固定批大小口径

- 源位置：1049行，“显存不省，只加速”。
- 理由：DDP不分片参数/梯度/优化器状态成立，但固定全局批时将数据分到K卡，每卡局部B变小，激活可能减少。这也与本章前面激活随B近似线性的分析冲突。通信缓冲会增加占用，但不能据此宣称所有场景都不省。
- 反例：忽略缓冲时每卡内存S+aB_global/K，小于单卡S+aB_global（a>0，K>1）；这不是声称完整模型状态被切分。
- 建议：改为“不分片模型状态；固定全局批时每卡激活可减少，保持每卡批大小时则通常不减少该部分”。同时说明输入分片由用户/sampler负责。

### CH16-02（P2）：checkpoint的4/3被提升成普遍数学上限

- 源位置：726行“上限（数学结论）”；778行“上限约33%”；1096行再次称“上限”。
- 理由：原计算F+B，额外重算R，比例为1+R/(F+B)。只有假定B≈2F且R≤F，才能得到约4/3；B≈2F本身是可训练矩阵乘法的近似，不能产生所有训练场景的上限。冻结权重但仍向输入传梯度时，矩阵乘法反向可只有一次乘法，B≈F，完整重算会达到1.5倍。
- CPU证据：PyTorch `FlopCounterMode`，x形状[8,32]且requires_grad=True，冻结w形状[32,32]，对`(x@w).sum()`反传；普通版32768 FLOPs，`checkpoint(lambda z:z@w,x,use_reentrant=True)`版49152 FLOPs，比值1.5。这里使用完整重算作为“全部重算上限”的反例，不是说本章非重入实测25%有错。
- 建议：保留全参数训练下约33%的估计，删去无条件“数学上限”，给一般式1+R/(F+B)，指出冻结/LoRA、算子结构与early-stop会改变比例；FLOPs比例也不等于墙钟时间比例。

## 待验证与已声明局限

本章已明确16 bytes/parameter是状态估算而非峰值；静态+激活也不能机械当作严格峰值上下界。不能再以“遗漏所有临时分配”为新错误。MPS即时占用不是峰值、CPU性能不能代替GPU吞吐、硬件阈值待实测、历史数据未形成全面基准，均已说明。显式fp32 master weights教学路径与autocast路径也已分别解释，不能把前者误认成PyTorch所有训练的强制存储方式。

## 跨章关联

|目标章节|本章依赖或输出|需宏观复核的命题|
|---|---|---|
|ch13|6ND及完整注意力矩阵FLOPs|ch13的因果半矩阵系数与本章实算系数是否清楚区分|
|ch19、ch21|有效token加权累积|不同长度response与padding的loss分母是否贯通|
|ch20、ch33|冻结参数、可训练参数Nt与master weights|冻结反传成本及精度路径是否支持相同资源估算|
|附录A|DDP不分片状态、ZeRO/FSDP分片|固定全局批与每卡批的内存口径是否一致|
|附录B、E|手工注意力保存激活与实际峰值测量|融合注意力不能直接套用手工T²保存量|

## 事实核查与边界

- 已打开并定位[PyTorch checkpoint文档](https://docs.pytorch.org/docs/2.15/checkpoint.html)中非重入early_stop及随机状态保存说明，支持本章关于提前终止和dropout重算的描述。
- 已打开并定位[AMP示例](https://docs.pytorch.org/docs/2.15/notes/amp_examples.html)中unscale后clip、累积期间scale不变的说明，支持本章流程。
- 已打开并定位[DDP文档](https://docs.pytorch.org/docs/2.15/generated/torch.nn.parallel.DistributedDataParallel.html)关于同步副本梯度、输入分片由用户负责的说明。
- 文档为核查时公开2.15版本，不声称重跑全部历史版本API；CPU小反例只证明一般“上限”不成立，不验证历史训练速度。

## 本章结论

确认2处P2，均为把带条件估计写成一般结论；其余精度、累积、显存口径的主要叙述能够自洽。跨章表仅登记待核对契约，不代替全书结论。
