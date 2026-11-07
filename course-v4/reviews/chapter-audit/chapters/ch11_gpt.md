# ch11 搭建GPT与初始化：逐章审查

## 阅读与版本

- 源文件：`course-v4/book/part2_lm/ch11_gpt/chapter.md`
- SHA256：`99e38dd9df8ad0e6827a193feb6782f643dfb702b7d00584fb0aa8986f522385`
- 完整逐行阅读：1–240、241–480、481–720、721–976。
- 核查：公式/形状/计数、项目模型CPU实例及mask不改标签验证，实际打开官方模型配置、nanoGPT源码、原始论文/作者文章。未重跑47秒全tour、350步故障训练或完整预训练。

## 主线与关键定义/协议

GPT由token embedding、L个Pre-Norm块、final RMSNorm与LM head组成。无learned位置表，RoPE共享buffer按pos0切片；forward可直接接inputs_embeds供视觉前缀，要求pos0+T≤block_size，cache逐层传递。标签由数据端移位一次，forward原样算CE；loss_mask=False转负100的新标签张量，不原地改调用者标签。

从本章起训练不屏蔽BOS输出，只忽略BOS目标；验证仍屏蔽BOS、EOS计目标、原文字节不计边界，与ch07–10基线评测一致。195–208行明确区分两操作、训练与验证不是完全相同概率口径，生成屏蔽交调用方。

普通MHA、无bias、RMSNorm、SwiGLU、共享head时参数`Vd+L(4d²+3df+2d)+d`，f上取整64倍数；默认本书V2258,d256,L4,f704共3791616。non_embedding本书减token表，nanoGPT减position表，922行明确两种约定不同，不能误判为代码互相冲突。GQA/QKnorm须改公式，335行已拒绝套用。

Weight tying是同Parameter，不是数值复制，梯度为输入稀疏路径与输出密集路径之和。初始loss≈lnV+σ²/2的推导明确小σ、大V、独立正态logits、目标无关，σ≈s√d另忽略共享相关；共享造成当前token复制偏好会让错位标签在第1步异常偏低，也承认真实重复数据可有相同信号。残差分支o/down初始化std=0.02/sqrt(2L)，方差相加依赖独立简化，实际RMS增长不严格遵循sqrt深度，最终norm又使RMS与logits尺度分开。

固定批次接近0只验证学习链路，不证明目标正确/泛化；批内相同历史冲突可造成合法非零下界。故障训练统一恢复正确评测：错位错误与全可见mask分别示范。因果测试改变未来看过去，在eval或dropout=0及确定性条件下比较；HF CausalLM内部shift与本书数据侧shift不同。

## 已确认问题

本轮没有确认独立的章内P1/P2/P3问题。多个容易误判的差别在正文中已经主动说明，尤其是BOS口径、non_embedding口径、共享初始化相关性、batch冲突、残差缩放理论边界。以下记录证据及跨章需保持的条件。

## 本次验证与待核线索

- 本次CPU构建`GPT(GPTConfig(vocab_size=2258,block_size=128))`：`n_params()`=3791616，`n_params(True)`=3213568，head.weight is tok.weight为True，与正文精确数值一致。
- CPU传入小随机idx/targets和部分False的loss_mask，前向后targets与原副本逐元素相同，支持192行非原地协议。本次随机目标loss8.087不是正文7.7634实验，输入/种子不同，不能混作初始化复现。
- 868行Qwen3.5-0.8B的hidden1024/layers24/vocab248320/intermediate3584/heads8/kvheads2/共享、full_attention_interval4、output gate、partial_rotary_factor0.25，均与本次实际打开的官方config相符。不是纯标准Transformer，正文已明确不能硬套参数公式。其head_dim256也不等于1024/8，这是本章未展开配置的重要后章差异，不能将本书dh=d/h套进去。
- 实际打开nanoGPT源码，确认c_proj输出权重缩放、共享和get_num_params减position表；这里原文922“待验证”的窄范围可解除，不意味着整个外部库所有版本都已测试。
- 743行“作弊只有一种”若按最广义泄漏分类过强（训练/验证重复、预处理信息泄漏等不必直接看到下一token），但本节在讨论训练任务内标签可见性，两种故障只是局部机制；建议写“本节重点看答案泄漏的两种常见方式”。不把同书ch05另有泄漏类别自动判定为互斥定义。
- 360/379行“每行都有梯度”指输出路径一般密集，不保证每次batch非零：隐藏向量0、贡献相消或浮点下溢可使某行0。这是常见教学泛称，本文公式g_i h清楚，无需将边界病例当核心推导错误。
- 612行u的尺度“不随残差流变化”需理解为Pre-Norm削弱整体正数缩放影响；输入方向、gamma、上游qkv变化仍影响u。604/662行已限定理论近似，不另指控未承认的严格独立假设。
- 700行批内冲突正确限定相同可见历史；若预训练流跨公式带别的前缀，条件历史未必真正相同。后章实际packing要以它自己的可见输入判断冲突。
- 786/841行移位断言作用于未mask原始targets；后章若已经把提示/pad目标写负100，应仅比较未被遮罩且有邻接关系的位置，不能盲用整矩阵torch.equal。
- 本次只确认官方config/算法逻辑与小实例，未独立复现表格训练loss、梯度范数、RMS逐层统计或模型故障后整段输出。

## 跨章关联表

| 目标章 | 当前章依赖/输出 | 待宏观复核命题 |
|---|---|---|
| 03、04、07 | CE/token平均、移位、初始lnV | 初始化诊断仅限本书小随机输出；loss-mask全False需防全忽略batchNaN |
| 07–10、12、13 | 验证排BOS且EOS计loss，训练忽略BOS目标但不排BOS分母 | 比较基线保持同一评测范围；训练流packing可见跨公式历史须注明 |
| 09、10、15 | RoPEbuffer、pos0历史长度、残差缩放、finalnorm | cache非方阵因果mask/长度边界；位置重置与整段参考一致；缩放不保证任何梯度都稳 |
| 13、16 | non_embedding减token表、Vd共享输出仍有计算 | scaling参数口径与6ND近似一致，不因参数只算一次而漏输出投影FLOPs |
| 14 | 生成调用方负责屏蔽BOS | 不让BOS/不希望特殊token重新采样；采样约定与验证分布区分 |
| 17、19 | HF内部shift，nanoGPT外部shift；真实模型结构差别 | 避免双移位；masked labels不能直接整矩阵移位断言；外部配置版本固定 |
| 31、32 | inputs_embeds视觉前缀，Qwen混合注意力且head_dim256 | 图像位置/前缀标签mask、语言block不改的具体边界；真实GQA维度和参数另算 |

## 实际打开的来源

1. [Qwen3.5-0.8B官方config](https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/config.json)：实际读取JSON，核对868行文本配置、门、层型、共享及partialRoPE；没加载模型权重。
2. [nanoGPT作者model.py](https://raw.githubusercontent.com/karpathy/nanoGPT/master/model.py)：实际打开并检索c_proj.weight/get_num_params，核对残差输出std、共享与非嵌入计数的不同约定。
3. [Press与Wolf原论文PDF](https://aclanthology.org/E17-2025.pdf)：实际打开，核对权重共享与输入/输出更新路径动机；不把文献任务收益作为当前LaTeX必然收益。
4. [GPT-2原技术报告](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)：实际打开，核查2.3节Pre-Norm、final norm与残差层权重缩放描述，和本章把N取2L的工程约定区分。
5. [Karpathy作者文章](https://karpathy.github.io/2019/04/25/recipe/)：实际打开并查verify loss @ init等训练健全性建议，支持检查流程的出处；本文正文也已声明这些检查不充分。
