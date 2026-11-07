# 第 15 章逐章审核

## 阅读覆盖

- 原文：`course-v4/book/part2_lm/ch15_kvcache/chapter.md`
- SHA256：`570503d8a62fe8d9e203207306952273be06b80fe70f2b64f404a6f8bfc6816d`
- 完整通读：1–250、251–500、501–740行，覆盖全章740行。
- 辅助核查：缓存相关代码片段、形状/算式、官方Qwen配置与HF缓存文档。未重跑CPU历史计时和GPU服务测试。

## 主要内容和关键约定

本章将第14章整段重算生成优化成prefill加逐token decode。固定参数、相同前文且因果可见时旧位置在数学模型中不变，故每层只缓存K/V，不需缓存历史Q。K应位于QK norm及RoPE之后、repeat_kv之前；缓存形状(B,kv_heads,T,head_dim)。正确性以同前缀逐步logits容差对拍为主，贪心文本一致只是辅助，不能保证也不必强求极近argmax时逐字相同。

分段预填充新块Tnew、历史Tpast时mask为(Tnew,Tpast+Tnew)，条件j<=Tpast+i，pos0=Tpast。缓存字节数2*层数*kv头*头维*T*B*字节，显式head_dim不能总用hidden_size/n_heads推。Qwen3-0.6B的28层、16Q头、8KV头、128头维给bf16每token112KiB；Qwen3.5-0.8B只有6层full attention，随T增长部分每token12KiB，18层线性注意力固定状态另算。

粗略roofline仅算矩阵乘和指定读写：预填充线性层AI约B*Tnew，decode约B，注意力AI约Q头/KV头。批量摊薄权重，不摊薄每条不同历史的KV。CPU观测仅支持本机小模型趋势，不能推广GPU。torch.cat逐步复制、未拆分开销、等长批限制均已明确。PagedAttention与前缀共享作为概括，为附录D和VLM视觉前缀预算铺垫。

## 已确认问题

本轮完整通读与核查未确认新的P1或P2章内错误。主要mask、位置、GQA字节计数与算术强度推导自洽；不为填满问题栏把已声明的简化列成缺陷。

## 待验证线索和已声明局限

- 15.1.2把数学上的前文不变与某次CPU“差恰好为0”连在一起；15.3.4已明确矩阵形状和计算顺序可能产生浮点误差。跨章引用时保留“数学不变、实现容差一致”的区分，不要推广成任何设备逐位相同。
- “因果模型旧位置不变”还要求推理时参数固定、无随机dropout等条件。章节讨论推理场景可理解其默认条件，但建议首次给出条件，避免移用于训练前向。
- `torch.cat`缓存每步复制是已承认的实现代价，不能把理论AI表当真实实现的全部内存流量。
- Qwen3.5线性注意力固定状态明确未计入，12KiB只代表随T增长的KV，不代表整个模型总推理状态。
- 模型/框架当前官方配置可支持数字核对；历史快照可复现性仍需项目保存snapshot，不以当前main替代历史实测证据。

## 跨章关联

| 章节 | 具体关联 | 宏观复核重点 |
|---|---|---|
| 第8、10、11章 | 因果mask、RoPE、GPT接口 | pos0的零基位置、RoPE后的K、QK norm顺序保持一致 |
| 第13、14章 | 生成与缓存对照 | BOS/EOS、可预测token屏蔽、生成上限和精度是否一致 |
| 第16章与附录E | FLOPs与roofline | 理论访存假设不能替代实测瓶颈或实际cat读写 |
| 第17章 | Qwen3骨架 | head_dim128但hidden_size1024，不能强制Q总宽=hidden_size |
| 第31、32章 | 视觉token前缀、混合架构 | 视觉位置计入T；Qwen3.5全注意力KV与线性状态分别核算 |
| 附录D | PagedAttention、TTFT、TPOT、并发 | 页块与前缀共享改变分配/读写，不能直接使用本章cat实现速度 |

## 来源和验证

- [Qwen官方Qwen3-0.6B配置](https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json)，已完整打开30行，核对28层、16Q头、8KV头、head_dim128、hidden_size1024，支持112KiB计算。
- [Qwen官方Qwen3.5-0.8B配置](https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/config.json)，已完整打开96行，核对24层中6个full_attention，8Q头、2KV头、head_dim256，支持12KiB的随T增长部分。
- [HF官方缓存策略文档](https://huggingface.co/docs/transformers/kv_cache)，已打开DynamicCache/StaticCache相关段，核对动态增长和固定预分配的区别，不据其当前API替代教材历史版本所有细节。
- [Ainslie等GQA作者论文页](https://arxiv.org/abs/2305.13245)，已打开摘要，质量与效率的折中方向与本章一致；未复现论文质量结论。
- 形状mask条件、各字节单位与总和、线性与注意力AI比值均按原公式独立检查，未发现算式错误。

## 结论

完整通读完成，已核查的核心定义与公式通过。宏观审查重点是数学等价和数值容差的表述、显式head_dim、以及VLM混合架构缓存口径。
