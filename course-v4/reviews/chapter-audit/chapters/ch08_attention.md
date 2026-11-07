# ch08 注意力：逐章审查

## 阅读与版本

- 源文件：`course-v4/book/part2_lm/ch08_attention/chapter.md`
- SHA256：`d29d0d5558afaece2beaf1b9ee04e1bb09fd190d5520a6fde8f0d81e307aba3a`
- 完整逐行阅读：1–240、241–480、481–720、721–960、961–1078。
- 验证：矩阵/概率/形状推导，PyTorch2.14.1 CPU边界案例，实际打开原论文与官方API。未重跑历史查表/LaTeX训练、全部注意力干预统计或GPU内核。

## 主线与关键约定

本章把内容寻址软查找推到Q/K/V投影、缩放点积、因果/补齐mask和多头。分数矩阵行是查询、列是键，softmax最后一维，输出为AV；q/k维度相同而v可不同。独立、零均值、单位方差假设下点积方差d，除sqrt(d)使标准差1；正文已说明训练时尺度可变、Q/K norm也有可学习gain。

本书bool mask的True为允许，补齐遮键；单头(B,1,Tk)，多头(B,1,1,Tk)。因果下三角含自己。手写softmax整行负无穷得到NaN；多头先按d切h份再交换维度，合并用reshape。固定d、标准MHA、含bias时参数4(d²+d)，独立于h。无位置且无mask自注意力置换等变，因果mask会打破对称但不等于完整位置编码。

标准显式注意力核心4T²d FLOPs、投影8Td²，显式权重BhT²元素；FlashAttention不必物化整个矩阵。LaTeX AttentionLM是单头裸注意力加learned position，无残差/FFN/norm，最长64历史，按逐目标滑窗、每窗位置重置。其EOS/字节/BOS屏蔽协议与ch07相同。一次短训练比MLP差，正文已承认目标数、参数、结构、种子不齐，不能用此判定注意力一般较差。干预非零说明信息通路，不证明学会括号配对；注意力权重不等于因果贡献。

## 已确认问题

### CH08-01 · P2 · 用−1e9修复全遮罩行的建议漏掉fp16不可表示条件

**位置与引文：**635–636行列为常见处理方法，建议“用一个很大的负数（比如−1e9）代替−∞……不再是NaN”；417–423行手写实现采用scores.masked_fill，1015/1022行又明确讨论fp16注意力分数与65504上限。

**复现：**CPU，`x=torch.zeros(1,3,dtype=torch.float16)`，对全False mask做`x.masked_fill(~mask,-1e9)`，PyTorch2.14.1报`RuntimeError: value cannot be converted to type c10::Half without overflow`。若先将常数强转为fp16张量，它会变成−inf，同样不能修复全遮罩softmax。该方法在正文float32实验成立，但不是可直接移用到低精度的通用处理。后面的fp16节没有为这个常数补条件。

**建议：**写明本例分数为fp32；首选确保每个查询至少有一键可见，且padding query不参与loss。如果保留有限负值法，应使用分数dtype可表示值（例如finfo.min）并说明全遮行输出无意义；也可先提升scores到fp32。CPU以finfo(float16).min填全行得到均匀[1/3,1/3,1/3]，支持这一局部修正，但不把它声称为所有混合精度/非有限输入的完整安全方案。

## 已有局限与待核线索

- 本章“整行mask得到NaN”准确描述手写实现，不宜无条件推广所有框架内核。本次CPU调用2.14.1 `F.scaled_dot_product_attention`，全False二维mask的输出是全0，而手写softmax为NaN。1032行对应关系需理解为正常有定义行的语义对应；建议加框架边界提示，未认定正文承诺异常输入逐项相同。
- 1032行“不要同时给attn_mask和is_causal=True”与官方文档的接口约束一致；具体后端可能接受部分组合，也不应把可移植建议反判为错误。
- 744–749行API可传独立kv，但causal=True构造的是Tq×Tq方阵，不能直接支持任意不同Tk的因果交叉注意力/cache。本章交叉注意力用途应选causal=False；等待9/15章确认调用条件。未把没有承诺的泛化场景判实现错误。
- 106行“无窗口限制”在机制层面指参数共享可拓展，906、947、949行明确实际模型只有64窗口，位置表也有限，不能忽略后文限定报自相矛盾。
- 288行自匹配非负并不意味着注意力概率一定大，正文已给反例并承认这一点。955–957行区分数学正权重与浮点下溢，已排除“可见就必变”的错误保证。
- 484–491行饱和论述针对局部导数，不是任意大的分数扰动也不改变输出。第3章CE消除softmax小梯度的p-y结论用于最终分类器，不自动适用于中间注意力权重，两章不矛盾。
- 980–986行明确权重解释限制，925–932行列出不公平实验条件，968行未用低平均权重证明完全不用括号。无需重复指控。
- Qwen3使用Q/K norm需以实际模型代码为最终证据；本次只打开官方config确认其RMSNorm、head配置，不从config推导不存在的norm实现细节。

## 跨章关联表

| 目标章 | 当前章输出/依赖 | 待统一核对命题 |
|---|---|---|
| 03、04 | 中间softmax饱和与最终CE的p-y；mask、NaN反向 | finite loss忽略padding不保证上游NaN无害；fp16常数应dtype合适 |
| 07、11–13 | EOS计loss、原文字节不含特殊符、BOS排除、逐目标64滑窗 | GPT基线比较是否同计分对象，packing/训练BOS变化如何解释 |
| 09、10 | 无mask置换等变、causal mask部分位置信息、learned position | 不把无位置信息的双向证明直接用于causal模型；无位置模型能力和长度泛化需限定 |
| 15 | causal方阵假设、缓存Tk>Tq | cache mask对齐不能直接复用上左三角；只算新query不等于无顺序依赖 |
| 16、17 | fp16先缩放、fp32softmax、FlashAttention | 实际实现是否提升中间乘积；SDPA/Flash异常全mask行为不同于手写 |
| 26、31、32 | cross-attention与视觉前缀是两种连接法 | ch31宣称使用视觉前缀而非cross-attention是否全书一致；视觉token长度纳入成本 |
| 22 | 干预信息通路与因果用途区别 | 非零变化不可当学会长程结构，attention heatmap不能单独证明解释 |

## 实际核查来源

- [Attention Is All You Need原文](https://arxiv.org/html/1706.03762v7)：实际打开HTML，核查scaled dot-product、多头与位置机制；metadata另打开arXiv abstract确认2017。公式与形状主要由直接代数验证，不借摘要背书实验数字。
- [PyTorch2.14 SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)：实际打开并检索签名/mask说明，核对bool True含义、dropout_p与is_causal位置、正常点积注意力对应；全mask输出另由CPU实际调用验证。
- [Qwen3-0.6B官方config](https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json)：实际打开，作为RMSNorm/head/model配置的窄范围核对，未将它当Q/K norm源码证明。
- CH08-01由本次fp16 masked_fill CPU反例直接确认；并验证dtype有限最小值法的局部效果。没有执行GPU精度或FlashAttention后端实验。
