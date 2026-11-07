# 第17章逐章审核

## 阅读记录

源文件：`course-v4/book/part3_post_training/ch17_gpt2/chapter.md`；SHA256：`39c73bc8258ff0ef7edd0a62b37badaaf47033b59b4b60a4890652d27199d0c7`。完整通读1–230、231–460、461–673，共673行。未运行历史GPT-2权重tour，未复现其logits数值表。

## 章节主线与关键约定

由GPT-2结构树进入真实权重加载。参数数目按独立Parameter去重：N=Vd+Pd+L(12d²+13d)+2d=124439808；共享输出矩阵不重复计数。state_dict按名称列举，可能重复共享权重，缓冲区也不是可训练参数。

架构为学习式位置嵌入、Pre-LN、带偏置MHA、tanh近似GELU、最终LayerNorm。Conv1D存[in,out]，转入Linear时转置；合并qkv沿输出维切三片后分别转置。strict加载检查键/形状，不检查语义或数值位置；方阵漏转置、qkv互换可静默通过。

对拍固定token、eval、dtype、设备和注意力实现，检查logits和各块边界；L+1个hidden边界最后一个已经经过ln_f。生成一致不足以证明实现正确，近似并列logits又可能因浮点误差改变argmax。容差只对指定CPU/fp32设置标定。最后比较Qwen3配置：GQA、显式head_dim、QK-Norm、RMSNorm、SwiGLU、RoPE和共享嵌入。

## 章内已确认问题

未确认新的P1/P2/P3问题。公式、qkv方向、参数去重和对拍逻辑相互一致。没有为了满足问题数量而把明确的模型版本/设备边界列为错误。

## 待验证与已声明局限

- GPT-2报告的117M与代码124.44M差异，本章明确列为待验证，没有给无来源历史解释。
- 本地safetensors文件大小、mask缓冲区dtype、160个张量是项目快照观察，不能推广为所有GPT-2发布文件的共同格式；本章已有本地来源限定，未重新下载大型权重核验。
- Qwen表格只依赖配置和键名，本章已说明不能由它证明运算顺序/运行时mask。没有把它当作完成Qwen前向对拍。
- 1024“硬上限”适用于本章未扩展位置表的原始结构；不能据此推出学习式位置嵌入永远无法扩表或重新训练。本章实际语境为固定预训练模型，尚不构成确定错误。

## 跨章关联

|目标章节|具体契约|待跨章复核|
|---|---|---|
|ch09、ch11|Pre-Norm、残差缩放、共享嵌入及参数公式|默认GPT配置和SwiGLU中间宽是否统一|
|ch06、ch18|字节BPE、预分词、token长度影响上下文|小BPE/GPT-2/Qwen词表口径与特殊token是否分清|
|ch15、ch20|Qwen3 head_dim=128，q16/kv8，hidden1024|缓存与LoRA参数计算不能误用hidden/n_heads|
|ch19、ch21|Qwen真实词表及模型参数|151936模型输出行数与实际tokenizer词表应区分|
|ch32|对拍方法的迁移边界|processor对拍与内部模型数值等价验证不是同一证据|

## 来源与验证方式

已打开并阅读[OpenAI原始GPT-2 model.py](https://raw.githubusercontent.com/openai/gpt-2/master/src/model.py)相关实现：norm/gelu、conv1d、qkv切分、Pre-LN块及共享wte输出，支持本章结构解释。已打开完整[官方社区GPT-2 config](https://huggingface.co/openai-community/gpt2/raw/main/config.json)，核对宽度、层数、位置、词表、dropout、特殊token和激活字段。已打开[GPT-2技术报告](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)并定位输入表示及2.3 Model，核对字节BPE类别约束、Pre-LN、最终norm与残差初始化；不声称完整复读该24页报告或重跑预训练。

## 本章结论

本章主要链条成立；保留快照文件细节和数值tour的复现边界。跨章关联仅登记具体契约，暂不下全书结论。
