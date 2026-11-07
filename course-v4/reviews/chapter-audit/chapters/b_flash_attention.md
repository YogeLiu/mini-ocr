# 附录B逐章审查

## 阅读记录

源文件`course-v4/book/appendix/b_flash_attention/chapter.md`；SHA256 `438156e9bcedb40c252875f432dedbc7f9b16be393ca6187a4e92894da9ec657`。完整通读1–229行，单次输出未截断。

## 主线与关键公式

CPU float64教学实现，非GPU FlashAttention性能复现。online softmax维护m最大值、l归一化和、o未归一化加权输出；新块先按新最大值用exp(m_old−m_new)缩放旧l/o，最后o/l。Q外循环、KV内循环的具体计数模型：标准IO=4ND+4N²、分块IO=2ND+2N²D/Br，Br整除N；SRAM容量2BrD+2BcD+2BrBc+2Br。FLOP主项皆4N²D，重算反向避免保存N²概率。直接autograd分块保存更多，checkpoint模拟重算。safe_m避免全mask块的−inf相减。

## 已确认问题

未确认P1/P2。online递推、IO计数和数值例子相容。该章明确只统计所写循环、只算前向、因果mask不跳块，不能以真实FA因果内核减少乘法指控教学计数错误。

## 待验证线索及已声明边界

- 183行最终o/l未定义整行所有key均被mask的情况；safe_m仅处理块，不处理整行l=0。当前验证是因果mask，每行至少自己可见，所以不是已展示实验失败。若向读者开放任意mask，应约定每行至少一个合法key或对l=0返回约定输出，再与SDPA对拍。
- 78行渐近范围应显式写原论文SRAM假设，避免任意极小M也宣称同阶最优；正文已有“定理前提以原文为准”。
- 216行“反向要靠重算”限定“在不保存完整P、保持精确稠密注意力的此路线中”；并非任何注意力反向的普遍要求。
- 标准访问式在S/P放不下时使用，147/151行已展示放得下则无IO优势，不存在式子无条件应用冲突。
- 没GPU计时、Python循环速度无参考价值、真实内核手写反向待版本核对均已说明。

## 跨章关联

|章|具体接口/结果|复核命题|
|---|---|---|
|3/8|稳定softmax与稠密注意力|分块改求和次序会有浮点差，数学精确不等于逐位一致|
|15/16|访存受限与重算|IO下降不是FLOP下降；checkpoint存储和反向临时峰值不同|
|29/31/33|视觉序列长度与SDPA|真实内核依设备选后端；CPU教学计数不能当VLM GPU提速预测|
|附录E|性能证据|须实测后端、shape、GPU同步和峰值，不用访问计数替代延迟|

## 实际核查来源与范围

打开FlashAttention作者论文摘要 https://arxiv.org/abs/2205.14135 ，核对IO-aware exact attention定位；打开PyTorch官方SDPA文档 https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html ，核对可选实现与设备选择入口。递推及表内算术逐项核对；未运行GPU内核、未复现论文性能。整体无新大规模实验。
