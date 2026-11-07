# ch08_attention 修正记录

- CH08-01｜成立｜8.7 节“怎样处理”第二条末尾补：本例分数为 fp32；fp16 下 masked_fill(-1e9) 报溢出，转 fp16 变 −∞ 仍 NaN；低精度用 torch.finfo(dtype).min 或先升 fp32；首选仍是每行至少一个可见键。CPU 复现确认（PyTorch .venv）。
- AR-03（置换对象）｜成立（本章仅补一行）｜8.9.2 “置换等变”后补段：P 置换的是整行 token，token 内数值不变；不能推广为任何空间关系不可辨，token 内部内容改变（如第 29 章 patch 内像素换位）不属此定理。
- 待核线索（SDPA 全遮罩输出 0、causal 方阵与 Tk≠Tq 等）｜未作为问题｜报告本身未认定为错误，不改。
- Codex 复核（gpt-6.1-sol，仅本章 diff）：未发现事实错误。
