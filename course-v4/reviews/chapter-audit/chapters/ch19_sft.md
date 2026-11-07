# 第19章逐章审核

## 阅读记录

源文件：`course-v4/book/part3_post_training/ch19_sft/chapter.md`；SHA256：`91988c654c1941ab8851ec3db99eb65d4eb160db7b12cdaaf299fd07b1c1a83d`。完整通读1–240、241–480、481–720、721–963，共963行。读取本地data/sft.py与train/sft.py核对机制；未重跑全参数GPU训练和历史故障训练。

## 主线、公式与协议

本章将SFT目标限定为答案条件似然，loss为所有受监督答案token（含EOS）的NLL总和除总数。prompt通过隐藏状态参与注意力并接收间接梯度，但不作为预测目标。labels和input_ids同位置，loss唯一一次移位：logits[:,:-1]对labels[:,1:]。因此labels[t]=-100屏蔽的是logits[t-1]，prompt最后一个位置要预测首个答案。

模板关闭思考，空think块属于prompt；联合编码prompt+target并检查prompt token前缀，拒绝BPE跨界不稳定样本，追加EOS后判断长度。右padding按位置屏蔽，不能在pad=EOS时按ID屏蔽。训练超长整条拒收，生成不据训练预算过滤。microbatch按sum除统一有效token数，避免平均的平均；packing须关注跨样本注意力与位置口径。

全参数Qwen3实验参数保持fp32、bf16 autocast，基线同精度重跑；val128选步、val500报告有重叠偏差且明确声明。raw/抽取后两种评分并列，抽取器破坏句末标点不能被混成模型质量下降。故障训练用正确labels的统一NLL作比较，不比较不同监督目标的训练loss。漏EOS只去掉正向的终止目标，预训练模型可能保留停止能力。

## 已确认问题

未确认新的P1/P2/P3问题。因果移位、EOS计数、token加权平均和主实验的基线重跑说明自洽。将常见错误、教学简化与后文已给的限制结合阅读后，没有把它们重复报成新问题。

## 待验证线索与已声明限制

- 19.3.2“对应logits梯度为0”单独阅读容易漏因果位移，但19.3.3、19.4.3及小结均明确前一个logits，按全章不能判为未纠正概念错误。
- 完整assistant对话渲染与人工prompt+answer渲染的细节依赖快照，训练/推理本章明确复用同一个render_chat，不能以当前远端模板不同否定历史观察。
- 约2轮train loss低于val不足以证明逐条记忆；“接近背下来”宜理解为泛化差距的口语解释，不将其当成已做membership测量。
- no_eos仍在非终止位置对EOS有softmax竞争产生的负梯度；本章“未提供别停监督”用于解释原有习惯保留，严格表述可补充“未在答案末端监督继续生成某个非EOS目标”。主要结论已限定80步单次观察，不把此措辞提升为确认训练失效。
- 本章没有评test，val128是val500子集、指标未覆盖所有语法支持域等均已明确，不能当隐瞒测试泄漏。

## 跨章关系

|目标章节|本章输出的具体定义|待复核命题|
|---|---|---|
|ch03、ch07|ignore_index及唯一一次因果shift|基础交叉熵位置记法是否和因果LM一致|
|ch16、ch21|一次更新所有有效token总分母|Trainer累积、DDP归约是否维持同一目标|
|ch18、ch22|postcorrect-v1、原串clean/noisy、raw/抽取双口径|跨章指标不可混合抽取前后或精度/子集|
|ch20|596049920参数、fp32权重bf16计算|LoRA比较需重统计到相同val200、资源记账区分冻结状态|
|ch23、ch25|SFT checkpoint及只监督答案协议|偏好/策略logprob同样一次shift、EOS含入且prompt排除|

## 核查来源与证据边界

读取本地`miniocr/data/sft.py:26–83`及`miniocr/train/sft.py`相关loss/累积实现，核对mask、拒收和分母。已打开[HF chat templating文档](https://huggingface.co/docs/transformers/chat_templating)，核对消息仍是token续写、控制token/模板匹配的重要性。已打开并定位[官方data_collator.py](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/data/data_collator.py)中CLM按pad ID屏蔽labels的分支，支持pad=EOS的警告。Qwen模板和特殊token沿用本轮ch18已经实际打开核对的官方tokenizer_config。远端main只作当前机制核查，不等同复现文中transformers5.19.0/TRL1.15.0全部历史行为；未训练596M模型。

## 结论

本章核心监督信号逻辑成立，保留若干表述精度与版本边界供全书统稿。没有新增确认问题。
