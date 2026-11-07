# 第21章逐章审核

## 阅读记录

源文件：`course-v4/book/part3_post_training/ch21_trl_sft/chapter.md`；SHA256：`166a59c63379e4ad09655170beb07807df62bc4a1dab85af28312ef9baad538a`。完整通读1–225、226–450、451–675、676–894，共894行。读取本地桥接代码、安装包相关源码及已有val逐条记录；不访问test错例，不重新训练GPU模型。

## 主线与关键协议

七项检查分别验证真实dataloader的mask/EOS、completion-only效果、可训练参数、loss/梯度/一次手动更新对拍、保存重载、同val比较、固定train探针与val的分叉。数据预编码labels且skip_prepare_dataset，completion_only_loss开关不负责此路径mask；train和generate模板仍来自同一个来源。

一次更新分母是所有microbatch的labels[...,1:]非-100总数，num_items_in_batch传入compute_loss；CUDA用融合LM head避免完整logits，CPU/MPS子类用原生loss，二者对拍证据不可互换。rank32 q/v有4587520参数；总参数包含adapter，基座数另减去。对拍用fp32、专门eps=1e-6，正式训练bf16 autocast/eps=1e-8，二者明示不同。

主实验train2023条、800步、每200步保存，val200选400步；每轮末microbatch不足2，实际样本次数3199/6397。选步与报告val相同有偏差；test预注册选中配置只评一次。adapter重载同机逐条复现，仅证明保存评测路径；训练与重载代码哈希有变，已明确未验证当前代码从零再训同adapter。

## 已确认问题

### CH21-01（P3）：对比表把可由已有计数确定的fix_rate列为“未记录”

- 源位置：762行，自写LoRA r16 q,k,v,o行“129/200…6/59…未记录”。
- 理由：raw-clean59条，改坏6条，则clean正确53条；总EM正确129条，因此noisy正确129−53=76条，fix_rate=76/141≈0.539。无需模型重跑或逐条test资料即可恢复。四指标的分母、分组契约已在本章明确定义，不能把此量当作无法获得。
- 证据：同表给出的EM、改坏与clean/noisy分母；代数恒等式N_correct=(N_clean−N_broken)+N_noisy_fixed。CPU算式结果0.53900709。
- 建议：补76/141，注明由汇总计数恢复；避免留下不必要的必报面板缺口。

## 待验证与明确局限

- `loss_step_parity`两侧都新建相同AdamW并手动backward/clip/step；它验证compute_loss及相同优化器作用下更新的接近程度，未直接对拍Trainer正式优化器/调度器/完整train循环。452行把“优化器也是框架行为”作为理由，覆盖范围宜讲清；已有实际Trainer一步参数改变测试也不等同全循环数值等价。此项作为验证覆盖线索，而非断言已发现训练错误。
- “改了代码没有影响评测路径”只适用于所检查200条和设置，本章已紧接着限制不保证从头重训相同，未重复指控历史可复现性。
- pairing的7条“copy对SFT错=原串无错改坏”并非所有任务恒等式（copy还可规范化修extra_space），但实际读取reloaded-val-predictions.json后重算：copy-only确为7条且7条均raw-clean，故本章这个具体观察成立。
- val NLL197条、生成200条、共享GPU时间不能排名、参数不同/数据不同不能比较框架优劣、未做多卡/packing均已说明。

## 跨章关联

|目标章节|具体契约|需复核|
|---|---|---|
|ch19|预编码labels、一次shift、token总分母|正式Trainer累积与自写大batch的目标一致范围|
|ch20|同一val200及LoRA形状/状态|不同rank/targets/训练数据混杂已列，不能归因框架|
|ch18、ch22|raw/extracted、原串clean/noisy|copy-only并不一般等于broken子集，具体数据须实际核对|
|ch23–25|adapter保存、版本、fused loss|偏好/策略训练是否沿用相同结束符与prompt排除|
|附录A、E|单机loss分母与性能|多卡world-size缩放和跨卡总token数尚不能由本章单卡对拍证明|

## 事实核查来源

已打开并定位[TRL v1.15.0官方sft_trainer.py](https://raw.githubusercontent.com/huggingface/trl/v1.15.0/trl/trainer/sft_trainer.py)：collator已有labels原样右补、skip_prepare保护、add_fused_lm_head及compute_loss的统一分母，支持主要叙述。文档页面v1.15.0工具无法访问，改读同版本官方源码，未只凭搜索摘要。读取本地transformers/trainer.py `_get_num_items_in_batch`核对labels[...,1:]计数。读取`miniocr/train/trl_sft.py`确认对拍手动更新与恢复参数。仅重统计已有val记录，没有重跑Triton/GPU训练或test评测。

## 结论

确认1处P3可恢复指标缺失；核心mask、版本路径和同机重载证据边界基本清楚，完整Trainer优化器等价属于尚未覆盖的验证范围。
