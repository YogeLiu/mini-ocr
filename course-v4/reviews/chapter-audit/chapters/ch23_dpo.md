# 第23章逐章审核

## 阅读记录

源文件：`course-v4/book/part3_post_training/ch23_dpo/chapter.md`；SHA256：`1b6e883b6e89c90722d9dcbbad4b72a438ca755f01b86d2f08408dbe42c2db88`。完整通读1–210、211–420、421–630、631–823，共823行。读取偏好训练tour与TRL同版本官方源码；未重跑GPU训练、采样或test。

## 主线与数学契约

BT以sigmoid(rw−rl)建模偏好，奖励加同prompt常数不可辨。固定参考与beta>0，KL正则目标最优分布pi*=pi_ref exp(r/beta)/Z，假设Z有限；取log反解奖励后同prompt的log Z抵消，得到−log sigmoid(beta[delta_w−delta_l])。隐式奖励是beta log(pi/pi_ref)，reward accuracy为margin>0，初始策略等于参考时loss=log2。回答logprob含EOS、排除prompt/pad、一次shift、按token求和；每对loss取平均。

梯度只优化chosen/rejected的相对差，chosen绝对概率也可能下降；章内C例已明示。CPU替换Triton后的tiny逐对前向对拍不覆盖原CUDA核/梯度更新/真实模板，本章完整披露这一边界。数据从SFT采四候选按规范化字符编辑率排序，min_gap筛选，仅387对，42%chosen仍错；离线覆盖不足与代理指标偏差需单独分析。

beta扫描小值退化、大值可能微增；配对区间仍受val选择、看结果扩扫描影响，无DPO test结论。新LoRA关闭后作为参考，冻结及数据顺序有源码和日志间接证据，不能把训练reward accuracy当泛化成绩。

## 已确认问题

### CH23-01（P3）：自动评分不会自动排除长度进入标签

- 源位置：549行，“用自动指标构造偏好对，长度不会偷偷混进标签”；547行由边际长度比例与均值断言“基本没有长度偏好”。
- 理由：自动指标可以含长度相关偏差；本章编辑距离惩罚插入/删除，且并列时明确选择短候选，所以不能从“没有人类打分”推出标签不依赖长度。chosen/rejected总体均长接近也只能排除明显边际偏移，不能排除不同题型上的偏差互相抵消。
- 反例：一半题始终偏好长候选、另一半始终偏好短候选，chosen/rejected全局均长和长短比例仍可完全相同。平均输出长度不漂移同样不能证明标签机制无长度偏差。
- 建议：收窄为“本批数据未见明显的整体偏长倾向”，保留自动编辑指标和短者优先规则可能引入长度相关选择的限制。

## 待验证线索与已明确边界

- 477行“2048之后所以没训练过”在原则上只能保证行号不重叠；latex-v1保留同规范键组内重复，需要另查提示/目标内容，不能由偏移推导内容未见。已在本地build_split(train,4096)复核：后2048条与前2048条原source交集命中0条，规范目标键命中3条。这支持“新source”，不能称目标内容全未见；尚未核对前缀SFT长度拒收后的接受集合，故不宣称实际训练污染。
- 293行“策略和奖励一一对应”严格应为奖励加prompt常数的等价类与策略对应；前面已证明常数不可辨，适合统稿补准确措辞，不把全章推导当错误。
- beta对KL的理论意义针对可实现的最优策略；有限离线确定偏好未给显式KL硬界，IPO小节已有margin发散说明。不能从beta扫描直接称测得了全分布KL（本章实际只报训练对margin和输出变化）。
- 参考同值日志只是间接一致性证据，CPU替换核没有覆盖GPU更新、末批3对和累积等价未逐数验证，均已声明，不重复报错。
- TRL max_length384/keep_start可能截去EOS或chosen/rejected差异，与ch19整条拒收原则应跨章核对；当前未统计387对实际截断数，不宣称已发生。

## 跨章关系

|目标章节|本章输入/输出契约|需宏观核对|
|---|---|---|
|ch19、ch21|response logprob求和、EOS、shift和mask|DPO keep_start截断与SFT拒收协议是否允许且披露|
|ch21、ch22|step400 SFT、val500基线|本章先merge再bf16保存/FP32加载，基线却沿用ch22未合并fp32基座adapter输出；需补DPO零步合并快照基线，排除初始化精度变化混入2.8点收益|
|ch12、ch18|按规范键分组但组内可重复|train偏移2048不自动保证新目标/新source内容|
|ch24、ch25|beta KL、参考策略、序列logprob|DPO beta与PPO/GRPO约束在有限优化时不能机械等同|
|ch22、ch36|配对区间和val扩扫描|未做test，不能把0.706当冻结后最终收益|

## 核查来源与证据范围

已打开并阅读[DPO原论文v3](https://arxiv.org/html/2305.18290v3)§3–5相关段落，核对BT、KL闭式解、Eq7、梯度及奖励等价类；独立代数核对手算例子。已打开并定位[TRL v1.15.0 dpo_trainer.py](https://raw.githubusercontent.com/huggingface/trl/v1.15.0/trl/trainer/dpo_trainer.py)的sigmoid损失、回答求和、no_grad及adapter参考分支、collator keep_start截断。读取本地ch23 tour采样前merge/to(bfloat16)/save与后续数据偏移流程。未声称读完全部DPO文献或复现历史GPU输出。

## 结论

确认1处P3长度偏好推断过强；推导及实际对拍边界基本成立。最值得主agent宏观复核的是SFT基线与DPO初始化合并/精度路径，以及截断协议。
