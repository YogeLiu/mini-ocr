# 第24章逐章审核

## 阅读记录

源文件：`course-v4/book/part3_post_training/ch24_rl_basics/chapter.md`；SHA256：`c666fc6bd2fc1043dfc917799cc14f8fad7be393cb9779f79bce174ed043b967`。完整通读1–215、216–430、431–645、646–857，共857行。CPU复算KL和PPO梯度；未复现历史多种子GPU训练/序列实验。

## 主线与关键数学

语言模型状态是prompt+历史token，动作是next token，回合至EOS或预算，终局奖励不依赖参数。REINFORCE来自score trick：grad J=E[R grad log pi]，序列logprob是response token之和。softmax score=e_a−pi，bandit真梯度pi*(r−J)。loss数值不等同奖励或性能。

基线须独立当前动作/奖励且停止反传，合法状态基线不改期望；最优常数基线按score平方范数加权。批均值含自身使期望缩为(B−1)/B，leave-one-out无偏，detach不能修统计依赖。GAE仅直觉，lambda1/gamma1在终止情形回到R−V。

KL方向固定当前p到参考q，样本来自p，log_ratio=log q−log p。k1=−logrho，k2=.5logrho²，k3=expm1(logrho)−logrho；正支持下k1/k3值无偏，k2仅局部近似，非负不等于梯度正确。24.5.7独立推导直接反传k3在on-policy点的期望对应反向参数次序KL(q||p)，需重要性权重修正。PPO min只在优势正/比率高及优势负/比率低两格饱和，不是全KL硬约束。

toy GPT固定8token、首尾$奖励诱发全$输出，未证明KL能阻止投机。项目OCR面板不适用于无标准答案符号任务；test只一次。TRL对应明确版本、token/sequence ratio与grpo/dapo不同聚合路径。

## 已确认问题

### CH24-01（P2）：由奖励不可导推出“只能强化学习”

- 源位置：728行，“它不可导，只能走强化学习这条路”；5行也以不是现成偏好对为理由说这类任务“绕不过去”。
- 理由：分数不可导只阻止直接把该分数当普通路径损失反传，不排除先采候选再依分数生成偏好对，用DPO训练；也可选高分候选作拒绝采样SFT。标量分数本来就可以给两个候选排序，恰是上一章23.7的流程。若“直接最大化原始期望奖励”是限定，应明确限定，不能写所有利用渲染评分训练的唯一方法。
- 反例：同图采样A、B，渲染相似度0.9与0.4，组成chosen=A/rejected=B，按ch23的可微DPO logprob损失更新；整个过程无需对渲染器求导，也无需在线RL更新。
- 建议：改为“若要直接优化当前策略采样的期望分数，可用策略梯度；也可以把评分转成偏好或筛选监督数据，走上一章方法”。

## 待验证与已有局限

- 固定状态bandit的基线效果不能推成所有任务最终分数更好，本章已用32种子实验主动排除这一误读。
- 直接k3梯度方向与值估计是不同命题，本章明确推了on-policy点，不能指控未处理采样分布梯度。
- 价值模型“一样大”及四模型同时驻留是常见实现的资源图景，不是PPO数学必要条件（可共享骨干或用不同大小critic）；统稿时宜保留典型实现限定，当前未据此指控实验资源账错误。
- GAE方差与偏差的lambda直觉不是所有环境下严格单调定理；本章标直觉且不做该实验。
- PPO logp_old未detach导致零梯度，正文明确要求来自同一个计算图，不把“仅requires_grad=True但独立旧参数”误判成必为零。

## 跨章契约

|目标章节|本章具体输出/依赖|待核查|
|---|---|---|
|ch19、ch23|response mask和唯一shift、序列logprob求和|EOS是采样动作须监督/计入，prompt不属于策略动作|
|ch23|评分可转换成偏好，beta KL与参考支持|本章“只能RL”与DPO自动配对路线冲突应统一|
|ch25|组均值含自身偏差、标准差缩放、PPO及k3修正|GRPO不能直接称原始无偏REINFORCE；token聚合与序列比率分开|
|ch34|手写VLM策略梯度、KL与无效组|修正是否包含重要性权重梯度、mask和有效token分母|
|ch22、ch36|玩具奖励与任务能力|满格式奖励不是EM/CER收益，不能进入R2 OCR成绩|

## 核查来源与验证

读取`miniocr/train/policy_gradient.py`，CPU float64枚举p=[.1,.2,.3,.4]、q反转：真KL=0.4564348191，k1/k3期望同值，k2=0.5215535024；CPU autograd核对ratio=[.5,1,1.5]时优势+1梯度[1,1,0]、−1梯度[0,−1,−1]。支持本章关键数值。

已打开[PPO原论文摘要及PDF](https://arxiv.org/pdf/1707.06347)，核对裁剪目标与重复minibatch更新背景；HTML转换页只有PDF链接，不把它当正文。已打开并定位[TRL v1.15.0 grpo_trainer.py](https://raw.githubusercontent.com/huggingface/trl/v1.15.0/trl/trainer/grpo_trainer.py)的ratio、k3、use_bias_correction_kl乘coef_1、min目标及不同聚合分支。Schulman KL博客两次打开均超时，未声称已读；k1/k3期望和反传方向由独立代数验证。DPO替代反例依赖本轮ch23已实际打开核对的原论文Eq7与自动配对方法。

## 本章结论

确认1处P2关于训练路线的排他性推断。REINFORCE、基线、KL值/梯度区别和PPO裁剪的主要数学链条成立。
