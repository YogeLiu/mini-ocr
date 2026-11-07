# 附录D逐章审查

## 阅读记录

源文件`course-v4/book/appendix/d_vllm/chapter.md`；SHA256 `3dc42bbcd9f150df9e7945c8fdd46ac6bb3a5c7660680ebbff5b30eb9fc18701`。完整通读1–233行，单次无截断。

## 主线与关键约定

纯模拟不导入vLLM。PagedAttention分块KV，n长占ceil(n/b)块，内部碎片<b；共享完整前缀块用引用计数。连续批处理是迭代级补位，所有请求同时到达、一步每活跃请求出一token、忽略prefill和调度时下界max(ceil(Σlength/B),max length)。有限池模拟后入请求抢占、释放重算。更大batch摊薄权重读取，但KV开销不等摊薄；步数非时间。单adapter可合并部署，多adapter共基座挂载，示例serve/enable-lora/lora-modules和OpenAI completions请求均标未实跑。部署验证模板/EOS/缓存容量、TTFT/TPOT。

## 已确认问题

未确认P1/P2。步数、碎片与权重账公式一致，静态“整批返回”的延迟假设107行已明确；不把该模拟结果当真实服务普遍延迟规律。

## 待验证线索与已声明局限

- 199行“确认基座和adapter输出不同”只能作挑选有效探针后的检查，非加载正确性的必要条件。正确挂载的轻微LoRA可能只改logits，不改若干题贪心输出；应查请求路由/加载日志并比logprob或选择适配任务探针。
- 198行temperature0不保证跨HF/vLLM逐字一致：内核、精度、批量归约可改变接近并列的logits；除模板/EOS还需允许数值差并做质量回归。正文写先查模板停止没有禁止其他原因，属补充限定。
- 35行块小“内存越不连续”是常见布局直觉而非严格单调定理，碎片/映射和内核开销更准确。
- 211行低bit合并效果差需与20/C区分直接低精度舍入和高精度反量化合并再量化评估。
- 59行模拟抢占策略、5行全部服务命令未运行、132行访存非吞吐均已披露；没有部署失败的实测证据。

## 跨章关联

|章|具体接口|待复核|
|---|---|---|
|15|KV每token112KiB、摊薄访存、TTFT/TPOT|GQA缓存用KV头数；粗算忽略上下文实现特征，不当真实延迟|
|18/21/33|提示模板、adapter目录与EOS|服务必须同训练格式；HF/vLLM生成等价限于配置/数值误差|
|20/附录C|LoRA合并、量化基座|支持挂载模块/rank需按版本核查；直接4bit舍入与先高精度合并分开|
|附录E|并发压测与GPU计时|模拟steps与服务TTFT/TPOT、吞吐和队列时间分别报告|

## 事实核查来源与范围

实际打开vLLM官方LoRA文档 https://docs.vllm.ai/en/latest/features/lora/ 、OpenAI兼容服务文档 https://docs.vllm.ai/en/latest/serving/openai_compatible_server/ ，核对serve/enable-lora/lora-modules/model别名部署路径。打开PagedAttention作者论文摘要 https://arxiv.org/abs/2309.06180 ，核对块管理与共享KV定位。算术核对13/10步、18token利用率及ceil碎片；没有启动服务、没有GPU基准，当前线上文档不能替代用户部署版本实跑。
