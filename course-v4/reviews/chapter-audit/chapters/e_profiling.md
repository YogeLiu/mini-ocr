# 附录E逐章审查

## 阅读记录

源文件`course-v4/book/appendix/e_profiling/chapter.md`；SHA256 `6ef2c12061d01f436fce3abdf75a8fe76c36d90409087fc0ea7f7d4ca87ef306`。完整通读1–218行，单次输出无截断。

## 主线与关键定义

CPU两线程tiny训练，故意逐元素weightnorm热点；record_function划span，cpu_time_total含子调用，自身时间看aten热点，count解释Python循环。schedule wait1/warmup1/active3排除初始化和预热，prof.step推进。向量化后不开profiler重测，避免instrumentation对小算子的放大。roofline用I=FLOP/byte和平衡点峰值算力/带宽判断理论上界；AXPY I=1/6，方阵GEMM理想每矩阵一次IO时I=n/6。加线程和工作集大小只是证据；小工作集不并行不代表算力受限。GPU只给未运行入口和同步计时提醒，不给实际GPU性能结论。

## 已确认问题

未确认P1/P2。数学公式和例子一致，对CPU量测的结论有明确机器/负载/计数边界。未把线程趋势当充分证明。

## 待验证线索与已声明局限

- 183行迁移到CUDA时应补充CPU时间与device/CUDA时间区分：只加ProfilerActivity.CUDA、仍用171行cpu_time_total排序，只看到host分派/等待，不一定找到设备kernel热点；应读GPU时间列及时间线。正文GPU未实跑且建议看trace，因此本轮列为实施补充，不判CPU示例错误。
- 45行“次数不受影响”以同执行路径和固定逻辑步数为前提；动态分支、异步数据/超时/编译路径可随观测条件改变。此处固定训练循环的count解释正确。
- 69/140行工作集大小对比应比较单位数据吞吐与相同操作次数，不能只看更大数组总耗时更长；正文已说该实验证据信号不强，无确定带宽推断。
- 68行计算受限线程扩展近线性还受线程布局、cache、矩阵大小与频率影响，72行已有其他原因限定。
- 未GPU测量、不写trace、名次除故意热点外不作断言均清楚披露。未重新运行CPU性能测试，以避免不必要负载且数值本就机器相关。

## 跨章关联

|章|具体公式/证据|宏观核对|
|---|---|---|
|15/16|访存模型、MFU与计时|单位FLOP/byte一致，GPU异步计时先后同步，不把CPU span时间当GPU算子耗时|
|31/33/34/35|共享卡墙钟与峰值|书内训练秒数含其他任务和CPU开销，不能据此归因架构kernel效率|
|附录B|IO与FLOP分离|FlashAttention访存减少需目标GPU验证，Python分块CPU速度无代表性|
|附录D|服务吞吐和延迟|并发、队列、prefill/decode、TTFT/TPOT各自计量，模拟steps不等时间|

## 实际来源与验证范围

打开PyTorch官方profiler教程 https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html 和API https://docs.pytorch.org/docs/2.14/profiler.html ，核对schedule、record_function、key_averages与活动类型入口。解析核对AXPY/GEMM强度及8192元素×3步的item/select计数；未复现机器特定性能或GPU trace，不宣称部署性能已验证。
