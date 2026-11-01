# 第 16 章实验手册：训练工程

## 0. 准备

在 `course-v4/` 下运行：

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q book/part2_lm/ch16_training_engineering
CUDA_VISIBLE_DEVICES="" python -m book.part2_lm.ch16_training_engineering.tour
```

部分号范围是 **1–11**，可单独运行，也可按指定顺序组合，编号与下面各节及 `tour.PARTS` 一致：

```bash
CUDA_VISIBLE_DEVICES="" python -m book.part2_lm.ch16_training_engineering.tour 7 8
CUDA_VISIBLE_DEVICES="" python -m book.part2_lm.ch16_training_engineering.tour 9 --seed 1
CUDA_VISIBLE_DEVICES="" python -m book.part2_lm.ch16_training_engineering.tour 10 11
```

`--device cpu|mps|cuda` 默认 `cpu`；`--seed` 默认 `0`。`--peak-tflops` 接受有限正数，由学习者填写当前设备、当前计算精度对应的峰值 TFLOP/s，注意乘加和稀疏口径；不填时 MFU 明确显示“待验证：请用 --peak-tflops 填写你设备的峰值”。本手册不提供任何 GPU 峰值数字。位置参数保留顺序，重复编号会重复运行；各部分重置种子，单独运行也可复现数值实验。

本章验收只用 CPU，两线程，离线、不下载、不训练课程模型、不读语料、不写产物。仅有一次更新的状态初始化和少数完整训练步的计时，不运行长训练。程序设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`OMP_NUM_THREADS=2`、`MKL_NUM_THREADS=2`、`OPENBLAS_NUM_THREADS=2`，并在 `main` 中调用 `torch.set_num_threads(2)`。第 1–5 部分及第 6 部分的 bf16 静态账固定使用 CPU；其余部分遵循 `--device`。显式选择不存在的设备会报错。

每行采用 `key=value`，标记“数学结论”“本项目观察（设备、线程）”或“待验证”。数学结论依赖所写假设；观察只代表本机 CPU、当前模型、PyTorch 实现与线程设置，不推广到 GPU。模型全部随机初始化，损失与输出不能代表学习效果。计时不含下载、语料加载或分词。

对照代码：[tour.py](tour.py)、[test_ch16_training_engineering.py](test_ch16_training_engineering.py)、[precision.py](../../../miniocr/train/precision.py)、[resources.py](../../../miniocr/train/resources.py)。复用的口径在 [memory.py](../../../miniocr/train/memory.py)、[lm.py](../../../miniocr/train/lm.py)、[pretrain.py](../../../miniocr/train/pretrain.py)。先写预测，再运行，把结果填入末尾空表。

## 1. 浮点格式（16.1）

**做什么：** 从符号位、指数位、尾数位推导 fp32、fp16、bf16 的最大值、最小正规数、最小非正规数和 eps，与 `torch.finfo` 对拍。`finfo` 没有最小非正规数属性，该列是 `tiny*eps` 的派生值；另用 `nextafter(0,1)` 实测。尾数位不含隐含的首位一。

再观察 `[70000, 1e-8, 1+1e-3, 1+1e-2, 0.1]` 转换后回到 fp32 的数值，检查 bf16 的 `256+1`，计算各尺度的间隔。正规数 binade 内 `spacing=2^floor(log2|x|)*eps`；它是所处区间的步长，不是任意实数到舍入值的距离。最大有限数处也不把“到 inf 的距离”当作间隔。

**预测题：** bf16 与 fp32 的指数位相同，为何它们的最大有限值仍不同？为什么 eps 更大的格式并不更容易发生指数溢出？小于最小正规数是否必然为零？

**记录：** 位数、公式值/finfo 值/一致性、nextafter 值、舍入后的数组，以及各 x 的间隔。区分指数范围、正规数间隔、非正规数与舍入到零。

## 2. 前向溢出与累加（16.2）

**做什么：** 对元素均为 300 的 64 维向量，在 fp16 里计算均方和朴素 RMSNorm，与内部转 fp32 的项目 `RMSNorm` 对照。比较包含 12 的 logits 的朴素 `exp/sum`、减最大值的稳定写法及 fp32 softmax。最后把 0.01 在 fp16、fp32 中各逐次累加一万次，每次加法都落回对应 dtype。

**预测题：** 归一化结果本应接近一，为何中间值仍可能溢出？稳定 softmax 为什么能改变中间范围而不改变数学结果？顺序累加何时会开始丢掉每次的小增量？放大 loss 能修复已溢出的前向吗？

**记录：** 均方、朴素与项目 RMSNorm、三个 softmax 结果以及两种最终累加值。保留实际出现的 `inf`、`nan` 或零，不把它们改写成预期答案。累加实验不是允许内部高精度累加的 reduction，也不是用 `cumsum` 假定硬件的累加精度。

## 3. 梯度下溢与loss scaling（16.3）

**做什么：** 固定种子生成一万个标准差为 `1e-7` 的正态梯度，分别乘 `1、2^10、2^16、2^24` 后转 fp16，再在 fp32 中除回。打印零比例、inf 比例和相对 L2 误差 `||g恢复−g参考||₂/||g参考||₂`。

另用微小 MLP 的小 loss，一次前向保留计算图，复用该图对 fp32 参考、不缩放和缩放路径反向对拍。低精度路径在梯度保存边界模拟 fp16 舍入；模型权重与前向仍为 fp32，不使用 CPU `autocast(float16)`。

**预测题：** 放大 S 是否能无限提高精度？放大后的梯度变成 inf 时，除回 S 能恢复吗？这里模拟的是低精度反向计算的全部误差，还是梯度存储舍入的一处误差？

**记录：** 每个 S 的三项统计、MLP 的 loss、零比例和相对误差。区分“放大范围以躲开下溢”和“增加尾数位数”；本实验没有增加尾数位。

## 4. GradScaler与裁剪顺序（16.4）

**做什么：** 教学版 `SimpleGradScaler` 按 `scale → backward → unscale_ → clip → step → update` 执行。用增长间隔为两步的八步序列，在第二、第三步人为注入 inf，打印当前 scale、是否跳步、参数是否更新和下一步 scale。同样序列与 `torch.amp.GradScaler("cpu")` 逐步对拍；只有构造不受支持时才显示跳过。

再对 `[3,4]` 方向的同一组梯度比较先 unscale 再 clip、先 clip 再 unscale，记录裁剪输入范数、实际更新梯度范数与更新量。缩放器状态字典兼容 torch 的键，测试还验证保存/恢复连续成功计数、隐式 unscale、重复 unscale/step 报错和 nan 跳步。

**预测题：** 连续两次 inf 后，需要多少次正常更新才会增长？为什么裁剪后的梯度都是有限值，也可能仍然更新错了？应在一次迭代的哪个边界保存 scaler 状态？

**记录：** 两个实现的全部逐步结果、裁剪顺序的差异。教学版只覆盖稠密 fp32 梯度与普通优化器；没有实现 fused/AMP-aware 优化器或分布式缩放。`step` 返回是否执行，与 torch 版返回优化器返回值的接口语义不同。

## 5. 主权重为什么要fp32（16.5）

**做什么：** 从一开始，在 bf16、fp16、fp32 中分别执行一百次 `p -= 1e-4`，打印最终权重，与实数结果比较。这里每次更新直接舍入回参数 dtype。

**预测题：** 一百次小更新的总量明明可表示，为什么低精度参数仍可能不动？“最后一次才转换成低精度”与“每一步都转换”相同吗？主权重与优化器的一阶矩是同一份状态吗？

**记录：** 三种最终权重、逐步舍入与实数运算的差异。fp32 主权重累积更新，低精度计算副本按需同步；第 6 部分只演示一次更新后的字节账，并不实现完整的副本同步训练器。

## 6. 静态显存账（16.6）

**做什么：** 先列数学口径：fp32 + AdamW 为 `4+4+8=16` 字节/参数；显式 bf16 计算副本、bf16 梯度、fp32 主权重和 fp32 m/v 为 `2+2+4+4+4=16` 字节/参数。两者总量相同，组成不同；冻结参数只有权重，不再按全参开销收费。

用 tiny 骨架、演示词表做一次反向与 AdamW step，真实遍历参数、梯度和优化器张量。共享 storage 只算一次，`step` 标量单列，`exp_avg` 与 `exp_avg_sq` 分列。混合精度例子先将 bf16 梯度转给主权重，step 后释放主梯度，保留 bf16 梯度；若保留主梯度，会多出一份 fp32 梯度，测量函数会如实记录 `master_grads`。

再为项目 medium、large 和本地 Qwen3 骨架列全参 fp32、全参混合精度、仅训练百分之一参数的静态账。项目阶梯在本实验统一使用演示词表大小 512；可训练参数量按四舍五入取整数，不是某个实际 LoRA 配置。

**预测题：** 为什么 `AdamW.step()` 前后优化器状态字节不同？为什么两种精度静态总量可能一样，训练峰值却不同？共享输出层与不共享输出层对 Qwen 骨架参数量影响多大？

**记录：** 四类估算、五类混合精度实测（m/v 分开）、step 标量、估算与实测的差值、冻结口径与参数来源。`qwen_like_params` 使用嵌入、q/k/v/o、q/k norm、三个 MLP 矩阵、两个块 norm 和最终 norm，不将显式 `head_dim` 错写成 `hidden_size/n_head`。

Qwen 配置只由 `local_model_configs` 读取本地 JSON；直接读取 Qwen3-0.6B 生成模型的配置，缺文件时使用内置回退配置并标“待验证”。共享输出层字段取自同一 JSON，缺失则暂按共享并标“待验证”，同时打印共享/不共享的两种参数量。不得把模型名中的量级直接当精确参数数目。静态测量不包括 buffer、分配器缓存、Python 对象或临时工作区。

## 7. 激活显存：估算、实测与拟合（16.7）

**做什么：** 对 medium、fp32、manual 注意力扫描 `(B,T)=(4,64)、(8,64)、(4,128)、(4,256)`，先给逐项公式，再用 `SavedTensors` 按 storage 去重实测。估算精确域是全参训练、MHA、RMSNorm、无 dropout、无 learned position、无检查点，且输入和目标为独立连续 int64 张量。测试以小模型验证误差小于百分之二。

分解包括 norm 输入/结果与 rsqrt、共用的线性输入、q/k/v、输出投影输入、SwiGLU 四个中间量、softmax 权重、布尔 mask、最终 norm/head 输入、交叉熵 log-softmax、idx/targets 和标量。参数和 RoPE buffer 不重复计入。**返回的 logits 本身没有被反向保存，单独列字节，不加进 `SavedTensors` 总量；scores 本身也未保存。** 因此这张表是保存量，不是所有仍存活张量的总量，也不是峰值显存。

固定 B 后，用三个 T 的实测扣掉最终 norm、交叉熵、索引和标量，除以层数，拟合每层 `aT+bT²`，与公式系数对照。完整保存量实际为 `a总T+b总T²+4`。另对 SDPA 只实测，不声称设备/后端内部保存项可由 manual 公式预测。

**预测题：** B 翻倍时哪些项不翻倍？为何 T 翻倍通常超过两倍？只把所有 fp32 字节除以二能正确估计混合精度吗？将 targets 直接设为 idx 而不是克隆，会改变哪个 storage 的去重结果？

**记录：** 四组估算/实测/相对误差/基准倍数，分解占比，拟合的 a、b 和 SDPA 保存量。布尔 mask 每层为 `T²` 字节，另有固定标量，所以 B 翻倍只是接近两倍。`bytes_per=2` 的扩展公式针对整个计算模型（含 buffer）转为 bf16/fp16 的副本；RMSNorm 中间量和交叉熵仍为 fp32，不能与普通 autocast 的转换缓存混为一谈。

## 8. 梯度检查点（16.8）

**做什么：** 复制同一模型，给副本逐块加 `checkpoint_blocks`。在相同输入目标上比较保存量和梯度；每条路径预热一次，前向加反向计时三次取中位数。hooks、FLOP 计数器和计时分开运行，计时不含优化器。

再用 `FlopCounterMode` 分别计数前向加反向，比较 FLOPs 与额外重算量。这里采用非重入检查点，只重算块；输出层不会因此重算，自动提前停止也可能省去块末尾不再需要的算子。观察实际比值，不把理论约 `4/3` 当作必须满足的恒等式。

**预测题：** 检查点为何没有消除所有激活？重算应增加哪部分计算？同样 FLOPs 比能否推出同样的时间比？如果打开 dropout，梯度对拍还需要控制什么？

**记录：** 两条路径的保存字节、中位秒、FLOPs、梯度最大绝对差和三个比值。不要断言加检查点在每次短 CPU 计时中都更慢；正确性依据梯度，重算依据计数，时间受调度影响。

## 9. 变长样本的梯度累积（16.9）

**做什么：** 四条样本有效长度为 `[2,16,16,10]`。比较整批 token 平均交叉熵与 `token`、`micro_mean`、`sample` 三种累积方式。分组 A 为 `[0,1] / [2,3]`，分组 B 为 `[0,2] / [1,3]`；交换的是两条同长度但内容不同的样本，两个微批的有效 token 数仍分别为 18 和 26。

每个 token 的权重：token 方式统一为 `1/44`；micro_mean 两组分别为 `1/18×1/2`、`1/26×1/2`；sample 四条依次为 `1/2×1/4`、`1/16×1/4`、`1/16×1/4`、`1/10×1/4`。换分组会让不同内容的 token 得到不同的 micro_mean 权重。

最后把补齐位置替换为 BOS=0，复现 `pretrain.py` 的非 BOS `count/valid`：每个微批 mean loss 乘本微批有效数除以总有效数，跳过有效数为零的微批，与 token 方式对拍。

**预测题：** sample 方式是在算错同一个目标，还是在优化另一种目标？为何两个微批的总长度没变，micro_mean 梯度仍会随分组改变？什么时候除以微批数刚好正确？

**记录：** 每种方式、每种分组与整批的相对 L2 差，以及换组前后的梯度差。整批参照目标是有效 token 平均；按样本平均可以有合理用途，但必须明确目标。

## 10. 6ND、实测FLOPs与MFU（16.10）

**做什么：** 对同一小 GPT 分别计前向及前向加反向 FLOPs。N 使用旧 `matmul_params` 口径，包含输出矩阵及少量 norm 参数，输入嵌入查表不按矩阵乘重复收费。比较前向 `2ND`、训练 `6ND`、含注意力项的估算。

manual 完整 `QKᵀ+AV` 每层前向为 `4BT²d`，训练近似乘三；计数器包括被因果 mask 遮住的位置。Kaplan 的有效因果半矩阵近似为每层每 token 前向 `2Td`，训练乘三，二者不要混用。计数器主要覆盖矩阵乘，并非所有逐元素指令或真实硬件指令计数。

再预热两个完整更新步，测五步 `zero_grad + forward + backward + AdamW.step` 的中位秒，得到 tokens/s。打印 `6N×tokens/s` 与按实测 FLOPs 换算的实现 FLOP/s；峰值由学习者填写，才计算 MFU。耗时预算为 `6ND/(峰值FLOP/s×MFU)`，峰值和 MFU 在示例中保留“学习者填写”。

**预测题：** 只增加上下文而不增加 N，为什么 `6ND` 的误差仍可能扩大？如果填写 bf16 峰值，却计时 fp32 路径，MFU 说明了什么？实测 FLOPs 更多是否一定更慢？

**记录：** N、tokens、前向及总 FLOPs、各估算比、训练步中位时间、tokens/s、两种实现 FLOP/s。默认 tour 的吞吐是 fp32；合成输入提前放到设备上，不含数据读取。小模型 CPU 短计时不能用于宣称 GPU 利用率。

## 11. 资源核算报告（16.11）

**做什么：** `resource_report` 先给本章模型静态、激活保存量与合计；一次短计时初始化优化器状态后，用 `measure_static_bytes`、`SavedTensors` 对照三列表：估算、实测、相对差。静态差值中单列 AdamW 的 step 张量。

报告中的 `batch` 是微批大小，每次更新 tokens 为 `batch×context×accum`，每步 FLOPs 也乘 accum；激活只计一个微批，不把各微批的计算图一起保存。`mixed` 表示显式 bf16 计算副本加 fp32 主权重，激活使用上述副本公式，不等同于通用 autocast。冻结时按可训练比例估静态，激活用全参保存量作预算参考并标待验证；具体冻结位置会影响反向图，FLOPs 也不能直接乘可训练比例。SDPA 等不支持的公式返回待验证与空值，不伪造合计。

给出 token 预算和实测吞吐即可计算 `预算/tokens_per_sec`；同时给出峰值则可计算 MFU 并核对等价的 `6ND/(峰值×MFU)`。tour 最后打印 Mac / 5080 的自填表，实测和相对差列留空。

**预测题：** 静态加保存量为什么不是训练峰值？累积增加到多步，会让静态、保存量、每步 tokens 和 FLOPs 分别怎样变化？仅有一个冻结比例，能否知道需要保留哪些激活？

**记录：** 三列表、step 标量字节、每步 tokens/FLOPs、预算与预计时间。不要将前向保存量与 AdamW 工作区、返回 logits、分配器碎片等忽略项混成同一概念。

## 在 Mac / 5080 上自己测

本节是学习者在自己设备上执行的步骤，本章验收没有运行 GPU；GPU 上的数全部由学习者测，本书不提供。示例仍在 `course-v4/` 下；Mac 请将 Python 路径换为自己已装好 PyTorch 的解释器，不要求下载模型。

```bash
python -m book.part2_lm.ch16_training_engineering.tour 6 7 8 10 11 --device mps
python -m book.part2_lm.ch16_training_engineering.tour 6 7 8 10 11 --device cuda
```

CUDA 实验先取消 CPU 验收时的 `CUDA_VISIBLE_DEVICES=""` 设置。先跑默认 fp32 路径，确认数值对拍，再测低精度；`--device` 只选择设备，不自动启用 autocast。提供 `--peak-tflops` 时填与当前精度及乘加/稀疏口径匹配的峰值，并记录来源；未确定则留空。某些设备/版本不支持所需 dtype 或 FLOP 计数算子时，保留错误与配置，标记待验证，不能把别的设备数字填进去。

1. 记录设备型号、系统、PyTorch 版本、dtype、注意力实现、B/T、层数、词表、线程和随机种子，先看第 7 部分的估算/实测对拍。`SavedTensors` 记录 autograd 保存的存储，不是设备分配器峰值。
2. CUDA 峰值：创建模型/输入/优化器，先预热使 AdamW 状态建立，清理临时 Python 引用，调用 `torch.cuda.synchronize()` 和 `torch.cuda.reset_peak_memory_stats()`。执行一个或少数完整更新步，再同步并读 `torch.cuda.max_memory_allocated()`。这是重置以来张量分配的峰值字节；基线模型及状态仍计入，不是显卡的全部物理占用，也不要和 `memory_reserved` 混用。不要在峰值试验中安装保存张量 hooks 或 FLOP 计数器。
3. MPS：同步用 `torch.mps.synchronize()`；`torch.mps.current_allocated_memory()` 是当前张量占用，`torch.mps.driver_allocated_memory()` 是驱动为进程分配的当前总量，口径还含缓存等。这里没有把任一 API 说成重置以来的峰值 API。可在前向、反向、step 后同步采样并取观察到的最大值，但可能漏掉算子内部瞬时峰值；真正峰值仍标“待验证”。
4. bf16 autocast 吞吐短脚本思路：使用相同随机 GPT、fp32 参数与 AdamW，在设备支持时，仅将 forward/loss 放入 `with torch.autocast(device_type=device, dtype=torch.bfloat16):`；退出该上下文再 backward、clip、step。预热后同步计时，若干完整步取中位数，不扩展成训练任务。bf16 不在此实验中使用 scaler；若另试 fp16，则按第 4 部分顺序使用真实 GradScaler。dtype/后端不支持时标待验证，不悄悄换精度。autocast 的静态与转换缓存不等同于第 6 部分的显式低精度副本账，需重新测量。
5. 从实测 tokens/s 计算两种实现 FLOP/s，再除以匹配精度的峰值。把 fp32/manual、SDPA、bf16 autocast 分行记录；不要用本机 CPU 的时间比推断自己的 GPU 时间比。

| 项目 | 估算 | 实测（你填） | 相对差（你填） |
|---|---|---|---|
| 静态显存 | | | |
| 激活保存量 | | | |
| CUDA 峰值张量显存 / MPS 驱动当前量采样最大值（注明口径） | | | |
| tokens/s | | | |
| MFU | | | |

## 记录表

数据单元格全部留空，按实际输出填写。可复制行区分设备、dtype、种子与重复测量；不要只保留最快的一次。

| 部分 | 要记的量 | 配置／来源／种子 | 预测 | 实际记录 | 解释与疑问 |
|---|---|---|---|---|---|
| 1 | 格式位数、公式/finfo、非正规数、舍入、spacing | | | | |
| 2 | 均方与RMSNorm、softmax、逐步累加 | | | | |
| 3 | 各S的零/inf比例和相对误差、MLP对照 | | | | |
| 4 | scaler逐步scale/跳步/参数、裁剪范数与更新量 | | | | |
| 5 | 各dtype反复小更新后的主权重 | | | | |
| 6 | 静态分项、step差值、冻结账、Qwen配置与共享来源 | | | | |
| 7 | 四组估算/实测/误差/倍数、分解、a/b拟合、SDPA | | | | |
| 8 | 检查点前后字节、梯度差、时间、FLOPs与比值 | | | | |
| 9 | 三种累积及pretrain、两种分组、与整批及跨分组差 | | | | |
| 10 | 前向/训练FLOPs、估算比、完整步时间、tokens/s、MFU | | | | |
| 11 | 资源三列表、step标量、每步tokens/FLOPs、预算时间 | | | | |
| 自测 | Mac/5080配置、fp32/AMP、峰值或采样口径与来源 | | | | |
| 全部 | 各部分与总用时、测试结果 | | | | |
