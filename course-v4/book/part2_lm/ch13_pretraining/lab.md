# 第 13 章实验手册：预训练过程与规模规律

## 0. 准备

在 `course-v4/` 下运行：

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q book/part2_lm/ch13_pretraining
CUDA_VISIBLE_DEVICES="" python -m book.part2_lm.ch13_pretraining.tour
```

模块名后可指定部分号，范围为 **1–10**，可独立运行或按给定顺序运行：

```bash
CUDA_VISIBLE_DEVICES="" python -m book.part2_lm.ch13_pretraining.tour 1 2
CUDA_VISIBLE_DEVICES="" python -m book.part2_lm.ch13_pretraining.tour 3 4 9
CUDA_VISIBLE_DEVICES="" python -m book.part2_lm.ch13_pretraining.tour 5 6 7 8
CUDA_VISIBLE_DEVICES="" python -m book.part2_lm.ch13_pretraining.tour 4 --log outputs/book/ch13/runs/smoke-cpu-seed0
```

默认 CPU、两线程，HF 离线，只读本地数据，不下载。第 3 部分默认冒烟自动对拍；显式给 `--preset`（包括 cpu）或 `--steps` 时默认只训练一次，`--check-resume` 可显式开启对拍；第 5、6 部分默认只展示，须加 `--run` 才训练全部模型。第 4 部分默认日志缺失、第 9 部分默认检查点缺失时，会先跑一次当前配置的训练。因此不要把 `--preset mps/cuda` 当成只切换设备：预设会同时扩大训练预算。

默认值见 `tour --help` 和 `pretrain.PRESETS`。`--size tiny|small|medium|large|xl` 选阶梯；`--steps --batch --context --lr --warmup --amp --eval-every` 覆盖训练参数；`--n-train` 控制标准训练集前缀；`--samples` 控制生成条数。`--run-dir RUN_DIR` 指定第 3 部分新训练目录（第 4/9 部分沿用）；`--resume RUN_DIR` 在第 3 部分真正续训；`--ladder-fracs "0.25,0.5,1"` 指定各阶梯独立训练的预算比例。`--preset cpu|mps|cuda` 先填入建议配置，显式参数再覆盖；**设备仍由 `--device` 选择**。三套建议同时适用于完整训练、消融和阶梯任务，阶梯训练会覆盖模型大小。所有性能、内存、效果门槛均为 **待校准**，不要把建议当成已验证承诺。

`load_latex_lm` 固定遵循 latex-v1：分词器只从标准训练前缀训练；没有本地语料时会警告并降级为样例。样例运行只能证明流程通，不参与真实语料结果比较。所有产物位于 `outputs/book/ch13/`，包括 token 文件、run 日志、检查点与 PNG；未加 `--resume` 时重复运行第 3 部分会重写对应 run；长训建议用 `--run-dir` 单独命名。消融使用带种子的目录与 CSV；阶梯目录是 `ladder/<size>/frac<frac>/`，不同种子实验前应先归档整个 ladder。

### 数据口径

| 使用位置 | 输入与上限 | token / 字节 / 参数口径 |
|---|---|---|
| 1、3、5、6 | 标准 train 前 `--n-train` 条 | 每条 BOS…EOS 拼成一维流；训练忽略 BOS 目标，输出 softmax 仍含 BOS 类别 |
| 周期验证 | val 前 `eval_formulas` 条 | 每条独立滑窗；屏蔽 BOS 输出、EOS 作目标；目标加权，不截断长公式 |
| 最终验证 | val 前 `final_eval_formulas` 条，标准默认上限为 2000 | 原文 UTF-8 字节与被评测公式严格对应；BOS/EOS 不算字节 |
| 训练预算与日志 | `tokens_seen = step × batch × context` | 处理位置数，含被屏蔽 BOS 目标；batch 是梯度累积后的有效批量 |
| 6、7 | 三种参数口径都显式记录 | Kaplan 非嵌入参数扣除 token/学习式位置表；6ND 采用既有 `matmul_params` 口径，含共享输出矩阵，也含少量 norm 参数 |
| 7 合成演示 | 人造计算量和损失 | 明确标“合成数据”，不能冒充真实模型结果 |
| 8 | 外部拟合系数 | Hoffmann et al. 2022 §3.3 approach 3，只演示公式，不是本项目拟合 |
| 9 参考上限 | val 标准前缀的原文本身 | mathtext 可解析率；总体、矩阵类环境、非矩阵类环境分桶 |
| 9 生成 | `--samples` 条，从 BOS 开始 | 禁 BOS，遇 EOS 停；未遇 EOS 的截断样本仍计分母，同时报结束条数 |
| 10 泄漏 | 所用 train 前缀与 val 前 2000 条 | `normalization_key` 不同键交集数，不是重复行数 |

BOS/EOS 的处理使训练 loss 和验证 loss 的任务不同；不能把二者的差全部解释成过拟合。周期与末次验证的前缀可能不同，读曲线时核对 `eval_formulas`、`eval_tokens`、`eval_bytes`。tokens_seen 对应计算量，不能当作独立语料量或有效监督目标数。

缓存路径按训练前缀大小区分，元数据校验分词器完整指纹、数据版本与格式版本；已有缓存过期会抛 `StaleCacheError`，不会静默覆盖。改变数据内容或顺序须升级版本/使用新路径；确认不再需要旧缓存后，由学习者归档或删除旧的 `.bin` 与 `.json` 配对文件。

每步 JSONL 有 `type=train`，每次验证另写 `type=eval`，同一更新步出现两种记录是正常的；同一种类型的步数不能重复。CSV 只有验证记录。检查点用临时文件加 `os.replace` 原子替换，保存模型、AdamW、GradScaler、取样生成器、全局 RNG、配置、未完成验证区间的累计 loss 与日志字节位置。续训严格检查配置和数据摘要，截断日志到检查点字节位置后从下一步追加；不能在续训时修改总步数重定义余弦调度。

## 1. 数据与口径（13.1）

**做什么：** 加载标准数据，写/读本章 token 文件，校验读回一致；故意换版本触发拒绝。打印训练和验证分母，将初始化 loss 与 ln V 对照；计算学习曲线的路标：训练目标的一元熵 H1，以及仅用训练计数的加一一元/二元模型验证交叉熵。打印一个 epoch 的训练 token 数和当前计划 tokens_seen 折算的 epoch 数。

**预测题：** 初始 loss 是否必须精确等于 ln V？若忽略 BOS 目标，是否意味着训练 softmax 已经移除 BOS？原文每字节与每 token 损失为何不能直接互换？只学会频率时验证 loss 应与哪个路标比较？二元模型比一元模型多用了什么信息？

**读哪些输出行：** `数据版本`、`分词器哈希`、`训练token数`、`验证原文字节数`、`验证目标数`、`每验证目标平均字节数`、`读回一致`、`过期缓存拒绝`、`初始loss`、`lnV`；新增 `学习曲线路标`、`H1_nat_per_token`、`训练非BOS目标数`、`平滑k`、`nat_per_token`、`bits_per_byte`、`可预测类别数`、`一个epoch训练token数`、`计划tokens_seen`、`对应epoch数`。

**对照代码：** [tour.py](tour.py) 的 `part1`；[pretrain.py](../../../miniocr/train/pretrain.py) 的 `token_stream`、`initial_loss`、`ngram_landmarks`；共享 `latex_lm.encode_formulas`、`token_file.load_token_file`。

**怎样读结果：** ln V 是均匀分布交叉熵参照，不是初始化损失的断言。字节分母只来自对应原文，不能拿训练字节去除验证损失；缓存指纹匹配也不能替代数据版本管理。

**学习曲线的路标：** H1 是训练流中排除 BOS 目标后的 plug-in 一元熵，单位 nat/token，不是验证集或完整语言的不可约熵。一元模型使用 `(count(y)+k)/(训练非BOS目标数+k·(V−1))`，默认 **k=1**；二元模型使用 `(count(x,y)+k)/(count(x,非BOS目标)+k·(V−1))`。两者均在完整分词器词表中移除 BOS 类别再归一化；未见上文退回 V−1 类均匀分布，不用验证频数。二元第一目标的上文是 BOS，不统计跨公式 EOS→BOS。

验证按第 11 章协议：前 2000 条公式的内容 token 和 EOS 各作一次目标，BOS 不作目标且不可预测；bits/byte = 总 NLL / (ln 2 × 同一批原文 UTF-8 字节数)。把实际验证曲线与一元路标对照，判断是否超过了“只学频率”；与二元路标比较短程相关性学习进度，不能把这些路标当作性能保证或硬下界。`一个epoch训练token数` 就是含 BOS/EOS 的训练流长度；`对应epoch数 = steps×batch×context / 训练token数`，随机窗口允许重复，这只是处理量折算。

## 2. 学习率调度（13.2）

**做什么：** 对比完整训练示例与当前冒烟配置的 warmup+cosine，在边界步打印学习率并保存 `schedule.png`。从同一初始化、同一固定批做两次独立 AdamW 首步更新，分别使用原 loss 和放大后的 loss，逐参数打印平均绝对更新除以 lr。

**预测题：** warmup 最后一步与其后一步的 lr 有什么关系？放大梯度会让 Adam 首步更新也同比放大吗？零梯度、epsilon 和衰减会怎样影响这个直觉？

**读哪些输出行：** `调度配置`、`索引步`、`lr`、`调度图`、`Adam首步参数`、`loss倍率`、`平均绝对更新除以lr`。

**对照代码：** `tour.part2`、`_schedule`；共享 `optim.warmup_cosine`。

**怎样读结果：** 调度索引从零开始，日志更新步从一开始。lr（含正确/错误续训 lr）、C、N_opt、D_opt、6ND、FLOPs 用科学计数法，避免跨数量级输出失真；完整示例索引 0、1 的非零 lr 可以直接辨认。Adam 实验特意关闭权重衰减以隔离自适应项；每个参数张量内部的标量更新取均值。接近单位比例来自偏差修正后的自适应归一化，不表示任意一步或任意优化器都如此。

## 3. 冒烟训练与续训（13.3、13.4）

**做什么：** 默认冒烟同配置训练两条路径：不中断，与半程保存后重新构建模型再续训；比较参数最大差，并主动修改总步数验证拒绝。显式 `--preset` 或 `--steps` 时默认只训练一次，只有加 `--check-resume` 才额外对拍。`--resume RUN_DIR` 则从已有同配置目录继续。所有路径末次使用标准验证前缀，再只计算调度重启反例。

**预测题：** 只恢复模型与优化器是否足够？在非验证步中断，最近区间平均 train loss 怎样接续？有 dropout 时还必须保存哪种随机数？

**读哪些输出行：** `训练配置`、`训练模式`、`日志目录`、`初始loss`、`ln V`、`续训起点`、`val_nat_per_token`、`val_bits_per_byte`、`grad_norm`（JSONL）、`续训对拍`、`参数最大差`、`CPU逐位一致`、`续训配置错误`、`正确lr`、`错误从头lr`。

**对照代码：** `tour.part3`；`pretrain.train_pretrain`、`validation_metrics`、`window_logits`；共享 `get_batch`、`_param_groups`、`FormulaWindows`、`evaluate_lm`。

**怎样读结果：** CPU 相同配置要求参数逐位一致；MPS/CUDA 需另做后端确定性校准，不承诺跨设备逐位一致。`grad_norm` 是裁剪前全局范数。梯度累积按非 BOS 目标数量加权，微批有效目标数不同不能简单平均。`tok_per_sec` 当前记录最近一次训练更新的吞吐，不含验证；`elapsed_sec` 含当前进程内训练、评测和保存耗时，加上检查点累计时间，不含停机时间。

复用库进行真正续训时，保留同一 `cfg.steps`，调用 `train_pretrain(..., resume=True)`；`stop_after` 仅用于模拟中断，不属于训练计划。`--resume RUN_DIR` 必须包含第 3 部分，保持原来的 preset、seed、数据前缀、steps、batch、context、warmup、lr、amp、评测间隔等全部配置；缺检查点或配置不一致就报错。它不会重新初始化日志，已完成的计划不再更新。不与 `--check-resume` 或 `--run-dir` 合用。新训练若无 preset 且未显式给 steps，会自动开启对拍（默认 60 步）；任何显式 preset 或 steps 默认关闭，也可显式 `--check-resume` 开启。只有自己的检查点才应使用 `torch.load(weights_only=False)`。

## 4. 学习曲线（13.3）

**做什么：** 读取第 3 部分或 `--log` 指定目录的 JSONL，画训练逐步 loss、滑动平均、验证 loss、ln V，以及 lr 与裁剪前梯度范数两子图；输出 `summarize_log`。

**预测题：** 验证曲线末尾突然变化可能只是更换评测前缀吗？训练 loss 尖峰一定会导致 NaN 吗？短期 train/val 差能证明过拟合吗？

**读哪些输出行：** `学习曲线图`、`initial_loss`、`ln_vocab`、`best_val`、`best_step`、`final_train_minus_val`、`spike_steps`、`first_nonfinite_step`。

**对照代码：** `tour.part4`；`pretrain.plot_learning_curves`、`summarize_log`。

**怎样读结果：** 先对照第 1 部分同一验证前缀的一元/二元交叉熵路标，再看学习过程；训练 H1 只作分布统计参照。尖峰定义为超过此前有限 loss 的滑动中位数乘阈值；历史不足一个窗口时用已有历史。最早非有限步检测日志里的数值字段。最佳验证值可能来自较小前缀，不能直接宣称优于末次标准验证。摘要是诊断线索，不自动选择“最好模型”；默认检查点保存的是最近一次保存状态。

## 5. 组件消融（13.5）

**做什么：** 列出基线、无位置、学习式位置、LayerNorm、无残差缩放、不共享权重，比较参数口径和同一批上的初始 loss。默认不训练；`--run` 时逐项使用同一数据、种子和训练预算，写验证汇总 CSV。

**预测题：** 哪些改动会增加参数？移除位置编码是否一定改变参数数目？学习式位置表怎样影响总参数与 Kaplan 口径？初始 loss 较低是否预示最终更好？

**读哪些输出行：** `消融`、`修改项`、`参数量`、`非嵌入参数量`、`固定批初始loss`；训练后另读 `variant`、`seed`、`val_nat_per_token`、`bits_per_byte`、`delta`、`消融汇总`。

**对照代码：** `tour.ABLATIONS`、`part5`；[gpt.py](../../../miniocr/models/gpt.py) 的 `GPTConfig`、`GPT.n_params`。

**怎样读结果：** 每次只改变一个指定组件，但增加模块可能改变随机数消费顺序；同种子不代表所有共有权重都逐元素相同。至少做两个种子再讨论趋势，不把单次差异当成稳健结论。旧 `rope=False` 优先，强制无位置；新实验统一用 `pos_emb`。默认 RMSNorm 的属性/参数名保持不变，LayerNorm 额外含 beta；学习式位置同时支持 cache 的 `pos0`。

## 6. 模型阶梯与 6ND（13.6）

**做什么：** 打印层数、维度、头数、三种参数口径、每 token 估算 FLOPs、注意力额外 FLOPs 占比。D 分列为当前配置计划量 `D_plan=steps×batch×context` 和训练流一个 epoch 的 `D_epoch`，各有 6ND；不把默认冒烟的计划量冒充真实语料规模。对 `--ladder-fracs` 中每个比例先打印各 size 的 steps、warmup、D、6ND 和全部 run 的总预算，然后 `--run` 才训练。

**预测题：** 相同 token 预算时，各模型训练步数是否相同？相同计算量时能否继续给每个模型相同步数？context 增大时忽略注意力额外成本还合理吗？

**读哪些输出行：** `size`、`layers`、`dim`、`heads`、`total`、`non_embedding`、`matmul`、`flops_per_token`、`compute`、`D_plan`、`D_epoch`、`epoch_6ND`、`attention_extra_ratio`、`阶梯预算`、`frac`、`steps`、`warmup`、`D`、`compute_6ND`、`阶梯run数`、`总6ND合计`。

**对照代码：** [scaling.py](../../../miniocr/train/scaling.py) 的 `LADDER`、`parameter_counts`、`ladder_table`、`fraction_config`、`budget_steps`；`tour.part6`。

**怎样读结果：** 默认比例是 `0.25,0.5,1`，须为 (0,1] 内不重复的有限数。每个模型每个比例从同一种子初始化，**独立训练完整 warmup+cosine**，`steps=max(1,round(frac×steps))`，`warmup=min(缩放后steps,max(1,round(frac×warmup)))`；round 使用 Python 最近偶数舍入，极小预算会有重复步数；仅一步时只有预热。学习率上下限保持不变。目录为 `outputs/book/ch13/ladder/<size>/frac<frac>/`。不能拿长周期余弦的中间检查点替代短预算完整调度的最终点：Hoffmann et al. 2022 指出这种周期不匹配会高估中间 loss。

参数量用解析公式并与真实 GPT 对拍。Kaplan 2020 的前向估算是 `2N + 2·n_layer·n_ctx·d_attn`；以当前 context 和 `d_attn≈d_model`，额外项相对于参数矩阵乘法项约为 `n_layer·context·d_model/N_matmul`（`attention_extra_ratio`），不是额外项除以总 FLOPs。此处用含共享输出的 N_matmul 近似代入，6ND 仍不含这项，也不是实测成本。全部预算在第一场训练前打印；同 size 的比例预算因取整可能不精确成比例。固定 C 的预算仍可通过 `budget_steps(..., compute=C)` 另算。

## 7. scaling 拟合（13.7）

**做什么：** 先在明确标注的合成数据上拟合 `a·C^(-alpha)+Linf`，比较忽略 Linf 的纯幂律。再读取 `ladder/<size>/frac*/` 所有兼容 run 的最终标准验证点（也兼容旧 `ladder/<size>/`，标 frac=1），逐点打印 size、frac、N、D、C=6·N_matmul·tokens_seen、val nat/token。构造 log C 分箱的最低损失前沿，拟合并画图：散点按 size 着色、前沿点用黑圈高亮、拟合曲线用连续计算量网格绘制。

**预测题：** 为什么固定同一个 D 的最终点只能给出固定 D 下的 L(N)？为什么延长余弦周期后读取中间点会影响比较？前缀不同的周期验证为何不能混进前沿？少量点能可靠推断无限算力极限吗？

**读哪些输出行：** `数据=合成数据`、`偏移幂律拟合`、`忽略Linf`、`阶梯最终点`、`size`、`frac`、`C`、`val_nat_per_token`、`二维损失数据`、`实测评测点数`、`前沿点数`、`前沿点`、`阶梯偏移拟合`、`阶梯拟合图`，或未找到日志/跳过不兼容日志的说明。

**对照代码：** `tour.part7`；`read_ladder_points`、`fit_power_law`、`fit_power_law_offset`、`compute_frontier`、`synthetic_scaling`。

**怎样读结果：** 只取 step=该 run 计划 steps 的最终标准评测（前 final_eval_formulas 条），不采用周期评测。未完成 run 不参与；训练流/数据版本、分词器、验证数据、BOS/EOS、上下文、最终验证上限及实际公式/token/字节分母须兼容。每个 size 的多 frac 点是固定 N 下不同 D 的数据；跨 size 可组成 L(N,D) 数据，但本部分只拟合 L(C) 前沿，不声称已验证 Chinchilla 最优分配。

纯幂律在 log 空间闭式最小二乘；偏移搜索在每个候选 Linf 下拟合 log(y−Linf)，外层按原始 loss 空间 MSE 选最佳。前沿保留每个 log C 区间最低实测点及真实 C，不插值、不强制单调；少于三个可比较前沿点时仍画散点，暂不拟合。合成无噪声恢复只验证算法，不证明本项目服从同一规律；窄预算跨度与少量点不能支撑可靠的极限外推。

## 8. Chinchilla 分配（13.8）

**做什么：** 用注明来源的外部系数，计算 `C=6ND` 下 N、D 的闭式最优，与独立 log N 网格搜索比较；打印 N、D 随计算预算增长的指数，并展示小预算的公式外推。

**预测题：** N 与 D 的指数为何相加为一？D/N 是固定常数吗？网格解不是精确闭式值时，应比较绝对误差还是相对误差？

**读哪些输出行：** `系数`、`N预算指数`、`D预算指数`、`C`、`N_opt`、`D_opt`、`D_per_N`、`L_opt`、`网格N相对误差`、`网格D相对误差`、`网格loss相对误差`、`小预算提示`。

**对照代码：** `HOFFMANN2022_APPROACH3`、`chinchilla_loss`、`chinchilla_optimal`、`chinchilla_grid`。

**怎样读结果：** 来源为 Hoffmann et al. 2022 §3.3 第三种方法的拟合值；该 loss、N、D 的原实验范围与本项目不同。本章的小预算远在原范围之外，只演示约束优化，不提供硬件选型或训练配比结论。参考系数不应未经拟合就用于本项目决策。

## 9. 生成与可编译率（13.9）

**做什么：** 先评测验证原文的 mathtext 可解析率，按矩阵类环境分桶；加载第 3 部分检查点从 BOS 采样，报告条数、比例、Wilson 区间、EOS 结束条数和若干样例。缺检查点时先跑一次当前配置训练。

**预测题：** 合法矩阵为什么可能被 mathtext 拒绝？参考上限是否是模型性能的严格上界？短串能解析就表示数学正确吗？增加样本会怎样影响置信区间？

**读哪些输出行：** `参考上限桶`、`count`、`successes`、`rate`、`interval`、`生成检查点`、`实际训练步数`、`指标=mathtext 可解析率（可编译率近似）`、`ended_with_eos`、`样例`、`生成提示`、`截断提示`。

**对照代码：** `reference_parse_report`、`MATRIX_ENV`、`sample_report`、`parse_report`、`wilson_interval`；共享 `generate.generate`、`render.can_parse`。

**怎样读结果：** 矩阵桶匹配 `matrix/pmatrix/bmatrix/Bmatrix/vmatrix/Vmatrix/smallmatrix` 和 `array/cases/aligned/alignedat/align/gathered` 环境，允许星号和空格；不是完整 LaTeX 环境解析器。空桶比例与区间为 None，不能写成零。全失败或全成功仍有区间宽度。生成长度受 block_size 限制，未结束的样本保留在分母；可解析也不代表语义正确。冒烟模型的结果只证明流程跑通。

## 10. 常见错误（13.11）

**做什么：** 固定同一个 CPU 小模型初值，对比 3e-3、3e-2、3e0 三档学习率，各跑 60 步，每步随机取新批；每档重置相同种子的独立取样生成器，保证数据序列相同。每 10 步打印 loss/梯度范数，最后打印最后 10 步平均 loss 与首次非有限步。调用规范键交集函数审计实际划分，再人为混入训练公式；重印续训调度重启反例。

**预测题：** lr 太大一定马上 NaN 吗？如何从多步平均 loss 分辨学习停滞和单批噪声？按原串不同划分能否避免规范等价公式泄漏？

**读哪些输出行：** `学习率故障`、`lr`、`step`、`loss`、`grad_norm`、`学习率故障汇总`、`最后10步平均loss`、`平均所用步数`、`首次非有限步`、`泄漏检查`、`注入条数`、`规范键交集`、`正确lr`、`错误从头lr`。

**对照代码：** `tour.part10`、`_resume_schedule`；`pretrain.leakage_overlap`；共享 `latex_corpus.normalization_key`。

**怎样读结果：** 故障实验固定 CPU 小模型，不随完整训练预设扩大；不使用梯度裁剪，以暴露不稳定。没有非有限值时输出 None，应观察震荡、停滞和损失水平，不能预设“发散”或编造 NaN。检测到非有限 loss/梯度立即停止该档，汇总为最后最多 10 个实际执行步并报告数量；没有非有限值时则完整执行 60 步。交集按不同规范键计数，注入条数与交集数未必相等；标准数据若出现非零交集应先查数据管道。

## 完整训练（在 Mac 或 5080 上）

### 运行前检查与预测

先由学习者释放 5080：停掉占用的 vLLM，再确认设备可用。不要在仍设置 `CUDA_VISIBLE_DEVICES=""` 的终端里运行 CUDA 命令。先激活已安装本书依赖的 Python 环境，下面命令统一使用 `python`。

长训练前必须先跑通 **前向 → 反向 → 保存 → 加载**，同一设备、同一 dtype 检查短训；再估算内存和预算。下面的完整训练由你运行，正文只提供 CPU 冒烟结果。先记录：预计哪个环节最耗时、能否逐位对齐、可接受内存和效果门槛（均待校准）。

```bash
# Mac 短训与加载检查
python -m book.part2_lm.ch13_pretraining.tour 3 --device mps --preset cpu --check-resume
# 5080 短训与加载检查；保留小模型/小预算，单独试 bf16
python -m book.part2_lm.ch13_pretraining.tour 3 --device cuda --preset cpu --amp bf16 --check-resume
```

| 设备与dtype | 前向 | 反向 | 保存 | 加载 | 参数最大差 | 实际内存/用时 | 下一步 |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

### 完整训练与曲线

**运行前预测：** warmup 是否足够？验证最佳点预计出现在什么阶段？末尾 train/val 差该如何区分上下文和输出类别口径的影响？先填写假设，再启动。

```bash
python -m book.part2_lm.ch13_pretraining.tour 3 4 9 --device mps --preset mps --eval-every 200 --run-dir outputs/book/ch13/runs/full-mps-seed0
python -m book.part2_lm.ch13_pretraining.tour 3 4 9 --device cuda --preset cuda --eval-every 200 --run-dir outputs/book/ch13/runs/full-cuda-seed0
```

上述命令默认只训练一次；`--check-resume` 会额外消耗一次完整计划的更新预算，先在短训中使用。中断后使用相同配置续训（两种设备任选其一）：

```bash
python -m book.part2_lm.ch13_pretraining.tour 3 4 9 --device mps --preset mps --eval-every 200 --resume outputs/book/ch13/runs/full-mps-seed0
python -m book.part2_lm.ch13_pretraining.tour 3 4 9 --device cuda --preset cuda --eval-every 200 --resume outputs/book/ch13/runs/full-cuda-seed0
```

不要改变 `--steps` 延长已有计划。首次校准可用命令行缩小 steps 并相应缩小 warmup，但续训须保留缩小后的全部配置。门槛：**待校准**。

| 设备 | 模型 | 种子 | 实际步数/token | 最低val及步数 | 末次val nat/token | bits/byte | train−val | 用时 |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |

### 完整消融

**运行前预测：** 先给变体排序并解释机制；哪些差异可能只来自参数量？两个种子是否会改变排序？运行后逐行核对相同训练数据、有效批量、步数和标准最终验证分母。

```bash
python -m book.part2_lm.ch13_pretraining.tour 5 --device mps --preset mps --eval-every 200 --run --seed 0
python -m book.part2_lm.ch13_pretraining.tour 5 --device mps --preset mps --eval-every 200 --run --seed 1
# 5080：同样至少两个种子
python -m book.part2_lm.ch13_pretraining.tour 5 --device cuda --preset cuda --eval-every 200 --run --seed 0
python -m book.part2_lm.ch13_pretraining.tour 5 --device cuda --preset cuda --eval-every 200 --run --seed 1
```

| 变体 | 种子 | 步数 | val nat/token | bits/byte | 与基线之差 | 用时 |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

统计显著性与实际收益门槛：**待校准**。保留每个种子的原始数据，不只保留最好的那次。

### 阶梯训练与拟合

**运行前预测：** 固定 token 数时最大模型一定最好吗？固定计算量时这个判断是否还成立？准备怎样防止只拟合少数低预算点产生误导性极限？

```bash
python -m book.part2_lm.ch13_pretraining.tour 6 --device mps --preset mps --eval-every 200 --ladder-fracs "0.25,0.5,1" --run
python -m book.part2_lm.ch13_pretraining.tour 6 --device cuda --preset cuda --eval-every 200 --ladder-fracs "0.25,0.5,1" --run
# 训练完成后，拟合本身只需 CPU
CUDA_VISIBLE_DEVICES="" python -m book.part2_lm.ch13_pretraining.tour 7
```

Mac 与 5080 命令是两种选择，连续运行会重写同名阶梯目录；先归档再换设备或种子。可先去掉 `--run` 只看每个比例和所有模型的预算总计；同一 frac 各模型 D 相同，不同 frac 提供不同 D，每个点均完成自己的调度。先核对最后标准验证分母，再解释计算前沿与 L(N,D) 数据；仍不能直接称为已经验证 Chinchilla 最优分配。拟合门槛、误差容许范围与所需预算覆盖：**待校准**。

| 模型 | frac | 种子 | steps/warmup | N非嵌入 | Nmatmul | 实际D | 6ND | val nat/token | bits/byte | 用时 |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

| 数据来源/评测协议 | 前沿点数 | a | alpha | Linf | 残差 | 拟合范围 | 局限与下一步 |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

### 实验记录

| 部分/命令 | 运行前预测 | 种子/数据版本/配置 | 关键输出与分母 | 实际用时 | 解释与下一步 |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

所有表留待运行后填写；不要将配置值、预测或合成恢复结果当成真实训练效果。快速测试完全使用人造 token 流、小模型与临时文件，不加载真实语料。
