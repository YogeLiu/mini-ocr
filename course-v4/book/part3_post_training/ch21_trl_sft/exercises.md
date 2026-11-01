# 第 21 章　习题：用 TRL 做 SFT 实验

除特别说明，题中的实测数字都来自正文的"主实验"（本项目观察，RTX 4090 租用，FP32 权重 + bf16 autocast，train 接受 2023 条，val 前 200 条，800 步，原始文件在 `outputs/book/ch21/gpu/`）。

---

## 一、概念题

**21-1.** 本章配置里写了 `completion_only_loss=True`，正文却说它"在本章的数据路径上没被用到"。说明 TRL 1.15 在纯文本 SFT 中按什么顺序决定 labels 的来源，以及本章为什么走的是第一条路。

**21-2.** 为什么"打印第一个 batch"一定要从 `trainer.get_train_dataloader()` 取，而不是从自己传进去的数据列表取？举出两种框架可能在中间改动数据的操作。

**21-3.** 判断对错并说明理由：
(a) 可训练参数的个数等于公式值，就说明 LoRA 挂对了。
(b) 本地 CPU 上 `loss_step_parity` 通过，说明 GPU 上 TRL 的 loss 也是对的。
(c) adapter 重载后 val EM_norm 和训练时相同，就说明输出复现了。
(d) 训练日志 loss 从第 600 步的 0.0089 变成第 800 步的 0.0091，说明模型在第 600 步以后开始过拟合。

**21-4.** 对拍时 AdamW 的 eps 用 1e-6，正式训练用默认的 1e-8。为什么对拍要这样改？为什么这不会让对拍"作弊"？

**21-5.** 本章 TRL LoRA（131/200）和第 20 章自写 LoRA r64 q,k,v,o（134/200）都在同一批 val 上评测。列出至少四个两者之间不同的地方，并说明为什么这张表不能回答"TRL 和自写哪个好"。

**21-6.** 复现一次实验，除了 adapter 权重还要保存什么？按"会不会改变输出"给出至少五类。

---

## 二、手算题

**21-7.** Qwen3-0.6B 有 28 层，隐藏维 1024，`q_proj` 为 1024 → 2048，`k_proj`、`v_proj` 为 1024 → 1024，`o_proj` 为 2048 → 1024。
(a) 算 r = 32、只挂 q、v 时的 LoRA 可训练参数个数。
(b) 算 r = 16、挂 q、k、v、o 时的个数，和 (a) 比较。
(c) 可训练参数的名字应当有多少个（两种情形各算）？
(d) 已知本章 `trainable-parameters.json` 里参数总数是 600,637,440，求基座参数量和可训练比例。

**21-8.** 一次更新有两个 microbatch。第 1 个监督 3 个 token，token 损失之和 0.9；第 2 个监督 7 个 token，损失之和 0.7。
(a) 按 token 加权的 loss 是多少？
(b) "平均的平均"是多少？
(c) 在 `Trainer` 的做法里，每个 microbatch 反向时用的 loss 分别是多少？它们的和是多少？

**21-9.** 一行 labels 是 `[-100, -100, -100, 12, 7, 151645, -100, -100]`，attention 是 `[1,1,1,1,1,1,0,0]`。
(a) 按 `labels[1:]` 数，监督 token 有几个？
(b) 如果把 labels 错误地设成和 input_ids 完全相同（pad 位置仍为 −100），监督 token 变成几个？其中几个属于提示？
(c) 这一行的 EOS 在哪个位置？它被监督了吗？

**21-10.** 主实验 train 接受 2023 条，有效 batch 8。
(a) 第 400 步和第 800 步各相当于多少轮？
(b) 第 20 章选中配置是 4049 条、400 步、有效 batch 8，相当于多少轮？
(c) 两章选中点"见过的样本次数"一样吗？"每条样本平均被看的次数"一样吗？

**21-11.** 第 400 步 val 上：SFT 和照抄都对 66 条，只有 SFT 对 65 条，只有照抄对 7 条。
(a) 求 SFT 和照抄各自的 EM_norm 成功数。
(b) 验证两者之差等于"只有 SFT 对"减"只有照抄对"。
(c) 照抄在无错的 59 条上全对。已知"只有照抄对"的 7 条恰好都是无错样本，SFT 的改坏数是多少？

**21-12.** LoRA 参数、梯度和 AdamW 两个动量都用 FP32 存。r = 32、q、v 时这四样一共占多少字节？换算成 MiB（1 MiB = 2²⁰ 字节）。和主实验 PyTorch 峰值 allocated 3,608,087,552 字节比，它占多大比例？峰值的大头可能是什么？

---

## 三、编程题

**练习 21.1（必做）　把监督合同交给框架，并检查真实 batch。**
实现 `miniocr/train/trl_sft.py` 中 `# >>> 练习 21.1` 的 `dataset_rows` 和 `audit_batch`：
- `dataset_rows` 把 `SFTExample` 转成 `{input_ids, labels}` 记录；复用 `collate_sft` 的合同检查，空输入或全 −100 必须报 `ValueError`。
- `audit_batch` 检查 pad 不监督、监督位置 label 等于 id、答案起点等于 `prompt_length`、整行 labels 与原编码一致、最后一个真实 token 是 EOS 且被监督、答案中间没有洞；返回逐 token 记录和 `labels[:,1:]` 上的监督数。

验证：`test_ch21_invalid_labels`、`test_ch21_collation_masks_eos_and_padding`；有本地 Qwen3 权重时再看 `test_ch21_real_tokenizer_contract`。

**练习 21.2（必做）　对拍 loss、梯度和一次 AdamW 更新。**
实现 `# >>> 练习 21.2` 的 `loss_step_parity`：两侧都用整个更新的监督 token 总数作分母；比较 loss、每个可训练参数的梯度、裁剪后一步 AdamW 的结果；要求参数确实改变；结束时无论成败都恢复参数。

验证：`test_ch21_fused_loss_gradient_and_adam_parity`（它还检查函数返回后 state_dict 逐位不变）；`python -m book.part3_post_training.ch21_trl_sft.tour 1 --device cpu`（lab 第 1 部分）。注意：CPU 上走的是 transformers 原生 loss，不是融合 loss。

**练习 21.3（必做）　训练一步、保存、重载。**
读懂 `test_ch21_trainer_update_and_adapter_reload`，回答：它怎样确认"冻结参数没动、可训练参数动了"？怎样确认重载的 adapter 与训练中的模型输出相同？然后在 CPU 上用 lab 第 0 节的小规模命令（`tour 2 3 4 6 --n 20 --train-n 32 --steps 2`，需要本地 Qwen3 权重）跑一遍，确认产出 `reload-verified.json`。

**练习 21.4（选做）　对已有逐条输出重新计数。**
不加载任何模型，只读 `outputs/book/ch21/gpu/reloaded-val-predictions.json`，用 `miniocr.eval.postcorrect_eval` 的规范化比较：
(a) 统计 SFT 与照抄的配对表（都对 / 只 SFT 对 / 只照抄对 / 都错）；
(b) 找出"原始输出对、抽取后错"的样本，统计其中目标以 `.`、`,`、`;` 结尾的有多少。
结果应与正文 21.8.3 节一致（66 / 65 / 7；89 条中 88 条）。

**练习 21.5（选做）　让检查真的能抓住错误。**
给 `audit_batch` 写三个新的"故意弄坏"用例（可加在本章测试文件的副本里）：(a) 整行 labels 左移一位；(b) 删掉 EOS 的监督；(c) 只有 `input_ids`、没有 labels（labels 等于 input_ids）。每个用例都要求抛出 `AssertionError`。用 `tiny_fixture()` 构造数据。

**练习 21.6（选做，需 GPU）　把"框架 vs 自写"比干净。**
设计一个只改变训练框架的对照：固定 train 条数、步数、rank、目标模块、学习率、数据顺序和选择方式，一边用第 20 章的自写循环，一边用本章的 `make_trainer`。写出你要固定的每一项和检查方法（例如怎样让两边的数据顺序相同）。跑出来的数字按"本项目观察"标注设备和规模。

---

## 四、调试题

**21-13.** 同学 A 换了一个数据格式：给 TRL 传了只有 `input_ids` 的预编码数据，保留 `skip_prepare_dataset=True` 和 `completion_only_loss=True`。训练第 1 步的 loss 比本章低很多，val EM 却很差。最可能的原因是什么？用哪一项检查能在训练前发现？

**21-14.** 同学 B 打印的可训练参数是 596,049,920 + 4,587,520，显存比预期高很多。可能出了什么错？应该看哪两样东西？

**21-15.** 同学 C 自定义了 `compute_loss`：`loss = cross_entropy(logits, labels, ignore_index=-100)`（默认 `reduction='mean'`），忽略了 `num_items_in_batch`。累积 4 步、每个 microbatch 监督数不同。训练能跑，但日志里的 loss 大约是另一位同学同样设置下的 4 倍。解释两个问题：为什么 token 的权重不对？为什么日志 loss 会放大约 4 倍？用哪一项检查能发现？

**21-16.** 同学 D 把 adapter 拷到另一台机器，重载后 val EM_norm 从 0.655 变成 0.650。他检查了 adapter 文件的哈希，完全一致。列出三个可能的原因，并说明各自去哪个文件里查。

**21-17.** 同学 E 看到 test 是 0.615、低于 val 的 0.655，于是用第 200 步的 adapter 又评了一次 test，得到 0.630，准备在报告里写 0.630。指出问题，并说明本章代码会怎样阻止这件事。

**21-18.** 同学 F 的训练日志里 loss 在第 800 步还在降，他据此把步数加到 1600。参照本章 21.9 节，他应该先看哪两个量？如果它们和本章的形状一样，他的改动大概率会带来什么？

---

## 五、闭卷手写

**21-19.** 默写本章检查清单的七项，每项写一句"查什么"和一句"不查会出什么事"。

**21-20.** 写出跨 microbatch 按 token 加权的 loss 公式，并说明 `num_items_in_batch` 是怎样数出来的（为什么从 `labels[..., 1:]` 数）。

**21-21.** 画出一条单轮 SFT 样本经 Qwen3 非思考模板后的 token 布局示意（system、user、assistant 开头、空思考块、答案、EOS、pad），在每一段下面标出 label 是 −100 还是 token id，attention 是 0 还是 1。

**21-22.** 写出 LoRA 可训练参数个数的公式，并解释为什么它与 α 无关。

---

## 答案与提示

**21-1.** 顺序：已有 `labels` 列 → 原样使用；否则在数据准备阶段由 `completion_mask`（仅当 `completion_only_loss=True`）和 `assistant_masks` 构造；再否则 labels = `input_ids`。`skip_prepare_dataset=True` 会跳过整个准备阶段（也不截断）。本章数据带 `labels` 列又跳过准备，所以走第一条，开关不起作用；监督范围完全由第 19 章 `encode_sft` 决定。

**21-2.** 因为模型看到的是 collator 的输出。中间可能发生：补 pad（以及 pad 位置的 label 怎样设）、截断、重新套模板、packing 拼接、删除未用列（`remove_unused_columns` 可能删掉 `labels`）。

**21-3.** (a) 错。个数对，名字和位置可能错（例如挂到了别的同形状模块），还要看名字；也要确认冻结参数的值没动。(b) 错。CPU 上用的是 transformers 原生 loss 的子类，GPU 上是 TRL 的融合 LM head，两个实现不同。(c) 错。汇总相同不代表逐条相同，要逐条比对（本章写 `predictions_identical`）。(d) 错。日志 loss 是不同随机 batch 上 10 步的平均，差 0.0002 主要是 batch 噪声；判断过拟合要看固定 train 探针与 val。

**21-4.** Adam 第一步每个分量的更新约为 lr × g/(|g| + eps)。g 接近 0 的分量上，两种实现的微小舍入差可能让比值从 +1 翻到 −1。eps 取大能压住这些分量。两侧用同一个 eps，比的仍是"同一个优化器面对两份梯度"，所以是公平的；但它只用于对拍，正式训练按 `sft-config.json` 用 1e-8。

**21-5.** 不同处：训练框架与 loss 实现；rank 和目标模块（32/q,v 对 64/q,k,v,o）；可训练参数（4.59M 对 18.35M）；train 条数（2023 对 4049）；选中点的轮数（1.58 对 0.79）；选择对象（4 个 checkpoint 对 6 个配置）；数据顺序；梯度检查点。多个变量同时变，任何差别都无法归因到"框架"。

**21-6.** 训练配置（含默认值）、生成配置（贪心、模板、左 padding、上限、批大小）、数据身份（版本、划分、seed、ID 哈希）、基座快照（commit 哈希）、库版本、代码哈希、选择规则与选中步、精度口径。

**21-7.** (a) 28 × 32 × [(1024+2048) + (1024+1024)] = 28 × 32 × 5120 = 4,587,520。(b) 每层 16 × [3072 + 2048 + 2048 + 3072] = 16 × 10240 = 163,840，28 层 4,587,520，和 (a) 相同。(c) (a) 28 × 2 × 2 = 112；(b) 28 × 4 × 2 = 224。(d) 基座 600,637,440 − 4,587,520 = 596,049,920；比例 4,587,520 / 596,049,920 ≈ 0.77%（全部数学结论）。

**21-8.** (a) (0.9 + 0.7)/10 = 0.16。(b) (0.3 + 0.1)/2 = 0.20。(c) 0.9/10 = 0.09 和 0.7/10 = 0.07，和为 0.16，与 (a) 相同。

**21-9.** (a) `labels[1:]` 中非 −100 的是 12、7、151645，共 3 个。(b) 位置 1–5 都被监督，共 5 个，其中位置 1、2 属于提示，2 个。（位置 0 不计，因为没有 logits 预测它。）(c) 位置 5（id 151645，即 `<|im_end|>`），被监督了。

**21-10.** (a) 400 × 8 / 2023 ≈ 1.58；800 × 8 / 2023 ≈ 3.16。(b) 3200 / 4049 ≈ 0.79。(c) 名义上一样（都是 400 × 8 = 3200），实际几乎一样但不完全相同：本章 `dataloader_drop_last=false`，每轮最后一个 microbatch 只有 1 条，`Trainer` 按轮划分更新，第 400 步实际处理 2023 + 147 × 8 = 3199 次；第 20 章不到一轮，正好 3200 次。每条平均被看的次数本章是第 20 章的两倍左右。

**21-11.** (a) SFT 66 + 65 = 131；照抄 66 + 7 = 73。(b) 131 − 73 = 58 = 65 − 7。(c) 无错样本里照抄全对；SFT 在其中错的就是"只照抄对"中的无错样本，即 7 条，改坏 7/59，与 `stages.json` 一致。

**21-12.** 4,587,520 × 4 样 × 4 字节 = 73,400,320 字节 = 70 MiB（数学结论）。占峰值约 2%。大头是 FP32 冻结基座（596,049,920 × 4 字节 ≈ 2.22 GiB，数学结论）和激活（梯度检查点已省掉一部分）；这是对第 20 章 20.8 节结论的又一次印证，具体拆分本章未测。

**21-13.** 只有 `input_ids` 且跳过准备时，collator 让 labels = input_ids，整条序列（含提示模板）都被监督；提示几乎是固定文本，很好预测，所以 loss 一开始就低，答案部分的权重被稀释。TRL 1.15 只在数据带 mask 列却没有 labels 时报错，这种情况不报。用检查一/二：第一批中提示位置的 label 不是 −100，`active[0]` 不等于提示长度，监督数远大于答案长度。

**21-14.** 基座没有冻住（或者打印的模型和训练的不是同一个），实际在做全参训练。看 `trainer.model` 的可训练参数个数和名字：名字里应全部含 `lora_`，个数应等于公式值；再确认训练一步后冻结参数逐位不变。

**21-15.** 权重：每个 microbatch 各自除以自己的监督数，短 microbatch 里每个 token 的权重更大，变成"平均的平均"，不是 token 加权。放大：transformers 5.19 的 `training_step` 在"模型接受 loss kwargs 且 `num_items_in_batch` 不为空"时，认为 `compute_loss` 已按总数缩放，不再除以累积步数（`transformers/trainer.py` 的 `loss_is_scaled_for_ga` 判断，Qwen3 属于这种情况）。于是 4 个均值直接相加，日志 loss 和梯度都约为正确值的 4 倍。梯度裁剪和 Adam 对整体尺度不敏感，训练可能看起来正常，权重错误却一直在。检查四（两个监督数不同的 microbatch 对拍 loss 和梯度）能发现；也可以把日志 loss 和固定探针的 `mean_loss` 对比。

**21-16.** 可能原因：(1) 基座快照不同（查 `environment-*.json` 的 `base_snapshot`）；(2) 生成参数不同，例如生成批大小变了导致左 padding 不同（查 `environment-*.json` 的 `generation`，重载代码会检查）；(3) 库版本或 GPU 不同，bf16 下个别样本分叉（查 `versions`、`device`）。另外也要查数据身份（`data-manifest.json` 的 val 哈希）。先做逐条比对，找出变了的那几条。

**21-17.** 用 test 挑 checkpoint，test 分数就不再是无偏估计；报告必须报冻结设置的那一次（0.615），并写明。本章代码：`FINAL_TEST_STARTED.json` 独占创建后，同一目录再评 test 直接报错；test 只评 `selection.json` 选中的步，不存逐条输出。如果真的根据 test 改了设置，按课程规则要在报告里写明，并补一份新的留出测试集。

**21-18.** 先看固定 train 探针 NLL 和 val 指标（val NLL、val EM）。如果和本章一样是"train 探针在降、val NLL 回升、val EM 不动"，加步数只会让模型更熟悉训练集，val 大概率不会改善，还多花一倍算力。

**21-19.** 参见正文 21.1.3 节表格及各节"会出什么错"。

**21-20.** loss = Σ_k S_k / Σ_k n_k，S_k 是第 k 个 microbatch 被监督 token 的负对数似然之和，n_k 是其监督 token 数。`Trainer` 在一次更新前取出全部 microbatch，按 `labels[..., 1:] != -100` 数出总数 N 传给每次 `compute_loss`；从位置 1 数，是因为因果对齐下位置 t 的 logits 预测位置 t+1，位置 0 的 label 不是任何预测的目标。

**21-21.** system、user、`<|im_start|>assistant\n`、`<think>\n\n</think>\n\n`：label −100，attention 1；答案各 token：label = 自身 id，attention 1；`<|im_end|>`：label 151645，attention 1；pad `<|endoftext|>`：label −100，attention 0（训练时右 padding）。可对照正文 21.3.3 节的表。

**21-22.** 参数个数 = Σ r × (in + out)，对每个挂 LoRA 的线性层求和（A 是 r × in，B 是 out × r）。α 只出现在缩放 α/r 里，是一个常数，不是参数，所以不影响个数。
