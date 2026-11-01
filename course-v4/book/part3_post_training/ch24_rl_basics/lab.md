# 第 24 章实验手册：强化学习基础

## 0. 准备

在 `course-v4/` 下运行。只需现有环境，不下载模型或数据，不安装新包：

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q book/part3_post_training/ch24_rl_basics
CUDA_VISIBLE_DEVICES="" python -m book.part3_post_training.ch24_rl_basics.tour
# 单独运行 KL 和 PPO；位置参数保留顺序。
python -m book.part3_post_training.ch24_rl_basics.tour 3 4 --device cpu
# 可选小 GPT；默认验证 n=20，不读 test。
MPLCONFIGDIR=/tmp/ch24-mpl python -m book.part3_post_training.ch24_rl_basics.tour 5 --out outputs/book/ch24/cpu
```

部分号 **1–5** 与下文及 `tour.PARTS` 对应；无部分号时只跑 1–4。`--device cpu|mps|cuda` 默认 cpu，显式指定不可用设备会报错。每部分重置种子。CPU 两线程，所有实验均为 **fp32**，没有自动混合精度。第 5 部分是随机初始化的课程 GPT，不是预训练 GPT-2/Qwen，不需要大权重；因此本章没有缺权重才 skip 的测试。

参数为实验配置，不是性能承诺：

| 参数 | 默认值 | 控制范围 |
|---|---:|---|
| `--seed` | 0 | 所有部分的随机起点 |
| `--seeds` | 8 | 第 1 部分独立种子数 |
| `--steps` / `--lr` | 200 / 0.1 | 第 1 部分 SGD 更新数和学习率 |
| `--batch-size` | 16 | 第 1 部分每种子每步采样数；第 5 部分训练与推理批量 |
| `--repeats` / `--mc-batch` | 1000 / 32 | 第 2、3 部分独立批次个数 / 每批样本数 |
| `--n` | 20 | 第 5 部分 val/test 提示前缀数；不改变划分 |
| `--seq-steps` / `--seq-lr` | 20 / 0.001 | 第 5 部分 AdamW 更新数和学习率 |
| `--width` / `--layers` | 64 / 2 | 第 5 部分 GPT 宽度 / 层数 |
| `--response-length` | 8 | 固定生成长度；至少两个 token，无 EOS/早停 |
| `--log-every` | 50 | 第 1、5 部分记录间隔；末步总会记录 |
| `--out` | 不写文件 | JSON、JSONL、CSV、PNG；第 5 部分另存 final.pt |
| `--final-test` | 关闭 | 仅在冻结配置后，显式部分 5 末尾评一次 test |

推荐的 GPU 配置就是本次实跑配置。租用实例上先 `nvidia-smi` 检查空闲；在代码快照目录运行，并将全部写入路径放到章节独占目录。下文 `$RUN_DIR` 是本章运行目录的绝对路径，`$LOG_DIR` 是日志目录的绝对路径；运行前设置这两个变量、创建目录，并将代码快照放在 `$RUN_DIR/source`：

```bash
# 进入代码快照目录。
cd "$RUN_DIR/source"
export PYTHONPYCACHEPREFIX="$RUN_DIR/pycache"
export MPLCONFIGDIR="$RUN_DIR/mpl"
export XDG_CACHE_HOME="$RUN_DIR/cache"
# 每次复跑换一个 out 和日志名，避免覆盖证据。
nohup python -u -m book.part3_post_training.ch24_rl_basics.tour 5 \
  --device cuda --width 128 --layers 4 --batch-size 128 \
  --seq-steps 30 --log-every 10 --n 256 \
  --out "$RUN_DIR/smoke" > "$LOG_DIR/ch24_smoke.log" 2>&1 &
# 确认上述任务完成、loss/梯度有限、mask 正确，再执行：
nohup python -u -m book.part3_post_training.ch24_rl_basics.tour 1 2 3 4 \
  --device cuda --seeds 32 --steps 2000 --batch-size 64 \
  --repeats 20000 --mc-batch 64 --log-every 50 \
  --out "$RUN_DIR/mechanisms" > "$LOG_DIR/ch24_mechanisms.log" 2>&1 &
nohup python -u -m book.part3_post_training.ch24_rl_basics.tour 5 \
  --device cuda --width 128 --layers 4 --batch-size 128 \
  --seq-steps 300 --log-every 25 --n 256 --final-test \
  --out "$RUN_DIR/sequence" > "$LOG_DIR/ch24_sequence.log" 2>&1 &
```

本次远程环境：Python 3.11.10、torch 2.14.1+cu130、matplotlib 3.11.2；系统已有 transformers 5.19.0、peft 0.21.2、trl 1.15.0、accelerate 1.15.0，本章不调用它们。**没有 pip 安装、没有模型/数据下载、没有图片渲染**。环境与显存检查保存在 [environment.txt](../../../outputs/book/ch24/gpu/environment.txt) 和 [preflight.txt](../../../outputs/book/ch24/gpu/preflight.txt)。

口径：stdout 每行一个 JSON 对象，含来源、设备、精度与事件；start 保存完整参数、UTC 时间和 torch/GPU 版本。`part_end` 计时包括该部分采样、训练、评测、绘图与写盘，并在结束前同步 CUDA；`end.seconds` 是任务内墙钟时长，不是 CUDA kernel profiler 时间。显存是本进程 PyTorch 峰值 allocated/reserved，不含其他进程与 CUDA 上下文。CSV 中 `exact_*` 是按已知分布求和的**数学结论的数值计算**，`mc_*`/`empirical_*` 是**本项目观察**；浮点残差不等于统计偏差。

实现沿用第 18、23 章的部分号、设备参数、小规模默认值和结构化日志约定。

## 1. 多臂老虎机：学习曲线

四臂奖励为均值 `[1, 1.2, 1.5, 2]`、标准差 1 的独立高斯变量。策略是四个 logits 的 softmax。从均匀策略开始，比较无基线、当前策略解析期望奖励（oracle）、历史奖励 EMA 三种 REINFORCE。

同一种子三种方法复用动作抽样的均匀数和奖励噪声；策略不同，因此动作不必相同。EMA 在当前梯度计算**之后**更新，系数为 0.9。oracle 使用已知环境均值，是机制实验的控制组，不是要求真实任务提供正确答案。每个种子分别优化，损失缩放保证种子数增加不会改变每个策略的学习率。

记录每种子的解析期望奖励及最优臂概率；图中阴影是**跨种子标准差，不是置信区间**。随机单批奖励不会用来代替真正的策略期望。

## 2. 固定策略：直接量梯度方差

固定 `p=[0.1,0.2,0.3,0.4]`，奖励分布沿用第 1 部分。对相同动作和噪声比较 `b=0` 与 `b=E[R]`。

**数学结论：** softmax logits 的单样本上升梯度是 `(R-b)(one_hot(a)-p)`。动作无关基线的期望梯度为 `p*(mu-E[R])`。方差口径是批平均梯度协方差矩阵的迹，即各 logits 分量方差之和；独立批大小为 B 时，单样本协方差的迹除以 B。代码用高斯奖励的二阶矩精确计算它，同时采样重复批次核对。

输出 `gradient_mean`、`exact_gradient`、最大均值误差及均值标准误，避免只报告方差下降却不检查期望。detach 只截断反传；若基线包含本条动作的奖励，仍可能有统计偏差。任意基线并不必然降低方差，`E[R]` 也不总是方差最优常数；本实验只验证指定策略和奖励分布。

## 3. KL 的 k1、k2、k3

固定样本来自 p，方向为 **KL(p || q)**，定义 `log_r=log(q)-log(p)`：

- k1 = `-log_r`；
- k2 = `0.5*log_r**2`；
- k3 = `expm1(log_r)-log_r`。

**数学结论：** p、q 具有相同的正支持时，k1 与 k3 无偏，k2 一般有偏；k3 逐样本非负。用 near `q=[0.11,0.19,0.29,0.41]` 和 far `q=[0.4,0.3,0.2,0.1]` 展示局部近似与远距离的区别。

每种分布先精确枚举真 KL、估计器期望及批均值方差，再做 Monte Carlo；三种估计器共享样本。`mc_batch_variance` 是独立批次均值之间的方差，不是所有样本均值的标准误。总体 MC 均值标准误另列。这里比较估计器数值，不把直接反传 k3 当作完整 PPO/GRPO 的 KL 梯度实现。

## 4. PPO：裁剪只阻止有利方向的过度更新

计算 `min(r*A, clip(r,1-eps,1+eps)*A)`，`eps=0.2`，正负优势分别测试比率 0.5、0.9、1.0、1.1、1.5，并通过 autograd 输出对 **ratio** 的导数。

**数学结论：** A>0 且 r>1.2，或 A<0 且 r<0.8，目标在该样本上饱和，梯度为零。A>0 且 r<0.8、A<0 且 r>1.2 时仍有梯度。避开分段边界的不可导点。这是单样本 surrogate 演示；PPO 不保证整体策略 KL 被硬性限制，多个样本共享参数也可能改变已饱和样本的概率。

## 5. 可选：小 GPT 的序列级 REINFORCE

复用 `miniocr.models.gpt.GPT`，随机初始化，全参数训练。词表 `$xyz+0123`；提示是全部非 `$` 符号组成的四 token 串。按提示 SHA256 固定分为 train/val/test（本次为 3280/431/385 个，无交集）；每条生成固定 8 token。首尾是 `$` 各奖励 0.5，在回合末一次性发放奖励。

这是为了精确控制格式奖励与其漏洞的**合成符号任务**，没有使用开源公式图像，也没有模拟 OCR。已检查第 18 章 `miniocr/data/postcorrect.py` 和 `miniocr/eval/postcorrect_eval.py`，本章不需要加载后纠错数据；没有修改它们的接口或 latex-v1 划分。实际后纠错实验仍应使用这些共用模块。本章不能作为 R2 的后纠错提升结果。

采样使用当前策略、temperature=1、无 top-k/top-p、无 dropout；一次采样对应一次更新。训练时重算整条序列，next-token 对齐后只对响应位置的 log 概率**求和**，乘 detached 终局 advantage；提示不计 loss。本任务无 padding/EOS，通用 `sequence_log_prob` 的测试另覆盖 prompt、padding=-100 和有效终止 token 的 mask。基线为之前训练批次的 EMA；不使用当前 batch 自身均值。优化器 AdamW，无 weight decay，梯度范数上限 1。

30 步冒烟先检查 loss/梯度有限、全部参数参与反传、响应 mask=8、提示 mask=0；正式配置放大到 300 步，固定最后一步为最终模型。所有中途评测只看 val；`--final-test` 只在正式训练结束时评一次 test，不输出 test 逐条错例，不根据 test 修改设置。

额外记录整个响应中 `$` 的占比。这是预先定义的奖励漏洞探针：输出 `$$$$$$$$` 能得到满分，但没有公式内容。训练结果确实走向这一漏洞，因此格式成功不能写成任务准确率或模型数学能力。Wilson 区间描述本受控任务的成功比例；不是 OCR 泛化区间。

## 运行前的预测题

1. 三种基线最终都能选择最优臂时，带基线方法是否一定有更高的最终奖励？应该在哪个实验直接验证方差？
2. 从 reward 减去不依赖当前动作的常数，期望梯度为何不变？用当前 batch 均值代替历史均值有什么区别？
3. 哪一种 KL 估计器可能给出负的单样本值？哪一种在 far 分布仍然有偏？
4. A=-1、r=1.5 时，PPO 梯度是否为零？r=0.5 呢？
5. 只奖励首尾 `$` 时，模型是否必须读懂提示？若全部输出都是 `$`，应如何评价“成功率 100%”？
6. 如果把序列 log 概率之和改成平均，在响应长度可变时会改变什么目标？

## 怎样读结果与实测记录

下表全部为 **本项目观察，RTX 4090（租用），fp32**；样本数、步数见各行。CPU/MPS/5080 列留空，未用 CPU 冒烟结果代填不同配置。来源为 [GPU 日志与指标目录](../../../outputs/book/ch24/gpu/)，每个原始数字可由对应 CSV/JSON 追溯。

| 实验及规模 | RTX 4090（租用）实测 | CPU | MPS | 5080 |
|---|---|---|---|---|
| 1：32 种子、各 2000 步、batch=64，无基线 | 最终期望奖励 1.995937，种子 SD 0.000286；最优臂概率 0.994253 | | | |
| 1：同上，oracle | 1.995930，SD 0.000207；最优臂概率 0.994236 | | | |
| 1：同上，历史 EMA | 1.995934，SD 0.000208；最优臂概率 0.994244 | | | |
| 2：20000 独立批次，各 64 样本，无训练 | 梯度方差迹 0.0363291 → 0.0120059（无基线 → oracle），比值 0.330477 | | | |
| 3：near，20000×64 样本，无训练 | k1/k2/k3 MC 均值：0.00100604 / 0.00101052 / 0.00101991 | | | |
| 3：far，同上 | k1/k2/k3 MC 均值：0.456078 / 0.521550 / 0.456403 | | | |
| 4：两种优势×五个比率，无训练 | A=+1,r=1.5 导数 0；A=-1,r=0.5 导数 0；A=-1,r=1.5 导数 -1 | | | |
| 5：冒烟，30 步、batch=128、val n=256 | val 奖励 0.0859375 → 0.998047，双端成功率 0 → 0.996094 | | | |
| 5：正式，300 步、batch=128、val n=256 | val 奖励 0.0859375 → 1；双端成功率 0 → 1；最终 `$` 占比 1 | | | |
| 5：最终 test，仅一次，n=256 | 奖励 1，成功率 1，Wilson 95% 区间 [0.985216,1]；`$` 占比 0.999512 | | | |
| 1–4 总用时 / 峰值 allocated | 5.950931 秒 / 74.956MiB | | | |
| 5 冒烟 / 正式用时 | 6.687251 / 15.873527 秒；两次峰值 allocated 均 92.396MiB | | | |

学习曲线几乎重合，不支持“加基线一定提高最终奖励”。第 2 部分在固定策略上测得方差下降约 66.95%，才是本配置降方差的直接证据；不能外推到任意奖励分布。

KL 详表（本项目观察，RTX 4090（租用），fp32，20000 批×64 样本）：

| 分布 / 估计器 | MC 偏差 | 批均值方差 | MC 总均值标准误 |
|---|---:|---:|---:|
| near / k1 | -1.50205e-5 | 3.12866e-5 | 3.95516e-5 |
| near / k2 | -1.05432e-5 | 2.35008e-8 | 1.08399e-6 |
| near / k3 | -1.14914e-6 | 2.51033e-8 | 1.12034e-6 |
| far / k1 | -3.56495e-4 | 0.0128766 | 0.000802390 |
| far / k2 | +0.0651152 | 0.00299766 | 0.000387147 |
| far / k3 | -3.19183e-5 | 0.00334951 | 0.000409238 |

数学结论的 fp32 精确枚举：near/far 真 KL 为 0.001021064 / 0.456434846；far 的 k2 偏差为 +0.06511867。k1/k3 的非零 MC 偏差应和标准误一起读；k2 的偏差不随增加样本而消失。near k3 精确偏差约 -2.79e-9 是浮点残差。

产物索引：

- [机制实验日志](../../../outputs/book/ch24/gpu/ch24_mechanisms.log)、[指标 JSON](../../../outputs/book/ch24/gpu/mechanisms/metrics.json)、[种子级学习曲线 CSV](../../../outputs/book/ch24/gpu/mechanisms/bandit_curves.csv)、[学习曲线 PNG](../../../outputs/book/ch24/gpu/mechanisms/bandit_curves.png)。
- [梯度方差 CSV](../../../outputs/book/ch24/gpu/mechanisms/gradient_variance.csv)、[KL CSV](../../../outputs/book/ch24/gpu/mechanisms/kl.csv)、[PPO CSV](../../../outputs/book/ch24/gpu/mechanisms/ppo.csv)。
- [冒烟日志](../../../outputs/book/ch24/gpu/ch24_smoke.log)、[正式序列日志](../../../outputs/book/ch24/gpu/ch24_sequence.log)、[序列曲线 CSV](../../../outputs/book/ch24/gpu/sequence/sequence_curve.csv)、[曲线 PNG](../../../outputs/book/ch24/gpu/sequence/sequence_curve.png)。
- [本章正文](chapter.md)、[本地全量测试日志](../../../outputs/book/ch24/gpu/local_pytest.log)、[执行代码哈希](../../../outputs/book/ch24/gpu/source.sha256)。检查点保留在远程 `$RUN_DIR/{smoke,sequence}/final.pt`。

机制代码：[policy_gradient.py](../../../miniocr/train/policy_gradient.py)，保留练习 24.1–24.4 标记；[CPU 测试](test_ch24_policy_gradient.py) 验证精确期望、KL 方向与偏差、PPO 两侧分支、mask 和采样/重算一致性。已有数据与评测接口未改动。
