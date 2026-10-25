# A5：对齐——策略梯度、DPO、GRPO，以及"学了新的、忘了旧的"

| | |
|---|---|
| 位置 | 第 32 章之后 |
| 预计用时 | 30–40 小时 |
| 算力 | 写代码、跑测试：CPU。实验：采样很慢，强烈建议 GPU；CPU 可以做小配置（`--smoke` 和几十步的短跑）。排行榜：一张租用的 24 GB 显卡，A5 的训练时间不超过 20 分钟（CPU 赛道 90 分钟） |
| 交付 | ① `a5/` 里的代码，`tests/adapters.py` 接好后全部测试通过；② 书面报告 `writeup.md`（从 `writeup_template.md` 复制）；③ 排行榜记录 |
| 满分 | 100 分：代码 52 分，书面 10 分，实验 32 分，排行榜 6 分 |

A4 用 SFT 教会了模型读文字行。SFT 只能模仿给定的答案；这个作业让模型**从自己的回答里学**：

1. **策略梯度的零件**（第 29 章）：组内标准化和留一的优势、"白算的组"、KL 的三个估计量、重要性比值的裁剪；
2. **DPO**（第 30 章）：KL 约束下的最优策略、DPO 损失和它的梯度系数、从组内采样挑成对数据、在正确描述上制造一处幻觉；
3. **GRPO 与 DAPO**（第 31 章）：三种汇总方式与长度偏差、GRPO 损失、clip-higher、动态采样、软超长惩罚、带格式闸门的可验证奖励；
4. **采样**（第 29、31 章）：带温度和 KV cache 的采样、每题 G 个回答、把回答整理成 (ids, mask)、整条序列的对数概率；
5. **遗忘与配比**（第 32 章）：帕累托前沿、重放、拒绝采样微调（RFT）的挑选、探测 KL。

最后你会从**同一个 A4 检查点**出发，用 SFT、RFT、DPO、GRPO 四种方法继续训练，在"新任务学得多好"和"旧任务忘了多少"两个轴上比较它们。

---

## 1. 规则

### 1.1 能用什么、不能用什么

A1–A4 的规则照样适用（`torch.nn.functional`、现成的层和损失、`torchvision`、`scipy.stats` 等都不能用）。这个作业另外：

| 不能用 | 自己写 |
|---|---|
| `trl`、`openrlhf`、`verl` 等 RL/偏好训练库 | GRPO、DPO、RFT 的训练循环 |
| `torch.distributions`（`Categorical`、`kl_divergence`） | 采样之前的 softmax、KL 的估计 |
| `F.logsigmoid`、`torch.nn.functional.kl_div` | 数值稳定的 log σ、KL |
| HF 的 `generate` | 带 KV cache 的采样 |

**可以用**：`torch.multinomial`（规定用它采样，见 5.1）、`torch.Generator`、numpy。

**可以把你自己 A1–A4 的代码复制进 `a5/`**，不能 import 前面的作业。

`miniocr` 里能用的仍然只有**数据生成器**：A3、A4 用过的那些，加上 `miniocr.data.shapes` 的 `COLORS`、`SHAPES`、`SIZES`、`describe`（第 2 部分要用）。

### 1.2 AI 助手和参考实现

与 A1 相同。

### 1.3 前提

你需要 A4 的检查点（学过 read 的 VLM）。没有的话可以用 A3 的检查点，把目标任务换成 A3 的任务（例如 `count`、`ground`），实验的结论照样成立，只是第 9 节的表要相应改。完全从随机初始化开始做 RL 是不行的：模型从来不写 eos，所有回答都被截断，奖励全是 0（第 29 章 29.5 节）。

---

## 2. 准备

```
assignments/a5_align/
├── handout.md
├── a5/                   你的代码
├── tests/
│   ├── adapters.py       ← 你要填的接线
│   └── test_a5_*.py
├── leaderboard.py        排行榜（复用 A3、A4 的测试集）
└── writeup_template.md
```

```bash
python -m pytest -q assignments/a5_align/tests
```

第一到三部分、第五部分只是对张量和列表的计算，互不依赖，可以先做；第四部分要用你 A4 的 VLM。

**记号**：一批是 N 道题、每题 G 个回答，奖励 `rewards` 是 (N·G,) 的张量，**每连续 G 个是同一道题的回答**。逐 token 的量（logp、mask 等）都是 (B, T−1)，第 t 列对应"预测第 t+1 个 token"（与 A4 的约定二相同）。

---

## 3. 第一部分：策略梯度的零件

### 3.1 问题（advantages）：两种优势和白算的组（3 分）

1. `group_advantages(rewards, G, std=True, eps=1e-6)`：每组减去组内均值；`std=True` 时再除以（**无偏**）组内标准差 + eps。全相同的组得到 0。
2. `loo_advantages(rewards, G)`：留一（RLOO）：Aᵢ = rᵢ − 其余 G − 1 个的平均。
3. `zero_variance_fraction(rewards, G)`：组内最大与最小相差 < 1e-8 的组占多少（返回 float）。

为什么组内标准化的优势在一组只有一个对的时候特别大？这是好事还是坏事（第 29 章 29.5 节）？

### 3.2 问题（kl_estimate）：KL 的三个估计量（3 分）

`kl_estimate(logp, ref_logp, kind)`：样本来自当前策略 π，记 log r = ref_logp − logp，

- k1 = −log r（无偏，可能为负，方差大）；
- k2 = (log r)² / 2（有偏，总 ≥ 0）；
- k3 = (r − 1) − log r（无偏，总 ≥ 0）。

逐元素返回，形状与输入相同。测试会从一个已知分布采 20 万个样本，检查 k1、k3 的平均等于真实的 KL、k3 没有负数、方差比 k1 小。

### 3.3 问题（clipped_objective）：裁剪的目标（4 分）

`clipped_objective(logp, old_logp, adv, eps_low, eps_high=None) → (目标, 是否被裁剪)`：

ρ = exp(logp − old_logp)，目标 = min(ρA, clip(ρ, 1 − ε_low, 1 + ε_high)·A)；`eps_high` 为 None 时等于 `eps_low`。"被裁剪"定义为**裁剪项严格小于未裁剪项**（这时这个 token 没有梯度）。adv 可以是和 logp 同形状，也可以是 (B, 1)（广播到每个 token）。

测试有四个手算的例子（ρ = 1.5 或 0.5，A = ±1），并检查被裁剪的 token 梯度恰好为 0、没被裁剪的不为 0。

**交付**：`run_group_advantages`、`run_loo_advantages`、`run_zero_variance_fraction`、`run_kl_estimate`、`run_clipped_objective`；`test_a5_pg.py` 全部通过。

### 问题（kl_by_hand）：手算 KL 的估计量（3 分，书面）

p = (0.5, 0.3, 0.15, 0.05) 是当前策略，q = (0.25, 0.25, 0.25, 0.25) 是参考策略。

1. 精确算 KL(p ‖ q)。
2. 对四个可能的样本，各写出 log r、k1、k2、k3。哪些样本的 k1 是负的？
3. 用 p 加权，算三个估计量的期望和方差。哪个有偏？偏多少？k1 和 k3 的方差哪个小？
4. 在 GRPO 里，KL 项是逐 token 算的，"样本"就是模型自己采到的 token。如果模型几乎是确定的（每步最大概率 0.99），三个估计量的差别还大吗？为什么？

---

## 4. 第二部分：DPO

### 4.1 问题（optimal_policy）：最优策略（2 分）

`optimal_policy(ref_logp, rewards, beta)`：在有限个候选上，max_π E_π[R] − β·KL(π ‖ π_ref) 的解 π* = softmax(log π_ref + R / β)。用你自己的 softmax。测试随机扰动 π*，检查没有一个扰动后的分布目标更高。

### 4.2 问题（dpo_loss）：DPO 损失（4 分）

`dpo_loss(pol_w, pol_l, ref_w, ref_l, beta=0.1, sft_weight=0.0, n_w=None) → (平均损失, 统计)`：四个输入都是 (B,) 的**整条回答的对数概率之和**。

- 隐式奖励 r_w = β(pol_w − ref_w)，r_l = β(pol_l − ref_l)；每条的损失 −log σ(r_w − r_l)，**用你自己的数值稳定的 log σ**（margin 是 ±10⁴ 时也不能出 inf 或 nan）；
- `sft_weight > 0` 时每条再加 sft_weight · (−pol_w / n_w)，n_w 是 chosen 回答的 token 数（缓解似然位移，第 30 章 30.5 节）；
- 对批求平均；
- 统计（不带梯度的 float）：`reward_w`、`reward_l`（批平均）、`margin`（r_w − r_l 的平均）、`acc`（r_w > r_l 的比例）。

测试会用 autograd 检查梯度：∂L/∂pol_w = −β·σ(r_l − r_w) / B。

### 4.3 问题（pairs_from_groups）：从组内采样挑成对数据（2 分）

`pairs_from_groups(rewards, G, min_gap=0.0) → [(chosen 的全局下标, rejected 的全局下标)]`：每组最高的当 chosen、最低的当 rejected（并列时取下标小的，即 `argmax`/`argmin` 的约定），只保留 最高 − 最低 **>** min_gap 的组。

### 4.4 问题（hallucination）：制造一处幻觉（2 分）

`hallucinated_caption(objs, rng, kind) → (rejected 描述, (size, color, shape))`：objs 是 `sample_scene` 返回的物体列表，正确描述是 `describe(objs)`。

1. 候选 = 图里**没有**的 (颜色, 形状) 组合，按 `for c in COLORS for s in SHAPES` 的顺序列出；先 `rng.integers(len(候选))` 抽一个。
2. `"substitute"`：再 `rng.integers(len(objs))` 抽一个物体，把它的颜色和形状换成候选（大小和框不变），然后 `describe`；返回被换掉后的那个物体 (原来的 size, 新 color, 新 shape)。
3. `"add"`：再从 `list(SIZES)` 里抽一个大小，在正确描述末尾接上 `", a {size} {color} {shape}"`。

**随机数的使用顺序是规定死的**，测试按它对照。为什么候选要排除图里已有的组合？

**交付**：`run_optimal_policy`、`run_dpo_loss`、`run_pairs_from_groups`、`run_hallucinated_caption`；`test_a5_dpo.py` 全部通过。

### 问题（dpo_derivation）：推导 DPO（4 分，书面）

1. 从 max_π E_{y∼π}[R(y)] − β·KL(π ‖ π_ref) 出发，用拉格朗日乘子（或者把目标改写成一个 KL），推出 π*(y) = π_ref(y)·exp(R(y)/β) / Z。
2. 把它反解成 R(y) = β log(π*(y)/π_ref(y)) + β log Z，代进 Bradley–Terry 模型，说明为什么 Z 消掉了。没有 Z 意味着什么（我们不需要做什么）？
3. 对 pol_w 求导，写出梯度系数 β·σ(r_l − r_w)。它什么时候大、什么时候接近 0？和 SFT 的梯度（每条系数都是 1）相比，这意味着什么？
4. chosen 和 rejected 有共同的前缀（例如 substitute 只改了一个物体），前缀上的 token 对损失的贡献是多少？为什么？

### 问题（gc_by_hand）：手算梯度的大小（3 分，书面）

1. DPO，β = 0.1：margin（ρ_w − ρ_l，ρ = pol − ref）分别是 −5、0、0.7、5、50 时，梯度系数 β·σ(−β·margin) 各是多少？
2. GRPO 的一组 G = 8，奖励 [1, 0, 0, 1, 1, 0, 0, 0]：写出组内标准化（无偏标准差）和留一的优势。如果这组全是 1 呢？
3. 一批两条回答，长度 3 和 1，每个 token 的损失都乘 1。在 sample、token、constant（max_len = 4）三种汇总方式下，每个 token 分到的权重各是多少？哪种方式让长回答的每个 token 权重更小？

---

## 5. 第三部分：GRPO、DAPO 与奖励

### 5.1 问题（aggregate）：三种汇总方式（3 分）

`aggregate(per_token, mask, mode, max_len=None)`：

- `"sample"`：每条先 Σ / Σmask，再对条平均（原版 GRPO）；
- `"token"`：Σ / Σmask，所有 token 一起平均（DAPO）；
- `"constant"`：Σ / (B · max_len)，max_len 默认是 T（Dr. GRPO）。

测试检查数值，也用 autograd 检查每个 token 分到的权重。

### 5.2 问题（grpo_loss）：GRPO 损失（5 分）

`grpo_loss(logp, old_logp, adv, mask, ref_logp=None, beta=0.0, eps_low=0.2, eps_high=None, mode="token", max_len=None) → (损失, 统计)`：adv 是 (B,)（每条回答一个），

- 每个 token：−裁剪目标 + β·k3（有 ref_logp 且 β > 0 时）；
- 按 mode 汇总；
- 统计（float，只在 mask 为 1 的位置上算）：`clip_frac`（被裁剪的比例）、`kl`（k3 的平均；没有 ref_logp 时为 0）、`ratio_max`（ρ 的最大值）。

第一次更新时 old_logp = logp，ρ 处处为 1，损失是多少？（测试会检查。）

### 5.3 问题（dapo）：超长惩罚与动态采样（2 分）

- `overlong_penalty(length, max_len, cache)`：长度 ≤ max_len − cache 时 0；在 (max_len − cache, max_len] 之间线性地从 0 降到 −1，即 (max_len − cache − length) / cache；超过 max_len 时 −1。
- `keep_groups(rewards, G) → (N,) bool`：组内最大与最小相差 > 1e-8 的组才留下。

### 5.4 问题（rewards）：带格式闸门的可验证奖励（4 分）

十个任务：A3 的九个加 read。奖励都在 [0, 1]。

1. `well_formed(task, pred)`：count、count_color 是数字（`str.isdigit`）；left 是 `COLORS` 里的一个；exists 是 yes 或 no；compare 是 "the first"、"the second"、"the same" 之一；ground 是 "none" 或者 `parse_box` 能解析的框；refer 以 "a " 开头；caption、text、read 非空。**都区分大小写**。
2. `task_reward(task, pred, ref)`：格式不对 0（先过闸门，再算别的）；count、count_color、left、exists、compare、refer 完全相同 1、否则 0；ground：有一方是 none 时两者相同 1、否则 0，都是框时取 IoU（你自己的 `box_iou`）；caption、text、read：max(0, 1 − NED)（你自己的 NED）。

为什么 read 用 1 − NED 而不用 0/1（第 31 章 31.2 节第 4 部分）？

**交付**：`run_aggregate`、`run_grpo_loss`、`run_overlong_penalty`、`run_keep_groups`、`run_well_formed`、`run_task_reward`；`test_a5_grpo.py` 全部通过。

---

## 6. 第四部分：采样

### 6.1 问题（sample）：带温度的采样（5 分）

`sample(vlm, images, prompt, max_new, eos, temperature=1.0, generator=None, banned=()) → (B, n)`：

- images 是 float 的 (B, C, H, W)（或 (B, k, C, H, W)），prompt 是 (B, T0)，每行长度相同；
- 先把整个提示送进去建 KV cache，之后每步只送一个新 token（和 A2、A3 的 `generate` 一样）；
- 每步：banned 里的 token 的 logit 设为 −∞；`temperature == 0` 时取 argmax，否则 p = softmax(logits / temperature)，`torch.multinomial(p, 1, generator=generator)`；
- 所有行都写出过 eos，或者长度到了 `block_size`，就提前停止；
- 写出 eos 之后的位置内容无所谓（rollout 会截掉）。

测试检查：温度 0 时和"每步重新跑整条序列取 argmax"逐 token 相同（KV cache 写对了没有）；同一个 generator 种子得到相同的样本；T = 1、T = 2 时第一个 token 的频率和 softmax(logits / T) 一致（3000 个样本）；banned 的 token 从来不出现。

实际训练时 banned 要包括 `<pad>`、`<bos>`、`<img>`：模型生成一个 `<img>`，下一次把这条回答送回模型时占位符就多了一个（A4 遇到过的错误）。

### 6.2 问题（rollout）：每题 G 个回答（2 分）

`rollout(vlm, items, G, max_new, eos, temperature=1.0, generator=None, device="cpu", banned=()) → (responses, finished)`：

- items 是 [(图像 uint8 (k, 3, H, W), 提示的编号列表)]，图像送进模型前按 x / 127.5 − 1 转成 float；
- 每题采 G 个回答。图的形状和提示长度不同的题不能拼成一批：按 (形状, 提示长度) 分组，每组的每道题重复 G 次一起采；
- 返回两个长度为 N·G 的列表，**顺序是 [第 0 题的 G 个, 第 1 题的 G 个, …]**（不管你内部怎么分组）；responses 是回答的编号列表，**截掉 eos 及之后**；finished 表示写出了 eos。

### 6.3 问题（make_batch）：整理成 (ids, mask)（1 分）

`make_batch(prompts, responses, finished, eos, pad) → (ids (B, T), mask (B, T−1))`：每条序列 = 提示 + 回答 + [eos]（finished 时才加），右侧补 pad；mask[b, t] = 1 当且仅当 ids[b, t+1] 是回答的 token（含加上的 eos）。没写完的回答不加 eos——为什么不能给它补一个？

### 6.4 问题（policy_logprobs）：整条序列的对数概率（2 分）

`policy_logprobs(vlm, images, ids) → (B, T−1)`：images 是每条序列一个 float 张量 (k, 3, H, W) 的列表（一批里可以有 64×64 的场景和 32×256 的文字行，用 A4 的 `embed_list`）。输出就是 A4 的 `token_logprobs`。需要梯度时不要包 `no_grad`。

**交付**：`get_vlm`（与 A3、A4 相同）、`run_vlm_logits`、`run_sample`、`run_rollout`、`run_make_batch`、`run_policy_logprobs`；`test_a5_rollout.py` 全部通过。

---

## 7. 第五部分：遗忘与配比

### 7.1 问题（pareto）：帕累托前沿（2 分）

`pareto_front(points)`：points 是 [(新任务的分数, 旧任务的分数)]，都是越大越好。返回不被任何其他点支配的点的下标（升序）。a 支配 b：a 的两个分数都 ≥ b 且至少一个 >。两个完全相同的点互不支配。

### 7.2 问题（mixed_batch）：重放（2 分）

`mixed_batch(rng, B, ratio, make_new, make_old) → (样本, 是否旧任务)`：n_old = `round(ratio · B)`（Python 的 round）；先调用 n_old 次 `make_old(rng)`，再调用 B − n_old 次 `make_new(rng)`，按"先旧后新"排好，最后按 `rng.permutation(B)` 打乱（`out[j] = items[perm[j]]`）。顺序是规定死的，测试按它对照。

### 7.3 问题（rft_select）：拒绝采样微调的挑选（2 分）

`rft_select(rewards, G, threshold=1.0, k=1) → [全局下标]`：每组里奖励 ≥ threshold 的回答，按奖励从高到低（相同时下标小的在前）最多取 k 个。第 32 章 32.6 节的手算例子是测试之一。

### 7.4 问题（probe_kl）：探测 KL（2 分）

`probe_kl(logits_new, logits_base, mask) → float`：在 mask 为 1 的位置上，KL(p_base ‖ p_new) 的平均（不带梯度，用 float32）。注意方向：期望是对**旧模型**的分布取的。第 32 章 32.1 节的手算值是 0.0851；方向反了会得到 0.0920。

**交付**：`run_pareto_front`、`run_mixed_batch`、`run_rft_select`、`run_probe_kl`；`test_a5_mix.py` 全部通过。

---

## 8. 训练与评测脚本

写下面几个入口（可以共用代码），要求与 A1 第 7 节相同（可复现、`log.jsonl`、`config.json`、检查点、`--smoke`、`--set`、默认 `device=cpu`）。不单独给分，但实验都要用它们。

**一个共同的骨架**：从 `init` 检查点出发；视觉塔冻结；另存一份冻结的副本当**参考模型**（DPO 的 ref、GRPO 的 KL、探测 KL 都用它）；每步按 `method` 走一条分支；定期评测。

1. `rl`：配置至少包括
   - `init`、`method`（sft / rft / dpo / grpo）、`tasks`（目标任务，例如 `read,text`）、`replay` 和 `replay_tasks`（重放比例和任务，默认其余任务）；
   - 采样：`n_prompts`（每步多少题）、`G`、`temperature`、`max_new`；
   - GRPO：`beta`、`eps_low`、`eps_high`、`agg`、`mu`（同一批采样上更新几次）、`dynamic`、`overlong_cache`；
   - DPO：`dpo_beta`、`sft_weight`、`dpo_source`（`online`：从组内采样挑成对；`caption`：正确描述 vs 制造的幻觉）、`min_gap`；
   - RFT：`rft_threshold`、`rft_k`；
   - 学习率（RL 通常比 SFT 小一个数量级）、步数、`eval_every`。

   四种方法的一步：

   | 方法 | 一步做什么 |
   |---|---|
   | sft | 从生成器拿真值答案，A4 的 SFT 损失（可混重放） |
   | rft | 每题采 G 个，`rft_select` 挑出来，**只用写完了的**当 SFT 数据（可混重放） |
   | dpo | online：采 G 个，`pairs_from_groups`，只保留 chosen 写完了的对；caption：每张图一对（正确描述 vs `hallucinated_caption`） |
   | grpo | 采 G 个，算奖励（可加超长惩罚）和优势，（可选）动态采样丢掉白算的组，算 old_logp（和 ref_logp），更新 μ 次 |

   每步记录：损失、平均奖励、`zero_var`、格式正确率、回答长度、GRPO 的 `clip_frac`/`kl`/`ratio_max`、DPO 的 margin/acc/chosen 的对数概率、每步毫秒数。评测时记录：每个任务的准确率、目标任务和其余任务的平均、read 的 CER、探测 KL。
2. `eval --ckpt … [--base …]`：每个任务 n 题贪心生成，逐样本写进 `.jsonl`；打印每个任务的准确率和 95% 区间；给了 `--base` 时再打印相对起点的配对差和区间。
3. `predict --ckpt … --out-shapes … --out-lines …`：排行榜的两个测试集，各写一个 `.jsonl`。

**建议**：每次先 `--smoke`；正式跑之前先用 20 步确认奖励在动。如果 `zero_var` 接近 1，说明题太难（全是 0）或太容易（全是 1），换任务或换起点，不要硬跑。

---

## 9. 实验

要求与 A4 第 9 节相同：先写预测，再运行；所有比较都附置信区间或检验；只跑过一次的结论标明"单次运行"。RL 的方差比 SFT 大得多：**需要比较大小的结论至少跑 2 个种子**。

### 问题（exp_kl_estimators）：KL 的估计量（3 分）

**交付**：(a) 在 kl_by_hand 的两个分布上，样本数 n = 10、100、1000、10000，各重复 1000 次：三个估计量平均值的偏差和标准差，画成图；(b) 在你的 VLM 上：参考模型和一个训练了 50 步 GRPO 的模型，用当前模型的采样算每个 token 的 k1、k2、k3，再用整个词表精确算 KL，比较三者的平均和方差。k1 有多少比例是负的？

### 问题（exp_dpo）：DPO（7 分）

从同一个 A4 检查点，`dpo_source=caption`（目标任务 caption）：

**交付**：(a) 训练前：参考模型在 200 对（正确描述, 幻觉）上偏好正确描述的比例（第 30 章 30.6 节第 4 部分），substitute 和 add 分开报告；(b) β = 0.03、0.1、0.3 各训练一次：margin、acc、chosen 和 rejected 的对数概率随步数的曲线；**chosen 的对数概率下降了吗**（似然位移）？(c) 选一个 β，`sft_weight` = 0 和 1 比较：似然位移、caption 的奖励、其余任务的准确率、探测 KL；(d) 训练后在新的 200 对上的偏好准确率，以及贪心生成的描述里"图里没有的物体"出现的比例（自己写一个检查：把描述解析回物体列表）。

### 问题（exp_grpo）：GRPO（8 分）

目标任务 read（和 text）：

**交付**：(a) 每步的采样总数固定（例如 64），G = 4、8、16：`zero_var` 和奖励的曲线，最终的 CER（带区间）。G 太小、太大各有什么问题？(b) `agg` = sample、token、constant：回答长度的曲线、最终 CER。和 gc_by_hand 第 3 题的预测一致吗？(c) 原版（ε = 0.2，不做动态采样）和 DAPO（`eps_high=0.28`、`dynamic=true`、`overlong_cache`）：奖励、`clip_frac`、每步用时、熵（或每 token 平均对数概率）的曲线；(d) β = 0 和 0.04：KL 的曲线、旧任务的准确率、探测 KL。

### 问题（exp_reward_hacking）：奖励投机（4 分）

**交付**：(a) 设计一个**有漏洞**的奖励（例如 read 的奖励 = 回答里出现了多少个参考中的字符 / 参考长度，不管顺序和多余的字符），预测模型会学出什么，用 GRPO 跑到它学会为止；(b) 打印 5 个例子，画"有漏洞的奖励"和"真实的 1 − NED"随步数的两条曲线，指出它们分开的地方；(c) 修好奖励，或者加 KL 惩罚，或两者都做，再跑一次。哪一个有用？

### 问题（exp_four_methods）：同一起点，四种方法（10 分）

从同一个 A4 检查点，目标任务 read（和 text），**同样的计算预算**：

**交付**：(a) 先算出每种方法一步的计算量（前向、反向各多少个 token，第 32 章 32.6 节"预算"），据此定下每种方法的步数，使总计算量相同；(b) SFT、RFT、在线 DPO、GRPO，各跑一次不带重放、一次带 25% 重放（共 8 次，最好 2 个种子）。一张表：目标任务的准确率和 CER、其余任务的平均准确率、探测 KL、总用时，都带区间；(c) 把 8 个点画在（目标任务, 其余任务）的平面上，标出帕累托前沿（用你的 `pareto_front`）。起点也画上；(d) 用 `eval --base` 对比起点：哪些任务的下降在区间之外？(e) 第 32 章 32.3 节提到一个有争议的假说："在线 RL 遗忘得更少"。你的数据支持吗？探测 KL 和旧任务的下降是什么关系？

---

## 10. 排行榜

### 问题（leaderboard）：学了新的，不忘旧的（6 分）

目标：在两个测试集上**同时**好。分数 = ½ · 九个 ShapesWorld 任务准确率的平均 + ½ · (1 − min(CER, 1))，两项都附 95% 区间。

```bash
python -m assignments.a5_align.leaderboard score shapes.jsonl lines.jsonl
```

两个测试集与 A3（`a3_vlm/data/test.npz`，540 题）、A4（`a4_eval_sft/data/lines_test.npz`，500 行）的排行榜相同。

**规则**

1. **起点**：你自己 A4（或 A3）的检查点，不管它们花了多久。不能用下载的权重。
2. **硬件和时间**：一张租用的 24 GB 显卡，A5 的训练墙钟时间不超过 **20 分钟**（含采样，不含评测）。CPU 赛道：90 分钟，写明 CPU 型号。
3. **必须用到在线采样**：至少一半的更新步用的是**模型自己当时采样的回答**（RFT、在线 DPO、GRPO，或你自己设计的方法）。纯 SFT 的提交放在 A4 的排行榜。
4. **数据**：只能用 `miniocr.data` 的生成器，种子不能用 20261022–20261099。**不能用测试集训练、调参或选检查点**。
5. **代码**：第 1.1 节的限制照样适用。奖励、方法、配比都可以自由设计。
6. **报告**：`leaderboard.md` 一行要写分数、两项各自的值和 95% 区间、起点的分数。

**交付**：`leaderboard.md` 加一行；报告里一段：试过什么、什么有用，附训练曲线和起点 → 终点每个任务的变化（`eval --base`）。不按名次给分。

---

## 11. 交付清单与评分

| 部分 | 问题 | 分数 |
|---|---|---|
| 代码：策略梯度 | advantages 3、kl_estimate 3、clipped_objective 4 | 10 |
| 代码：DPO | optimal_policy 2、dpo_loss 4、pairs_from_groups 2、hallucination 2 | 10 |
| 代码：GRPO 与奖励 | aggregate 3、grpo_loss 5、dapo 2、rewards 4 | 14 |
| 代码：采样 | sample 5、rollout 2、make_batch 1、policy_logprobs 2 | 10 |
| 代码：遗忘与配比 | pareto 2、mixed_batch 2、rft_select 2、probe_kl 2 | 8 |
| 书面 | kl_by_hand 3、dpo_derivation 4、gc_by_hand 3 | 10 |
| 实验 | exp_kl_estimators 3、exp_dpo 7、exp_grpo 8、exp_reward_hacking 4、exp_four_methods 10 | 32 |
| 排行榜 | leaderboard 6 | 6 |
| **合计** | | **100** |

下一站是第 33 章（文档引擎），A6 在第 34 章之后：数据引擎。
