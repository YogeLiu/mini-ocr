# A6：数据引擎——去重、共识伪标签、渲染校验、配额，以及"固定配方，比数据"

| | |
|---|---|
| 位置 | 第 34 章之后 |
| 预计用时 | 25–35 小时 |
| 算力 | 写代码、跑测试、数据引擎：CPU（整个数据池跑一遍三元组共识约半分钟到一分钟）。固定配方的训练：GPU 赛道一张租用的 24 GB 显卡；CPU 赛道用小模型（作者的 CPU 上每步约 0.12 秒、3000 步，只测了步速，没有跑完整的训练） |
| 交付 | ① `a6/` 里的代码，`tests/adapters.py` 接好后全部测试通过；② 你的数据引擎脚本和它生成的训练数据；③ 书面报告 `writeup.md`（从 `writeup_template.md` 复制）；④ 排行榜记录 |
| 满分 | 100 分：代码 52 分，书面 10 分，实验 32 分，排行榜 6 分 |

A1–A5 都是"数据给定，改模型和训练"。这一次反过来：**模型、步数、优化器全部固定，你只能决定训练数据**（和 DataComp 一样）。

数据池是 2.4 万张"爬下来的"公式图。**每张图没有真值**，只有 6 个"专家"模型各写的一个 LaTeX 答案。数据池故意带着真实数据池常见的毛病：

- 同一篇文档里的公式风格相近，还常有"改了一个符号"的变体；有重复、有跨文档的复制；
- 一部分图是**测试集公式的重新渲染**（污染）；
- 有些图噪声很重，专家们基本在乱写；
- 专家们各有毛病：alpha 很准；beta 也准，但写法不统一（`x^2`、`{a \over b}`、`x^{2}_{n}`）；gamma 一般；**delta 和 epsilon 在难图上犯同一个结构性错误**（像用相似数据训练出来的两个模型）；zeta 常把答案截断；
- 500 张图有人工核对过的真值（**审核集**）。

你要写的是第 33、34 章的数据引擎：

1. **清洗、去重、划分**（第 33 章 33.7 节）：规范化、MinHash 与 LSH、近似重复的簇、按组划分、去污染；
2. **多模型共识**（第 33 章 33.4 节、第 34 章 34.3 节）：两两一致度、三元组共识、分档、加权投票、估计专家可靠度的 EM；
3. **渲染校验**（第 33 章 33.5 节）：把候选重新画出来和原图比，在候选里重选；
4. **难例挖掘与配比**（第 33 章 33.6 节、第 34 章练习 34.9）：自洽度、特征、k-means、按簇配额、簇内抽样权重。

最后，用你的引擎从数据池里生成一份训练数据，交给**固定的训练配方**，在测试集上比结果。

---

## 1. 规则

### 1.1 能用什么、不能用什么

| 不能用 | 自己写 |
|---|---|
| `datasketch`、`sklearn`（`KMeans`、`MinHashLSH`）、`scipy` | MinHash、LSH、k-means |
| `Levenshtein`、`editdistance`、`rapidfuzz`、`jellyfish` | 编辑距离 |
| `miniocr.engine`、`miniocr.xiaomi`、`miniocr.eval`、`miniocr.metrics` | 第 33、34 章的那些函数 |
| `assignments.a6_data.pool` 的 `_truth()`、`planted()`，`fixed_train oracle` 生成的文件 | ——它们是数据池的真值，**只给实验当上限、给排行榜查污染** |

**可以用**：Python 标准库（`hashlib`、`re`、`itertools`）、numpy、`miniocr.data` 的全部（`latex.parse`/`normalize`/`tokenize`、`render.render_formula`、`imgops.resize_nearest` 等）、`assignments.a6_data.pool` 的 `load_pool`、`load_test`。`test_a6_rules.py` 会检查 `a6/` 里的 import 和对真值的访问（查不出所有绕过的办法，第 1.2 节的诚信规则才是正式的）。

**可以把你自己 A1–A5 的代码复制进 `a6/`**（例如 A4 的 bootstrap、McNemar、AUROC），不能 import 前面的作业。

**固定配方**（`fixed_train.py`）、数据池（`pool.py`）、排行榜（`leaderboard.py`）是基础设施，**不要改**；排行榜会核对 `fixed_train.py` 的哈希。

### 1.2 AI 助手、参考实现与诚信

与 A1 相同。另外：数据池和测试集都是由代码确定地生成的，你当然能读到生成它们的代码；**用任何方式拿到数据池的真值（包括照着 pool.py 重新生成）都等于作弊**。审核集的 500 个真值是允许用的。

### 1.3 前提

`miniocr.data.latex` 的 `parse`、`normalize`（练习 3.4、3.5）和 `miniocr.data.imgops.resize_nearest`（练习 2.4）要先做完：规范化、渲染校验和固定配方的评分都要用它们。`fixed_train.py` 开始时会检查 `normalize`，没做完或有错就停下。

---

## 2. 准备

```
assignments/a6_data/
├── handout.md
├── a6/                   你的代码（库函数 + 你的数据引擎脚本）
├── tests/
│   ├── adapters.py       ← 你要填的接线
│   └── test_a6_*.py
├── pool.py               数据池与测试集（基础设施）
├── fixed_train.py        固定的训练配方（基础设施）
├── leaderboard.py        排行榜（基础设施）
└── writeup_template.md
```

```bash
python -m pytest -q assignments/a6_data/tests
python -m assignments.a6_data.pool info                 # 数据池概况和几个例子（第一次生成约 10 秒）
```

```python
from assignments.a6_data.pool import load_pool, load_test, EXPERTS
pool = load_pool()   # images (N, 48, 192) uint8；outputs：N 行 × 6 个字符串；doc (N,)；audit {下标: 真值}
test = load_test()   # images (1000, 48, 192)；labels（规范写法）；levels（难度 1–4）
```

第一到四部分互不依赖。第三部分要渲染公式，要先做完练习 3.4、3.5。

---

## 3. 第一部分：清洗、去重、划分

### 3.1 问题（latex_key）：不解析的规范化（3 分）

`latex_key(s)`：对任意字符串都能用（解析不了的也行），按顺序：

1. 去首尾空白；整个字符串是 `$$…$$`、`\[…\]` 或 `$…$` 时去掉外壳（再去首尾空白）；
2. 整个字符串是 `\begin{E}…\end{E}`（E ∈ align、equation、gather、displaymath，可带 `*`）时只留中间；
3. 单个字符的上下标加花括号：`_` 或 `^`（前面不是反斜杠）后面紧跟一个字母或数字时，`x^2` → `x^{2}`；
4. 删掉 `\displaystyle`；`\,`、`\;`、`\:`、`\!`、`\ `（反斜杠加空格）、`~`、`\quad`、`\qquad` 换成空格；
5. 删掉所有空白，但"命令名 + 空白 + 字母"里的空白换成一个空格（`\alpha x` 不能变成 `\alphax`）。

第 3 步在第 5 步之前：`a ^ b` 的结果是 `a^b`（这是这个规范化的已知缺陷，不用修）。

### 3.2 问题（canon）：规范写法（1 分）

`canon(s)`：先 `latex_key`，再用 `miniocr.data.latex.normalize`；解析失败（`LatexError`）时返回 `latex_key` 的结果。这样 `x^2`、`{a \over b}`、`x^{2}_{n}` 都会写成唯一的规范写法。

### 3.3 问题（dedup_basics）：精确去重、n-gram、Jaccard（2 分）

- `dedup(items, key)`：返回每个 key 第一次出现的下标；
- `shingles(s, n=3)`：所有长度为 n 的子串组成的集合；比 n 短的字符串返回 `{s}`（公式很短，所以默认 n = 3，不是网页去重常用的 5 或更大）；
- `jaccard(a, b)`：|a ∩ b| / |a ∪ b|，两个都空时为 1。

### 3.4 问题（minhash）：MinHash（3 分）

**哈希规定死**，测试按它逐位对照：

- 元素 → 整数：`int.from_bytes(hashlib.blake2b(e.encode(), digest_size=4).digest(), "little")`；
- P = 2³¹ − 1；系数 `AB = np.random.default_rng(12345).integers(1, P, size=(2, 1024), dtype=np.int64)`；
- 第 i 个哈希函数 hᵢ(x) = (AB[0, i] · x + AB[1, i]) mod P（int64 不会溢出）；
- `minhash(sh, n_hash)` 的第 i 位 = 集合里所有元素的 hᵢ 的最小值，返回长度 n_hash 的 int64 数组。

### 3.5 问题（lsh）：LSH 与近似重复的簇（4 分）

1. `lsh_pairs(sigs, bands)`：sigs 是 (N, n_hash)。每段 r = n_hash // bands 位；任一段完全相同的两条成为候选对。返回 `{(i, j)}`，i < j。
2. `near_dup_clusters(sets, pairs, tau)`：只看候选对；对每一对（按 `sorted(pairs)` 的顺序），Jaccard **≥** tau 就把两条连到同一个簇（并查集）。返回每条所在簇的编号 = 簇里**最小的下标**。

### 3.6 问题（group_split）：按组划分（2 分）

`group_split(groups, val_frac, seed)`：不同的组按 `str` 排序，用 `np.random.default_rng(seed).permutation(组数)` 打乱，依次放进验证集，直到验证集的**样本数**第一次 ≥ val_frac × 总数；其余进训练集。返回 (train 下标, val 下标)，都升序。

### 3.7 问题（decontaminate）：去污染（1 分）

`decontaminate(pool_sets, test_sets, tau, n_hash=64, bands=16)`：数据池在前、测试集在后拼在一起算 MinHash 和 LSH 候选对；一端是数据池、一端是测试集的候选对，用**精确的** Jaccard 核对 ≥ tau。返回数据池里被判为污染的下标（升序）。

**交付**：`run_latex_key`、`run_canon`、`run_dedup`、`run_shingles`、`run_jaccard`、`run_minhash`、`run_lsh_pairs`、`run_near_dup_clusters`、`run_group_split`、`run_decontaminate`；`test_a6_clean.py` 全部通过。

### 问题（minhash_math）：MinHash 与 LSH 的数学（4 分，书面）

1. 证明：对一个随机排列（或"足够随机"的哈希），两个集合的 MinHash 相等的概率等于它们的 Jaccard 相似度。
2. n_hash = 64 时，(bands, r) 可以取 (32, 2)、(16, 4)、(8, 8)、(4, 16)。写出成为候选的概率 1 − (1 − Jʳ)^bands，在 J = 0.3、0.5、0.7、0.8、0.9 时各是多少？S 曲线的"阈值"大约是 (1/bands)^(1/r)，各是多少？
3. 想要"J ≥ 0.8 的几乎都成为候选、J ≤ 0.5 的很少成为候选"，选哪一组？如果 n_hash 可以加到 128 呢？
4. 去污染和去重对这两种错误的容忍度一样吗？（漏掉一对 vs 多报一对，各自的代价是什么？）

---

## 4. 第二部分：多模型共识

### 4.1 问题（ned）：编辑距离与 NED（1 分）

`ned(a, b)` = 编辑距离 / max(两者长度)，两个都空时 0。相似度 sim(a, b) = 1 − NED。

### 4.2 问题（pairwise）：两两一致度（2 分）

`pairwise_agreement(preds)`：(n, n) 的 numpy 数组，A[i, j] = sim(preds[i], preds[j])，对角线为 1。

### 4.3 问题（triplet）：三元组共识（4 分）

`triplet_consensus(preds, k=3) → (中心点下标, 组内平均一致度, 组的下标元组)`：在所有 C(n, k) 个组合（`itertools.combinations` 的顺序）里，选组内两两一致度的平均值**最高**的一组（严格更高才替换：平局取顺序靠前的）；组内的中心点 = 和组内其他成员一致度之和最大的（平局取下标小的）。n < k 时用全部 n 个；只有一个成员时一致度记 1。

### 4.4 问题（votes）：分档、多数票、加权投票（2 分）

- `tier(score, hi=0.9, lo=0.6)`：≥ hi 为 `"high"`，≥ lo 为 `"medium"`，否则 `"low"`；
- `majority_vote(preds)`：出现次数最多的（平局取最先出现的）；
- `weighted_vote(preds, weights)`：每个不同的答案得分 = 写它的专家的权重之和（权重可以为负），取最高分的（平局取最先出现的）。

### 4.5 问题（one_coin_em）：专家的可靠度（3 分）

不知道真值，能不能估计每个专家有多准？最简单的"单参数"模型（Dawid–Skene 的简化）：每个专家以概率 acc_k 写对。`one_coin_em(outputs, iters=10, eps=1e-3)`，outputs 是 N 行 × K 个答案：

1. 权重 w 从全 1 开始；
2. 每轮：用 `weighted_vote` 得到每行的标签；acc_k = (专家 k 和标签相同的行数 + 1) / (N + 2)，裁到 [eps, 1 − eps]；w_k = log(acc_k / (1 − acc_k))；
3. 最后用最终的权重再投一次票。返回 (标签列表, acc (K,) 数组)。

为什么 w 用对数几率？（提示：假设各专家独立，"某个答案是对的"的后验对数几率是什么？）独立性假设在 delta、epsilon 上成立吗？

**交付**：`run_ned`、`run_pairwise_agreement`、`run_triplet_consensus`、`run_tier`、`run_majority_vote`、`run_weighted_vote`、`run_one_coin_em`；`test_a6_consensus.py` 全部通过。

### 问题（consensus_by_hand）：手算一次共识（3 分，书面）

一张图的真值是 `\frac{a}{b}+x^{2}`。五个专家（规范化之后）：

| 专家 | 答案 |
|---|---|
| 0 | `\frac{a}{b}+x^{2}` |
| 1 | `\frac{a}{b}+x^{2}` |
| 2 | `\frac{a}{b}+x_{2}` |
| 3 | `\frac{a}{b}+x_{2}` |
| 4 | `\frac{a}{b}+x^{2`（截断） |

1. 写出 5×5 的编辑距离矩阵和一致度矩阵。
2. 十个三元组的平均一致度各是多少？三元组共识选了哪一组、哪个中心点？对吗？如果平局时改成"取顺序靠后的"，结果还对吗？
3. 多数票的结果是什么？对吗？靠的是什么？
4. 如果权重是 log(acc/(1 − acc))、acc = (0.9, 0.85, 0.7, 0.7, 0.6)，加权投票选哪个？2、3 号的错误是相关的——在这个例子里，什么信息能把它们和 0、1 区分开？（提示：第三部分。）

---

## 5. 第三部分：渲染校验

### 5.1 问题（ink）：墨迹的外接框（2 分）

1. `ink_crop(img, thresh=128, min_neighbors=0)`：墨迹 = 比 thresh 暗、**而且 8 个邻居里至少有 min_neighbors 个也暗**的像素（图外当作不暗）；裁到墨迹的外接框；没有墨迹时返回 None。
2. 为什么需要 min_neighbors？（噪声 0.2 的图上，一个孤立的暗点就能把外接框撑到整张图。）它的代价是什么？（1 像素粗的线的两端只有一个邻居。）

### 5.2 问题（blur_ncc）：模糊与归一化互相关（2 分）

- `blur3(m)`：3×3 的均值模糊，边缘复制（`np.pad(..., mode="edge")`），输出和输入一样大；
- `ncc(a, b)`：两个数组各自减去均值之后的余弦相似度；任一方全是常数时为 0。

### 5.3 问题（render_score）：渲染校验（4 分）

`render_score(latex, img, size=16, thresh=128, shift=2, min_neighbors=2)`：

1. 用 `parse` 解析，在 3H × 3W 的白底上用 `render_formula(tree, np.random.default_rng(0), H=3H, W=3W, size=size)` 渲染（不加噪声）；解析失败或放不下（`LatexError`、`ValueError`）得 0；
2. 渲染图用 `ink_crop(pred, thresh)`，原图用 `ink_crop(img, thresh, min_neighbors)`；任一边没有墨迹得 0；
3. 两个外接框的高或宽相差超过原图墨迹的 25%：得 0；相差超过 shift 个像素：把渲染图缩放（`resize_nearest`）到原图墨迹的大小；
4. 渲染图四周补 shift 个像素的白边（255），对所有平移 (dy, dx) 取一个和原图墨迹一样大的窗口；墨迹强度 255 − 像素，两边都 `blur3`，算 `ncc`，取最大值。

### 5.4 问题（render_select）：在候选里重选（2 分）

`render_select(cands, img, size=16) → (最高分的下标（平局取小的）, 最高分, 全部分数)`。

**交付**：`run_ink_crop`、`run_blur3`、`run_ncc`、`run_render_score`、`run_render_select`；`test_a6_verify.py` 全部通过（测试检查：正确的公式 > 0.99、等价写法得分相同、上下标写反得分更低、解析失败和大小差太多得 0、噪声 0.2 的图仍然 > 0.9、平移一个像素不影响）。

---

## 6. 第四部分：难例挖掘与配比

### 6.1 问题（self_consistency）：自洽度（2 分）

`self_consistency(samples)`：同一张图的 G 个回答两两之间相似度的平均（不含自己和自己）；G < 2 时为 1。

### 6.2 问题（features）：公式的特征（2 分）

`formula_features(s)` → 7 个 float：[记号数, `\frac` 的个数, `\sqrt` 的个数, `^` 和 `_` 的个数, 花括号的最大嵌套深度, `+ - =` 的个数, 能否解析（1/0）]。记号按正则 `\\[A-Za-z]+|\S` 切（不管能否解析）；深度：遇到 `{` 加一、`}` 减一（不低于 0），记录最大值。

### 6.3 问题（kmeans）：k-means（3 分）

`kmeans(X, k, rng, iters=50) → (labels, centers)`：

- k-means++ 初始化：第一个中心 `X[rng.integers(N)]`；之后每个中心按"到最近已有中心的距离平方"成比例地抽（`rng.choice(N, p=d / d.sum())`）；
- 交替：每个点分给最近的中心（平局取编号小的）；每个中心更新为它的点的平均（没有点的中心不动）；中心不再变化（`np.allclose`）或到了 iters 次就停；
- **返回的 labels 要按最终的 centers 重新分一次**。测试检查：分开的三团点各自成一簇、每个点都在最近的中心、中心等于簇内平均、同一个 rng 种子结果相同。

### 6.4 问题（quota_sample）：按簇配额（4 分）

`quota_sample(labels, weights, n, rng)`：

1. **名额**：簇按编号排序。只要还有名额没分完：在"还有剩余样本的簇"之间平均分 `divmod(剩余名额, 簇数)`，余数给排在前面的簇（一簇一个），每簇最多拿到它剩下的样本数；重复直到分完或所有簇都满了。
2. **簇内**：`rng.choice(簇内下标, size=名额, replace=False, p=…)`，p 按 weights 成比例；权重全为 0、或者正权重的样本少于名额时，退回均匀（p=None）。
3. 按簇编号、簇内抽取的顺序返回下标数组。

### 6.5 问题（sampling_weights）：簇内抽样权重（2 分）

第 34 章练习 34.9（式 10）：wᵢ = (aᵢ + eps)^(f_a/τ) · (1 − uᵢ + eps)^(f_u/τ)，a 是专家一致度（越高标签越可信），u 是模型自洽度（越低模型越没把握）；在每个簇内归一化。

### 6.6 问题（hard）：难例候选（1 分）

`hard_candidates(agreement, consistency, tau_a=0.9, tau_u=0.7)`：agreement ≥ tau_a 且 consistency < tau_u 的布尔数组。

**交付**：`run_self_consistency`、`run_formula_features`、`run_kmeans`、`run_quota_sample`、`run_sampling_weights`、`run_hard_candidates`；`test_a6_mining.py` 全部通过。

### 问题（quota_by_hand）：手算配额与权重（3 分，书面）

1. 三个簇各有 12、1、2 个样本，一共抽 6 个。按 6.4 的规则，每簇几个名额？写出每一轮的分配。如果改成"按簇的大小成比例分配"呢？哪一种更适合"让少见的结构也被学到"？
2. 一个簇里三个样本：a = (1.0, 0.9, 0.6)，u = (0.9, 0.3, 0.3)，f_a = f_u = 1，eps = 0。τ = 1 和 τ = 0.5 时的抽样概率各是多少？第一个样本的标签最可信，为什么抽到它的概率最小？
3. 什么样的样本是"难例候选"？为什么既要专家一致度高、又要模型自洽度低？只看其中一个会挑出什么？

---

## 7. 你的数据引擎脚本

在 `a6/` 里写一个脚本（不单独给分，但实验和排行榜都要用），从数据池生成训练数据 `.jsonl`：每行 `{"i": 数据池下标, "label": "LaTeX"}`；同一个下标可以出现多次（等于加权）。要求：

- 可复现（同样的配置得到逐字节相同的文件）、`--set key=value` 改配置、`--smoke` 只处理前 2000 张；
- **每一步都在审核集上报告覆盖率和标签精度**（例如"去污染之后""只要高档""渲染校验之后"），写进日志；
- 至少包括：规范化、去污染、共识（三元组或你自己的方法）、过滤（分档、渲染校验或别的）。重选标签、配额、难例加权、用固定配方训练出来的模型当第 7 个专家或算自洽度，都是可选的。

固定配方的用法：

```bash
F=assignments.a6_data.fixed_train
python -m $F train --data 你的数据.jsonl --track cpu --out outputs/a6/run1    # 或 --track gpu（一张 24 GB 的卡）
python -m $F train --data 你的数据.jsonl --smoke --out outputs/a6/smoke         # 几步，只检查能跑
python -m $F predict --ckpt outputs/a6/run1/model.pt --indices idx.txt --G 8 --temperature 1 --out s.jsonl   # 采样（自洽度）
python -m assignments.a6_data.leaderboard check 你的数据.jsonl                    # 格式、规模、污染
python -m assignments.a6_data.leaderboard score 你的数据.jsonl outputs/a6/run1   # 测试集分数（规范化后整式全对率）
```

`train` 结束后 `run1/result.json` 里有测试集的整式全对率、95% 区间、平均 1 − NED、按难度的结果；`test_preds.jsonl` 是逐题的预测（做配对比较用）。

---

## 8. 实验

要求与 A4 第 9 节相同：先写预测，再运行；所有比较附置信区间或检验；只跑过一次的结论标明"单次运行"。标签精度都在**审核集**上估计（500 条：区间不窄，记得报告）。`leaderboard score` 会打印你的标签在整个数据池真值上的准确率，**只能在报告里引用，不能用来调参**。

### 问题（exp_pool_audit）：先看数据（4 分）

**交付**：(a) 审核集上每个专家的整式全对率：原样比较和 `canon` 之后比较，带区间。哪个专家的差别最大，为什么？(b) 按公式的结构（用 `formula_features` 分桶：有没有分式、根号、上下标）和图的"脏"程度（你自己设计一个只看图的估计，例如孤立暗点的比例）分桶，每个专家的准确率怎样变？(c) 找出 delta、epsilon 同错的例子，归纳它们错在哪里。

### 问题（exp_dedup）：去重与泄漏（5 分）

**交付**：(a) 用每张图的共识标签：精确重复（`canon` 相同）有多少组、多少张？近似重复的簇（τ = 0.8、0.9）呢？(b) S 曲线：从数据池随机取 3000 对（按精确 Jaccard 分桶），比较 (bands, r) = (16, 4) 和 (8, 8) 下成为候选的实际比例与理论曲线；(c) 3000 条标签上，LSH + 核对和"两两全算"的近似重复对：召回率、用时；(d) 你自己的验证集：按图随机划分 vs 按 doc 的 `group_split`，验证集里"在训练集中有近似重复"的比例各是多少？这会让验证分数偏高多少（用固定配方训练一次、在两种验证集上评测来说明，或者给出估计）？

### 问题（exp_consensus）：共识（7 分）

**交付**：(a) 一张表：单个专家、多数票、三元组（k = 2、3、4）、单参数 EM、EM 权重的加权投票——审核集上的标签精度（带区间）；(b) 三元组的一致度阈值从 0.6 到 1.0：覆盖率–精度曲线；在哪里收益递减？(c) EM 估计出的 acc 和审核集上的真实准确率比较：谁被高估了？为什么（相关的专家）？(d) 共识还错的审核集样本，有多少是 delta、epsilon 同错造成的？

### 问题（exp_render）：渲染校验（6 分）

**交付**：(a) 审核集上，共识标签对的和错的两组的渲染分分布（直方图）、用渲染分找错标签的 AUROC；(b) 渲染分为 0 的样本按你在 exp_pool_audit 里的"脏"程度分桶：渲染校验在哪里失效？改 `min_neighbors`（或你自己的去噪）能救回多少？(c) 中档样本用 `render_select` 在 6 个候选里重选：标签精度和覆盖率变化多少？修好了多少 delta、epsilon 的结构性错误？(d) 渲染分 0.95 意味着标签"对"吗？举一个渲染分很高但标签错的例子（或者说明为什么在这个数据池里找不到）。

### 问题（exp_fixed_train）：固定配方，比数据（10 分）

同一个赛道（CPU 或 GPU，写清楚）、固定配方：

**交付**：(a) 先预测，再训练这些数据：① `oracle`（真值标签，上限；只用来对照）；② 单个专家 gamma 的全部标签；③ 单个专家 alpha；④ 三元组中心点，全部；⑤ 只要高档；⑥ 你的引擎的最终版本。一张表：条数、审核集上估计的标签精度、测试集整式全对率 [区间]、按难度的结果、训练用时；(b) 用 `test_preds.jsonl` 做配对比较（A4 的配对 bootstrap 或 McNemar）：⑤ vs ④、⑥ vs ⑤ 的差别是真的吗？(c) **数量和质量**：从 ⑥ 里随机保留 25%、50%、100%，固定步数下结果怎样变？和"标签精度从 96% 提到 99%"相比，哪个更值？(d) 标签精度和测试分数的关系：把 (a)、(c) 的点画在一张图上，能看出什么？标签噪声被"平均掉"了多少（第 33 章 33.9 节第 6 部分）？

---

## 9. 排行榜

### 问题（leaderboard）：谁的数据好（6 分）

目标：固定配方在测试集（1000 个公式，4 个难度各约 1/4）上的**规范化后整式全对率**尽量高。

**规则**

1. **训练**：只能用没改过的 `fixed_train.py train`，两个赛道分开排名：GPU 赛道（`--track gpu`，一张租用的 24 GB 显卡）和 CPU 赛道（`--track cpu`，写明 CPU 型号）。
2. **图**：只能用数据池里的图（用下标指定），不能自己渲染新图、不能改图。
3. **标签**：只能来自专家的输出（选一个、规范化、拼接或别的变换）、审核集的真值，或者你用**只在数据池上训练**的模型（例如固定配方训练出来的）生成的答案。不能人工标注、不能用 `_truth`/`planted`/`oracle`、不能照着 `pool.py` 重新生成真值。
4. **算力**：数据引擎（共识、渲染校验、聚类……）在 CPU 上不超过 2 小时；如果用模型辅助（训练一个模型来采样、算自洽度或当第 7 个专家），这部分的 GPU 时间不超过 20 分钟（CPU 赛道：CPU 时间不超过 2 小时）。固定配方的那一次训练不算。
5. **污染**：数据里植入的测试集公式不能超过 5 条（`leaderboard check` 会告诉你有几条，但不告诉你是哪几条）。
6. **报告**：`leaderboard.md` 一行要写分数和 95% 区间、条数、审核集上估计的标签精度、引擎用时、`leaderboard score` 打印的"有效"。

**交付**：`leaderboard.md` 加一行；报告里一段：试过什么、什么有用，附 exp_fixed_train 里的对照。不按名次给分。

---

## 10. 交付清单与评分

| 部分 | 问题 | 分数 |
|---|---|---|
| 代码：清洗、去重、划分 | latex_key 3、canon 1、dedup_basics 2、minhash 3、lsh 4、group_split 2、decontaminate 1 | 16 |
| 代码：多模型共识 | ned 1、pairwise 2、triplet 4、votes 2、one_coin_em 3 | 12 |
| 代码：渲染校验 | ink 2、blur_ncc 2、render_score 4、render_select 2 | 10 |
| 代码：难例挖掘与配比 | self_consistency 2、features 2、kmeans 3、quota_sample 4、sampling_weights 2、hard 1 | 14 |
| 书面 | minhash_math 4、consensus_by_hand 3、quota_by_hand 3 | 10 |
| 实验 | exp_pool_audit 4、exp_dedup 5、exp_consensus 7、exp_render 6、exp_fixed_train 10 | 32 |
| 排行榜 | leaderboard 6 | 6 |
| **合计** | | **100** |

六个大作业到这里结束。第 35 章把我们的代码搬到真实的模型上，第 36 章的毕业项目会把 A1–A6 的零件组合成一个完整的系统。
