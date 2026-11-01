# 第 3 章　习题：概率、softmax 与交叉熵

习题分五类：概念题、手算题、编程题、调试题、闭卷手写。答案和提示在最后，**先自己做，再看答案。** 答错或答不完整的题，记进复习队列，一周后和三周后各重做一次。

---

## 一、概念题

用自己的话回答，每题三到五句。

**C1.** logits 和概率有什么区别？为什么说 logits 是"概率的对数尺度"？给所有 logits 加 5，概率怎样变化？乘以 5 呢？

**C2.** softmax 为什么用指数，而不是直接除以总和或者先平方再除以总和？各举一个反例。

**C3.** 不减最大值的 softmax 在 fp32 和 fp16 下分别从多大的 logit 开始溢出？溢出以后得到的是 inf 还是 nan？为什么一个位置的 nan 可能让经过它的梯度、以及更新后的相关参数都变成 nan？

**C4.** 减去最大值以后，为什么既不会上溢、分母也不会是 0？减最大值会不会改变梯度？

**C5.** 已经有了稳定的 softmax，为什么还要单独写 log_softmax？举一个 `softmax(z).log()` 出错的例子。

**C6.** 从最大似然推出平均 NLL 的四步是什么？每一步为什么不改变最优解？其中哪一步用到了"样本独立"的假设？

**C7.** 模型给正确类别 0.9 的概率，预测是对的，为什么 loss 不是 0？loss 能不能等于 0？

**C8.** 交叉熵、熵、KL 散度之间是什么关系？为什么最小化交叉熵等价于最小化 KL？语言模型的 loss 为什么降不到 0？

**C9.** 什么叫"恰当评分规则"？为什么"罚 1 − p"不是恰当评分规则，而"罚 −ln p"是？这对训练出来的模型有什么影响？

**C10.** 用一句话解释 ∂L/∂z = p − y 的含义：正确类别的 logit 怎样变？错误类别呢？哪个错误类别被压得最多？为什么各分量之和是 0？

**C11.** 为什么 softmax 后面接交叉熵，而不接均方误差？从"自信地犯错"的情况说明。

**C12.** `ignore_index` 要做到哪两件事？只做到第一件会怎样？为什么清零时用 `torch.where` 而不是乘以 0？

**C13.** 按 token 平均和按序列平均有什么区别？用班级平均分打个比方。各适合什么情况？

**C14.** 初始 loss 为什么应该约等于 ln C？如果初始 loss 远大于 ln C，可能是什么原因？明显小于 ln C 呢？

**C15.** 做梯度检查时，为什么要用 float64、中心差分、h ≈ 1e-5？h 太大和太小分别有什么问题？

**C16.** 公式 OCR 的解码器每一步的 loss 是什么？为什么"交叉熵低"不等于"公式识别对了"？

---

## 二、手算题

只用纸笔和计算器（可以用计算器算指数和对数）。

**H1.** z = [1, 2, 3]。
  (a) 求 softmax(z)。
  (b) 求 softmax(z + 10)，不用重新算指数，说明理由。
  (c) 求 softmax(2z)。和 (a) 比，分布变尖还是变平了？

**H2.** z = [1, 2, 3]。
  (a) 正确类别分别是 0 和 2 时，loss 各是多少？
  (b) 正确类别是 0 时，写出 ∂L/∂z。
  (c) 取学习率 1，沿负梯度走一步，新的 z 是多少？新的 p_0 变大了吗？（可以只估计方向。）

**H3.** z = [100, 101, 103]，用"减最大值"的方法手算 softmax 和 log_softmax，写出每一个中间量。

**H4.** 一枚硬币抛 12 次，3 次正面。
  (a) 写出似然 L(θ) 和平均 NLL。
  (b) 求最大似然估计 θ*。
  (c) 求 θ* 处的平均 NLL，并和 θ = 0.5 时比较。

**H5.** KL 散度的方向。
  (a) q = [0.7, 0.2, 0.1]，p = softmax([2, 1, 0.1]) = [0.6590, 0.2424, 0.0986]。分别求 KL(q ‖ p) 和 KL(p ‖ q)。
  (b) q = [0.5, 0.5]，p = [0.9, 0.1]。分别求 KL(q ‖ p) 和 KL(p ‖ q)。两个方向相等吗？

**H6.** 一个 batch 有 3 个序列，每个位置的 loss 如下（"·"表示补齐位置）：

```
序列 A：1.0  2.0  3.0
序列 B：0.5   ·    ·
序列 C：2.0  2.0   ·
```

  (a) 按 token 平均是多少？
  (b) 按序列平均是多少？
  (c) 如果有人把补齐位置的 loss 当 0，又除以 3 × 3 = 9，得到多少？
  (d) 按 token 平均时，序列 B 那个 token 的梯度权重是多少？按序列平均时呢？

**H7.**
  (a) 一个词表大小为 32000 的语言模型，第一步的 loss 应该是多少？
  (b) 一个 1000 类的图像分类器，第一步 loss 是 0.7。你会先检查什么？
  (c) 一个 10 类分类器，类别 0 占 50%，其余 9 类平分剩下的 50%。只学会类别频率的模型 loss 是多少？

**H8.** 线性 softmax 分类器：X = [[1, 2], [0, 1]]（N = 2，D = 2），W = 0（D = 2，C = 2），y = [0, 1]，reg = 0。
  (a) 求 scores、P、loss。
  (b) 求 dS = (P − Y)/N 和 dW = Xᵀ dS，写出每一步的形状。

**H9.** 有人先 softmax 再把结果传给 `F.cross_entropy`，C = 5。即使模型完美（softmax 输出 one-hot），loss 最低能降到多少？

**H10.**
  (a) fp16 下，naive softmax 处理 logits [11, 0] 和 [12, 0]，哪个会出 nan？
  (b) 一个公式 30 个 token，假设每个 token 独立地以 99% 的概率预测对，整个公式完全正确的概率是多少？如果每个 token 99.9% 呢？

---

## 三、编程题

所有编程题都在 `miniocr/nn/losses.py` 里，参考实现已经写好并标了练习标记。想从空白开始做，先生成挖空版：

```bash
python -m tools.make_student ../my-course
cd ../my-course && python -m pytest -q book/part1_foundations/ch03_softmax_ce
```

挖空版里这些函数只剩下签名和文档字符串，测试应该全部失败；你写完以后，测试应该全部通过。**不许调用 `torch.softmax`、`torch.log_softmax`、`F.cross_entropy`、`F.nll_loss`、`torch.logsumexp`**，只用 `max`/`amax`、`exp`、`log`、`sum`、`gather`、`where` 和矩阵乘法。

### 练习 3.1　稳定的 softmax（必做）

实现 `softmax(z, dim=-1)`。要求：任意形状、任意 dim；fp32 下 [1000, 999, 998] 和 fp16 下 [12, 10, 8] 都不出 nan；每一行各减各的最大值（`[[1000, 999], [0, −1]]` 两行都要对）；fp16 下 65536 个全 0 的 logits 也要得到 1/65536（求和不能溢出，正文 3.3.4 节）；输出的 dtype 和输入相同。

### 练习 3.2　稳定的 log-softmax（必做）

实现 `log_softmax(z, dim=-1)`。要求：z = [0, −200] 时输出有限值，第二个约为 −200；和练习 3.1 一样，fp16 大词表时求和不能溢出，输出 dtype 和输入相同。不许写成 `softmax(z).log()`。

### 练习 3.3　带 ignore_index 的交叉熵（必做）

实现 `cross_entropy(logits, targets, ignore_index=-100, reduction="mean")`。要求：

- logits 形状 (..., V)，类别在**最后一维**；targets 形状 (...)。
- 支持 `reduction` 为 `"none"`、`"sum"`、`"mean"`；`"none"` 时被忽略的位置为 0。
- `"mean"` 只除以有效位置的个数。
- 被忽略位置的梯度精确为 0；改动被忽略位置的 logits，loss 不变。

### 练习 3.4　按序列平均的交叉熵（必做）

实现 `sequence_mean_cross_entropy(logits, targets, ignore_index=-100)`：先在每个序列内部对有效 token 平均，再对序列平均。没有有效 token 的序列不参加平均。

### 练习 3.5　softmax 交叉熵的梯度（必做）

实现 `softmax_ce_grad(logits, targets)`，返回 (P − Y)/N。要求：不构造 one-hot 矩阵；不修改传进来的 logits。

### 练习 3.6　向量化的 softmax 分类器损失（必做）

实现 `softmax_loss_vectorized(W, X, y, reg)`，返回 `(loss, dW)`。不许用循环，不许用 autograd。

### 练习 3.7　数值梯度（必做）

实现 `numerical_gradient(f, x, h=1e-5)`，用中心差分。要求：函数返回前把 x 恢复成原值。

**检查：**

```bash
python -m pytest -q book/part1_foundations/ch03_softmax_ce
```

测试会做这些检查：天气例子的数值；和 PyTorch 在各种 dim 上对拍；fp32 和 fp16 的大 logits 不出 nan；log_softmax 不下溢；(B, T, V) 加 ignore_index 和 `F.cross_entropy` 对拍；被忽略位置梯度为 0；初始 loss = ln C；按序列平均的例子（1.00 对 1.75）；p − y 的数值和 autograd 对拍；分类器的 loss 和 dW 和 autograd 对拍；数值梯度检查。

### 练习 P1　训练你自己的分类器（必做）

不看 `tour.py`，自己写一个脚本：

1. 生成 3 类二维点云（每类 100 个点，中心随意，标准差 0.8），再生成同样分布的验证集。
2. 补一列 1 做偏置。用 float64 和一个小的随机 W 做梯度检查，相对误差应在 1e-7 以下。
3. 全批量梯度下降训练 300 步，每 50 步打印训练 loss 和正确率。
4. 检查第 0 步的 loss 是否接近 ln 3。
5. 报告验证集的 loss 和正确率。

把结果记在实验手册的记录表里。学习率换成 0.01、1.0、10 各跑一次，记录 loss 曲线有什么不同。

### 练习 P2　和 F.cross_entropy 对拍（必做）

写一个测试函数，随机生成 50 组 (B, T, V)（B、T、V 都在 1 到 20 之间随机取），随机把一部分 targets 设成 −100（但每组至少留一个有效位置），检查你的 `cross_entropy` 和 `F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))` 的差小于 1e-5。再对 `reduction="sum"` 和 `"none"` 各做一次。

### 练习 P3　循环版本（选做）

按正文 3.9.3 节，写 `softmax_loss_naive(W, X, y, reg)`（两层循环）。验证它和向量化版本的 loss 与 dW 一致（float64 下差小于 1e-10），再在 N = 2000、D = 100、C = 10 上比较两者的用时。

### 练习 P4　log-sum-exp（选做）

写 `logsumexp(z, dim)`，要求数值稳定，并和 `torch.logsumexp` 对拍。然后用它改写你的 `log_softmax`。思考：如果某一行全是 −inf（比如被 mask 掉了），你的 logsumexp 会返回什么？`torch.logsumexp` 呢？

### 练习 P5　标签平滑（选做）

给你的 `cross_entropy` 加一个 `label_smoothing=0.0` 参数，含义和 `F.cross_entropy` 相同：目标分布是 (1 − ε)·onehot + ε/V。和 `F.cross_entropy(..., label_smoothing=ε)` 对拍。注意被忽略的位置仍然要跳过。再验证：这时的梯度是 p − y'。

---

## 四、调试题

下面每段代码都有一个错误。说出：(1) 错在哪里；(2) 运行时会看到什么现象（报错，还是不报错但结果不对）；(3) 怎样改；(4) 写一个能抓住这个错误的测试。

**D1.**

```python
def softmax(z, dim=-1):
    m = z.max()
    e = (z - m).exp()
    return e / e.sum(dim, keepdim=True)
```

**D2.**

```python
def log_softmax(z, dim=-1):
    return softmax(z, dim).log()
```

**D3.**

```python
logits = model(x)                       # (B, C)
loss = F.cross_entropy(logits.softmax(-1), y)
```

**D4.**

```python
logits = lm(inp)                        # (B, T, V)
loss = F.cross_entropy(logits, tgt)     # tgt: (B, T)
```

**D5.**

```python
def cross_entropy(logits, targets, ignore_index=-100):
    logp = log_softmax(logits)
    nll = -logp.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    nll = nll * (targets != ignore_index)
    return nll.mean()
```

**D6.**

```python
tgt = inp.clone()                       # 补齐位置是 pad_id = 0
loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))
```

**D7.**

```python
def softmax_ce_grad(logits, targets):
    p = softmax(logits)
    p[torch.arange(len(targets)), targets] -= 1
    return p
```

（提示：这段代码有一个问题。）

---

## 五、闭卷手写

限时 30 分钟，不看任何资料，在纸上或空白文件里写出：

1. 稳定的 `softmax` 和 `log_softmax`；
2. 支持 `(B, T, V)` 和 `ignore_index` 的 `cross_entropy`（只要 `"mean"`）；
3. 推导 ∂L/∂z = p − y（任选 3.6.2 或 3.6.3 节的一种方法），以及 dW = Xᵀ(P − Y)/N 的形状检查。

写完后：

- 把代码放进挖空版里，运行 `python -m pytest -q book/part1_foundations/ch03_softmax_ce` 检查。
- 对每一行，口头说出张量的形状。
- 不看书，说出 fp32 和 fp16 下 e^z 溢出的阈值大约是多少，以及 3 类、10 类问题的初始 loss。

---

## 答案与提示

### 概念题要点

- **C1.** logits 是任意实数，概率非负且和为 1。p_i / p_j = e^{z_i − z_j}，logits 的差是概率之比的对数。加 5：不变（平移不变）。乘 5：分布变尖，更接近 argmax。
- **C2.** 直接除以总和：logits 有负数时得到负的"概率"，总和为 0 时除以 0，例如 [2, −1, 0] → [2, −1, 0]。先平方：破坏顺序，−3 和 3 得到相同概率。指数永远为正、严格递增、把差变成比，和 ln 配合能干净地抵消。
- **C3.** fp32 约 88.72，fp16 约 11.09。e^z 溢出成 inf，inf/inf 得到 nan，有限数/inf 得到 0。nan 和其他数做加减乘除一般还是 nan：在朴素 softmax 的溢出路径上，nan 会流进 loss，反向传播时经过这条路径的梯度也会变成 nan，相关参数更新一步后就坏了。但不能反过来说"loss 是 nan，所有梯度必然是 nan"：梯度取决于具体运算和依赖路径，比如全部位置被忽略、`reduction="mean"` 时 loss 是 nan，对 logits 的梯度却全是 0（正文 3.7.3 节）。
- **C4.** 减完后所有指数 ≤ 0，e^{…} ≤ 1，不会上溢；最大那一项是 e^0 = 1，分母 ≥ 1。不改变梯度：softmax(z − m) 和 m 无关，对 m 的导数为 0。
- **C5.** 稳定 softmax 仍然可能把很小的概率下溢成 0，ln 0 = −inf，梯度为 nan。例：z = [0, −200]，p_1 ≈ 1.4e-87 下溢为 0。log_softmax 直接算 (z − m) − ln Σ e^{z − m}，不经过小概率。
- **C6.** 似然（连乘）→ 对数似然（取 ln，单调递增）→ 负对数似然（取负，最大化变成最小化）→ 平均（除以常数 N）。"独立"用在第一步：联合概率等于各样本概率的连乘。
- **C7.** loss 是 −ln 0.9 = 0.1054，交叉熵评价的是"给正确答案多少把握"，不是"答没答对"。只有 p = 1 时为 0。在精确的实数运算里，只要类别数大于 1、logits 都是有限数，softmax 给不出恰好为 1 的概率，所以 loss 大于 0。浮点数是例外：fp32 下 softmax([0, −200]) 舍入成 [1, 0]，正确类别是 0 时，本章实现算出的交叉熵恰好是 0（本项目观察，tour 第 2 部分）。
- **C8.** H(q, p) = H(q) + KL(q ‖ p)。q 来自数据，H(q) 是常数，所以最小化交叉熵就是最小化 KL。语言模型的真实分布本身有不确定性（同样的前文后面可以接不同的词），条件熵 H(q) > 0，所以**真实分布下的期望** loss 下限大于 0，验证 loss 估计的就是它。有限训练集上的经验 loss 不受这个下限约束：模型可以背下训练语料，让训练 loss 接近 0。
- **C9.** 恰当评分规则：报告真实信念时期望罚分最小。线性罚分的期望是 1 − q·r，把 r 全押在 q 最大的那一类上最小（正文 3.4.6 节：0.3110 < 0.4600）。−ln 罚分的期望是 H(q, r)，在 r = q 时最小。用 −ln 训练的模型被鼓励报告真实的概率，而不是只猜对。
- **C10.** "你给的减去应该给的"。正确类别 p_c − 1 < 0，沿负梯度更新时 logit 变大；错误类别 p_k > 0，logit 变小，模型给它的概率越大压得越多。Σ p = Σ y = 1，所以和为 0；这对应平移不变性。
- **C11.** 均方误差的梯度里乘着 softmax 的雅可比，模型很自信时（p 接近 one-hot）这些因子接近 0，梯度消失，自信地犯错时反而学不动（正文 3.6.6 节：z = [−10, 10, 0] 时均方误差的梯度约 1e-4，交叉熵是 ±1）。交叉熵的 −1/p 恰好抵消了这个因子。
- **C12.** 不贡献 loss，不计入分母。只做第一件：loss 被稀释，偏小，而且随 batch 里补齐的多少而变化。乘以 0 时如果被忽略位置的 nll 是 inf，inf × 0 = nan；`where` 直接选 0。
- **C13.** 按 token 平均：所有有效 token 权重相同，像"全年级平均分"；按序列平均：每个序列权重相同，像"先算班级平均，再对班级平均"，短序列里的 token 权重更大。预训练和多数 SFT 用按 token 平均（最大似然的定义）；希望长短样本同等重要时用按序列平均。关键是前后一致并写清楚。
- **C14.** 初始化小时 logits ≈ 0，softmax 均匀，loss = −ln(1/C)。远大于：初始化太大、logits 被放大、偏置不为 0、加载了旧分类头。明显小于：标签泄漏进输入、分母算错（包含了补齐位置）、类别不均衡且偏置已学到频率。
- **C15.** 截断误差 ∝ h²（中心差分），舍入误差 ∝ ε/h，总误差在 h ≈ ε^{1/3} 附近最小。float64 的 ε ≈ 2.2e-16，所以 h ≈ 1e-5；float32 的 ε ≈ 1.2e-7，最好也只有约 1e-4 的误差，不够用来判断梯度对不对。h 太大：截断误差大；h 太小：舍入误差大，float32 下甚至 x + h 和 x 相同，数值梯度为 0。
- **C16.** 这一步的 V 分类交叉熵，V 是词表大小。交叉熵在 teacher forcing 下计算（给定正确的前文），生成时模型看到自己的输出，错误会累积；而且每个 token 99% 正确，30 个 token 的公式完全正确也只有约 74%。

### 手算题答案

- **H1.** (a) e^z = [2.7183, 7.3891, 20.0855]，和 30.1929，p = [0.0900, 0.2447, 0.6652]。(b) 不变，平移不变性。(c) softmax([2, 4, 6]) = [0.0159, 0.1173, 0.8668]，变尖了。
- **H2.** (a) y = 0：−ln 0.0900 = 2.4076；y = 2：−ln 0.6652 = 0.4076。(b) p − y = [−0.9100, 0.2447, 0.6652]。(c) z' = [1.9100, 1.7553, 2.3348]；z'_0 变大了，z'_2 变小了，而且 z'_0 和最大值之间的差距从 2 缩小到约 0.42，所以 p_0 一定变大。（精确值：从 0.0900 变成 0.2953。）
- **H3.** m = 103，z − m = [−3, −2, 0]，e^{z−m} = [0.0498, 0.1353, 1.0000]，和 1.1851，p = [0.0420, 0.1142, 0.8438]；ln 1.1851 = 0.1698，log_softmax = [−3.1698, −2.1698, −0.1698]。
- **H4.** (a) L(θ) = θ³(1 − θ)⁹，平均 NLL = −(3 ln θ + 9 ln(1 − θ))/12。(b) 3/θ − 9/(1 − θ) = 0 ⟹ θ* = 3/12 = 0.25。(c) θ* 处：(3 × 1.3863 + 9 × 0.2877)/12 = 0.5623；θ = 0.5 时 ln 2 = 0.6931。
- **H5.** (a) KL(q ‖ p) = 0.0052，KL(p ‖ q) = 0.0054，很接近但不相等。(b) KL(q ‖ p) = 0.5 ln(0.5/0.9) + 0.5 ln(0.5/0.1) = 0.5108；KL(p ‖ q) = 0.9 ln(0.9/0.5) + 0.1 ln(0.1/0.5) = 0.3681。不相等，KL 不是距离。
- **H6.** (a) (1 + 2 + 3 + 0.5 + 2 + 2)/6 = 10.5/6 = 1.75。(b) (2.0 + 0.5 + 2.0)/3 = 1.5。(c) 10.5/9 = 1.1667。(d) 按 token：1/6；按序列：1/(3 × 1) = 1/3。
- **H7.** (a) ln 32000 = 10.3735。(b) ln 1000 = 6.9078，0.7 远小于它：先查标签是否泄漏进输入（例如标签被当成了一个特征）、loss 的分母是否包含了大量被忽略的位置、是否在用训练过的权重而不是随机初始化。(c) H = −0.5 ln 0.5 − 0.5 ln(0.5/9) = 1.7918。
- **H8.** (a) scores = X W = 0，形状 (2, 2)；P = [[0.5, 0.5], [0.5, 0.5]]；loss = ln 2 = 0.6931。(b) P − Y = [[−0.5, 0.5], [0.5, −0.5]]，dS = [[−0.25, 0.25], [0.25, −0.25]]，形状 (2, 2)；Xᵀ = [[1, 0], [2, 1]]，形状 (2, 2)；dW = Xᵀ dS = [[−0.25, 0.25], [−0.25, 0.25]]，形状 (D, C) = (2, 2)。
- **H9.** −ln(e/(e + 4)) = ln(6.7183/2.7183) = 0.9048。
- **H10.** (a) e^11 ≈ 59874 < 65504，不溢出；e^12 ≈ 162755 > 65504，溢出成 inf，得到 [nan, 0]。(b) 0.99³⁰ ≈ 0.740；0.999³⁰ ≈ 0.970。

### 编程题提示

- **3.1** `m = z.amax(dim, keepdim=True)`，然后 `(z - m).exp()`。别用 `z.max()`（全局最大值，见 D1）。fp16/bf16 输入先 `zf = z.float()` 再算，最后 `.to(z.dtype)` 转回去。
- **3.2** `s = z - m; return s - s.exp().sum(dim, keepdim=True).log()`；低精度输入同样先转 fp32。
- **3.3** 先 `valid = targets != ignore_index`；用 `torch.where(valid, targets, 0)` 得到安全的下标；`gather(-1, idx.unsqueeze(-1)).squeeze(-1)`；用 `where` 把被忽略的位置清零；mean 除以 `valid.sum()`。
- **3.4** 调用 `cross_entropy(..., reduction="none")` 得到 (B, T)；每行求和除以每行的有效个数；用布尔下标去掉有效个数为 0 的行。
- **3.5** `g = softmax(logits).clone()`；`g[torch.arange(N), targets] -= 1`；`return g / N`。`softmax` 返回的是新张量，严格说不 clone 也不会改到 logits，但写上 clone 是好习惯。
- **3.6** 正文 3.9.3 节，四行。
- **3.7** 正文 3.9.4 节。注意用 `view(-1)` 得到和 x 共享内存的一维视图，改它就是改 x。
- **P2** `torch.randint` 取形状；`mask = torch.rand(B, T) < 0.3; tgt[mask] = -100`；如果全被 mask，就把 `tgt[0, 0]` 恢复成一个合法值。
- **P4** 全是 −inf 时 m = −inf，z − m = −inf − (−inf) = nan。`torch.logsumexp` 对这种情况返回 −inf（本项目观察，torch 2.14.1，CPU，tour 第 2 部分）。可以把 m 中的 −inf 替换成 0 再算。
- **P5** 目标分布 y' 和 log_softmax 逐元素相乘再求和取负；被忽略的位置照样清零、不计分母。

### 调试题答案

- **D1.** `z.max()` 取的是整个张量的最大值，不是每一行的最大值。单个样本时没问题；batch 里如果某一行的 logits 都远小于另一行（比如一行是 [1000, …]，另一行是 [0, …]），第二行减完后全是 −1000 左右，全部下溢成 0，得到 0/0 = nan。不报错，只在某些 batch 出现 nan。改成 `z.amax(dim, keepdim=True)`。测试：`softmax(torch.tensor([[1000., 999.], [0., -1.]]))` 不应有 nan。
- **D2.** 先 softmax 再 log，小概率下溢成 0，得到 −inf，梯度 nan。前向不报错。改成正文 3.3.5 节的写法。测试：`log_softmax(torch.tensor([0., -200.]))` 应为有限值。
- **D3.** `F.cross_entropy` 内部已经做了 log_softmax，再传 softmax 的结果等于做了两次。不报错，loss 降得慢且有下限（C 类时 −ln(e/(e + C − 1))）。改成直接传 logits。测试：模型输出 one-hot 方向很大的 logits 时，loss 应接近 0。
- **D4.** `F.cross_entropy` 要求类别在第 1 维。T ≠ V 时报形状错误；T = V 时不报错，但在"对位置做 softmax"。改成 `F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))`。测试：和逐位置手算的结果对拍；特意用 T = V 的输入测。
- **D5.** 两个问题：最后用 `mean()` 除以了全部位置（包括被忽略的），loss 被稀释；用乘法清零，被忽略位置的 nll 若为 inf 会得到 nan。改成 `torch.where` 清零，再除以 `(targets != ignore_index).sum()`。测试：和 `F.cross_entropy(..., ignore_index=-100)` 对拍。
- **D6.** 两个问题：目标没有错开一位（模型在学"复制当前输入"，而不是预测下一个），loss 会低得不正常；补齐位置用的是 pad_id = 0，而 `F.cross_entropy` 默认的 ignore_index 是 −100，所以补齐位置都参与了 loss。改成：目标 = 输入左移一位并在末尾补 `<eos>`，补齐位置设成 −100（正文 3.11.2 节）。测试：检查 `tgt[:, :-1]` 和 `inp[:, 1:]` 在有效位置上相等；检查补齐位置的 tgt 都是 −100。
- **D7.** 没有除以 N，梯度大了 N 倍。本章的 `softmax` 返回新张量，所以对 `p` 原地减 1 不会改到 logits，这不算错误（写上 `.clone()` 也无妨，但它救不了一个本身就原地修改输入的 softmax）。改成 `return p / len(targets)`。测试：和 `F.cross_entropy` 的 autograd 梯度对拍（N > 1）。
