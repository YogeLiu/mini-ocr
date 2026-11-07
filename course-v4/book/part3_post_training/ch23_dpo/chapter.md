# 第 23 章 偏好优化与 DPO

## 本章导读

到第 22 章为止，我们教模型的方式只有一种：给它一个标准答案，让它照着写。这叫 SFT。第 21 章的 TRL LoRA 在 val 500 条上 EM_norm 是 0.678（本项目观察，第 22 章 `analysis.json`），已经比照抄、比提示词强得多。

但 SFT 有一个盲点：**它从来没见过自己犯的错。**训练时它只看标准答案；推理时它写出来的东西，对也好错也好，没人告诉它"这个比那个好"。

这一章换一种教法：让模型自己写几个答案，我们告诉它"这个比那个好"，它从比较里学。这就是**偏好优化**。我们要把它最流行的一种算法 **DPO**（Direct Preference Optimization，直接偏好优化）从头推导、从零实现，再用 TRL 在 Qwen3-0.6B 上真跑一次。

本章的路线：

1. 23.1 为什么要从"比较"里学。
2. 23.2 RLHF 的三步：SFT → 奖励模型 → 强化学习。DPO 就是为了绕开后两步的麻烦而生的。
3. 23.3 Bradley-Terry 模型：怎样把"A 比 B 好"变成一个概率。
4. 23.4 从"带 KL 约束的奖励最大化"推出最优策略的闭式解。
5. 23.5 把闭式解代入 Bradley-Terry，得到 DPO 损失；手算一对样本。
6. 23.6 从零实现 DPO 损失，对照 `miniocr/train/preference.py` 逐行讲。
7. 23.7 偏好数据：怎样从后纠错的候选里自动构造，以及噪声、矛盾、长度偏好。
8. 23.8 实验：TRL `DPOTrainer` 与 β 扫描。
9. 23.9 IPO、KTO 等变体（概括）。
10. 23.10 到 23.12 和项目的关系、常见错误、和真实框架的对应。

本章的两个结论，先放在这里：

- **DPO 的损失只是一个二分类的交叉熵**：它的"分数"是策略相对参考模型的对数概率比，乘上 β。推导的关键一步是：奖励最大化问题有闭式解，所以奖励可以用策略本身表示出来（数学结论，23.4、23.5 节）。
- **β 太小，模型会跑偏**：本项目的 β 扫描里，β = 0.05 时 val EM_norm 从 SFT 的 0.678 掉到 0.414，chosen 的 log-prob 也被一起压低了；β = 1.0 时 EM_norm 是 0.706，但这个数是在 val 上挑出来的，还没有 test 确认（本项目观察，23.8 节；选择偏差见 23.8.5 节）。

**本章用的面板。**本章用第 0 章 0.9.3 节定义的文本后纠错面板：EM_norm、CER、改坏、fix_rate，四项都在 23.8.3 节，那里的"修好"就是 fix_rate。可选诊断项 parse_rate 本章没有列。

和 TRL 的对拍只做了一部分，做到了哪些、哪些还没验证，见 23.6.5 节。

先修回顾：

> **回顾**：交叉熵与 sigmoid（第 3 章）、序列的对数概率是逐 token 对数概率之和（第 7 章）、SFT 的 label mask（第 19 章）、LoRA 与 TRL 检查清单（第 20、21 章）、bootstrap 与配对差、"val 做决定、test 只报告"（第 22 章）。KL 散度只用到定义 $\mathrm{KL}(p\|q)=\sum_y p(y)\log\frac{p(y)}{q(y)}\ge 0$，等号当且仅当 $p=q$。

---

## 23.1 为什么要从"比较"里学

### 23.1.1 SFT 只教"对的长什么样"

SFT 只给正例，偏好优化同时给正例和反例。

回忆第 19 章：SFT 的损失是标准答案每个 token 的负对数概率。它把标准答案的概率往上推。至于模型自己最容易写错成什么样子，SFT 一概不管，反正那些错答案从来不出现在训练数据里。

这就像学开车只看示范视频。你看了一百遍完美的倒车入库，但从来没人在你自己蹭到柱子的时候说一句"刚才那把方向打早了"。

偏好优化补上的就是这句话。它的训练数据长这样：

- 一个提示 $x$（带噪的 OCR 公式）；
- 两个回答 $y_w$ 和 $y_l$，下标 w 是 win（更好，也叫 chosen），l 是 lose（更差，也叫 rejected）。

两个回答都可以是**模型自己写的**。模型从自己的输出里学，学到的就是它真会犯的错。

### 23.1.2 比较比打分容易

在很多任务里，写出唯一正确的答案很难，给答案打绝对分也很难，但**比较两个答案**容易得多。

- 让人写一首好诗：难。让人给一首诗打 7.3 分还是 7.5 分：标准飘忽。问"这两首哪首好"：大多数人能答，而且答得比较一致。
- 对聊天助手，"有帮助、无害"没有标准答案，但两个回复放在一起，标注员能挑出更好的那个。

我们的后纠错任务其实有标准答案，比较可以**自动完成**：哪个候选离标准答案更近，哪个就更好。所以本章不需要人工标注。这是本项目的一个便利，23.7 节会说它也有代价。

### 23.1.3 本章要回答的问题

有了"A 比 B 好"这样的数据，怎样更新模型？经典答案是 RLHF：先训练一个奖励模型，再用强化学习。DPO 的答案是：不用奖励模型，也不用强化学习，直接用一个分类损失。下面一步一步推导这两者为什么等价。

---

## 23.2 RLHF 的三步

RLHF（Reinforcement Learning from Human Feedback，从人类反馈中强化学习）是 InstructGPT（Ouyang et al., 2022）让大模型"听话"的方法。它分三步。

### 23.2.1 第一步：SFT

用人写的示范回答做监督微调，得到 $\pi^{\mathrm{SFT}}$。这就是第 19 到 21 章做的事。SFT 模型有两个用途：它是后面强化学习的起点，也是后面的**参考模型** $\pi_{\mathrm{ref}}$。

### 23.2.2 第二步：训练奖励模型

对同一个提示，让 SFT 模型生成几个回答，请人排序，拆成两两的偏好对。然后训练一个**奖励模型** $r_\phi(x,y)$：输入提示和回答，输出一个实数分数。训练目标是让"更好"的回答分数更高。分数怎样变成可训练的损失，就是 23.3 节的 Bradley-Terry 模型。

奖励模型通常就是一个语言模型，把最后的词表头换成一个输出标量的线性层。

### 23.2.3 第三步：带 KL 约束的强化学习

固定奖励模型，用强化学习（InstructGPT 用的是 PPO）更新策略 $\pi_\theta$，目标是：

$$\max_{\pi_\theta}\ \mathbb{E}_{x\sim\mathcal D,\ y\sim\pi_\theta(\cdot\mid x)}\big[r_\phi(x,y)\big]-\beta\,\mathbb{E}_{x\sim\mathcal D}\Big[\mathrm{KL}\big(\pi_\theta(\cdot\mid x)\,\big\|\,\pi_{\mathrm{ref}}(\cdot\mid x)\big)\Big]$$

两项的意思：

- 第一项：**多拿奖励**。
- 第二项：**别离参考模型太远**。可以把它想成一根把策略拴在参考模型旁边的绳子，$\beta>0$ 控制这根绳子有多紧。

为什么要第二项？因为奖励模型只是一个近似。它在 SFT 模型常写的那类回答上训练过，换到奇怪的回答上可能乱打高分。没有绳子，策略会去钻奖励模型的空子（这叫 reward hacking，第 25 章还会遇到），同时把语言能力也弄坏。

### 23.2.4 三步的代价

第三步很贵，也很难调：

- 训练时显存里要同时有**策略、参考模型、奖励模型**，PPO 还要一个**价值模型**。
- 每一步都要先**采样**（生成文本），再打分，再更新。生成是最慢的环节。
- PPO 有一堆超参数，训练不稳定是出了名的。

DPO 论文（Rafailov et al., 2023）的观察是：第三步的目标函数有**闭式最优解**。利用它，第二步和第三步可以合成一个监督学习问题。下面三节就是这件事的完整推导。强化学习本身留到第 24 章。

---

## 23.3 Bradley-Terry：把偏好变成概率

### 23.3.1 直觉：棋手的等级分

**Bradley-Terry 模型**假设每个选手有一个实力分，A 赢 B 的概率只取决于两人的分差。

国际象棋的 Elo 等级分就是这个思路。你 1800 分，我 1600 分，你赢我的概率大约 76%（Elo 的标准公式，数学结论）。我们俩各加 100 分，这个概率不变，只看差。

把"选手"换成"回答"，"实力分"换成"奖励"，就是 RLHF 的奖励模型。

### 23.3.2 模型

给定提示 $x$ 和两个回答 $y_1,y_2$，Bradley-Terry 模型说：

$$P(y_1\succ y_2\mid x)=\frac{\exp r(x,y_1)}{\exp r(x,y_1)+\exp r(x,y_2)}$$

分子分母同时除以 $\exp r(x,y_1)$：

$$P(y_1\succ y_2\mid x)=\frac{1}{1+\exp\big(-(r(x,y_1)-r(x,y_2))\big)}=\sigma\big(r(x,y_1)-r(x,y_2)\big)$$

其中 $\sigma(z)=1/(1+e^{-z})$ 是 sigmoid。这一步是纯代数（数学结论）。

所以偏好概率就是分差的 sigmoid。分差为 0，概率是 0.5；分差越大，越接近 1。

### 23.3.3 训练奖励模型就是二分类

有一批偏好数据 $\{(x,y_w,y_l)\}$，每条都说"$y_w$ 比 $y_l$ 好"。用最大似然估计奖励模型 $r_\phi$：最大化 $\prod P(y_w\succ y_l\mid x)$。取对数把连乘变成求和，再加负号把最大化变成最小化，就得到等价的负对数似然：

$$\mathcal L_{\mathrm{BT}}(\phi)=-\mathbb E_{(x,y_w,y_l)}\Big[\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)\Big]$$

这是第 3 章交叉熵在两类时的形式：logit 是分差，标签永远是 1（"$y_w$ 赢"）。

### 23.3.4 手算

设奖励模型给两个回答打分 $s_A=2.0$、$s_B=0.5$。

- 分差：$2.0-0.5=1.5$
- $P(A\succ B)=\sigma(1.5)=1/(1+e^{-1.5})=1/(1+0.2231)=0.8176$
- 若数据说 A 赢，损失 $=-\log 0.8176=0.2014$

若两个分数相等，$P=\sigma(0)=0.5$，损失 $=\log 2\approx 0.6931$。（以上为数学结论；`test_bradley_terry_zero_margin_is_log2` 检查的正是最后这个值。）

记住 0.6931 这个数。后面 DPO 训练日志的第一步，loss 就是它。

### 23.3.5 一个后面要用的性质：只看分差

如果对同一个提示下所有回答的奖励都加上同一个常数 $c(x)$：

$$\sigma\big((r(x,y_w)+c(x))-(r(x,y_l)+c(x))\big)=\sigma\big(r(x,y_w)-r(x,y_l)\big)$$

概率不变（数学结论）。也就是说，Bradley-Terry 看不见只依赖于提示、不依赖于回答的那部分奖励。23.5 节推导 DPO 时，一个算不出来的配分函数就靠这个性质消掉。

> **停下来想一想**：Bradley-Terry 只用分差，那奖励模型输出的绝对数值有意义吗？如果你拿到一个奖励模型，它对所有回答都打 100 分左右，这说明它坏了吗？

---

## 23.4 从 KL 约束目标推出最优策略

这一节只做一件事：解出 23.2.3 节那个优化问题的最优策略。结论很短，推导需要耐心。

### 23.4.1 先固定一个提示

目标对每个 $x$ 是分开的（不同提示之间没有耦合），所以只看一个提示 $x$。为了少写字，下面把 $x$ 省掉，写 $\pi(y)$、$\pi_{\mathrm{ref}}(y)$、$r(y)$。回答 $y$ 取值于所有可能的文本，它是一个离散集合（很大，但离散）。

把 23.2.3 节的期望和 KL 都按定义写成对 $y$ 的求和（期望是按 $\pi(y)$ 加权求和，KL 用回顾里的定义），要最大化的是：

$$J(\pi)=\sum_y\pi(y)\,r(y)-\beta\sum_y\pi(y)\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}$$

约束：$\pi(y)\ge 0$，$\sum_y\pi(y)=1$。

### 23.4.2 一步一步推

**第 1 步：把两项合到一个求和里。**

$$J(\pi)=\sum_y\pi(y)\Big[r(y)-\beta\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}\Big]$$

**第 2 步：提出 $-\beta$。**

$$J(\pi)=-\beta\sum_y\pi(y)\Big[\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}-\frac{r(y)}{\beta}\Big]$$

**第 3 步：把 $r/\beta$ 也写成对数。**因为 $\frac{r(y)}{\beta}=\log\exp\frac{r(y)}{\beta}$，方括号就成了两个对数相减，再用 $\log a-\log b=\log\frac{a}{b}$ 并成一个：

$$J(\pi)=-\beta\sum_y\pi(y)\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)\exp\big(r(y)/\beta\big)}$$

**第 4 步：定义一个新分布。**现在的式子已经很像 $-\beta\,\mathrm{KL}$ 了：$\sum_y\pi(y)\log\frac{\pi(y)}{\text{某个东西}}$。差的一点是，KL 的分母必须是一个概率分布。分母 $\pi_{\mathrm{ref}}(y)\exp(r(y)/\beta)$ 全是正数，但加起来不一定是 1。把它归一化：

$$Z=\sum_y\pi_{\mathrm{ref}}(y)\exp\big(r(y)/\beta\big),\qquad \pi^*(y)=\frac{1}{Z}\,\pi_{\mathrm{ref}}(y)\exp\big(r(y)/\beta\big)$$

$Z$ 叫**配分函数**（partition function），它就是所有权重的总和，除以它是为了让 $\pi^*$ 加起来等于 1。回答的集合可以无限大，这里假设 $Z$ 有限，比如奖励有上界时就成立。它只依赖于提示（和 $\pi_{\mathrm{ref}}$、$r$、$\beta$），**不依赖于我们要优化的 $\pi$**。

**第 5 步：代回去。**由定义，$\pi_{\mathrm{ref}}(y)\exp(r(y)/\beta)=Z\,\pi^*(y)$。代进第 3 步，再用 $\log\frac{\pi}{Z\pi^*}=\log\frac{\pi}{\pi^*}-\log Z$ 把 $Z$ 拆出来：

$$J(\pi)=-\beta\sum_y\pi(y)\log\frac{\pi(y)}{Z\,\pi^*(y)}=-\beta\sum_y\pi(y)\log\frac{\pi(y)}{\pi^*(y)}+\beta\log Z\sum_y\pi(y)$$

最后一项里 $\sum_y\pi(y)=1$，于是

$$J(\pi)=-\beta\,\mathrm{KL}(\pi\,\|\,\pi^*)+\beta\log Z$$

**第 6 步：读出最优解。**$\beta\log Z$ 和 $\pi$ 无关。$\mathrm{KL}\ge 0$，当且仅当 $\pi=\pi^*$ 时为 0。因为前面乘的是 $-\beta<0$，所以 $J$ 在 $\pi=\pi^*$ 时取最大值，最大值是 $\beta\log Z$。$\pi^*$ 本身非负、加起来是 1，满足约束，所以这个最大值确实取得到。

把 $x$ 写回来，结论是（数学结论，DPO 论文 §4）：

$$\boxed{\ \pi^*(y\mid x)=\frac{1}{Z(x)}\,\pi_{\mathrm{ref}}(y\mid x)\exp\Big(\frac{1}{\beta}r(x,y)\Big)\ }$$

也就是说，最优策略是参考模型乘上奖励的指数加权，再归一化。

### 23.4.3 小例子：三个候选

假设某个提示只有三种可能的回答，参考模型的概率和奖励如下：

| 回答 | $\pi_{\mathrm{ref}}$ | $r$ |
|---|---|---|
| $y_1$ | 0.5 | 0 |
| $y_2$ | 0.3 | 1 |
| $y_3$ | 0.2 | 2 |

参考模型最喜欢 $y_1$，奖励最喜欢 $y_3$。最优策略会怎样折中？按公式算 $\pi_{\mathrm{ref}}\cdot e^{r/\beta}$，再除以总和：

**β = 1：**权重 $0.5\times e^0=0.5$，$0.3\times e^1=0.8155$，$0.2\times e^2=1.4778$；$Z=2.7933$；
$\pi^*=(0.179,\ 0.292,\ 0.529)$。

**β = 10：**权重 $0.5$，$0.3\times e^{0.1}=0.3316$，$0.2\times e^{0.2}=0.2443$；$Z=1.0758$；
$\pi^*=(0.465,\ 0.308,\ 0.227)$，几乎还是参考模型。

**β = 0.1：**权重 $0.5$，$0.3\times e^{10}\approx 6.6\times10^3$，$0.2\times e^{20}\approx 9.7\times10^7$；
$\pi^*\approx(0.000,\ 0.000,\ 1.000)$，全押在奖励最高的回答上。

（以上全部为数学结论。）

### 23.4.4 β 和参考模型各管什么

从这个例子能直接读出两个旋钮的作用：

- **β 是温度。**β 大，$e^{r/\beta}$ 都接近 1，最优策略贴着参考模型；β 小，奖励的差异被放大，最优策略变得贪心。
- **参考模型是先验。**$\pi_{\mathrm{ref}}(y)=0$ 的回答，乘上多大的 $e^{r/\beta}$ 都还是 0。参考模型认为"不可能"的东西，最优策略也不会去。所以参考模型一般选 SFT 模型：它已经会说"像样的话"，偏好优化只在像样的话之间重新分配概率。

> **停下来想一想**：如果奖励模型对某个乱码回答错误地打了很高的分，β 小和 β 大，哪种情况下最优策略更容易被它骗？

---

## 23.5 DPO 损失

### 23.5.1 把奖励反解出来

23.4 节的公式是"已知奖励，求最优策略"。DPO 把它倒过来用：已知策略，奖励是什么？

对 $\pi^*=\frac{1}{Z}\pi_{\mathrm{ref}}e^{r/\beta}$ 两边取对数：

$$\log\pi^*(y\mid x)=\log\pi_{\mathrm{ref}}(y\mid x)+\frac{1}{\beta}r(x,y)-\log Z(x)$$

把 $r$ 单独移到一边，乘以 β：

$$r(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+\beta\log Z(x)$$

（数学结论。）奖励被写成了策略相对参考模型的对数概率比，加上一个**只依赖于提示**的项。

麻烦在于 $Z(x)$：它要对所有可能的文本求和，根本算不出来。

### 23.5.2 代入 Bradley-Terry，Z 消掉了

还记得 23.3.5 节的性质吗？Bradley-Terry 只看同一个提示下两个回答的**分差**。把上式代入：

$$
\begin{aligned}
P(y_w\succ y_l\mid x)&=\sigma\big(r(x,y_w)-r(x,y_l)\big)\\
&=\sigma\Big(\beta\log\frac{\pi^*(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}+\beta\log Z(x)-\beta\log\frac{\pi^*(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}-\beta\log Z(x)\Big)\\
&=\sigma\Big(\beta\log\frac{\pi^*(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi^*(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\Big)
\end{aligned}
$$

两个 $\beta\log Z(x)$ 一正一负，抵消（数学结论）。偏好概率现在**只用策略和参考模型**就能写出来。

### 23.5.3 用参数化的策略去拟合

最后一步：我们不知道真正的 $\pi^*$，就用一个可训练的策略 $\pi_\theta$ 代替它，然后像 23.3.3 节训练奖励模型那样做最大似然。

这样替换为什么合理？反过来看：任给一个策略 $\pi_\theta$，只要它和 $\pi_{\mathrm{ref}}$ 在同一批回答上概率都大于 0，下面的对数比就有定义。用 softmax 输出的语言模型对每个回答的概率都大于 0，自然满足这一条。定义奖励 $r_\theta(x,y)=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}$。把它代进 23.4 节的闭式解，权重 $\pi_{\mathrm{ref}}e^{r_\theta/\beta}$ 恰好就是 $\pi_\theta$，加起来是 1，所以 $Z(x)=1$，最优策略就是 $\pi_\theta$ 自己。也就是说，每个策略都是某个奖励下的最优策略。在策略上做最大似然，就等于在这一族奖励里挑最符合偏好数据的那个，同时直接拿到它的最优策略：

$$\boxed{\ \mathcal L_{\mathrm{DPO}}(\theta)=-\mathbb E_{(x,y_w,y_l)\sim\mathcal D}\Big[\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\Big)\Big]\ }$$

这就是 DPO 损失（数学结论，DPO 论文 §4）。回头看它做了什么：

- 原来要两步：先用 BT 损失训奖励模型，再用 RL 找 KL 约束下的最优策略。
- 现在一步：直接用 BT 损失训策略，因为在最优点上，"策略"和"奖励"是一一对应的。论文的副标题把这件事说得很俏皮："你的语言模型其实偷偷是个奖励模型。"

有两点要说清楚：

1. 推导用的 KL 约束目标和 RLHF 第三步**是同一个**，所以 β 的含义也一样：β 越大，越贴近参考模型。
2. DPO 只在**偏好数据覆盖到的回答**上施加信号。它不采样，不知道模型在数据以外会写出什么。这是它比 PPO 便宜的原因，也是它的局限（23.7 节）。

### 23.5.4 隐式奖励与 margin

定义

$$\hat r_\theta(x,y)=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}$$

叫**隐式奖励**。DPO 损失就是 $-\log\sigma(\hat r_\theta(x,y_w)-\hat r_\theta(x,y_l))$。

- $\hat r_w-\hat r_l$ 叫 **margin**（间隔）。
- margin > 0 的比例叫 **reward accuracy**。

TRL 训练日志里的 `rewards/chosen`、`rewards/rejected`、`rewards/margins`、`rewards/accuracies` 就是这四个量的批平均（23.12 节核对了源码）。

序列的对数概率怎么算？自回归模型的 $\pi_\theta(y\mid x)=\prod_t\pi_\theta(y_t\mid x,y_{<t})$，所以

$$\log\pi_\theta(y\mid x)=\sum_{t}\log\pi_\theta(y_t\mid x,y_{<t})$$

是回答部分每个 token 的对数概率**之和**（不是平均）。提示部分不算。这和第 19 章的 label mask 是同一件事。

### 23.5.5 梯度：DPO 在推什么

令 $u=\hat r_w-\hat r_l$，损失 $L=-\log\sigma(u)$。

**第 1 步：**用 sigmoid 的导数 $\sigma'(u)=\sigma(u)(1-\sigma(u))$（第 3 章）和对称性 $1-\sigma(u)=\sigma(-u)$：$\frac{dL}{du}=-\frac{\sigma(u)(1-\sigma(u))}{\sigma(u)}=-(1-\sigma(u))=-\sigma(-u)$。

**第 2 步：**$u=\beta[\log\pi_\theta(y_w)-\log\pi_{\mathrm{ref}}(y_w)]-\beta[\log\pi_\theta(y_l)-\log\pi_{\mathrm{ref}}(y_l)]$，参考模型不参与求导，所以 $\frac{\partial u}{\partial\log\pi_\theta(y_w)}=\beta$，$\frac{\partial u}{\partial\log\pi_\theta(y_l)}=-\beta$。

**第 3 步：链式法则。**

$$\frac{\partial L}{\partial\log\pi_\theta(y_w)}=-\beta\,\sigma(-u),\qquad\frac{\partial L}{\partial\log\pi_\theta(y_l)}=+\beta\,\sigma(-u)$$

$$\nabla_\theta L=-\beta\,\sigma\big(\hat r_l-\hat r_w\big)\Big[\nabla_\theta\log\pi_\theta(y_w\mid x)-\nabla_\theta\log\pi_\theta(y_l\mid x)\Big]$$

（数学结论。）读法：

- **方向**：把 chosen 的对数概率往上推，把 rejected 的往下压。
- **力度**：$\sigma(\hat r_l-\hat r_w)$。隐式奖励排错了（$\hat r_l>\hat r_w$），这个权重大于 0.5，错得越多越接近 1，使劲推；已经排对而且拉开了，权重接近 0，几乎不管。这和 SFT 很不一样：SFT 的损失里每个样本权重相同，没有这种按排没排对来调节的系数。

注意方括号里是**差**。梯度只关心"chosen 比 rejected 高多少"，不关心两者各自的绝对值。这一点在 23.8.4 节会变成一个真实的失败现象。

### 23.5.6 手算一对样本

取一个偏好对，序列对数概率（回答部分 token 之和）如下：

| | 参考模型 $\log\pi_{\mathrm{ref}}$ | 策略 $\log\pi_\theta$ |
|---|---|---|
| chosen $y_w$ | −5.0 | ? |
| rejected $y_l$ | −6.0 | ? |

**情形 A：训练刚开始，策略 = 参考模型。**取 β = 0.1。

- 对数概率比：$\Delta_w=-5-(-5)=0$，$\Delta_l=0$
- 隐式奖励：$\hat r_w=\hat r_l=0$，margin $u=0$
- 损失：$-\log\sigma(0)=\log2=0.6931$
- 梯度：$\partial L/\partial\log\pi_\theta(y_w)=-0.1\times\sigma(0)=-0.05$；对 $y_l$ 是 $+0.05$

只要策略从参考模型出发，任何 DPO 训练的第一步 loss 都是 0.6931，与数据无关。这是一个免费的检查。

**情形 B：训练一段时间后。**策略 chosen = −3.0，rejected = −7.0。取 β = 0.5。

- $\Delta_w=-3-(-5)=+2$，$\Delta_l=-7-(-6)=-1$
- $\hat r_w=0.5\times2=1.0$，$\hat r_l=0.5\times(-1)=-0.5$
- $u=1.0-(-0.5)=1.5$；$\sigma(1.5)=0.8176$
- 损失：$-\log0.8176=0.2014$（和 23.3.4 节同一个数：分差同为 1.5）
- 梯度：$\sigma(-1.5)=0.1824$；对 $y_w$：$-0.5\times0.1824=-0.0912$；对 $y_l$：$+0.0912$

**情形 C：两个都往下掉。**策略 chosen = −6.0，rejected = −9.0，β = 0.5。

- $\Delta_w=-1$，$\Delta_l=-3$；$\hat r_w=-0.5$，$\hat r_l=-1.5$；$u=1.0$
- 损失：$-\log\sigma(1.0)=0.3133$

（以上全部为数学结论；用 `dpo_loss` 在 CPU 上算，得到同样的 0.6931、−0.05、0.2014、±0.0912、0.3133。）

再看情形 C：chosen 的序列概率变成了原来的 $e^{-1}\approx0.37$ 倍，**变小了**，但损失从 0.6931 降到了 0.3133。DPO 很满意，因为 rejected 掉得更多。损失下降不等于"好答案变得更可能"。

> **停下来想一想**：情形 B 里，如果把 β 从 0.5 换成 0.05，要让损失同样降到 0.2014，策略的对数概率比之差 $\Delta_w-\Delta_l$ 要多大？这对"策略离参考模型多远"意味着什么？

---

## 23.6 从零实现 DPO 损失

代码在 `miniocr/train/preference.py`，一共三个函数加一个辅助。`# >>> 练习 23.1` 标记的 `bradley_terry_loss` 和 `dpo_loss` 要你自己从零写出（作业 A4 机制层的一部分）。下面用 23.5.6 节的同一组数字讲。

### 23.6.1 `bradley_terry_loss`

```python
def bradley_terry_loss(score_chosen, score_rejected):
    return -F.logsigmoid(score_chosen - score_rejected)
```

- 输入：两个形状 `[B]` 的张量，每对一个分数。
- `F.logsigmoid(z)` 直接算 $\log\sigma(z)$。**不要写成 `torch.log(torch.sigmoid(z))`**：$z$ 很负时 sigmoid 下溢成 0，log 得到 −inf。`logsigmoid` 用 $\log\sigma(z)=-\mathrm{softplus}(-z)$ 计算，数值稳定。
- 输出：形状 `[B]`，逐对损失，不求平均。批平均由调用者决定。

代入 $s_A=2.0$、$s_B=0.5$：输出 0.2014。

### 23.6.2 `dpo_loss`

```python
def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    if not beta > 0:
        raise ValueError("beta 必须为正")
    for name, value in (...):
        if value.dim() != 1:
            raise ValueError(f"{name} 必须是一维逐对向量")
    reward_chosen = beta * (policy_chosen - ref_chosen)
    reward_rejected = beta * (policy_rejected - ref_rejected)
    return bradley_terry_loss(reward_chosen, reward_rejected), reward_chosen, reward_rejected
```

逐行：

- `if not beta > 0`：β ≤ 0 在推导里没有意义（KL 项变成奖励，或者除以 0）。写成 `not beta > 0` 而不是 `beta <= 0`，是为了让 NaN 也报错。测试 `test_dpo_rejects_nonpositive_beta` 检查这一条。
- 维度检查：四个输入都必须是 `[B]`。最常见的 bug 是把 token 级的 `[B, T]` 直接传进来，没有先对回答部分求和。
- `reward_chosen = beta * (policy_chosen - ref_chosen)`：隐式奖励 $\hat r_w$。情形 B：$0.5\times(-3-(-5))=1.0$。
- `reward_rejected`：$0.5\times(-7-(-6))=-0.5$。
- 最后一行：把两个隐式奖励交给 BT 损失。$-\log\sigma(1.5)=0.2014$。

DPO 就是"用隐式奖励当分数的 BT 损失"，代码和 23.5.3 节的公式一一对应。

测试 `test_dpo_hand_value_and_rewards` 用的是策略 (2, −1)、参考 (0, 0)、β = 0.5。只有差值重要，所以它和情形 B 是同一道题：$r_c=1$，$r_r=-0.5$，损失 $\log(1+e^{-1.5})=0.2014$。`test_dpo_gradient_pushes_chosen_up` 是情形 A：β = 0.1，策略 = 参考，检查 chosen 的梯度为负（实际值 −0.05）。

**为什么返回隐式奖励？**为了日志。训练时你要看 margin 和 reward accuracy，而不只是 loss（23.8.2 节）。

### 23.6.3 `response_logprob`：从 logits 到序列对数概率

```python
def response_logprob(logits, labels):
    targets = labels[:, 1:]
    mask = targets != -100
    return sequence_log_prob(logits[:, :-1].float(), targets, mask)
```

- `logits`：`[B, T, V]`；`labels`：`[B, T]`，提示和 padding 位置是 −100（第 19 章的约定）。
- **错位**：第 t 个位置的 logits 预测第 t+1 个 token。所以 logits 去掉最后一位，labels 去掉第一位，两者都变成长度 T−1。
- `mask`：只有回答 token 参与。
- `sequence_log_prob`（第 24 章 `policy_gradient.py` 里的函数，这里复用）：对词表做 `log_softmax`，用 `gather` 取出目标 token 的对数概率，把 mask 外的位置置 0，再沿时间**求和**。
- `.float()`：在 FP32 里做 log_softmax。bf16 的显式尾数只有 7 位，含隐含位共 8 位（约 2 到 3 位十进制有效数字，第 16 章 16.1 节），上百个 token 的对数概率加起来，误差会累积。

测试 `test_response_logprob_ignores_prompt_and_padding`：logits 全 0（词表 5，均匀分布），labels = `[-100, -100, 2, 1]`。错位后 targets = `[-100, 2, 1]`，两个有效 token，每个 $\log(1/5)$，总和 $2\log(1/5)=-3.219$。提示位置被正确跳过。

### 23.6.4 `dpo_step_loss`：一步训练

```python
logits = model(input_ids=..., attention_mask=..., use_cache=False).logits   # [2B, T, V]
policy = response_logprob(logits, batch['labels'])                          # [2B]
ref = ref_logprob_fn(batch)                                                 # [2B]，no_grad
pc, pr = policy[0::2], policy[1::2]                                         # [B], [B]
rc, rr = ref[0::2], ref[1::2]
loss, chosen, rejected = dpo_loss(pc, pr, rc, rr, beta=beta)
return loss.mean(), chosen.detach(), rejected.detach()
```

- **交错排列**：一个 batch 有 B 个偏好对，展开成 2B 行，偶数行是 chosen，奇数行是 rejected。一次前向同时算出两者。TRL 的做法是前 B 行 chosen、后 B 行 rejected（`logps.chunk(2, dim=0)`），排法不同，意思一样。
- **参考模型**：`ref_logprob_fn` 内部必须在 `torch.no_grad()` 下算。参考模型是固定的，它的梯度不该存在。
- `loss.mean()`：对 B 个偏好对取平均。
- 返回的奖励 `.detach()`：只用于日志，别让它们挂着计算图。

tour 部分 1 在一个 tiny 随机 Qwen3（词表 64，隐藏 32，1 层）上用一个偏好对跑三步 AdamW（学习率 0.01，β = 0.5）。参考模型是策略的一份拷贝。三步的 loss 是 0.6931 → 0.3606 → 0.1824（本项目观察，CPU，lab 第 1 部分）。第一步严格是 log 2，因为策略 = 参考。这只证明机制通了，不代表任何效果。

### 23.6.5 与 TRL 对拍：做到了什么，没做到什么

第 21 章的原则是"框架替你做的事，就是可能出错的事"，所以要对拍。本章的对拍只做了一部分：

- **做到的**：(1) 23.12 节对照 TRL 1.15.0 的源码，确认它的默认损失（`loss_type=["sigmoid"]`）就是 $-\log\sigma(\beta(\Delta_w-\Delta_l))$，序列对数概率是回答 token 之**和**，`rewards/*` 是 β 乘以对数概率比，和 `dpo_loss` 是同一个公式。(2) 23.8.2 节用日志做了两项一致性检查：第一步 loss = 0.6931，以及各 β 运行反推出的参考模型对数概率完全一致。(3) 每个日志点都满足 loss ≥ $-\log\sigma(\text{平均 margin})$（Jensen 不等式：$-\log\sigma$ 是凸函数），例如 β = 0.05 最后一个日志点 0.2858 ≥ 0.1431（本项目观察 + 数学结论）。
- **替换内核后做到的**：逐对比较损失。TRL 1.15.0 的 `DPOTrainer` 强制走 Triton 融合内核（第 21 章 21.6.3 节的融合 LM head），没有 CPU 分支，在 CPU 上报 `RuntimeError: 0 active drivers ([]). There should only be one.`。所以只在实验脚本里把这个内核换成 CPU 上的 linear + cross_entropy。TRL 自己的数据处理（补 EOS、`common_prefix_length` 分界）、collator、骨干前向、shift/mask、求和和 `compute_loss` 都保留。`dpo_loss` 一侧用 TRL 处理好的同一批 token，走普通 logits 前向和 `response_logprob`。设置是 tiny 随机 Qwen3（1 层）、4 个手写偏好对、CPU、FP32、权重不更新、β = 0.1、batch 为 1，容差事先定为绝对差 ≤ 1e-5。四对的损失差都是 0；把 chosen 和 rejected 对调的否定对照，差为 0.0117 到 0.1220，都超过容差（本项目观察，脚本和结果在 `outputs/book/ch23/trl_compare/` 的 `compare_trl.py`、`compare.log`、`compare.json`）。
- **没做到的**：这次比较没有覆盖原版 Triton 内核的数值路径、反向传播和训练更新、GPU、真实预训练权重、聊天模板，所以它不能说明 `dpo_loss` 与原版 TRL 逐对一致。TRL 日志只记批平均，日志里也只有平均 margin，所以单从日志做不了逐对比较，loss 和 margin 的关系只能以上面的凸性下界的形式检查。练习 23.5（选做）让你自己做这个比较；有 GPU 的话，可以直接跑原版内核。

---

## 23.7 偏好数据：从后纠错候选自动构造

### 23.7.1 流程

对后纠错任务，"哪个候选更好"可以自动判断：离标准答案越近越好。本章的构造流程（lab 第 2、3 部分，代码 `tour.py` 的 `part2`、`part3`）：

1. **起点**：把第 21 章第 400 步的 LoRA adapter 合并进 Qwen3-0.6B，得到 SFT 策略。它同时是 DPO 的起点和参考模型。
2. **新提示**：取 train 第 2048 条之后的 2048 条。第 21 章 SFT 只用了前 2048 条，所以这些提示模型没训练过，偏好对不是在背过的题上做的。
3. **采样**：每个提示用 SFT 策略采样 4 个候选，温度 1.0，不用 top-p/top-k。候选是**模型自己写的**，所以偏好对正好落在模型自己的输出分布上。
4. **打分**：每个候选与标准答案算规范化后的字符编辑率（`sample_error_rate`）。
5. **配对**：编辑率最低的做 chosen（并列取较短的），最高的做 rejected。两者编辑率差小于 0.05 的提示丢掉。
6. **边界检查**：提示的 token 必须是"提示 + 回答"token 序列的前缀，否则丢弃。第 19 章 19.2.3 节讲过，分词在拼接处合并会让提示和回答的分界落在一个 token 中间。

本章用规范化字符编辑率来比较，它简单，也便宜，但它是一个会碰撞的代理距离：共享实现先把规范 token 无分隔拼回字符，命令边界就丢了。例如 `\sin h(x)`（sin 作用于 h(x)）和 `\sinh(x)`（双曲正弦）字符距离为 0，EM_norm 却判为不同；再加上“并列取较短”，较短的错误写法可能排在正确候选前面。本章 `candidates.jsonl` 的 8192 个候选里，字符距离为 0 而非 EM 的计数是 0，所以本章这批偏好对没有被这种零距离碰撞污染；非零距离下边界丢失对排序的影响尚未统计。本章不改共享实现，历史数字保持旧协议。换成"渲染后与标准答案的相似度"能容忍写法不同但长得一样的 LaTeX，这个改进可以留给项目。

### 23.7.2 实测：2048 个提示，只剩 387 对

| 项 | 数 | 来源 |
|---|---|---|
| 提示 | 2048 | `pair-stats.json` |
| 4 个候选完全相同 | 941 | lab 第 7 部分；对 `candidates.jsonl` 重新计数一致 |
| 因编辑率差 < 0.05 丢弃（含上一行） | 1661 | `pair-stats.json` 的 `no_gap` |
| 分词边界不一致 | 0 | `pair-stats.json` |
| **偏好对** | **387** | `pair-stats.json` |

（本项目观察，RTX 4090（租用），SFT 策略 bf16，采样墙钟 793 秒，`part2.json`。）

按不同候选的个数重新计数 `candidates.jsonl`：1 种 941 个，2 种 597 个，3 种 326 个，4 种 184 个（本项目观察，重新计数）。

为什么这么少？SFT 模型太自信了。温度 1.0 下，将近一半的提示（941/2048 = 45.9%，数学结论）四次采样一字不差。没有差异，就没有比较。这是对"自己的输出"做偏好学习的一个通病：模型越确定，能挖出的信号越少，而它确定的地方恰恰未必是对的。可以试的办法：每个提示多采几个、提高温度、或混入别的模型的输出（后者偏离了"自己的分布"，有代价）。本章没有试，留作练习 23.9。

### 23.7.3 噪声："更好"不等于"对"

两个例子（本项目观察，`candidates.jsonl` 与 `pairs.jsonl`）：

**干净的一对。**目标 `I_n(\Pi;q)=1+q.`

- chosen：`I_n(\Pi;q)=1+q.`（编辑率 0）
- rejected：`I_n(\Pi;q)=1+q^2.`（编辑率 0.235）

chosen 完全正确，rejected 凭空多了个平方。这是我们想让模型学的。

**有噪声的一对。**目标 `L(s, \pi \otimes \pi)=L(s, \vee^2)L(s, \wedge^2),`（输入里是拼错的 `\otlmes`）

- chosen：`L(s, \pi \leq \pi)=...`（编辑率 0.104）
- rejected：`L(s, \Pi \Rightarrow \Pi)=...`（编辑率 0.229）

两个都错了。`\leq` 只是"错得少一点"。DPO 会把 `\otlmes → \leq` 的概率往上推，这可不是我们想要的。

整体看：387 对里 chosen 完全正确的 225 对，占 0.581；rejected 完全正确的 0 对（本项目观察，`pair-stats.json`；重新计数一致）。也就是说，**约 42% 的 chosen 本身就是错的**（162/387，数学结论）。chosen 的编辑率中位数是 0、均值 0.025；rejected 中位数 0.135、均值 0.163。两者的编辑率差：中位数 0.111，四分位数 0.074 和 0.171（本项目观察，重新计数）。

这类噪声叫**偏好标签正确、但比较对象都不好**。另一类更常见于人工标注的噪声是**标签本身错了**（标注员看走眼）。DPO 对两者都很敏感：每一对都在说"往这个方向推"，错的方向也会被认真执行。

### 23.7.4 矛盾

**矛盾**是指数据里同时有"A 比 B 好"和"B 比 A 好"，或者 A > B、B > C、C > A 这样的循环。

- 人工标注里矛盾很常见：不同标注员口味不同，同一个人前后也不一致。Bradley-Terry 假设存在一个统一的分数，所以循环偏好它根本无法拟合。数据里有矛盾时，DPO 的 reward accuracy 天然达不到 1。
- 本章用一个确定的指标打分，同一提示内不会自相矛盾：编辑率就是一个分数，按分数排序不可能出现循环。
- 但"指标与真实质量矛盾"依然存在：字符编辑率小的候选，未必在数学上更接近原意（上面 `\leq` 的例子）。这叫**代理指标偏差**，第 25 章的奖励函数会再碰到它。

### 23.7.5 长度偏好

人工偏好数据里有一个被反复报告的现象：标注员倾向于选**更长**的回答。DPO 学到的可能不是"更有帮助"，而是"更长"。（这是文献中的普遍观察，本项目未测，待验证。）

DPO 还有一个机制上的原因让长度更危险：序列对数概率是 token **之和**。回答越长，求和项越多，$\Delta_w-\Delta_l$ 的数值范围也越大。TRL 的 `ld_alpha`（23.12 节）、SimPO 的长度归一化都是针对这一点的修补。

我们的数据有没有长度偏好？看 `pair-stats.json`（本项目观察）：

| 统计 | 值 |
|---|---|
| chosen 比 rejected 长的比例 | 0.388（150/387） |
| 一样长 | 61/387（重新计数） |
| chosen 更短 | 176/387（重新计数） |
| 平均长度（字符） | chosen 108.2，rejected 107.7 |
| rejected 有格式问题的比例 | 0.044（17/387） |

chosen 更短的比例反而略高，平均长度几乎一样：本批数据**未见明显的整体长度偏向**。这只排除了全局的边际偏移；若一类题偏好长候选、另一类偏好短候选，两者可以互相抵消，全局比例和均长照样相同。而且规则本身就和长度有关：编辑距离惩罚插入和删除，并列时又明确选较短的。训练后的输出长度也没有漂移：val 500 条原始输出的平均字符数，SFT 120.5，β = 1.0 的 DPO 120.8（本项目观察，对 `preds-val.jsonl` 重新计数）。

用自动指标构造偏好对，并不能自动排除长度混进标签：没有人类打分，只是去掉了一种来源，指标和平局规则仍可能引入长度相关的选择。要按题型分层统计才能更有把握。

---

## 23.8 实验：TRL DPOTrainer 与 β 扫描

操作步骤见 lab 第 4 到 6 部分，这里只讲结果怎样读。

### 23.8.1 设置

本项目观察，RTX 4090（租用），TRL 1.15.0、PEFT 0.21.2、transformers 5.19.0、torch 2.14.1（`environment.json`）：

- 策略起点与参考模型：第 21 章 step-400 adapter 合并后的 SFT 模型（bf16 保存，以 FP32 加载，bf16 autocast）。
- 可训练部分：新的 LoRA，r = 16，只加在 `q_proj`、`v_proj`。
- **参考模型 = 关闭 LoRA 的同一个模型**。TRL 发现策略是 PEFT 模型、又没传 `ref_model`，就在算参考对数概率时临时禁用 adapter（23.12 节）。好处是显存里只有一份基座。
- 偏好对 387 对；每步最多 16 对（batch 8 × 累积 2）；387 对每轮分成 48 个满批和最后 3 对，未丢弃末批，所以每轮 25 步，其中第 25 步只有 3 对；75 步，正好 3 轮；学习率 5 × 10⁻⁵ 常数；`max_length=384`。
- β 取 0.05、0.1、0.3、1.0，每个单独训练一次。
- 评测：val 500 条，贪心解码，与第 22 章同一协议、同一批 SFT 预测。

成本（本项目观察，与其他实验共享 GPU，只看量级）：每个 β 训练 61 到 137 秒（`train-summary.json`），PyTorch 峰值 allocated 14,007,232,000 字节 ≈ 13.0 GiB（换算为数学结论）；每个 β 的 val 生成 317 到 345 秒（`val-metrics.json`）。

### 23.8.2 训练日志怎样读

以 β = 1.0 为例（本项目观察，`beta-1.0/trainer-history.json`）：

| 步 | loss | rewards/chosen | rewards/rejected | margin | accuracy |
|---|---|---|---|---|---|
| 1 | 0.6931 | 0.000 | 0.000 | 0.000 | 0.000 |
| 25 | 0.4208 | 0.060 | −1.150 | 1.210 | 0.778 |
| 50 | 0.1702 | 0.100 | −3.740 | 3.840 | 0.931 |
| 75 | 0.0376 | 0.291 | −7.232 | 7.523 | 0.986 |

读法：

1. **第一步 loss = 0.6931**，所有 β 都一样。这就是 23.5.6 节情形 A：策略 = 参考，margin = 0。如果你的第一步不是 log 2，说明参考模型和策略起点不一致（23.11 节）。第一步的 accuracy 是 0 而不是 0.5，因为 margin 恰好等于 0，"大于 0"不成立。
2. **margin 稳步上升，accuracy 超过 0.5**：模型在训练对上越来越能区分 chosen 和 rejected。
3. **主要靠压低 rejected**：β = 1.0 时 chosen 的隐式奖励只涨到 0.291，rejected 降到 −7.232。DPO 的梯度是"差"，它选了更省力的方向。

注意两件事。第一，TRL 的每个日志点是**自上次记录以来各步的平均**（`_metrics` 在记录时求平均后清空，23.12 节），而第 1 步只含一步。第二，训练对只有 387 对，跑了 3 轮，末段的 loss 和 accuracy 是**训练集上**的数字，接近过拟合的信号，不能当成泛化能力（第 21 章 21.9 节）。

**一个免费的一致性检查。**`logps/chosen` 是策略的序列对数概率，`rewards/chosen` 是 β 乘以对数概率比，所以

$$\log\pi_{\mathrm{ref}}(y_w)=\texttt{logps/chosen}-\texttt{rewards/chosen}/\beta$$

（对批平均同样成立，因为它是线性的；数学结论。）对四个 β 在第 75 步的日志算一遍：

| β | logps/chosen | rewards/chosen | 反推参考 $\log\pi_{\mathrm{ref}}(y_w)$ |
|---|---|---|---|
| 0.05 | −11.425 | −0.493 | −1.567 |
| 0.1 | −5.623 | −0.406 | −1.567 |
| 0.3 | −2.350 | −0.235 | −1.567 |
| 1.0 | −1.276 | 0.291 | −1.567 |

（本项目观察，四个 `trainer-history.json` 的末行；反推为数学结论。rejected 一侧同样四个都是 −6.793。对其余 15 个日志点做同样的反推，四个 β 在每个点上也都相同，保留两位小数。）

四个独立运行反推出同一个参考值。这说明三件事：参考模型在四次运行里是同一个；四次运行看到的数据顺序相同（同一 seed）；TRL 的 reward 定义和 `dpo_loss` 一致。如果参考模型在训练中跟着更新，四个 β 的更新各不相同，同一步的反推值很难完全对上，所以这也和参考没有被更新相符。但它只是一致性证据，参考冻结的直接证据是 TRL 源码里参考前向包在 `torch.no_grad()` 下（23.12 节）。

注意不能拿不同步的反推值互相比较：第 1 步反推出 −3.304，第 75 步是 −1.567，这只是两批数据不同。这是本章训练日志里与 TRL 最强的一条一致性证据。

### 23.8.3 val 结果

本项目观察，RTX 4090（租用），387 对，75 步，val n = 500，贪心解码。SFT 一行来自第 22 章的同一批预测，它是**第 22 章旧实现的历史输出**（FP32 基座挂 adapter），不是本章 DPO 实际起点（合并后 bf16 保存、FP32 加载）在同一设置下的零步输出；本章没有生成这个零步基线。合并与降精度未必改变任何一题，但下面的配对差因此不能全部归因于 DPO 训练，只能读作“相对第 22 章参照”的差。区间：EM_norm 一列是 Wilson 区间，DPO 各行取自各 `val-metrics.json`，SFT 一行取自第 22 章 `analysis.json`（同一函数、同一口径）；配对差（DPO − SFT，配对 bootstrap，1000 次，seed 0）：β = 1.0 一行取自 `selection.json`；其余三行用同一个 `paired_bootstrap_diff` 对已有预测重新计算，结果与 lab 第 7 部分的表一致。lab 表里 SFT 一行的 [0.638, 0.718] 是 `selection.json` 里的 bootstrap 区间，这里和 DPO 各行统一成 Wilson。

| 设置 | 末段 loss | 末段 margin | 末段 logps/chosen | val EM_norm（95% 区间） | val CER | 与 SFT 配对差 |
|---|---|---|---|---|---|---|
| SFT | 不适用 | 不适用 | 不适用 | 0.678 [0.636, 0.717] | 0.0257 | 不适用 |
| β = 0.05 | 0.286 | 1.87 | −11.43 | 0.414 [0.372, 0.458] | 0.0505 | −0.264 [−0.308, −0.216] |
| β = 0.1 | 0.210 | 2.54 | −5.62 | 0.590 [0.546, 0.632] | 0.0234 | −0.088 [−0.122, −0.050] |
| β = 0.3 | 0.101 | 4.49 | −2.35 | 0.688 [0.646, 0.727] | 0.0164 | +0.010 [−0.018, 0.038] |
| **β = 1.0（val 选中）** | 0.038 | 7.52 | −1.28 | **0.706 [0.665, 0.744]** | 0.0228 | **+0.028 [0.004, 0.052]** |

再按第 22 章的口径看面板的另外两项，"改坏"和"修好"（即 fix_rate）（本项目观察，`val-metrics.json` 与第 22 章 `analysis.json`）：

| 设置 | 改坏（输入本来对） | 修好，即 fix_rate（输入本来错） |
|---|---|---|
| SFT | 11/163 | 187/337 |
| β = 0.05 | 51/163 | 95/337 |
| β = 0.1 | 28/163 | 160/337 |
| β = 0.3 | 12/163 | 193/337 |
| β = 1.0 | 8/163 | 198/337 |

三个观察：

- **β 越小越糟。**β = 0.05 把原本正确的输入改坏了 51 条，是 SFT 的四倍多。
- **β 较大时，DPO 和 SFT 打成平手或略好。**β = 0.3 的配对差区间包含 0；β = 1.0 的区间刚好不含 0。
- **两个指标不完全一致。**按 EM_norm，β = 1.0 最好；按 CER，β = 0.3 最好（0.0164 对 0.0228）。如果事先没定好"用哪个指标选"，这里就又多了一条分岔路。

### 23.8.4 小 β 时发生了什么

**现象一：chosen 的对数概率也被压下去了。**从 23.8.2 节的表看，同一批训练对，参考模型给 chosen 的对数概率是 −1.567。训练 75 步后，β = 1.0 的策略给 −1.276（略升），β = 0.05 给 −11.425，平均对数概率比为 −9.86。换回概率，就是这一窗口里各条 chosen 的概率比 $\pi_\theta/\pi_{\mathrm{ref}}$ 的几何平均约为 $e^{-9.86}\approx5\times10^{-5}$（数学结论）。这个几何平均是 $\exp(\mathbb{E}[\log(\pi_\theta/\pi_{\mathrm{ref}})])$，不是每条 chosen 的倍率，也不是平均概率之比。模型在"学偏好"的同时，把自己本来很确定的好答案也弄得不确定了。这是 23.5.6 节情形 C 的放大版。

这里要小心比较的对象。不能拿日志首末的约 −3.3 和 −11.4 比：−3.3 是第 1 步那一批（16 对）的值，−11.4 是第 71 到 75 步那几批的平均，不是同一批数据。不同批的回答长短不同，对数概率本来就不同。同一批上的公平比较是参考 −1.567 对策略 −11.425。

**现象二：小 β 让策略离参考模型更远。**把末段 margin 除以 β，得到对数概率比之差 $\Delta_w-\Delta_l$：

| β | margin | $\Delta_w-\Delta_l$ = margin / β |
|---|---|---|
| 0.05 | 1.872 | 37.4 |
| 0.1 | 2.541 | 25.4 |
| 0.3 | 4.489 | 15.0 |
| 1.0 | 7.523 | 7.5 |

（本项目观察，末段日志；除法为数学结论。）

**现象三：输出离 SFT 越来越远。**val 500 条里，DPO 输出与 SFT 输出逐字相同的条数（本项目观察，对 `preds-val.jsonl` 重新计数）：

| β | 0.05 | 0.1 | 0.3 | 1.0 |
|---|---|---|---|---|
| 与 SFT 相同 | 176 | 256 | 345 | 401 |

三个现象都符合 23.4.4 节的说法：β 是那根绳子，β 小，绳子松，策略被允许跑得更远。在只有 387 对的小数据上，跑远换来的是 val 上更差的分数，chosen 的概率也被压低了。

为什么会这样？有两个推测（待验证，本章没有做专门的对照实验）：

1. **chosen 和 rejected 共享大部分 token。**编辑率差的中位数只有 0.111（23.7.3 节），两个回答大部分字符相同。压低 rejected 时，共享部分的概率也被一起压低，而梯度里"推高 chosen"那一半只能抵消一部分。
2. **β 小，损失难以饱和。**要让损失降到同样水平，margin（以 β 为单位）要一样大，所以 β 越小，需要的对数概率比之差越大（上表）。损失长时间不饱和，梯度权重 $\sigma(-u)$ 就一直不小。另一方面，AdamW 的更新量是 $m/\sqrt v$，梯度整体乘以一个常数时 $m$ 和 $\sqrt v$ 同比缩放，比值近似不变（忽略 ε 时为数学结论），所以 β 小带来的梯度变小并不会让步子变小。

**怎样发现？**不只看 loss，还要看三样东西：`logps/chosen` 是否明显低于同一批的参考值（按 23.8.2 节的公式反推，不要拿第一步的日志值比）；`rewards/chosen` 是否持续为负；以及在 val 上和 SFT 做配对比较，看"改坏"的条数。只看 loss 和 reward accuracy，四个 β 看起来都很成功：β = 0.05 的末段 accuracy 也有 0.917。

### 23.8.5 β = 1.0 的提升可信吗

先看配对差是怎么来的。按第 22 章的 2×2 表重新计数（本项目观察，对两份预测逐条计数）：

| | DPO β=1.0 对 | DPO β=1.0 错 |
|---|---|---|
| **SFT 对** | 326 | 13 |
| **SFT 错** | 27 | 134 |

差 = (27 − 13)/500 = 0.028。只有 40 道题不一致。按第 22 章 22.3.3 节的公式：$\mathrm{Var}(d)=40/500-0.028^2=0.0792$，SE = $\sqrt{0.0792/500}\approx0.0126$，正态 95% 区间约 [0.003, 0.053]（数学结论），与配对 bootstrap 的 [0.004, 0.052] 吻合。

区间不含 0，但下界只有 0.004。而且这个数字**带着选择偏差**，理由有三条：

1. **四选一。**β 是在 val 上从 4 个值里挑的，报告的正是挑出来那个的 val 分数。按第 22 章 22.7.3 节，若 4 个候选真实水平相同、互相独立，p ≈ 0.7、n = 500 时"挑最大"的偏差约 $0.0205\times1.029\approx0.021$（数学结论，正态近似）。我们的 4 个候选水平差别很大，真正和 β = 1.0 竞争的只有 β = 0.3，而且两者的对错高度相关，所以实际偏差应该小于这个数。但它和 0.028 是同一个量级。
2. **扫描范围是看了结果才扩的。**lab 第 7 部分有记录：最初只扫 {0.05, 0.1, 0.3}，最优在边界 0.3；看到 val 结果后才加了 β = 1.0。规则（val EM_norm 最高、并列取小）是事先定的，但"再试一个"的决定是看了 val 才做的。第 22 章 22.7 节说过，每一次根据评测结果做的决定都是一次选择，这里的选择发生在 val 上。
3. **β = 1.0 又在边界上。**更大的 β 没测，不知道曲线有没有到顶。

**本章没有在 test 上评 DPO。**按第 0 章 0.10.1 节和第 5 章 5.1.3 节的协议，val 上的这些选择是允许的，代价就是 val 数字偏乐观。要对外声称"DPO 比 SFT 好"，需要冻结 β = 1.0 这一设置，作为预先登记的候选在 test 上评**一次**，报告那个数字，不据此改任何设置。

这批数字能支持的结论是：**在这 500 道 val 上，β ≤ 0.1 的 DPO 明显比 SFT 差；β = 0.3 与 SFT 分不出高下；β = 1.0 可能略好，但证据很弱，还没有 test 确认。**这里的“SFT”是第 22 章的历史输出，差值里混有合并与精度路径的变化，没有零步基线就不能全部算在 DPO 头上。

---

## 23.9 变体概览：IPO、KTO 等（概括）

DPO 之后出现了大量变体。这里只说每个想解决什么问题。

- **IPO**（Azar et al., 2023）：DPO 的 sigmoid 损失在偏好是确定的（数据总说 A > B）时，会把 margin 推向无穷大，β 的约束作用被削弱。IPO 把损失换成平方：$\big(\Delta_w-\Delta_l-\frac{1}{2\tau}\big)^2$，让 margin 停在一个有限的目标值。TRL 里是 `loss_type="ipo"`，这时 `beta` 就是 τ；TRL 1.15.0 的实现还按回答 token 数做了平均（源码注释说明了这一点）。
- **KTO**（Ethayarajh et al., 2024）：不需要成对数据，只要每个回答单独标"好"或"不好"。适合只有点赞/点踩的场景。TRL 里是单独的 `KTOTrainer`。
- **SimPO** 等：去掉参考模型，并用长度归一化的对数概率当隐式奖励，省显存，也缓解长度偏好。
- **在线 / 迭代 DPO**：定期用当前策略重新采样、重新构造偏好对，让数据跟上策略的变化。本章是"离线"的：偏好对只在 SFT 策略上采样一次。

TRL 1.15.0 的 `DPOConfig.loss_type` 还列了 `hinge`、`robust`、`bco_pair`、`apo_zero`、`discopop` 等十几种（源码 `dpo_config.py`）。其中大多数共享 23.5 节的骨架：用对数概率比做分数，只是把 $-\log\sigma(\cdot)$ 换成了别的函数。

---

## 23.10 和项目的关系

**偏好数据可以自动构造。**"两个识别结果哪个更好"在公式 OCR 里可以自动判断：与标准答案比编辑率，或者渲染后比图像相似度。所以 TexPixel 项目不需要人工标注就能做偏好优化。23.7 节的流程可以直接搬到 VLM 上（第 33 章以后）：用模型对同一张公式图片采样几个 LaTeX 候选，比较，配对。

**R2 中期报告的 DPO 行。**R2 要求"提示词基线 → SFT/LoRA → DPO 或 GRPO 的结果表（附置信区间）"。DPO 这一行要写：

- 偏好对的来源与数量：2048 个新提示 × 4 个候选，min_gap 0.05，剩 387 对；941 个提示四个候选完全相同。
- 偏好对的质量统计：chosen 精确匹配率 0.581；长度统计（未见明显的整体长度偏向，不排除分题型的偏差）。
- 训练设置：LoRA r16 q/v，β 扫描 {0.05, 0.1, 0.3, 1.0}，75 步，学习率 5 × 10⁻⁵。
- 选择：val EM_norm 最高者，β = 1.0；**写明扫描范围是看了 val 后扩展的**，以及 β = 1.0 在扫描上界。
- val：0.706 [0.665, 0.744]，与 SFT 配对差 +0.028 [0.004, 0.052]（SFT 为第 22 章旧实现的历史输出，不是合并快照的零步输出，见 23.8.3 节）；CER、改坏、fix_rate 见 23.8.3 节。
- test：本章未评。R2 定稿时冻结设置评一次；在那之前这一格留空，不要拿 val 的数字填。
- 成本：采样 793 秒 + 每个 β 的训练与 val 生成（23.8.1 节）。

**作业 A4。**机制层：`dpo_loss` 是 A4 的一部分。A4 的讲义是 `assignments/a4/handout.md`，测试在 `assignments/a4/tests/`；本章的 `test_ch23_dpo.py` 已经覆盖手算值、梯度方向和非法 β 三项，可以先用它自查。训练层："小模型上奖励达到门槛"，**门槛待校准**，由你第一次完整训练后确定。

---

## 23.11 常见错误

**1. 参考模型和策略起点不一致。**
- 现象：第一步 loss 不是 0.6931，rewards 第一步就不是 0。
- 原因：比如策略加载了 SFT 合并模型，参考模型却是没合并的基座；或者一个是 bf16、另一个是 FP32。
- 怎样发现：看第 1 步日志。本章四次运行第 1 步都是 0.6931、rewards 全 0（本项目观察）。

**2. 参考模型被训练了。**
- 现象：loss 降得飞快、margin 涨得也快，但 `logps/chosen` 和 `logps/rejected` 几乎不动，margin 是参考模型在动造出来的。
- 原因：参考模型的计算没放在 `no_grad` 下，而且优化器里混进了参考模型的参数。
- 怎样发现：用 23.8.2 节的反推检查。固定同一批样本，在不同时刻反推（或直接重算）参考对数概率，应该对得上；不同运行在同一步的反推值也应该对得上。不同批次的均值本来就不同，不能拿来比。

**3. 对数概率用了平均而不是求和，或者把提示也算进去了。**
- 现象：按 token 平均会让对数概率比缩小约"回答 token 数"倍，β 的"手感"和别人报告的完全对不上，目标本身也变了。把提示算进求和则不同：chosen 和 rejected 用同一个提示、分词边界相同时，提示部分在 $\Delta_w-\Delta_l$ 里严格抵消，标准 DPO 的 margin 和损失不变（数学结论）；但单侧的隐式奖励 `rewards/chosen`、`rewards/rejected` 和 `logps/*` 日志会变，含义就不对了。
- 怎样发现：手算一个只有 2 个回答 token 的例子（`test_response_logprob_ignores_prompt_and_padding`）。注意 IPO 在 TRL 里按 token 平均，那是那个损失的有意设计，不是 DPO 的默认。

**4. `torch.log(torch.sigmoid(z))`。**
- 现象：margin 很负时出现 −inf 或 NaN。
- 怎样发现：喂一个 margin = −200 的样本进损失函数。用 `F.logsigmoid`。

**5. β 太小。**
- 现象：loss 和 reward accuracy 都很漂亮，但 `logps/chosen` 一路下跌，生成质量变差。本章 β = 0.05：末段 accuracy 0.917，val EM_norm 却从 0.678 掉到 0.414（本项目观察）。
- 怎样发现：同时看 `logps/chosen`、`rewards/chosen` 的符号，以及 val 上的"改坏"数（23.8.4 节）。

**6. 把训练对上的 reward accuracy 当成效果。**
- 现象：accuracy 接近 1，以为模型学好了。
- 原因：这是训练集上的数字。387 对跑 3 轮，背下来很容易。
- 怎样发现：只在 val 上用任务指标（EM_norm、改坏数）做判断。

**7. 偏好对的提示和回答在分词边界上粘连。**
- 现象：TRL 打出 "The tokenized prompt is not a prefix of the tokenized prompt+completion" 警告；chosen/rejected 的第一个 token 被算进提示或丢失。
- 怎样发现：构造时就检查（本章 `tokenization_mismatch = 0`）。

**8. 看了 val 再扩扫描范围，却不写出来。**
- 现象：报告里的提升在 test 上消失。
- 怎样发现：在报告里写清每一个"看了结果才做"的决定（本章的 β = 1.0 就是一例），最终数字用冻结后的 test。

**9. 偏好对太少或太同质。**
- 现象：大部分提示被丢掉；训练很快过拟合。本章 2048 个提示只剩 387 对。
- 怎样发现：构造后先看 `pair-stats.json`，包括候选完全相同的提示数。

---

## 23.12 和真实框架的对应

本节按本地安装的 TRL 1.15.0 源码（`trl/trainer/dpo_trainer.py`、`dpo_config.py`）核对。

| 本章 | TRL 1.15.0 | 说明 |
|---|---|---|
| `dpo_loss` | `DPOTrainer._compute_loss`，`loss_type="sigmoid"`（默认） | `per_sequence_loss = -F.logsigmoid(self.beta * delta_score)`；`delta_score` 在默认 `f_divergence_type="reverse_kl"` 下就是 $\Delta_w-\Delta_l$ |
| `response_logprob`（求和） | `logps = per_token_logps.sum(dim=1)` | 默认 `ld_alpha=None` 时是回答 token 之和 |
| 偶/奇行交错 | `logps.chunk(2, dim=0)` | TRL 是前一半 chosen、后一半 rejected |
| `ref_logprob_fn`（no_grad） | `torch.no_grad()` + `use_adapter(model, adapter_name=None)` | 未传 `ref_model`、通过 `peft_config` 新建 adapter 时（本章做法），禁用 adapter 当参考；若传入的是已带 adapter 的 PEFT 模型，TRL 一般复制出一个 `"ref"` adapter 当参考 |
| 返回的隐式奖励 | `rewards/chosen` = `beta * chosen_logratios` 的批平均 | `rewards/margins`、`rewards/accuracies` 同理；`logps/chosen` 是策略序列对数概率的批平均 |
| `loss.mean()` | `per_sequence_loss.mean()` | 再由 Trainer 处理梯度累积 |
| β | `DPOConfig.beta`，默认 0.1 | 本章扫描 0.05 到 1.0 |
| 学习率 | `DPOConfig.learning_rate`，默认 1e-6 | 本章用 LoRA，设 5e-5 |
| 长度上限 | `max_length`，默认 1024；`truncation_mode` 默认 `"keep_start"` | 本章 384 |
| 自动加 EOS | 非对话格式数据，chosen/rejected 不以 EOS 结尾时自动补上 | 本章构造时已经加了 `<\|im_end\|>`，所以不会重复 |
| 预先算参考对数概率 | `precompute_ref_log_probs`，默认 False | 设 True 可在训练前算好，省掉训练中的参考前向 |
| 变体 | `loss_type` 列表：`ipo`、`hinge`、`robust`、`apo_zero` 等；`loss_weights` 可组合多个 | KTO 是单独的 `KTOTrainer`；奖励模型是 `RewardTrainer` |
| 日志点 | `_metrics` 累积后在记录时求平均并清空 | 每个日志点是自上次记录以来的平均 |

几个没有核对到底、标为**待验证**的地方：

- 梯度累积时，TRL 的批平均损失和"16 对整体平均"是否严格相等（两个 microbatch 都是 8 对时应当相等，但没有逐数验证；每轮最后一步只有 3 对，不涉及这个问题）。
- `use_adapter(..., None)` 在本章 LoRA 配置下是否完全等价于"未加 LoRA 的 SFT 合并模型"。23.8.2 节的反推检查（四次运行同一参考值）是间接证据，不是直接验证。

---

## 23.13 小结

偏好优化从"哪个更好"的比较中学习，RLHF 把它拆成 SFT、奖励模型、带 KL 约束的强化学习三步。Bradley-Terry 模型把偏好写成分差的 sigmoid，KL 约束的奖励最大化有闭式解 $\pi^*\propto\pi_{\mathrm{ref}}e^{r/\beta}$；把奖励反解出来代入 Bradley-Terry，配分函数抵消，就得到 DPO：一个以 β 乘对数概率比为分数的二分类损失。参考模型是先验，β 是绳子：本项目 β = 0.05 时 chosen 的对数概率被一起压低，val EM_norm 从 0.678 掉到 0.414；β = 1.0 时为 0.706，但这是在 val 上看了结果才扩展扫描得到的，没有 test 确认。自动构造的偏好对便宜，本批也未见明显的整体长度偏向，但约 42% 的 chosen 本身是错的，而且 SFT 模型太确定，2048 个提示只剩 387 对。和 TRL 的对拍有源码核对、日志一致性检查，以及替换 Triton 内核后在 CPU、tiny 随机模型上的逐对比较；原版内核的数值路径和训练更新没有验证。

复习时抓住下面几条：

- **Bradley-Terry**：$P(y_w\succ y_l\mid x)=\sigma(r_w-r_l)$，只看分差，所以给同一提示下所有回答加同一个 $c(x)$ 不改变概率。训练奖励模型就是标签恒为 1 的二分类交叉熵 $-\log\sigma(r_w-r_l)$；两个分数相等时损失是 $\log2\approx0.6931$。
- **KL 约束下的最优策略**：$\max_\pi\ \mathbb E_\pi[r]-\beta\,\mathrm{KL}(\pi\|\pi_{\mathrm{ref}})$ 的解是 $\pi^*(y\mid x)=\pi_{\mathrm{ref}}(y\mid x)e^{r(x,y)/\beta}/Z(x)$。推导的关键是把目标凑成 $-\beta\,\mathrm{KL}(\pi\|\pi^*)+\beta\log Z$，再用 KL ≥ 0。β 大时贴着参考模型，β 小时贪心地押在高奖励回答上；参考模型给 0 概率的回答，最优策略也给 0。
- **DPO 损失**：对闭式解取对数反解出 $r=\beta\log\frac{\pi^*}{\pi_{\mathrm{ref}}}+\beta\log Z(x)$，代入 Bradley-Terry 后两个 $\beta\log Z(x)$ 相减抵消，再用 $\pi_\theta$ 代替 $\pi^*$：$\mathcal L_{\mathrm{DPO}}=-\log\sigma\big(\hat r_\theta(x,y_w)-\hat r_\theta(x,y_l)\big)$，隐式奖励 $\hat r_\theta=\beta\log\frac{\pi_\theta}{\pi_{\mathrm{ref}}}$。序列对数概率是回答 token 的对数概率之**和**，提示不算。
- **梯度**：$\nabla_\theta L=-\beta\,\sigma(\hat r_l-\hat r_w)\big[\nabla\log\pi_\theta(y_w)-\nabla\log\pi_\theta(y_l)\big]$。它只管 chosen 和 rejected 之差，排错得越多推得越狠，排对而且拉开了的几乎不管。所以损失下降时 chosen 的概率也可能在下降（23.5.6 节情形 C）。
- **β 的作用**：β 是绳子的松紧。本章 387 对、75 步的设置下，β 越小，策略离参考越远、输出离 SFT 越远，β = 0.05 时 val 明显变差；β = 1.0 是 val 上挑出来的，还没有 test 确认（本项目观察，23.8 节）。
- **偏好数据**：用编辑率给自己的采样候选排序，自动、便宜、同一提示内不会出现循环偏好；代价是 chosen 本身可能是错的、SFT 太确定时大部分提示配不出对。构造后先看 `pair-stats.json`。
- **最常见的错误**：策略从参考出发时，第一步 loss 必须是 0.6931，不是就说明两者起点不一致；用 `torch.log(torch.sigmoid(z))` 而不是 `F.logsigmoid`；只看 loss 和训练对上的 reward accuracy，不看 `logps/chosen` 和同一批参考值的比较、不在 val 上数改坏。

---

## 23.14 精读

**1. Rafailov et al., 2023.** *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* 读 §4（推导）和 §5（理论分析）。

带着这些问题读：

1. §4 从 RL 目标到最优策略的那一步，论文把细节放在了附录。对照本章 23.4.2 节，作者用的是同一个"凑成 KL"的技巧吗？
2. §5 说 DPO 的重参数化"不损失一般性"：任何奖励函数都能写成 $\beta\log\frac{\pi}{\pi_{\mathrm{ref}}}$ 的形式（相差一个只依赖于提示的项）。为什么"相差一个只依赖于提示的项"不影响 Bradley-Terry？用 23.3.5 节回答。
3. §5 还解释了 PPO 这类 actor-critic 方法为什么不稳定。它说的"归一化项"和本章的 $Z(x)$ 是什么关系？

**2. Ouyang et al., 2022.** *Training language models to follow instructions with human feedback*（InstructGPT）. 读方法一节（高层流程图和三步的描述）。

带着这些问题读：

1. 奖励模型是怎样从"对 K 个回答的排序"里拆出训练对的？为什么作者把同一提示的所有对放在同一个 batch 里？
2. 第三步除了 KL 惩罚，还加了一项预训练损失（PPO-ptx）。它想解决什么问题？和本章 β 太小时 chosen 概率下降的现象有关系吗？
