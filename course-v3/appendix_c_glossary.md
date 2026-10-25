# 附录 C　术语表（中英对照）

全书各章"术语卡"的合集（由 `tools/appendix_tables.py` 生成）。同一个术语在几章里出现时，释义取第一次出现的那一章，"章"一栏列出所有出现过它的章，第一个是首次出现。
前半部分按课程的部分排列，便于复习一个部分；后半部分是按英文字母排序的索引，便于查一个英文词。

共 432 个术语，来自第 0–36 章。

## 第 0 部分　准备（第 0–3 章）

| 中文 | 英文 | 一句话 | 章 |
|---|---|---|---|
| 模型 | model | 带参数的函数 | 0 |
| 参数 | parameter | 模型里可以调节的数（"旋钮"） | 0、9 |
| 训练 | training | 用数据自动调整参数 | 0 |
| 损失 | loss | 衡量模型输出和正确答案差多少的数，越小越好 | 0 |
| 梯度 | gradient | 每个参数往哪个方向变一点，损失下降最快 | 0、5 |
| 学习率 | learning rate | 每次更新参数时迈多大一步 | 0、5 |
| 训练集 / 验证集 / 测试集 | train / validation / test set | 练习题 / 模拟考 / 正式考 | 0、10 |
| 视觉语言模型 | VLM | 输入图像和文字、输出文字的模型 | 0 |
| 光学字符识别 | OCR | 把图像里的文字、公式、表格读成文本 | 0 |
| 视觉编码器 | vision encoder | 把图像变成一串向量的网络 | 0 |
| 连接器 | connector / projector | 把视觉向量变成语言模型能读的 token | 0、22 |
| 冻结 | freeze | 某个部件的参数在这一阶段不训练 | 0、22、28 |
| 指令微调 | SFT（supervised fine-tuning） | 用"指令—回答"配对继续训练 | 0、22 |
| 列表推导式 | list comprehension | `[表达式 for x in 序列]` | 1 |
| 生成器 | generator | 用 `yield` 一个一个产出结果的函数 | 1 |
| 类 / 对象 | class / object | 模板 / 用模板造出的具体东西 | 1 |
| 模块 / 包 | module / package | 一个 `.py` 文件 / 一个带 `__init__.py` 的文件夹 | 1 |
| 数组的形状 / 类型 | shape / dtype | 每一维的长度 / 元素的类型 | 1 |
| 视图 / 副本 | view / copy | 共享内存 / 独立的一份 | 1 |
| 广播 | broadcasting | 形状不同的数组自动对齐后逐元素运算 | 1 |
| 向量化 | vectorization | 用数组运算代替 Python 循环 | 1 |
| 独热向量 | one-hot | 只有一个位置为 1 的向量 | 1 |
| 补齐 / 掩码 | padding / mask | 把序列补成一样长 / 标记哪些位置是真实内容 | 1 |
| 小批量 | mini-batch | 每次训练用的一小部分数据 | 1 |
| 种子 | seed | 伪随机数序列的起点 | 1 |
| 断言 | assertion | 条件不成立时立即报错的检查 | 1 |
| 像素 | pixel | 图像网格里的一个小格子 | 2 |
| 灰度图 / 彩色图 | grayscale / RGB image | 每像素 1 个数 / 3 个数 | 2 |
| 通道 | channel | 彩色图的每一种颜色分量 | 2 |
| 边界框 | bounding box | `(x0, y0, x1, y1)`，一个矩形区域 | 2、21 |
| 渲染 | rendering | 把文字画成图 | 2 |
| 字形 | glyph | 一个字符画出来的样子 | 2 |
| 基线 | baseline | 一行字"站"着的那条线 | 2、29 |
| 抗锯齿 | anti-aliasing | 边缘用灰色过渡，让画面平滑；缩小前先平滑以免混叠 | 2 |
| 最近邻 / 双线性 / 区域平均 | nearest / bilinear / area | 三种缩放方法 | 2 |
| 混叠 | aliasing | 采样太稀导致的图案失真 | 2 |
| 长宽比 | aspect ratio | 宽除以高 | 2 |
| letterbox | letterbox | 保持比例缩放、补边成目标尺寸 | 2 |
| 标准化 | standardization | 减均值除标准差 | 2、7 |
| HWC / CHW / BCHW | — | 维度顺序约定 | 2 |
| 码位 | code point | Unicode 给每个字符的编号 | 3 |
| 编码 | encoding | 码位到字节的规则，如 UTF-8 | 3 |
| 字符集 / 词表 | charset / vocabulary | 模型能输出的所有符号 | 3 |
| 结构化输出 | structured output | 文本里藏着结构：Markdown、HTML、LaTeX | 3 |
| 记号 | token | 语法上不可再分的最小单位 | 3 |
| 语法树 | syntax tree | 用树表示的结构；叶子是符号，内部节点是结构 | 3 |
| 文法 | grammar | 规定什么样的记号序列合法、结构是什么的规则 | 3 |
| 解析 | parsing | 字符串 → 语法树 | 3 |
| 递归下降 | recursive descent | 每条文法规则写成一个函数的解析方法 | 3 |
| 规范写法 | canonical form | 每棵树唯一的标准字符串 | 3 |
| 归一化 | normalization | 解析后按规范写法写回 | 3 |
| 幂等 | idempotent | 做两次和做一次结果相同 | 3 |
| 往返性质 | round-trip | `parse(to_latex(t)) == t` | 3 |
| 拒绝采样 | rejection sampling | 随便生成，不合要求就丢掉重来 | 3 |

## 第一部分　数学（第 4–6 章）

| 中文 | 英文 | 一句话 | 章 |
|---|---|---|---|
| 点积 / 内积 | dot product | 对应分量相乘再相加 | 4 |
| 范数 / 长度 | norm | `√(a·a)` | 4 |
| 余弦相似度 | cosine similarity | 点积除以两个长度，只看方向 | 4 |
| 单位向量 | unit vector | 长度为 1 的向量 | 4 |
| 转置 | transpose | 行列互换 | 4 |
| 矩阵乘法 | matrix multiplication | 行与列的点积 | 4 |
| 单位矩阵 | identity matrix | 对角线为 1，乘任何矩阵都不变 | 4 |
| 正交矩阵 | orthogonal matrix | `QᵀQ = I`，不改变长度和夹角 | 4 |
| 线性变换 | linear transformation | 矩阵代表的变换 | 4 |
| 线性层 | linear / fully connected layer | `y = xW + b` | 4 |
| 权重 / 偏置 | weight / bias | 线性层的参数 | 4 |
| 浮点运算次数 | FLOPs | 衡量计算量 | 4 |
| 外积 | outer product | 列向量乘行向量，得到矩阵 | 4 |
| 灾难性抵消 | catastrophic cancellation | 两个接近的大数相减损失精度 | 4 |
| 导数 | derivative | 函数在一点的变化率 | 5 |
| 极限 | limit | h 越来越小时式子趋近的值 | 5 |
| 链式法则 | chain rule | 复合函数的导数是各段导数之积 | 5 |
| 偏导数 | partial derivative | 只让一个变量变化时的导数 | 5 |
| 方向导数 | directional derivative | 沿某方向的变化率，等于 `∇f · u` | 5 |
| 等高线 | contour | 函数值相同的点连成的线 | 5 |
| 梯度下降 | gradient descent | 反复沿负梯度走一小步 | 5 |
| sigmoid | sigmoid / logistic | `1/(1 + e⁻ᶻ)`，压到 (0, 1) | 5 |
| ReLU | rectified linear unit | `max(0, x)` | 5、10 |
| 中心差分 | central difference | `(f(x+h) − f(x−h))/2h` | 5 |
| 梯度检查 | gradient check | 用数值梯度验证解析梯度 | 5 |
| 泰勒展开 | Taylor expansion | 用多项式近似函数 | 5 |
| Hessian / 条件数 | Hessian / condition number | 二阶导数矩阵 / 最大与最小弯曲度之比 | 5 |
| 随机变量 / 分布 | random variable / distribution | 取值不确定的量 / 各取值的概率 | 6 |
| 类别分布 | categorical distribution | K 个类各有一个概率 | 6 |
| 期望 / 方差 | expectation / variance | 加权平均 / 散开程度 | 6 |
| 大数定律 | law of large numbers | 样本越多，均值越接近期望 | 6 |
| 熵 | entropy | 平均的意外程度 `−Σ p log p` | 6、27 |
| logits | logits | 模型输出的未归一化分数 | 6 |
| softmax | softmax | `e^z / Σ e^z`，分数变概率 | 6 |
| 温度 | temperature | logits 除以 T，控制分布尖锐程度 | 6、17、20 |
| logsumexp | log-sum-exp | `log Σ e^z` 的稳定算法 | 6 |
| 交叉熵 | cross-entropy | `−Σ p log q`；分类时为 `−log p_y` | 6 |
| KL 散度 | KL divergence | `Σ p log(p/q)`，分布之间的差异 | 6 |
| 似然 / 最大似然 | likelihood / MLE | 数据在模型下的概率 / 让它最大的参数 | 6 |
| 困惑度 | perplexity | `e^交叉熵` | 6、14、27 |
| 平滑 | smoothing | 给没见过的情况留一点概率 | 6、14 |

## 第二部分　神经网络（第 7–12 章）

| 中文 | 英文 | 一句话 | 章 |
|---|---|---|---|
| 经验风险 | empirical risk | 训练集上的平均损失 | 7 |
| 均方误差 | MSE | 预测与目标之差的平方的平均 | 7 |
| 残差 | residual | 预测减目标 | 7 |
| 闭式解 / 正规方程 | closed-form / normal equation | `θ = (AᵀA)⁻¹Aᵀy` | 7 |
| 全批量梯度下降 | full-batch GD | 每步用全部数据 | 7 |
| 小批量随机梯度下降 | mini-batch SGD | 每步用一小批数据 | 7 |
| 轮 | epoch | 把训练集完整过一遍 | 7、26 |
| 条件数 | condition number | 最大与最小弯曲度之比，决定收敛快慢 | 7 |
| 逻辑回归 | logistic regression | sigmoid 输出概率的二分类模型 | 7 |
| 二元交叉熵 | binary cross-entropy | `−[y log p + (1−y) log(1−p)]` | 7 |
| softmax 回归 | softmax / multinomial regression | 多分类的线性模型 | 7 |
| 线性可分 | linearly separable | 存在超平面完全分开两类 | 7 |
| 线性探针 | linear probe | 在冻结的特征上训练线性分类器来评价特征 | 7、20 |
| 自动求导 | automatic differentiation / autograd | 让计算机按链式法则自动算梯度 | 8 |
| 计算图 | computational graph | 节点是值、边是运算的有向无环图 | 8 |
| 前向 / 反向 | forward / backward | 算值 / 算梯度 | 8 |
| 反向传播 | backpropagation | 沿计算图逆向传播梯度 | 8 |
| 局部导数 | local derivative | 一个运算的输出对其输入的导数 | 8 |
| 拓扑排序 | topological sort | 使每个节点排在其输入之后的顺序 | 8 |
| 深度优先搜索 | depth-first search | 先走到底再回头的遍历方法 | 8 |
| 闭包 | closure | 记住了外部变量的内部函数 | 8 |
| 梯度累加 / 清零 | gradient accumulation / zero_grad | 多个来源的梯度相加 / 每步开始前归零 | 8 |
| 叶子节点 | leaf node | 不是由运算产生的节点（参数、输入） | 8 |
| 向量-雅可比积 | vector-Jacobian product | 反向传播每一步实际计算的东西 | 8 |
| 多层感知机 | multilayer perceptron (MLP) | 若干全连接层加激活函数 | 8、10 |
| 张量 | tensor | PyTorch 的多维数组 | 9 |
| 设备 | device | 张量所在的位置：cpu、cuda、mps | 9 |
| 连续 | contiguous | 内存里按行优先顺序紧密排列 | 9 |
| 叶子张量 | leaf tensor | 用户直接创建的张量（如参数） | 9 |
| grad_fn | grad_fn | 记录张量由哪个运算产生，用于反向传播 | 9 |
| 模块 | nn.Module | 包含参数和前向计算的组件 | 9 |
| 状态字典 | state_dict | 名字到参数张量的字典 | 9 |
| 数据集 / 数据加载器 | Dataset / DataLoader | 定义样本 / 打乱分批取样本 | 9 |
| 优化器 | optimizer | 管理参数更新规则的对象 | 9 |
| kernel | kernel | GPU 上执行的一个运算程序 | 9 |
| 同步 | synchronize | 等待 GPU 上提交的任务全部完成 | 9 |
| 泛化 | generalization | 在没见过的数据上表现好 | 10 |
| 数据泄漏 | data leakage | 测试信息以某种方式进入了训练 | 10、25 |
| 仿射变换 | affine transformation | 线性变换加平移：旋转、缩放、错切 | 10 |
| 轮 / 步 | epoch / step | 过一遍训练集 / 一次参数更新 | 10 |
| 提前停止 | early stopping | 用验证集最好时的参数 | 10 |
| 检查点 | checkpoint | 保存到磁盘的模型参数 | 10、18 |
| 混淆矩阵 | confusion matrix | 真实类别 × 预测类别的计数表 | 10 |
| 召回率 / 精确率 | recall / precision | 某类样本被认出的比例 / 预测为某类时的正确率 | 10 |
| 过拟合 / 欠拟合 | overfitting / underfitting | 记住训练数据 / 连训练数据都学不好 | 10 |
| 泛化差距 | generalization gap | 训练与验证表现之差 | 10 |
| 类别不均衡 | class imbalance | 各类别样本数相差悬殊 | 10 |
| 先验 | prior | 不看输入时各类别的概率 | 10 |
| 重采样 | resampling | 按权重有放回地抽样 | 10 |
| 动量 | momentum | 用过去梯度的加权和作为更新方向 | 11 |
| 自适应学习率 | adaptive learning rate | 每个参数按其梯度大小调整步长 | 11 |
| 一阶矩 / 二阶矩 | first / second moment | 梯度 / 梯度平方的移动平均 | 11 |
| 偏差修正 | bias correction | 补偿 m、v 从 0 开始造成的偏小 | 11 |
| 权重衰减 | weight decay | 每步把参数缩小一点 | 11 |
| 解耦 | decoupled | 衰减不经过自适应缩放（AdamW） | 11 |
| 学习率调度 | learning rate schedule | 学习率随步数变化的规则 | 11 |
| 预热 | warmup | 开始时学习率从小逐渐增大 | 11 |
| 余弦衰减 | cosine decay | 学习率按余弦曲线降到最小值 | 11 |
| 扇入 | fan-in | 一个神经元的输入个数 n | 11 |
| Xavier / Kaiming 初始化 | Xavier / Kaiming (He) init | 方差 1/n / 2/n 的初始化 | 11 |
| 梯度消失 / 爆炸 | vanishing / exploding gradient | 梯度逐层变小 / 变大 | 11 |
| 批归一化 / 层归一化 | BatchNorm / LayerNorm | 沿批 / 沿特征归一化 | 11 |
| 滑动平均 | running average | 评测时 BatchNorm 使用的统计量 | 11 |
| 残差连接 | residual connection | h ← h + f(h) | 11 |
| 梯度裁剪 | gradient clipping | 限制梯度向量的总长度 | 11 |
| 混合精度 | mixed precision | 计算用 16 位，参数用 32 位 | 11、28 |
| 损失缩放 | loss scaling | 放大损失以免 fp16 梯度下溢 | 11 |
| 卷积 / 互相关 | convolution / cross-correlation | 小模板在每个位置做点积（深度学习里不翻转） | 12 |
| 卷积核 / 滤波器 | kernel / filter | 卷积用的小权重矩阵 | 12 |
| 特征图 | feature map | 卷积层的一个输出通道 | 12 |
| 步长 / 填充 | stride / padding | 卷积核每次移动几格 / 四周补几圈 0 | 12 |
| 局部连接 | local connectivity | 每个输出只看输入的一小块 | 12 |
| 权重共享 | weight sharing | 所有位置用同一个卷积核 | 12、16 |
| 平移等变 / 不变 | translation equivariance / invariance | 输入平移，输出跟着平移 / 输出不变 | 12 |
| 感受野 | receptive field | 一个输出能看到的输入范围 | 12 |
| 最大池化 | max pooling | 每个区域取最大值 | 12 |
| 全局平均池化 | global average pooling | 每个通道对所有位置求平均 | 12 |
| im2col | im2col | 把每个窗口拉平成一行，卷积变成矩阵乘法 | 12 |
| 数据增强 | data augmentation | 对训练数据做随机变换以增加多样性 | 12 |
| 残差块 | residual block | 输出 = 输入 + 变换(输入) | 12 |
| 退化问题 | degradation problem | 普通网络加深后训练误差反而上升 | 12 |

## 第三部分　序列与语言模型（第 13–18 章）

| 中文 | 英文 | 一句话 | 章 |
|---|---|---|---|
| 连接时序分类 | Connectionist Temporal Classification, CTC | 对所有对齐方式求和的序列损失 | 13 |
| 空白 | blank | CTC 的特殊符号，分隔重复、填充时间步 | 13 |
| 路径 | path | 每个时间步选一个类别得到的序列 | 13 |
| 折叠 | collapse | 合并相邻重复、删去空白 | 13 |
| 前向算法 | forward algorithm | 用动态规划求所有路径的概率之和 | 13 |
| 贪心解码 | greedy decoding | 每步取最大，再折叠 | 13、17 |
| 前缀束搜索 | prefix beam search | 按折叠后的前缀合并路径的束搜索 | 13 |
| 循环神经网络 | recurrent neural network, RNN | 沿序列传递状态的网络 | 13 |
| 长短期记忆 | long short-term memory, LSTM | 带门和细胞状态的 RNN | 13 |
| 双向 RNN | bidirectional RNN | 从左到右、从右到左各一个 RNN | 13 |
| 编辑距离 | edit (Levenshtein) distance | 最少的插入、删除、替换次数 | 13 |
| 字符错误率 | character error rate, CER | 总编辑距离 ÷ 总字符数 | 13、25 |
| 尖峰行为 | peaky behavior | CTC 模型每个字符只在一个时间步上输出 | 13 |
| 分词 / 分词器 | tokenization / tokenizer | 把文字切成 token 并编号 | 14 |
| 子词 | subword | 介于字符和单词之间的片段 | 14 |
| 字节对编码 | byte pair encoding, BPE | 反复合并最常见相邻对的分词算法 | 14 |
| 合并规则 | merge rule | BPE 学到的一条 (左, 右) → 新符号 | 14 |
| 预切分 | pre-tokenization | BPE 之前用规则把文本切成块 | 14 |
| 字节级 BPE | byte-level BPE | 以 256 个字节为初始符号的 BPE | 14 |
| 语言模型 | language model | 给 token 序列分配概率的模型 | 14 |
| n 元语法 | n-gram | 只看前 n−1 个 token 的语言模型 | 14 |
| 数据稀疏 | data sparsity | 大部分组合在训练数据里从未出现 | 14 |
| 每字符比特 | bits per character | 与分词方式无关的语言模型指标 | 14 |
| 特殊 token | special token | 不由 BPE 产生、直接加进词表的 token | 14 |
| 注意力 | attention | 按查询与键的相似度对值加权平均 | 15 |
| 查询 / 键 / 值 | query / key / value | 我在找什么 / 我能被怎样匹配 / 我交出的内容 | 15 |
| 缩放点积注意力 | scaled dot-product attention | softmax(QKᵀ/√d) V | 15 |
| 自注意力 | self-attention | Q、K、V 来自同一个序列 | 15 |
| 交叉注意力 | cross-attention | Q 来自一个序列，K、V 来自另一个 | 15 |
| 因果 mask | causal mask | 位置 t 只能看 ≤ t 的位置 | 15 |
| 多头注意力 | multi-head attention | 并行的多个注意力，各学一种看法 | 15 |
| 置换等变 | permutation equivariance | 打乱输入，输出跟着同样打乱 | 15 |
| 位置编码 | positional encoding | 把位置信息加进模型的方法 | 15 |
| 归纳头 | induction head | 两层注意力配合完成"找到上次出现的位置，复制它后面的内容" | 15 |
| 分组查询注意力 | grouped-query attention, GQA | 多个查询头共用一组键值头 | 15、17 |
| FlashAttention | FlashAttention | 分块计算、不存完整权重矩阵的注意力实现 | 15 |
| 解码器式 Transformer | decoder-only Transformer | 只有因果自注意力的 Transformer，即 GPT | 16 |
| 残差流 | residual stream | 从嵌入到输出、各块往上做加法的主干 | 16 |
| 前置归一化 | Pre-Norm | 归一化只作用于支路的输入 | 16 |
| 均方根归一化 | RMSNorm | 除以均方根，不减均值 | 16 |
| 旋转位置编码 | rotary position embedding, RoPE | 按位置旋转 q、k | 16 |
| 门控线性单元 | SwiGLU | 一条支路经 silu 后乘上另一条 | 16 |
| 预训练 | pre-training | 在大量文本上预测下一个 token | 16 |
| 长度外推 | length extrapolation | 处理比训练时更长的序列 | 16 |
| 机制可解释性 | mechanistic interpretability | 研究模型内部具体电路的方法 | 16 |
| 自回归 | autoregressive | 每步依赖之前生成的所有 token | 17 |
| top-k 采样 | top-k sampling | 只在概率最大的 k 个里采样 | 17 |
| 核采样 | nucleus (top-p) sampling | 只在累计概率达到 p 的最少几个里采样 | 17 |
| 重复惩罚 | repetition penalty | 压低已出现 token 的 logit | 17 |
| 键值缓存 | KV cache | 保存过去位置的 k、v，避免重算 | 17 |
| 预填充 | prefill | 一次处理整个提示 | 17 |
| 解码 | decode | 逐个生成 token 的阶段 | 17 |
| 束搜索 | beam search | 保留累计概率最高的 k 个候选 | 17 |
| 投机解码 | speculative decoding | 小模型猜、大模型验证 | 17、34 |
| 浮点运算 | FLOP | 一次乘法或加法 | 18 |
| 模型算力利用率 | MFU | 实际 FLOP/秒 ÷ 硬件峰值 | 18 |
| 激活 | activations | 前向时为反向保存的中间结果 | 18 |
| 激活检查点 | activation checkpointing | 只存块的输入，反向时重算 | 18 |
| 梯度累积 | gradient accumulation | 多个小批的梯度相加后再更新 | 18 |
| 续训 | resume training | 从检查点接着训练 | 18 |
| 缩放定律 | scaling law | 损失与参数量、数据量的幂律关系 | 18 |
| 不可约损失 | irreducible loss | 模型再大也降不下去的部分 | 18 |
| 计算最优 | compute-optimal (Chinchilla) | 计算量固定时参数与数据的最优分配 | 18 |
| 吞吐 | throughput | 每秒处理的 token 数 | 18 |

## 第四部分　视觉编码器（第 19–21 章）

| 中文 | 英文 | 一句话 | 章 |
|---|---|---|---|
| 图像块 | patch | 图像切成的 p×p 小方块，ViT 的"词" | 19 |
| 块嵌入 | patch embedding | 把 patch 拉平后乘一个矩阵（= 卷积） | 19 |
| 分类 token | CLS token | 拼在序列最前面、用来汇总信息的可学习向量 | 19 |
| 视觉编码器 / 视觉塔 | vision encoder / vision tower | 把图像变成向量序列的网络 | 19 |
| 归纳偏置 | inductive bias | 模型结构里预先写进去的先验假设 | 19 |
| 置换不变 / 等变 | permutation invariant / equivariant | 打乱输入顺序，输出不变 / 跟着打乱 | 19 |
| 二维 RoPE | 2D RoPE | 按行号和列号分别旋转的位置编码 | 19 |
| 位置插值 | position interpolation | 把学到的位置表缩放到新的网格大小 | 19 |
| 窗口注意力 | window attention | 只在局部窗口内做注意力，降低计算量 | 19 |
| 对比学习 | contrastive learning | 拉近正例、推开负例的表示学习 | 20 |
| 双塔 | dual encoder / two-tower | 图像、文字各一个编码器，在同一个空间里比较 | 20 |
| 正例 / 负例 | positive / negative | 配对的 / 不配对的样本对 | 20 |
| 难负例 | hard negative | 和正例很像、容易混淆的负例 | 20 |
| 假负例 | false negative | 实际上配对、却被当成负例的样本对 | 20 |
| InfoNCE | InfoNCE | CLIP 用的 B 选 1 交叉熵损失 | 20 |
| 零样本 | zero-shot | 不用该任务的训练样本，直接用文字描述做分类 | 20 |
| 召回率@k | Recall@k | 正确答案排在前 k 名的比例 | 20 |
| 模态鸿沟 | modality gap | 图像向量和文字向量在空间里分开聚集的现象 | 20 |
| 交并比 | IoU | 交集面积 / 并集面积 | 21 |
| 非极大值抑制 | NMS | 去掉与高分框重叠的重复框 | 21 |
| 精确率 / 召回率 | precision / recall | 预测对的比例 / 找到的比例 | 21 |
| 平均精确率 | AP | 精确率–召回率曲线下的面积 | 21 |
| 锚框 | anchor | 预设的模板框，检测器预测相对它的修正 | 21 |
| 热图 | heatmap | 每个位置是物体中心的概率 | 21 |
| focal loss | focal loss | 降低易分样本权重的交叉熵 | 21 |
| 版面分析 | layout analysis | 把页面分成有语义的区块 | 21 |
| 阅读顺序 | reading order | 区块或文字行应该被阅读的先后 | 21 |
| XY-cut | XY-cut | 按投影空隙递归切分版面 | 21 |
| 流水线 OCR | pipeline OCR | 检测 → 排序 → 识别的多模型系统 | 21 |

## 第五部分　视觉语言模型（第 22–24 章）

| 中文 | 英文 | 一句话 | 章 |
|---|---|---|---|
| 视觉塔 | vision tower / vision encoder | 把图像变成特征序列的编码器 | 22 |
| 视觉 token | visual token | 送进语言模型的、代表图像的向量 | 22 |
| 占位符 | placeholder token | 序列里代表视觉 token 位置的特殊 token | 22 |
| 空间到深度 | space-to-depth / pixel shuffle | 把相邻 patch 拼接成一个更宽的 token | 22 |
| Q-Former | Q-Former | 用可学习查询从图像特征中提取固定数量 token 的模块 | 22 |
| 对齐预训练 | alignment pretraining | 只训练连接器的第一阶段 | 22 |
| 早期融合 | early fusion | 视觉 token 直接进入语言模型的输入序列 | 22 |
| 聊天模板 | chat template | 把对话排成 token 序列的固定格式 | 23、35 |
| 系统提示 | system prompt | 设定助手行为的、不计算损失的开头消息 | 23 |
| 定位 | grounding / detection | 根据描述输出物体的框 | 23 |
| 指代 | referring | 根据框输出描述 | 23 |
| 场景文字 | scene text | 自然图像里出现的文字 | 23 |
| 指令数据 | instruction data | 问答、对话形式的训练数据 | 23 |
| 数据配比 | data mixture | 各类数据在训练中所占的比例 | 23 |
| 幻觉 | hallucination | 输出与图像不符的内容 | 23 |
| 负样本 | negative sample | 答案为"没有 / 否"的样本 | 23 |
| 对抗负样本 | adversarial negative | 与图中内容相似、最容易被误认的不存在物体 | 23 |
| 处理器 | processor | 把原始图片和文字变成模型输入张量的组件 | 24 |
| 原生分辨率 | native resolution | 保持宽高比、按需调整 token 数的图像输入方式 | 24 |
| 智能缩放 | smart_resize | 把长宽变成 factor 的倍数、像素数落在上下限之间 | 24 |
| 时间 patch | temporal patch | 视频的"时空小块"；静态图复制成 2 帧 | 24 |
| 网格大小 | image_grid_thw | 每张图切块后的 (时间, 行, 列) | 24 |
| 多模态旋转位置编码 | M-RoPE | 每个 token 有时间、行、列三个位置编号 | 24 |
| 交错式 | interleaved | 频率按 T H W T H W … 轮流分给三个轴 | 24 |
| 线性注意力 | linear attention | 用固定大小的状态代替 KV cache 的注意力 | 24 |
| delta 规则 | delta rule | 只把"预测误差"写进联想记忆的更新方式 | 24 |
| 切子图 | image splitting / tiling | 把大图切成固定大小的块分别编码 | 24 |
| 像素重排 | pixel shuffle | 把相邻的 k×k 个 patch 拼成一个 token | 24 |
| meta 设备 | meta device | 只有形状、没有数据的"假"设备，用于数参数 | 24 |

## 第六部分　评测与监督微调（第 25–28 章）

| 中文 | 英文 | 一句话 | 章 |
|---|---|---|---|
| 词错误率 | word error rate, WER | 词级的编辑距离 / 参考答案的总词数 | 25 |
| 归一化编辑距离 | normalized edit distance, NED | 编辑距离 / 两者中较长的长度 | 25 |
| 平均归一化编辑相似度 | ANLS | 1 − NED，NED ≥ 0.5 记 0，多个答案取最好 | 25 |
| 字符检测匹配 | Character Detection Matching, CDM | 渲染公式后按符号和位置配对，算 F1 | 25 |
| 树编辑距离相似度 | TEDS | 1 − 树的编辑距离 / 最大节点数 | 25 |
| 语料级 | corpus-level | 先把所有样本的分子、分母分别加起来再相除 | 25 |
| 宏平均 | macro-average | 先算每个样本（或每类）的指标再平均 | 25 |
| 自助法 | bootstrap | 有放回地重抽测试集，估计指标的波动 | 25 |
| 配对比较 | paired comparison | 两个系统在同一批样本上比较差，抵消样本难度的波动 | 25、36 |
| 分桶 | slicing / bucketing | 按属性分组后分别算指标 | 25 |
| 基准污染 | benchmark contamination | 公开基准的题目和答案进入了预训练语料 | 25 |
| 多重比较 | multiple comparisons | 比较很多次，总会有"显著"的结果 | 25、36 |
| 评测协议 | evaluation protocol | 跑实验之前写好的评测规则 | 25 |
| 监督微调 | supervised fine-tuning, SFT | 用"问题 → 正确答案"的数据和交叉熵训练 | 26 |
| 只在回答上算损失 | completion-only loss | 提示部分的标签设成 −100 | 26 |
| 提示权重 | prompt loss weight | 提示部分的损失乘的系数 | 26 |
| 按 token 平均 | token-level averaging | 所有计算损失的 token 一样重要 | 26 |
| 按样本平均 | sample-level averaging | 每条样本一样重要 | 26 |
| 补齐 | padding | 把短序列用 `<pad>` 补到同样长 | 26 |
| 打包 | packing | 把多条短样本首尾相接装进一条序列 | 26 |
| 分块 mask | block-diagonal mask | 只允许看同一条样本内的位置 | 26 |
| 变长注意力 | varlen attention | 用 cu_seqlens 代替 mask 的注意力内核 | 26 |
| 平台期 | plateau | 学习曲线长时间不动的阶段 | 26 |
| 达到阈值的步数 | steps to threshold | 第一次达到某个指标的训练步数 | 26 |
| 对数概率 | log-probability | log π(token \| 前文)，≤ 0 | 27 |
| 对数似然 | log-likelihood | 一个序列所有 token 对数概率之和 | 27 |
| 长度偏差 | length bias | 求和偏向短回答，平均偏向长回答 | 27、31 |
| 似然打分 | likelihood scoring | 比较候选答案的对数概率来做选择题 | 27 |
| 校准 | calibration | 置信度与实际准确率是否一致 | 27 |
| 期望校准误差 | expected calibration error, ECE | 各段 \|准确率 − 置信度\| 的加权平均 | 27 |
| 可靠性图 | reliability diagram | 按置信度分段，画出每段的准确率 | 27 |
| AUROC | area under the ROC curve | 随机正例分数高于随机负例的概率 | 27 |
| 温度缩放 | temperature scaling | logits 除以 τ 来调整置信度 | 27 |
| 重新分词 | re-tokenization | 把文字再分一次词，可能得到和生成时不同的 token | 27 |
| 静态内存 | static memory | 参数、梯度、优化器状态，与批大小无关 | 28 |
| 激活值 | activations | 前向时保存、反向时要用的中间结果 | 28 |
| 主权重 | master weights | 混合精度训练时 fp32 的那份权重 | 28 |
| 梯度检查点 | gradient checkpointing / activation recomputation | 前向少存、反向重算 | 28 |
| 低秩适配 | LoRA (Low-Rank Adaptation) | 冻结 W，训练 ΔW = B·A | 28 |
| 秩 | rank (r) | LoRA 小矩阵的内侧维度 | 28 |
| 合并 | merge | 把 (α/r)·B·A 加进 W | 28、35 |
| 元设备 | meta device | 只有形状没有数据的张量，用来数参数 | 28 |
| 缓存分配器 | caching allocator | PyTorch 留着已释放的显存以便复用 | 28 |
| ZeRO / FSDP | Zero Redundancy Optimizer / Fully Sharded Data Parallel | 把静态内存切分到多张卡 | 28 |

## 第七部分　偏好与强化学习（第 29–32 章）

| 中文 | 英文 | 一句话 | 章 |
|---|---|---|---|
| 策略 | policy | 模型：给定输入，回答的概率分布 | 29 |
| 轨迹 | trajectory / rollout | 一个完整的回答 | 29 |
| 采样 | rollout / sampling | 用当前的模型生成回答 | 29 |
| 可验证奖励 | verifiable reward | 能用程序自动、客观地算出的奖励 | 29 |
| 策略梯度 | policy gradient | E[R · ∇log π] | 29 |
| 对数导数技巧 | log-derivative trick / score function | ∇π = π · ∇log π | 29 |
| 优势 | advantage | R − 基线：比平均好多少 | 29 |
| 留一基线 | leave-one-out (RLOO) | 用其余样本的平均当基线 | 29 |
| 组内标准化 | group normalization (GRPO) | 每道题的 G 个回答内部减均值、除标准差 | 29 |
| 参考模型 | reference model | RL 开始前的模型，冻结 | 29 |
| KL 估计量 | KL estimator (k1/k2/k3) | 用采样的 token 估计 KL | 29 |
| 重要性比值 | importance ratio | ρ = π / π_old | 29 |
| 裁剪 | clipping | 把 ρ 限制在 [1 − ε, 1 + ε] | 29 |
| 奖励投机 | reward hacking | 钻奖励函数的空子拿高分 | 29 |
| 价值网络 | value network / critic | 预测"从这里出发能拿多少分"的网络 | 29 |
| 偏好数据 | preference data | (输入, 好回答, 坏回答) | 30 |
| 被偏好的 / 不被偏好的回答 | chosen / rejected（y_w / y_l） |  | 30 |
| Bradley–Terry 模型 | Bradley–Terry model | P(y_w ≻ y_l) = σ(r_w − r_l) | 30 |
| 奖励模型 | reward model | 从偏好数据学出来的打分模型 | 30 |
| 配分函数 | partition function（Z） | 让 π* 的和为 1 的归一化常数 | 30 |
| 直接偏好优化 | DPO | 不用奖励模型、不采样，直接从偏好数据训练策略 | 30 |
| 隐含奖励 | implicit reward | β (log π_θ − log π_ref) | 30 |
| 偏好准确率 | preference accuracy | 隐含奖励把 chosen 排在前面的比例 | 30 |
| 似然位移 | likelihood displacement | DPO 训练中 chosen 的概率也下降 | 30 |
| 幻觉修正对 | hallucination correction pair | 修正前（rejected）和修正后（chosen）的描述 | 30 |
| 离线 / 在线 DPO | offline / online DPO | 偏好数据是否用当前模型重新采样 | 30 |
| 组相对策略优化 | GRPO | 组内标准化的优势 + 裁剪 + KL | 31 |
| 格式闸门 | format gate | 格式不对一律 0 分 | 31 |
| 部分得分 | partial credit | 用 IoU、1 − NED 等连续的奖励 | 31 |
| 汇总方式 | loss aggregation | 逐 token 的损失怎样合成一个数 | 31 |
| 熵崩溃 | entropy collapse | 输出越来越确定，组内回答变得一样 | 31 |
| clip-higher | clip-higher | 裁剪的上界放宽（ε_high > ε_low） | 31 |
| 动态采样 | dynamic sampling | 去掉没有梯度的组并补采 | 31 |
| 超长奖励整形 | overlong reward shaping | 被截断的回答过滤掉或按长度扣分 | 31 |
| 被裁剪的比例 | clip fraction | 被裁剪的 token 占回答 token 的比例 | 31 |
| 多任务 RL | multi-task RL / Mix-RL | 多种任务、各自的奖励，混在一个批里训练 | 31 |
| 遗忘 / 灾难性遗忘 | (catastrophic) forgetting | 新任务上训练后，旧能力下降 | 32 |
| 探测 KL | probe KL | 在固定的探测输入上，新模型和起点模型的输出分布之差 | 32 |
| 支配 | dominate | 两项都不差、至少一项更好 | 32 |
| 帕累托前沿 | Pareto front | 不被任何点支配的点 | 32 |
| 重放 | replay / rehearsal | 训练新任务时混入旧任务的数据 | 32 |
| 能力配比 / 数据配比 | data mixture | 各种任务的数据各占多少 | 32 |
| 拒绝采样微调 | rejection sampling fine-tuning (RFT) | 只保留模型自己做对的样本，做 SFT | 32 |
| 模型合并 | model merging / weight interpolation | 新旧模型的权重加权平均 | 32 |
| 稳定性—可塑性 | stability–plasticity | 不忘旧的和学会新的之间的矛盾 | 32 |
| 计算预算 | compute budget | 一种方法总共要算多少 | 32 |

## 第八部分　文档 OCR 与真实世界（第 33–36 章）

| 中文 | 英文 | 一句话 | 章 |
|---|---|---|---|
| 两阶段解析 | two-stage / pipeline parsing | 先版面检测，再逐个区块识别 | 33 |
| 端到端解析 | end-to-end parsing | 整页图直接输出全部内容 | 33 |
| 伪标签 | pseudo-label | 模型写的、当作标签用的答案 | 33 |
| 共识 | consensus | 多个专家的答案互相印证 | 33 |
| 中心点 | medoid | 和组内其他成员最相似的那个成员 | 33、34 |
| 渲染校验 | render verification | 把答案重新画出来和原图比 | 33 |
| 归一化互相关 | normalized cross-correlation (NCC) | 去掉均值之后两张图的余弦相似度 | 33 |
| 自洽度 | self-consistency | 同一个输入多次采样的回答之间的一致程度 | 33 |
| 难例挖掘 | hard example mining | 挑出模型没把握、标签可信的样本 | 33 |
| 按簇配额 | cluster quota sampling | 先聚类，每簇分配名额，再簇内抽样 | 33 |
| 近似去重 | near-duplicate detection | 找出几乎相同的样本 | 33 |
| MinHash | MinHash | 签名相同的概率等于 Jaccard 相似度的哈希 | 33 |
| 局部敏感哈希 | locality-sensitive hashing (LSH) | 相似的东西更可能落进同一个桶 | 33 |
| 按组划分 | group split | 同一组的样本只进训练或只进验证 | 33 |
| 定向合成 | targeted synthesis | 针对覆盖空白和失败模式生成数据 | 33 |
| 三元组共识 | ensemble triplet consensus (ETC) | 在 N 个专家里选两两最一致的三个，取中心点当标签 | 34 |
| 一致度分档 | agreement tiers | 高（≥ 0.9）直接收下、中低送去修正 | 34 |
| 渲染引导的修正与评判 | render-guided refine-and-judge | 把读法渲染出来给大模型对照原图修改，修改结果再投票 | 34 |
| 多因素挖掘 | multi-factor sample mining | 按专家一致度、模型自洽度、簇加权抽样 | 34 |
| 模型自洽度 | self-consistency | 多次采样的读法之间有多一致 | 34 |
| 文字锚定 | text anchoring | 训练时同时预测答案在图里的位置 | 34 |
| 按查询条件化的掩码解码器 | query-conditioned mask decoder | 根据问题预测图里相关区域的掩码 | 34 |
| Dice 损失 | Dice loss | 1 − 2·重叠 / (预测 + 真值)，对小前景敏感 | 34 |
| 多 token 预测 | multi-token prediction (MTP) | 额外预测往后第二个 token | 34 |
| Mix-RL | mixed-task RL | 多种任务、各自的奖励，一起做 RL | 34 |
| 多教师在线蒸馏 | MOPD | 本报告对照的另一种能力整合方法 | 34 |
| 生成提示 | generation prompt | 提示末尾的"助手开始说话"的标记 | 35 |
| 前缀法 | prefix masking | 提示是完整序列的前缀，前缀之后的是回答 | 35 |
| LoRA 目标 | target modules | 加 LoRA 的线性层 | 35 |
| 阻断项检查 | blocker check | 训练前确认模型、kernel、库、显存都没有问题 | 35 |
| 参考实现 | reference implementation | 没有优化 kernel 时用的纯 PyTorch 实现（正确但慢） | 35 |
| 融合的 LM head | fused LM head | 不生成完整 logits、直接算目标 token 对数概率的 kernel | 35 |
| 奖励函数 | reward function | trl 里 `f(completions, **kwargs) -> list[float]` | 35 |
| 研究问题 | research question | 能用"是 / 否 / 看不出"回答的一句话，写明和谁比、怎么测 | 36 |
| 预测 | prediction | 运行之前写下的、带方向和大致大小的预期结果 | 36 |
| 对照（基线） | control / baseline | 除了要研究的那个因素之外，其他条件都相同的配置 | 36 |
| 预算口径 | compute budget | 怎样定义"同样的代价"：步数、样本数、计算量或墙钟时间 | 36 |
| 种子表 | seed table | 同一配置多个种子的 n、均值、标准差、最小、最大 | 36 |
| 相变 | phase transition | 同一配置的不同种子落进几种明显不同的结果（学会 / 没学会） | 36 |
| 运行的身份证（manifest） | run manifest | 配置、种子、数据内容哈希，以及由它们算出的短 id | 36 |
| 方案冻结 | protocol freeze | 所有选择都已确定、之后不再修改，此时才评测测试集 | 36 |
| 分桶报告 | bucketed report | 按事先定好的维度分组，每组报告样本数、均值和区间 | 36 |
| 效应大小 | effect size | 差异本身有多大，和它是否显著是两回事 | 36 |
| 能检测的最小差异 | minimum detectable difference | 给定样本数下，这次实验有能力区分的最小差异 | 36 |

## 英文索引

| 英文 | 中文 | 首次出现 |
|---|---|---|
| 2D RoPE | 二维 RoPE | 第 19 章 |
| activation checkpointing | 激活检查点 | 第 18 章 |
| activations | 激活 | 第 18 章 |
| activations | 激活值 | 第 28 章 |
| adaptive learning rate | 自适应学习率 | 第 11 章 |
| advantage | 优势 | 第 29 章 |
| adversarial negative | 对抗负样本 | 第 23 章 |
| affine transformation | 仿射变换 | 第 10 章 |
| agreement tiers | 一致度分档 | 第 34 章 |
| aliasing | 混叠 | 第 2 章 |
| alignment pretraining | 对齐预训练 | 第 22 章 |
| anchor | 锚框 | 第 21 章 |
| ANLS | 平均归一化编辑相似度 | 第 25 章 |
| anti-aliasing | 抗锯齿 | 第 2 章 |
| AP | 平均精确率 | 第 21 章 |
| area under the ROC curve | AUROC | 第 27 章 |
| aspect ratio | 长宽比 | 第 2 章 |
| assertion | 断言 | 第 1 章 |
| attention | 注意力 | 第 15 章 |
| automatic differentiation / autograd | 自动求导 | 第 8 章 |
| autoregressive | 自回归 | 第 17 章 |
| backpropagation | 反向传播 | 第 8 章 |
| baseline | 基线 | 第 2 章 |
| BatchNorm / LayerNorm | 批归一化 / 层归一化 | 第 11 章 |
| beam search | 束搜索 | 第 17 章 |
| benchmark contamination | 基准污染 | 第 25 章 |
| bias correction | 偏差修正 | 第 11 章 |
| bidirectional RNN | 双向 RNN | 第 13 章 |
| binary cross-entropy | 二元交叉熵 | 第 7 章 |
| bits per character | 每字符比特 | 第 14 章 |
| blank | 空白 | 第 13 章 |
| block-diagonal mask | 分块 mask | 第 26 章 |
| blocker check | 阻断项检查 | 第 35 章 |
| bootstrap | 自助法 | 第 25 章 |
| bounding box | 边界框 | 第 2 章 |
| Bradley–Terry model | Bradley–Terry 模型 | 第 30 章 |
| broadcasting | 广播 | 第 1 章 |
| bucketed report | 分桶报告 | 第 36 章 |
| byte pair encoding, BPE | 字节对编码 | 第 14 章 |
| byte-level BPE | 字节级 BPE | 第 14 章 |
| caching allocator | 缓存分配器 | 第 28 章 |
| calibration | 校准 | 第 27 章 |
| canonical form | 规范写法 | 第 3 章 |
| catastrophic cancellation | 灾难性抵消 | 第 4 章 |
| (catastrophic) forgetting | 遗忘 / 灾难性遗忘 | 第 32 章 |
| categorical distribution | 类别分布 | 第 6 章 |
| causal mask | 因果 mask | 第 15 章 |
| central difference | 中心差分 | 第 5 章 |
| chain rule | 链式法则 | 第 5 章 |
| channel | 通道 | 第 2 章 |
| Character Detection Matching, CDM | 字符检测匹配 | 第 25 章 |
| character error rate, CER | 字符错误率 | 第 13 章 |
| charset / vocabulary | 字符集 / 词表 | 第 3 章 |
| chat template | 聊天模板 | 第 23 章 |
| checkpoint | 检查点 | 第 10 章 |
| chosen / rejected（y_w / y_l） | 被偏好的 / 不被偏好的回答 | 第 30 章 |
| class / object | 类 / 对象 | 第 1 章 |
| class imbalance | 类别不均衡 | 第 10 章 |
| clip fraction | 被裁剪的比例 | 第 31 章 |
| clip-higher | clip-higher | 第 31 章 |
| clipping | 裁剪 | 第 29 章 |
| closed-form / normal equation | 闭式解 / 正规方程 | 第 7 章 |
| closure | 闭包 | 第 8 章 |
| CLS token | 分类 token | 第 19 章 |
| cluster quota sampling | 按簇配额 | 第 33 章 |
| code point | 码位 | 第 3 章 |
| collapse | 折叠 | 第 13 章 |
| completion-only loss | 只在回答上算损失 | 第 26 章 |
| computational graph | 计算图 | 第 8 章 |
| compute budget | 计算预算 | 第 32 章 |
| compute budget | 预算口径 | 第 36 章 |
| compute-optimal (Chinchilla) | 计算最优 | 第 18 章 |
| condition number | 条件数 | 第 7 章 |
| confusion matrix | 混淆矩阵 | 第 10 章 |
| Connectionist Temporal Classification, CTC | 连接时序分类 | 第 13 章 |
| connector / projector | 连接器 | 第 0 章 |
| consensus | 共识 | 第 33 章 |
| contiguous | 连续 | 第 9 章 |
| contour | 等高线 | 第 5 章 |
| contrastive learning | 对比学习 | 第 20 章 |
| control / baseline | 对照（基线） | 第 36 章 |
| convolution / cross-correlation | 卷积 / 互相关 | 第 12 章 |
| corpus-level | 语料级 | 第 25 章 |
| cosine decay | 余弦衰减 | 第 11 章 |
| cosine similarity | 余弦相似度 | 第 4 章 |
| cross-attention | 交叉注意力 | 第 15 章 |
| cross-entropy | 交叉熵 | 第 6 章 |
| data augmentation | 数据增强 | 第 12 章 |
| data leakage | 数据泄漏 | 第 10 章 |
| data mixture | 数据配比 | 第 23 章 |
| data mixture | 能力配比 / 数据配比 | 第 32 章 |
| data sparsity | 数据稀疏 | 第 14 章 |
| Dataset / DataLoader | 数据集 / 数据加载器 | 第 9 章 |
| decode | 解码 | 第 17 章 |
| decoder-only Transformer | 解码器式 Transformer | 第 16 章 |
| decoupled | 解耦 | 第 11 章 |
| degradation problem | 退化问题 | 第 12 章 |
| delta rule | delta 规则 | 第 24 章 |
| depth-first search | 深度优先搜索 | 第 8 章 |
| derivative | 导数 | 第 5 章 |
| device | 设备 | 第 9 章 |
| Dice loss | Dice 损失 | 第 34 章 |
| directional derivative | 方向导数 | 第 5 章 |
| dominate | 支配 | 第 32 章 |
| dot product | 点积 / 内积 | 第 4 章 |
| DPO | 直接偏好优化 | 第 30 章 |
| dual encoder / two-tower | 双塔 | 第 20 章 |
| dynamic sampling | 动态采样 | 第 31 章 |
| early fusion | 早期融合 | 第 22 章 |
| early stopping | 提前停止 | 第 10 章 |
| edit (Levenshtein) distance | 编辑距离 | 第 13 章 |
| effect size | 效应大小 | 第 36 章 |
| empirical risk | 经验风险 | 第 7 章 |
| encoding | 编码 | 第 3 章 |
| end-to-end parsing | 端到端解析 | 第 33 章 |
| ensemble triplet consensus (ETC) | 三元组共识 | 第 34 章 |
| entropy | 熵 | 第 6 章 |
| entropy collapse | 熵崩溃 | 第 31 章 |
| epoch | 轮 | 第 7 章 |
| epoch / step | 轮 / 步 | 第 10 章 |
| evaluation protocol | 评测协议 | 第 25 章 |
| expectation / variance | 期望 / 方差 | 第 6 章 |
| expected calibration error, ECE | 期望校准误差 | 第 27 章 |
| false negative | 假负例 | 第 20 章 |
| fan-in | 扇入 | 第 11 章 |
| feature map | 特征图 | 第 12 章 |
| first / second moment | 一阶矩 / 二阶矩 | 第 11 章 |
| FlashAttention | FlashAttention | 第 15 章 |
| FLOP | 浮点运算 | 第 18 章 |
| FLOPs | 浮点运算次数 | 第 4 章 |
| focal loss | focal loss | 第 21 章 |
| format gate | 格式闸门 | 第 31 章 |
| forward / backward | 前向 / 反向 | 第 8 章 |
| forward algorithm | 前向算法 | 第 13 章 |
| freeze | 冻结 | 第 0 章 |
| full-batch GD | 全批量梯度下降 | 第 7 章 |
| fused LM head | 融合的 LM head | 第 35 章 |
| generalization | 泛化 | 第 10 章 |
| generalization gap | 泛化差距 | 第 10 章 |
| generation prompt | 生成提示 | 第 35 章 |
| generator | 生成器 | 第 1 章 |
| global average pooling | 全局平均池化 | 第 12 章 |
| glyph | 字形 | 第 2 章 |
| grad_fn | grad_fn | 第 9 章 |
| gradient | 梯度 | 第 0 章 |
| gradient accumulation | 梯度累积 | 第 18 章 |
| gradient accumulation / zero_grad | 梯度累加 / 清零 | 第 8 章 |
| gradient check | 梯度检查 | 第 5 章 |
| gradient checkpointing / activation recomputation | 梯度检查点 | 第 28 章 |
| gradient clipping | 梯度裁剪 | 第 11 章 |
| gradient descent | 梯度下降 | 第 5 章 |
| grammar | 文法 | 第 3 章 |
| grayscale / RGB image | 灰度图 / 彩色图 | 第 2 章 |
| greedy decoding | 贪心解码 | 第 13 章 |
| grounding / detection | 定位 | 第 23 章 |
| group normalization (GRPO) | 组内标准化 | 第 29 章 |
| group split | 按组划分 | 第 33 章 |
| grouped-query attention, GQA | 分组查询注意力 | 第 15 章 |
| GRPO | 组相对策略优化 | 第 31 章 |
| hallucination | 幻觉 | 第 23 章 |
| hallucination correction pair | 幻觉修正对 | 第 30 章 |
| hard example mining | 难例挖掘 | 第 33 章 |
| hard negative | 难负例 | 第 20 章 |
| heatmap | 热图 | 第 21 章 |
| Hessian / condition number | Hessian / 条件数 | 第 5 章 |
| idempotent | 幂等 | 第 3 章 |
| identity matrix | 单位矩阵 | 第 4 章 |
| im2col | im2col | 第 12 章 |
| image splitting / tiling | 切子图 | 第 24 章 |
| image_grid_thw | 网格大小 | 第 24 章 |
| implicit reward | 隐含奖励 | 第 30 章 |
| importance ratio | 重要性比值 | 第 29 章 |
| induction head | 归纳头 | 第 15 章 |
| inductive bias | 归纳偏置 | 第 19 章 |
| InfoNCE | InfoNCE | 第 20 章 |
| instruction data | 指令数据 | 第 23 章 |
| interleaved | 交错式 | 第 24 章 |
| IoU | 交并比 | 第 21 章 |
| irreducible loss | 不可约损失 | 第 18 章 |
| kernel | kernel | 第 9 章 |
| kernel / filter | 卷积核 / 滤波器 | 第 12 章 |
| KL divergence | KL 散度 | 第 6 章 |
| KL estimator (k1/k2/k3) | KL 估计量 | 第 29 章 |
| KV cache | 键值缓存 | 第 17 章 |
| language model | 语言模型 | 第 14 章 |
| law of large numbers | 大数定律 | 第 6 章 |
| layout analysis | 版面分析 | 第 21 章 |
| leaf node | 叶子节点 | 第 8 章 |
| leaf tensor | 叶子张量 | 第 9 章 |
| learning rate | 学习率 | 第 0 章 |
| learning rate schedule | 学习率调度 | 第 11 章 |
| leave-one-out (RLOO) | 留一基线 | 第 29 章 |
| length bias | 长度偏差 | 第 27 章 |
| length extrapolation | 长度外推 | 第 16 章 |
| letterbox | letterbox | 第 2 章 |
| likelihood / MLE | 似然 / 最大似然 | 第 6 章 |
| likelihood displacement | 似然位移 | 第 30 章 |
| likelihood scoring | 似然打分 | 第 27 章 |
| limit | 极限 | 第 5 章 |
| linear / fully connected layer | 线性层 | 第 4 章 |
| linear attention | 线性注意力 | 第 24 章 |
| linear probe | 线性探针 | 第 7 章 |
| linear transformation | 线性变换 | 第 4 章 |
| linearly separable | 线性可分 | 第 7 章 |
| list comprehension | 列表推导式 | 第 1 章 |
| local connectivity | 局部连接 | 第 12 章 |
| local derivative | 局部导数 | 第 8 章 |
| locality-sensitive hashing (LSH) | 局部敏感哈希 | 第 33 章 |
| log-derivative trick / score function | 对数导数技巧 | 第 29 章 |
| log-likelihood | 对数似然 | 第 27 章 |
| log-probability | 对数概率 | 第 27 章 |
| log-sum-exp | logsumexp | 第 6 章 |
| logistic regression | 逻辑回归 | 第 7 章 |
| logits | logits | 第 6 章 |
| long short-term memory, LSTM | 长短期记忆 | 第 13 章 |
| LoRA (Low-Rank Adaptation) | 低秩适配 | 第 28 章 |
| loss | 损失 | 第 0 章 |
| loss aggregation | 汇总方式 | 第 31 章 |
| loss scaling | 损失缩放 | 第 11 章 |
| M-RoPE | 多模态旋转位置编码 | 第 24 章 |
| macro-average | 宏平均 | 第 25 章 |
| master weights | 主权重 | 第 28 章 |
| matrix multiplication | 矩阵乘法 | 第 4 章 |
| max pooling | 最大池化 | 第 12 章 |
| mechanistic interpretability | 机制可解释性 | 第 16 章 |
| medoid | 中心点 | 第 33 章 |
| merge | 合并 | 第 28 章 |
| merge rule | 合并规则 | 第 14 章 |
| meta device | meta 设备 | 第 24 章 |
| meta device | 元设备 | 第 28 章 |
| MFU | 模型算力利用率 | 第 18 章 |
| MinHash | MinHash | 第 33 章 |
| mini-batch | 小批量 | 第 1 章 |
| mini-batch SGD | 小批量随机梯度下降 | 第 7 章 |
| minimum detectable difference | 能检测的最小差异 | 第 36 章 |
| mixed precision | 混合精度 | 第 11 章 |
| mixed-task RL | Mix-RL | 第 34 章 |
| modality gap | 模态鸿沟 | 第 20 章 |
| model | 模型 | 第 0 章 |
| model merging / weight interpolation | 模型合并 | 第 32 章 |
| module / package | 模块 / 包 | 第 1 章 |
| momentum | 动量 | 第 11 章 |
| MOPD | 多教师在线蒸馏 | 第 34 章 |
| MSE | 均方误差 | 第 7 章 |
| multi-factor sample mining | 多因素挖掘 | 第 34 章 |
| multi-head attention | 多头注意力 | 第 15 章 |
| multi-task RL / Mix-RL | 多任务 RL | 第 31 章 |
| multi-token prediction (MTP) | 多 token 预测 | 第 34 章 |
| multilayer perceptron (MLP) | 多层感知机 | 第 8 章 |
| multiple comparisons | 多重比较 | 第 25 章 |
| n-gram | n 元语法 | 第 14 章 |
| native resolution | 原生分辨率 | 第 24 章 |
| near-duplicate detection | 近似去重 | 第 33 章 |
| nearest / bilinear / area | 最近邻 / 双线性 / 区域平均 | 第 2 章 |
| negative sample | 负样本 | 第 23 章 |
| NMS | 非极大值抑制 | 第 21 章 |
| nn.Module | 模块 | 第 9 章 |
| norm | 范数 / 长度 | 第 4 章 |
| normalization | 归一化 | 第 3 章 |
| normalized cross-correlation (NCC) | 归一化互相关 | 第 33 章 |
| normalized edit distance, NED | 归一化编辑距离 | 第 25 章 |
| nucleus (top-p) sampling | 核采样 | 第 17 章 |
| OCR | 光学字符识别 | 第 0 章 |
| offline / online DPO | 离线 / 在线 DPO | 第 30 章 |
| one-hot | 独热向量 | 第 1 章 |
| optimizer | 优化器 | 第 9 章 |
| orthogonal matrix | 正交矩阵 | 第 4 章 |
| outer product | 外积 | 第 4 章 |
| overfitting / underfitting | 过拟合 / 欠拟合 | 第 10 章 |
| overlong reward shaping | 超长奖励整形 | 第 31 章 |
| packing | 打包 | 第 26 章 |
| padding | 补齐 | 第 26 章 |
| padding / mask | 补齐 / 掩码 | 第 1 章 |
| paired comparison | 配对比较 | 第 25 章 |
| parameter | 参数 | 第 0 章 |
| Pareto front | 帕累托前沿 | 第 32 章 |
| parsing | 解析 | 第 3 章 |
| partial credit | 部分得分 | 第 31 章 |
| partial derivative | 偏导数 | 第 5 章 |
| partition function（Z） | 配分函数 | 第 30 章 |
| patch | 图像块 | 第 19 章 |
| patch embedding | 块嵌入 | 第 19 章 |
| path | 路径 | 第 13 章 |
| peaky behavior | 尖峰行为 | 第 13 章 |
| permutation equivariance | 置换等变 | 第 15 章 |
| permutation invariant / equivariant | 置换不变 / 等变 | 第 19 章 |
| perplexity | 困惑度 | 第 6 章 |
| phase transition | 相变 | 第 36 章 |
| pipeline OCR | 流水线 OCR | 第 21 章 |
| pixel | 像素 | 第 2 章 |
| pixel shuffle | 像素重排 | 第 24 章 |
| placeholder token | 占位符 | 第 22 章 |
| plateau | 平台期 | 第 26 章 |
| policy | 策略 | 第 29 章 |
| policy gradient | 策略梯度 | 第 29 章 |
| position interpolation | 位置插值 | 第 19 章 |
| positional encoding | 位置编码 | 第 15 章 |
| positive / negative | 正例 / 负例 | 第 20 章 |
| Pre-Norm | 前置归一化 | 第 16 章 |
| pre-tokenization | 预切分 | 第 14 章 |
| pre-training | 预训练 | 第 16 章 |
| precision / recall | 精确率 / 召回率 | 第 21 章 |
| prediction | 预测 | 第 36 章 |
| preference accuracy | 偏好准确率 | 第 30 章 |
| preference data | 偏好数据 | 第 30 章 |
| prefill | 预填充 | 第 17 章 |
| prefix beam search | 前缀束搜索 | 第 13 章 |
| prefix masking | 前缀法 | 第 35 章 |
| prior | 先验 | 第 10 章 |
| probe KL | 探测 KL | 第 32 章 |
| processor | 处理器 | 第 24 章 |
| prompt loss weight | 提示权重 | 第 26 章 |
| protocol freeze | 方案冻结 | 第 36 章 |
| pseudo-label | 伪标签 | 第 33 章 |
| Q-Former | Q-Former | 第 22 章 |
| query / key / value | 查询 / 键 / 值 | 第 15 章 |
| query-conditioned mask decoder | 按查询条件化的掩码解码器 | 第 34 章 |
| random variable / distribution | 随机变量 / 分布 | 第 6 章 |
| rank (r) | 秩 | 第 28 章 |
| re-tokenization | 重新分词 | 第 27 章 |
| reading order | 阅读顺序 | 第 21 章 |
| recall / precision | 召回率 / 精确率 | 第 10 章 |
| Recall@k | 召回率@k | 第 20 章 |
| receptive field | 感受野 | 第 12 章 |
| rectified linear unit | ReLU | 第 5 章 |
| recurrent neural network, RNN | 循环神经网络 | 第 13 章 |
| recursive descent | 递归下降 | 第 3 章 |
| reference implementation | 参考实现 | 第 35 章 |
| reference model | 参考模型 | 第 29 章 |
| referring | 指代 | 第 23 章 |
| rejection sampling | 拒绝采样 | 第 3 章 |
| rejection sampling fine-tuning (RFT) | 拒绝采样微调 | 第 32 章 |
| reliability diagram | 可靠性图 | 第 27 章 |
| render verification | 渲染校验 | 第 33 章 |
| render-guided refine-and-judge | 渲染引导的修正与评判 | 第 34 章 |
| rendering | 渲染 | 第 2 章 |
| repetition penalty | 重复惩罚 | 第 17 章 |
| replay / rehearsal | 重放 | 第 32 章 |
| resampling | 重采样 | 第 10 章 |
| research question | 研究问题 | 第 36 章 |
| residual | 残差 | 第 7 章 |
| residual block | 残差块 | 第 12 章 |
| residual connection | 残差连接 | 第 11 章 |
| residual stream | 残差流 | 第 16 章 |
| resume training | 续训 | 第 18 章 |
| reward function | 奖励函数 | 第 35 章 |
| reward hacking | 奖励投机 | 第 29 章 |
| reward model | 奖励模型 | 第 30 章 |
| RMSNorm | 均方根归一化 | 第 16 章 |
| rollout / sampling | 采样 | 第 29 章 |
| rotary position embedding, RoPE | 旋转位置编码 | 第 16 章 |
| round-trip | 往返性质 | 第 3 章 |
| run manifest | 运行的身份证（manifest） | 第 36 章 |
| running average | 滑动平均 | 第 11 章 |
| sample-level averaging | 按样本平均 | 第 26 章 |
| scaled dot-product attention | 缩放点积注意力 | 第 15 章 |
| scaling law | 缩放定律 | 第 18 章 |
| scene text | 场景文字 | 第 23 章 |
| seed | 种子 | 第 1 章 |
| seed table | 种子表 | 第 36 章 |
| self-attention | 自注意力 | 第 15 章 |
| self-consistency | 自洽度 | 第 33 章 |
| self-consistency | 模型自洽度 | 第 34 章 |
| SFT（supervised fine-tuning） | 指令微调 | 第 0 章 |
| shape / dtype | 数组的形状 / 类型 | 第 1 章 |
| sigmoid / logistic | sigmoid | 第 5 章 |
| slicing / bucketing | 分桶 | 第 25 章 |
| smart_resize | 智能缩放 | 第 24 章 |
| smoothing | 平滑 | 第 6 章 |
| softmax | softmax | 第 6 章 |
| softmax / multinomial regression | softmax 回归 | 第 7 章 |
| space-to-depth / pixel shuffle | 空间到深度 | 第 22 章 |
| special token | 特殊 token | 第 14 章 |
| speculative decoding | 投机解码 | 第 17 章 |
| stability–plasticity | 稳定性—可塑性 | 第 32 章 |
| standardization | 标准化 | 第 2 章 |
| state_dict | 状态字典 | 第 9 章 |
| static memory | 静态内存 | 第 28 章 |
| steps to threshold | 达到阈值的步数 | 第 26 章 |
| stride / padding | 步长 / 填充 | 第 12 章 |
| structured output | 结构化输出 | 第 3 章 |
| subword | 子词 | 第 14 章 |
| supervised fine-tuning, SFT | 监督微调 | 第 26 章 |
| SwiGLU | 门控线性单元 | 第 16 章 |
| synchronize | 同步 | 第 9 章 |
| syntax tree | 语法树 | 第 3 章 |
| system prompt | 系统提示 | 第 23 章 |
| target modules | LoRA 目标 | 第 35 章 |
| targeted synthesis | 定向合成 | 第 33 章 |
| Taylor expansion | 泰勒展开 | 第 5 章 |
| TEDS | 树编辑距离相似度 | 第 25 章 |
| temperature | 温度 | 第 6 章 |
| temperature scaling | 温度缩放 | 第 27 章 |
| temporal patch | 时间 patch | 第 24 章 |
| tensor | 张量 | 第 9 章 |
| text anchoring | 文字锚定 | 第 34 章 |
| throughput | 吞吐 | 第 18 章 |
| token | 记号 | 第 3 章 |
| token-level averaging | 按 token 平均 | 第 26 章 |
| tokenization / tokenizer | 分词 / 分词器 | 第 14 章 |
| top-k sampling | top-k 采样 | 第 17 章 |
| topological sort | 拓扑排序 | 第 8 章 |
| train / validation / test set | 训练集 / 验证集 / 测试集 | 第 0 章 |
| training | 训练 | 第 0 章 |
| trajectory / rollout | 轨迹 | 第 29 章 |
| translation equivariance / invariance | 平移等变 / 不变 | 第 12 章 |
| transpose | 转置 | 第 4 章 |
| two-stage / pipeline parsing | 两阶段解析 | 第 33 章 |
| unit vector | 单位向量 | 第 4 章 |
| value network / critic | 价值网络 | 第 29 章 |
| vanishing / exploding gradient | 梯度消失 / 爆炸 | 第 11 章 |
| varlen attention | 变长注意力 | 第 26 章 |
| vector-Jacobian product | 向量-雅可比积 | 第 8 章 |
| vectorization | 向量化 | 第 1 章 |
| verifiable reward | 可验证奖励 | 第 29 章 |
| view / copy | 视图 / 副本 | 第 1 章 |
| vision encoder | 视觉编码器 | 第 0 章 |
| vision encoder / vision tower | 视觉编码器 / 视觉塔 | 第 19 章 |
| vision tower / vision encoder | 视觉塔 | 第 22 章 |
| visual token | 视觉 token | 第 22 章 |
| VLM | 视觉语言模型 | 第 0 章 |
| warmup | 预热 | 第 11 章 |
| weight / bias | 权重 / 偏置 | 第 4 章 |
| weight decay | 权重衰减 | 第 11 章 |
| weight sharing | 权重共享 | 第 12 章 |
| window attention | 窗口注意力 | 第 19 章 |
| word error rate, WER | 词错误率 | 第 25 章 |
| Xavier / Kaiming (He) init | Xavier / Kaiming 初始化 | 第 11 章 |
| XY-cut | XY-cut | 第 21 章 |
| Zero Redundancy Optimizer / Fully Sharded Data Parallel | ZeRO / FSDP | 第 28 章 |
| zero-shot | 零样本 | 第 20 章 |
