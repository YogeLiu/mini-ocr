# 第 17 章实验手册：解剖 GPT-2

## 0. 准备

在 `course-v4/` 下运行：

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q book/part3_post_training/ch17_gpt2
CUDA_VISIBLE_DEVICES="" python -m book.part3_post_training.ch17_gpt2.tour
```

部分号范围是 **1–11**，可单独运行或按指定顺序组合；编号与下文、`tour.PARTS` 一致。位置参数保留顺序，重复编号会重复实验；各部分重置随机种子，故障演练结束后恢复模型。

```bash
CUDA_VISIBLE_DEVICES="" python -m book.part3_post_training.ch17_gpt2.tour 3 4
CUDA_VISIBLE_DEVICES="" python -m book.part3_post_training.ch17_gpt2.tour 8 9 10 --seed 1
CUDA_VISIBLE_DEVICES="" python -m book.part3_post_training.ch17_gpt2.tour 11
```

CLI 支持 `--device cpu|mps|cuda`，默认 `cpu`；正文与验收以 **CPU fp32** 为准。显式选择 `mps` 或 `cuda` 时会先检查设备可用性，不可用则清楚报错并退出；可用时打印“待验证：非 CPU 设备的误差量级未在本章标定”，然后照常运行，HF 模型与本章模型及输入放在同一设备，仍用 fp32。第 10 部分的独立激活函数网格扫描保持 CPU float64；非 CPU 对拍误差不能直接沿用本章验收阈值。`--seed` 默认 `0`。程序设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`OMP_NUM_THREADS=2`、`MKL_NUM_THREADS=2`、`OPENBLAS_NUM_THREADS=2`，并在 `main` 中调用 `torch.set_num_threads(2)`。参照模型使用 **HF eager 注意力、fp32、eval**；手写模型使用显式矩阵乘和因果 mask，省略 dropout。默认不使用 GPU，无语言模型训练、无 KV cache，只有第 6 部分训练一个小 BPE 分词器。完整 tour 的耗时目标是几分钟以内，实际总时间运行后填写；语料的标准划分需扫描本地语料，时间也计入。

输出使用 `key=value`，注明“数学结论”“本项目观察（设备、线程）”或“待验证”。参数公式是数学结论；误差、生成文本、分词结果和时间是本机本版本的观察。结构树逐行加 `HF结构=`，内容等同 `print(model)`。记录 `torch`、`transformers` 版本与确切权重路径，不把一次 top-1 一致推广到所有输入。

对照代码：[tour.py](tour.py)、[test_ch17_gpt2.py](test_ch17_gpt2.py)、[gpt2.py](../../../miniocr/models/gpt2.py)、[本书 GPT](../../../miniocr/models/gpt.py)、[第 6 章分词器](../../../miniocr/tokenizer.py)。练习标记在 `gpt2.py`：17.1 参数账、17.2 权重转换、17.3 注意力、17.4 块前向。先预测，再运行，最后填末尾空表。

### 本地权重与一次性准备

实验只从本地 HF 缓存快照读取，支持 `HF_HUB_CACHE` / `HF_HOME`；默认位置是 `~/.cache/huggingface/hub/`。多个快照按路径排序选取，输出会说明实际来源。GPT-2 需要 `config.json`、`model.safetensors` 和 tokenizer 文件。给定环境已备好，不需要重新获取。找不到权重的部分会打印：

```text
跳过：未找到本地 GPT-2 权重；本章不会自动下载
```

随机小模型测试不需要权重；官方权重测试在缺缓存时 skip。差异表读取本地 Qwen3-0.6B 生成模型配置；若有 safetensors，只读取键名，不加载 Qwen 张量。没有对应证据则标“待验证”，不要求下载 Qwen。

对尚未准备缓存的学习者，本章唯一需要另行获取的模型是 `openai-community/gpt2`。以下是**在允许联网的准备环境中手动执行一次**的示例，本次验收不执行，tour 也不会调用它；若外部环境已设置离线变量，准备时需自行取消：

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="openai-community/gpt2",
    revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
    allow_patterns=[
        "model.safetensors", "config.json", "generation_config.json",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "vocab.json", "merges.txt",
    ],
)
```

准备后所有模型和 tokenizer 加载都使用本地路径；HF 加载显式 `local_files_only=True`。代码本身直接用 safetensors 读 GPT-2 权重，不导入 transformers 建模实现。

## 1. 模型结构与参数账（17.1–17.2）

**做什么：** 打印 HF 模型树，找到 token/位置嵌入、Pre-LN 块、最终 LayerNorm 与共享输出头。逐项计算参数，并比较 `sum(p.numel() for p in model.parameters())` 和遍历 `state_dict()` 的总数。

**运行：** `python -m book.part3_post_training.ch17_gpt2.tour 1`（以下简写命令均使用准备节中的解释器及 CPU 环境）。

**看什么：** 共享参数只算一次时，数学公式为

`V·d + P·d + L·(12d² + 13d) + 2d`。

其中注意力权重/偏置为 `4d²+4d`，MLP 为 `8d²+5d`，块内两个 LayerNorm 为 `4d`，最终 LayerNorm 为 `2d`。GPT-2 small 的数学总数为 **124,439,808**。`state_dict` 同时有嵌入和输出头两个键，键数或逻辑元素总数不等于独立参数量；旧版本还可能保存 mask buffer。

**预测：** 若取消共享，参数量会增加哪一项？LayerNorm 的 β 和线性层偏置分别藏在哪一项？

**记录：** 公式逐项、本章统计、HF 去重统计、HF state_dict 未去重统计、差值与共享指针检查，实测运行后填写。

## 2. 配置逐字段解读（17.3）

**做什么：** 从本地 `config.json` 逐字段输出原始值与中文解释，不用模型名称猜数值。

**运行：** `python -m book.part3_post_training.ch17_gpt2.tour 2`。

**看什么：** `n_ctx` 是历史上下文字段，官方配置与 `n_positions` 相同；本章位置表以 `n_positions` 为准。`*_pdrop` 是各处 dropout，eval 时不生效。`summary_*` 属于序列分类/摘要头，普通 LM head 不使用。`task_specific_params` 是任务建议，实验显式选择贪心。官方 GPT-2 的 BOS/EOS 共用编号 **50256**。

**预测：** 修改 `task_specific_params` 是否改变已加载权重？只改 `n_ctx` 能否扩展学习式位置表？

**记录：** 将字段分为结构、训练行为、生成行为、其他头的元数据；未识别字段保留原文并标待验证。

## 3. 权重映射（17.6）

**做什么：** 列出嵌入、最终归一化、输出头及第一个块的 HF 键名/形状与手写模型键名/形状，逐行标记照搬、切分、转置、共享；其他块同理。

**运行：** `python -m book.part3_post_training.ch17_gpt2.tour 3`。

**看什么：** HF `Conv1D` 在这里实现仿射映射，不是沿序列滑动的卷积。权重布局为 `(in,out)`，`nn.Linear` 为 `(out,in)`。GPT-2 small 的 `c_attn.weight` 数学形状为 **(768, 2304)**，按输出维切成 q/k/v，然后各自转置。偏置仅切分，不转置。`c_fc` 和 `mlp.c_proj` 也需转置。

**看什么（原始文件账）：** 追加统计直接读取本地 `model.safetensors` 的 8 字节长度字段和 JSON 头，不加载张量。核对张量键总数（正文预期 160）、是否有 `lm_head` 键、是否带 `transformer.` 前缀、dtype 集合，以及 `h.{i}.attn.bias` 缓冲区的个数和各自形状；这些是本项目观察，不能用 HF `state_dict()` 的键替代。参数元素总数由头中的形状相乘求和，排除因果 mask 缓冲区；fp32 每元素 4 字节是数学结论。再看由数据偏移统计的参数字节数、缓冲区字节数、JSON 头长度和实际文件大小，核对 `文件字节 = 8 + 头长度 + 参数字节 + 缓冲区字节` 是否成立。具体数值以本地输出为准，并记录文件路径。

**预测：** 为什么不能把 qkv 的第一个维度当作拼接维度？为什么 `(768,768)` 的矩阵特别容易漏检？

**记录：** 手画 `x @ W + b` 与 `F.linear(x, W, b)` 的形状链，核对每项转换。转换器接受有/无 `transformer.` 前缀，忽略旧因果 mask buffer，检查已有 LM head 与嵌入是否一致。

## 4. logits 与残差流对拍（17.7）

**做什么：** 对固定英文和含 LaTeX 的英文各做一次 CPU fp32 前向，与 HF eager 比较全部 logits 及每层 hidden states。

**运行：** `python -m book.part3_post_training.ch17_gpt2.tour 4`。

**看什么：** 最大绝对误差、HF 最大 `|logit|`、两者之比（相对 logits 量级），以及各边界误差。该比值不是每个 logit 的逐元素相对误差。hidden 边界 `0` 是嵌入相加；`1..L-1` 是前一块输出；边界 `L` 是最后一块之后**再经过 ln_f** 的结果，共 `L+1` 项，严格对应 HF 的口径。它没有单独列出最后一块未经 ln_f 的残差。

**预测：** 正确转换为何仍可能有浮点误差？若边界 0 一致而边界 1 首次超差，应从哪些模块开始检查？

**记录：** 两段输入各自的误差、量级、逐层误差和首个超出绝对容差的边界；所有数值运行后填写。tiny 测试使用约 `1e-5` 容差，官方模型测试另用明确的累计误差容差；不要把它们当成跨设备保证。

## 5. 下一 token 与贪心续写（17.7）

**做什么：** 对同一英文提示打印两边下一 token 的 top-5 编号、文本和概率，然后固定续写 20 个 token，逐 token 比较。

**运行：** `python -m book.part3_post_training.ch17_gpt2.tour 5`。

**看什么：** top-5 编号是否一致、概率最大误差、续写 token 是否完全一致和两边续写文本。HF 显式 `do_sample=False`，双方都不用 KV cache，并禁用 EOS 提前停止以保持固定步数。手写 `generate_greedy` 要求输入加续写总长不超过 `n_positions`；超过时报告错误，不滑窗重置位置编号。

**预测：** 很小的 logits 差是否必然保持 argmax？仅最终文本一样能否说明 logits 转换正确？

**记录：** top-5、续写与一致性，运行后填写；仅对这些输入成立。

## 6. 字节级 BPE 与 LaTeX（17.9）

**做什么：** GPT-2 tokenizer 与第 6 章 `train_bpe`/`BPETokenizer` 对比中文、单个汉字、空格和典型公式，包括 `\frac{a+b}{\sqrt{x^2+1}}`、`\alpha+\beta=\gamma`、`\text{面积}=ab`。

**运行：** `python -m book.part3_post_training.ch17_gpt2.tour 6`。

**看什么：** 第 6 章 BPE 使用本地 `standard_split` 训练集的至多 256 条前缀，最多 256 次合并；缺语料时使用 `SAMPLE_FORMULAS` 并缩小留出集，输出实际规模。标准协议先完整划分再截训练前缀，不先截原始语料。每个示例打印切分、token 数、各 token 原始字节数；GPT-2 另打印 hex 并验证拼回原文 UTF-8。

`Ġ` 是空格字节的词表显示映射，不是原文字母。中文字符的多字节 UTF-8 编码可能被切到多个 token；独立解码这些片段可能出现替换字符，不能用替换字符的字节数当原始 token 长度。小 BPE 可能见过演示公式，不能将这些例子称为留出集性能评测。

**预测：** 为什么更少的 token 不一定表示更好的语言模型？小 BPE 的词表更小时能否在个别公式上更紧凑？

**记录：** 语料来源、实际训练规模、合并次数、每种文本的两组切分与字节数，运行后填写。

## 7. 三种骨架的差异（17.10）

**做什么：** 比较 GPT-2、本书默认 GPT、Qwen3 骨架，逐格标明来源。

**运行：** `python -m book.part3_post_training.ch17_gpt2.tour 7`。

**看什么：** 位置编码、归一化、激活/MLP、q/kv 头数、Q/K norm、偏置、权重共享、词表、上下文、层数、宽度、head_dim、norm eps 与 rope_theta。本书值来自 `GPTConfig` 默认值及实现，GPT-2 数值来自本地 JSON，结构来自手写实现并由前向对拍验证。Qwen3 **直接读取本地 Qwen3-0.6B 生成模型**；配置未声明 Q/K norm 时，只从 safetensors 键名判断是否有 q_norm/k_norm，不读取张量。缺证据明确标待验证。

**预测：** Qwen 的 `head_dim` 能否总用 `hidden_size/n_head` 推导？配置有 `silu` 就足以证明门控 MLP 的全部结构吗？

**记录：** 表格、每格来源与尚待验证项；配置与权重键能确认的项目不再标待验证；键名不能证明前向运算顺序或 mask 实现。

## 8. 故障 A：漏转置（17.8.1）

**做什么：** 先把非方阵原始权重交给 Linear，捕获真实 shape mismatch；再运行 `bug="no_transpose"` 的可静默加载版本。

**运行：** `python -m book.part3_post_training.ch17_gpt2.tour 8`。

**看什么：** 先保持 c_attn 的合并形式，用一个 `Linear(d,3d)` 展示原始 `(d,3d)` 权重漏转置时的加载失败；c_fc/mlp.c_proj 同样展示直接漏转置，都打印原始/目标形状及错误摘要。这一步只隔离矩阵方向，不引入错误的切分维度。静默版本**只漏掉原始完整正方阵 attn.c_proj 的转置**，其余仍正确切分/转置；不是所有矩阵都原样加载。打印 logits 误差、下一位置及全部位置 top-1 一致性、首个超差 hidden 边界及 `(batch, position, channel)`（均从零计）。

**预测：** `strict=True` 为什么不能保证矩阵方向正确？首个超差位置是否必然与最大误差位置相同？

**记录：** shape mismatch 摘要、静默矩阵名称、误差和首个超差边界，运行后填写。

## 9. 故障 B：qkv 顺序（17.8.2）

**做什么：** 把按 q/k/v 拼接的权重和偏置误认作 k/q/v，保持所有形状不变。

**运行：** `python -m book.part3_post_training.ch17_gpt2.tour 9`。

**看什么：** strict 加载成功后的 logits 与逐层误差，以及最早偏离的边界和位置。单位置只有一个可见 key，不一定能揭示 q/k 互换，因此测试和实验使用多 token 提示。

**预测：** q/k 交换为什么不是无害的名字互换？因果 mask 与按行 softmax 对对称性有什么影响？

**记录：** 与漏转置故障比较：误差规模、top-1、首个超差位置，运行后填写。

## 10. 故障 C：GELU 版本（17.8.3）

**做什么：** 在 `[-5,5]` 的固定 float64 网格上比较精确式和 tanh 近似，再把模型激活改为精确式，与官方输出和固定步数续写对拍。

**运行：** `python -m book.part3_post_training.ch17_gpt2.tour 10`。

**看什么：** 精确式 `x/2*(1+erf(x/sqrt(2)))` 和 GPT-2 近似式 `x/2*(1+tanh(sqrt(2/pi)*(x+0.044715*x³)))`；网格最大差和对应 x 是数值观察，不是连续实数域最大值的证明。然后观察逐层累计、logits、top-1 和 20 token 贪心续写。

**预测：** 激活函数差很小是否保证最后输出相同？这段输入续写一致是否允许忽略转换中的激活版本？

**记录：** 网格误差、模型误差、首个超差边界、top-1 与续写是否改变，全部运行后填写，不预设必须改变。

## 11. 故障 D：位置越界（17.8.4）

**做什么：** 对官方配置，手写模型在长度 1024 做一次完整前向；长度 1025 分别调用手写模型与 HF，捕获并打印行为。

**运行：** `python -m book.part3_post_training.ch17_gpt2.tour 11`。

**看什么：** 正常长度的 logits 形状及是否全部有限；手写模型越界时在嵌入索引之前给出中文 `ValueError`。HF 若抛错则打印异常类型与摘要；若未抛错则如实记录。不重复跑长序列，不以某个 HF 异常类型作为跨版本断言。

**预测：** 将位置编号取模能消除索引错误，但能等价扩展预训练上下文吗？学习式位置表和 RoPE 的边界限制是否完全相同？

**记录：** 两边长度、正常输出、异常类型及摘要、该部分耗时，运行后填写。

## 自己填写的实测记录

环境：设备 ______；系统 ______；torch ______；transformers ______；dtype ______；线程 ______；HF 注意力 ______；seed ______；权重快照 ______；完整 tour 用时 ______。

| 部分 | 观察项 | 预测（先填） | 实测（运行后填写） |
|---|---|---|---|
| 1 | 参数公式 / 代码 / state_dict 差值 | | |
| 2 | n_ctx、n_positions、summary_* 的作用 | | |
| 3 | c_attn / c_proj 转换形状、原始文件键 / dtype / 缓冲区 / 字节账 | | |
| 4 | 英文 logits 误差 / 相对量级 / hidden 最大误差 | | |
| 4 | LaTeX logits 误差 / 相对量级 / hidden 最大误差 | | |
| 5 | top-5 / 概率误差 / 续写一致性 | | |
| 6 | 语料来源 / 规模 / 中文与公式 token 数 | | |
| 7 | 配置证据 / 权重键证据 / 待验证项 | | |
| 8 | 静默漏转置误差 / top-1 / 首个超差边界位置 | | |
| 9 | qkv 顺序误差 / top-1 / 首个超差边界位置 | | |
| 10 | GELU 网格误差 / logits 误差 / 续写变化 | | |
| 11 | 正常边界 / 越界错误 / HF 实际行为 | | |
| 全部 | 总用时 / 测试通过与跳过数量 | | |
