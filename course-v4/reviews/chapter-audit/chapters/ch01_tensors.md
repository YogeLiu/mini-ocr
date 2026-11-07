# 第 1 章审核

## 阅读覆盖与基准

- 源文件：`course-v4/book/part1_foundations/ch01_tensors/chapter.md`，846 行。
- SHA256：`c167c91515825282467ba5d0c2794bb21d52c7322e7eb33b52541cd9552186ef`。
- 已完整读取并通读 1–220、221–440、441–650、651–846 行。核查广播推导、距离例子、stride 表、多头拆分、einops 对照、one-hot 和图片管线，没有用关键词扫描代替通读。
- 本地 CPU/meta 小例子核查 view 和 expand 行为；没有使用真实 GPU，也没有重新测历史计时。

## 章节主线与关键约定

本章建立 shape、dtype、device 的基础词汇。广播按右对齐、左补 1、逐维相等或含 1 判定，强调长度匹配不保证语义维度正确。两两平方距离的直接广播需要 NMD 中间张量，矩阵展开减少中间量但有消减误差，clamp 只能防负数，不能恢复精度。

矩阵乘只在矩阵维上收缩，批量维广播。以 storage_offset 加下标乘 stride 描述寻址；view 需要目标形状与 stride 兼容，非连续本身并不禁止 view；reshape 必要时复制。多头先拆特征维再换轴，不能用形状相同作为内容正确的证据。einops 用显式维名表达同样的操作。

后续全书的张量词汇为 B 批大小、T 序列长、V 词表大小、D 隐藏维、C 通道、H/W 像素高宽。one-hot 乘权重等于索引查表；语言模型 logits 为 BTV。图片常见 HWC uint8，卷积批量采用 BCHW float；逻辑轴顺序与实际内存布局必须区分。

## 已确认问题

本章未确认影响核心推导的 P1/P2 错误，以下是两处过度绝对化表述。

### CH01-01 P3 同设备要求漏掉 CPU 零维标量例外

- 位置：77、129、820 行，“参与同一次运算的张量必须在同一个设备上”。
- 证据：[PyTorch 2.14 Tensor Attributes](https://docs.pytorch.org/docs/2.14/tensor_attributes.html) 的 device 注释明确允许需要时把 CPU 零维标量自动用于 GPU 运算，而非标量不自动搬运。本地 `torch.tensor(1.) + torch.ones(2, device='meta')` 也可执行，结果在 meta；这只是辅助 CPU/meta 复现，不冒充 GPU 测量。
- 修改建议：“通常应放在同一设备；CPU 的零维标量在部分运算中是例外。”保留模型和非标量输入必须显式对齐的实践建议。

### CH01-02 P3 expand 不是所有原地写入都被禁止

- 位置：552 行，“expand 出来的张量不能原地修改”。
- 问题：该建议的方向正确，但若读成运行时保证就不准确。有些逐元素写入会成功，并同时改变所有别名位置，而非必然报错。
- 本地复现：`x=torch.tensor([[1.],[2.]]).expand(2,3); x[0,0]=9`，得到 `[[9,9,9],[2,2,2]]`。官方 [expand 文档](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.expand.html) 的警告是写入可能造成错误行为，需要先 clone，并非所有写入都会拒绝。
- 修改建议：“不要直接写入展开视图；多个位置可能共享同一地址，批量原地运算可能报错，逐元素赋值也可能连带修改别处。需要写入时先 clone。”

## 已声明局限与待验证事项

- 非连续张量也能 view、先 transpose 后正确拆维同样成立、clamp 不消除距离舍入误差、channels_last 与 BCHW 形状不同层次，章内均已有准确说明，不再标为错误。
- 355 行的 80 MiB 是 ND+MD+NM 量级估算，不等于实际 eager 表达式的峰值；此处明确说“量级”，尚不作为确定内存数值错误。若后章用于资源规划，应区分张量量级与实测峰值。
- NumPy 互转共享内存的概述隐含 CPU、可转换 dtype、不需要梯度等条件；本章以普通 CPU 张量为语境。后章若将该句直接套到 GPU 或 requires_grad 张量，需补条件。
- 没有复现 matplotlib 字体相关图片尺寸、随机误差条数和耗时，不据此声称所有历史测量数字已复核。

## 跨章节关联

| 目标章节 | 本章交付 | 待跨章核查 |
|---|---|---|
| 2、3、4 | shape 检查、BTV、类别轴与设备规则 | loss 分母、广播静默错误、参数更新是否遵循维度语义 |
| 8、11、15 | 拆头顺序、批量矩阵乘、mask 广播 | B/h/T 恰好相等时测试能否抓到轴错；KV cache 的维度与布局 |
| 16、20 | dtype 字节数、中间张量量级 | 资源估算是否把量级、存储分配和峰值混用 |
| 27、28、29、31、32 | HWC 到 BCHW、stride、patch 拆分 | 图像颜色约定、归一化与底层 memory_format 的含义是否一致 |
| 6、7 | token 编号与 one-hot 查表 | 不把手工 LaTeX token、BPE token 和隐藏向量混为一谈 |

## 核查来源与方式

已打开并读取相关正文：[PyTorch 2.14 Tensor Attributes](https://docs.pytorch.org/docs/2.14/tensor_attributes.html)、[Tensor.view](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.view.html)、[Tensor.expand](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.expand.html)。view 的 stride 兼容条件与本章一致。本地还核查 `arange(24).view(2,3,4)[:,:,0].view(-1)`，得到 stride `(12,4)` 与 `[0,4,8,12,16,20]`。
