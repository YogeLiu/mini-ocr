"""只接线到自己的 mechanism；不得把实现写在本文件。整数接口无梯度要求。
所有非法输入按下面契约抛 ValueError；不要求实现类名或内部目录结构。
"""

def run_encode_sft(tokenizer, source, target, *, max_length=512):
    """返回有 input_ids、labels、prompt_length 属性的对象，序列为 tuple[int,...]。
    system 为 'Correct OCR errors in the following LaTeX formula.'，user 为 source。
    使用 apply_chat_template(tokenize=False, add_generation_prompt=True, enable_thinking=False)。
    encode(...,add_special_tokens=False) 联合编码 prompt+target，核对单独 prompt 是非空真前缀。
    追加 eos_token_id；labels 不 shift，prompt=-100，答案及 EOS 原 ID。
    超过 max_length 整条返回 None；空白答案、无 EOS、不稳定边界、max_length<2 抛 ValueError。
    """
    raise NotImplementedError


def run_collate(examples, pad_token_id, *, multiple=1):
    """返回 input_ids/labels/attention_mask 三个 CPU long (B,T) 张量。
    右补齐到最长长度的 multiple 倍；按位置填 pad/-100/0，真实位置 attention=1。
    pad==EOS 也不能遮掉真实 EOS。不修改输入；拒绝空批、multiple<1、pad=None、
    长度<2、ids/labels 长度不等、labels[1:] 全 -100。
    """
    raise NotImplementedError


def run_causal_loss(logits, labels, *, reduction="mean"):
    """(B,T,V) 浮点 logits 和 (B,T) long labels；只 shift 一次，忽略 -100。
    稳定交叉熵；支持 mean/sum，mean 除有效 token 总数；保留梯度。
    拒绝维度/形状不符、非法 reduction、全 mask、非 -100 的越界标签。
    """
    raise NotImplementedError


def get_lora(weight, bias=None, *, rank=4, alpha=8):
    """weight 浮点 (out,in)，bias None 或 (out,)；复制成冻结基座，不共享存储。
    返回 Module：base.weight/base.bias、可训练 A(rank,in)/B(out,rank)，B 初始零、A 非零随机。
    forward(x)=x@W.T+b+(alpha/rank)*(x@A.T)@B.T，支持任意前导维。
    merged() 返回独立冻结可调用层，有 weight/bias，保持 dtype/device/train 状态；可反复调用不累加。
    rank 正整数，alpha 正有限，非法尺寸/参数抛 ValueError；不限制 rank<=min(in,out)。
    """
    raise NotImplementedError
