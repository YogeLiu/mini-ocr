"""第 28 章：训练时的内存从哪里来。先估算，再测量。

    BYTES               每种数据类型每个数占几个字节
    static_memory       参数、梯度、优化器状态各占多少字节（与批大小、序列长度无关）
    SavedTensors        上下文管理器：记录反向传播要用的、前向时保存下来的张量（激活值）一共多少字节
    CheckpointBlock     梯度检查点：前向时不保存块内部的激活值，反向时重新算一遍
    checkpoint_blocks   把一个 ModuleList 里的每个块都换成 CheckpointBlock

激活值是"前向时算出来、反向时还要用"的中间结果。它们和参数不同：参数量固定，激活值随批大小 × 序列长度增长。
PyTorch 在前向时把这些张量交给 autograd 保存；saved_tensors_hooks 可以截获每一个被保存的张量，于是在 CPU 上也能
精确地数出它们的大小，不需要 GPU 的显存统计。
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

BYTES = {"fp32": 4, "bf16": 2, "fp16": 2, "int8": 1}


# >>> 练习 28.3：参数、梯度、优化器状态
def static_memory(n_params, n_trainable, param_dtype="fp32", optimizer="adamw", master_fp32=False):
    """返回 dict（单位：字节）：
        params：所有参数，n_params × param_dtype 的字节数；
        grads：只有可训练参数有梯度，n_trainable × param_dtype 的字节数；
        optim：AdamW 每个可训练参数存两个 fp32 的数（一阶矩 m、二阶矩 v）= 8 字节；SGD（无动量）为 0；
        master：混合精度训练时（param_dtype 为 bf16/fp16 且 master_fp32=True），可训练参数再存一份 fp32 的"主权重"，4 字节；
        total：以上之和。"""
    b = BYTES[param_dtype]
    out = dict(params=n_params * b, grads=n_trainable * b,
               optim={"adamw": 8, "sgd": 0}[optimizer] * n_trainable,
               master=4 * n_trainable if master_fp32 and param_dtype != "fp32" else 0)
    out["total"] = sum(out.values())
    return out
# <<<


# >>> 练习 28.4：数出保存下来的激活值
class SavedTensors:
    """with SavedTensors(model) as s:  loss = model(...)   →  s.bytes 是这次前向保存下来、留给反向用的张量的总字节数。
    规则：同一块存储（storage）只算一次（很多张量是同一块内存的不同视图）；模型的参数不算（它们本来就在内存里）。
    实现：torch.autograd.graph.saved_tensors_hooks(pack, unpack)。pack(t) 在每个张量被保存时调用，
    用 t.untyped_storage().data_ptr() 识别存储、t.untyped_storage().nbytes() 得到大小；返回 t 本身；unpack 原样返回。"""

    def __init__(self, model=None):
        self.skip = set()
        if model is not None:
            self.skip = {p.untyped_storage().data_ptr() for p in model.parameters()}
            self.skip |= {b.untyped_storage().data_ptr() for b in model.buffers()}
        self.seen, self.bytes, self.count = set(), 0, 0

    def _pack(self, t):
        st = t.untyped_storage()
        ptr = st.data_ptr()
        if ptr not in self.skip and ptr not in self.seen:
            self.seen.add(ptr)
            self.bytes += st.nbytes()
            self.count += 1
        return t

    def __enter__(self):
        self._ctx = torch.autograd.graph.saved_tensors_hooks(self._pack, lambda t: t)
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        self._ctx.__exit__(*exc)
        return False
# <<<


# >>> 练习 28.5：梯度检查点
class CheckpointBlock(nn.Module):
    """包住一个块。训练时用 torch.utils.checkpoint.checkpoint(self.block, *args, use_reentrant=False)：
    前向只保存块的输入，不保存块内部的激活值；反向传播到这个块时，用保存的输入把块的前向重新算一遍，再求梯度。
    不训练（torch.is_grad_enabled() 为 False）或者没有任何输入需要梯度时，直接调用 self.block。"""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, *args, **kw):
        if torch.is_grad_enabled() and self.training:
            return checkpoint(self.block, *args, use_reentrant=False, **kw)
        return self.block(*args, **kw)
# <<<


def checkpoint_blocks(blocks: nn.ModuleList):
    """原地把 ModuleList 里的每个块换成 CheckpointBlock（已经换过的不再换）。返回换了几个。"""
    n = 0
    for i, b in enumerate(blocks):
        if not isinstance(b, CheckpointBlock):
            blocks[i] = CheckpointBlock(b)
            n += 1
    return n


def fmt_bytes(n):
    for unit, k in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= k:
            return f"{n / k:.2f} {unit}"
    return f"{n} B"
