"""A4 参考实现：训练内存。静态部分（参数、梯度、优化器、主权重）的估算，保存下来的激活值的测量，梯度检查点。"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

BYTES = {"fp32": 4, "bf16": 2, "fp16": 2}


def static_memory(n_params, n_trainable, param_dtype="fp32", optimizer="adamw", master_fp32=False):
    """字节数：params = 全部参数 × 类型字节；grads = 可训练 × 类型字节；optim：AdamW 每个可训练参数 8 字节（fp32 的 m、v），
    SGD 0；master：混合精度（bf16/fp16 且 master_fp32）时可训练参数再存 4 字节；total。"""
    b = BYTES[param_dtype]
    out = dict(params=n_params * b, grads=n_trainable * b, optim={"adamw": 8, "sgd": 0}[optimizer] * n_trainable,
               master=4 * n_trainable if master_fp32 and param_dtype != "fp32" else 0)
    out["total"] = sum(out.values())
    return out


class SavedTensors:
    """with SavedTensors(model) as s: 前向 → s.bytes：为反向保存下来的张量的总字节数。同一块存储只算一次，
    模型的参数和缓冲区不算。"""

    def __init__(self, model=None):
        self.skip = set()
        if model is not None:
            self.skip = {t.untyped_storage().data_ptr() for t in list(model.parameters()) + list(model.buffers())}
        self.seen, self.bytes = set(), 0

    def _pack(self, t):
        st = t.untyped_storage()
        ptr = st.data_ptr()
        if ptr not in self.skip and ptr not in self.seen:
            self.seen.add(ptr)
            self.bytes += st.nbytes()
        return t

    def __enter__(self):
        self._ctx = torch.autograd.graph.saved_tensors_hooks(self._pack, lambda t: t)
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        self._ctx.__exit__(*exc)
        return False


class CheckpointBlock(nn.Module):
    """训练且需要梯度时用 torch.utils.checkpoint：前向只存输入，反向时重算块内部。"""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, *args, **kw):
        if torch.is_grad_enabled() and self.training:
            return checkpoint(self.block, *args, use_reentrant=False, **kw)
        return self.block(*args, **kw)


def checkpoint_blocks(blocks):
    n = 0
    for i, b in enumerate(blocks):
        if not isinstance(b, CheckpointBlock):
            blocks[i] = CheckpointBlock(b)
            n += 1
    return n


def uncheckpoint_blocks(blocks):
    """把 CheckpointBlock 换回原来的块（存盘前用，state_dict 的键才和原模型一致）。"""
    for i, b in enumerate(blocks):
        if isinstance(b, CheckpointBlock):
            blocks[i] = b.block
