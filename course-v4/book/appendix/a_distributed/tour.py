"""附录 A：分布式训练。不开进程组：显存账是纯计算，梯度等价用单进程模拟。

python -m book.appendix.a_distributed.tour [部分号 1–7 ...]
"""
import argparse
from dataclasses import replace
import math
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "2"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DistributedSampler

from book.appendix import emit
from miniocr.kvcache import local_model_configs
from miniocr.models.gpt import GPT, RMSNorm
from miniocr.train.resources import activation_bytes_estimate, qwen_like_params
from miniocr.train.scaling import ladder_config

# 第 16 章 16.6 的每参数字节拆法。
ITEMS = {"fp32": dict(params=4, grads=4, optim=8),
         "mixed": dict(params=2, grads=2, optim=12)}  # mixed 的 optim = 主权重 4 + m、v 共 8
SHARDED = {"ddp": (), "zero1": ("optim",), "zero2": ("optim", "grads"),
           "zero3": ("optim", "grads", "params"), "fsdp": ("optim", "grads", "params")}


def static_per_card(N, K, method, precision="fp32"):
    """每张卡的静态字节。被切分的项除以 K，其余每卡一份。

    假设：参数能均匀切分；不含通信缓冲、临时聚合（见 peak_gather）、分配器开销。
    """
    if N < 0 or K < 1 or method not in SHARDED or precision not in ITEMS:
        raise ValueError("N、K、method 或 precision 非法")
    out = {item: bytes_per*N/K if item in SHARDED[method] else float(bytes_per*N)
           for item, bytes_per in ITEMS[precision].items()}
    out["total"] = sum(out.values())
    return out


def peak_gather(unit_params, precision="fp32"):
    """ZeRO-3 / FSDP 的临时开销：算到一个切分单元时，把它的完整参数和梯度聚到本卡。"""
    item = ITEMS[precision]
    return unit_params*(item["params"]+item["grads"])


def tensor_parallel_params(model, K):
    """张量并行：矩阵（含嵌入）按一个维度切成 K 份，一维的 norm 每卡一份。返回每卡参数量。"""
    return sum(math.ceil(p.numel()/K) if p.ndim >= 2 else p.numel() for p in model.parameters())


def pipeline_parallel_params(model, K):
    """流水线并行：块按顺序连续分到 K 个阶段；嵌入在首段，末 norm 在末段。

    权重共享时输出层也要用嵌入，所以末段再放一份。K 须能整除层数。返回每阶段参数量。
    """
    L = len(model.blocks)
    if K < 1 or L % K:
        raise ValueError("K 须整除层数")
    per = [sum(sum(p.numel() for p in model.blocks[i].parameters()) for i in range(s*L//K, (s+1)*L//K))
           for s in range(K)]
    per[0] += model.tok.weight.numel()
    per[-1] += model.norm.weight.numel()
    if K > 1 and model.cfg.tie_weights:
        per[-1] += model.tok.weight.numel()
    return per


def linear_example(xs=(1., 2., 3., 4.), w=1.):
    """一个参数的回归 loss=(wx)²：单样本梯度 2wx²，可手算。"""
    return [2*w*x*x for x in xs]


def averaged(grads, sizes=None):
    """把各卡梯度合成一份。sizes=None 是 DDP 的简单平均；给出时按份额加权。"""
    if sizes is None:
        sizes = [1]*len(grads)
    total = sum(sizes)
    return sum(g*(n/total) for g, n in zip(grads, sizes))


def _grad(model, x, y):
    # GPT.forward 的交叉熵在 fp32 里算；对拍要全程 float64，所以这里自己算 loss。
    model.zero_grad(set_to_none=True)
    logits = model(x)[0]
    F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100).backward()
    return torch.cat([p.grad.flatten() for p in model.parameters()]).clone()


class DoubleRMSNorm(nn.Module):
    """项目的 RMSNorm 内部转 fp32。对拍要全程 float64，否则舍入差停在 1e-8 附近。"""

    def __init__(self, norm):
        super().__init__()
        self.weight, self.eps = norm.weight, norm.eps

    def forward(self, x):
        return x*torch.rsqrt(x.square().mean(-1, keepdim=True)+self.eps)*self.weight


def tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = replace(ladder_config("tiny", 64, 16), attn_impl="manual")
    model = GPT(cfg).double()
    for name, module in list(model.named_modules()):
        if isinstance(module, RMSNorm):
            parent, _, leaf = name.rpartition(".")
            setattr(model.get_submodule(parent), leaf, DoubleRMSNorm(module))
    return model


def batch(seed=0, B=8, T=16, V=64):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(V, (B, T), generator=g), torch.randint(V, (B, T), generator=g)


def shard_gradients(model, x, y, sizes):
    grads, start = [], 0
    for n in sizes:
        grads.append(_grad(model, x[start:start+n], y[start:start+n]))
        start += n
    return grads


def relative(a, b):
    return ((a-b).norm()/b.norm()).item()


def sampler_indices(n, K, drop_last=False):
    """真实 DistributedSampler 的切分。显式给 num_replicas、rank，不需要进程组。"""
    return [list(DistributedSampler(range(n), num_replicas=K, rank=r, shuffle=False, drop_last=drop_last))
            for r in range(K)]


def bn_gradients(seed=0, B=8, K=4):
    """BatchNorm 用本卡的 batch 算统计量。返回 (整批梯度, K 份平均梯度)。"""
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Tanh(), nn.Linear(8, 1)).double()
    x, y = torch.randn(B, 4).double(), torch.randn(B, 1).double()

    def grad(xx, yy):
        net.zero_grad(set_to_none=True)
        (net(xx)-yy).square().mean().backward()
        return torch.cat([p.grad.flatten() for p in net.parameters()]).clone()
    full = grad(x, y)
    return full, averaged([grad(a, b) for a, b in zip(x.chunk(K), y.chunk(K))])


def trajectory(seed=0, K=4, steps=5, lr=.1, broken=False):
    """K 个副本同初值、用同一平均梯度走 SGD。broken=True 时每个副本只用本卡梯度。

    返回 (副本间最大参数差, 副本 0 与整批单卡的最大参数差)。
    """
    x, y = batch(seed)
    replicas = [tiny_model(seed) for _ in range(K)]
    single = tiny_model(seed)
    size = len(x)//K
    for _ in range(steps):
        grads = [_grad(m, x[i*size:(i+1)*size], y[i*size:(i+1)*size]) for i, m in enumerate(replicas)]
        mean = averaged(grads)
        for m, own in zip(replicas, grads):
            _apply(m, own if broken else mean, lr)
        _apply(single, _grad(single, x, y), lr)
    flat = [_flat(m) for m in replicas]
    return max((f-flat[0]).abs().max().item() for f in flat), (flat[0]-_flat(single)).abs().max().item()


def _flat(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def _apply(model, grad, lr):
    start = 0
    with torch.no_grad():
        for p in model.parameters():
            p -= lr*grad[start:start+p.numel()].view_as(p)
            start += p.numel()


def part1(_):
    N = qwen_like_params(local_model_configs()[0])["total"]
    for precision in ITEMS:
        emit(来源="数学结论", 口径=precision, 每参数字节=ITEMS[precision], N=N, K=4)
        for method in SHARDED:
            row = static_per_card(N, 4, method, precision)
            emit(来源="数学结论", 模型="Qwen3-0.6B骨架", 口径=precision, 方法=method, K=4,
                 每参数字节=row["total"]/N, 每卡GiB=row["total"]/2**30,
                 相对DDP=row["total"]/static_per_card(N, 4, "ddp", precision)["total"])
    for K in (2, 4, 8, 64):
        emit(来源="数学结论", 口径="fp32", K=K, **{m: static_per_card(1, K, m)["total"] for m in SHARDED})
    emit(来源="数学结论", 说明="K趋于无穷时每参数下限", fp32=dict(zero1=8, zero2=4, zero3=0),
         mixed=dict(zero1=4, zero2=2, zero3=0))


def part2(_):
    model = GPT(replace(ladder_config("medium", 512, 64), attn_impl="manual"))
    N, K, B, T = model.n_params(), 4, 16, 64
    block = sum(p.numel() for p in model.blocks[0].parameters())
    emit(来源="数学结论", 模型="medium", N=N, K=K, 单块参数=block, 嵌入参数=model.tok.weight.numel())
    tp = tensor_parallel_params(model, K)
    pp = pipeline_parallel_params(model, K)
    for name, row in (("DDP", static_per_card(N, K, "ddp")["total"]),
                      ("ZeRO-3/FSDP", static_per_card(N, K, "zero3")["total"]),
                      ("张量并行", tp*16),
                      ("流水线最大阶段", max(pp)*16)):
        emit(来源="数学结论", 模型="medium", 方法=name, K=K, 每卡静态字节=row, 占单卡比例=row/(16*N))
    emit(来源="数学结论", 张量并行每卡参数=tp, 流水线各阶段参数=pp,
         说明="张量并行的norm不切；流水线的末段多一份共享嵌入")
    emit(来源="数学结论", 临时聚合字节=peak_gather(block), 假设="每块一个切分单元；fp32；参数+梯度各聚一块",
         ZeRO3峰值字节=static_per_card(N, K, "zero3")["total"]+peak_gather(block))
    whole = activation_bytes_estimate(model.cfg, B, T)["total"]
    share = activation_bytes_estimate(model.cfg, B//K, T)["total"]
    emit(来源="数学结论", 全局批=B, T=T, 单卡整批激活字节=whole, 切到K卡后每卡激活字节=share,
         比例=share/whole, 适用="DDP、ZeRO、FSDP：同样的全局批切到K卡")


def part3(_):
    per = linear_example()
    emit(来源="数学结论", x=[1, 2, 3, 4], w=1, 单样本梯度=per, 整批=sum(per)/4)
    a, b = sum(per[:2])/2, sum(per[2:])/2
    emit(来源="数学结论", 切分="[1,2]|[3,4]", 各卡=[a, b], 简单平均=(a+b)/2)
    c, d = per[0], sum(per[1:])/3
    emit(来源="数学结论", 切分="[1]|[2,3,4]", 各卡=[c, d], 简单平均=(c+d)/2, 按样本数加权=(c+3*d)/4)


def part4(args):
    model = tiny_model(args.seed)
    x, y = batch(args.seed)
    full = _grad(model, x, y)
    emit(模型="tiny", dtype="float64", B=8, T=16, 参数量=model.n_params(), 设备="cpu")
    for sizes in ((8,), (4, 4), (2, 2, 2, 2), (1,)*8):
        emit(切分=list(sizes), 与整批相对差=relative(averaged(shard_gradients(model, x, y, sizes)), full))
    sizes = (5, 1, 1, 1)
    grads = shard_gradients(model, x, y, sizes)
    emit(切分=list(sizes), 简单平均相对差=relative(averaged(grads), full),
         按样本数加权相对差=relative(averaged(grads, sizes), full))
    # 样本数相同，有效 token 数不同。
    yy = y.clone()
    lengths = (2, 16, 16, 10, 3, 16, 8, 16)
    for i, n in enumerate(lengths):
        yy[i, n:] = -100
    masked = _grad(model, x, yy)
    grads = shard_gradients(model, x, yy, (2, 2, 2, 2))
    tokens = [sum(lengths[i:i+2]) for i in range(0, 8, 2)]
    emit(场景="等样本数、不等token数", 有效长度=list(lengths), 各卡token=tokens,
         简单平均相对差=relative(averaged(grads), masked), 按token加权相对差=relative(averaged(grads, tokens), masked))


def part5(args):
    n, K = 10, 4
    for drop in (False, True):
        emit(来源="本项目观察（真实DistributedSampler）", 设备="cpu", 样本数=n, K=K, drop_last=drop,
             各卡下标=sampler_indices(n, K, drop))
    model = tiny_model(args.seed)
    x, y = batch(args.seed, B=n)
    full = _grad(model, x, y)
    for drop in (False, True):
        grads = [_grad(model, x[i], y[i]) for i in sampler_indices(n, K, drop)]
        emit(drop_last=drop, 与全部10条整批相对差=relative(averaged(grads), full),
             说明="补齐时有样本算两次；丢弃时有样本没算")


def part6(args):
    full, mean = bn_gradients(args.seed)
    emit(场景="BatchNorm，等分4卡", 与整批相对差=relative(mean, full),
         说明="统计量按本卡batch算；真实框架要用SyncBatchNorm，本书模型用RMSNorm没有这个问题")


def part7(args):
    same = trajectory(args.seed)
    broken = trajectory(args.seed, broken=True)
    emit(场景="同一平均梯度，SGD 5步，K=4", 副本间最大差=same[0], 与整批单卡最大差=same[1])
    emit(场景="错误：各用本卡梯度，SGD 5步，K=4", 副本间最大差=broken[0], 与整批单卡最大差=broken[1])


PARTS = {1: (part1, "DDP、ZeRO、FSDP的静态显存（A.4.1）"),
         2: (part2, "medium上五种方法每卡切分（A.4.2）"),
         3: (part3, "手算梯度等价（A.4.3）"),
         4: (part4, "tiny GPT梯度对拍（A.4.3）"),
         5: (part5, "DistributedSampler的不等长处理（A.4.4）"),
         6: (part6, "BatchNorm打破等价（A.4.4）"),
         7: (part7, "副本同步（A.4.4）")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="*", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if any(part not in PARTS for part in args.parts):
        parser.error("部分号须为1–7")
    torch.set_num_threads(2)
    emit(来源="环境", 附录="A", 种子=args.seed, torch版本=torch.__version__, 默认行为="CPU；离线；不开进程组；不写文件")
    for part in args.parts or PARTS:
        torch.manual_seed(args.seed)
        fn, title = PARTS[part]
        emit(来源="环境", 部分=part, 内容=title)
        fn(args)


if __name__ == "__main__":
    main()
