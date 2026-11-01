"""附录 B：FlashAttention。分块注意力 + online softmax，并按外存模型计数。不做 GPU 测量。

python -m book.appendix.b_flash_attention.tour [部分号 1–6 ...]

计数模型（Dao et al. 2022 的外存模型的教学版，字节换成“标量个数”）：
- 两级存储：慢速存储（HBM，显存）无限大；快速存储（SRAM）容量 M 个标量。
- 输入 Q、K、V 和输出 O 都在慢速存储；运算只在快速存储里做。
- 标量每次进出慢速存储记一次访问。不算反向，不算缓存行、突发传输、并行度。
- 标准注意力：S、P 是 N×N，只要装不下快速存储（N² > M）就写回慢速存储。
  Q、K、V 只读一遍，这是对标准注意力有利的假设。
- FLOPs：一次乘加算 2；除、减、exp、比较各算 1；因果 mask 不省计算。
"""
import argparse
from dataclasses import dataclass
import math
import os

os.environ["HF_HUB_OFFLINE"] = "1"
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "2"

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from book.appendix import emit
from miniocr.train.memory import SavedTensors


@dataclass
class Counter:
    """记录快速/慢速存储之间的访问和 FLOPs。"""
    reads: int = 0
    writes: int = 0
    flops: int = 0
    peak_fast: int = 0

    @property
    def accesses(self):
        return self.reads+self.writes

    def fast(self, n):
        self.peak_fast = max(self.peak_fast, n)


def _null():
    return Counter()


def standard_attention(q, k, v, mask=None, counter=None, M=None):
    """标准注意力。q、k、v 形状 (N, D)；mask 为 bool (N, N)，True 表示可见。"""
    c = counter if counter is not None else _null()
    N, D = q.shape
    spill = M is None or fast_needed(N, N, D) > M  # S、P 放不下快速存储，要写回慢速存储。
    s = q @ k.T/math.sqrt(D)
    c.reads += 2*N*D             # 读 Q、K
    c.flops += 2*N*N*D+N*N       # 矩阵乘 + 除以 √D
    if spill:
        c.writes += N*N          # 写 S
        c.reads += N*N           # 读 S
    if mask is not None:
        s = s.masked_fill(~mask, -math.inf)
    p = s.softmax(-1)
    c.flops += 5*N*N             # 每个元素：比较、减最大值、exp、求和、除
    if spill:
        c.writes += N*N          # 写 P
        c.reads += N*N           # 读 P
    o = p @ v
    c.reads += N*D               # 读 V
    c.flops += 2*N*N*D
    c.writes += N*D              # 写 O
    return o


def online_softmax_update(s, m_old, l_old, o_old, v):
    """处理一块分数 s (Br, Bc)。m 是目前的最大值，l 是目前 exp(s−m) 的和，o 是未归一化的输出。"""
    m_new = torch.maximum(m_old, s.amax(-1))
    # 全被遮挡时 m 还是 −inf。权重和缩放系数直接取 0，避免 inf−inf。
    safe_m = torch.where(m_new > -math.inf, m_new, torch.zeros_like(m_new))
    weights = torch.where(s > -math.inf, (s-safe_m.unsqueeze(-1)).exp(), torch.zeros_like(s))
    alpha = torch.where(m_old > -math.inf, (m_old-safe_m).exp(), torch.zeros_like(m_old))
    return m_new, alpha*l_old+weights.sum(-1), alpha.unsqueeze(-1)*o_old+weights @ v


def tile_row(q_i, k, v, Bc, mask_i=None, counter=None):
    """算一个外层块（Br 行）的输出。沿 N 列分块走一遍，不保存 S、P。"""
    c = counter if counter is not None else _null()
    Br, D = q_i.shape
    N = k.shape[0]
    c.reads += Br*D                                 # 读 Q_i
    m = torch.full((Br,), -math.inf, dtype=q_i.dtype)
    l = torch.zeros(Br, dtype=q_i.dtype)
    o = torch.zeros(Br, D, dtype=q_i.dtype)
    for j in range(0, N, Bc):
        k_j, v_j = k[j:j+Bc], v[j:j+Bc]
        c.reads += 2*k_j.shape[0]*D                 # 读 K_j、V_j
        s = q_i @ k_j.T/math.sqrt(D)
        if mask_i is not None:
            s = s.masked_fill(~mask_i[:, j:j+Bc], -math.inf)
        w = k_j.shape[0]
        c.flops += 2*Br*w*D+Br*w                    # QKᵀ + 除以 √D
        c.flops += 4*Br*w                           # 每个元素：比较、减 m、exp、求和
        c.flops += Br*(5+D)                         # 每行：比 m、减、exp、l 乘加（2）；o 缩放（D）
        c.flops += 2*Br*w*D+Br*D                    # P·V 乘加；累加到 o
        m, l, o = online_softmax_update(s, m, l, o, v_j)
        c.fast(2*Br*D+2*w*D+2*Br*w+2*Br)
    c.flops += Br*D                                 # 最后除以 l
    c.writes += Br*D                                # 写 O_i
    return o/l.unsqueeze(-1)


def flash_attention(q, k, v, Br, Bc, mask=None, counter=None, recompute=False):
    """分块注意力。recompute=True 时每个外层块用 checkpoint，反向时重算，不保存块内结果。"""
    outs = []
    for i in range(0, q.shape[0], Br):
        mask_i = None if mask is None else mask[i:i+Br]
        if recompute:
            outs.append(checkpoint(tile_row, q[i:i+Br], k, v, Bc, mask_i, counter, use_reentrant=False))
        else:
            outs.append(tile_row(q[i:i+Br], k, v, Bc, mask_i, counter))
    return torch.cat(outs)


def standard_accesses(N, D, M=None):
    """闭式：4ND+4N²（S、P 装不下）或 4ND（装得下）。"""
    return 4*N*D+(0 if M is not None and fast_needed(N, N, D) <= M else 4*N*N)


def standard_flops(N, D):
    """闭式：4N²D+6N²。"""
    return 4*N*N*D+6*N*N


def flash_flops(N, D, Bc):
    """闭式（Bc 整除 N）：4N²D+5N²+(5+2D)N²/Bc+ND。第一项和标准注意力相同。"""
    return 4*N*N*D+5*N*N+(5+2*D)*N*N//Bc+N*D


def flash_accesses(N, D, Br):
    """闭式（Br 整除 N）：外层块数 N/Br，每块读 2ND，再加读 Q、写 O。"""
    return 2*N*D+2*N*N*D//Br


def fast_needed(Br, Bc, D):
    """快速存储峰值：Q_i、O_i、K_j、V_j、S、P、m、l。"""
    return 2*Br*D+2*Bc*D+2*Br*Bc+2*Br


def choose_blocks(N, D, M):
    """在 2 的幂里搜索放得下的 (Br, Bc)：先让访问最少，再让 FLOPs 最少（Bc 大）。"""
    best = None
    for r in range(int(math.log2(N))+1):
        for b in range(int(math.log2(N))+1):
            Br, Bc = 2**r, 2**b
            if fast_needed(Br, Bc, D) <= M and (best is None or (Br, Bc) > best):
                best = (Br, Bc)
    if best is None:
        raise ValueError("M 太小，放不下一个块")
    return best


def random_inputs(N, D, seed=0, requires_grad=False):
    g = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(N, D, generator=g, dtype=torch.float64) for _ in range(3))
    return tuple(t.requires_grad_(requires_grad) for t in (q, k, v))


def causal(N):
    return torch.ones(N, N, dtype=torch.bool).tril()


def count_pair(N, D, Br, Bc):
    q, k, v = random_inputs(N, D)
    a, b = Counter(), Counter()
    standard_attention(q, k, v, counter=a)
    flash_attention(q, k, v, Br, Bc, counter=b)
    return a, b


def part1(args):
    N, D = 64, 8
    q, k, v = random_inputs(N, D, args.seed)
    reference = F.scaled_dot_product_attention(q, k, v)
    emit(设备="cpu", dtype="float64", N=N, D=D, 手写标准与SDPA最大差=(standard_attention(q, k, v)-reference).abs().max().item())
    for Br, Bc in ((1, 1), (4, 4), (8, 16), (16, 8), (64, 64), (5, 7)):
        emit(设备="cpu", dtype="float64", Br=Br, Bc=Bc,
             无mask最大差=(flash_attention(q, k, v, Br, Bc)-reference).abs().max().item(),
             因果mask最大差=(flash_attention(q, k, v, Br, Bc, causal(N))-standard_attention(q, k, v, causal(N))).abs().max().item())


def part2(args):
    s = torch.tensor([[1., 2., 3., 0.]], dtype=torch.float64)
    v = torch.eye(4, dtype=torch.float64)[:, :2]
    m = torch.full((1,), -math.inf, dtype=torch.float64)
    l = torch.zeros(1, dtype=torch.float64)
    o = torch.zeros(1, 2, dtype=torch.float64)
    for j in (0, 2):
        m, l, o = online_softmax_update(s[:, j:j+2], m, l, o, v[j:j+2])
        emit(来源="数学结论", 处理到列=j+1, m=m.item(), l=l.item(), o=o[0].tolist())
    emit(来源="数学结论", 分块结果=(o/l.unsqueeze(-1))[0].tolist(), 一次softmax=s.softmax(-1)[0, :2].tolist())


def part3(args):
    N, D, Br, Bc = 16, 4, 4, 4
    a, b = count_pair(N, D, Br, Bc)
    emit(来源="数学结论（计数器）", 算法="标准", N=N, D=D, 读=a.reads, 写=a.writes, 访问=a.accesses, 闭式=standard_accesses(N, D), FLOPs=a.flops, FLOPs闭式=standard_flops(N, D))
    emit(来源="数学结论（计数器）", 算法="分块", N=N, D=D, Br=Br, Bc=Bc, 读=b.reads, 写=b.writes, 访问=b.accesses,
         闭式=flash_accesses(N, D, Br), FLOPs=b.flops, FLOPs闭式=flash_flops(N, D, Bc), 快速存储峰值=b.peak_fast, 公式=fast_needed(Br, Bc, D))
    emit(来源="数学结论（计数器）", 访问比_分块比标准=b.accesses/a.accesses, FLOPs比_分块比标准=b.flops/a.flops)


def part4(args):
    N, D = 64, 8
    a = Counter()
    q, k, v = random_inputs(N, D)
    standard_attention(q, k, v, counter=a)
    emit(来源="数学结论（计数器）", 算法="标准", N=N, D=D, 访问=a.accesses, FLOPs=a.flops)
    for Br, Bc in ((1, 1), (1, 64), (4, 4), (8, 8), (16, 16), (32, 32), (64, 64)):
        b = Counter()
        flash_attention(q, k, v, Br, Bc, counter=b)
        emit(来源="数学结论（计数器）", Br=Br, Bc=Bc, 访问=b.accesses, 闭式=flash_accesses(N, D, Br), 快速存储峰值=b.peak_fast,
             FLOPs=b.flops, 访问比=b.accesses/a.accesses, FLOPs比=b.flops/a.flops)


def part5(args):
    N, D = 64, 8
    emit(来源="数学结论", 说明="给定M搜索块大小；标准注意力在N²>M时写回S、P", N=N, D=D, N平方=N*N)
    for M in (256, 1024, 4096, 16384):
        Br, Bc = choose_blocks(N, D, M)
        b = Counter()
        q, k, v = random_inputs(N, D)
        flash_attention(q, k, v, Br, Bc, counter=b)
        emit(来源="数学结论（计数器）", M=M, Br=Br, Bc=Bc, 快速存储峰值=b.peak_fast, 分块访问=b.accesses,
             标准访问=standard_accesses(N, D, M), 访问比=b.accesses/standard_accesses(N, D, M),
             FLOPs比=b.flops/standard_flops(N, D))
    for M in (65536, 262144):
        N2, D2 = 1024, 64
        Br, Bc = choose_blocks(N2, D2, M)
        emit(来源="数学结论（闭式）", N=N2, D=D2, M=M, Br=Br, Bc=Bc, 分块访问=flash_accesses(N2, D2, Br),
             标准访问=standard_accesses(N2, D2, M), 访问比=flash_accesses(N2, D2, Br)/standard_accesses(N2, D2, M),
             FLOPs比=flash_flops(N2, D2, Bc)/standard_flops(N2, D2), N方D方除M=N2*N2*D2*D2//M, 说明="M只是教学取值，不代表某款GPU")


def saved_bytes(fn, N=32, D=8):
    q, k, v = random_inputs(N, D, requires_grad=True)
    with SavedTensors() as saved:
        loss = fn(q, k, v).square().sum()
    loss.backward()
    return saved.bytes, torch.cat([t.grad.flatten() for t in (q, k, v)])


def part6(args):
    for N in (32, 64, 128):
        std, g0 = saved_bytes(standard_attention, N)
        plain, g1 = saved_bytes(lambda q, k, v: flash_attention(q, k, v, 8, 8), N)
        redo, g2 = saved_bytes(lambda q, k, v: flash_attention(q, k, v, 8, 8, recompute=True), N)
        emit(设备="cpu", dtype="float64", N=N, D=8, Br=8, Bc=8, 标准保存字节=std, 分块autograd保存字节=plain,
             分块加重算保存字节=redo, 分块梯度差=(g0-g1).abs().max().item(), 重算梯度差=(g0-g2).abs().max().item(),
             说明="SavedTensors口径；autograd会保存块内结果；真实内核不保存S、P，靠重算或手写反向")


PARTS = {1: (part1, "和标准注意力对拍（B.4.1）"),
         2: (part2, "online softmax手算（B.4.2）"),
         3: (part3, "计数：N=16的数值例子（B.4.3）"),
         4: (part4, "计数：块大小扫描（B.4.4）"),
         5: (part5, "计数：给定快速存储容量（B.4.5）"),
         6: (part6, "反向与重算（B.4.6）")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="*", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if any(part not in PARTS for part in args.parts):
        parser.error("部分号须为1–6")
    torch.set_num_threads(2)
    emit(来源="环境", 附录="B", 种子=args.seed, torch版本=torch.__version__, 默认行为="CPU；float64小张量；计数非GPU测量；不写文件")
    for part in args.parts or PARTS:
        torch.manual_seed(args.seed)
        fn, title = PARTS[part]
        emit(来源="环境", 部分=part, 内容=title)
        fn(args)


if __name__ == "__main__":
    main()
