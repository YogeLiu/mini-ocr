"""第 1 章　张量、形状与广播：配套实验脚本。正文里标为“本项目观察”的数字来自这里。

    python -m book.part1_foundations.ch01_tensors.tour              # 全部 10 个部分，CPU 上几秒钟
    python -m book.part1_foundations.ch01_tensors.tour 2 3          # 只运行第 2、3 部分
    python -m book.part1_foundations.ch01_tensors.tour 1 10 --device mps    # 第 1、10 部分在 Mac 上跑
    python -m book.part1_foundations.ch01_tensors.tour 1 10 --device cuda   # 在 NVIDIA GPU 上跑

不加 --device 时一律用 CPU。只有第 1 部分（设备移动）和第 10 部分（跨设备对照）会用到 --device。

部分与正文小节的对应：
    1  张量的三个属性：shape、dtype、device（1.1）   6  einops 与等价的 PyTorch 写法（1.7）
    2  广播规则：一步步对齐形状（1.2）              7  one-hot 与 (B, T, V)（1.8）
    3  两两距离（1.3）                            8  公式图片变成张量（1.9）
    4  逐元素乘法与矩阵乘法（1.4）                  9  常见错误（1.10）
    5  内存布局：stride、view、reshape（1.5–1.6）   10 跨设备对照与计时（1.1，实验手册）
"""
import io
import os
import sys
import time

# 除非明确要求 --device cuda，否则对 PyTorch 隐藏 GPU（本机的 GPU 正被别的服务占用）。
if "cuda" not in sys.argv:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

from miniocr.device import device_from_argv, synchronize
from miniocr.tensor_utils import (batch_to_image, broadcast_shape, image_to_batch, one_hot,
                                  pairwise_sq_dists, pairwise_sq_dists_mm)

OUT = "outputs/book/ch01"
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(min(8, os.cpu_count() or 1))
DEVICE = device_from_argv(default="cpu")   # 只有第 1、10 部分用它


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def fmt(t, nd=4):
    """把张量打印成嵌套列表，浮点数保留 nd 位小数。"""
    if t.dim() == 0:
        return round(t.item(), nd) if t.is_floating_point() else t.item()
    if t.dim() > 1:
        return [fmt(r, nd) for r in t]
    return [round(x, nd) for x in t.tolist()] if t.is_floating_point() else t.tolist()


def shp(t):
    return tuple(t.shape)


# ---------------------------------------------------------------------- 1
def part1_attributes():
    section("1. 张量的三个属性：shape、dtype、device（1.1）")
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f"  x = {x.tolist()}")
    print(f"    shape {shp(x)}  dim {x.dim()}  numel {x.numel()}  dtype {x.dtype}  device {x.device}")
    print(f"    每个元素 {x.element_size()} 字节，共 {x.numel() * x.element_size()} 字节")
    y = torch.tensor([[1.0, 2, 3], [4, 5, 6]])
    print(f"  写成 1.0 以后：dtype {y.dtype}，每个元素 {y.element_size()} 字节")
    print("  常见 dtype 每个元素占的字节数：")
    for dt in (torch.float64, torch.float32, torch.bfloat16, torch.float16, torch.int64, torch.int32, torch.uint8, torch.bool):
        print(f"    {str(dt):16s} {torch.empty(0, dtype=dt).element_size()} 字节")
    print(f"  dtype 转换：x.float() → {x.float().dtype}；x.to(torch.uint8) → {x.to(torch.uint8).dtype}")
    print(f"  整数相除：x / 2 = {fmt(x / 2, 1)}（dtype {(x / 2).dtype}）；x // 2 = {(x // 2).tolist()}（dtype {(x // 2).dtype}）")
    print()
    print(f"  设备：本次运行选的是 {DEVICE}（默认 cpu；加 --device mps 或 --device cuda 换设备）")
    print(f"    torch.cuda.is_available() = {torch.cuda.is_available()}；torch.backends.mps.is_available() = {torch.backends.mps.is_available()}")
    z = x.to(DEVICE)
    print(f"    z = x.to({str(DEVICE)!r}) → z.device = {z.device}；x 本身还在 {x.device}（.to 返回新张量，不改原来的）")
    back = z.cpu()
    print(f"    z.cpu() 搬回来，和 x 相等：{torch.equal(back, x)}")
    same = x.to("cpu")
    print(f"    已经在目标设备上时，.to 直接返回原张量本身：x.to('cpu') is x → {same is x}")


# ---------------------------------------------------------------------- 2
def show_align(a, b):
    """把广播的三步打印出来：右对齐 → 左边补 1 → 逐维取值。"""
    a, b = tuple(a), tuple(b)
    n = max(len(a), len(b))
    pa, pb = (1,) * (n - len(a)) + a, (1,) * (n - len(b)) + b
    print(f"    第 1 步 右对齐：     {str(a):>16s}")
    print(f"                       {str(b):>16s}")
    print(f"    第 2 步 左边补 1：   {str(pa):>16s}")
    print(f"                       {str(pb):>16s}")
    try:
        out = broadcast_shape(a, b)
        print(f"    第 3 步 逐维取值：   {str(out):>16s}")
        return out
    except ValueError as e:
        print(f"    第 3 步 逐维取值：   失败——{e}")
        return None


def part2_broadcast():
    section("2. 广播规则：一步步对齐形状（1.2）")
    a = torch.tensor([[0], [10], [20]])           # (3, 1)
    b = torch.tensor([0, 1, 2, 3])                # (4,)
    print(f"  例 1：a = {a.tolist()} 形状 {shp(a)}，b = {b.tolist()} 形状 {shp(b)}")
    show_align(a.shape, b.shape)
    print(f"    a + b = {(a + b).tolist()}  形状 {shp(a + b)}")
    print(f"    等价于先把两者都“复制”成 (3, 4)：a.expand(3, 4) = {a.expand(3, 4).tolist()}")
    print(f"                                   b.expand(3, 4) = {b.expand(3, 4).tolist()}")
    print(f"    但 expand 不复制数据：a.expand(3, 4).stride() = {a.expand(3, 4).stride()}（第 1 维步长 0：一直读同一个数）")
    print()
    X = torch.tensor([[1, 2, 3], [4, 5, 6]])     # (2, 3)
    bias = torch.tensor([10, 20, 30])            # (3,)
    print(f"  例 2：每一行加同一个偏置。X 形状 {shp(X)}，bias = {bias.tolist()} 形状 {shp(bias)}")
    show_align(X.shape, bias.shape)
    print(f"    X + bias = {(X + bias).tolist()}")
    print()
    rowv = torch.tensor([100, 200])               # (2,)
    print(f"  例 3：想给每一行加不同的数。r = {rowv.tolist()} 形状 {shp(rowv)}")
    show_align(X.shape, rowv.shape)
    try:
        X + rowv
    except RuntimeError as e:
        print(f"    PyTorch 的报错：{str(e).splitlines()[0]}")
    print(f"    改成 r[:, None]，形状 {shp(rowv[:, None])}：")
    show_align(X.shape, rowv[:, None].shape)
    print(f"    X + r[:, None] = {(X + rowv[:, None]).tolist()}")
    print()
    print("  例 4：后面章节会遇到的形状（只算形状）")
    for name, s1, s2 in (("logits + 词表偏置（第 7 章）", (8, 16, 100), (100,)),
                         ("注意力分数 + 补齐 mask（第 8 章）", (8, 4, 16, 16), (8, 1, 1, 16)),
                         ("图片减去每通道均值（第 27 章）", (8, 3, 64, 256), (3, 1, 1)),
                         ("每通道均值写错成 (3,)", (8, 3, 64, 256), (3,))):
        print(f"    {name}：{s1} 与 {s2}")
        show_align(s1, s2)
    print(f"  和 PyTorch 对拍：torch.broadcast_shapes((8, 4, 16, 16), (8, 1, 1, 16)) = {tuple(torch.broadcast_shapes((8, 4, 16, 16), (8, 1, 1, 16)))}")


# ---------------------------------------------------------------------- 3
A3 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])     # (N=3, D=2)
B3 = torch.tensor([[1.0, 1.0], [3.0, 0.0]])                 # (M=2, D=2)


def part3_pairwise():
    section("3. 两两距离（1.3）：用广播代替两重循环")
    A, B = A3, B3
    print(f"  A = {A.tolist()}  形状 {shp(A)}（3 个点）")
    print(f"  B = {B.tolist()}  形状 {shp(B)}（2 个点）")
    print("  两重循环的写法：")
    loop = torch.zeros(3, 2)
    for i in range(3):
        for j in range(2):
            loop[i, j] = ((A[i] - B[j]) ** 2).sum()
    print(f"    D = {fmt(loop, 1)}")
    a3, b3 = A[:, None, :], B[None, :, :]
    print("  广播的写法，每一步的形状：")
    print(f"    A[:, None, :]  形状 {shp(a3)}")
    print(f"    B[None, :, :]  形状 {shp(b3)}")
    show_align(a3.shape, b3.shape)
    diff = a3 - b3
    print(f"    diff = A[:, None, :] − B[None, :, :]  形状 {shp(diff)}")
    print(f"      diff[1] = {fmt(diff[1], 1)}  （A 的第 1 个点减去 B 的每一个点）")
    sq = diff ** 2
    D = sq.sum(-1)
    print(f"    (diff ** 2).sum(-1) = {fmt(D, 1)}  形状 {shp(D)}")
    print(f"    和循环结果一致：{torch.equal(D, loop)}；和 pairwise_sq_dists 一致：{torch.equal(pairwise_sq_dists(A, B), loop)}")
    print(f"    和 torch.cdist(A, B) ** 2 的最大差：{(torch.cdist(A, B) ** 2 - D).abs().max().item():.1e}")
    print(f"    每个 A 点最近的 B 点：D.argmin(1) = {D.argmin(1).tolist()}")
    print()
    print("  展开写法 ||a||² − 2a·b + ||b||²：")
    a2, b2, ab = (A ** 2).sum(-1)[:, None], (B ** 2).sum(-1)[None, :], A @ B.T
    print(f"    ||a||²  形状 {shp(a2)} = {fmt(a2, 1)}")
    print(f"    ||b||²  形状 {shp(b2)} = {fmt(b2, 1)}")
    print(f"    A @ B.T 形状 {shp(ab)} = {fmt(ab, 1)}")
    print(f"    合起来 = {fmt(pairwise_sq_dists_mm(A, B), 1)}")
    print()
    print("  中间张量的大小（数学结论）：广播写法要存 N·M·D 个数；展开写法不产生 (N, M, D)，"
          "额外内存是 N·D + M·D + N·M 量级")
    for N, M, Dm in ((1000, 1000, 64), (4096, 4096, 512)):
        print(f"    N = M = {N}，D = {Dm}，float32：广播 {N * M * Dm * 4 / 2**20:,.0f} MiB，展开 {(N * Dm + M * Dm + N * M) * 4 / 2**20:,.0f} MiB（其中 (N, M) 占 {N * M * 4 / 2**20:,.0f} MiB）")
    torch.manual_seed(0)
    X, Y = torch.randn(1000, 64), torch.randn(1000, 64)
    for name, fn in (("广播", pairwise_sq_dists), ("展开", pairwise_sq_dists_mm)):
        fn(X, Y)
        t0 = time.perf_counter()
        for _ in range(10):
            fn(X, Y)
        print(f"    实测 N = M = 1000、D = 64（CPU，10 次平均）：{name} {(time.perf_counter() - t0) / 10 * 1000:.2f} ms")
    print(f"    两种写法的最大差：{(pairwise_sq_dists(X, Y) - pairwise_sq_dists_mm(X, Y)).abs().max().item():.1e}")


# ---------------------------------------------------------------------- 4
def part4_mul_vs_matmul():
    section("4. 逐元素乘法 * 与矩阵乘法 @（1.4）")
    P = torch.tensor([[1, 2], [3, 4]])
    Q = torch.tensor([[5, 6], [7, 8]])
    print(f"  P = {P.tolist()}，Q = {Q.tolist()}")
    print(f"    P * Q = {(P * Q).tolist()}   （对应位置相乘）")
    print(f"    P @ Q = {(P @ Q).tolist()}   （行乘列再求和：19 = 1×5 + 2×7）")
    v = torch.tensor([1, 10])
    print(f"  v = {v.tolist()} 形状 {shp(v)}")
    print(f"    P * v = {(P * v).tolist()}  （广播：每一行逐元素乘 v，即每一列乘一个数）")
    print(f"    P @ v = {(P @ v).tolist()}  形状 {shp(P @ v)}  （矩阵乘向量）")
    print(f"    v @ v = {(v @ v).item()}  形状 {shp(v @ v)}  （两个 1 维张量 @ 是点积）")
    print("  @ 的形状规则：(..., n, k) @ (k, m) → (..., n, m)；前面的维度按广播对齐")
    for s1, s2 in (((3, 4), (4, 5)), ((2, 5, 8), (8, 11)), ((2, 5, 8), (2, 8, 11)), ((2, 1, 5, 8), (3, 8, 11)), ((3, 4), (3, 5))):
        try:
            out = shp(torch.zeros(s1) @ torch.zeros(s2))
            print(f"    {str(s1):14s} @ {str(s2):12s} → {out}")
        except RuntimeError as e:
            print(f"    {str(s1):14s} @ {str(s2):12s} → 报错：{str(e).splitlines()[0]}")


# ---------------------------------------------------------------------- 5
def part5_layout():
    section("5. 内存布局：stride、view、reshape、连续性（1.5–1.6）")
    x = torch.arange(12).view(3, 4)
    print(f"  x = torch.arange(12).view(3, 4) = {x.tolist()}")
    print(f"    底层存储（一条线）：{x.flatten().tolist()}")
    print(f"    x.stride() = {x.stride()}：沿第 0 维走一步跳 4 个数，沿第 1 维走一步跳 1 个数")
    print(f"    x[2, 1] 在存储里的位置 = 2×4 + 1×1 = 9，值 {x[2, 1].item()}")
    print(f"    x.is_contiguous() = {x.is_contiguous()}")
    y = x.view(4, 3)
    print(f"  y = x.view(4, 3) = {y.tolist()}  stride {y.stride()}")
    print(f"    y 和 x 共用存储：y.data_ptr() == x.data_ptr() → {y.data_ptr() == x.data_ptr()}")
    y[0, 0] = 100
    print(f"    改 y[0, 0] = 100 之后，x[0, 0] = {x[0, 0].item()}（一起变了）")
    x[0, 0] = 0
    print(f"  view 的 -1：x.view(2, -1) 形状 {shp(x.view(2, -1))}；x.view(-1) 形状 {shp(x.view(-1))}")
    t = x.t()
    print(f"  t = x.t()（转置）= {t.tolist()}")
    print(f"    t.stride() = {t.stride()}，t.is_contiguous() = {t.is_contiguous()}，共用存储：{t.data_ptr() == x.data_ptr()}")
    print(f"    转置只是交换了 stride，没有移动任何数据")
    try:
        t.view(12)
    except RuntimeError as e:
        print(f"    t.view(12) 报错：{str(e).splitlines()[0][:70]}…")
    r = t.reshape(12)
    print(f"    t.reshape(12) = {r.tolist()}，共用存储：{r.data_ptr() == x.data_ptr()}（reshape 在需要时复制）")
    c = t.contiguous()
    print(f"    t.contiguous().stride() = {c.stride()}，共用存储：{c.data_ptr() == x.data_ptr()}（复制成新的连续存储）")
    r2 = x.reshape(4, 3)
    print(f"    对连续的 x，x.reshape(4, 3) 共用存储：{r2.data_ptr() == x.data_ptr()}（能不复制就不复制）")
    print()
    z = torch.arange(24).view(2, 3, 4)
    print(f"  三维的例子：z = torch.arange(24).view(2, 3, 4)，stride {z.stride()}")
    for name, w in (("z.transpose(0, 2)", z.transpose(0, 2)), ("z.permute(2, 0, 1)", z.permute(2, 0, 1)),
                    ("z.permute(0, 2, 1)", z.permute(0, 2, 1)), ("z.unsqueeze(1)", z.unsqueeze(1)),
                    ("z[:, None]", z[:, None]), ("z[:, :, 0]", z[:, :, 0])):
        print(f"    {name:20s} 形状 {str(shp(w)):14s} stride {str(w.stride()):16s} 连续 {w.is_contiguous()}")
    zc = z[:, :, 0]
    print(f"  不连续不等于不能 view：z[:, :, 0] 连续 {zc.is_contiguous()}，但 z[:, :, 0].view(-1) = {zc.view(-1).tolist()}"
          f"（目标形状能用一组 stride 表示，view 就成功）")
    s = torch.arange(12).view(3, 4)[1:]
    print(f"  偏移量：s = torch.arange(12).view(3, 4)[1:]，s.storage_offset() = {s.storage_offset()}，stride {s.stride()}，"
          f"s[0, 0] = {s[0, 0].item()}，位置 = 偏移量 + 0×4 + 0×1 = {s.storage_offset()}")
    print(f"  transpose 只能交换两维，permute 一次给出所有维的新顺序：z.transpose(1, 2) 等于 z.permute(0, 2, 1) → "
          f"{torch.equal(z.transpose(1, 2), z.permute(0, 2, 1))}")


# ---------------------------------------------------------------------- 6
def part6_einops():
    section("6. einops 与等价的 PyTorch 写法（1.7）：代码就是形状说明")
    x = torch.arange(2 * 3 * 4 * 6).view(2, 3, 4, 6).float()        # (B, C, H, W)
    print(f"  x 形状 {shp(x)}，按 (B, C, H, W) 理解")
    cases = [
        ("通道放到最后", "rearrange(x, 'b c h w -> b h w c')", rearrange(x, "b c h w -> b h w c"),
         "x.permute(0, 2, 3, 1)", x.permute(0, 2, 3, 1)),
        ("把每个像素变成一个 token", "rearrange(x, 'b c h w -> b (h w) c')", rearrange(x, "b c h w -> b (h w) c"),
         "x.flatten(2).transpose(1, 2)", x.flatten(2).transpose(1, 2)),
        ("切成 2×2 的小块（ViT，第 29 章）", "rearrange(x, 'b c (h p) (w q) -> b (h w) (c p q)', p=2, q=2)",
         rearrange(x, "b c (h p) (w q) -> b (h w) (c p q)", p=2, q=2),
         "x.view(2, 3, 2, 2, 3, 2).permute(0, 2, 4, 1, 3, 5).reshape(2, 6, 12)",
         x.view(2, 3, 2, 2, 3, 2).permute(0, 2, 4, 1, 3, 5).reshape(2, 6, 12)),
        ("每张图每个通道求平均", "reduce(x, 'b c h w -> b c', 'mean')", reduce(x, "b c h w -> b c", "mean"),
         "x.mean(dim=(2, 3))", x.mean(dim=(2, 3))),
        ("灰度图复制成 3 个通道", "repeat(x[:, :1], 'b 1 h w -> b 3 h w')", repeat(x[:, :1], "b 1 h w -> b 3 h w"),
         "x[:, :1].expand(-1, 3, -1, -1)", x[:, :1].expand(-1, 3, -1, -1)),
    ]
    for desc, e_code, e_val, t_code, t_val in cases:
        print(f"  {desc}：")
        print(f"    einops  {e_code}")
        print(f"    PyTorch {t_code}")
        print(f"    形状 {shp(e_val)}，两者相等：{torch.equal(e_val, t_val)}")
    print()
    h = torch.arange(2 * 5 * 8).view(2, 5, 8).float()                  # (B, T, h·d)，h = 2，d = 4
    print(f"  拆成多头（第 8 章）：h 形状 {shp(h)}，按 (B, T, 头数×每头维度) = (2, 5, 2×4) 理解")
    e = rearrange(h, "b t (n d) -> b n t d", n=2)
    good = h.view(2, 5, 2, 4).transpose(1, 2)
    bad = h.transpose(1, 2).reshape(2, 2, 5, 4)
    print(f"    einops  rearrange(h, 'b t (n d) -> b n t d', n=2)        形状 {shp(e)}")
    print(f"    PyTorch h.view(2, 5, 2, 4).transpose(1, 2)              相等：{torch.equal(e, good)}")
    print(f"    错误写法 h.transpose(1, 2).reshape(2, 2, 5, 4)           形状也对，相等：{torch.equal(e, bad)}")
    print(f"      正确：第 0 个头、第 0 个 token = {e[0, 0, 0].tolist()}（token 0 的前 4 维）")
    print(f"      错误：第 0 个头、第 0 个 token = {bad[0, 0, 0].tolist()}（把 4 个不同 token 的第 0 维拼在了一起）")
    alt = h.transpose(1, 2).view(2, 2, 4, 5).transpose(2, 3)
    print(f"    先 transpose 也可以写对：h.transpose(1, 2).view(2, 2, 4, 5).transpose(2, 3) 相等：{torch.equal(e, alt)}"
          f"（拆的是 8 这一维，没有和 token 维混在一起）")
    print("  einops 会检查形状：")
    try:
        rearrange(h, "b t (n d) -> b n t d", n=3)
    except Exception as ex:
        print(f"    rearrange(h, 'b t (n d) -> b n t d', n=3) 报错：{str(ex).splitlines()[-1][:80]}")


# ---------------------------------------------------------------------- 7
def part7_one_hot():
    section("7. one-hot 与 (B, T, V)（1.8）")
    idx = torch.tensor([[0, 2, 1], [3, 3, 0]])                      # (B=2, T=3)
    V = 4
    print(f"  idx = {idx.tolist()}  形状 {shp(idx)}（B = 2 条序列，每条 T = 3 个 token，词表 V = {V}）")
    print(f"    idx[..., None] 形状 {shp(idx[..., None])}，torch.arange(4) 形状 (4,)")
    show_align(idx[..., None].shape, (V,))
    oh = one_hot(idx, V)
    print(f"    one_hot(idx, 4) 形状 {shp(oh)}，dtype {oh.dtype}")
    print(f"      第 1 条序列：{fmt(oh[1], 0)}")
    print(f"    和 F.one_hot(idx, 4) 一致：{torch.equal(oh, F.one_hot(idx, V).float())}")
    print(f"    每个 token 恰好一个 1：oh.sum(-1) = {fmt(oh.sum(-1), 0)}")
    print(f"    还原编号：oh.argmax(-1) = {oh.argmax(-1).tolist()}")
    W = torch.arange(V * 2).view(V, 2).float()                     # 嵌入表 (V, 2)
    print(f"  one-hot 乘矩阵 = 查表：W 形状 {shp(W)} = {fmt(W, 0)}")
    print(f"    oh @ W 形状 {shp(oh @ W)}，和 W[idx] 相等：{torch.equal(oh @ W, W[idx])}")
    print()
    torch.manual_seed(0)
    logits = torch.randn(2, 3, V)
    print(f"  语言模型的输出 logits 形状 {shp(logits)} = (B, T, V)：第 b 条序列第 t 个位置，对词表里每个 token 的打分")
    print(f"    logits[0, 1] = {fmt(logits[0, 1])}  （第 0 条序列第 1 个位置的 4 个分数）")
    p = logits.softmax(-1)
    print(f"    softmax(-1) 后每个位置的概率和：{fmt(p.sum(-1))}")
    print(f"    argmax(-1) 形状 {shp(logits.argmax(-1))} = {logits.argmax(-1).tolist()}（每个位置分数最高的 token）")
    print(f"    沿错了维度 softmax(1) 后的和：{fmt(logits.softmax(1).sum(-1))}（不报错，但不是概率分布）")
    flat = F.cross_entropy(logits.reshape(-1, V), idx.reshape(-1))
    chan = F.cross_entropy(logits.permute(0, 2, 1), idx)
    print(f"  F.cross_entropy 要 (N, V)：logits.reshape(-1, 4) 形状 {shp(logits.reshape(-1, V))}，idx.reshape(-1) 形状 {shp(idx.reshape(-1))}")
    print(f"    或者 (B, V, T)：logits.permute(0, 2, 1)。两种写法的 loss：{flat.item():.4f} 和 {chan.item():.4f}（第 3 章细讲）")


# ---------------------------------------------------------------------- 8
def render_formula(tex, fontsize=24, dpi=100):
    """用 matplotlib 自带的 mathtext 把 LaTeX 公式渲染成图片（不需要安装 LaTeX）。
    返回 (H, W, 3) 的 uint8 张量：白底黑字。"""
    fig = plt.figure(figsize=(2, 1), dpi=dpi)
    fig.text(0.5, 0.5, f"${tex}$", fontsize=fontsize, ha="center", va="center")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    rgba = plt.imread(buf)                                      # numpy (H, W, 4)，float32，取值 [0, 1]
    return (torch.from_numpy(rgba[..., :3].copy()) * 255).round().to(torch.uint8)


def part8_formula_image():
    section("8. 公式图片变成张量（1.9）")
    img = render_formula(r"\frac{a}{b}")
    H, W, C = img.shape
    print(f"  用 mathtext 渲染 \\frac{{a}}{{b}}：张量形状 {shp(img)} = (H, W, C)，dtype {img.dtype}，"
          f"取值 [{img.min().item()}, {img.max().item()}]")
    print(f"    H = {H} 行像素，W = {W} 列像素，C = {C} 个颜色通道（红、绿、蓝）")
    print(f"    占内存 {img.numel() * img.element_size()} 字节 = {H}×{W}×{C}×1")
    print(f"    stride {img.stride()}：同一个像素的 3 个通道在内存里挨着放（通道在最后）")
    print(f"    左上角像素 img[0, 0] = {img[0, 0].tolist()}（白色）")
    dark = (img.float().mean(-1) < 128)
    rows = dark.any(1).nonzero().flatten()
    print(f"    深色像素（三通道平均 < 128）{dark.sum().item()} 个，分布在第 {rows.min().item()}–{rows.max().item()} 行")
    frac = dark.float().mean(1)                                 # (H,)：每一行深色像素占的比例
    bar = (frac > 0.4).nonzero().flatten()
    print(f"    深色像素超过一行 40% 的行：{bar.tolist()}（每行 {frac[bar].mul(W).round().int().tolist()} 个深色像素）——这就是分数线")
    print("    用字符画出来（# 深色 < 128，+ 灰色 < 230，. 白色），第 5–42 行：")
    g = img.float().mean(-1)                                     # (H, W) 灰度
    for i in range(5, 43):
        print(f"      {i:2d} " + "".join("#" if v < 128 else ("+" if v < 230 else ".") for v in g[i].tolist()))
    x = image_to_batch(img)
    print(f"  image_to_batch：形状 {shp(x)} = (B, C, H, W)，dtype {x.dtype}，取值 [{x.min().item():.1f}, {x.max().item():.1f}]")
    y = (img.permute(2, 0, 1)[None].float() / 255)
    print(f"    只做 permute + float：stride {y.stride()}，连续 {y.is_contiguous()}（float() 沿用了原来的内存顺序）")
    print(f"    再 contiguous()：stride {x.stride()}，连续 {x.is_contiguous()}：同一个通道的所有像素挨着放")
    print(f"    占内存 {x.numel() * x.element_size()} 字节")
    print(f"    einops 写法相等：{torch.equal(x, rearrange(img, 'h w c -> 1 c h w').float() / 255)}")
    print(f"    还原回去和原图一致：{torch.equal(batch_to_image(x), img)}")
    gray = x.mean(1, keepdim=True)
    print(f"  转成灰度：x.mean(1, keepdim=True) 形状 {shp(gray)}；不加 keepdim 是 {shp(x.mean(1))}")
    conv = torch.nn.Conv2d(1, 4, kernel_size=3)                    # 1 个输入通道、4 个输出通道
    print(f"    送进 Conv2d(1, 4, 3)：(1, 1, H, W) 输出 {shp(conv(gray))}；"
          f"(1, H, W) 被当成无批量维的 (C, H, W)，不报错，输出 {shp(conv(x.mean(1)))}，批量维没了")
    img2 = render_formula(r"x^2")
    print(f"  再渲染 x^2：形状 {shp(img2)}")
    try:
        torch.stack([image_to_batch(img)[0], image_to_batch(img2)[0]])
    except RuntimeError as e:
        print(f"    两张大小不同的图不能直接 stack：{str(e).splitlines()[0]}")
    Hm, Wm = max(H, img2.shape[0]), max(W, img2.shape[1])
    def pad(t):                                         # 右边和下边补白色
        t = image_to_batch(t)[0]
        return F.pad(t, (0, Wm - t.shape[2], 0, Hm - t.shape[1]), value=1.0)
    batch = torch.stack([pad(img), pad(img2)])
    print(f"    右边、下边补白到 ({Hm}, {Wm}) 再 stack：batch 形状 {shp(batch)} = (B, C, H, W)")
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.5))
    axes[0].imshow(img.numpy()); axes[0].set_title(f"(H, W, C) = {shp(img)}")
    axes[1].imshow(gray[0, 0].numpy(), cmap="gray", vmin=0, vmax=1); axes[1].set_title(f"gray (1, 1, H, W)")
    axes[2].imshow(batch_to_image(batch, 1).numpy()); axes[2].set_title(f"batch[1] of {shp(batch)}")
    for ax in axes:
        ax.set_xlabel("W"); ax.set_ylabel("H")
    fig.tight_layout()
    path = os.path.join(OUT, "formula_tensor.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  图片已保存：{path}")


# ---------------------------------------------------------------------- 9
def part9_pitfalls():
    section("9. 常见错误（1.10）")
    pred = torch.tensor([1.0, 2.0, 3.0, 4.0])                 # (4,)
    target = torch.tensor([[1.0], [2.0], [3.0], [5.0]])       # (4, 1)
    wrong = ((pred - target) ** 2).mean()
    right = ((pred - target.squeeze(1)) ** 2).mean()
    print(f"  (a) 静默广播：pred 形状 {shp(pred)}，target 形状 {shp(target)}")
    print(f"      pred − target 形状 {shp(pred - target)}（不是 (4,)！）")
    print(f"      均方误差：写错 {wrong.item():.4f}，正确 {right.item():.4f}")
    print("      防御：关键位置写 assert pred.shape == target.shape")
    m = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    print(f"  (b) 求和后丢了维度：想让每一行和为 1，m = {fmt(m, 0)}")
    print(f"      m / m.sum(1) 的每行和：{fmt((m / m.sum(1)).sum(1))}（方阵时不报错，除的是列）")
    print(f"      m / m.sum(1, keepdim=True) 的每行和：{fmt((m / m.sum(1, keepdim=True)).sum(1))}")
    u = torch.tensor([200, 100], dtype=torch.uint8)
    print(f"  (c) uint8 溢出：图片像素 {u.tolist()}，u + u = {(u + u).tolist()}（200 + 200 = 400，超过 255，400 − 256 = 144）")
    print(f"      先转浮点：u.float() + u.float() = {fmt(u.float() + u.float(), 0)}")
    print(f"      u − 250 = {(u - torch.tensor(250, dtype=torch.uint8)).tolist()}（负数绕回到大正数）")
    a = torch.ones(2)
    try:
        a + torch.ones(2, device="meta")
    except RuntimeError as e:
        print(f"  (d) 设备不一致（用 meta 设备在 CPU 上模拟）：{str(e).splitlines()[0]}")
    try:
        torch.zeros(3).add_(torch.ones(2, 3))
    except RuntimeError as ex:
        print(f"  (e0) 原地运算不能靠广播改变自己的形状：torch.zeros(3).add_(torch.ones(2, 3)) 报错：{str(ex).splitlines()[0][:70]}")
    s = torch.zeros(3)
    try:
        e = s.expand(2, 3)
        e.add_(1)
    except RuntimeError as ex:
        print(f"  (e) 对 expand 出来的张量原地修改：{str(ex).splitlines()[0][:80]}…")
    torch.manual_seed(0)
    X = torch.randn(200, 64, dtype=torch.float32) + 100
    d_mm = (X ** 2).sum(-1)[:, None] - 2 * (X @ X.T) + (X ** 2).sum(-1)[None, :]
    diag = d_mm.diagonal()
    print(f"  (f) 展开写法的舍入误差：200 个 64 维点（每维加 100），float32，点到自己的距离本应是 0")
    print(f"      对角线最小 {diag.min().item():.2f}、最大 {diag.max().item():.2f}，负数 {(diag < 0).sum().item()} 个")
    print(f"      广播写法的对角线最大绝对值 {pairwise_sq_dists(X, X).diagonal().abs().max().item():.2f}")
    print(f"      float64 下展开写法对角线最大绝对值 {pairwise_sq_dists_mm(X.double(), X.double()).diagonal().abs().max().item():.1e}")
    print("      pairwise_sq_dists_mm 用 clamp_min(0) 截掉负数，但误差本身还在；数据先减去均值能大幅减小误差：")
    Xc = X - X.mean(0)
    print(f"      减去均值后 float32 对角线最大绝对值 {pairwise_sq_dists_mm(Xc, Xc).diagonal().abs().max().item():.1e}")


# ---------------------------------------------------------------------- 10
def part10_devices():
    section(f"10. 跨设备对照与计时（1.1，实验手册）：当前设备 {DEVICE}")
    torch.manual_seed(0)
    X, Y = torch.randn(2000, 64), torch.randn(3000, 64)
    ref = pairwise_sq_dists(X.double(), Y.double())                     # float64、CPU 作为参照
    print(f"  X {shp(X)}、Y {shp(Y)}，参照答案：CPU 上 float64 的广播写法")
    Xd, Yd = X.to(DEVICE), Y.to(DEVICE)
    for name, fn in (("广播", pairwise_sq_dists), ("展开", pairwise_sq_dists_mm)):
        fn(Xd, Yd); synchronize(DEVICE)                                 # 预热
        t0 = time.perf_counter()
        for _ in range(5):
            out = fn(Xd, Yd)
        synchronize(DEVICE)
        ms = (time.perf_counter() - t0) / 5 * 1000
        err = (out.cpu().double() - ref).abs().max().item()
        print(f"    {DEVICE.type} float32 {name}：{ms:7.2f} ms，和参照的最大差 {err:.1e}，结果在 {out.device}")
    print("  在 --device mps 或 --device cuda 下再跑一次，对照用时和最大差（实验手册实验三）")


PARTS = {1: part1_attributes, 2: part2_broadcast, 3: part3_pairwise, 4: part4_mul_vs_matmul, 5: part5_layout,
         6: part6_einops, 7: part7_one_hot, 8: part8_formula_image, 9: part9_pitfalls, 10: part10_devices}

if __name__ == "__main__":
    t_start = time.perf_counter()
    for k in [int(a) for a in sys.argv[1:] if a.isdigit()] or list(PARTS):
        PARTS[k]()
    print(f"\n第 1 章的实验全部完成，用时 {time.perf_counter() - t_start:.1f} 秒。")
