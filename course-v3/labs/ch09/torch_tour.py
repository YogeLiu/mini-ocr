"""第 9 章：PyTorch 基础。逐段运行，对照正文阅读。

    python -m labs.ch09.torch_tour
    python -m labs.ch09.torch_tour --cpu      # 强制只用 CPU
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from miniocr.torch_basics import GlyphDataset, MyLinear, pick_device

DEV = torch.device("cpu") if "--cpu" in sys.argv else pick_device()


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def part1_tensors():
    section("1. 张量：和 NumPy 数组几乎一样")
    a = torch.arange(12).reshape(3, 4)
    print("torch.arange(12).reshape(3, 4):\n", a, "\n形状", tuple(a.shape), "类型", a.dtype, "设备", a.device)
    print("a[1, :]", a[1, :].tolist(), "  a[:, 2]", a[:, 2].tolist(), "  a.sum(dim=0)", a.sum(dim=0).tolist())
    x = torch.ones(2, 3) * 0.5
    print("默认的小数类型:", x.dtype, "   torch.tensor([1, 2]) 的类型:", torch.tensor([1, 2]).dtype)
    n = np.zeros(3, dtype=np.float32)
    t = torch.from_numpy(n)
    t[0] = 7
    print("torch.from_numpy 共享内存：改张量后 NumPy 数组变成", n.tolist())
    t2 = torch.tensor(n)
    t2[1] = 9
    print("torch.tensor 复制一份：改张量后 NumPy 数组仍是", n.tolist())
    v = a.T
    print("a.T 是视图，contiguous 吗？", v.is_contiguous(), "  a.T.view(-1) 会报错，a.T.reshape(-1) =", v.reshape(-1).tolist())
    print("unsqueeze/squeeze：", tuple(torch.zeros(32, 32).unsqueeze(0).shape), tuple(torch.zeros(1, 32, 32).squeeze(0).shape))


def part2_autograd():
    section("2. autograd：第 5 章的迷你神经元")
    x, y = torch.tensor(2.0), torch.tensor(1.0)
    w = torch.tensor(0.5, requires_grad=True)
    b = torch.tensor(-0.3, requires_grad=True)
    z = w * x + b
    p = torch.sigmoid(z)
    L = (p - y) ** 2
    print(f"  L = {L.item():.4f}   L.grad_fn = {L.grad_fn.__class__.__name__}   "
          f"它的上一步 {L.grad_fn.next_functions[0][0].__class__.__name__}")
    L.backward()
    print(f"  dL/dw = {w.grad.item():.6f}   dL/db = {b.grad.item():.6f}   （第 5、8 章：-0.294268、-0.147134）")
    print(f"  x.requires_grad = {x.requires_grad}，x.grad = {x.grad}（输入不需要梯度，PyTorch 不为它算）")
    L = (torch.sigmoid(w * x + b) - y) ** 2
    L.backward()
    print(f"  不清零再反向一次：w.grad = {w.grad.item():.6f}（累加了！）")
    w.grad = None
    with torch.no_grad():
        z2 = w * x + b
    print(f"  torch.no_grad() 里算出的 z2.requires_grad = {z2.requires_grad}（不记录计算图：评测和参数更新时用）")
    d = (w * x).detach()
    print(f"  detach() 得到的张量 requires_grad = {d.requires_grad}（从计算图上\"剪断\"）")


def part3_module():
    section("3. nn.Module：自己写的 MyLinear 与 nn.Linear")
    torch.manual_seed(0)
    mine, ref = MyLinear(1024, 62), nn.Linear(1024, 62)
    print("  MyLinear 的参数:", [(n, tuple(p.shape)) for n, p in mine.named_parameters()])
    print("  nn.Linear 的参数:", [(n, tuple(p.shape)) for n, p in ref.named_parameters()], " ← 权重是 (out, in)")
    net = nn.Sequential(nn.Flatten(), nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 62))
    print("  一个 MLP:\n", net)
    print("  参数总数:", sum(p.numel() for p in net.parameters()), "（第 4 章 mlp_param_count([1024,256,62]) = 278334）")
    x = torch.rand(8, 1, 32, 32)
    print("  输入", tuple(x.shape), "-> 输出", tuple(net(x).shape))
    sd = net.state_dict()
    print("  state_dict 的键:", list(sd.keys()))


def part4_data():
    section("4. Dataset 与 DataLoader")
    t = time.perf_counter()
    ds = GlyphDataset(seed=0, n_per_class=20, size=32)
    print(f"  渲染 {len(ds)} 张字符图用时 {time.perf_counter() - t:.2f} 秒")
    x, y = ds[5]
    print(f"  ds[5]：图像 {tuple(x.shape)} {x.dtype}，标签 {y.item()}（字符 {ds.chars[y]!r}）")
    loader = DataLoader(ds, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(0))
    xb, yb = next(iter(loader))
    print(f"  一批：图像 {tuple(xb.shape)}，标签 {tuple(yb.shape)}，前 10 个标签 {yb[:10].tolist()}")
    print(f"  一轮有 {len(loader)} 批（{len(ds)} / 64 向上取整）")


def bench(fn, n):
    fn()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n


def part5_gpu():
    section(f"5. CPU 与 {DEV.type.upper()}：矩阵乘法的速度")
    if DEV.type == "cpu":
        print("  没有可用的 GPU，跳过。")
        return
    print(f"  GPU：{torch.cuda.get_device_name(0) if DEV.type == 'cuda' else DEV}")
    for n in (256, 1024, 2048):
        a, b = torch.randn(n, n), torch.randn(n, n)
        ag, bg = a.to(DEV), b.to(DEV)
        tc = bench(lambda: a @ b, 5 if n == 2048 else 20)
        tg = bench(lambda: ag @ bg, 20)
        flops = 2 * n ** 3
        print(f"  {n:5d}×{n:<5d} CPU {tc * 1000:8.2f} ms ({flops / tc / 1e12:5.2f} TFLOPS)   "
              f"GPU {tg * 1000:7.3f} ms ({flops / tg / 1e12:6.2f} TFLOPS)   快 {tc / tg:5.0f} 倍")
    a = torch.randn(2048, 2048)
    t_copy = bench(lambda: a.to(DEV), 10)
    print(f"  把 16 MB 的张量从内存拷到显存：{t_copy * 1000:.2f} ms（{16 / 1024 / t_copy:.1f} GB/s）")
    ag = torch.randn(2048, 2048, device=DEV)
    t0 = time.perf_counter(); _ = ag @ ag; t_nosync = time.perf_counter() - t0
    torch.cuda.synchronize()
    t0 = time.perf_counter(); _ = ag @ ag; torch.cuda.synchronize(); t_sync = time.perf_counter() - t0
    print(f"  不同步就计时：{t_nosync * 1000:.3f} ms（只测到了\"提交任务\"）；同步后计时：{t_sync * 1000:.3f} ms")


def part6_train():
    section(f"6. 用 PyTorch 重做第 8 章的任务（设备：{DEV.type}）")
    tr = GlyphDataset(seed=0, chars="0123456789", n_per_class=30, size=8, align=True)
    te = GlyphDataset(seed=1, chars="0123456789", n_per_class=30, size=8, align=True)
    torch.manual_seed(0)
    net = nn.Sequential(nn.Flatten(), nn.Linear(64, 16), nn.Tanh(), nn.Linear(16, 10)).to(DEV)
    opt = torch.optim.SGD(net.parameters(), lr=0.5)
    loader = DataLoader(tr, batch_size=25, shuffle=True, generator=torch.Generator().manual_seed(0))
    t0 = time.perf_counter()
    for epoch in range(6):
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad()
            loss = F.cross_entropy(net(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
        with torch.no_grad():
            acc = (net(te.x.to(DEV)).argmax(1).cpu() == te.y).float().mean().item()
        print(f"    第 {epoch + 1} 轮  训练损失 {total / len(tr):.4f}  测试准确率 {acc:.1%}")
    dt = time.perf_counter() - t0
    print(f"  6 轮用时 {dt:.2f} 秒，平均每个样本 {dt / (6 * len(tr)) * 1e6:.0f} µs（第 8 章的 Value 版：约 45000 µs）")


if __name__ == "__main__":
    print(f"PyTorch {torch.__version__}，使用设备 {DEV}")
    part1_tensors()
    part2_autograd()
    part3_module()
    part4_data()
    part5_gpu()
    part6_train()
    print("\n第 9 章的演示全部完成。")
