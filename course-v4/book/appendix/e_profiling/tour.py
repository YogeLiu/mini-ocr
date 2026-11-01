"""附录 E：性能分析。用 torch.profiler 在 CPU 上分析一段小训练循环，抓一个故意放的慢点。

python -m book.appendix.e_profiling.tour [部分号 1–5 ...]

只断言故意放的慢点排第一；其他操作的名次随机器和负载变化，只打印。
"""
import argparse
from dataclasses import replace
import os
import time

os.environ["HF_HUB_OFFLINE"] = "1"
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "2"

import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule

from book.appendix import emit
from miniocr.kvcache import local_model_configs, step_cost
from miniocr.models.gpt import GPT
from miniocr.train.scaling import ladder_config

SPANS = ("span/forward", "span/backward", "span/optimizer", "span/weight_norm")


def slow_weight_norm(model):
    """故意放的慢点：逐元素 Python 循环算嵌入权重的平方和。每步都调一次，像一行“顺手记日志”。"""
    weight = model.tok.weight.detach()
    total = 0.
    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            total += weight[i, j].item()**2
    return total


def fast_weight_norm(model):
    """向量化的写法，同一个数。用 float64 算，避免和 Python float 的累加差出舍入。"""
    return model.tok.weight.detach().double().square().sum().item()


def make_setup(seed=0, V=128, B=4, T=32):
    torch.manual_seed(seed)
    model = GPT(replace(ladder_config("tiny", V, T), attn_impl="manual"))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(V, (B, T), generator=g)
    return model, opt, x, torch.randint(V, (B, T), generator=g)


def train_step(model, opt, x, y, norm_fn):
    opt.zero_grad(set_to_none=True)
    with record_function("span/forward"):
        loss = model(x, targets=y)[1]
    with record_function("span/backward"):
        loss.backward()
    with record_function("span/optimizer"):
        opt.step()
    with record_function("span/weight_norm"):
        return loss.item(), norm_fn(model)


def profile_loop(norm_fn, seed=0, wait=1, warmup=1, active=3):
    """跑 wait+warmup+active 步，只有 active 步进统计。不写文件。"""
    setup = make_setup(seed)
    with profile(activities=[ProfilerActivity.CPU], schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=1)) as prof:
        for _ in range(wait+warmup+active):
            train_step(*setup, norm_fn)
            prof.step()
    return prof


def span_rows(prof):
    """按 CPU 总时间从大到小排四个 span。返回 [(名字, 总微秒, 次数)]。"""
    rows = [(e.key, e.cpu_time_total, e.count) for e in prof.key_averages() if e.key in SPANS]
    return sorted(rows, key=lambda row: -row[1])


def op_rows(prof, top=5):
    """按自身 CPU 时间排 aten 算子。"""
    rows = [(e.key, e.self_cpu_time_total, e.count) for e in prof.key_averages() if e.key.startswith("aten::")]
    return sorted(rows, key=lambda row: -row[1])[:top]


def op_count(prof, name):
    return sum(e.count for e in prof.key_averages() if e.key == name)


def intensity(flops, bytes_):
    return flops/bytes_


def daxpy_intensity():
    """y += a·x：每元素 1 次乘加（2 FLOPs），读 x、读 y、写 y，fp32 共 12 字节。"""
    return intensity(2, 12)


def matmul_intensity(n):
    """n×n 矩阵乘：2n³ FLOPs；读 A、读 B、写 C，fp32 共 12n² 字节。假设每个矩阵只进出一次。"""
    return intensity(2*n**3, 12*n*n)


def measure_rate(fn, flops, repeats=5):
    """返回 FLOP/s（中位数）。只打印，不断言。"""
    times = []
    for _ in range(repeats+1):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter()-start)
    return flops/sorted(times[1:])[repeats//2]


def part1(args):
    prof = profile_loop(slow_weight_norm, args.seed)
    emit(设备="cpu", 配置="tiny V=128 B=4 T=32", 计划="wait=1 warmup=1 active=3", 嵌入元素数=128*64)
    for rank, (name, total, count) in enumerate(span_rows(prof), 1):
        emit(设备="cpu", 带慢点=True, 名次=rank, span=name, 次数=count, 占四段比例=total/sum(r[1] for r in span_rows(prof)))
    for rank, (name, total, count) in enumerate(op_rows(prof), 1):
        emit(设备="cpu", 带慢点=True, 算子名次=rank, 算子=name, 次数=count)


def part2(args):
    prof = profile_loop(fast_weight_norm, args.seed)
    for rank, (name, total, count) in enumerate(span_rows(prof), 1):
        emit(设备="cpu", 带慢点=False, 名次=rank, span=name, 次数=count, 占四段比例=total/sum(r[1] for r in span_rows(prof)))
    for rank, (name, total, count) in enumerate(op_rows(prof), 1):
        emit(设备="cpu", 带慢点=False, 算子名次=rank, 算子=name, 次数=count)


def part3(args):
    model = make_setup(args.seed)[0]
    emit(设备="cpu", 慢写法=slow_weight_norm(model), 快写法=fast_weight_norm(model),
         差=abs(slow_weight_norm(model)-fast_weight_norm(model)), 说明="同一个数")
    for fn in (slow_weight_norm, fast_weight_norm):
        setup = make_setup(args.seed)
        for _ in range(2):
            train_step(*setup, fn)
        start = time.perf_counter()
        for _ in range(3):
            train_step(*setup, fn)
        emit(设备="cpu", 写法=fn.__name__, 不开profiler三步总秒=time.perf_counter()-start, 说明="用时随负载变化")


def part4(_):
    emit(来源="数学结论", 操作="y+=a·x", 每元素FLOPs=2, 每元素字节=12, 算术强度=daxpy_intensity())
    for n in (16, 256, 512, 1024):
        emit(来源="数学结论", 操作="矩阵乘", n=n, FLOPs=2*n**3, 字节=12*n*n, 算术强度=matmul_intensity(n))
    cfg = local_model_configs()[0]
    for name, T, past, B in (("预填充", 512, 0, 1), ("解码", 1, 512, 1), ("解码", 1, 512, 32)):
        emit(来源="数学结论（第15章粗算）", 模型=cfg.name, 阶段=name, T_new=T, T_past=past, B=B, 算术强度=step_cost(cfg, T, past, B)["intensity"])


def part5(_):
    small, big = 2**12, 2**25  # fp32 下 x+y 共 32 KiB 与 256 MiB
    n = 512
    A, B = torch.randn(n, n), torch.randn(n, n)
    x1, y1, x2, y2 = torch.randn(small), torch.randn(small), torch.randn(big), torch.randn(big)
    for threads in (1, 2, 4):
        torch.set_num_threads(threads)
        emit(设备="cpu", 线程=threads,
             小工作集daxpy_GFLOPs=measure_rate(lambda: [y1.add_(x1, alpha=.001) for _ in range(2000)], 2*small*2000)/1e9,
             大工作集daxpy_GFLOPs=measure_rate(lambda: y2.add_(x2, alpha=.001), 2*big)/1e9,
             矩阵乘GFLOPs=measure_rate(lambda: A @ B, 2*n**3)/1e9)
    torch.set_num_threads(2)
    emit(来源="待验证", 说明="这几个数只说明本机当时的趋势，换机器要重测；机器的峰值算力比带宽由你填写")


PARTS = {1: (part1, "抓慢点（E.4.1）"),
         2: (part2, "修掉以后再看（E.4.1）"),
         3: (part3, "慢写法和快写法结果相同（E.4.1）"),
         4: (part4, "算术强度（E.4.2）"),
         5: (part5, "工作集和线程实验（E.4.3）")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="*", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if any(part not in PARTS for part in args.parts):
        parser.error("部分号须为1–5")
    torch.set_num_threads(2)
    emit(来源="环境", 附录="E", 种子=args.seed, torch版本=torch.__version__, 默认行为="CPU；不写trace文件；用时不进断言")
    for part in args.parts or PARTS:
        torch.manual_seed(args.seed)
        fn, title = PARTS[part]
        emit(来源="环境", 部分=part, 内容=title)
        fn(args)


if __name__ == "__main__":
    main()
