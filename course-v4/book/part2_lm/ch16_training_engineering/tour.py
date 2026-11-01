"""第 16 章：训练工程。离线、不下载；默认 CPU 两线程，仅短数值与计时实验。

python -m book.part2_lm.ch16_training_engineering.tour [部分号 1–11 ...]
"""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import math
import os
from statistics import median
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "2"

import torch

from miniocr.device import pick_device, synchronize
from miniocr.kvcache import local_model_configs
from miniocr.models.gpt import GPT, GPTConfig, RMSNorm
from miniocr.train.lm import accumulate_grads
from miniocr.train.memory import checkpoint_blocks, static_memory, matmul_params, train_flops
from miniocr.train.precision import format_table, spacing, quantize_gradients, SimpleGradScaler
from miniocr.train.resources import (
    measure_static_bytes, activation_bytes_estimate, measure_activations,
    qwen_like_params, training_flops, count_flops, resource_report,
)
from miniocr.train.scaling import ladder_config, parameter_counts

_DEVICE = "cpu"


def emit(*, 来源="本项目观察", 设备=None, **values):
    """每行注明来源；观察行标设备/线程，float 保留有效数字。"""
    if 来源.startswith("本项目观察"):
        values = dict(设备=设备 or _DEVICE, 线程=torch.get_num_threads(), **values)
    values = dict(来源=来源, **values)
    print("；".join(f"{key}={value:.7g}" if isinstance(value, float) else f"{key}={value}"
                   for key, value in values.items()), flush=True)


class Experiment:
    def __init__(self, args):
        self.args = args

    def model(self, size="medium", vocab=512, context=256):
        torch.manual_seed(self.args.seed)
        cfg = replace(ladder_config(size, vocab, context), attn_impl="manual")
        return GPT(cfg).to(self.args.device)


def _gradient(model):
    return torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None])


def _relative(actual, expected):
    return ((actual-expected).norm()/expected.norm().clamp_min(1e-30)).item()


def part1(exp):
    for row in format_table():
        fi = torch.finfo(row["dtype"])
        emit(来源="数学结论", 格式=row["name"], 符号位=row["sign_bits"],
             指数位=row["exponent_bits"], 尾数位=row["mantissa_bits"])
        for key in ("max", "tiny", "eps", "subnormal"):
            # finfo 没有 subnormal 属性；用其 tiny*eps 推导，并另用 nextafter 对拍。
            actual = fi.tiny*fi.eps if key == "subnormal" else getattr(fi, key)
            emit(来源="数学结论（torch.finfo 对拍）", 格式=row["name"], 项=key,
                 公式值=row[key], finfo值=actual, 一致=row[key] == actual,
                 finfo口径="tiny*eps（派生）" if key == "subnormal" else key)
        sub = torch.nextafter(torch.tensor(0., dtype=row["dtype"]), torch.tensor(1., dtype=row["dtype"]))
        emit(设备="cpu", 格式=row["name"], nextafter零到正一=sub.item(), 最小非正规数一致=sub.item() == row["subnormal"])
    values = torch.tensor([70000., 1e-8, 1+1e-3, 1+1e-2, .1])
    for dt in (torch.float16, torch.bfloat16):
        emit(设备="cpu", dtype=str(dt), 原值=values.tolist(), 舍入后=values.to(dt).float().tolist())
        for x in (1, 256, 1000, 65504):
            emit(来源="数学结论", dtype=str(dt), x=x, binade间隔=spacing(x, dt))
    emit(设备="cpu", bf16的256加1=(torch.tensor(256., dtype=torch.bfloat16)+1).item())


def part2(exp):
    x = torch.full((1, 64), 300., dtype=torch.float16)
    square = x.square().mean(-1, keepdim=True)
    emit(设备="cpu", fp16均方=square.item(), 朴素RMSNorm=(x*torch.rsqrt(square+1e-6))[0, :3].tolist(),
         项目RMSNorm=RMSNorm(64)(x)[0, :3].tolist(), 说明="项目 RMSNorm 内部转 fp32")
    logits = torch.tensor([12., 11., 0.], dtype=torch.float16)
    naive = logits.exp()
    stable = (logits-logits.max()).exp()
    emit(设备="cpu", exp12_fp16=naive[0].item(), exp12_fp32=logits.float().exp()[0].item(),
         朴素softmax=(naive/naive.sum()).tolist(), 减最大值=(stable/stable.sum()).tolist(),
         fp32_softmax=logits.float().softmax(0).tolist())
    for dt in (torch.float16, torch.float32):
        acc, increment = torch.tensor(0., dtype=dt), torch.tensor(.01, dtype=dt)
        for _ in range(10000):
            acc.add_(increment)  # 真正逐步在该 dtype 中舍入，不使用更高精度的 cumsum。
        emit(设备="cpu", dtype=str(dt), 逐次累加次数=10000, 每次输入=increment.item(), 最终和=acc.item())


def part3(exp):
    torch.manual_seed(exp.args.seed)
    grad = torch.randn(10000)*1e-7
    for scale in (1, 2**10, 2**16, 2**24):
        stats = quantize_gradients(grad, scale)
        emit(设备="cpu", 场景="随机梯度标准差1e-7", S=scale, **{k:v for k,v in stats.items() if k != "restored"})
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
    x = torch.randn(4, 8)
    # 同一个前向图，分别反向；模拟梯度写入 fp16，不依赖 CPU autocast(float16)。
    loss = model(x).square().mean()*1e-8
    params = tuple(model.parameters())
    reference = torch.cat([g.flatten() for g in torch.autograd.grad(loss, params, retain_graph=True)])
    for scale in (1, 2**16):
        scaled = torch.cat([g.flatten() for g in torch.autograd.grad(loss*scale, params, retain_graph=True)])
        restored = scaled.half().float()/scale
        emit(设备="cpu", 场景="MLP一次前向，复用图反向对拍", loss=loss.item(), S=scale,
             零比例=(restored == 0).float().mean().item(), 相对fp32误差=_relative(restored, reference))


def _scaler_sequence(scaler):
    p = torch.nn.Parameter(torch.tensor([1.]))
    opt = torch.optim.SGD([p], lr=.1)
    rows = []
    for index in range(1, 9):
        opt.zero_grad(set_to_none=True)
        scale, before = scaler.get_scale(), p.detach().clone()
        scaler.scale(p.sum()).backward()
        if index in (2, 3):
            p.grad.fill_(float("inf"))
        scaler.unscale_(opt)
        scaler.step(opt)
        changed = not torch.equal(p, before)
        scaler.update()
        rows.append(dict(step=index, scale=scale, skipped=not changed, parameter_updated=changed,
                         next_scale=scaler.get_scale(), parameter=p.item()))
    return rows


def part4(exp):
    rows = _scaler_sequence(SimpleGradScaler(growth_interval=2))
    for row in rows:
        emit(设备="cpu", 实现="SimpleGradScaler", **row)
    try:
        reference = torch.amp.GradScaler("cpu", init_scale=2.**16, growth_interval=2)
    except (TypeError, RuntimeError, AttributeError) as exc:
        emit(来源="待验证", 对拍="跳过对拍：本机不支持", 原因=str(exc))
    else:
        other = _scaler_sequence(reference)
        for a, b in zip(rows, other):
            emit(设备="cpu", 实现="torch.amp.GradScaler", **b, 与教学版一致=a == b)
    for correct in (True, False):
        p = torch.nn.Parameter(torch.ones(2))
        opt = torch.optim.SGD([p], lr=.1)
        scaler = SimpleGradScaler(init_scale=1024)
        scaler.scale((p*torch.tensor([3., 4.])).sum()).backward()
        if correct:
            scaler.unscale_(opt)
        clip_input = torch.nn.utils.clip_grad_norm_([p], 1.).item()
        if not correct:
            scaler.unscale_(opt)
        effective = p.grad.norm().item()
        before = p.detach().clone()
        scaler.step(opt)
        scaler.update()
        emit(设备="cpu", 顺序="unscale再clip" if correct else "错误：clip再unscale",
             clip输入范数=clip_input, 实际更新梯度范数=effective, 更新量范数=(p-before).norm().item())


def part5(exp):
    for dt in (torch.bfloat16, torch.float16, torch.float32):
        p = torch.tensor(1., dtype=dt)
        for _ in range(100):
            p.sub_(1e-4)
        emit(设备="cpu", dtype=str(dt), 初始值=1., 每步更新量=1e-4, 步数=100, 最终值=p.item())
    emit(来源="数学结论", 精确实数结果=1-100*1e-4, 说明="fp32主权重累积小更新；计算副本可按需转低精度")


def _static_example(mixed=False, device="cpu"):
    cfg = replace(ladder_config("tiny", 512, 64), attn_impl="manual")
    master = GPT(cfg).to(device)
    model = deepcopy(master).to(torch.bfloat16) if mixed else master
    opt = torch.optim.AdamW(master.parameters(), lr=1e-3)
    x = torch.randint(cfg.vocab_size, (2, 16), device=device)
    model(x, targets=x.clone())[1].backward()
    if mixed:
        for p, low in zip(master.parameters(), model.parameters()):
            p.grad = low.grad.float()
    opt.step()
    if mixed:
        opt.zero_grad(set_to_none=True)  # 只释放主梯度；计算副本的 bf16 梯度仍保留。
    measured = measure_static_bytes(model, opt, master if mixed else None)
    n = master.n_params()
    estimated = static_memory(n, n, "bf16" if mixed else "fp32", master_fp32=mixed)
    return estimated, measured


def part6(exp):
    for dt, master in (("fp32", False), ("bf16", True)):
        emit(来源="数学结论", 口径=dt, 每参数字节=static_memory(1, 1, dt, master_fp32=master))
    for mixed in (False, True):
        # bf16 静态账演示固定 CPU，避免把 MPS 的 dtype 支持当作前提。
        device = "cpu" if mixed else exp.args.device
        est, actual = _static_example(mixed, device)
        emit(设备=device, 混合精度=mixed, 估算=est, 实测=actual,
             差值=actual["total"]-est["total"], 说明="差值应为step标量；不含buffer/分配器；mixed主梯度step后释放")
    configs = []
    for size in ("medium", "large"):
        cfg = ladder_config(size, 512, 64)
        configs.append((size, parameter_counts(cfg)["total"], "项目ladder解析参数量，V=512"))
    cfg = local_model_configs()[0]
    qwen = qwen_like_params(cfg)
    emit(来源="数学结论（配置来源与待验证项见字段）", 模型=cfg.name, **qwen)
    configs.append((cfg.name, qwen["total"], qwen["source"]+"；"+qwen["tie_source"]+"；"+qwen["note"]))
    for name, n, source in configs:
        for label, dt, master, fraction in (("全参fp32", "fp32", False, 1),
                                           ("全参混合", "bf16", True, 1),
                                           ("仅训练1%混合", "bf16", True, .01)):
            nt = round(n*fraction)
            emit(来源="数学结论", 模型=name, 参数来源=source, 参数量=n, 可训练参数量=nt,
                 口径=label, 静态字节=static_memory(n, nt, dt, master_fp32=master))


def part7(exp):
    model = exp.model()
    emit(配置=asdict(model.cfg), 口径="fp32/manual；独立idx与targets；saved storage去重，不是峰值")
    base, observations = None, []
    for B, T in ((4, 64), (8, 64), (4, 128), (4, 256)):
        estimate = activation_bytes_estimate(model.cfg, B, T)
        actual = measure_activations(model, B, T)
        base = base or actual
        emit(B=B, T=T, 估算字节=estimate["total"], 实测字节=actual, 相对基准=actual/base,
             相对误差=abs(actual-estimate["total"])/actual)
        if B == 4:
            observations.append((T, actual))
        if (B, T) == (4, 64):
            for row in estimate["breakdown"]:
                emit(来源="数学结论", **row)
            emit(来源="数学结论", 未计入saved的返回logits字节=estimate["logits_bytes"],
                 说明="scores未保存，softmax权重保存；B翻倍时mask不翻倍，另有固定标量")
    # 去掉最终norm、CE、索引及标量，再按层拟合块的 aT+bT²。
    cfg, B = model.cfg, 4
    tail_linear = 4*B*(3*cfg.d_model+1+cfg.vocab_size)+16*B
    A = torch.tensor([[T, T*T] for T, _ in observations], dtype=torch.float64)
    y = torch.tensor([(n-tail_linear*T-4)/cfg.n_layer for T, n in observations], dtype=torch.float64)
    a, b = torch.linalg.lstsq(A, y).solution.tolist()
    formula = activation_bytes_estimate(cfg, B, 64)
    emit(拟合口径="扣除非块项后，每层bytes(T)=aT+bT²，三组T最小二乘", a拟合=a, b拟合=b,
         a公式=formula["block_linear_coefficient"], b公式=formula["block_quadratic_coefficient"])
    sdpa = GPT(replace(cfg, attn_impl="sdpa")).to(exp.args.device)
    emit(attention="sdpa", B=4, T=128, 实测字节=measure_activations(sdpa, 4, 128),
         公式="待验证：SDPA内部保存随设备/后端变化，本章不作精确预测")


def _timed_backward(model, x, y, repeats=3):
    times = []
    for run in range(repeats+1):
        model.zero_grad(set_to_none=True)
        synchronize(x.device)
        start = time.perf_counter()
        model(x, targets=y)[1].backward()
        synchronize(x.device)
        if run:
            times.append(time.perf_counter()-start)
    return median(times), _gradient(model).clone()


def part8(exp):
    model = exp.model()
    checked = deepcopy(model)
    checkpoint_blocks(checked.blocks)
    x = torch.randint(model.cfg.vocab_size, (4, 128), device=exp.args.device)
    y = x.clone()
    results = []
    for name, m in (("无检查点", model), ("逐块检查点", checked)):
        saved = measure_activations(m, 4, 128)
        sec, grad = _timed_backward(m, x, y)
        flops = count_flops(m, x, y)
        results.append((saved, sec, grad, flops))
        emit(路径=name, B=4, T=128, 激活字节=saved, 前后向中位秒=sec,
             前后向FLOPs=flops, 预热=1, 计时重复=3, 口径="计时不装hooks/计数器，不含optimizer")
    a, b = results
    emit(梯度最大绝对差=(a[2]-b[2]).abs().max().item(), 激活比=b[0]/a[0],
         时间比=b[1]/a[1], FLOPs比=b[3]/a[3], 额外重算FLOPs=b[3]-a[3],
         说明="仅块重算，输出层不重算；非重入checkpoint可提前停止，比例不必正好4/3")


def _accumulation_experiment(seed=0, device="cpu"):
    torch.manual_seed(seed)
    model = GPT(GPTConfig(vocab_size=50, d_model=32, n_head=2, n_layer=2,
                          block_size=16, attn_impl="manual")).to(device)
    x = torch.randint(1, 50, (4, 16), device=device)
    y = torch.randint(1, 50, (4, 16), device=device)
    for i, length in enumerate((2, 16, 16, 10)):
        y[i, length:] = -100
    model(x, targets=y)[1].backward()
    full = _gradient(model).clone()
    outputs = {}
    for label, groups in (("A", [[0, 1], [2, 3]]), ("B", [[0, 2], [1, 3]])):
        batches = [(x[g], y[g]) for g in groups]
        for mode in ("token", "micro_mean", "sample", "pretrain"):
            model.zero_grad(set_to_none=True)
            if mode == "pretrain":
                # 将补齐位置换成 BOS=0，复现 pretrain 的非 BOS count/valid。
                packed = [(xx, yy.masked_fill(yy == -100, 0)) for xx, yy in batches]
                counts = [int((yy != 0).sum()) for _, yy in packed]
                valid = sum(counts)
                for (xx, yy), count in zip(packed, counts):
                    if count:
                        loss = model(xx, yy, loss_mask=yy != 0)[1]*(count/valid)
                        loss.backward()
            else:
                accumulate_grads(model, batches, mode)
            outputs[label, mode] = _gradient(model).clone()
    return full, outputs


def part9(exp):
    full, outputs = _accumulation_experiment(exp.args.seed, exp.args.device)
    emit(来源="数学结论", 有效长度=[2, 16, 16, 10], 分组A=[[0, 1], [2, 3]], 分组B=[[0, 2], [1, 3]],
         token权重="所有token:1/44", micro_mean权重="两组各:1/18×1/2；1/26×1/2",
         sample权重="逐样本:1/2×1/4；1/16×1/4；1/16×1/4；1/10×1/4",
         pretrain权重="微批mean×count/valid = sum/valid；BOS不计入count")
    for (group, mode), grad in outputs.items():
        emit(分组=group, 方式=mode, 与整批相对差=_relative(grad, full))
    for mode in ("token", "micro_mean", "sample", "pretrain"):
        emit(方式=mode, 换分组后相对差=_relative(outputs["B", mode], outputs["A", mode]))


def _throughput(model, B=4, T=64, repeats=5):
    device = next(model.parameters()).device
    x = torch.randint(model.cfg.vocab_size, (B, T), device=device)
    y = x.clone()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    times = []
    for run in range(2+repeats):
        synchronize(device)
        start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        model(x, targets=y)[1].backward()
        opt.step()
        synchronize(device)
        if run >= 2:
            times.append(time.perf_counter()-start)
    return B*T/median(times), median(times), opt


def part10(exp):
    model = exp.model()
    B, T = 4, 64
    x = torch.randint(model.cfg.vocab_size, (B, T), device=exp.args.device)
    y = x.clone()
    forward = count_flops(model, x, y, backward=False)
    measured = count_flops(model, x, y)
    estimate = training_flops(model.cfg, B, T)
    emit(来源="数学结论", 配置=asdict(model.cfg), **estimate,
         注意力口径="manual完整QKᵀ+AV，前向4LBT²d，训练12LBT²d；Kaplan因果半矩阵减半")
    emit(前向实测FLOPs=forward, 前向与2ND比=forward/estimate["forward_2nd"],
         前后向实测FLOPs=measured, 与6ND比=measured/estimate["six_nd"],
         与含完整注意力估算比=measured/estimate["with_attention"],
         说明="FlopCounterMode主要计矩阵乘；matmul_params还含少量norm参数")
    speed, sec, _ = _throughput(model, B, T)
    achieved = train_flops(matmul_params(model), speed)
    counted = measured/(B*T)*speed
    emit(完整训练步中位秒=sec, tokens每秒=speed, 实现FLOPs每秒_6N=achieved,
         实现FLOPs每秒_计数器=counted, 预热步数=2, 计时步数=5,
         口径="fp32；forward+backward+AdamW+zero_grad；合成输入，不含数据加载")
    if exp.args.peak_tflops is None:
        emit(来源="待验证", MFU="待验证：请用 --peak-tflops 填写你设备的峰值")
    else:
        emit(来源="本项目观察（峰值由学习者填写）", 峰值TFLOPs=exp.args.peak_tflops,
             MFU_6N=achieved/(exp.args.peak_tflops*1e12), MFU_计数器=counted/(exp.args.peak_tflops*1e12))
    emit(来源="数学结论", N=matmul_params(model), D=B*T*5,
         时间公式="6ND/(峰值FLOP/s×MFU)", 峰值="学习者填写", MFU="学习者填写")


def part11(exp):
    model = exp.model()
    B, T = 4, 64
    report = resource_report(model.cfg, B, T)
    emit(来源="数学结论", 阶段="先估算", 静态字节=report["static"],
         激活字节=report["activation"]["total"], 合计字节=report["total_estimate"],
         每更新步tokens=report["tokens_per_step"], 每更新步FLOPs=report["flops"], 口径=report["scope"])
    speed, sec, opt = _throughput(model, B, T)
    measured = measure_static_bytes(model, opt)
    activation = measure_activations(model, B, T)
    pairs = [("静态（含step小张量）", report["static"]["total"], measured["total"]),
             ("激活保存量", report["activation"]["total"], activation),
             ("静态+保存量（非峰值）", report["total_estimate"], measured["total"]+activation)]
    for name, expected, actual in pairs:
        emit(项目=name, 估算=expected, 实测=actual, 相对差=(actual-expected)/expected)
    emit(step标量字节=measured["step"], tokens每秒=speed, 完整步中位秒=sec)
    budget = B*T*100
    report = resource_report(model.cfg, B, T, peak_tflops=exp.args.peak_tflops,
                             tokens_per_sec=speed, token_budget=budget)
    emit(来源="本项目观察（预算为教学设定，峰值由学习者填写）", token预算=budget,
         按实测吞吐预计秒=report["estimated_seconds"],
         MFU=report["mfu"] if report["mfu"] is not None else "待验证：请用 --peak-tflops 填写你设备的峰值")
    emit(来源="待验证", 表="Mac / 5080 自填；峰值测量口径见lab，不把CPU结果外推")
    for name, expected in (("静态显存", report["static"]["total"]),
                           ("激活保存量", report["activation"]["total"]),
                           ("峰值显存：CUDA max_memory_allocated / MPS driver_allocated_memory采样（非峰值API）", "待验证"),
                           ("tokens/s", "待验证"), ("MFU", "待验证")):
        emit(来源="数学结论" if isinstance(expected, int) else "待验证", 项目=name, 估算=expected,
             **{"实测（你填）": "", "相对差（你填）": ""})


PARTS = {1: (part1, "浮点格式（16.1）"),
         2: (part2, "前向溢出与累加（16.2）"),
         3: (part3, "梯度下溢与loss scaling（16.3）"),
         4: (part4, "GradScaler与裁剪顺序（16.4）"),
         5: (part5, "主权重为什么要fp32（16.5）"),
         6: (part6, "静态显存账（16.6）"),
         7: (part7, "激活显存：估算、实测与拟合（16.7）"),
         8: (part8, "梯度检查点（16.8）"),
         9: (part9, "变长样本的梯度累积（16.9）"),
         10: (part10, "6ND、实测FLOPs与MFU（16.10）"),
         11: (part11, "资源核算报告（16.11）")}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="*", type=int)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--peak-tflops", type=float)
    args = parser.parse_args(argv)
    if any(part not in PARTS for part in args.parts):
        parser.error("部分号须为1–11")
    if args.seed < 0:
        parser.error("seed 须非负")
    if args.peak_tflops is not None and (not math.isfinite(args.peak_tflops) or args.peak_tflops <= 0):
        parser.error("peak-tflops 须为有限正数")
    try:
        pick_device(args.device)
    except RuntimeError as exc:
        parser.error(str(exc))
    return args


def main(argv=None):
    global _DEVICE
    args = parse_args(argv)
    _DEVICE = args.device
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    emit(章节=16, 种子=args.seed, torch版本=torch.__version__,
         默认行为="CPU；离线；不下载；不读语料；不写产物；仅几步计时，无训练任务",
         设备范围="1–5及6的bf16静态演示固定CPU；其余遵循--device")
    start = time.perf_counter()
    exp = Experiment(args)
    for part in args.parts or PARTS:
        torch.manual_seed(args.seed)  # 部分单独跑与整套跑使用同一随机起点。
        fn, title = PARTS[part]
        emit(部分=part, 内容=title)
        synchronize(args.device)
        t0 = time.perf_counter()
        fn(exp)
        synchronize(args.device)
        emit(部分=part, 用时秒=time.perf_counter()-t0)
    emit(总用时秒=time.perf_counter()-start)


if __name__ == "__main__":
    main()
