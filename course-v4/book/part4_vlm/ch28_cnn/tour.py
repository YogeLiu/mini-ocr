"""第 28 章：1 卷积/梯度，2 几何，3 下采样，4 BN/LN，5 CNN/MLP。

在 course-v4 下 python -m book.part4_vlm.ch28_cnn.tour 1 2 --device cpu。
全部产物仅写 --out，远程运行时指定本章运行目录下的新子目录，见 lab.md。
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from miniocr.data.formula_images import load_formula_images
from miniocr.data.symbols import make_dataset, group_split, count_shared_groups, render_symbol, augment
from miniocr.device import pick_device, synchronize
from miniocr.nn.cnn import (conv_forward, conv_backward, numerical_gradient, output_size,
                            receptive_field, ManualBatchNorm2d, SymbolClassifier)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def csv_rows(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def part1(args):
    """CPU float64：全部坐标中心差分，并与优化库前后向独立对拍。"""
    rng = np.random.default_rng(args.seed)
    rows = []
    for stride, padding in ((1, 0), (1, 1), (2, 1)):
        x = rng.normal(size=(2, 2, 4, 5))
        w = rng.normal(size=(3, 2, 2, 3))
        b = rng.normal(size=3)
        y, cache = conv_forward(x, w, b, stride, padding)
        dy = rng.normal(size=y.shape)
        gradients = conv_backward(dy, cache)
        values = [x, w, b]
        tx, tw, tb = [torch.tensor(a, requires_grad=True) for a in values]
        ref = F.conv2d(tx, tw, tb, stride=stride, padding=padding)
        (ref * torch.tensor(dy)).sum().backward()
        for k, name in enumerate(("x", "weight", "bias")):
            def objective(v):
                inputs = values.copy()
                inputs[k] = v
                return (conv_forward(*inputs, stride, padding)[0] * dy).sum()
            num = numerical_gradient(objective, values[k])
            analytic = gradients[k]
            absolute = float(np.max(np.abs(num - analytic)))
            relative = float(np.max(np.abs(num - analytic) / np.maximum(1e-8, np.abs(num) + np.abs(analytic))))
            library_error = float(np.max(np.abs(analytic - (tx, tw, tb)[k].grad.numpy())))
            rows.append(dict(stride=stride, padding=padding, gradient=name,
                             coordinates=num.size, max_abs=absolute, max_relative=relative,
                             torch_max_abs=library_error,
                             forward_max_abs=float(np.max(np.abs(y - ref.detach().numpy())))))
            assert absolute < 1e-7 and library_error < 1e-10
    csv_rows(args.out / "gradients.csv", rows)
    return dict(device="cpu", dtype="float64", checks=rows)


def part2(args):
    layers = [(3, 1, 1, 1), (2, 2, 0, 1), (3, 1, 1, 1), (2, 2, 0, 1)]
    size, rows = 32, receptive_field(layers)
    for layer, row in zip(layers, rows):
        size = output_size(size, *layer)
        row["output_size"] = size
    csv_rows(args.out / "geometry.csv", rows)
    return dict(source="数学结论", input_size=32, layers=layers, results=rows)


def downsample(ink, levels, device="cpu"):
    """连续 2×2 平均池化，再最近邻放回原尺寸；奇数边补白纸至 2**levels 倍数。

    同一原图坐标测量损伤；这不是 OCR 准确率，也不是逆变换。
    """
    x = torch.as_tensor(ink, dtype=torch.float32, device=device)[None, None]
    h, w = x.shape[-2:]
    unit = 2 ** levels
    x = F.pad(x, (0, (-w) % unit, 0, (-h) % unit))
    for _ in range(levels):
        x = F.avg_pool2d(x, 2)
    x = F.interpolate(x, scale_factor=unit, mode="nearest")
    return x[0, 0, :h, :w].cpu().numpy()


def damage(ink, restored):
    """原图墨迹>0.5 像素的恢复对比度及阈值召回；MSE 另计整图，避免只看白纸。"""
    mask = ink > .5
    return dict(mse=float(np.mean((ink-restored)**2)),
                foreground_pixels=int(mask.sum()),
                foreground_contrast=float(restored[mask].mean()) if mask.any() else None,
                foreground_recall=float((restored[mask] > .5).mean()) if mask.any() else None)


def part3(args):
    print("读取完整语料以确定 latex-v1，再选 val 图片……", flush=True)
    examples = load_formula_images(args.n)
    if not examples and args.require_data:
        raise RuntimeError("真实 PNG 缓存缺失；--require-data 禁止跳过")
    rows, panels = [], []
    if examples:
        for e in examples:
            ink = e["ink"]
            for level in range(5):
                restored = downsample(ink, level, args.device)
                rows.append(dict(id=e["id"], key_sha256=e["key_sha256"],
                                 has_script=("^" in e["formula"] or "_" in e["formula"]),
                                 height=ink.shape[0], width=ink.shape[1], level=level,
                                 **damage(ink, restored)))
            if len(panels) < 4:
                panels.append((e["id"].split(":")[-1], ink))
        csv_rows(args.out / "real_downsampling.csv", rows)
    # 同一个符号、仅字号不同，用于控制变量对照。
    controls = []
    for fontsize in (14, 40):
        ink = augment(render_symbol("x", fontsize=fontsize), out=64)
        panels.append((f"controlled x fontsize={fontsize}", ink))
        for level in range(5):
            controls.append(dict(fontsize=fontsize, level=level,
                                 **damage(ink, downsample(ink, level, args.device))))
    csv_rows(args.out / "controlled_downsampling.csv", controls)
    fig, axes = plt.subplots(len(panels), 5, figsize=(13, 2.3*len(panels)), squeeze=False)
    for i, (label, ink) in enumerate(panels):
        for level in range(5):
            axes[i, level].imshow(downsample(ink, level, args.device), cmap="gray_r", vmin=0, vmax=1)
            axes[i, level].set_title(f"{label} / {2**level}x", fontsize=8)
            axes[i, level].axis("off")
    fig.tight_layout()
    fig.savefig(args.out / "downsampling.png", dpi=150)
    plt.close(fig)
    means = []
    for level in range(5):
        subset = [r for r in rows if r["level"] == level and r["foreground_contrast"] is not None]
        if subset:
            means.append(dict(level=level, n=len(subset), **{
                k: float(np.mean([r[k] for r in subset]))
                for k in ("mse", "foreground_contrast", "foreground_recall")}))
    return dict(real_status="completed" if examples else "skipped: no local PNG cache",
                dataset="OleehyO/latex-formulas cleaned_formulas", split="latex-v1 val",
                n=len(examples), means=means, controlled=controls,
                caveat="像素损伤指标不等于 OCR；has_script 仅为公式标签，不是小字框标注")


def part4(args):
    device = args.device
    dtype = torch.float32
    x = torch.randn(8, 3, 6, 5, device=device, dtype=dtype, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    ours, ref = ManualBatchNorm2d(3).to(device), nn.BatchNorm2d(3).to(device)
    a, b = ours(x), ref(y)
    upstream = torch.randn_like(a)
    (a * upstream).sum().backward()
    (b * upstream).sum().backward()
    parity = dict(forward_max_abs=float((a-b).abs().max().detach()),
                  dx_max_abs=float((x.grad-y.grad).abs().max()),
                  dweight_max_abs=float((ours.weight.grad-ref.weight.grad).abs().max()),
                  running_var_max_abs=float((ours.running_var-ref.running_var).abs().max()))
    assert max(parity.values()) < 1e-4
    anchor = x[:1].detach()
    same = torch.cat((anchor, x[1:].detach()))
    shifted = torch.cat((anchor, x[1:].detach() + 5))
    ln = nn.LayerNorm((3, 6, 5)).to(device)
    rows = []
    for mode in ("train", "eval"):
        ours.train(mode == "train")
        left = ours(same)[0].detach()
        right = ours(shifted)[0].detach()
        rows.append(dict(mode=mode, bn_anchor_mean_abs_change=float((left-right).abs().mean()),
                         ln_anchor_mean_abs_change=float((ln(same)[0]-ln(shifted)[0]).abs().mean().detach())))
    csv_rows(args.out / "normalization.csv", rows)
    return dict(dtype="float32", batch=8, shape=[8, 3, 6, 5], parity=parity, context=rows,
                ln_axes="C,H,W per sample; BN axes=N,H,W per channel")


@torch.no_grad()
def evaluate(net, x, y, device, batch_size=512):
    net.eval()
    total_loss, correct = 0., 0
    for start in range(0, len(y), batch_size):
        xb, yb = x[start:start+batch_size].to(device), y[start:start+batch_size].to(device)
        logits = net(xb)
        total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += int((logits.argmax(1) == yb).sum())
    return dict(loss=total_loss/len(y), accuracy=correct/len(y), correct=correct, n=len(y))


def group_interval(net, x, y, groups, device):
    """按来源组 bootstrap（1000 次，固定种子），不假设组内样本独立。"""
    net.eval()
    with torch.no_grad():
        ok = torch.cat([(net(x[i:i+512].to(device)).argmax(1).cpu() == y[i:i+512])
                        for i in range(0, len(y), 512)]).numpy()
    unique, inv = np.unique(groups, return_inverse=True)
    success = np.bincount(inv, weights=ok)
    counts = np.bincount(inv)
    rng = np.random.default_rng(0)
    selected = rng.integers(len(unique), size=(1000, len(unique)))
    samples = success[selected].sum(1) / counts[selected].sum(1)
    return np.quantile(samples, [.025, .975]).tolist()


def part5(args):
    if (args.out / "training.json").exists():
        raise FileExistsError("training.json exists; use a fresh --out to preserve the experiment")
    cache = args.out / "symbols.npz"
    if cache.exists():
        raise FileExistsError("symbols.npz exists; use a fresh --out")
    print(f"生成控制数据 sources/class={args.sources} per_source={args.per_source}", flush=True)
    data = make_dataset(args.sources, args.per_source, seed=args.seed)
    np.savez_compressed(cache, images=data["images"], labels=data["labels"], groups=data["groups"])
    tr, va, te = group_split(data["groups"], seed=0)
    assert count_shared_groups(data["groups"], tr, va, te) == 0
    x = torch.from_numpy(data["images"])[:, None]
    y = torch.from_numpy(data["labels"])
    train_x, train_y, val_x, val_y = x[tr], y[tr], x[va], y[va]
    data_hash = hashlib.sha256(data["images"].tobytes() + data["labels"].tobytes() + data["groups"].tobytes()).hexdigest()
    configs, curves = [], []
    # 全部模型先训练、只看 val 选检查点；最后才进行一次预声明的 test 比较。
    for seed in args.seeds:
        for kind in ("mlp", "cnn"):
            torch.manual_seed(seed)
            net = SymbolClassifier(kind, args.width).to(args.device)
            params = sum(p.numel() for p in net.parameters() if p.requires_grad)
            opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            gen = torch.Generator().manual_seed(seed)
            best, best_step, first_loss, last_loss = float("inf"), 0, None, None
            checkpoint = args.out / f"{kind}_seed{seed}.pt"
            synchronize(args.device)
            started = time.perf_counter()
            if args.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            for step in range(1, args.steps + 1):
                idx = torch.randint(len(tr), (args.batch_size,), generator=gen)
                xb, yb = train_x[idx].to(args.device), train_y[idx].to(args.device)
                net.train()
                logits = net(xb)
                assert logits.shape == (args.batch_size, 24) and int(yb.min()) >= 0 and int(yb.max()) < 24
                loss = F.cross_entropy(logits, yb)  # 每张图一个有效标签，没有 padding / label mask。
                assert torch.isfinite(loss), "non-finite loss"
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if step == 1:
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
                opt.step()
                last_loss = float(loss.detach())
                if first_loss is None:
                    first_loss = last_loss
                    print(f"smoke model={kind} seed={seed} trainable={params} labels=all_valid loss={first_loss:.6f}", flush=True)
                if step == 1 or step % args.eval_every == 0 or step == args.steps:
                    train = evaluate(net, train_x, train_y, args.device)
                    val = evaluate(net, val_x, val_y, args.device)
                    row = dict(model=kind, seed=seed, step=step, train_loss=train["loss"],
                               train_acc=train["accuracy"], val_loss=val["loss"], val_acc=val["accuracy"])
                    curves.append(row)
                    print(json.dumps(row), flush=True)
                    if val["loss"] < best:
                        best, best_step = val["loss"], step
                        torch.save(dict(model=net.state_dict(), kind=kind, width=args.width, seed=seed,
                                        step=step, val_loss=best, data_sha256=data_hash), checkpoint)
            synchronize(args.device)
            elapsed = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated() / 2**20 if args.device.type == "cuda" else None
            saved = torch.load(checkpoint, map_location=args.device, weights_only=True)
            net.load_state_dict(saved["model"])
            validation = evaluate(net, val_x, val_y, args.device)
            # 下采样对识别率的影响也只看 val，使用同一个已选定模型，不重新训练。
            degradation = []
            for level in (1, 2, 3):
                low = F.avg_pool2d(val_x, 2**level)
                restored = F.interpolate(low, size=(32, 32), mode="nearest")
                degradation.append(dict(level=level, **evaluate(net, restored, val_y, args.device)))
            configs.append(dict(model=kind, seed=seed, parameters=params, best_step=best_step,
                                seconds=elapsed, peak_allocated_mib=peak, first_batch_loss=first_loss,
                                last_batch_loss=last_loss, validation=validation, degradation=degradation,
                                checkpoint=checkpoint.name))
    csv_rows(args.out / "learning_curves.csv", curves)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for r in configs:
        history = [v for v in curves if v["model"] == r["model"] and v["seed"] == r["seed"]]
        label = f"{r['model']} seed={r['seed']}"
        axes[0].plot([v["step"] for v in history], [v["train_loss"] for v in history], "--", alpha=.6, label=label+" train")
        axes[0].plot([v["step"] for v in history], [v["val_loss"] for v in history], label=label+" val")
        axes[1].plot([v["step"] for v in history], [v["val_acc"] for v in history], label=label)
    for ax in axes:
        ax.set_xlabel("step")
        ax.legend(fontsize=6)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("cross entropy")
    axes[1].set_ylabel("validation accuracy")
    fig.tight_layout()
    fig.savefig(args.out / "learning_curves.png", dpi=150)
    plt.close(fig)
    result = dict(source="ch05 controlled synthetic symbols (24 classes, 5 font families)",
                  split="source groups 80/10/10, seed=0", counts=dict(train=len(tr), val=len(va), test=len(te)),
                  shared_groups=0, data_sha256=data_hash, selection="minimum validation CE per model/seed",
                  dtype="float32; TF32 disabled", configs=configs, final_test=[])
    write_json(args.out / "frozen_validation.json", result)
    if args.final_test:
        # 所有设置、模型、检查点在 frozen_validation.json 写完后冻结；test 不反馈到上面的循环。
        # 一次前向同时收集准确率、CE 和组 bootstrap，避免重复评测。
        for r in configs:
            net = SymbolClassifier(r["model"], args.width).to(args.device)
            net.load_state_dict(torch.load(args.out / r["checkpoint"], map_location=args.device, weights_only=True)["model"])
            net.eval()
            with torch.no_grad():
                logits = torch.cat([net(x[te][i:i+512].to(args.device)).cpu() for i in range(0, len(te), 512)])
            ok = (logits.argmax(1) == y[te]).numpy()
            _, inv = np.unique(data["groups"][te], return_inverse=True)
            successes, counts = np.bincount(inv, weights=ok), np.bincount(inv)
            rng = np.random.default_rng(0)
            picks = rng.integers(len(counts), size=(1000, len(counts)))
            boot = successes[picks].sum(1) / counts[picks].sum(1)
            result["final_test"].append(dict(model=r["model"], seed=r["seed"], n=len(te),
                correct=int(ok.sum()), accuracy=float(ok.mean()), loss=float(F.cross_entropy(logits, y[te])),
                group_bootstrap_95=np.quantile(boot, [.025, .975]).tolist()))
        write_json(args.out / "final_test.json", result["final_test"])
    write_json(args.out / "training.json", result)
    return result


PARTS = {1: part1, 2: part2, 3: part3, 4: part4, 5: part5}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="*", type=int)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--n", type=int, default=20, help="real validation PNGs")
    parser.add_argument("--out", type=Path, default=Path("outputs/book/ch28/local"))
    parser.add_argument("--seed", type=int, default=0, help="data/mechanism seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="paired training seeds")
    parser.add_argument("--sources", type=int, default=8, help="sources per class")
    parser.add_argument("--per-source", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--weight-decay", type=float, default=.0001)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--require-data", action="store_true")
    parser.add_argument("--final-test", action="store_true", help="evaluate frozen final settings once; never use in smoke/tuning")
    args = parser.parse_args(argv)
    if any(p not in PARTS for p in args.parts):
        parser.error("parts must be 1..5")
    if min(args.n, args.sources, args.per_source, args.steps, args.batch_size, args.width, args.eval_every, args.threads) < 1:
        parser.error("size arguments must be positive")
    if args.lr <= 0 or args.weight_decay < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("invalid optimizer or duplicate seeds")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    args.device = pick_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    started = datetime.now(timezone.utc).isoformat()
    settings = {k: str(v) if isinstance(v, (Path, torch.device)) else v for k, v in vars(args).items()}
    meta = dict(start_utc=started, python=platform.python_version(), torch=torch.__version__,
                numpy=np.__version__, gpu=torch.cuda.get_device_name() if args.device.type == "cuda" else None,
                settings=settings, source="本项目观察；1=float64 CPU，其他=float32 指定设备")
    print(json.dumps(meta, ensure_ascii=False), flush=True)
    results = {}
    for part in args.parts or PARTS:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        synchronize(args.device)
        begin = time.perf_counter()
        print(f"\n部分 {part}", flush=True)
        result = PARTS[part](args)
        synchronize(args.device)
        result["part_wall_seconds"] = time.perf_counter() - begin
        results[str(part)] = result
        write_json(args.out / f"part{part}.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    meta["end_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.out / "run.json", dict(metadata=meta, results=results))
    print(f"DONE out={args.out} end={meta['end_utc']}", flush=True)


if __name__ == "__main__":
    main()
