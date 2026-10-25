"""A1 参考实现：训练脚本。在 course-v3/ 下运行：

    python -m assignments.solutions.a1_basics.a1.train run --smoke                 # 冒烟：几十秒
    python -m assignments.solutions.a1_basics.a1.train run --set max_lr=1e-2 --set out_dir=outputs/a1/lr1e-2
    python -m assignments.solutions.a1_basics.a1.train run --set device=cuda ...   # 在 GPU 上
    python -m assignments.solutions.a1_basics.a1.train run --resume outputs/a1/run/ckpt.pt --set max_iters=4000
    python -m assignments.solutions.a1_basics.a1.train predict --ckpt outputs/a1/run/ckpt.pt \\
        --x assignments/a1_basics/data/leaderboard_test.npz --out outputs/a1/preds.npy
"""
import argparse
import dataclasses
import json
import os
import time

import numpy as np
import torch

from .data import build_split, get_batch, remove_overlap
from .model import CharCNN, count_params, load_checkpoint, save_checkpoint
from .nn import cross_entropy
from .optim import AdamW, clip_grad_norm, lr_cosine_schedule


@dataclasses.dataclass
class TrainConfig:
    n_per_class: int = 100
    val_per_class: int = 10
    distort: float = 1.0
    noise: float = 0.05
    train_seed: int = 100
    val_seed: int = 101
    channels: tuple = (32, 64, 128)
    batch_size: int = 128
    max_lr: float = 3e-3
    min_lr: float = 3e-4
    warmup_iters: int = 100
    max_iters: int = 2000
    lr_decay_iters: int = 0              # 0 表示等于 max_iters；续训时固定它，调度就不会因为 max_iters 改变而改变
    weight_decay: float = 0.01
    clip: float = 1.0
    label_smoothing: float = 0.0
    eval_every: int = 200
    seed: int = 0
    device: str = "cpu"
    out_dir: str = ""                    # 空字符串：不写日志和检查点


SMOKE = dict(n_per_class=5, val_per_class=2, channels=(8, 16, 32), max_iters=20, warmup_iters=5, eval_every=10)


def make_data(cfg):
    X, y = build_split(cfg.train_seed, cfg.n_per_class, cfg.distort, cfg.noise)
    Xv, yv = build_split(cfg.val_seed, cfg.val_per_class, cfg.distort, cfg.noise)
    X, y, n = remove_overlap(X, y, Xv)
    return X, y, Xv, yv, n


@torch.no_grad()
def evaluate(model, X, y, device, bs=1024):
    was = model.training
    model.eval()
    loss, correct = 0.0, 0
    for i in range(0, len(X), bs):
        xb, yb = torch.from_numpy(X[i:i + bs]).to(device), torch.from_numpy(y[i:i + bs]).to(device)
        logits = model(xb)
        loss += cross_entropy(logits, yb).item() * len(xb)
        correct += (logits.argmax(-1) == yb).sum().item()
    model.train(was)
    return loss / len(X), correct / len(X)


def train(cfg, data=None, resume=None, log=print):
    """训练 CharCNN。data = (X, y, Xv, yv)，None 时按 cfg 生成。resume：检查点路径，从那里接着训练。
    返回 {"model", "optimizer", "history"}；history 里每一步一条 {"iter", "loss", "lr"}，评测的步还有 val_loss、val_acc。"""
    if data is None:
        X, y, Xv, yv, n = make_data(cfg)
        log(f"训练 {len(X)} 张（去掉与验证集重复的 {n} 张），验证 {len(Xv)} 张")
    else:
        X, y, Xv, yv = data
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    model = CharCNN(num_classes=int(max(y.max(), yv.max())) + 1, channels=tuple(cfg.channels)).to(device)
    opt = AdamW(model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    start = 0
    if resume:
        start, extra = load_checkpoint(resume, model, opt)
        gen.set_state(extra["gen"])
        log(f"从 {resume} 的第 {start} 步继续")
    decay = cfg.lr_decay_iters or cfg.max_iters
    log(f"参数量 {count_params(model):,}；设备 {device}")

    out = cfg.out_dir
    if out:
        os.makedirs(out, exist_ok=True)
        json.dump(dataclasses.asdict(cfg), open(os.path.join(out, "config.json"), "w"), indent=1)
        flog = open(os.path.join(out, "log.jsonl"), "a")
    history, t0 = [], time.time()
    model.train()
    for it in range(start, cfg.max_iters):
        lr = lr_cosine_schedule(it, cfg.max_lr, cfg.min_lr, cfg.warmup_iters, decay)
        for g in opt.param_groups:
            g["lr"] = lr
        xb, yb = get_batch(X, y, cfg.batch_size, gen, device)
        loss = cross_entropy(model(xb), yb, cfg.label_smoothing)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.clip:
            clip_grad_norm(model.parameters(), cfg.clip)
        opt.step()
        rec = {"iter": it + 1, "loss": loss.item(), "lr": lr}
        last = it + 1 == cfg.max_iters
        if (it + 1) % cfg.eval_every == 0 or last:
            rec["val_loss"], rec["val_acc"] = evaluate(model, Xv, yv, device)
            rec["elapsed"] = round(time.time() - t0, 2)
            log(f"  步 {it + 1:5d}  loss {rec['loss']:.4f}  val_loss {rec['val_loss']:.4f}  "
                f"val_acc {rec['val_acc']:.2%}  lr {lr:.2e}  {rec['elapsed']:.0f}s")
            if out:
                save_checkpoint(model, opt, it + 1, os.path.join(out, "ckpt.pt"), extra={"gen": gen.get_state()})
        if out:
            flog.write(json.dumps(rec) + "\n")
        history.append(rec)
    if out:
        flog.close()
    return {"model": model, "optimizer": opt, "history": history}


def _parse_set(items, cfg):
    for item in items:
        k, v = item.split("=", 1)
        old = getattr(cfg, k)                                     # 键写错时这里会报错
        if isinstance(old, tuple):
            v = tuple(int(s) for s in v.split(","))
        elif isinstance(old, bool):
            v = v.lower() in ("1", "true", "yes")
        else:
            v = type(old)(v)
        setattr(cfg, k, v)
    return cfg


@torch.no_grad()
def predict(ckpt, x_path, out, device="cpu"):
    cfg = TrainConfig(**json.load(open(os.path.join(os.path.dirname(ckpt), "config.json"))))
    if x_path.endswith(".npz"):                                     # 排行榜的测试集：uint8 原图，换算成训练时的格式
        X = (1.0 - np.load(x_path)["img"].astype(np.float32) / 255.0)[:, None]
    else:
        X = np.load(x_path)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    model = CharCNN(num_classes=sd["head.weight"].shape[0], channels=tuple(cfg.channels))
    model.load_state_dict(sd)
    model.to(device).eval()
    preds = np.concatenate([model(torch.from_numpy(X[i:i + 1024]).to(device)).argmax(-1).cpu().numpy()
                            for i in range(0, len(X), 1024)])
    np.save(out, preds)
    print(f"{len(preds)} 个预测存到 {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--set", action="append", default=[], help="覆盖配置，例如 --set max_lr=1e-2")
    r.add_argument("--smoke", action="store_true", help="极小的配置，只检查能不能跑通")
    r.add_argument("--resume", default=None)
    p = sub.add_parser("predict")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--x", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.cmd == "predict":
        return predict(a.ckpt, a.x, a.out, a.device)
    cfg = TrainConfig(**(SMOKE if a.smoke else {}))
    cfg = _parse_set(a.set, cfg)
    train(cfg, resume=a.resume)


if __name__ == "__main__":
    main()
