"""A3 参考实现：四个训练入口和评测。在 course-v3/ 下运行（M = assignments.solutions.a3_vlm.a3.train）：

    python -m $M clip --set out_dir=outputs/a3/clip                     # 对比学习预训练视觉塔 → vit.pt
    python -m $M lm   --set out_dir=outputs/a3/lm                       # 字符级 GPT 预训练 → lm.pt
    python -m $M vlm  --set stage=align --set vit=outputs/a3/clip/vit.pt --set lm=outputs/a3/lm/lm.pt --set out_dir=outputs/a3/align
    python -m $M vlm  --set stage=instruct --set init=outputs/a3/align/vlm.pt --set out_dir=outputs/a3/instruct
    python -m $M vlm  --set stage=instruct --set out_dir=outputs/a3/scratch          # 不给 vit/lm/init：从零训练
    python -m $M eval --ckpt outputs/a3/instruct/vlm.pt                 # 自己的验证集，逐任务准确率
    python -m $M predict --ckpt outputs/a3/instruct/vlm.pt --out outputs/a3/preds.jsonl     # 排行榜
所有入口都支持 --smoke 和 --set key=value；默认 device=cpu。
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from miniocr.data.instruct import TASKS

from .contrastive import DualEncoder, TextEncoder
from .data import (BOS, EOS, IMG, PAD, TOK, VOCAB_SIZE, caption_batch, convo, correct, eval_set, example,
                   prompt_ids, sample_batch, to_float)
from .model import GPT, GPTConfig
from .optim import AdamW, clip_grad_norm, lr_cosine_schedule
from .vision import ViT, ViTConfig
from .vlm import VLM, MergeProjector, MLPProjector, QFormer, set_trainable

VIT = dict(image_size=64, patch=8, in_ch=3, d=192, n_layer=4, n_head=4, pos="rope2d")
GPT_CFG = dict(vocab_size=VOCAB_SIZE, block_size=320, n_layer=4, n_head=4, d_model=256)
COMMON = dict(batch_size=32, max_lr=3e-4, min_lr=3e-5, warmup_iters=100, max_iters=3000, weight_decay=0.01, clip=1.0,
              eval_every=500, seed=0, device="cpu", out_dir="")
CLIP = dict(COMMON, loss="siglip", batch_size=128, max_lr=1e-3, max_iters=2000)
LM = dict(COMMON, train_docs=20000, max_lr=3e-3, batch_size=32)
VLMC = dict(COMMON, stage="instruct", connector="merge", vit="", lm="", init="", train_vision=True, train_lm=True,
            multi_turn=0.3)
SMOKE = dict(batch_size=4, max_iters=4, warmup_iters=1, eval_every=2, train_docs=50)
TINY_VIT = dict(VIT, d=32, n_layer=1, n_head=2)
TINY_GPT = dict(GPT_CFG, n_layer=1, n_head=2, d_model=32)


def _cfg(defaults, smoke, items):
    cfg = dict(defaults, **({k: v for k, v in SMOKE.items() if k in defaults} if smoke else {}))
    cfg["smoke"] = smoke
    for item in items:
        k, v = item.split("=", 1)
        old = cfg[k]
        cfg[k] = v.lower() in ("1", "true") if isinstance(old, bool) else type(old)(v)
    return cfg


def _arch(cfg):
    return (TINY_VIT, TINY_GPT) if cfg["smoke"] else (VIT, GPT_CFG)


class Run:
    def __init__(self, cfg):
        self.out, self.t0 = cfg["out_dir"], time.time()
        if self.out:
            os.makedirs(self.out, exist_ok=True)
            json.dump(cfg, open(os.path.join(self.out, "config.json"), "w"), indent=1)
            self.f = open(os.path.join(self.out, "log.jsonl"), "a")

    def log(self, rec, show=False):
        rec["elapsed"] = round(time.time() - self.t0, 2)
        if self.out:
            self.f.write(json.dumps(rec) + "\n")
            self.f.flush()
        if show:
            print("  " + "  ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()))

    def save(self, name, obj):
        if self.out:
            torch.save(obj, os.path.join(self.out, name))


def _loop(cfg, model, params, batch_fn, loss_fn, run, eval_fn=None):
    dev = torch.device(cfg["device"])
    opt = AdamW(params, lr=cfg["max_lr"], weight_decay=cfg["weight_decay"])
    rng = np.random.default_rng(cfg["seed"])
    model.train()
    for it in range(cfg["max_iters"]):
        lr = lr_cosine_schedule(it, cfg["max_lr"], cfg["min_lr"], cfg["warmup_iters"], cfg["max_iters"])
        for g in opt.param_groups:
            g["lr"] = lr
        loss = loss_fn(*[b.to(dev) for b in batch_fn(rng)])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg["clip"]:
            clip_grad_norm(params, cfg["clip"])
        opt.step()
        rec = {"iter": it + 1, "loss": loss.item(), "lr": lr}
        last = it + 1 == cfg["max_iters"]
        if (it + 1) % cfg["eval_every"] == 0 or last:
            if eval_fn is not None:
                rec.update(eval_fn())
            run.log(rec, show=True)
        else:
            run.log(rec)


# ---------------- 对比学习 ----------------
def train_clip(cfg):
    vcfg, _ = _arch(cfg)
    dev = torch.device(cfg["device"])
    torch.manual_seed(cfg["seed"])
    vit = ViT(ViTConfig(**vcfg))
    txt = TextEncoder(VOCAB_SIZE, max_len=128, d=vcfg["d"], n_layer=2, n_head=vcfg["n_head"], pad_id=PAD)
    model = DualEncoder(vit, txt, vcfg["d"], vcfg["d"], loss=cfg["loss"]).to(dev)
    run = Run(cfg)
    _loop(cfg, model, list(model.parameters()), lambda rng: caption_batch(rng, cfg["batch_size"]),
          lambda x, ids: model(to_float(x), ids), run)
    run.save("vit.pt", {"vit": vit.state_dict(), "cfg": vcfg})
    return model


# ---------------- 语言模型 ----------------
def get_batch(ids, batch_size, context, generator):
    """同 A2：随机起点，x = ids[s : s+context]，y 右移一位。"""
    starts = torch.randint(0, len(ids) - context, (batch_size,), generator=generator).numpy()
    x = np.stack([ids[s:s + context] for s in starts])
    y = np.stack([ids[s + 1:s + context + 1] for s in starts])
    return torch.from_numpy(x), torch.from_numpy(y)


def train_lm(cfg):
    from miniocr.data.corpus import mixed_corpus
    _, gcfg = _arch(cfg)
    ids = np.array(TOK.encode(mixed_corpus(0, cfg["train_docs"])), dtype=np.int64)
    dev = torch.device(cfg["device"])
    torch.manual_seed(cfg["seed"])
    model = GPT(GPTConfig(**gcfg)).to(dev)
    run = Run(cfg)
    T = 256 if not cfg["smoke"] else 32
    g = torch.Generator().manual_seed(cfg["seed"])
    _loop(cfg, model, list(model.parameters()), lambda rng: get_batch(ids, cfg["batch_size"], T, g),
          lambda x, y: model(x, y)[1], run)
    run.save("lm.pt", {"lm": model.state_dict(), "cfg": gcfg})
    return model


# ---------------- VLM ----------------
def build_vlm(cfg):
    vcfg, gcfg = _arch(cfg)
    torch.manual_seed(cfg["seed"])
    vit, lm = ViT(ViTConfig(**vcfg)), GPT(GPTConfig(**gcfg))
    if cfg["vit"]:
        vit.load_state_dict(torch.load(cfg["vit"], map_location="cpu")["vit"])
    if cfg["lm"]:
        lm.load_state_dict(torch.load(cfg["lm"], map_location="cpu")["lm"])
    d_v, d = vcfg["d"], gcfg["d_model"]
    conn = {"mlp": MLPProjector, "merge": MergeProjector, "qformer": QFormer}[cfg["connector"]](d_v, d)
    m = VLM(vit, conn, lm, IMG)
    if cfg["init"]:
        m.load_state_dict(torch.load(cfg["init"], map_location="cpu")["vlm"])
    return m


def n_img_tokens(cfg):
    g = (64 // VIT["patch"]) ** 2
    return {"mlp": g, "merge": g // 4, "qformer": 16}[cfg["connector"]]


@torch.no_grad()
def answer(m, items, n_img, device, max_new=96):
    """items：[(图像 uint8 (k, 3, 64, 64), 问题)]。按（图数, 提示长度）分组批量生成，返回答案字符串列表。"""
    out = [None] * len(items)
    groups = {}
    for i, (img, q) in enumerate(items):
        p = prompt_ids(q, n_img)
        groups.setdefault((img.shape[0], len(p)), []).append((i, p))
    for (k, _), group in groups.items():
        for s in range(0, len(group), 64):
            sub = group[s:s + 64]
            X = torch.from_numpy(np.stack([items[i][0] for i, _ in sub])).to(device)
            X = to_float(X[:, 0] if k == 1 else X)
            gen = m.generate(X, torch.tensor([p for _, p in sub], device=device), max_new, eos=EOS)
            for (i, _), g in zip(sub, gen.tolist()):
                g = g[:g.index(EOS)] if EOS in g else g
                out[i] = TOK.decode(g)
    return out


def evaluate(m, cfg, n_per_task=50):
    dev = torch.device(cfg["device"])
    data = eval_set(seed=7, n_per_task=2 if cfg["smoke"] else n_per_task)
    preds = answer(m, [(img, q) for _, img, q, _ in data], n_img_tokens(cfg), dev)
    acc = {}
    for (t, _, _, a), p in zip(data, preds):
        acc.setdefault(t, []).append(correct(t, p, a))
    res = {f"acc_{t}": float(np.mean(v)) for t, v in acc.items()}
    res["acc_mean"] = float(np.mean(list(res.values())))
    return res


def train_vlm(cfg):
    dev = torch.device(cfg["device"])
    m = build_vlm(cfg).to(dev)
    if cfg["stage"] == "align":                                       # 对齐：只训连接器，只用描述数据
        set_trainable(m, vision=False, connector=True, lm=False)
        tasks, weights = ["caption"], [1.0]
    else:
        set_trainable(m, vision=cfg["train_vision"], connector=True, lm=cfg["train_lm"])
        tasks, weights = list(TASKS), [1.0] * len(TASKS)
    params = [p for p in m.parameters() if p.requires_grad]
    n_img = n_img_tokens(cfg)
    run = Run(cfg)
    print(f"阶段 {cfg['stage']}；连接器 {cfg['connector']}（每张图 {n_img} 个视觉 token）；"
          f"可训练参数 {sum(p.numel() for p in params):,} / {sum(p.numel() for p in m.parameters()):,}；设备 {dev}")

    def batch(rng):
        X, ids, tg = sample_batch(rng, tasks, weights, cfg["batch_size"], n_img, cfg["multi_turn"])
        return X, ids, tg

    def loss_fn(X, ids, tg):
        X = to_float(X[:, 0] if X.shape[1] == 1 else X)
        return m(X, ids, tg)[1]

    _loop(cfg, m, params, batch, loss_fn, run, eval_fn=lambda: evaluate(m, cfg))
    run.save("vlm.pt", {"vlm": m.state_dict(), "cfg": cfg})
    return m


def _load(ckpt, device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"], vit="", lm="", init="", device=device)
    m = build_vlm(cfg)
    m.load_state_dict(ck["vlm"])
    return m.to(device), cfg


def predict(ckpt, out, device="cpu"):
    from assignments.a3_vlm.leaderboard import load_test
    m, cfg = _load(ckpt, device)
    test = load_test()
    preds = answer(m, [(img, q) for _, img, q, _ in test], n_img_tokens(cfg), torch.device(device))
    with open(out, "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    print(f"{len(preds)} 个回答存到 {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("clip", "lm", "vlm"):
        p = sub.add_parser(name)
        p.add_argument("--set", action="append", default=[])
        p.add_argument("--smoke", action="store_true")
    for name in ("eval", "predict"):
        p = sub.add_parser(name)
        p.add_argument("--ckpt", required=True)
        p.add_argument("--device", default="cpu")
        if name == "predict":
            p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "clip":
        train_clip(_cfg(CLIP, a.smoke, a.set))
    elif a.cmd == "lm":
        train_lm(_cfg(LM, a.smoke, a.set))
    elif a.cmd == "vlm":
        train_vlm(_cfg(VLMC, a.smoke, a.set))
    elif a.cmd == "eval":
        m, cfg = _load(a.ckpt, a.device)
        print(evaluate(m, cfg))
    else:
        predict(a.ckpt, a.out, a.device)


if __name__ == "__main__":
    main()
