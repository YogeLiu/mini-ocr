"""A2 参考实现：两个训练入口。在 course-v3/ 下运行：

    python -m assignments.solutions.a2_sequences.a2.train ocr --smoke
    python -m assignments.solutions.a2_sequences.a2.train ocr --set device=cuda --set out_dir=outputs/a2/ocr
    python -m assignments.solutions.a2_sequences.a2.train ocr-predict --ckpt outputs/a2/ocr/ckpt.pt --out outputs/a2/ocr_preds.txt
    python -m assignments.solutions.a2_sequences.a2.train lm --smoke
    python -m assignments.solutions.a2_sequences.a2.train lm --set vocab_size=1024 --set out_dir=outputs/a2/lm
    python -m assignments.solutions.a2_sequences.a2.train lm-eval --ckpt outputs/a2/lm/ckpt.pt       # 排行榜测试文本的 bits/char
    python -m assignments.solutions.a2_sequences.a2.train lm-sample --ckpt outputs/a2/lm/ckpt.pt --prompt "# The"
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from .ctc import ctc_loss, greedy_decode, prefix_beam_search
from .generate import generate
from .lm import (LINE_CLASSES, accumulate_grads, bits_per_char, cer, decode_line, encode_line, get_batch,
                 nll_sum)
from .model import GPT, GPTConfig
from .optim import AdamW, clip_grad_norm, lr_cosine_schedule
from .rnn import CRNN
from .tokenizer import Tokenizer, train_bpe

OCR = dict(n_train=20000, n_val=500, noise=0.15, distort=1.0, train_seed=0, val_seed=1, c=128, hidden=128, layers=2,
           batch_size=64, max_lr=1e-3, min_lr=1e-4, warmup_iters=200, max_iters=3000, lr_decay_iters=0,
           weight_decay=0.01, clip=1.0, eval_every=500, seed=0, device="cpu", out_dir="", beam=0)
OCR_SMOKE = dict(n_train=64, n_val=16, c=16, hidden=16, layers=1, batch_size=8, max_iters=10, warmup_iters=2, eval_every=5)
LM = dict(train_docs=20000, val_docs=400, train_seed=0, val_seed=1, vocab_size=1024, block_size=256, n_layer=4,
          n_head=4, d_model=256, batch_size=32, micro_batches=1, max_lr=3e-3, min_lr=3e-4, warmup_iters=200,
          max_iters=3000, lr_decay_iters=0, weight_decay=0.1, clip=1.0, eval_every=500, seed=0, device="cpu", out_dir="")
LM_SMOKE = dict(train_docs=200, val_docs=20, vocab_size=200, block_size=64, n_layer=1, n_head=2, d_model=32,
                batch_size=4, max_iters=10, warmup_iters=2, eval_every=5)


def _cfg(defaults, smoke, items):
    cfg = dict(defaults, **(smoke or {}))
    for item in items:
        k, v = item.split("=", 1)
        old = cfg[k]                                                  # 键写错时这里报错
        cfg[k] = type(old)(v) if not isinstance(old, bool) else v.lower() in ("1", "true")
    return cfg


class Run:
    """日志（log.jsonl）、配置（config.json）、检查点（ckpt.pt）。out_dir 为空时什么也不写。"""

    def __init__(self, cfg):
        self.out = cfg["out_dir"]
        self.t0 = time.time()
        if self.out:
            os.makedirs(self.out, exist_ok=True)
            json.dump(cfg, open(os.path.join(self.out, "config.json"), "w"), indent=1)
            self.f = open(os.path.join(self.out, "log.jsonl"), "a")

    def log(self, rec):
        rec["elapsed"] = round(time.time() - self.t0, 2)
        if self.out:
            self.f.write(json.dumps(rec) + "\n")
            self.f.flush()

    def save(self, obj):
        if self.out:
            torch.save(obj, os.path.join(self.out, "ckpt.pt"))


def _step(model, opt, cfg, it, loss_fn):
    lr = lr_cosine_schedule(it, cfg["max_lr"], cfg["min_lr"], cfg["warmup_iters"], cfg["lr_decay_iters"] or cfg["max_iters"])
    for g in opt.param_groups:
        g["lr"] = lr
    loss = loss_fn()
    if cfg["clip"]:
        clip_grad_norm(model.parameters(), cfg["clip"])
    opt.step()
    return loss, lr


# ---------------- 文字行识别 ----------------
def line_data(seed, n, noise, distort):
    from miniocr.data.text import LineDataset
    ds = LineDataset(seed, n, noise=noise, distort=distort)
    return ds.x.numpy(), ds.texts


@torch.no_grad()
def ocr_decode(model, X, device, beam=0, bs=256):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        lp = model(torch.as_tensor(X[i:i + bs]).to(device)).float().cpu()
        for b in range(lp.shape[1]):
            ids = prefix_beam_search(lp[:, b], beam) if beam else greedy_decode(lp[:, b])
            out.append(decode_line(ids))
    model.train()
    return out


def train_ocr(cfg):
    X, texts = line_data(cfg["train_seed"], cfg["n_train"], cfg["noise"], cfg["distort"])
    Xv, tv = line_data(cfg["val_seed"], cfg["n_val"], cfg["noise"], cfg["distort"])
    labels = [encode_line(t) for t in texts]
    dev = torch.device(cfg["device"])
    torch.manual_seed(cfg["seed"])
    model = CRNN(len(LINE_CLASSES), cfg["c"], cfg["hidden"], cfg["layers"]).to(dev)
    opt = AdamW(model.parameters(), lr=cfg["max_lr"], weight_decay=cfg["weight_decay"])
    gen = torch.Generator().manual_seed(cfg["seed"] + 1)
    run = Run(cfg)
    print(f"训练 {len(X)} 行，验证 {len(Xv)} 行；参数量 {sum(p.numel() for p in model.parameters()):,}；设备 {dev}")
    for it in range(cfg["max_iters"]):
        idx = torch.randint(0, len(X), (cfg["batch_size"],), generator=gen).tolist()
        xb = torch.from_numpy(X[idx]).to(dev)
        S = max(len(labels[i]) for i in idx)
        tg = torch.zeros(len(idx), S, dtype=torch.long)
        for r, i in enumerate(idx):
            tg[r, :len(labels[i])] = torch.tensor(labels[i])
        tl = torch.tensor([len(labels[i]) for i in idx])

        def loss_fn():
            lp = model(xb)
            il = torch.full((len(idx),), lp.shape[0], dtype=torch.long)
            loss = ctc_loss(lp.float(), tg.to(dev), il.to(dev), tl.to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            return loss.item()

        loss, lr = _step(model, opt, cfg, it, loss_fn)
        rec = {"iter": it + 1, "loss": loss, "lr": lr}
        if (it + 1) % cfg["eval_every"] == 0 or it + 1 == cfg["max_iters"]:
            rec["val_cer"] = cer(ocr_decode(model, Xv, dev, cfg["beam"]), tv)
            print(f"  步 {it + 1:5d}  loss {loss:.4f}  val_cer {rec['val_cer']:.4f}  lr {lr:.2e}  {time.time() - run.t0:.0f}s")
            run.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "iteration": it + 1, "cfg": cfg})
        run.log(rec)
    return model


def ocr_predict(ckpt, out, device="cpu", beam=0):
    from assignments.a2_sequences.leaderboard import load_ocr_test
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    c = ck["cfg"]
    model = CRNN(len(LINE_CLASSES), c["c"], c["hidden"], c["layers"])
    model.load_state_dict(ck["model"])
    X, _ = load_ocr_test()
    preds = ocr_decode(model.to(device), X, torch.device(device), beam)
    open(out, "w").write("\n".join(preds) + "\n")
    print(f"{len(preds)} 行预测存到 {out}")


# ---------------- 语言模型 ----------------
def corpus(seed, n):
    from miniocr.data.corpus import mixed_corpus
    return mixed_corpus(seed, n)


def train_lm(cfg):
    tr, va = corpus(cfg["train_seed"], cfg["train_docs"]), corpus(cfg["val_seed"], cfg["val_docs"])
    t = time.time()
    vocab, merges = train_bpe(tr, cfg["vocab_size"])
    tok = Tokenizer(vocab, merges)
    ids, vids = np.array(tok.encode(tr), dtype=np.int32), tok.encode(va)
    print(f"BPE：词表 {len(vocab)}，{time.time() - t:.1f} 秒；训练 {len(tr):,} 字符 → {len(ids):,} token（{len(tr) / len(ids):.2f} 字符/token）")
    dev = torch.device(cfg["device"])
    torch.manual_seed(cfg["seed"])
    gcfg = GPTConfig(vocab_size=len(vocab), block_size=cfg["block_size"], n_layer=cfg["n_layer"],
                     n_head=cfg["n_head"], d_model=cfg["d_model"])
    model = GPT(gcfg).to(dev)
    opt = AdamW(model.parameters(), lr=cfg["max_lr"], weight_decay=cfg["weight_decay"])
    gen = torch.Generator().manual_seed(cfg["seed"] + 1)
    run = Run(cfg)
    print(f"参数量 {model.n_params():,}；设备 {dev}")
    for it in range(cfg["max_iters"]):
        x, y = get_batch(ids, cfg["batch_size"], cfg["block_size"], gen, dev)
        loss, lr = _step(model, opt, cfg, it, lambda: accumulate_grads(model, x, y, cfg["micro_batches"]))
        rec = {"iter": it + 1, "loss": loss, "lr": lr}
        if (it + 1) % cfg["eval_every"] == 0 or it + 1 == cfg["max_iters"]:
            rec["val_bpc"] = bits_per_char(nll_sum(model, vids, cfg["block_size"], dev), len(va))
            print(f"  步 {it + 1:5d}  loss {loss:.4f}  val_bpc {rec['val_bpc']:.4f}  lr {lr:.2e}  {time.time() - run.t0:.0f}s")
            run.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "iteration": it + 1, "cfg": cfg,
                      "gpt": gcfg.__dict__, "vocab": vocab, "merges": merges})
        run.log(rec)
    return model, tok


def _load_lm(ckpt, device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ck["gpt"]))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), Tokenizer(ck["vocab"], ck["merges"]), ck["cfg"]


def lm_eval(ckpt, device="cpu"):
    from assignments.a2_sequences.leaderboard import load_lm_test
    model, tok, cfg = _load_lm(ckpt, device)
    text = load_lm_test()
    bpc = bits_per_char(nll_sum(model, tok.encode(text), cfg["block_size"], device), len(text))
    print(f"排行榜测试文本：{len(text):,} 字符，{bpc:.4f} bits/char")


def lm_sample(ckpt, prompt, n=200, temperature=0.8, top_p=0.95, device="cpu", seed=0):
    model, tok, _ = _load_lm(ckpt, device)
    idx = torch.tensor([tok.encode(prompt)], device=device)
    out = generate(model, idx, n, temperature=temperature, top_p=top_p, generator=torch.Generator(device).manual_seed(seed))
    print(tok.decode(out[0].tolist()))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("ocr", "lm"):
        p = sub.add_parser(name)
        p.add_argument("--set", action="append", default=[])
        p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("ocr-predict")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--beam", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("lm-eval")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("lm-sample")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default="")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    a = ap.parse_args()
    if a.cmd == "ocr":
        train_ocr(_cfg(OCR, OCR_SMOKE if a.smoke else None, a.set))
    elif a.cmd == "lm":
        train_lm(_cfg(LM, LM_SMOKE if a.smoke else None, a.set))
    elif a.cmd == "ocr-predict":
        ocr_predict(a.ckpt, a.out, a.device, a.beam)
    elif a.cmd == "lm-eval":
        lm_eval(a.ckpt, a.device)
    else:
        lm_sample(a.ckpt, a.prompt, a.n, a.temperature, a.top_p)


if __name__ == "__main__":
    main()
