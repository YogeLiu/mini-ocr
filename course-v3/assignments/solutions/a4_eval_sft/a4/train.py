"""A4 参考实现：SFT、评测、配对比较、内存、排行榜预测。在 course-v3/ 下运行（M = assignments.solutions.a4_eval_sft.a4.train）：

    python -m $M sft --set init=outputs/a3/align/vlm.pt --set out_dir=outputs/a4/full          # 从 A3 的对齐检查点开始
    python -m $M sft --set init=... --set lora_r=8 --set out_dir=outputs/a4/lora8               # LoRA（存盘前合并）
    python -m $M sft --set init=... --set prompt_weight=0.1 --set pack=true --set out_dir=...
    python -m $M eval --ckpt outputs/a4/full/vlm.pt --out outputs/a4/full/eval.jsonl            # 逐样本结果 + 置信区间 + 校准
    python -m $M compare outputs/a4/full/eval.jsonl outputs/a4/lora8/eval.jsonl                 # 配对 bootstrap + McNemar
    python -m $M memory --set lora_r=8                                                          # 估算与实测
    python -m $M predict --ckpt outputs/a4/full/vlm.pt --out outputs/a4/preds.jsonl             # 排行榜
所有训练入口都支持 --smoke 和 --set key=value；默认 device=cpu。不给 init 时从随机初始化开始。
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from .data import (ALL_TASKS, BOS, EOS, IMG, PAD, TOK, VOCAB_SIZE, correct, eval_set, n_vis, prompt_ids, sft_example,
                   to_float)
from .logprobs import mask_from_lengths, seq_logprob, token_logprobs
from .lora import apply_lora, count_params, merge_lora
from .memory import SavedTensors, checkpoint_blocks, static_memory, uncheckpoint_blocks
from .metrics import edit_distance
from .model import GPT, GPTConfig
from .optim import AdamW, clip_grad_norm, lr_cosine_schedule
from .sft import pack, packed_mask, pad_weighted, sft_loss
from .stats import auroc, bootstrap_ci, ece, mcnemar, paired_bootstrap
from .vision import ViT, ViTConfig
from .vlm import VLM, MergeProjector, MLPProjector, QFormer

VIT = dict(image_size=64, patch=8, in_ch=3, d=192, n_layer=4, n_head=4, pos="rope2d")
GPT_CFG = dict(vocab_size=VOCAB_SIZE, block_size=320, n_layer=4, n_head=4, d_model=256)
TINY_VIT = dict(VIT, d=32, n_layer=1, n_head=2)
TINY_GPT = dict(GPT_CFG, n_layer=1, n_head=2, d_model=32)
SFT = dict(init="", connector="merge", tasks=",".join(ALL_TASKS), read_weight=1.0, prompt_weight=0.0, loss_mode="token",
           train_eos=True, pack=False, max_len=384, multi_turn=0.3, lora_r=0, lora_alpha=16,
           lora_targets="qkv,o,gate,up,down", train_vision=False, train_connector=True, ckpt_blocks=False,
           batch_size=32, max_lr=3e-4, min_lr=3e-5, warmup_iters=100, max_iters=2000, weight_decay=0.01, clip=1.0,
           eval_every=500, eval_n=50, seed=0, device="cpu", out_dir="")
SMOKE = dict(batch_size=4, max_iters=4, warmup_iters=1, eval_every=2, eval_n=2)


def _cfg(defaults, smoke, items):
    cfg = dict(defaults, **({k: v for k, v in SMOKE.items() if k in defaults} if smoke else {}))
    cfg["smoke"] = smoke
    for item in items:
        k, v = item.split("=", 1)
        old = cfg[k]
        cfg[k] = v.lower() in ("1", "true") if isinstance(old, bool) else type(old)(v)
    return cfg


# ---------------- 模型 ----------------
def build(arch):
    vit, lm = ViT(ViTConfig(**arch["vit"])), GPT(GPTConfig(**arch["gpt"]))
    d_v, d = arch["vit"]["d"], arch["gpt"]["d_model"]
    conn = {"mlp": MLPProjector, "merge": MergeProjector, "qformer": QFormer}[arch["connector"]](d_v, d)
    return VLM(vit, conn, lm, IMG)


def load_vlm(path, smoke=False):
    """A4 的检查点（带 arch），或 A3 参考实现的检查点（cfg 里有 connector 和 smoke）。返回 (模型, arch)。"""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    arch = ck.get("arch")
    if arch is None:
        c = ck["cfg"]
        tiny = c.get("smoke", False)
        arch = dict(vit=TINY_VIT if tiny else VIT, gpt=TINY_GPT if tiny else GPT_CFG, connector=c["connector"])
    m = build(arch)
    m.load_state_dict(ck["vlm"])
    return m, arch


def prepare(cfg):
    """建模型、装 LoRA、决定哪些参数可训练。返回 (模型, arch)。"""
    torch.manual_seed(cfg["seed"])
    if cfg["init"]:
        m, arch = load_vlm(cfg["init"])
    else:
        arch = dict(vit=TINY_VIT if cfg["smoke"] else VIT, gpt=TINY_GPT if cfg["smoke"] else GPT_CFG, connector=cfg["connector"])
        m = build(arch)
    for p in m.parameters():
        p.requires_grad_(True)
    if cfg["lora_r"]:
        apply_lora(m.lm, cfg["lora_targets"].split(","), cfg["lora_r"], cfg["lora_alpha"])
    for p in m.vision.parameters():
        p.requires_grad_(cfg["train_vision"])
    for p in m.connector.parameters():
        p.requires_grad_(cfg["train_connector"])
    if cfg["ckpt_blocks"]:
        checkpoint_blocks(m.lm.blocks)
        checkpoint_blocks(m.vision.blocks)
    return m, arch


# ---------------- 批 ----------------
def make_batch(rng, cfg, arch):
    tasks = cfg["tasks"].split(",")
    p = np.array([cfg["read_weight"] if t == "read" else 1.0 for t in tasks])
    exs = [sft_example(rng, tasks[int(rng.choice(len(tasks), p=p / p.sum()))], arch["connector"], cfg["prompt_weight"],
                       cfg["train_eos"], cfg["multi_turn"]) for _ in range(cfg["batch_size"])]
    if cfg["pack"]:
        P = pack([e[1:] for e in exs], cfg["max_len"], PAD)
        imgs = [exs[k][0] for b in P["order"] for k in b]
        return imgs, P["ids"], P["targets"], P["weights"], P["seg"], P["pos"]
    ids, tg, w = pad_weighted([e[1:] for e in exs], PAD)
    return [e[0] for e in exs], ids, tg, w, None, None


def batch_loss(m, batch, cfg, dev):
    imgs, ids, tg, w, seg, pos = batch
    ids, tg, w = ids.to(dev), tg.to(dev), w.to(dev)
    emb = m.embed_list([to_float(torch.from_numpy(x).to(dev)) for x in imgs], ids)
    if seg is None:
        logits, _ = m.lm(inputs_embeds=emb)
    else:
        logits, _ = m.lm(inputs_embeds=emb, pos=pos.to(dev), mask=packed_mask(seg.to(dev)))
    return sft_loss(logits, tg, w, cfg["loss_mode"]), int((w > 0).sum()), ids.numel()


# ---------------- 评测 ----------------
@torch.no_grad()
def answer(m, items, connector, device, max_new=40):
    """items：[(图像 uint8 (k, 3, H, W), 问题)]。按（图的形状, 提示长度）分组贪心生成，再给生成的回答（含 eos）打分。
    返回 [(回答, 回答的对数概率之和, 回答的 token 数)]。"""
    m.eval()
    out = [None] * len(items)
    groups = {}
    for i, (img, q) in enumerate(items):
        p = prompt_ids(q, n_vis(connector, *img.shape[2:]))
        groups.setdefault((img.shape, len(p)), []).append((i, p))
    for (shape, T0), group in groups.items():
        for s in range(0, len(group), 64):
            sub = group[s:s + 64]
            X = to_float(torch.from_numpy(np.stack([items[i][0] for i, _ in sub])).to(device))
            X = X[:, 0] if shape[0] == 1 else X
            P = torch.tensor([p for _, p in sub], device=device)
            gen = m.generate(X, P, max_new, eos=EOS, banned=(PAD, BOS, IMG)).tolist()
            gen = [g[:g.index(EOS) + 1] if EOS in g else g for g in gen]
            L = max(map(len, gen))
            full = torch.full((len(sub), T0 + L), PAD, dtype=torch.long, device=device)
            for b, g in enumerate(gen):
                full[b, :T0] = P[b]
                full[b, T0:T0 + len(g)] = torch.tensor(g, device=device)
            logits, _ = m(X, full)
            lp = seq_logprob(token_logprobs(logits, full),
                             mask_from_lengths([T0] * len(sub), [T0 + len(g) for g in gen], full.shape[1]).to(device))
            for b, ((i, _), g) in enumerate(zip(sub, gen)):
                text = TOK.decode([t for t in g if t != EOS])
                out[i] = (text, float(lp[b]), len(g))
    return out


def evaluate(m, connector, device, n_per_task=50, seed=7, tasks=ALL_TASKS):
    """逐样本结果：dict(task, pred, ref, correct, ed, ref_len, logprob, n_tok)。"""
    data = eval_set(seed, n_per_task, tasks)
    res = answer(m, [(img, q) for _, img, q, _ in data], connector, device)
    return [dict(task=t, pred=p, ref=a, correct=bool(correct(t, p, a)), ed=edit_distance(p, a), ref_len=len(a),
                 logprob=lp, n_tok=n) for (t, _, _, a), (p, lp, n) in zip(data, res)]


def summarize(rows):
    out = {}
    for t in dict.fromkeys(r["task"] for r in rows):
        rs = [r for r in rows if r["task"] == t]
        out[f"acc_{t}"] = float(np.mean([r["correct"] for r in rs]))
        if t == "read":
            out["cer_read"] = sum(r["ed"] for r in rs) / max(1, sum(r["ref_len"] for r in rs))
    out["acc_mean"] = float(np.mean([v for k, v in out.items() if k.startswith("acc_")]))
    return out


def report(rows):
    print(f"{'任务':<12}{'n':>5}{'准确率':>10}   95% 区间")
    for t in dict.fromkeys(r["task"] for r in rows):
        rs = [r for r in rows if r["task"] == t]
        acc, lo, hi = bootstrap_ci([r["correct"] for r in rs])
        print(f"{t:<12}{len(rs):>5}{acc:>10.3f}   [{lo:.3f}, {hi:.3f}]")
        if t == "read":
            c, lo, hi = bootstrap_ci([r["ed"] for r in rs], [r["ref_len"] for r in rs])
            print(f"{'  CER':<12}{'':>5}{c:>10.4f}   [{lo:.4f}, {hi:.4f}]")
    conf = [float(np.exp(r["logprob"] / max(1, r["n_tok"]))) for r in rows]
    ok = [r["correct"] for r in rows]
    print(f"置信度 = exp(每 token 平均对数概率)：ECE {ece(conf, ok):.4f}，"
          f"用它找错例的 AUROC {auroc([-c for c in conf], [not o for o in ok]):.4f}")


# ---------------- 入口 ----------------
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


def train_sft(cfg):
    dev = torch.device(cfg["device"])
    m, arch = prepare(cfg)
    m.to(dev)
    params = [p for p in m.parameters() if p.requires_grad]
    total, trainable = count_params(m)
    print(f"连接器 {arch['connector']}；LoRA r={cfg['lora_r']}；可训练参数 {trainable:,} / {total:,}；"
          f"{'打包' if cfg['pack'] else '补齐'}；损失按 {cfg['loss_mode']} 平均；提示权重 {cfg['prompt_weight']}")
    opt = AdamW(params, lr=cfg["max_lr"], weight_decay=cfg["weight_decay"])
    rng = np.random.default_rng(cfg["seed"])
    run = Run(cfg)
    tasks = tuple(cfg["tasks"].split(","))
    for it in range(cfg["max_iters"]):
        m.train()
        lr = lr_cosine_schedule(it, cfg["max_lr"], cfg["min_lr"], cfg["warmup_iters"], cfg["max_iters"])
        for g in opt.param_groups:
            g["lr"] = lr
        t0 = time.time()
        loss, n_sup, n_tok = batch_loss(m, make_batch(rng, cfg, arch), cfg, dev)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg["clip"]:
            clip_grad_norm(params, cfg["clip"])
        opt.step()
        rec = {"iter": it + 1, "loss": loss.item(), "lr": lr, "ms": round(1000 * (time.time() - t0), 1),
               "tokens": n_tok, "supervised": n_sup}
        if (it + 1) % cfg["eval_every"] == 0 or it + 1 == cfg["max_iters"]:
            rec.update(summarize(evaluate(m, arch["connector"], dev, cfg["eval_n"], tasks=tasks)))
            run.log(rec, show=True)
        else:
            run.log(rec)
    if cfg["lora_r"]:
        merge_lora(m.lm)
    uncheckpoint_blocks(m.lm.blocks)
    uncheckpoint_blocks(m.vision.blocks)
    if run.out:
        torch.save({"vlm": m.state_dict(), "arch": arch, "cfg": cfg}, os.path.join(run.out, "vlm.pt"))
    return m


def memory(cfg):
    m, arch = prepare(dict(cfg, ckpt_blocks=False))
    total, trainable = count_params(m)
    print(f"参数 {total:,}，可训练 {trainable:,}")
    for dt, master in (("fp32", False), ("bf16", True)):
        s = static_memory(total, trainable, dt, "adamw", master)
        print(f"  {dt}{' + fp32 主权重' if master else ''}：" + "，".join(f"{k} {v / 2**20:.2f} MB" for k, v in s.items()))
    rng = np.random.default_rng(0)
    batch = make_batch(rng, cfg, arch)
    m.train()
    for ck in (False, True):
        if ck:
            checkpoint_blocks(m.lm.blocks)
            checkpoint_blocks(m.vision.blocks)
        with SavedTensors(m) as s:
            loss, _, n_tok = batch_loss(m, batch, cfg, torch.device("cpu"))
        loss.backward()
        print(f"  激活值（批 {cfg['batch_size']}，{n_tok} 个位置{'，梯度检查点' if ck else ''}）：{s.bytes / 2**20:.2f} MB")


def compare(a_path, b_path):
    A = [json.loads(x) for x in open(a_path)]
    B = [json.loads(x) for x in open(b_path)]
    assert [(r["task"], r["ref"]) for r in A] == [(r["task"], r["ref"]) for r in B], "两个文件不是同一批样本"
    print(f"{'任务':<12}{'A':>7}{'B':>7}{'A−B':>8}   95% 区间          P(差≤0)  b10  b01  McNemar p")
    for t in list(dict.fromkeys(r["task"] for r in A)) + ["全部"]:
        ia = [i for i, r in enumerate(A) if t in ("全部", r["task"])]
        ca, cb = [A[i]["correct"] for i in ia], [B[i]["correct"] for i in ia]
        d, lo, hi, p_le = paired_bootstrap(ca, cb)
        b10, b01, p = mcnemar(ca, cb)
        print(f"{t:<12}{np.mean(ca):>7.3f}{np.mean(cb):>7.3f}{d:>+8.3f}   [{lo:+.3f}, {hi:+.3f}]   {p_le:>6.3f} {b10:>4} {b01:>4}   {p:.4f}")


def predict(ckpt, out, device="cpu"):
    from assignments.a4_eval_sft.leaderboard import load_test
    m, arch = load_vlm(ckpt)
    m.to(device)
    imgs, _ = load_test()
    items = [(np.repeat(g[None, None], 3, 1), "<image>Read the text.") for g in imgs]
    res = answer(m, items, arch["connector"], torch.device(device))
    with open(out, "w") as f:
        for p, _, _ in res:
            f.write(json.dumps(p) + "\n")
    print(f"{len(res)} 行存到 {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sft", "memory"):
        p = sub.add_parser(name)
        p.add_argument("--set", action="append", default=[])
        p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("eval")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    p = sub.add_parser("predict")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.cmd == "sft":
        train_sft(_cfg(SFT, a.smoke, a.set))
    elif a.cmd == "memory":
        memory(_cfg(SFT, a.smoke, a.set))
    elif a.cmd == "eval":
        m, arch = load_vlm(a.ckpt)
        rows = evaluate(m.to(a.device), arch["connector"], torch.device(a.device), a.n, a.seed)
        report(rows)
        if a.out:
            with open(a.out, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
    elif a.cmd == "compare":
        compare(a.a, a.b)
    else:
        predict(a.ckpt, a.out, a.device)


if __name__ == "__main__":
    main()
