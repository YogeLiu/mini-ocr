"""A6 的固定训练配方（基础设施：不是你的代码，**不要改**）。和 DataComp 一样：模型、步数、优化器全部固定，
你唯一能控制的是**训练数据**——从数据池里挑哪些图、配什么标签。

模型：8×8 的 patch（48×192 的图 → 6×24 = 144 个视觉 token）+ Transformer 编码器 + 带交叉注意力的 Transformer 解码器，
逐记号输出 LaTeX（记号见 VOCAB；不认识的记号变成 <unk>）。每步从你的数据里有放回地均匀抽一批（重复的条目就等于加权）。
这里用 torch.nn 的现成模块：它不是作业代码，第 1 节的限制不适用于它。

在 course-v3/ 下运行（F = assignments.a6_data.fixed_train）：

    python -m $F train --data 你的数据.jsonl --track gpu --out outputs/a6/run1      # 一张 24 GB 的卡
    python -m $F train --data 你的数据.jsonl --track cpu --out outputs/a6/run1      # CPU 赛道（小模型、少步数）
    python -m $F train --data ... --smoke                                          # 几步，只检查能跑
    python -m $F oracle --out outputs/a6/oracle.jsonl                               # 只给实验当上限：真值标签
    python -m $F predict --ckpt outputs/a6/run1/model.pt --indices idx.txt --out p.jsonl            # 模型当"第 7 个专家"
    python -m $F predict --ckpt ... --indices idx.txt --G 8 --temperature 1.0 --out s.jsonl          # 每张图采 G 个（自洽度）

数据文件：每行一个 JSON：{"i": 数据池的下标, "label": "LaTeX 字符串"}。同一个下标可以出现多次。
train 结束后在 --out 里留下：config.json、log.jsonl、model.pt、test_preds.jsonl、result.json（测试集结果）。
"""
import argparse
import hashlib
import json
import math
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn

from miniocr.data.latex import LatexError, normalize

from . import pool as P

SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
VOCAB = SPECIALS + list("xyzabcnk0123456789()+-={}^_") + ["\\frac", "\\sqrt", "\\over"]
STOI = {t: i for i, t in enumerate(VOCAB)}
PAD, BOS, EOS, UNK = range(4)
MAX_LEN = 64
TRACKS = {
    "gpu": dict(d=256, enc=4, dec=3, heads=4, batch=128, steps=4000, lr=5e-4, warmup=200),
    "cpu": dict(d=128, enc=3, dec=2, heads=4, batch=64, steps=3000, lr=1e-3, warmup=100),
}
SMOKE = dict(d=32, enc=1, dec=1, heads=2, batch=8, steps=3, lr=1e-3, warmup=1)
_TOK = re.compile(r"\\[A-Za-z]+|\S")


def encode(label):
    return [STOI.get(t, UNK) for t in _TOK.findall(label)][:MAX_LEN - 1]


def decode(ids):
    out, prev = "", ""
    for i in ids:
        if i == EOS:
            break
        t = VOCAB[i] if i >= 4 else ""
        if prev.startswith("\\") and t[:1].isalpha():
            out += " "
        out += t
        prev = t or prev
    return out


def canon(s):
    """评测用的规范化：能解析就写成规范写法，否则去掉空格。"""
    try:
        return normalize(s)
    except LatexError:
        return s.replace(" ", "")


def check_prereq():
    """评分要用 miniocr.data.latex 的 normalize（练习 3.4、3.5）。没做完或有错时，分数会不可信，直接停下。"""
    cases = {"x^2": "x^{2}", "{a \\over b}": "\\frac{a}{b}", "x^{2}_{n}": "x_{n}^{2}", "\\sqrt{ {a+b} }": "\\sqrt{a+b}"}
    try:
        ok = all(normalize(k) == v for k, v in cases.items())
    except (NotImplementedError, LatexError):
        ok = False
    if not ok:
        raise SystemExit("miniocr.data.latex 的 parse / normalize（练习 3.4、3.5）还没做完或有错：先让 labs/ch03 的测试通过。")


class Model(nn.Module):
    def __init__(self, d, enc, dec, heads, **_):
        super().__init__()
        self.patch = nn.Conv2d(1, d, 8, 8)
        self.pos_img = nn.Parameter(torch.randn(1, (P.H // 8) * (P.W // 8), d) * 0.02)
        layer = dict(d_model=d, nhead=heads, dim_feedforward=4 * d, dropout=0.0, batch_first=True, norm_first=True,
                     activation="gelu")
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(**layer), enc, enable_nested_tensor=False)
        self.tok = nn.Embedding(len(VOCAB), d)
        self.pos_txt = nn.Parameter(torch.randn(1, MAX_LEN + 1, d) * 0.02)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(**layer), dec)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, len(VOCAB))

    def encode(self, x):
        """x (B, 48, 192) uint8 → 记忆 (B, 144, d)。"""
        x = x.float().unsqueeze(1) / 127.5 - 1.0
        return self.enc(self.patch(x).flatten(2).transpose(1, 2) + self.pos_img)

    def forward(self, mem, ids):
        T = ids.shape[1]
        causal = torch.triu(torch.full((T, T), float("-inf"), device=ids.device), 1)
        h = self.dec(self.tok(ids) + self.pos_txt[:, :T], mem, tgt_mask=causal)
        return self.head(self.norm(h))


def load_data(path):
    rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    N = len(P.load_pool()["outputs"])
    bad = [r for r in rows if not (isinstance(r.get("i"), int) and 0 <= r["i"] < N and isinstance(r.get("label"), str))]
    if bad:
        raise ValueError(f"{len(bad)} 行格式不对，例如 {bad[0]}")
    return rows


def _batch(rows, idx, images, dev):
    seqs = [[BOS] + encode(rows[k]["label"]) + [EOS] for k in idx]
    T = max(map(len, seqs))
    ids = torch.full((len(seqs), T), PAD, dtype=torch.long)
    for b, s in enumerate(seqs):
        ids[b, :len(s)] = torch.tensor(s)
    x = torch.from_numpy(images[[rows[k]["i"] for k in idx]])
    return x.to(dev), ids.to(dev)


@torch.no_grad()
def generate(model, images, dev, temperature=0.0, generator=None, bs=250):
    """images (N, 48, 192) uint8 → N 个字符串。temperature 0 = 贪心。"""
    model.eval()
    out = []
    for s in range(0, len(images), bs):
        mem = model.encode(torch.from_numpy(images[s:s + bs]).to(dev))
        ids = torch.full((mem.shape[0], 1), BOS, dtype=torch.long, device=dev)
        done = torch.zeros(mem.shape[0], dtype=torch.bool, device=dev)
        for _ in range(MAX_LEN):
            logits = model(mem, ids)[:, -1].float()
            logits[:, [PAD, BOS]] = float("-inf")
            if temperature == 0:
                nxt = logits.argmax(-1)
            else:
                nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1, generator=generator)[:, 0]
            ids = torch.cat([ids, nxt[:, None]], 1)
            done |= nxt == EOS
            if done.all():
                break
        out += [decode(r[1:]) for r in ids.tolist()]
    return out


def _ned(a, b):
    if not a and not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(len(a), len(b))


def score(preds, labels, levels=None, n_boot=2000):
    """规范化之后的整式全对率（附 95% bootstrap 区间）与平均 1 − NED。"""
    ok = np.array([canon(p) == canon(r) for p, r in zip(preds, labels)], dtype=float)
    sim = np.array([1 - _ned(canon(p), canon(r)) for p, r in zip(preds, labels)])
    idx = np.random.default_rng(0).integers(0, len(ok), size=(n_boot, len(ok)))
    lo, hi = np.quantile(ok[idx].mean(1), [0.025, 0.975])
    res = dict(exact=float(ok.mean()), lo=float(lo), hi=float(hi), sim=float(sim.mean()))
    if levels is not None:
        res["by_level"] = {int(lv): float(ok[levels == lv].mean()) for lv in sorted(set(levels.tolist()))}
    return res


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def train(data, track, out, device, smoke=False, seed=0):
    cfg = dict(SMOKE if smoke else TRACKS[track], track=track, data=data, data_sha=_sha(data), recipe_sha=_sha(__file__),
               seed=seed, device=device, smoke=smoke)
    check_prereq()
    rows = load_data(data)
    images = P.load_pool()["images"]
    n_unk = sum(t == UNK for r in rows for t in encode(r["label"]))
    print(f"{len(rows):,} 条训练数据（{len({r['i'] for r in rows}):,} 张不同的图，<unk> {n_unk} 个）；赛道 {track}；{cfg['steps']} 步 × {cfg['batch']}")
    torch.manual_seed(seed)
    dev = torch.device(device)
    model = Model(**cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.05, betas=(0.9, 0.98))
    gen = torch.Generator().manual_seed(seed)
    os.makedirs(out, exist_ok=True)
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    log = open(os.path.join(out, "log.jsonl"), "w")
    use_amp = dev.type == "cuda"
    t0 = time.time()
    for step in range(cfg["steps"]):
        lr = cfg["lr"] * min(1.0, (step + 1) / cfg["warmup"]) * 0.5 * (1 + math.cos(math.pi * step / cfg["steps"]))
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        idx = torch.randint(len(rows), (cfg["batch"],), generator=gen).tolist()
        x, ids = _batch(rows, idx, images, dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(model.encode(x), ids[:, :-1])
        loss = nn.functional.cross_entropy(logits.float().reshape(-1, len(VOCAB)), ids[:, 1:].reshape(-1), ignore_index=PAD)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 50 == 0 or step + 1 == cfg["steps"]:
            rec = dict(step=step + 1, loss=round(loss.item(), 4), lr=lr, elapsed=round(time.time() - t0, 1))
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if (step + 1) % 500 == 0 or step + 1 == cfg["steps"]:
                print(f"  步 {rec['step']}  损失 {rec['loss']:.4f}  {rec['elapsed']:.0f} 秒")
    train_s = time.time() - t0
    torch.save({"model": model.state_dict(), "cfg": cfg}, os.path.join(out, "model.pt"))
    test = P.load_test()
    n = 20 if smoke else len(test["labels"])
    preds = generate(model, test["images"][:n], dev)
    with open(os.path.join(out, "test_preds.jsonl"), "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    res = dict(score(preds, test["labels"][:n], test["levels"][:n]), train_seconds=round(train_s, 1), n_rows=len(rows),
               data_sha=cfg["data_sha"], recipe_sha=cfg["recipe_sha"], track=track, smoke=smoke)
    json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1)
    print(f"测试集整式全对 {res['exact']:.4f} [{res['lo']:.4f}, {res['hi']:.4f}]；平均 1 − NED {res['sim']:.4f}；"
          f"训练 {train_s:.0f} 秒；按难度 {res.get('by_level')}")
    return res


def load_model(ckpt, device):
    sd = torch.load(ckpt, map_location=device)
    m = Model(**sd["cfg"]).to(device)
    m.load_state_dict(sd["model"])
    return m


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--data", required=True)
    p.add_argument("--track", choices=list(TRACKS), default="cpu")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="")
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("oracle")
    p.add_argument("--out", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--indices", required=True, help="每行一个数据池下标")
    p.add_argument("--out", required=True)
    p.add_argument("--G", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.cmd == "train":
        dev = a.device or ("cuda" if a.track == "gpu" and torch.cuda.is_available() else "cpu")
        train(a.data, a.track, a.out, dev, a.smoke)
    elif a.cmd == "oracle":
        truth, bad = P._truth(), set(P.planted())
        with open(a.out, "w") as f:
            for i, t in enumerate(truth):
                if i not in bad:
                    f.write(json.dumps({"i": i, "label": t}) + "\n")
        print(f"真值标签（去掉了污染的 {len(bad)} 条）写进 {a.out}。只用来给实验当上限，不能提交排行榜。")
    else:
        idx = [int(x) for x in open(a.indices) if x.strip()]
        m = load_model(a.ckpt, a.device)
        imgs = P.load_pool()["images"][idx]
        gen = torch.Generator(device=a.device).manual_seed(0)
        with open(a.out, "w") as f:
            samples = [generate(m, imgs, a.device, a.temperature, gen) for _ in range(a.G)]
            for k, i in enumerate(idx):
                f.write(json.dumps({"i": i, "preds": [s[k] for s in samples]}) + "\n")
        print(f"{len(idx)} 张图 × {a.G} 个回答写进 {a.out}")


if __name__ == "__main__":
    main()
