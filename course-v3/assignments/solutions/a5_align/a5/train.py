"""A5 参考实现：同一个起点，四种继续训练的方法（SFT、RFT、DPO、GRPO），可选重放；评测目标任务、旧任务和探测 KL。
在 course-v3/ 下运行（M = assignments.solutions.a5_align.a5.train）：

    python -m $M rl --set init=outputs/a4/full/vlm.pt --set method=grpo --set out_dir=outputs/a5/grpo
    python -m $M rl --set init=... --set method=dpo --set dpo_source=caption --set tasks=caption --set out_dir=...
    python -m $M rl --set init=... --set method=rft --set replay=0.25 --set out_dir=...
    python -m $M eval --ckpt outputs/a5/grpo/vlm.pt --base outputs/a4/full/vlm.pt --out outputs/a5/grpo/eval.jsonl
    python -m $M predict --ckpt outputs/a5/grpo/vlm.pt --out-shapes s.jsonl --out-lines l.jsonl      # 排行榜
所有训练入口都支持 --smoke 和 --set key=value；默认 device=cpu。
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch

from miniocr.data.shapes import describe

from .data import (ALL_TASKS, BOS, EOS, IMG, PAD, TOK, VOCAB_SIZE, correct, convo, encode, eval_set, example, n_vis,
                   prompt_ids, scene, sft_example, to_float)
from .dpo import dpo_loss, pairs_from_groups
from .grpo import grpo_loss, keep_groups, overlong_penalty
from .logprobs import seq_logprob, token_logprobs
from .metrics import edit_distance
from .mix import mixed_batch, probe_kl, rft_select
from .model import GPT, GPTConfig
from .optim import AdamW, clip_grad_norm, lr_cosine_schedule
from .pg import group_advantages, zero_variance_fraction
from .prefs import hallucinated_caption
from .rewards import task_reward, well_formed
from .rollout import make_batch, policy_logprobs, rollout
from .sft import chat_roles, pad_weighted, sft_loss, sft_targets
from .stats import bootstrap_ci, paired_bootstrap
from .vision import ViT, ViTConfig
from .vlm import VLM, MergeProjector, MLPProjector, QFormer

VIT = dict(image_size=64, patch=8, in_ch=3, d=192, n_layer=4, n_head=4, pos="rope2d")
GPT_CFG = dict(vocab_size=VOCAB_SIZE, block_size=320, n_layer=4, n_head=4, d_model=256)
TINY_VIT = dict(VIT, d=32, n_layer=1, n_head=2)
TINY_GPT = dict(GPT_CFG, n_layer=1, n_head=2, d_model=32)
RL = dict(init="", connector="merge", method="grpo", tasks="read,text", replay=0.0, replay_tasks="",
          batch_size=32, n_prompts=8, G=8, temperature=1.0, max_new=40,
          beta=0.0, eps_low=0.2, eps_high=0.2, agg="token", mu=1, dynamic=False, overlong_cache=0,
          dpo_beta=0.1, sft_weight=0.0, dpo_source="online", min_gap=0.0, rft_threshold=1.0, rft_k=1,
          max_lr=2e-5, min_lr=2e-6, warmup_iters=10, max_iters=200, weight_decay=0.0, clip=1.0,
          eval_every=50, eval_n=50, probe_n=8, seed=0, device="cpu", out_dir="")
SMOKE = dict(batch_size=4, n_prompts=2, G=3, max_new=8, max_iters=3, warmup_iters=1, eval_every=3, eval_n=2, probe_n=1)


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
    conn = {"mlp": MLPProjector, "merge": MergeProjector, "qformer": QFormer}[arch["connector"]](arch["vit"]["d"], arch["gpt"]["d_model"])
    return VLM(vit, conn, lm, IMG)


def load_vlm(path):
    """A4/A5 的检查点（带 arch），或 A3 参考实现的检查点。"""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    arch = ck.get("arch")
    if arch is None:
        tiny = ck["cfg"].get("smoke", False)
        arch = dict(vit=TINY_VIT if tiny else VIT, gpt=TINY_GPT if tiny else GPT_CFG, connector=ck["cfg"]["connector"])
    m = build(arch)
    m.load_state_dict(ck["vlm"])
    return m, arch


# ---------------- 数据 ----------------
def items_for(rng, tasks, n):
    """n 道题：[(task, 图像 uint8 (k, 3, H, W), 问题, 答案)]，任务均匀抽。"""
    out = []
    for _ in range(n):
        t = tasks[int(rng.integers(len(tasks)))]
        img, pairs = example(t, rng)
        out.append((t, img, pairs[0][0], pairs[0][1]))
    return out


def sft_seq(img, q, a, connector):
    ids, roles = chat_roles(convo([(q, a)]), n_vis(connector, *img.shape[2:]), encode, BOS, EOS, IMG)
    tg, w = sft_targets(ids, roles, 0.0, eos_id=EOS)
    return img, ids, tg, w


def sft_batch_loss(m, exs, dev):
    """exs：[(图像 uint8, ids, targets, weights)] → token 平均的 SFT 损失。"""
    ids, tg, w = pad_weighted([e[1:] for e in exs], PAD)
    emb = m.embed_list([to_float(torch.from_numpy(e[0]).to(dev)) for e in exs], ids.to(dev))
    logits, _ = m.lm(inputs_embeds=emb)
    return sft_loss(logits, tg.to(dev), w.to(dev), "token")


def imgs_float(imgs, dev):
    return [to_float(torch.from_numpy(x).to(dev)) for x in imgs]


# ---------------- 四种方法的一步 ----------------
class Trainer:
    def __init__(self, cfg):
        self.cfg, self.dev = cfg, torch.device(cfg["device"])
        torch.manual_seed(cfg["seed"])
        if cfg["init"]:
            self.m, self.arch = load_vlm(cfg["init"])
        else:
            self.arch = dict(vit=TINY_VIT if cfg["smoke"] else VIT, gpt=TINY_GPT if cfg["smoke"] else GPT_CFG, connector=cfg["connector"])
            self.m = build(self.arch)
        self.m.to(self.dev)
        for p in self.m.vision.parameters():
            p.requires_grad_(False)
        self.ref = copy.deepcopy(self.m).eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)
        self.params = [p for p in self.m.parameters() if p.requires_grad]
        self.opt = AdamW(self.params, lr=cfg["max_lr"], weight_decay=cfg["weight_decay"])
        self.rng = np.random.default_rng(cfg["seed"])
        self.gen = torch.Generator(device=self.dev).manual_seed(cfg["seed"])
        self.tasks = cfg["tasks"].split(",")
        rt = cfg["replay_tasks"].split(",") if cfg["replay_tasks"] else [t for t in ALL_TASKS if t not in self.tasks]
        self.replay_tasks = rt
        self.conn = self.arch["connector"]

    def _update(self, loss):
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg["clip"]:
            clip_grad_norm(self.params, self.cfg["clip"])
        self.opt.step()

    def _replay_loss(self, n):
        if n == 0:
            return 0.0
        exs = [sft_example(self.rng, self.replay_tasks[int(self.rng.integers(len(self.replay_tasks)))], self.conn)
               for _ in range(n)]
        return sft_batch_loss(self.m, exs, self.dev)

    def _sample(self):
        c = self.cfg
        items = items_for(self.rng, self.tasks, c["n_prompts"])
        prompts = [prompt_ids(q, n_vis(self.conn, *img.shape[2:])) for _, img, q, _ in items]
        resp, fin = rollout(self.m, [(img, p) for (_, img, _, _), p in zip(items, prompts)], c["G"], c["max_new"], EOS,
                            c["temperature"], self.gen, self.dev, banned=(PAD, BOS, IMG))
        texts = [TOK.decode(r) for r in resp]
        rewards = []
        for j, t in enumerate(texts):
            task, _, _, ref = items[j // c["G"]]
            r = task_reward(task, t, ref)
            if c["overlong_cache"]:
                r += overlong_penalty(len(resp[j]) + fin[j], c["max_new"], c["overlong_cache"])
            rewards.append(r)
        return items, prompts, resp, fin, texts, torch.tensor(rewards)

    def step_sft(self):
        c = self.cfg
        make_new = lambda rng: sft_example(rng, self.tasks[int(rng.integers(len(self.tasks)))], self.conn)          # noqa: E731
        make_old = lambda rng: sft_example(rng, self.replay_tasks[int(rng.integers(len(self.replay_tasks)))], self.conn)  # noqa: E731
        exs, _ = mixed_batch(self.rng, c["batch_size"], c["replay"], make_new, make_old)
        loss = sft_batch_loss(self.m, exs, self.dev)
        self._update(loss)
        return {"loss": loss.item()}

    def step_rft(self):
        c = self.cfg
        items, _, resp, fin, texts, rewards = self._sample()
        sel = rft_select(rewards, c["G"], c["rft_threshold"], c["rft_k"])
        exs = [sft_seq(items[j // c["G"]][1], items[j // c["G"]][2], texts[j], self.conn) for j in sel if fin[j]]
        n_old = int(round(c["replay"] * c["batch_size"]))
        loss = (sft_batch_loss(self.m, exs, self.dev) if exs else 0.0) + self._replay_loss(n_old)
        if torch.is_tensor(loss):
            self._update(loss)
        return {"loss": float(loss.detach()) if torch.is_tensor(loss) else loss, "reward": rewards.mean().item(), "n_selected": len(exs)}

    def _seq_logps(self, model, imgs, ids, mask):
        return seq_logprob(policy_logprobs(model, imgs_float(imgs, self.dev), ids.to(self.dev)), mask.to(self.dev))

    def step_dpo(self):
        c = self.cfg
        if c["dpo_source"] == "caption":
            imgs, prompts, chosen, rejected = [], [], [], []
            for _ in range(c["n_prompts"]):
                img, objs, _ = scene(self.rng)
                imgs.append(img[None])
                prompts.append(prompt_ids("<image>Describe the image.", n_vis(self.conn, 64, 64)))
                chosen.append(encode(describe(objs)))
                rejected.append(encode(hallucinated_caption(objs, self.rng, ("substitute", "add")[int(self.rng.integers(2))])[0]))
            reward = float("nan")
        else:
            items, prompts_all, resp, fin, _, rewards = self._sample()
            pairs = [(w, l) for w, l in pairs_from_groups(rewards, c["G"], c["min_gap"]) if fin[w]]
            if not pairs:
                return {"reward": rewards.mean().item(), "n_pairs": 0}
            imgs = [items[w // c["G"]][1] for w, _ in pairs]
            prompts = [prompts_all[w // c["G"]] for w, _ in pairs]
            chosen, rejected = [resp[w] for w, _ in pairs], [resp[l] for _, l in pairs]
            reward = rewards.mean().item()
        n = len(prompts)
        ids, mask = make_batch(prompts + prompts, chosen + rejected, [True] * (2 * n), EOS, PAD)
        both = imgs + imgs
        pol = self._seq_logps(self.m, both, ids, mask)
        with torch.no_grad():
            ref = self._seq_logps(self.ref, both, ids, mask)
        loss, st = dpo_loss(pol[:n], pol[n:], ref[:n], ref[n:], c["dpo_beta"], c["sft_weight"], mask[:n].sum(1).to(self.dev))
        loss = loss + self._replay_loss(int(round(c["replay"] * c["batch_size"])))
        self._update(loss)
        return dict(loss=loss.item(), reward=reward, n_pairs=n, chosen_logp=pol[:n].mean().item(), **st)

    def step_grpo(self):
        c = self.cfg
        items, prompts, resp, fin, texts, rewards = self._sample()
        G = c["G"]
        rec = {"reward": rewards.mean().item(), "zero_var": zero_variance_fraction(rewards, G),
               "format_ok": float(np.mean([well_formed(items[j // G][0], t) for j, t in enumerate(texts)])),
               "resp_len": float(np.mean([len(r) for r in resp]))}
        adv = group_advantages(rewards, G)
        keep = torch.ones(len(items), dtype=torch.bool)
        if c["dynamic"]:
            keep = keep_groups(rewards, G)
            if not keep.any():
                return rec
        rows = [j for j in range(len(resp)) if keep[j // G]]
        imgs = [items[j // G][1] for j in rows]
        ids, mask = make_batch([prompts[j // G] for j in rows], [resp[j] for j in rows], [fin[j] for j in rows], EOS, PAD)
        ids, mask, adv = ids.to(self.dev), mask.to(self.dev), adv[rows].to(self.dev)
        with torch.no_grad():
            old = policy_logprobs(self.m, imgs_float(imgs, self.dev), ids)
            ref = policy_logprobs(self.ref, imgs_float(imgs, self.dev), ids) if c["beta"] else None
        for _ in range(c["mu"]):
            logp = policy_logprobs(self.m, imgs_float(imgs, self.dev), ids)
            loss, st = grpo_loss(logp, old, adv, mask, ref, c["beta"], c["eps_low"], c["eps_high"], c["agg"], c["max_new"] + 1)
            loss = loss + self._replay_loss(int(round(c["replay"] * c["batch_size"])))
            self._update(loss)
        return dict(rec, loss=loss.item(), **st)


# ---------------- 评测 ----------------
@torch.no_grad()
def answer(m, items, connector, device, max_new=40):
    """贪心。items：[(图像 uint8, 问题)] → 回答字符串列表。"""
    m.eval()
    out = [None] * len(items)
    groups = {}
    for i, (img, q) in enumerate(items):
        p = prompt_ids(q, n_vis(connector, *img.shape[2:]))
        groups.setdefault((img.shape, len(p)), []).append((i, p))
    for (shape, _), group in groups.items():
        for s in range(0, len(group), 64):
            sub = group[s:s + 64]
            X = to_float(torch.from_numpy(np.stack([items[i][0] for i, _ in sub])).to(device))
            X = X[:, 0] if shape[0] == 1 else X
            gen = m.generate(X, torch.tensor([p for _, p in sub], device=device), max_new, eos=EOS,
                             banned=(PAD, BOS, IMG)).tolist()
            for (i, _), g in zip(sub, gen):
                out[i] = TOK.decode(g[:g.index(EOS)] if EOS in g else g)
    return out


def evaluate(m, connector, device, n_per_task=50, seed=7):
    data = eval_set(seed, n_per_task)
    preds = answer(m, [(img, q) for _, img, q, _ in data], connector, device)
    return [dict(task=t, pred=p, ref=a, correct=bool(correct(t, p, a)), reward=task_reward(t, p, a),
                 ed=edit_distance(p, a), ref_len=len(a)) for (t, _, _, a), p in zip(data, preds)]


@torch.no_grad()
def probe(m, base, connector, device, n_per_task=8, seed=11):
    """探测 KL：每个任务 n 条，真值答案当回答，回答位置上 KL(p_base ‖ p_new) 的平均。"""
    data = eval_set(seed, n_per_task)
    exs = [sft_seq(img, q, a, connector) for _, img, q, a in data]
    ids, tg, _ = pad_weighted([e[1:] for e in exs], PAD)
    imgs = [to_float(torch.from_numpy(e[0]).to(device)) for e in exs]
    ids = ids.to(device)
    lg_new = m.lm(inputs_embeds=m.embed_list(imgs, ids))[0]
    lg_base = base.lm(inputs_embeds=base.embed_list(imgs, ids))[0]
    return probe_kl(lg_new, lg_base, (tg != -100).to(device))


def summarize(rows, targets):
    out = {}
    for t in dict.fromkeys(r["task"] for r in rows):
        out[f"acc_{t}"] = float(np.mean([r["correct"] for r in rows if r["task"] == t]))
    out["acc_target"] = float(np.mean([r["correct"] for r in rows if r["task"] in targets]))
    out["acc_others"] = float(np.mean([r["correct"] for r in rows if r["task"] not in targets]))
    rs = [r for r in rows if r["task"] == "read"]
    if rs:
        out["cer_read"] = sum(r["ed"] for r in rs) / max(1, sum(r["ref_len"] for r in rs))
    return out


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


def train_rl(cfg):
    tr = Trainer(cfg)
    step = getattr(tr, "step_" + cfg["method"])
    run = Run(cfg)
    print(f"方法 {cfg['method']}；目标任务 {tr.tasks}；重放 {cfg['replay']}（{tr.replay_tasks}）；"
          f"可训练参数 {sum(p.numel() for p in tr.params):,}")
    for it in range(cfg["max_iters"]):
        tr.m.train()
        lr = lr_cosine_schedule(it, cfg["max_lr"], cfg["min_lr"], cfg["warmup_iters"], cfg["max_iters"])
        for g in tr.opt.param_groups:
            g["lr"] = lr
        t0 = time.time()
        rec = dict(iter=it + 1, lr=lr, **step())
        rec["ms"] = round(1000 * (time.time() - t0), 1)
        if (it + 1) % cfg["eval_every"] == 0 or it + 1 == cfg["max_iters"]:
            rec.update(summarize(evaluate(tr.m, tr.conn, tr.dev, cfg["eval_n"]), tr.tasks))
            rec["probe_kl"] = probe(tr.m, tr.ref, tr.conn, tr.dev, cfg["probe_n"])
            run.log(rec, show=True)
        else:
            run.log(rec)
    if run.out:
        torch.save({"vlm": tr.m.state_dict(), "arch": tr.arch, "cfg": cfg}, os.path.join(run.out, "vlm.pt"))
    return tr


def report(rows, base_rows=None):
    print(f"{'任务':<12}{'准确率':>8}   95% 区间" + ("          相对起点的差   95% 区间" if base_rows else ""))
    tasks = list(dict.fromkeys(r["task"] for r in rows))
    for t in tasks + ["全部"]:
        ia = [i for i, r in enumerate(rows) if t in ("全部", r["task"])]
        c = [rows[i]["correct"] for i in ia]
        acc, lo, hi = bootstrap_ci(c)
        line = f"{t:<12}{acc:>8.3f}   [{lo:.3f}, {hi:.3f}]"
        if base_rows:
            d, dlo, dhi, _ = paired_bootstrap(c, [base_rows[i]["correct"] for i in ia])
            line += f"   {d:+.3f}   [{dlo:+.3f}, {dhi:+.3f}]"
        print(line)


def predict(ckpt, out_shapes, out_lines, device="cpu"):
    """两个排行榜测试集：A3 的 ShapesWorld 问答（540 题）和 A4 的文字行（500 行）。"""
    from assignments.a3_vlm.leaderboard import load_test as shapes_test
    from assignments.a4_eval_sft.leaderboard import load_test as lines_test
    m, arch = load_vlm(ckpt)
    m.to(device)
    dev = torch.device(device)
    s = answer(m, [(img, q) for _, img, q, _ in shapes_test()], arch["connector"], dev)
    imgs, _ = lines_test()
    l = answer(m, [(np.repeat(g[None, None], 3, 1), "<image>Read the text.") for g in imgs], arch["connector"], dev)
    for path, preds in ((out_shapes, s), (out_lines, l)):
        with open(path, "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
    print(f"{len(s)} 个回答存到 {out_shapes}，{len(l)} 行存到 {out_lines}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("rl")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("eval")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--base", default="")
    p.add_argument("--out", default="")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("predict")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-shapes", required=True)
    p.add_argument("--out-lines", required=True)
    p.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.cmd == "rl":
        train_rl(_cfg(RL, a.smoke, a.set))
    elif a.cmd == "eval":
        dev = torch.device(a.device)
        m, arch = load_vlm(a.ckpt)
        rows = evaluate(m.to(dev), arch["connector"], dev, a.n)
        base_rows = None
        if a.base:
            b, _ = load_vlm(a.base)
            b.to(dev)
            base_rows = evaluate(b, arch["connector"], dev, a.n)
            print(f"探测 KL（相对 {a.base}）：{probe(m, b, arch['connector'], dev):.4f}")
        report(rows, base_rows)
        if a.out:
            with open(a.out, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
    else:
        predict(a.ckpt, a.out_shapes, a.out_lines, a.device)


if __name__ == "__main__":
    main()
