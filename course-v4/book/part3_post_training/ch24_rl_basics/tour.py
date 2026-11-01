"""第 24 章：1 学习曲线，2 梯度方差，3 KL，4 PPO，5 小 GPT 序列奖励。

默认只跑 1–4；第 5 部分显式选择，无需下载模型或数据。
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from miniocr.device import pick_device, synchronize
from miniocr.models.gpt import GPT, GPTConfig
from miniocr.train.policy_gradient import (
    categorical_score_gradient, kl_estimators, ppo_surrogate,
    reinforce_loss, sequence_log_prob,
)
from miniocr.train.pretrain import wilson_interval


class Recorder:
    """统一 JSONL stdout 与可选 JSON/CSV/PNG 产物，不隐式写缓存。"""
    def __init__(self, args):
        self.args = args
        self.rows = []
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)

    def emit(self, event, **values):
        row = dict(source="本项目观察", event=event, device=self.args.device,
                   dtype="fp32", **values)
        self.rows.append(row)
        print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
        if self.args.out:
            with (self.args.out / "events.jsonl").open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def table(self, name, rows):
        if self.args.out:
            with (self.args.out / f"{name}.csv").open("w") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    def plot(self, name, lines, ylabel):
        if not self.args.out:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for label, x, y, std in lines:
            ax.plot(x, y, label=label)
            if std is not None:
                ax.fill_between(x, [a-b for a, b in zip(y, std)],
                                [a+b for a, b in zip(y, std)], alpha=.15)
        ax.set(xlabel="step", ylabel=ylabel)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.args.out / f"{name}.png", dpi=150)
        plt.close(fig)


def part1(a, rec):
    means = torch.tensor([1., 1.2, 1.5, 2.], device=a.device)
    # 每个种子独立 CPU generator；三个方法配对复用均匀数和奖励噪声。
    us, zs = [], []
    for seed in range(a.seed, a.seed+a.seeds):
        gen = torch.Generator().manual_seed(seed)
        us.append(torch.rand(a.steps, a.batch_size, generator=gen))
        zs.append(torch.randn(a.steps, a.batch_size, generator=gen))
    u, z = torch.stack(us, 1).to(a.device), torch.stack(zs, 1).to(a.device)
    rows, lines, finals = [], [], {}
    for method in ("none", "oracle", "ema"):
        logits = torch.zeros(a.seeds, 4, device=a.device, requires_grad=True)
        opt = torch.optim.SGD([logits], lr=a.lr)
        baseline = torch.zeros(a.seeds, device=a.device)
        xs, ys, stds = [], [], []
        for step in range(a.steps+1):
            probs = logits.softmax(-1)
            expected = (probs * means).sum(-1)
            if step % a.log_every == 0 or step == a.steps:
                vals = expected.detach().cpu()
                for seed, val, best in zip(range(a.seed, a.seed+a.seeds), vals,
                                            probs[:, -1].detach().cpu()):
                    rows.append(dict(method=method, seed=seed, step=step,
                                     expected_reward=val.item(), best_arm_prob=best.item()))
                xs.append(step)
                ys.append(vals.mean().item())
                stds.append(vals.std(unbiased=a.seeds > 1).item())
            if step == a.steps:
                finals[method] = vals
                rec.emit("bandit_final", method=method, seeds=a.seeds, steps=a.steps,
                         batch_size=a.batch_size, expected_reward_mean=ys[-1],
                         expected_reward_seed_sd=stds[-1], best_arm_prob=probs[:, -1].mean().item())
                break
            actions = (u[step, :, :, None] > probs.detach().cumsum(-1)[:, None, :]).sum(-1).clamp_max(3)
            rewards = means[actions] + z[step]
            b = expected.detach() if method == "oracle" else baseline if method == "ema" else torch.zeros_like(baseline)
            logp = logits.log_softmax(-1).gather(1, actions)
            # mean over batches, sum independent seed losses: identical lr for every seed.
            loss = reinforce_loss(logp, rewards, b[:, None]) * a.seeds
            opt.zero_grad()
            loss.backward()
            opt.step()
            if method == "ema":
                baseline = .9*baseline + .1*rewards.mean(-1)
        lines.append((method, xs, ys, stds))
    rec.table("bandit_curves", rows)
    rec.plot("bandit_curves", lines, "expected reward (shade = seed SD)")
    rec.emit("bandit_design", means=means.tolist(), noise_sd=1., lr=a.lr,
             paired_oracle_minus_none=(finals["oracle"]-finals["none"]).mean().item(),
             baseline="oracle=当前策略的解析期望奖励；ema=此前批次奖励均值的 EMA")


def part2(a, rec):
    p = torch.tensor([.1, .2, .3, .4], device=a.device)
    means = torch.tensor([1., 1.2, 1.5, 2.], device=a.device)
    actions = torch.multinomial(p, a.repeats*a.mc_batch, replacement=True).view(a.repeats, a.mc_batch)
    rewards = means[actions] + torch.randn(actions.shape, device=a.device)
    exact_grad = p*(means-(p*means).sum())
    score = torch.eye(4, device=a.device)-p
    results = {}
    rows = []
    for name, b in (("none", 0.), ("oracle", (p*means).sum().item())):
        gradients = categorical_score_gradient(p, actions, rewards, b).mean(1)
        var = gradients.var(0, unbiased=True).sum().item()
        exact_var = ((p*((means-b).square()+1)*score.square().sum(-1)).sum()-exact_grad.square().sum())/a.mc_batch
        row = dict(method=name, baseline=b, empirical_trace_variance=var,
                   exact_trace_variance=exact_var.item(), gradient_mean=gradients.mean(0).tolist(),
                   exact_gradient=exact_grad.tolist(), max_abs_bias=(gradients.mean(0)-exact_grad).abs().max().item(),
                   mean_se_max=(gradients.var(0)/a.repeats).sqrt().max().item())
        rows.append(row)
        results[name] = var
        rec.emit("gradient_variance", repeats=a.repeats, mc_batch=a.mc_batch, **row)
    rec.emit("variance_ratio", oracle_over_none=results["oracle"]/results["none"])
    rec.table("gradient_variance", rows)


def part3(a, rec):
    p = torch.tensor([.1, .2, .3, .4], device=a.device)
    rows = []
    for case, q in (("near", [.11, .19, .29, .41]), ("far", [.4, .3, .2, .1])):
        q = torch.tensor(q, device=a.device)
        truth = (p*(p.log()-q.log())).sum()
        samples = torch.multinomial(p, a.repeats*a.mc_batch, replacement=True).view(a.repeats, a.mc_batch)
        for name, values in kl_estimators(q.log()-p.log()).items():
            exact_mean = (p*values).sum()
            exact_var = (p*(values-exact_mean).square()).sum()/a.mc_batch
            estimates = values[samples].mean(-1)
            row = dict(case=case, estimator=name, true_kl=truth.item(),
                       exact_mean=exact_mean.item(), exact_bias=(exact_mean-truth).item(),
                       exact_batch_variance=exact_var.item(), mc_mean=estimates.mean().item(),
                       mc_bias=(estimates.mean()-truth).item(), mc_batch_variance=estimates.var().item(),
                       mc_mean_se=(estimates.var()/a.repeats).sqrt().item())
            rows.append(row)
            rec.emit("kl", repeats=a.repeats, mc_batch=a.mc_batch, **row)
    rec.table("kl", rows)


def part4(a, rec):
    rows = []
    for advantage in (1., -1.):
        ratio = torch.tensor([.5, .9, 1., 1.1, 1.5], device=a.device, requires_grad=True)
        objective = ppo_surrogate(ratio, torch.full_like(ratio, advantage), .2)
        grad, = torch.autograd.grad(objective.sum(), ratio)
        for r, obj, g in zip(ratio.tolist(), objective.tolist(), grad.tolist()):
            row = dict(advantage=advantage, ratio=r, objective=obj, derivative_wrt_ratio=g)
            rows.append(row)
            rec.emit("ppo", **row)
    rec.table("ppo", rows)


VOCAB = "$xyz+0123"


def prompt_pool(split):
    """受控格式任务：全部 8^4 个非 $ 提示，经 SHA256 固定分组划分。

    此任务需要精确控制稀疏奖励，故用符号串而非图片或 OCR 数据。
    不使用 latex-v1，不声称测量后纠错能力。三个划分提示无重叠。
    """
    pools = {"train": [], "val": [], "test": []}
    for i in range(8**4):
        ids = [(i // 8**j) % 8 + 1 for j in range(4)]
        bucket = int(hashlib.sha256(bytes(ids)).hexdigest()[:8], 16) % 10
        pools["val" if bucket == 0 else "test" if bucket == 1 else "train"].append(ids)
    return torch.tensor(pools[split])


def format_reward(response):
    """终局奖励：首尾是 $ 各得 0.5；双端成功率另报，内容正确性不在奖励中。"""
    return .5*((response[:, 0] == 0).float()+(response[:, -1] == 0).float())


@torch.no_grad()
def rollout(model, prompts, length):
    ids = prompts
    for _ in range(length):
        logits, _ = model(ids)
        token = torch.multinomial(logits[:, -1].float().softmax(-1), 1)
        ids = torch.cat([ids, token], 1)
    return ids


def eval_sequence(model, a, split):
    pool = prompt_pool(split)
    if a.n > len(pool):
        raise ValueError(f"n={a.n} 超过 {split} 提示池 {len(pool)}")
    rewards, successes, dollars = [], [], []
    # 独立 fork RNG，不改变训练随机流；val 各检查点复用随机种子。
    devices = [torch.cuda.current_device()] if a.device == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(a.seed + (10000 if split == "val" else 20000))
        for prompts in pool[:a.n].split(a.batch_size):
            ids = rollout(model, prompts.to(a.device), a.response_length)
            response = ids[:, 4:]
            rewards.extend(format_reward(response).tolist())
            successes.extend(((response[:, 0] == 0)&(response[:, -1] == 0)).tolist())
            dollars.extend((response == 0).float().mean(-1).tolist())
    return dict(split=split, n=a.n, reward=sum(rewards)/a.n,
                success_rate=sum(successes)/a.n, success_interval=wilson_interval(sum(successes), a.n),
                dollar_fraction=sum(dollars)/a.n)


def part5(a, rec):
    cfg = GPTConfig(vocab_size=len(VOCAB), block_size=4+a.response_length,
                    n_layer=a.layers, n_head=4, d_model=a.width, dropout=0.)
    model = GPT(cfg).to(a.device)
    # dropout=0; eval 保持采样和重算为同一策略，仍然启用 autograd。
    model.eval()
    opt = torch.optim.AdamW(model.parameters(), lr=a.seq_lr, weight_decay=0.)
    pool = prompt_pool("train").to(a.device)
    rec.emit("sequence_setup", config=asdict(cfg), trainable_parameters=model.n_params(),
             training_prompts=len(pool), steps=a.seq_steps, batch_size=a.batch_size,
             reward="0.5*first_is_dollar + 0.5*last_is_dollar", pretrained=False,
             sequence_reduction="sum", prompt_tokens=4, response_tokens=a.response_length)
    initial = eval_sequence(model, a, "val")
    rec.emit("sequence_eval", step=0, **initial)
    curve = [dict(step=0, **initial)]
    baseline = 0.
    for step in range(1, a.seq_steps+1):
        prompts = pool[torch.randint(len(pool), (a.batch_size,), device=a.device)]
        ids = rollout(model, prompts, a.response_length)
        reward = format_reward(ids[:, 4:])
        logits, _ = model(ids[:, :-1])
        mask = torch.zeros_like(ids[:, 1:], dtype=torch.bool)
        mask[:, 3:] = True  # logits position 3 predicts first response at ids position 4.
        lp = sequence_log_prob(logits, ids[:, 1:], mask)
        loss = reinforce_loss(lp, reward, baseline)
        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
            raise RuntimeError("nonfinite loss/gradient")
        opt.step()
        baseline = .9*baseline + .1*reward.mean().item()
        if step == 1 or step % a.log_every == 0 or step == a.seq_steps:
            rec.emit("sequence_train", step=step, loss=loss.item(), reward=reward.mean().item(),
                     baseline_after_update=baseline, grad_norm=grad_norm.item(),
                     trainable_with_grad=sum(p.numel() for p in model.parameters() if p.grad is not None),
                     response_mask_tokens=mask.sum(-1).unique().tolist(),
                     prompt_mask_tokens=mask[:, :3].sum().item())
            stats = eval_sequence(model, a, "val")
            rec.emit("sequence_eval", step=step, **stats)
            curve.append(dict(step=step, **stats))
    rec.table("sequence_curve", curve)
    rec.plot("sequence_curve", [("val reward", [r["step"] for r in curve],
                                [r["reward"] for r in curve], None),
                               ("val both delimiters", [r["step"] for r in curve],
                                [r["success_rate"] for r in curve], None)], "rate / reward")
    # 最后一步固定为最终模型；不依据 test 选步数或超参数。
    if a.out:
        torch.save(dict(config=asdict(cfg), model=model.state_dict(), args=vars(a)),
                   a.out / "final.pt")
    if a.final_test:
        rec.emit("sequence_final_test_once", step=a.seq_steps, **eval_sequence(model, a, "test"))


PARTS = {1: ("多种子老虎机", part1), 2: ("固定策略的梯度方差", part2),
         3: ("KL k1/k2/k3", part3), 4: ("PPO 梯度", part4),
         5: ("小 GPT 序列级 REINFORCE（可选）", part5)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="*", type=int)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    for name, default in (("n", 20), ("seeds", 8), ("steps", 200), ("batch-size", 16),
                          ("repeats", 1000), ("mc-batch", 32), ("log-every", 50),
                          ("seq-steps", 20), ("response-length", 8), ("width", 64), ("layers", 2)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--lr", type=float, default=.1)
    parser.add_argument("--seq-lr", type=float, default=.001)
    parser.add_argument("--final-test", action="store_true", help="冻结设置后显式评一次 test；默认仅 val")
    a = parser.parse_args(argv)
    if any(p not in PARTS for p in a.parts):
        parser.error("部分号须为 1–5")
    for key in ("n", "seeds", "steps", "batch_size", "repeats", "mc_batch", "log_every", "seq_steps", "response_length", "width", "layers"):
        if getattr(a, key) < 1:
            parser.error(f"{key} 须为正整数")
    if a.repeats < 2 or a.response_length < 2 or a.width % 8 or a.seed < 0:
        parser.error("repeats/response-length 至少 2；width 须为 8 的倍数；seed 非负")
    if any(not math.isfinite(x) or x <= 0 for x in (a.lr, a.seq_lr)):
        parser.error("学习率须为有限正数")
    if a.final_test and 5 not in a.parts:
        parser.error("--final-test 仅用于显式部分 5")
    pick_device(a.device)
    return a


def main(argv=None):
    a = parse_args(argv)
    torch.set_num_threads(2)
    rec = Recorder(a)
    start = time.perf_counter()
    rec.emit("start", utc=datetime.now(timezone.utc).isoformat(), torch_version=torch.__version__,
             gpu=torch.cuda.get_device_name() if a.device == "cuda" else None,
             config={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()})
    if a.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for part in a.parts or [1, 2, 3, 4]:
        torch.manual_seed(a.seed)
        synchronize(a.device)
        begin = time.perf_counter()
        title, fn = PARTS[part]
        rec.emit("part_start", part=part, title=title)
        fn(a, rec)
        synchronize(a.device)
        rec.emit("part_end", part=part, seconds=time.perf_counter()-begin)
    rec.emit("end", utc=datetime.now(timezone.utc).isoformat(), seconds=time.perf_counter()-start,
             peak_allocated_bytes=torch.cuda.max_memory_allocated() if a.device == "cuda" else None,
             peak_reserved_bytes=torch.cuda.max_memory_reserved() if a.device == "cuda" else None)
    if a.out:
        (a.out / "metrics.json").write_text(json.dumps(rec.rows, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
