"""第 13 章：预训练过程与规模规律。默认 CPU、两线程、本地语料、离线且不下载。

python -m book.part2_lm.ch13_pretraining.tour [部分号 1–10 ...] [选项]
默认只跑冒烟；第 5/6 部分须显式 --run 才训练。GPU/MPS 由学习者显式选择。
"""
import argparse
import copy
import csv
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "2"
# 不设置 CUDA_VISIBLE_DEVICES：--device cuda 必须能使用学习者显式授权的设备；
# 默认 device=cpu，且自检命令显式设 CUDA_VISIBLE_DEVICES=""。

import numpy as np
import torch

from miniocr.data.latex_lm import load_latex_lm
from miniocr.data.token_file import StaleCacheError, load_token_file
from miniocr.device import pick_device
from miniocr.models.gpt import GPT, GPTConfig
from miniocr.train.lm import _param_groups, get_batch
from miniocr.train.optim import warmup_cosine
from miniocr.train.pretrain import (ROOT, PRESETS, PretrainConfig, initial_loss, leakage_overlap,
    plot_learning_curves, reference_parse_report, sample_report, summarize_log, token_stream,
    train_pretrain, format_metric, ngram_landmarks)
from miniocr.train.scaling import (LADDER, HOFFMANN2022_APPROACH3, chinchilla_grid,
    chinchilla_optimal, compare_offset_fits, compute_frontier, fit_power_law_offset,
    ladder_config, ladder_table, synthetic_scaling, fraction_config, read_ladder_points)


def emit(**values):
    """统一键=值；学习率/计算量等跨数量级字段用科学计数法。"""
    print("；".join(f"{k}={format_metric(v, k)}" for k, v in values.items()), flush=True)


class Experiment:
    def __init__(self, args):
        self.args, self.data = args, None
        self.result = None
        directory = args.resume or args.run_dir
        self.run_dir = Path(directory) if directory else ROOT / "runs" / f"smoke-{args.device}-seed{args.seed}"
        self.resume_dir = ROOT / "runs" / f"resume-{args.device}-seed{args.seed}"

    def prepare(self):
        if self.data is not None:
            return
        self.data = load_latex_lm(self.args.n_train)
        self.bos, self.eos = (self.data.tokenizer.special_ids[k] for k in ("<|bos|>", "<|eos|>"))
        self.version = f"latex-v1/train/n={self.args.n_train}"
        self.token_path = ROOT / f"train-n{self.args.n_train}"
        self.ids, self.meta = token_stream(self.data.tokenizer, self.data.train_text, self.version, self.token_path)
        emit(数据版本=self.version, 训练公式=len(self.data.train), 验证公式=len(self.data.val),
             词表=len(self.data.tokenizer), 分词器哈希=self.meta["tokenizer_hash"])

    def config(self):
        a = self.args
        return PretrainConfig(steps=a.steps, batch=a.batch, context=a.context, warmup=a.warmup,
                              lr=a.lr, min_lr=a.lr/10, seed=a.seed, device=a.device, amp=a.amp,
                              eval_every=a.eval_every)

    def model(self, size=None, **overrides):
        self.prepare()
        torch.manual_seed(self.args.seed)
        cfg = ladder_config(size or self.args.size, len(self.data.tokenizer), self.args.context)
        return GPT(replace(cfg, **overrides)).to(self.args.device)

    def train(self, model, run_dir, *, cfg=None, **options):
        return train_pretrain(model, self.ids, self.data.val, self.data.val_text, cfg or self.config(),
                              bos=self.bos, eos=self.eos, run_dir=run_dir, data_version=self.version,
                              tokenizer_hash=self.meta["tokenizer_hash"], **options)

    def checkpoint_model(self):
        self.prepare()
        path = self.run_dir/"checkpoint.pt"
        if not path.exists():
            # 独立第9部分只需一次冒烟，不额外跑续训对拍。
            model = self.model()
            self.result = self.train(model, self.run_dir)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        identity = saved["identity"]
        if identity["tokenizer_hash"] != self.meta["tokenizer_hash"] or identity["data_version"] != self.version:
            raise ValueError("样本检查点的数据/分词器不匹配；请先运行第3部分")
        model = GPT(GPTConfig(**identity["model"]))
        model.load_state_dict(saved["model"])
        emit(生成检查点=str(path), 实际训练步数=saved["step"])
        return model, saved["step"]


def part1(exp):
    exp.prepare()
    d = exp.data
    targets = sum(len(s)-1 for s in d.val)
    emit(口径="训练拼接流；训练忽略BOS目标；验证逐公式、BOS屏蔽输出、EOS计入、按目标加权",
         训练token数=len(exp.ids), 验证原文字节数=d.n_bytes, 验证目标数=targets,
         每验证目标平均字节数=d.n_bytes/targets, lnV=math.log(len(d.tokenizer)))
    landmarks = ngram_landmarks(exp.ids, d.val, d.val_text, exp.bos, exp.eos,
                                k=1.0, vocab_size=len(d.tokenizer))
    emit(学习曲线路标="训练一元熵", H1_nat_per_token=landmarks["H1"],
         训练非BOS目标数=landmarks["train_targets"], 估计="plug-in")
    for label, key in (("只学会频率的一元模型", "unigram"), ("二元模型", "bigram")):
        emit(学习曲线路标=label, 平滑k=landmarks["k"], **landmarks[key],
             验证公式数=landmarks["n_formulas"], 验证目标数=landmarks["n_tokens"],
             验证原文字节数=landmarks["n_bytes"], 可预测类别数=landmarks["predictable_classes"])
    seen = exp.args.steps*exp.args.batch*exp.args.context
    emit(一个epoch训练token数=len(exp.ids), 计划tokens_seen=seen, 对应epoch数=seen/len(exp.ids),
         epoch口径="随机窗口处理量折算；含BOS位置，不代表无放回遍历")
    reread = torch.tensor(np.asarray(load_token_file(exp.token_path, d.tokenizer, exp.version)).astype(np.int64))
    emit(token文件=str(exp.token_path), 读回一致=torch.equal(exp.ids, reread))
    try:
        load_token_file(exp.token_path, d.tokenizer, exp.version+"/stale")
    except StaleCacheError as exc:
        emit(过期缓存拒绝=str(exc))
    emit(初始loss=initial_loss(exp.model(), exp.ids, exp.bos, exp.config()), lnV=math.log(len(d.tokenizer)))


def _schedule(cfg, step):
    return warmup_cosine(step, cfg.steps, cfg.warmup, cfg.lr, cfg.min_lr)


def part2(exp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    full = replace(exp.config(), steps=5000, warmup=200, lr=.003, min_lr=.0003)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for ax, (label, cfg) in zip(axes, [("完整训练示例（待校准）", full), ("当前冒烟或自定配置", exp.config())]):
        emit(调度配置=label, steps=cfg.steps, warmup=cfg.warmup, lr=cfg.lr, min_lr=cfg.min_lr)
        steps = sorted({0, 1, max(0, cfg.warmup-1), cfg.warmup, cfg.steps//4,
                        cfg.steps//2, 3*cfg.steps//4, cfg.steps-1, cfg.steps})
        for step in steps:
            emit(调度=label, 索引步=step, lr=_schedule(cfg, step))
        ax.plot(range(cfg.steps+1), [_schedule(cfg, s) for s in range(cfg.steps+1)])
        ax.set(xlabel="zero-based schedule step", ylabel="lr")
    ROOT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(ROOT/"schedule.png")
    plt.close(fig)
    emit(调度图=str(ROOT/"schedule.png"))
    model = exp.model()
    x, y = get_batch(exp.ids, exp.args.batch, exp.args.context,
                     torch.Generator().manual_seed(exp.args.seed), exp.args.device)
    for multiplier in (1, 100):
        trial = copy.deepcopy(model)
        before = {n: p.detach().clone() for n, p in trial.named_parameters()}
        # 去掉衰减，单独观察自适应更新；零梯度与 eps 会使比例略小于1。
        opt = torch.optim.AdamW(_param_groups(trial, 0), lr=exp.args.lr, betas=(.9, .95))
        (trial(x, y, loss_mask=y != exp.bos)[1]*multiplier).backward()
        opt.step()
        for name, p in trial.named_parameters():
            emit(Adam首步参数=name, loss倍率=multiplier,
                 平均绝对更新除以lr=float((p.detach()-before[name]).abs().mean())/exp.args.lr)


def _resume_schedule(cfg):
    start = max(1, cfg.steps//2)
    for offset in range(min(5, cfg.steps-start)):
        emit(续训后更新步=start+offset+1, 正确lr=_schedule(cfg, start+offset),
             错误从头lr=_schedule(cfg, offset))


def part3(exp):
    full_model = exp.model()
    emit(训练配置="冒烟或用户显式覆盖", 设备=exp.args.device, 模型=exp.args.size,
         steps=exp.args.steps, batch=exp.args.batch, context=exp.args.context, amp=exp.args.amp,
         日志目录=str(exp.run_dir))
    emit(训练模式="真正续训" if exp.args.resume else "续训对拍" if exp.args.check_resume else "单次训练")
    full = exp.train(full_model, exp.run_dir, resume=bool(exp.args.resume))
    exp.result = full
    if exp.args.check_resume:
        partial = exp.model()
        exp.train(partial, exp.resume_dir, stop_after=max(1, exp.args.steps//2))
        resumed_model = exp.model()  # 丢弃内存模型，验证加载路径
        resumed = exp.train(resumed_model, exp.resume_dir, resume=True)
        delta = max(float((p.detach()-q.detach()).abs().max())
                    for p, q in zip(full_model.parameters(), resumed_model.parameters()))
        for name, result in (("不中断", full), ("中断后续训", resumed)):
            row = result["evals"][-1]
            emit(续训对拍=name, val_nat_per_token=row["val_nat_per_token"], bits_per_byte=row["val_bits_per_byte"])
        emit(参数最大差=delta, CPU逐位一致=delta == 0)
        try:
            train_pretrain(resumed_model, exp.ids, exp.data.val, exp.data.val_text,
                           replace(exp.config(), steps=exp.args.steps+1), bos=exp.bos, eos=exp.eos,
                           run_dir=exp.resume_dir, data_version=exp.version,
                           tokenizer_hash=exp.meta["tokenizer_hash"], resume=True)
        except ValueError as exc:
            emit(续训配置错误=str(exc))
    _resume_schedule(exp.config())


def part4(exp):
    path = Path(exp.args.log) if exp.args.log else exp.run_dir
    if not (path/"metrics.jsonl").exists():
        if exp.args.log:
            raise ValueError(f"指定日志不存在：{path}")
        model = exp.model()
        exp.result = exp.train(model, exp.run_dir)
    emit(学习曲线图=str(plot_learning_curves(path)))
    emit(**summarize_log(path))
    emit(读图提示="周期验证用前缀，末次扩大至标准前2000条；先比较eval_formulas/eval_bytes再解释折点；train-val差含评测口径差异")


ABLATIONS = {"baseline": {}, "no_position": {"pos_emb": "none"},
             "learned_position": {"pos_emb": "learned"}, "layernorm": {"norm": "layernorm"},
             "no_resid_scale": {"resid_scale": False}, "untied": {"tie_weights": False}}


def part5(exp):
    rows = []
    for name, changes in ABLATIONS.items():
        model = exp.model(**changes)
        emit(消融=name, 修改项=json.dumps(changes), 参数量=model.n_params(),
             非嵌入参数量=model.n_params(True), 固定批初始loss=initial_loss(model, exp.ids, exp.bos, exp.config()))
        if exp.args.run:
            path = ROOT/"runs"/f"ablation-{name}-seed{exp.args.seed}"
            result = exp.train(model, path)["evals"][-1]
            rows.append(dict(variant=name, seed=exp.args.seed, steps=exp.args.steps,
                             val_nat_per_token=result["val_nat_per_token"], bits_per_byte=result["val_bits_per_byte"],
                             delta=result["val_nat_per_token"]-(rows[0]["val_nat_per_token"] if rows else result["val_nat_per_token"])))
    if rows:
        dest = ROOT/f"ablations-seed{exp.args.seed}.csv"
        with dest.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for row in rows:
            emit(**row)
        emit(消融汇总=str(dest))
    else:
        emit(消融训练="默认不训练；--run 才逐个训练；至少比较两个种子；门槛待校准")


def part6(exp):
    exp.prepare()
    cfg = exp.config()
    d = cfg.steps*cfg.batch*cfg.context
    runs, total = [], 0
    for row in ladder_table(len(exp.data.tokenizer), d, cfg.context):
        emit(**row, D_plan=d, D_epoch=len(exp.ids), epoch_6ND=6*row["matmul"]*len(exp.ids))
        for frac in exp.args.ladder_fracs:
            scaled = fraction_config(cfg, frac)
            tokens = scaled.steps*scaled.batch*scaled.context
            compute = 6*row["matmul"]*tokens
            path = ROOT/"ladder"/row["size"]/f"frac{str(frac).removesuffix('.0')}"
            emit(阶梯预算=row["size"], frac=frac, steps=scaled.steps, warmup=scaled.warmup,
                 D=tokens, compute_6ND=compute, 目录=str(path))
            total += compute
            runs.append((row["size"], scaled, path))
    # 全部预算先打印完，再开始任何训练。
    emit(阶梯run数=len(runs), 总6ND合计=total)
    emit(预算口径="D_plan=当前配置计划tokens；D_epoch=训练流一个epoch；6ND的N=matmul（含共享输出）",
         注意力额外FLOPs占比="attention_extra_ratio≈n_layer×context×d_model/N_matmul；Kaplan 2020前向2N+2×n_layer×n_ctx×d_attn，额外项/2N的近似，非总量占比")
    if exp.args.run:
        for size, scaled, path in runs:
            exp.train(exp.model(size=size), path, cfg=scaled)
    else:
        emit(阶梯训练="默认不训练；--run 对每个size×frac从相同种子初值训练独立完整warmup+cosine；待校准")


def part7(exp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c, loss = synthetic_scaling(seed=exp.args.seed)
    fit = compare_offset_fits(c, loss)
    emit(数据="合成数据", 真值a=25.0, 真值alpha=.3, 真值Linf=1.5)
    emit(偏移幂律拟合="合成数据", **fit["offset"])
    emit(忽略Linf="合成数据", **fit["power"])
    fig, ax = plt.subplots()
    ax.semilogx(c, loss, "o", label="synthetic data")
    r = fit["offset"]
    ax.semilogx(c, r["a"]*c**(-r["alpha"])+r["Linf"], label="offset fit")
    ax.set(xlabel="compute C", ylabel="loss")
    ax.legend()
    ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT/"synthetic_scaling.png")
    plt.close(fig)
    rows = read_ladder_points(ROOT/"ladder")
    if not rows:
        emit(阶梯拟合="未找到阶梯日志，先运行第 6 部分 --run")
        return
    for row in rows:
        emit(阶梯最终点="标准验证", **row)
    for size in dict.fromkeys(r["size"] for r in rows):
        subset = [r for r in rows if r["size"] == size]
        emit(二维损失数据=size, 点数=len(subset), fracs=[r["frac"] for r in subset],
             说明="同size固定N、不同frac改变D；跨size组成L(N,D)数据，可另作二维拟合")
    points = [(r["C"], r["val_nat_per_token"]) for r in rows]
    frontier = compute_frontier(points)
    emit(实测评测点数=len(points), 前沿点数=len(frontier), 拟合口径="各run末次标准验证；logC区间内最低loss")
    for c_value, value in frontier:
        emit(前沿点="分箱最低loss", C=c_value, val_nat_per_token=value)
    fig, ax = plt.subplots()
    for size in dict.fromkeys(r["size"] for r in rows):
        subset = [r for r in rows if r["size"] == size]
        ax.scatter([r["C"] for r in subset], [r["val_nat_per_token"] for r in subset], label=size)
    c, loss = np.array(frontier).T
    ax.scatter(c, loss, s=100, facecolors="none", edgecolors="black", label="binned frontier")
    if len(frontier) >= 3 and len(set(c)) >= 2:
        fitted = fit_power_law_offset(c, loss)
        emit(阶梯偏移拟合="探索性；少量点不代表普适规律", **fitted)
        grid = np.geomspace(min(c), max(c), 200)
        ax.plot(grid, fitted["a"]*grid**(-fitted["alpha"])+fitted["Linf"], label="offset fit")
    else:
        emit(阶梯拟合="可比较前沿点不足三个，暂不拟合")
    ax.set(xscale="log", xlabel="6 N_matmul D", ylabel="val nat/token")
    ax.legend()
    fig.savefig(ROOT/"ladder_scaling.png")
    plt.close(fig)
    emit(阶梯拟合图=str(ROOT/"ladder_scaling.png"))


def part8(exp):
    p = HOFFMANN2022_APPROACH3
    emit(系数="外部系数，来源 Hoffmann et al. 2022 §3.3 第三种方法；不代表本项目", **p)
    emit(N预算指数=p["beta"]/(p["alpha"]+p["beta"]), D预算指数=p["alpha"]/(p["alpha"]+p["beta"]))
    for c in (1e18, 1e19, 1e20, 1e21, 1e22, 1e23, 1e24, 1e15, 1e16, 1e17):
        exact = chinchilla_optimal(c, **p)
        grid = chinchilla_grid(c, **p)
        emit(C=c, **exact, 网格N相对误差=abs(grid["N_opt"]/exact["N_opt"]-1),
             网格D相对误差=abs(grid["D_opt"]/exact["D_opt"]-1),
             网格loss相对误差=abs(grid["L_opt"]/exact["L_opt"]-1))
    emit(小预算提示="1e15/1e16/1e17 远在原拟合范围之外，只是演示公式；不能据此选本项目最优规模")


def part9(exp):
    exp.prepare()
    for label, row in reference_parse_report(exp.data.val_text).items():
        emit(参考上限桶=label, **row)
    model, steps = exp.checkpoint_model()
    result = sample_report(model, exp.data.tokenizer, count=exp.args.samples, seed=exp.args.seed)
    examples = result.pop("examples")
    emit(指标="mathtext 可解析率（可编译率近似）", **result)
    for i, text in enumerate(examples, 1):
        emit(样例序号=i, 样例=json.dumps(text, ensure_ascii=False))
    emit(生成提示="冒烟模型只训练了几十步，这个数只证明流程跑通" if steps <= 100 else "用户完整训练检查点；mathtext仅支持LaTeX子集",
         实际步数=steps, 截断提示="未遇EOS样本也纳入分母；截断串能解析不等于完整公式正确")


def part10(exp):
    exp.prepare()
    # 故障实验固定很小模型/短上下文，避免 --preset 把它放大成长训练。
    torch.manual_seed(exp.args.seed)
    baseline = GPT(ladder_config("tiny", len(exp.data.tokenizer), 16))
    for lr in (.003, .03, 3.):
        model = copy.deepcopy(baseline)
        opt = torch.optim.AdamW(_param_groups(model, .1), lr=lr, betas=(.9, .95))
        generator = torch.Generator().manual_seed(exp.args.seed)
        first_bad, losses = None, []
        for step in range(1, 61):
            # 每步新随机批；每种 lr 重置独立生成器，保证看到完全相同的数据序列。
            x, y = get_batch(exp.ids, 8, 16, generator)
            opt.zero_grad(set_to_none=True)
            loss = model(x, y, loss_mask=y != exp.bos)[1]
            loss.backward()
            # 只测范数，不调用裁剪；非有限梯度也不被裁剪操作改写。
            grad = float(torch.linalg.vector_norm(torch.stack([
                torch.linalg.vector_norm(p.grad.detach()) for p in model.parameters() if p.grad is not None])))
            value = loss.item()
            losses.append(value)
            finite = math.isfinite(value) and math.isfinite(grad)
            if step % 10 == 0 or not finite:
                emit(学习率故障="随机新批，无裁剪", lr=lr, step=step, loss=value, grad_norm=grad)
            if not finite:
                first_bad = step
                break
            opt.step()
        emit(学习率故障汇总="以实测判断", lr=lr, 最后10步平均loss=float(np.mean(losses[-10:])),
             平均所用步数=min(10, len(losses)), 首次非有限步=first_bad)
    injected = exp.data.train_text[:50]
    emit(泄漏检查="标准划分（所用训练前缀与val前2000条）", 规范键交集=leakage_overlap(exp.data.train_text, exp.data.val_text))
    emit(泄漏检查="人为混入训练公式", 注入条数=len(injected),
         规范键交集=leakage_overlap(exp.data.train_text, exp.data.val_text+injected))
    _resume_schedule(exp.config())


PARTS = {1: (part1, "数据与口径（13.1）"), 2: (part2, "学习率调度（13.2）"),
         3: (part3, "冒烟训练与续训（13.3、13.4）"), 4: (part4, "学习曲线（13.3）"),
         5: (part5, "组件消融（13.5）"), 6: (part6, "模型阶梯与6ND（13.6）"),
         7: (part7, "scaling拟合（13.7）"), 8: (part8, "Chinchilla分配（13.8）"),
         9: (part9, "生成与可编译率（13.9）"), 10: (part10, "常见错误（13.11）")}


def parse_args(argv=None):
    """解析部分号与覆盖项；preset 先应用，显式参数优先，所有非法数值走 parser.error。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="*", type=int)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", choices=LADDER)
    for name in ("steps", "batch", "context", "warmup"):
        parser.add_argument("--"+name, type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--amp", choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--n-train", type=int, default=20000)
    parser.add_argument("--run", action="store_true", help="实际训练第5/6部分的全部模型")
    parser.add_argument("--check-resume", action="store_true", default=None, help="额外跑中断续训路径进行对拍")
    parser.add_argument("--resume", metavar="RUN_DIR", help="第3部分从已有同配置目录真正续训")
    parser.add_argument("--run-dir", help="第3部分新训练目录；第4/9部分也使用它")
    parser.add_argument("--ladder-fracs", default="0.25,0.5,1", help="各模型独立完整调度的预算比例，逗号分隔")
    parser.add_argument("--log", help="第4部分读取的run目录")
    parser.add_argument("--preset", choices=PRESETS, help="待校准建议配置；设备仍由--device选择")
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args(argv)
    if args.resume and (args.check_resume or args.run_dir):
        parser.error("--resume 不与 --check-resume/--run-dir 合用；目录由 --resume 指定")
    if args.resume and args.parts and 3 not in args.parts:
        parser.error("--resume 必须包含第3部分")
    if args.check_resume is None:
        args.check_resume = args.preset is None and args.steps is None and not args.resume
    try:
        args.ladder_fracs = [float(v) for v in args.ladder_fracs.split(",")]
        if (not args.ladder_fracs or len(set(args.ladder_fracs)) != len(args.ladder_fracs) or
                any(not math.isfinite(v) or not 0 < v <= 1 for v in args.ladder_fracs)):
            raise ValueError
    except ValueError:
        parser.error("--ladder-fracs 须为不重复的 (0,1] 比例，如 0.25,0.5,1")
    for key, value in PRESETS[args.preset or "cpu"].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if any(p not in PARTS for p in args.parts):
        parser.error("部分号须为1–10")
    if min(args.steps, args.batch, args.context, args.n_train, args.eval_every, args.samples) < 1:
        parser.error("步数、批量、上下文、公式数、评测间隔和样本数须为正")
    if args.seed < 0 or not 0 <= args.warmup <= args.steps or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("seed>=0；0<=warmup<=steps；lr须为正有限数")
    if args.amp == "fp16" and args.device != "cuda":
        parser.error("fp16+GradScaler仅支持cuda；CPU/MPS请用fp32或受支持的bf16")
    try:
        pick_device(args.device)
    except RuntimeError as exc:
        parser.error(str(exc))
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(2)
    emit(章节=13, 设备=args.device, 线程=2, 种子=args.seed, 建议配置="待校准", 默认行为="CPU冒烟；离线；不下载")
    if args.preset:
        emit(所选预设=args.preset, 校准状态="待校准", 用途="完整训练/消融/阶梯共用训练配置；阶梯覆盖模型大小")
    started = time.perf_counter()
    exp = Experiment(args)
    for part in args.parts or PARTS:
        fn, title = PARTS[part]
        emit(部分=part, 内容=title)
        t0 = time.perf_counter()
        fn(exp)
        emit(**{f"第 {part} 部分用时秒": time.perf_counter()-t0})
    emit(tour总用时秒=time.perf_counter()-started)


if __name__ == "__main__":
    main()
