"""第 14 章：解码策略。默认 CPU、两线程、本地语料、离线且不下载。

python -m book.part2_lm.ch14_decoding.tour [部分号 1–7 ...] [选项]
无 --ckpt 时第 5 部分训练 tiny 冒烟模型；不含 KV cache（第 15 章）。
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "2"

import torch

from miniocr.decoding import (
    DecodeConfig, apply_repetition_penalty, apply_temperature, top_k_filter,
    top_p_filter, min_p_filter, process_logits, select_next, sample_sequences,
    gpt_step_fn, gnmt_length_penalty, beam_search, brute_force_search,
    short_bias_toy, decoding_report, speculative_accept_reject,
    speculative_output_distribution,
)
from miniocr.data.latex_corpus import corpus_available
from miniocr.data.latex_lm import load_latex_lm
from miniocr.data.token_file import _tokenizer_digest
from miniocr.device import pick_device, synchronize
from miniocr.models.gpt import GPT, GPTConfig
from miniocr.train.pretrain import PretrainConfig, token_stream, train_pretrain
from miniocr.train.scaling import ladder_config

ROOT = Path("outputs/book/ch14")
NAMES = ["x", "y", "^", "+", "}"]
FIVE = torch.tensor([[2.0, 1.0, 0.5, 0.0, -1.0]])
PREV = torch.tensor([[0, 4]])
TARGET = torch.tensor([.40, .25, .15, .15, .05])
DRAFT = torch.tensor([.15, .35, .25, .05, .20])
CONTENT_SCOPE = ("mathtext可解析率、unique_ratio与样例：完整文本（提示内容+续写，去BOS/EOS）；"
                 "distinct-1/2、平均生成长度（mean_length）：仅续写（去提示内容、BOS/EOS）")


def _format(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format(v) for v in value) + "]"
    return str(value)


def emit(**values):
    """统一键=值；浮点保留四位，概率与 logit 可直接对照手算。"""
    print("；".join(f"{key}={_format(value)}" for key, value in values.items()), flush=True)


def kept(logits, names=NAMES):
    return [name for name, value in zip(names, logits[0]) if torch.isfinite(value)]


def show_distribution(label, logits, names=NAMES):
    prob = logits.softmax(-1)[0]
    emit(阶段=label, 保留集合=kept(logits, names),
         被屏蔽=[name for name, v in zip(names, logits[0]) if not torch.isfinite(v)])
    for name, value, p in zip(names, logits[0], prob):
        emit(token=name, logit=float(value), 概率=float(p))


class Experiment:
    def __init__(self, args):
        self.args = args
        self.data = self.model = None

    def prepare(self):
        if self.model is not None:
            return
        args = self.args
        saved = torch.load(args.ckpt, map_location="cpu", weights_only=False) if args.ckpt else None
        version = saved["identity"]["data_version"] if saved else "latex-v1/train/n=20000"
        match = re.fullmatch(r"latex-v1/train/n=(\d+)", version)
        if not match or int(match[1]) < 1:
            raise ValueError(f"不支持的检查点数据版本：{version}")
        emit(数据来源="本地 LaTeX 语料" if corpus_available() else "降级为内置样例，只证明流程跑通",
             数据版本=version)
        self.data = load_latex_lm(int(match[1]))
        tok = self.data.tokenizer
        self.bos, self.eos = (tok.special_ids[key] for key in ("<|bos|>", "<|eos|>"))
        digest = _tokenizer_digest(tok)  # 与 pretrain.token_stream 完全相同的指纹函数。
        if saved:
            identity = saved["identity"]
            if identity["tokenizer_hash"] != digest:
                raise ValueError("检查点 tokenizer_hash 与本地分词器哈希不一致")
            cfg = GPTConfig(**identity["model"])
            if cfg.vocab_size != len(tok) or (identity["bos"], identity["eos"]) != (self.bos, self.eos):
                raise ValueError("检查点词表大小或 BOS/EOS 编号不一致")
            self.model = GPT(cfg).to(args.device)
            self.model.load_state_dict(saved["model"])
            self.source = str(args.ckpt)
            self.steps = saved["step"]
            emit(检查点路径=self.source, 训练步数=self.steps, 分词器哈希=digest)
        else:
            emit(模型说明=f"冒烟模型（训练 {args.smoke_steps} 步），只证明流程跑通，数字不代表质量")
            torch.manual_seed(args.seed)
            self.model = GPT(ladder_config("tiny", len(tok), 32)).to(args.device)
            ids, meta = token_stream(tok, self.data.train_text, version, use_file=False)
            cfg = PretrainConfig(steps=args.smoke_steps, batch=16, context=32,
                                 warmup=min(15, args.smoke_steps), seed=args.seed, device=args.device,
                                 eval_every=args.smoke_steps, eval_formulas=20, final_eval_formulas=20)
            run_dir = ROOT / f"smoke-{args.device}-seed{args.seed}"
            started = time.perf_counter()
            train_pretrain(self.model, ids, self.data.val, self.data.val_text, cfg,
                           bos=self.bos, eos=self.eos, run_dir=run_dir,
                           data_version=version, tokenizer_hash=meta["tokenizer_hash"])
            self.source, self.steps = str(run_dir / "checkpoint.pt"), args.smoke_steps
            emit(冒烟训练用时秒=time.perf_counter()-started, 检查点路径=self.source)
        self.model.eval()


def part1(exp):
    trace = []
    cfg = DecodeConfig(repetition_penalty=1.2, temperature=.8, top_k=4, top_p=.8)
    out = process_logits(FIVE, cfg, PREV, trace)
    emit(例子="五 token", token=NAMES, 已出现=["x", "}"],
         repetition_penalty=1.2, temperature=.8, top_k=4, top_p=.8)
    for stage, logits in trace:
        show_distribution(stage, logits)
    emit(最终采样分布=out.softmax(-1)[0].tolist())
    for temperature in (.5, 1., 2.):
        logits = apply_temperature(FIVE, temperature)
        p = logits.softmax(-1)
        emit(无重复惩罚=True, 温度=temperature, 概率=p[0].tolist(),
             熵_nat=float(-(p*p.log()).sum()), top_p_08保留数=len(kept(top_p_filter(logits, .8))))
    for label, logits in (("平坦", torch.zeros_like(FIVE)), ("尖锐", FIVE)):
        emit(分布=label, 原始概率=logits.softmax(-1)[0].tolist(),
             top_p_08保留数=len(kept(top_p_filter(logits, .8))),
             top_k_2保留数=len(kept(top_k_filter(logits, 2))))
    scaled = apply_temperature(apply_repetition_penalty(FIVE, PREV, 1.2), .8)
    emit(选做="min-p=0.1（重复惩罚与温度之后，未加top-k/top-p）",
         保留集合=kept(min_p_filter(scaled, .1)))


def wrong_negative_penalty(logits, prev, penalty):
    # 故意错：负数也除以 penalty，反而向零靠拢。
    return logits.clone().scatter(1, prev, logits.gather(1, prev)/penalty)


def part2(exp):
    correct = apply_repetition_penalty(FIVE, PREV, 1.2)
    wrong = wrong_negative_penalty(FIVE, PREV, 1.2)
    for name, row in (("原始", FIVE), ("正确：正除负乘", correct), ("错误：负数也除", wrong)):
        emit(重复惩罚=name, token="}", logit=float(row[0, 4]), 概率=float(row.softmax(-1)[0, 4]))
    emit(错误说明="负 logit 也除会让它变大；penalty>1 时正确写法让已出现 logit 不增")


def part3(exp):
    for temperature in (.5, 1., 1.5):
        generator = torch.Generator().manual_seed(exp.args.seed)
        cfg = DecodeConfig(temperature=temperature)
        samples = select_next(FIVE.expand(5000, -1), cfg, generator=generator)
        emit(温度=temperature, 抽样数=5000, token=NAMES,
             理论概率=apply_temperature(FIVE, temperature).softmax(-1)[0].tolist(),
             经验频率=(samples.bincount(minlength=5)/5000).tolist())
    emit(贪心永远选择=NAMES[int(select_next(FIVE, DecodeConfig(greedy=True))[0])])


def toy_text(toy, tokens):
    return " ".join(toy.vocab[i] for i in tokens)


def part4(exp):
    toy = short_bias_toy()
    prompt, eos = [toy.stoi["<bos>"]], toy.stoi["<eos>"]
    # 宽度覆盖整棵有限树，列出全部正概率路径（含到上限的截断路径）。
    all_paths = beam_search(toy.step_fn, prompt, 4, len(toy.vocab)**4, eos, return_all=True)
    emit(长度口径="生成 token 数，含 EOS、不含提示", EOS约定="占本步名额，完成后绝不扩展",
         剪枝="累计log概率", 长度惩罚="仅最终排序")
    for h in all_paths:
        emit(枚举序列=toy_text(toy, h.tokens), 概率=math.exp(h.logprob), log概率=h.logprob,
             长度=h.length, finished=h.finished, truncated=h.truncated,
             **{f"alpha={a}得分": h.logprob/gnmt_length_penalty(h.length, a) for a in (0., .6, 1.)})
    prefix = prompt.copy()
    for _ in range(4):
        token = int(toy.logits(prefix).argmax())
        prefix.append(token)
        if token == eos:
            break
    emit(贪心结果=toy_text(toy, prefix[len(prompt):]))
    for alpha in (0., 1.):
        for width in (1, 2, 3):
            h = beam_search(toy.step_fn, prompt, 4, width, eos, alpha)
            emit(beam=width, alpha=alpha, 结果=toy_text(toy, h.tokens), score=h.score)
    for alpha in (0., .6, 1.):
        h = brute_force_search(toy.step_fn, prompt, 4, eos, alpha)
        emit(暴力最优=toy_text(toy, h.tokens), alpha=alpha, score=h.score)
    for length in range(1, 7):
        emit(生成长度=length, alpha_06惩罚=gnmt_length_penalty(length, .6),
             alpha_10惩罚=gnmt_length_penalty(length, 1.))


def _sample_prompts(model, prompts, max_new, cfg, eos, bos, generator):
    # 不等长提示分组；不用填充 token 冒充真实上下文。
    result = [None] * len(prompts)
    for length in sorted(set(map(len, prompts))):
        indices = [i for i, p in enumerate(prompts) if len(p) == length]
        idx = torch.tensor([prompts[i] for i in indices], device=next(model.parameters()).device)
        rows = sample_sequences(model, idx, max_new, cfg, eos, [bos], generator).cpu().tolist()
        for i, row in zip(indices, rows):
            result[i] = row
    return result


def _content_report(rows, prompts, tok, eos):
    contents, generated, ended = [], [], []
    for row, prompt in zip(rows, prompts, strict=True):
        content = row[1:]  # 去 BOS，保留提示内容，使 mathtext 看见完整公式前缀。
        ended.append(eos in content)
        content = content[:content.index(eos)] if eos in content else content
        contents.append(content)
        generated.append(content[len(prompt)-1:])  # 按实际提示长度切出续写，已去 EOS 及其填充。
    texts = [tok.decode(row) for row in contents]
    return decoding_report(texts, generated, ended), texts


def part5(exp):
    exp.prepare()
    args, model, tok = exp.args, exp.model, exp.data.tokenizer
    prompts = [[exp.bos] + seq[1:-1][:args.prompt_tokens] for seq in exp.data.val[:args.samples]]
    if not prompts:
        raise ValueError("验证集没有可用提示")
    room = model.cfg.block_size - max(map(len, prompts))
    max_new = min(24, room) if args.max_new is None else args.max_new
    if max_new < 1 or max_new > room:
        raise ValueError("须满足提示长度 + max_new <= block_size；请减少 --prompt-tokens 或 --max-new")
    emit(指标="mathtext 可解析率（可编译率近似）", 请求条数=args.samples, 实际提示条数=len(prompts),
         prompt_tokens=args.prompt_tokens, max_new=max_new, block_size=model.cfg.block_size,
         截断说明="未遇 EOS 的也算分母；可解析不等于完整或正确",
         内容口径=CONTENT_SCOPE)
    strategies = [("greedy", DecodeConfig(greedy=True)),
                  ("temperature=0.7", DecodeConfig(temperature=.7)),
                  ("temperature=1.0", DecodeConfig()),
                  ("temperature=1.3", DecodeConfig(temperature=1.3)),
                  ("top-k=20,T=1", DecodeConfig(top_k=20)),
                  ("top-p=0.9,T=1", DecodeConfig(top_p=.9)),
                  ("greedy,repetition_penalty=1.2", DecodeConfig(greedy=True, repetition_penalty=1.2)),
                  ("beam=4,alpha=0", None), ("beam=4,alpha=0.6", None)]
    base_step = gpt_step_fn(model)

    def step_fn(prefixes):
        lp = base_step(prefixes)
        lp[:, exp.bos] = -torch.inf
        return lp.log_softmax(-1)  # 与采样一样，禁止 BOS 后重新归一化。

    records = []
    for label, cfg in strategies:
        torch.manual_seed(args.seed)
        generator = None if args.device == "mps" else torch.Generator(device=args.device).manual_seed(args.seed)
        synchronize(args.device)
        started = time.perf_counter()
        if cfg is None:
            alpha = .6 if label.endswith("0.6") else 0.
            rows = [p + beam_search(step_fn, p, max_new, 4, exp.eos, alpha).tokens for p in prompts[:10]]
        else:
            rows = _sample_prompts(model, prompts, max_new, cfg, exp.eos, exp.bos, generator)
        report, texts = _content_report(rows, prompts[:len(rows)], tok, exp.eos)
        synchronize(args.device)
        seconds = time.perf_counter()-started
        emit(策略=label, 条数=report["count"], mathtext可解析数=report["successes"],
             率=report["rate"], Wilson区间=report["interval"],
             **{k: report[k] for k in ("distinct_1", "distinct_2", "unique_ratio")},
             平均生成长度=report["mean_length"], ended_with_eos=report["ended_with_eos"], 用时秒=seconds)
        for i, text in enumerate(texts[:2], 1):
            emit(策略=label, 样例序号=i, 样例=json.dumps(text, ensure_ascii=False))
        records.append(dict(strategy=label, **report, seconds=seconds, texts=texts))
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / f"decoding-seed{args.seed}.json"
    path.write_text(json.dumps(dict(checkpoint=exp.source, steps=exp.steps, seed=args.seed,
                                   prompt_tokens=args.prompt_tokens, max_new=max_new,
                                   content_scope=CONTENT_SCOPE, strategies=records),
                               ensure_ascii=False, indent=2), encoding="utf-8")
    emit(报告文件=str(path), 比较口径="束搜索只取前 min(10,samples,验证集条数) 条；其余策略取全部所选提示")


def part6(exp):
    out, beta = speculative_output_distribution(TARGET, DRAFT)
    emit(目标p=TARGET.tolist(), 草稿q=DRAFT.tolist(), 解析输出分布=out.tolist(),
         与p最大差=float((out-TARGET).abs().max()), 接受率β=float(beta))
    generator = torch.Generator().manual_seed(exp.args.seed)
    counts, accepted = torch.zeros(5), 0
    for _ in range(20000):
        token, yes = speculative_accept_reject(TARGET, DRAFT, generator)
        counts[token] += 1
        accepted += yes
    empirical = counts / 20000
    emit(蒙特卡罗次数=20000, 经验分布=empirical.tolist(),
         最大误差=float((empirical-TARGET).abs().max()), 经验接受率=accepted/20000)
    emit(演示范围="单 token 接受-拒绝的正确性；未实现多 token 投机解码，不测加速比")


def wrong_temperature_order(logits):
    return apply_temperature(top_p_filter(logits, .8), 2.)


def wrong_missing_scatter(logits):
    # 故意错：排序位置的掩码直接应用到原始词表位置。
    p = logits.sort(descending=True, dim=-1).values.softmax(-1)
    keep = p.cumsum(-1)-p < .8
    keep[:, 0] = True
    return logits.masked_fill(~keep, -torch.inf)


def wrong_expand_eos(toy, prompt, eos, alpha=0.0):
    # 故意错：传 eos=None，导致 EOS 自循环也继续作为普通边扩展和计分。
    return beam_search(toy.step_fn, prompt, 4, 2, eos=None, alpha=alpha)


def part7(exp):
    show_distribution("正确：温度在top-p之前", top_p_filter(apply_temperature(FIVE, 2.), .8))
    show_distribution("错误：温度在top-p之后", wrong_temperature_order(FIVE))
    shuffled = FIVE[:, [4, 2, 0, 3, 1]]
    names = [NAMES[i] for i in [4, 2, 0, 3, 1]]
    show_distribution("正确：scatter回原索引", top_p_filter(shuffled, .8), names)
    show_distribution("错误：忘了scatter", wrong_missing_scatter(shuffled), names)
    toy = short_bias_toy()
    prompt, eos = [toy.stoi["<bos>"]], toy.stoi["<eos>"]
    for alpha in (0., 1.):
        good = beam_search(toy.step_fn, prompt, 4, 2, eos, alpha)
        bad = wrong_expand_eos(toy, prompt, eos, alpha)
        for label, h in (("正确：结束即移出", good), ("错误：EOS后继续扩展", bad)):
            emit(EOS处理=label, alpha=alpha, 结果=toy_text(toy, h.tokens), log概率=h.logprob,
                 长度=h.length, 长度惩罚=gnmt_length_penalty(h.length, alpha), score=h.score)
    emit(说明="EOS自循环概率为1：alpha=0时log概率虽不变，却多出EOS；"
         "alpha=1时虚增长度使负log概率除以更大的长度惩罚，多余EOS路径错误地胜出")
    part2(exp)


PARTS = {1: (part1, "五 token 例子：处理流水线（14.3–14.5）"),
         2: (part2, "重复惩罚的正负号（14.6）"),
         3: (part3, "贪心与温度采样（14.2、14.3）"),
         4: (part4, "束搜索与偏短（14.7、14.8）"),
         5: (part5, "在 LaTeX GPT 上比较解码策略（14.9）"),
         6: (part6, "投机解码的接受-拒绝（14.10）"),
         7: (part7, "常见错误重现（14.11）")}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="*", type=int)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--smoke-steps", type=int, default=150)
    parser.add_argument("--prompt-tokens", type=int, default=3)
    parser.add_argument("--max-new", type=int, help="默认 min(24, block_size-最长提示长度)")
    args = parser.parse_args(argv)
    if any(part not in PARTS for part in args.parts):
        parser.error("部分号须为1–7")
    if args.seed < 0 or args.prompt_tokens < 0 or min(args.samples, args.smoke_steps) < 1:
        parser.error("seed/prompt-tokens须非负；samples/smoke-steps须为正")
    if args.max_new is not None and args.max_new < 1:
        parser.error("max-new须为正")
    if args.ckpt is not None and not args.ckpt.is_file():
        parser.error(f"检查点不存在：{args.ckpt}")
    try:
        pick_device(args.device)
    except RuntimeError as exc:
        parser.error(str(exc))
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(2)
    emit(章节=14, 设备=args.device, 线程=2, 种子=args.seed, 默认行为="CPU；离线；不下载；不用KV cache")
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
