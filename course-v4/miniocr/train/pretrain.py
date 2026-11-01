"""第 13 章：可续训的 LaTeX GPT 预训练、日志和生成诊断。

训练用 BOS…EOS 拼接流，忽略 BOS 目标；验证逐公式滑窗、屏蔽 BOS 输出。
steps 是总更新数，batch 是累积后的批量；tokens_seen 计处理位置（含被屏蔽的
BOS 目标），6ND 是计算近似，不是有效监督目标数。检查点以原子替换写入，
续训将 JSONL/CSV 截断到保存的字节位置，再从 step+1 追加。仅加载自己的检查点。
"""
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time

import numpy as np
import torch

from miniocr.data.latex_lm import FormulaWindows, encode_formulas, evaluate_lm
from miniocr.data.token_file import load_token_file, write_token_file
from miniocr.device import autocast, synchronize
from miniocr.generate import generate
from miniocr.train.checkpoint import set_rng_state
from miniocr.train.lm import _param_groups, get_batch
from miniocr.train.memory import matmul_params, train_flops
from miniocr.train.optim import warmup_cosine

ROOT = Path("outputs/book/ch13")
DTYPES = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def format_metric(value, key=""):
    """跨数量级字段用科学计数法；小的非零浮点也不能显示成零。"""
    scientific = (key in {"C", "N_opt", "D_opt", "D_per_N"} or
                  key.lower().endswith("lr") or "学习率" in key or
                  any(s in key.lower() for s in ("compute", "flops", "6nd")))
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        if scientific:
            return f"{value:.4e}"
        if isinstance(value, (float, np.floating)):
            return f"{value:.4e}" if 0 < abs(value) < 1e-3 or abs(value) >= 1e6 else f"{value:.4f}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_metric(v, key) for v in value) + "]"
    return str(value)
EVAL_FIELDS = ["step", "tokens_seen", "train_loss", "val_nat_per_token", "val_bits_per_byte",
               "lr", "tok_per_sec", "eval_formulas", "eval_tokens", "eval_bytes"]
# 每套适用于完整训练/消融/阶梯；阶梯任务用其 batch/context/steps，模型由阶梯覆盖。
# 全部待校准，不承诺速度、显存或训练效果。CPU 默认 tour 使用单独的冒烟配置。
PRESETS = {
    "cpu": dict(size="tiny", batch=16, context=32, steps=60, warmup=6, lr=3e-3, amp="fp32"),
    "mps": dict(size="medium", batch=32, context=128, steps=5000, warmup=200, lr=3e-3, amp="fp32"),
    "cuda": dict(size="large", batch=64, context=128, steps=5000, warmup=200, lr=3e-3, amp="bf16"),
}


@dataclass
class PretrainConfig:
    steps: int = 60
    batch: int = 16
    context: int = 32
    accum: int = 1
    warmup: int = 6
    lr: float = 3e-3
    min_lr: float = 3e-4
    weight_decay: float = 0.1
    clip: float = 1.0             # 0 表示不裁剪，仍记录裁剪前范数
    eval_every: int = 20
    eval_formulas: int = 200
    final_eval_formulas: int = 2000
    eval_batch: int = 128
    seed: int = 0
    device: str = "cpu"
    amp: str = "fp32"


def token_stream(tokenizer, texts, data_version, path=ROOT / "train", *, use_file=True):
    """返回 (CPU LongTensor, 元数据)；已有缓存只校验读取，过期抛 StaleCacheError。

    use_file=False 直接编码；文件缺失才写，半写入的缓存不静默重建。
    texts 或其顺序改变时调用者必须升级 data_version。
    """
    from miniocr.data.token_file import _tokenizer_digest
    texts = list(texts)
    if not use_file:
        ids, n_bytes = encode_formulas(tokenizer, texts, stream=True)
        return ids, dict(data_version=data_version, tokenizer_hash=_tokenizer_digest(tokenizer),
                         formula_count=len(texts), token_count=len(ids), text_utf8_bytes=n_bytes)
    path = Path(path)
    if not Path(str(path)+".json").exists() and not Path(str(path)+".bin").exists():
        write_token_file(path, tokenizer, texts, data_version)
    ids = load_token_file(path, tokenizer, data_version)
    meta = json.loads(Path(str(path)+".json").read_text(encoding="utf-8"))
    return torch.tensor(np.asarray(ids).astype(np.int64)), meta


def ngram_landmarks(train_ids, val_seqs, val_texts, bos, eos, k=1.0, *, vocab_size=None):
    """训练目标的 plug-in H1 与加 k 一元/二元模型的验证交叉熵。

    train_ids 是完整 BOS…EOS 拼接流。BOS 不作目标，也不占平滑类别；
    二元行可用 BOS 作上文，EOS→BOS 不计数。所有验证内容和 EOS 各计一次。
    传完整 vocab_size 可覆盖训练中未出现的类别；省略时按所给 ID 的最大值+1
    推断连续词表（包括验证中未见 ID），不使用验证频数。字节只计对应原文。
    """
    ids = np.asarray(train_ids, dtype=np.int64)
    if (ids.ndim != 1 or len(ids) < 2 or ids[0] != bos or ids[-1] != eos or
            len(val_seqs) != len(val_texts) or not val_seqs or
            not math.isfinite(k) or k <= 0 or bos == eos):
        raise ValueError("需要完整训练流、对应验证集及正有限平滑 k")
    if any(len(s) < 2 or s[0] != bos or s[-1] != eos or bos in s[1:] or eos in s[:-1]
           for s in val_seqs):
        raise ValueError("验证序列须为 BOS…EOS")
    largest = max(int(ids.max()), max(max(s) for s in val_seqs), bos, eos)
    vocab_size = largest+1 if vocab_size is None else vocab_size
    if vocab_size <= largest or min(int(ids.min()), min(min(s) for s in val_seqs), bos, eos) < 0:
        raise ValueError("token ID 超出词表")
    n_bytes = sum(len(t.encode("utf-8")) for t in val_texts)
    if n_bytes <= 0:
        raise ValueError("验证原文字节数须为正")
    valid = ids[1:] != bos
    previous, targets = ids[:-1][valid], ids[1:][valid]
    counts = np.bincount(targets, minlength=vocab_size)
    p = counts[counts > 0] / len(targets)
    h1 = float(-np.dot(p, np.log(p)))
    pairs = np.bincount(previous*vocab_size+targets, minlength=vocab_size**2).reshape(vocab_size, vocab_size)
    row_counts = pairs.sum(axis=1)
    # 每行只在 V-1 个可预测类别上平滑并归一化；未见上文退回均匀分布。
    smooth_mass = k*(vocab_size-1)
    val_previous = np.array([a for s in val_seqs for a in s[:-1]], dtype=np.int64)
    val_targets = np.array([b for s in val_seqs for b in s[1:]], dtype=np.int64)
    unigram_nll = float(-np.log((counts[val_targets]+k)/(len(targets)+smooth_mass)).sum())
    bigram_nll = float(-np.log((pairs[val_previous, val_targets]+k)/
                               (row_counts[val_previous]+smooth_mass)).sum())
    def metrics(nll):
        return dict(nat_per_token=nll/len(val_targets), bits_per_byte=nll/(math.log(2)*n_bytes))
    return dict(H1=h1, unigram=metrics(unigram_nll), bigram=metrics(bigram_nll), k=k,
                train_targets=len(targets), n_tokens=len(val_targets), n_bytes=n_bytes,
                n_formulas=len(val_seqs), predictable_classes=vocab_size-1)


def window_logits(model, idx, lengths):
    """最后真实位置投影，与 GPT 前向等价；接收 CPU 窗口，返回 CPU logits。

    设备桥接让共享 evaluate_lm 的 CPU 索引/目标协议也能评测 MPS/CUDA 模型。
    去除整批都不需要的尾部 padding，不改变任何真实位置的因果注意力。
    """
    device = next(model.parameters()).device
    t = int(lengths.max())
    idx, lengths = idx[:, :t].to(device), lengths.to(device)
    x = model.tok(idx)
    if hasattr(model, "pos"):
        x = x + model.pos(torch.arange(t, device=device))
    for block in model.blocks:
        x = block(x, model.cos[:t], model.sin[:t])
    logits = model.head(model.norm(x[torch.arange(len(x), device=device), lengths-1]))
    return logits.float().cpu()


def validation_metrics(model, seqs, texts, bos, eos, context, *, limit=2000, batch=128):
    """seqs 含 BOS/EOS，与 texts 一一对应；返回该前缀的目标加权评测和分母。"""
    if len(seqs) != len(texts) or not seqs or limit < 1:
        raise ValueError("验证序列/原文须非空且逐条对应，limit 须为正")
    seqs, texts = seqs[:limit], texts[:limit]
    windows = FormulaWindows([s[1:-1] for s in seqs], bos, eos, context)
    out = evaluate_lm(model, windows, sum(len(t.encode("utf-8")) for t in texts),
                      batch=batch, window_logits=window_logits)
    return dict(out, n_formulas=len(seqs))


@torch.no_grad()
def initial_loss(model, ids, bos, cfg):
    """固定独立批次上的初始训练 loss；不消耗训练取样 RNG，也不启用 dropout。"""
    was = model.training
    model.eval()
    try:
        x, y = get_batch(ids, cfg.batch, cfg.context, torch.Generator().manual_seed(cfg.seed), cfg.device)
        return model(x, y, loss_mask=y != bos)[1].item()
    finally:
        model.train(was)


def _validate(model, ids, seqs, texts, cfg, bos, eos):
    if (min(cfg.steps, cfg.batch, cfg.context, cfg.accum, cfg.eval_every, cfg.eval_formulas,
            cfg.final_eval_formulas, cfg.eval_batch) < 1 or cfg.batch % cfg.accum or
            not 0 <= cfg.warmup <= cfg.steps or cfg.seed < 0):
        raise ValueError("训练整数配置非法：正步数/批量/上下文，batch 可整除 accum，0<=warmup<=steps，seed>=0")
    if (not all(math.isfinite(v) for v in (cfg.lr, cfg.min_lr, cfg.clip, cfg.weight_decay)) or
            not 0 <= cfg.min_lr <= cfg.lr or cfg.lr <= 0 or min(cfg.clip, cfg.weight_decay) < 0):
        raise ValueError("学习率/裁剪/衰减须有限且范围合法")
    if cfg.amp not in DTYPES or cfg.device not in ("cpu", "mps", "cuda"):
        raise ValueError("amp/device 不支持")
    if cfg.amp == "fp16" and cfg.device != "cuda":
        raise ValueError("本脚本 fp16+GradScaler 仅支持 cuda；CPU/MPS 请用 fp32 或受支持的 bf16")
    if (ids.device.type != "cpu" or ids.dtype != torch.long or ids.ndim != 1 or
            len(ids) <= cfg.context or cfg.context > model.cfg.block_size):
        raise ValueError("训练流须为 CPU 一维 LongTensor，长度大于 context，context 不超过 block_size")
    if ids.min() < 0 or ids.max() >= model.cfg.vocab_size or bos == eos or not 0 <= bos < model.cfg.vocab_size or not 0 <= eos < model.cfg.vocab_size:
        raise ValueError("token/BOS/EOS 编号非法")
    if len(seqs) != len(texts) or not seqs or sum(len(t.encode('utf-8')) for t in texts) <= 0:
        raise ValueError("验证序列和原文须对应且含原文字节")
    if any(len(s) < 2 or s[0] != bos or s[-1] != eos or bos in s[1:] or eos in s[:-1] for s in seqs):
        raise ValueError("验证序列必须逐条 BOS…EOS，内部不能含边界符")


def _identity(model, ids, seqs, texts, cfg, data_version, tokenizer_hash, bos, eos):
    # 版本由人维护；额外内容摘要还可抓住忘记升级版本的错误。
    return dict(model=asdict(model.cfg), training=asdict(cfg), data_version=data_version,
                tokenizer_hash=tokenizer_hash, bos=bos, eos=eos,
                train_hash=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                val_hash=hashlib.sha256(json.dumps([seqs, texts], ensure_ascii=False).encode()).hexdigest())


def _check_identity(saved, expected, prefix=""):
    for key, value in expected.items():
        name = prefix+key
        if isinstance(value, dict) and isinstance(saved.get(key), dict):
            _check_identity(saved[key], value, name+".")
        elif saved.get(key) != value:
            raise ValueError(f"续训配置不一致：{name}；检查点={saved.get(key)!r}；请求={value!r}")


def _rng_snapshot(generator, device):
    # 沿用 checkpoint.set_rng_state 的格式，但 CPU 任务绝不读取 CUDA RNG，
    # 即使机器上有可用 GPU，也不能因保存 CPU 检查点而初始化它。
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state().clone(),
                cuda=torch.cuda.get_rng_state_all() if device == "cuda" else None,
                generators={"data": generator.get_state().clone()})


def train_pretrain(model, train_ids, val_seqs, val_texts, cfg, *, bos, eos, run_dir,
                   data_version, tokenizer_hash="", resume=False, stop_after=None, log=print):
    """教学训练循环；返回 initial_loss/ln_vocab/evals/step/checkpoint，原地更新 model。

    模型初始化由调用者负责；新训练重置全局 RNG 到 cfg.seed。stop_after 是模拟
    中断点，不改变计划 steps，不增加非计划验证；保存未完成验证区间的 loss 累计。
    续训严格校验全部模型/训练配置及数据内容，并恢复 Python/NumPy/Torch/设备 RNG。
    fp16 支持 GradScaler；累积按非 BOS 目标数加权，不能平均各微批 loss。
    """
    _validate(model, train_ids, val_seqs, val_texts, cfg, bos, eos)
    if stop_after is not None and not 1 <= stop_after <= cfg.steps:
        raise ValueError("stop_after 须在 [1,steps] 内")
    identity = _identity(model, train_ids, val_seqs, val_texts, cfg, data_version, tokenizer_hash, bos, eos)
    run_dir = Path(run_dir)
    checkpoint = run_dir / "checkpoint.pt"
    saved = None
    if resume:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        _check_identity(saved["identity"], identity)
    model.to(cfg.device)
    opt = torch.optim.AdamW(_param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp == "fp16")
    g = torch.Generator().manual_seed(cfg.seed)
    start, recent_sum, recent_count, elapsed, history = 0, 0.0, 0, 0.0, []
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {"jsonl": run_dir/"metrics.jsonl", "csv": run_dir/"evals.csv"}
    if saved:
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        set_rng_state(saved["rng"], {"data": g})
        if cfg.device == "mps":
            torch.mps.set_rng_state(saved["mps_rng"])
        start, recent_sum, recent_count = saved["step"], saved["recent_sum"], saved["recent_count"]
        elapsed, history, init = saved["elapsed_sec"], saved["evals"], saved["initial_loss"]
        # 若崩溃前日志先于检查点落盘，删掉尾部；日志缺失则报错，不能伪造历史。
        for name, path in paths.items():
            if not path.exists() or path.stat().st_size < saved["log_positions"][name]:
                raise ValueError(f"续训日志缺失或短于检查点：{path}")
        for name, path in paths.items():
            with path.open("r+b") as handle:
                handle.truncate(saved["log_positions"][name])
    else:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed % 2**32)
        torch.manual_seed(cfg.seed)
        init = initial_loss(model, train_ids, bos, cfg)
        metadata = dict(identity, initial_loss=init, ln_vocab=math.log(model.cfg.vocab_size),
                        non_embedding_params=model.n_params(True), matmul_params=matmul_params(model),
                        planned_tokens=cfg.steps*cfg.batch*cfg.context,
                        planned_6ND=train_flops(matmul_params(model), cfg.steps*cfg.batch*cfg.context))
        (run_dir/"config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        paths["jsonl"].write_text("", encoding="utf-8")
        with paths["csv"].open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=EVAL_FIELDS).writeheader()
    if log:
        log(f"初始loss={init:.4f}；ln V={math.log(model.cfg.vocab_size):.4f}；续训起点={start}")
    end = cfg.steps if stop_after is None else stop_after
    if end < start:
        raise ValueError("stop_after 不能早于已保存步数")
    synchronize(cfg.device)
    t0 = time.perf_counter()
    model.train()
    with paths["jsonl"].open("a", encoding="utf-8") as jf, paths["csv"].open("a", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=EVAL_FIELDS)
        for step in range(start+1, end+1):
            update_start = time.perf_counter()
            lr = warmup_cosine(step-1, cfg.steps, cfg.warmup, cfg.lr, cfg.min_lr)
            for group in opt.param_groups:
                group["lr"] = lr
            opt.zero_grad(set_to_none=True)
            batches = [get_batch(train_ids, cfg.batch//cfg.accum, cfg.context, g, cfg.device)
                       for _ in range(cfg.accum)]
            counts = [int((y != bos).sum()) for _, y in batches]
            valid = sum(counts)
            if valid == 0:
                raise ValueError("本批没有非 BOS 目标")
            train_loss = 0.0
            for (x, y), count in zip(batches, counts):
                if not count:
                    continue
                with autocast(cfg.device, DTYPES[cfg.amp]):
                    loss = model(x, y, loss_mask=y != bos)[1] * (count/valid)
                scaler.scale(loss).backward()
                train_loss += loss.item()
            scaler.unscale_(opt)
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip or math.inf))
            scaler.step(opt)
            scaler.update()
            synchronize(cfg.device)
            update_sec = time.perf_counter()-update_start
            recent_sum += train_loss
            recent_count += 1
            seen = step*cfg.batch*cfg.context
            jf.write(json.dumps(dict(type="train", step=step, lr=lr, train_loss=train_loss,
                                     grad_norm=grad, tokens_seen=seen,
                                     elapsed_sec=elapsed+time.perf_counter()-t0))+"\n")
            do_eval = step % cfg.eval_every == 0 or step == cfg.steps
            if do_eval:
                metrics = validation_metrics(model, val_seqs, val_texts, bos, eos, cfg.context,
                                             limit=cfg.final_eval_formulas if step == cfg.steps else cfg.eval_formulas,
                                             batch=cfg.eval_batch)
                rec = dict(step=step, tokens_seen=seen, train_loss=recent_sum/recent_count,
                           val_nat_per_token=metrics["nat_per_token"], val_bits_per_byte=metrics["bits_per_byte"],
                           lr=lr, tok_per_sec=cfg.batch*cfg.context/max(update_sec, 1e-9),
                           eval_formulas=metrics["n_formulas"], eval_tokens=metrics["n_tokens"], eval_bytes=metrics["n_bytes"])
                history.append(rec)
                writer.writerow(rec)
                jf.write(json.dumps(dict(type="eval", **rec))+"\n")
                if log:
                    log("；".join(f"{k}={format_metric(v, k)}" for k, v in rec.items()))
                recent_sum, recent_count = 0.0, 0
            if do_eval or step == end:
                jf.flush()
                cf.flush()
                state = dict(identity=identity, model=model.state_dict(), optimizer=opt.state_dict(),
                             scaler=scaler.state_dict(), step=step, rng=_rng_snapshot(g, cfg.device),
                             mps_rng=torch.mps.get_rng_state() if cfg.device == "mps" else None,
                             recent_sum=recent_sum, recent_count=recent_count, evals=history, initial_loss=init,
                             elapsed_sec=elapsed+time.perf_counter()-t0,
                             log_positions={k: p.stat().st_size for k, p in paths.items()})
                torch.save(state, str(checkpoint)+".tmp")
                os.replace(str(checkpoint)+".tmp", checkpoint)
    return dict(initial_loss=init, ln_vocab=math.log(model.cfg.vocab_size), evals=history,
                step=end, checkpoint=str(checkpoint))


def wilson_interval(successes, count, z=1.959963984540054):
    """Wilson 95% 区间；空样本返回 (None,None)，不会声称零成功率。"""
    if not 0 <= successes <= count or count < 0 or z <= 0:
        raise ValueError("Wilson 计数/置信参数非法")
    if count == 0:
        return None, None
    p, z2 = successes/count, z*z
    center = (p+z2/(2*count))/(1+z2/count)
    width = z*math.sqrt(p*(1-p)/count+z2/(4*count*count))/(1+z2/count)
    return max(0.0, center-width), min(1.0, center+width)


def parse_report(texts):
    """mathtext 可解析率（可编译率近似），返回计数、比例、Wilson 区间和前五例。"""
    from miniocr.eval.render import can_parse
    texts = list(texts)
    ok = sum(can_parse(t) for t in texts)
    return dict(count=len(texts), successes=ok, rate=ok/len(texts) if texts else None,
                interval=wilson_interval(ok, len(texts)), examples=texts[:5])


# 明确按环境名分桶：matrix/pmatrix/bmatrix/Bmatrix/vmatrix/Vmatrix/smallmatrix，
# 以及 array/cases/aligned/alignedat/align/gathered（允许星号及空格）。
MATRIX_ENV = re.compile(r"\\begin\s*\{\s*(?:[pbBvV]?matrix|smallmatrix|array|cases|aligned(?:at)?|align|gathered)\*?\s*\}")


def reference_parse_report(texts, limit=2000):
    """验证原文本身的参考上限，按矩阵类环境分桶；不是模型可达到的数学上界。"""
    from miniocr.eval.render import can_parse
    texts = list(texts)[:limit]
    rows = [(t, bool(MATRIX_ENV.search(t)), can_parse(t)) for t in texts]
    out = {}
    for label, subset in (("all", rows), ("matrix", [r for r in rows if r[1]]),
                          ("non_matrix", [r for r in rows if not r[1]])):
        n, ok = len(subset), sum(r[2] for r in subset)
        out[label] = dict(count=n, successes=ok, rate=ok/n if n else None,
                          interval=wilson_interval(ok, n))
    return out


def sample_report(model, tokenizer, count=50, temperature=1.0, greedy=False, seed=0, max_new=None):
    """从 BOS 批量采样，禁 BOS、EOS 停止；截断长度不超过模型位置表，恢复训练模式。

    采样在 CPU 副本上执行也适用于从 MPS/CUDA 训练获得的权重，调用者可自行搬到 CPU。
    """
    if count < 1 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("count/temperature 须为正")
    device = next(model.parameters()).device
    bos, eos = (tokenizer.special_ids[k] for k in ("<|bos|>", "<|eos|>"))
    max_new = model.cfg.block_size if max_new is None else max_new
    if not 1 <= max_new <= model.cfg.block_size:
        raise ValueError("max_new 须在 [1,block_size] 内")
    # MPS 没有独立 torch.Generator；fork_rng+手动保存 MPS 状态避免改变训练 RNG。
    was = model.training
    mps_rng = torch.mps.get_rng_state() if device.type == "mps" else None
    try:
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
            torch.manual_seed(seed)
            generator = None if device.type == "mps" else torch.Generator(device=device).manual_seed(seed)
            ids = generate(model, torch.full((count, 1), bos, device=device), max_new,
                           temperature=temperature, greedy=greedy, eos=eos, banned=[bos], generator=generator)
        texts, ended = [], 0
        for row in ids.cpu().tolist():
            content = row[1:]
            if eos in content:
                content = content[:content.index(eos)]
                ended += 1
            texts.append(tokenizer.decode(content))
        return dict(parse_report(texts), ended_with_eos=ended, max_new=max_new)
    finally:
        model.train(was)
        if mps_rng is not None:
            torch.mps.set_rng_state(mps_rng)


def leakage_overlap(train_texts, val_texts):
    """返回训练/验证规范键交集的不同键数（不是重复行数）。"""
    from miniocr.data.latex_corpus import normalization_key
    return len({normalization_key(t) for t in train_texts} & {normalization_key(t) for t in val_texts})


def _read_log(run_dir):
    path = Path(run_dir)
    records = [json.loads(line) for line in (path/"metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads((path/"config.json").read_text(encoding="utf-8")), records


def summarize_log(run_dir, window=50, spike_factor=1.5):
    """报告初值、最佳验证、末尾 train-val 差、相对前 window 步中位数的尖峰与首个非有限步。"""
    if window < 1 or spike_factor <= 1:
        raise ValueError("window>=1，spike_factor>1")
    config, records = _read_log(run_dir)
    train = [r for r in records if r["type"] == "train"]
    ev = [r for r in records if r["type"] == "eval"]
    finite_ev = [r for r in ev if math.isfinite(r["val_nat_per_token"])]
    best = min(finite_ev, key=lambda r: r["val_nat_per_token"], default=None)
    spikes = []
    for i, r in enumerate(train):
        prior = [v["train_loss"] for v in train[max(0, i-window):i] if math.isfinite(v["train_loss"])]
        if prior and math.isfinite(r["train_loss"]) and r["train_loss"] > spike_factor*np.median(prior):
            spikes.append(r["step"])
    bad = next((r["step"] for r in records if any(isinstance(v, (int, float)) and not math.isfinite(v)
                                               for v in r.values())), None)
    return dict(initial_loss=config["initial_loss"], ln_vocab=config["ln_vocab"],
                best_val=best["val_nat_per_token"] if best else None, best_step=best["step"] if best else None,
                final_train_minus_val=ev[-1]["train_loss"]-ev[-1]["val_nat_per_token"] if ev else None,
                spike_steps=spikes, first_nonfinite_step=bad)


def plot_learning_curves(run_dir, window=10):
    """读取 JSONL（逐步训练+验证）画 loss/调度两子图，写 learning_curves.png；返回路径。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if window < 1:
        raise ValueError("window 须为正")
    config, records = _read_log(run_dir)
    train = [r for r in records if r["type"] == "train"]
    ev = [r for r in records if r["type"] == "eval"]
    x, losses = [r["step"] for r in train], [r["train_loss"] for r in train]
    smooth = [np.mean(losses[max(0, i-window+1):i+1]) for i in range(len(losses))]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(x, losses, alpha=.3, label="train (each update)")
    axes[0].plot(x, smooth, label="train rolling mean")
    axes[0].plot([r["step"] for r in ev], [r["val_nat_per_token"] for r in ev], "o-", label="val (prefix varies)")
    axes[0].axhline(config["ln_vocab"], ls="--", color="gray", label="ln V")
    axes[0].set_ylabel("nat/token")
    axes[0].legend()
    axes[1].plot(x, [r["lr"] for r in train], label="lr")
    right = axes[1].twinx()
    right.plot(x, [r["grad_norm"] for r in train], color="orange", label="grad norm (before clip)")
    axes[1].set(xlabel="optimizer step", ylabel="learning rate")
    right.set_ylabel("gradient norm")
    axes[1].legend(loc="upper left")
    right.legend(loc="upper right")
    fig.tight_layout()
    dest = Path(run_dir)/"learning_curves.png"
    fig.savefig(dest)
    plt.close(fig)
    return dest
