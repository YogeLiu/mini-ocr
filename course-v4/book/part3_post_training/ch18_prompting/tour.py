"""第 18 章提示与上下文学习：离线、只推理，所有性能评测只用 val。"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "2"

import numpy as np
import torch

from miniocr.data.postcorrect import (
    ERROR_TYPES, NoiseConfig, build_split, check_no_leak, error_type_counts,
    few_shot_pool, make_example, target_splits,
)
from miniocr.eval.postcorrect_eval import (
    copy_baseline, em_ignoring_final_punct, evaluate, extract_formula, format_issues,
    oracle_baseline, score_postcorrect,
)
from miniocr.models.qwen3 import find_local_qwen3, load_qwen3
from miniocr.prompting import (
    GenerationResult, PARAPHRASES, few_shot, few_shot_strict, generate_greedy, render_chat, render_completion, zero_shot,
)
from miniocr.train.pretrain import wilson_interval

DEFAULT_N = 20
_CONTEXT = dict(device="cpu", dtype="fp32", threads=2, n=DEFAULT_N, batch_size=4, sub_n=None)


def emit(*, origin="本项目观察", **values):
    """每行标明数值来源和实验环境；嵌套指标用 JSON 保留计数和区间。"""
    if origin == "本项目观察":
        origin += "（" + ",".join(f"{k}={v}" for k, v in _CONTEXT.items()) + "）"
    def show(value):
        return json.dumps(value, ensure_ascii=False, default=str)
    print("；".join(f"{k}={show(v)}" for k, v in dict(来源=origin, **values).items()), flush=True)


def read_generation(path, examples, prompts, *, model_path, dtype, device, batch_size=4):
    """严格按顺序核对保存记录；不信任旧 extracted/score，不推算丢失的耗时。

    generation_total_seconds 在每条记录重复保存同一配置的总秒数，读取时只取一次。
    旧文件可全部缺少该字段；部分缺失或值不一致则拒绝，避免混用不同次生成。
    """
    def fail(row, field, reason):
        raise ValueError(f"无法复用 {path}：第 {row} 条字段 {field} {reason}")

    rows = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(i, "JSON", f"无效：{exc.msg}")
            if not isinstance(row, dict):
                fail(i, "record", "必须是对象")
            rows.append(row)
    if len(rows) != len(examples):
        raise ValueError(f"无法复用 {path}：记录数不一致，保存={len(rows)}，本次={len(examples)}")
    for i, (row, ex, prompt) in enumerate(zip(rows, examples, prompts), 1):
        # 新参数引入前固定 batch_size=4；旧记录仍可按原配置复用。
        if row.get("batch_size", 4) != batch_size:
            fail(i, "batch_size", "不一致")
        expected = dict(id=ex.id, prompt=prompt, model_path=str(model_path), dtype=dtype, device=device)
        for field, value in expected.items():
            if field not in row:
                fail(i, field, "缺失")
            if row[field] != value:
                fail(i, field, f"不一致：保存={row[field]!r}，本次={value!r}")
        for field, kind in (("raw", str), ("pred", str), ("prompt_tokens", int),
                            ("generated_tokens", int), ("length_capped", bool)):
            if field not in row:
                fail(i, field, "缺失")
            if type(row[field]) is not kind or (kind is int and row[field] < 0):
                fail(i, field, "类型或取值无效")
    seconds = rows[0].get("generation_total_seconds") if rows else None
    for i, row in enumerate(rows, 1):
        value = row.get("generation_total_seconds")
        if (("generation_total_seconds" in row) != ("generation_total_seconds" in rows[0])
                or value != seconds):
            fail(i, "generation_total_seconds", "各条记录不一致")
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            fail(i, "generation_total_seconds", "必须是非负有限秒数或 null")
    return GenerationResult(
        texts=tuple(r["pred"] for r in rows), raw_texts=tuple(r["raw"] for r in rows),
        prompt_tokens=tuple(r["prompt_tokens"] for r in rows),
        generated_tokens=tuple(r["generated_tokens"] for r in rows), seconds=seconds,
        length_capped=tuple(r["length_capped"] for r in rows))


class Experiment:
    """按需加载 tokenizer/模型/全量 val；--reuse 读已有文件，缺文件才生成。"""
    def __init__(self, args):
        self.args = args
        self.path = find_local_qwen3()
        self._tokenizer = self._model = self._val = None
        self.cache = {}

    def available(self):
        if self.path is None:
            emit(origin="待验证", 状态="跳过：未找到本地 Qwen3-0.6B 权重；不会自动下载")
            return False
        return True

    def tok(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.path), local_files_only=True)
        return self._tokenizer

    def model(self):
        if self._model is None:
            dtype = torch.float32 if self.args.dtype == "fp32" else torch.bfloat16
            self._model, self._tokenizer = load_qwen3(self.path, self.args.device, dtype)
        return self._model

    def val(self, n=None):
        if self._val is None:
            self._val = build_split("val", seed=self.args.seed)
        return self._val[:n]

    def pool(self, k=3, seed=None):
        return few_shot_pool(k, self.args.seed if seed is None else seed)

    def run(self, mode, n, *, instruction=None, pool_seed=None, reverse=False):
        if mode == "zero_shot":
            instruction = instruction or PARAPHRASES[0]
        key = (mode, n, instruction, pool_seed, reverse)
        if key in self.cache:
            return self.cache[key]
        examples = self.val(n)
        pool = self.pool(seed=pool_seed) if mode != "zero_shot" else []
        if reverse:
            pool = list(reversed(pool))
        prompts = []
        for ex in examples:
            if mode == "completion":
                prompt = render_completion(ex.source, pool)
            else:
                messages = (zero_shot(ex.source, instruction or PARAPHRASES[0]) if mode == "zero_shot"
                            else few_shot_strict(ex.source, pool) if mode == "few_shot_strict"
                            else few_shot(ex.source, pool))
                prompt = render_chat(self.tok(), messages)
            prompts.append(prompt)
        path = None
        if self.args.out:
            suffix = f"{mode}-n{n}-p{PARAPHRASES.index(instruction) if instruction else 0}-seed{pool_seed}-rev{reverse}"
            path = self.args.out / f"{suffix}.jsonl"
        reused = bool(getattr(self.args, "reuse", False) and path is not None and path.exists())
        if reused:
            result = read_generation(path, examples, prompts, model_path=self.path,
                                     dtype=self.args.dtype, device=self.args.device,
                                     batch_size=self.args.batch_size)
            emit(复用=f"已保存生成（{path}）", 原生成用时=result.seconds if result.seconds is not None else "未记录",
                 时间口径="原记录；不含本次抽取/评分")
        else:
            result = generate_greedy(self.model(), self.tok(), prompts, sources=[e.source for e in examples],
                                     batch_size=self.args.batch_size, stop_at_newline=(mode == "completion"))
        raw = score_postcorrect(examples, result.texts)
        extracted = score_postcorrect(examples, [extract_formula(s) for s in result.texts])
        data = dict(result=result, raw=raw, extracted=extracted, reused_from=path if reused else None)
        self.cache[key] = data
        if path is not None and not reused:
            self.args.out.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                for i, (ex, row) in enumerate(zip(examples, extracted["rows"])):
                    record = dict(id=ex.id, source=ex.source, target=ex.target, error_types=ex.error_types,
                                  example=ex.to_dict(), raw=result.raw_texts[i], pred=result.texts[i],
                                  extracted=row["pred"], score=row, prompt=prompts[i],
                                  prompt_tokens=result.prompt_tokens[i], generated_tokens=result.generated_tokens[i],
                                  length_capped=result.length_capped[i], seed=self.args.seed,
                                  generation_total_seconds=result.seconds,
                                  model_path=str(self.path), device=self.args.device, dtype=self.args.dtype,
                                  device_name=(torch.cuda.get_device_name() if self.args.device == "cuda" else self.args.device),
                                  batch_size=self.args.batch_size, sub_n=self.args.sub_n,
                                  threads=torch.get_num_threads(), mode=mode)
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return data


def report(mode, data):
    result = data["result"]
    n = len(result.texts)
    timing = (dict(复用=f"已保存生成（{data['reused_from']}）",
                   原生成用时=result.seconds if result.seconds is not None else "未记录")
              if data.get("reused_from") else dict(生成总秒=result.seconds))
    emit(方法=mode, 实际n=n, **timing,
         每条平均秒=result.seconds/n if n and result.seconds is not None else None,
         平均提示token=np.mean(result.prompt_tokens).item() if n else None,
         平均生成token=np.mean(result.generated_tokens).item() if n else None,
         达到长度上限=sum(result.length_capped),
         时间口径="原记录；生成含编解码，不含加载/评分" if data.get("reused_from") else "生成含编解码，不含加载/评分；进程内缓存列首次耗时")
    emit(方法=mode, 评分="原始输出直接评分（completion 已按换行截断）", 指标=data["raw"]["summary"])
    emit(方法=mode, 评分="extract_formula 后评分", 指标=data["extracted"]["summary"])


def part1(exp):
    splits = target_splits()
    emit(数据协议="latex-v1 / postcorrect-v1", 划分大小={k: len(v) for k, v in splits.items()})
    # 用明确含可施加位置的教学公式逐一隔离错误；不是评测集成绩。
    targets = (r"l+O+\times", r"x^{2}", r"\alpha+\beta", r"\frac{a}{b}", r"x^{2}", "x+y")
    for kind, target in zip(ERROR_TYPES, targets):
        ex = make_example(target, "val", 0, exp.args.seed, NoiseConfig(0, {kind: 1}, 1))
        emit(教学错误类型=kind, source=ex.source, target=ex.target, errors=ex.to_dict()["errors"])
    val = exp.val()
    emit(统计范围="val 全集", 实际n=len(val), 错误类型样本数=error_type_counts(val),
         目标公式字符长度均值=float(np.mean([len(e.target) for e in val])),
         目标公式字符长度中位数=float(np.median([len(e.target) for e in val])),
         字符长度口径="目标原串的 Python len，含空白和 LaTeX 命令字符，未规范化",
         目标句末标点={p: dict(count=sum(e.target.rstrip().endswith(p) for e in val),
                              total=len(val), value=sum(e.target.rstrip().endswith(p) for e in val)/len(val))
                       for p in (".", ",", ";")},
         句末统计口径="目标原串 rstrip 后的最后字符，未规范化；分母为 val 全集",
         无错比例=sum(e.is_clean for e in val)/len(val), 平均错误数=sum(len(e.errors) for e in val)/len(val))
    emit(泄漏检查=check_no_leak(splits), 说明="检查干净目标的规范键，无需对 train 全量加噪；test 仅查身份不评分")


def part2(exp):
    for label, examples in (("val 全集", exp.val()), ("val 前 n 条", exp.val(exp.args.n))):
        emit(范围=label, 基线="copy", 指标=copy_baseline(examples)["summary"])
        emit(范围=label, 基线="oracle", 指标=oracle_baseline(examples)["summary"])
    emit(origin="数学结论", 说明="oracle 的 EM=1、CER=0；copy 对无错样本的改坏率=0。parse_rate 不保证=1，copy 是比较锚点而非所有指标必须严格超过。")


def part3(exp):
    if not exp.available():
        return
    tok = exp.tok()
    for thinking in (False, True):
        text = render_chat(tok, zero_shot(r"\frac{a}{b}"), enable_thinking=thinking)
        ids = tok.encode(text, add_special_tokens=False)
        emit(enable_thinking=thinking, 渲染repr=repr(text), token_ids=ids, token_strings=tok.convert_ids_to_tokens(ids))
    emit(special_tokens=tok.special_tokens_map, bos_token=tok.bos_token,
         说明="False 在 assistant 开头预填 <think>\\n\\n</think>\\n\\n；True 不预填。权重是 Qwen3-0.6B chat，不是 Base。")


def subset_n(exp, default):
    """显式子集仍取主实验前缀内的样本；未指定时沿用教学规模。"""
    return min(exp.args.n, default if exp.args.sub_n is None else exp.args.sub_n)


def part4(exp):
    if not exp.available():
        return
    for mode in ("completion", "few_shot"):
        report(mode, exp.run(mode, subset_n(exp, 30)))
    emit(说明="两种输入使用相同任务指令、train 示例与 val 子集；completion 在首个换行处截断，完整续写可用 --out 保存。")


def part5(exp):
    if not exp.available():
        return
    for mode in ("zero_shot", "few_shot", "few_shot_strict"):
        report(mode, exp.run(mode, exp.args.n))


def report_final_punct_diagnostic(mode, scored, *, scoring):
    """独立报告句末标点诊断；改坏沿用无错 source 的 broken_rate 分桶。"""
    rows = scored["rows"]
    matches = [em_ignoring_final_punct(r["target"], r["pred"]) for r in rows]
    successes, n = sum(matches), len(rows)
    broken = [ok for r, ok in zip(rows, matches) if r["is_clean"] and not r["em_norm"]]
    emit(方法=mode, 评分=scoring, 诊断="非官方指标，只用于错例分析；官方指标仍是 EM_norm",
         官方EM_norm=scored["summary"]["em_norm"],
         忽略句末标点的EM_norm=dict(value=successes/n if n else None, successes=successes,
                                    count=n, interval=wilson_interval(successes, n)),
         改坏样本中仅差句末标点的条数=sum(broken), 改坏样本数=len(broken),
         改坏口径="source 原本无错且本次评分 EM_norm 错（broken_rate）；不是抽取改坏",
         比较规则="target/pred 均先规范化，再去掉 token 序列末尾全部 . , ;")


def part6(exp):
    if not exp.available():
        return
    emit(抽取修好定义="原始输出 em_norm 错、抽取后对的样本数",
         抽取改坏定义="原始输出 em_norm 对、抽取后错的样本数",
         变化口径="同一批全部样本；变化=抽取后-抽取前（比例，非百分比）；不同于按 source 分桶的 fix_rate/broken_rate")
    report_final_punct_diagnostic("copy", copy_baseline(exp.val(exp.args.n)),
                                 scoring="照抄 source，不抽取")
    for mode in ("zero_shot", "few_shot", "few_shot_strict"):
        data = exp.run(mode, exp.args.n)
        counts = Counter({k: 0 for k in ("empty", "has_explanation", "dollar_wrapped", "code_fence", "think_block")})
        for raw in data["result"].texts:
            counts.update(format_issues(raw))
        pairs = list(zip(data["raw"]["rows"], data["extracted"]["rows"]))
        changes = {}
        for metric in ("parse_rate", "em_norm"):
            before = data["raw"]["summary"][metric]["value"]
            after = data["extracted"]["summary"][metric]["value"]
            changes[metric] = dict(抽取前=before, 抽取后=after,
                                   变化=after-before if before is not None and after is not None else None)
        emit(方法=mode, 格式问题=counts,
             抽取修好=sum(not a["em_norm"] and b["em_norm"] for a, b in pairs),
             抽取改坏=sum(a["em_norm"] and not b["em_norm"] for a, b in pairs),
             **changes,
             原始输出典型例子=[r for r in data["result"].texts if format_issues(r)][:3])
        report_final_punct_diagnostic(mode, data["extracted"], scoring="extract_formula 后评分")
    emit(说明="格式诊断是启发式，空 think 预填属于提示，不计入模型生成输出。抽取器不是正确性裁判。")


def part7(exp):
    if not exp.available():
        return
    tok, pool, examples = exp.tok(), exp.pool(8), exp.val(exp.args.n)
    base = None
    for k in (0, 1, 3, 5, 8):
        lengths = [len(tok.encode(render_chat(tok, few_shot(e.source, pool, k)), add_special_tokens=False)) for e in examples]
        average = float(np.mean(lengths))
        if base is None:
            base = average
        emit(k=k, 提示token平均=average, 提示token最大=max(lengths), 每个示例平均增量token=(average-base)/k if k else None)
    cfg = json.loads((exp.path/"config.json").read_text())
    emit(max_position_embeddings=cfg["max_position_embeddings"], 说明="配置上限不等于实际可用预算；还须给生成留位置。只加载 tokenizer/config。")


def part8(exp):
    if not exp.available():
        return
    scores = []
    for instruction in PARAPHRASES:
        data = exp.run("zero_shot", subset_n(exp, 50), instruction=instruction)
        summary = data["extracted"]["summary"]
        scores.append(summary)
        emit(指令=instruction, 实际n=summary["n"], em_norm=summary["em_norm"], cer=summary["cer"], broken_rate=summary["broken_rate"])
    emit(最大最小差={k: max(s[k]["value"] for s in scores)-min(s[k]["value"] for s in scores)
                      for k in ("em_norm", "cer")},
         说明="同一子集上的配对比较，不能只凭各自区间是否重叠判显著；第 22 章做配对 bootstrap。")


def part9(exp):
    if not exp.available():
        return
    for label, kwargs in (("原示例顺序", {}), ("逆序", {"reverse": True}), ("另一组", {"pool_seed": exp.args.seed+1})):
        data = exp.run("few_shot", subset_n(exp, 30), **kwargs)
        emit(示例选择=label, 示例ids=[e.id for e in (list(reversed(exp.pool())) if kwargs.get("reverse") else exp.pool(seed=kwargs.get("pool_seed")))])
        report(label, data)


def part10(exp):
    try:
        evaluate(lambda sources: sources, split="test")
    except ValueError as exc:
        emit(协议守卫=str(exc), 说明="在加载数据/调用预测函数前拒绝；未评测 test")


PARTS = {
    1: ("合成数据与泄漏检查", part1), 2: ("指标锚点", part2),
    3: ("chat template", part3), 4: ("直接续写与 chat", part4),
    5: ("三种提示基线", part5), 6: ("输出格式问题", part6),
    7: ("上下文占用", part7), 8: ("措辞敏感性", part8),
    9: ("示例选择与顺序", part9), 10: ("评测协议", part10),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="*", type=int)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--batch-size", type=int, default=4, help="生成批大小，默认 4")
    parser.add_argument("--sub-n", type=int, help="第 4/8/9 部分的 val 前缀大小（不超过 n）；默认分别最多 30/50/30")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="预测 JSONL 目录，含样本身份、错误类型、完整原输出及抽取结果")
    parser.add_argument("--reuse", action="store_true", help="校验并复用 --out 中已有的生成；缺文件时生成，不一致时报错")
    args = parser.parse_args(argv)
    if args.reuse and args.out is None:
        parser.error("--reuse 必须搭配 --out")
    if args.n < 1 or any(p not in PARTS for p in args.parts):
        parser.error("n 必须为正，部分号须在 1 至 10 之间")
    if args.batch_size < 1 or (args.sub_n is not None and args.sub_n < 1):
        parser.error("batch-size 和 sub-n 必须为正")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA 不可用，请使用 --device cpu")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS 不可用，请使用 --device cpu")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[name] = "1"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "2"
    torch.set_num_threads(2)
    device_name = torch.cuda.get_device_name() if args.device == "cuda" else args.device
    _CONTEXT.update(device=f"cuda({device_name})" if args.device == "cuda" else args.device,
                    dtype=args.dtype, threads=2, n=args.n, batch_size=args.batch_size, sub_n=args.sub_n)
    if args.device != "cpu" or args.dtype != "fp32":
        emit(说明="本次配置单独记录；设备、精度和左 padding 批大小变化可能使个别输出不同")
    import transformers
    start = time.perf_counter()
    exp = Experiment(args)
    emit(torch版本=torch.__version__, transformers版本=transformers.__version__, 权重快照=exp.path,
         seed=args.seed, 默认n=DEFAULT_N, GPU名称=device_name if args.device == "cuda" else None,
         batch_size=args.batch_size, sub_n=args.sub_n,
         生成预算="每批 min(256,2*最长源token+32)", 离线=True)
    for p in args.parts or PARTS:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        title, fn = PARTS[p]
        emit(部分=p, 标题=title)
        begun = time.perf_counter()
        fn(exp)
        emit(部分=p, 用时秒=time.perf_counter()-begun)
    emit(tour总用时秒=time.perf_counter()-start)


if __name__ == "__main__":
    main()
