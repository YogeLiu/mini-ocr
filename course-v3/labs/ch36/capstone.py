"""第 36 章：毕业项目的工具。分析工具的演示（合成数据）、逐样本评测、自动生成报告的表格。

    python -m labs.ch36.capstone 1                       # 分析工具的演示：用合成的逐样本结果（不是实验结果），几秒
    python -m labs.ch36.capstone 4 --seed 0               # 你自己跑：选题 A 的流水线（SFT → 继续 SFT / DPO / GRPO），每个种子一次
                                                         # 加 --quick：每段只跑几步，确认流水线能跑通（需要第 22 章的阶段 1 模型）
    python -m labs.ch36.capstone 5                       # 选题 B 的起点：公式数据按近似重复分组划分 + SmolVLM-256M 零样本基线（CPU）
    python -m labs.ch36.capstone 2 --model 路径 --name 名字 --seed 0
                                                         # 逐样本评测第 26 章格式的模型，结果写进 outputs/ch36/runs/名字__s种子__val.jsonl
    python -m labs.ch36.capstone 3 --baseline 名字        # 汇总 runs/ 里的全部 val 结果，生成 outputs/ch36/report_tables_val.md（加 --test 汇总测试集）
    python -m labs.ch36.capstone 2 --model 路径 --name 名字 --seed 0 --test
                                                         # 方案冻结之后：在测试集上评测（每个运行只允许一次，见 TestGuard）
"""
import glob
import json
import os
import sys
from collections import defaultdict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

from miniocr.research import TestGuard, bucket_report, compare, manifest, md_table, seed_table

OUT = "outputs/ch36"
RUNS = os.path.join(OUT, "runs")
os.makedirs(RUNS, exist_ok=True)


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


# ---------------------------------------------------------------------- 1
def synthetic_runs(seed=0, n=300, seeds=5):
    """合成的逐样本结果，用来演示分析工具。设定：SFT 有"相变"——每个种子以 60% 的概率学会（准确率 0.85），
    否则 0.55；"SFT + 新方法"在同一个种子上每个样本多 4% 的概率答对。两个桶：短（容易）、长（难，准确率低 0.2）。"""
    rng = np.random.default_rng(seed)
    bucket = np.where(rng.random(n) < 0.5, "短", "长")
    runs = {}
    for s in range(seeds):
        r = np.random.default_rng(100 + s)
        base = 0.85 if r.random() < 0.6 else 0.55
        p = np.clip(base + np.where(bucket == "长", -0.1, 0.1), 0, 1)
        u = r.random(n)
        runs[("SFT", s)] = (u < p).astype(float)
        runs[("SFT + 新方法", s)] = (u < np.clip(p + 0.04, 0, 1)).astype(float)
    return runs, bucket


def part1_demo():
    section("1. 分析工具的演示（合成数据，不是实验结果）：种子噪声、配对比较、分桶")
    runs, bucket = synthetic_runs()
    print("  设定：SFT 每个种子 60% 的概率\"学会\"（准确率约 0.85），否则约 0.55；新方法在同一种子上每个样本多 4% 的概率答对")
    a, b = runs[("SFT + 新方法", 0)], runs[("SFT", 1)]
    r = compare(a, b, names=("新方法（种子 0）", "SFT（种子 1）"))
    print(f"  只跑一个种子、而且两边种子不同：{r['mean_a']:.1%} vs {r['mean_b']:.1%}，差 {r['diff']:+.1%}"
          f"（95% 区间 {r['lo']:+.1%} ~ {r['hi']:+.1%}）→ {r['verdict']}")
    rows = seed_table([(k[0], k[1], v.mean()) for k, v in runs.items()])
    print("  5 个种子：")
    for name, n, m, sd, lo, hi, ok in rows:
        print(f"    {name:<10} n = {n}  均值 {m:.1%}  标准差 {sd:.1%}  最小 {lo:.1%}  最大 {hi:.1%}")
    print("  同一个种子上的配对比较（新方法 − SFT）：")
    for s in range(5):
        r = compare(runs[("SFT + 新方法", s)], runs[("SFT", s)], names=("新方法", "SFT"))
        print(f"    种子 {s}：{r['mean_a']:.1%} vs {r['mean_b']:.1%}，差 {r['diff']:+.1%}（{r['lo']:+.1%} ~ {r['hi']:+.1%}）→ {r['verdict']}")
    print("  分桶（新方法，种子 0）：")
    for k, n, m, lo, hi in bucket_report(runs[("SFT + 新方法", 0)], bucket):
        print(f"    {k}  n = {n:3d}  {m:.1%}（{lo:.1%} ~ {hi:.1%}）")


# ---------------------------------------------------------------------- 2
def bucket_of(task, ref):
    if task == "text":
        return f"text·{'≤3 字' if len(ref) <= 3 else ('4 字' if len(ref) == 4 else '≥5 字')}"   # 第 23 章的场景文字：中位数 4 个字
    return task


def per_sample_eval(m, n=100, test=False):
    """第 26 章格式的模型，在六个任务上逐样本评测。验证集用第 23 章的测试种子（1000 + 任务编号），
    测试集用另一组种子（5000 + 任务编号）——方案冻结之前不要碰测试集。返回 [{task, bucket, ref, pred, score}, …]。"""
    from labs.ch23.instruct_tour import TASKS, ask, prompt, test_examples
    from labs.ch26.sft_tour import EVAL_TASKS
    from miniocr.boxes import box_iou
    from miniocr.data.instruct import parse_box
    from miniocr.metrics import ned
    import torch
    out = []
    for t in EVAL_TASKS:
        X, qs, refs = test_examples(t, n=n, seed=(5000 if test else 1000) + TASKS.index(t))
        pred = ask(m, X, [prompt(q) for q in qs], max_new=90 if t == "caption" else 20)
        for p, r in zip(pred, refs):
            if t == "ground":
                if r == "none" or parse_box(p) is None:
                    s = float(p == r)
                else:
                    s = float(box_iou(torch.tensor([parse_box(p)], dtype=torch.float),
                                      torch.tensor([parse_box(r)], dtype=torch.float))[0, 0] >= 0.5)
            elif t == "caption":
                s = max(0.0, 1 - ned(p, r))                              # 描述用 1 − NED（完全一样太难）
            else:
                s = float(p == r)
            out.append({"task": t, "bucket": bucket_of(t, r), "ref": r, "pred": p, "score": s})
    return out


def part2_eval():
    section("2. 逐样本评测（你自己跑：需要训练好的模型）")
    import torch
    from labs.ch22.vlm_tour import build, need
    path, name, seed = arg("--model"), arg("--name"), int(arg("--seed", 0))
    if not path or not name:
        print("  用法：python -m labs.ch36.capstone 2 --model outputs/ch26/sft.pt --name sft --seed 0 [--test]")
        return
    need(path)
    test = "--test" in sys.argv
    man = manifest({"model_file": path, "split": "test" if test else "val"}, [path], seed)
    if test:
        TestGuard(os.path.join(OUT, "test_log.json")).check(f"{name}__s{seed}", reason=arg("--reason"))
    m = build("mlp")
    m.load_state_dict(torch.load(path, map_location="cpu"))
    recs = per_sample_eval(m.eval(), n=int(arg("--n", 100)), test=test)
    write_run(name, seed, "test" if test else "val", recs, man)


def write_run(name, seed, split, recs, man, extra=None):
    """逐样本结果的统一格式：第一行 {"manifest": …, "extra": {…}}，之后每行 {task, bucket, ref, pred, score}。
    extra 放整个模型一个数的指标（幻觉率、探测 KL……），第 3 部分单独列一张表。
    第 35 章的真实模型（或者你自己的任何模型）只要写成这个格式，第 3 部分就能汇总。"""
    f = os.path.join(RUNS, f"{name}__s{seed}__{split}.jsonl")
    with open(f, "w") as fh:
        fh.write(json.dumps({"manifest": man, "extra": extra or {}}, ensure_ascii=False) + "\n")
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    by = defaultdict(list)
    for r in recs:
        by[r["task"]].append(r["score"])
    print(f"  {name}（种子 {seed}，{split}，id {man['id']}）：" + "  ".join(f"{t} {np.mean(v):.1%}" for t, v in by.items()))
    print(f"  逐样本结果写进 {f}")
    return f


# ---------------------------------------------------------------------- 3
def load_runs(split="val"):
    runs = {}
    for f in sorted(glob.glob(os.path.join(RUNS, f"*__{split}.jsonl"))):
        name, s, _ = os.path.basename(f)[:-6].split("__")
        with open(f) as fh:
            lines = [json.loads(x) for x in fh]
        runs[(name, int(s[1:]))] = (lines[0]["manifest"], lines[1:])
        EXTRA[(name, int(s[1:]))] = lines[0].get("extra", {})
    return runs


EXTRA = {}


def part3_report(split=None, baseline=None):
    split = split or ("test" if "--test" in sys.argv else "val")
    baseline = baseline or arg("--baseline")
    section(f"3. 汇总报告的表格（{split}）")
    runs = load_runs(split)
    if not runs:
        print(f"  {RUNS} 里没有 {split} 的结果：先用第 2 部分评测")
        return
    tasks = sorted({r["task"] for _, recs in runs.values() for r in recs})
    parts = [f"# 结果表（{split}，自动生成）\n"]
    rows = []
    for (name, s), (man, recs) in sorted(runs.items()):
        by = defaultdict(list)
        for r in recs:
            by[r["task"]].append(r["score"])
        rows.append([name, s, man["id"]] + [f"{np.mean(by[t]):.1%}" for t in tasks] +
                    [f"{np.mean([r['score'] for r in recs]):.1%}"])
    parts.append("## 每个运行\n\n" + md_table(["配置", "种子", "运行 id"] + tasks + ["全部"], rows))
    st = seed_table([(name, s, np.mean([r["score"] for r in recs])) for (name, s), (_, recs) in sorted(runs.items())])
    parts.append("## 种子\n\n" + md_table(["配置", "种子数", "均值", "标准差", "最小", "最大", "种子 ≥ 3"],
                                          [[a, n, f"{m:.1%}", "—" if np.isnan(sd) else f"{sd:.1%}", f"{lo:.1%}", f"{hi:.1%}",
                                            "是" if ok else "**否**"] for a, n, m, sd, lo, hi, ok in st]))
    keys = sorted({k for (n, s) in runs for k in EXTRA.get((n, s), {})})
    if keys:
        parts.append("## 整体指标\n\n" + md_table(["配置", "种子"] + keys,
                     [[n, s] + [f"{EXTRA[(n, s)][k]:.4f}" if k in EXTRA[(n, s)] else "—" for k in keys] for (n, s) in sorted(runs)]))
    if baseline:
        cmp_rows = []
        for (name, s), (_, recs) in sorted(runs.items()):
            if name == baseline or (baseline, s) not in runs:
                continue
            base = runs[(baseline, s)][1]
            if [r["ref"] for r in base] != [r["ref"] for r in recs]:
                cmp_rows.append([name, s, "—", "—", "样本不同，不能配对"])
                continue
            r = compare([x["score"] for x in recs], [x["score"] for x in base], names=(name, baseline))
            cmp_rows.append([name, s, f"{r['diff']:+.1%}", f"{r['lo']:+.1%} ~ {r['hi']:+.1%}", r["verdict"]])
        parts.append(f"## 和 {baseline} 的配对比较（同一个种子）\n\n" +
                     md_table(["配置", "种子", "差", "95% 区间", "结论"], cmp_rows))
    for (name, s), (_, recs) in sorted(runs.items()):
        br = bucket_report([r["score"] for r in recs], [r["bucket"] for r in recs])
        parts.append(f"## 分桶：{name}（种子 {s}）\n\n" +
                     md_table(["桶", "样本数", "均值", "95% 区间"], [[k, n, f"{m:.1%}", f"{lo:.1%} ~ {hi:.1%}"] for k, n, m, lo, hi in br]))
    f = os.path.join(OUT, f"report_tables_{split}.md")
    with open(f, "w") as fh:
        fh.write("\n\n".join(parts) + "\n")
    print(f"  {len(runs)} 个运行；表格写进 {f}")


# ---------------------------------------------------------------------- 4
def part4_project_a(seed=None, sft_steps=2000, cont_steps=200, dpo_steps=200, rl_steps=200, n_eval=100):
    """选题 A 的流水线（一个种子）：阶段 1 → SFT → 三种继续训练（继续 SFT / DPO / GRPO，步数相同），
    每个模型存进 outputs/ch36/models/，逐样本评测写进 runs/。整体指标：描述的幻觉率与全对率（第 30 章的
    hallucination_eval），以及相对 SFT 起点的探测 KL（第 32 章，衡量"离起点多远"，遗忘的代理指标）。"""
    import copy
    import torch
    from labs.ch26.sft_tour import Pool, load_stage1, train
    from labs.ch30 import dpo_tour as c30
    from labs.ch31 import grpo_tour as c31
    from labs.ch32.forget_tour import probe_set, vlm_probe_kl
    seed = int(arg("--seed", 0)) if seed is None else seed
    if "--quick" in sys.argv:                                     # 冒烟测试：几步，只确认流水线能跑通
        sft_steps, cont_steps, dpo_steps, rl_steps, n_eval = 4, 2, 2, 2, 5
    section(f"4. 选题 A 的流水线，种子 {seed}：SFT {sft_steps} 步 → 继续 SFT / DPO / GRPO 各 {cont_steps}/{dpo_steps}/{rl_steps} 步")
    models = os.path.join(OUT, "models")
    os.makedirs(models, exist_ok=True)
    cfg = dict(sft_steps=sft_steps, cont_steps=cont_steps, dpo_steps=dpo_steps, rl_steps=rl_steps)

    probe = probe_set()

    def finish(name, m, extra):
        f = os.path.join(models, f"{name}__s{seed}.pt")
        torch.save(m.state_dict(), f)
        h, e = c30.hallucination_eval(m, n=n_eval * 2)
        k = vlm_probe_kl(m, sft, probe) if name != "sft" else 0.0
        write_run(name, seed, "val", per_sample_eval(m.eval(), n=n_eval), manifest(dict(cfg, **extra), [f], seed),
                  {"hallucination": h, "caption_exact": e, "probe_kl": float(k)})
        print(f"    描述中的幻觉 {h:.1%}，描述全对 {e:.1%}，探测 KL {k:.4f}")

    m = load_stage1()
    train(m, sft_steps, Pool(None, 10 + seed), seed=seed)
    sft = m
    finish("sft", m, {"stage": "sft"})
    m = copy.deepcopy(sft)
    train(m, cont_steps, Pool(None, 100 + seed), lr=1e-4, seed=seed)
    finish("sft+cont", m, {"stage": "cont", "lr": 1e-4})
    m = copy.deepcopy(sft)
    c30.train_dpo(m, dpo_steps, 0.1, seed=seed, log_every=0)
    finish("sft+dpo", m, {"stage": "dpo", "beta": 0.1, "lr": 1e-5})
    m = copy.deepcopy(sft)
    c31.grpo_train(m, c31.RLConfig(steps=rl_steps, lr=1e-5, seed=seed, log_every=0))
    finish("sft+grpo", m, {"stage": "grpo", "lr": 1e-5})


# ---------------------------------------------------------------------- 5
FORMULA_Q = "Convert the formula in the image to LaTeX."


def formula_split(n_pool=20000, near=0.8, seed=0):
    """选题 B 的数据划分：latex-formulas 第 1 个分片，按 latex_key 精确去重后取前 n_pool 条；
    Jaccard ≥ near 的近似重复用并查集连成组，按组划分 训练 90% / 验证 5% / 测试 5%（第 33 章的 group_split）。
    返回 {train, val, test: 分片里的行号}，和数据文件的路径。"""
    from labs.ch33.engine_tour import load_latex, LATEX_DIR
    from miniocr.engine.clean import dedup, group_split, jaccard, latex_key, lsh_pairs, minhash, shingles
    L = load_latex()
    if L is None:
        return None, None, None
    keep = dedup(L, key=latex_key)[:n_pool]
    U = [latex_key(L[i]) for i in keep]
    sh = [shingles(u, 5) for u in U]
    pairs = [(i, j) for i, j in lsh_pairs(np.stack([minhash(x) for x in sh]), bands=16) if jaccard(sh[i], sh[j]) >= near]
    parent = list(range(len(U)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j in pairs:
        parent[find(i)] = find(j)
    groups = [find(i) for i in range(len(U))]
    tr, rest = group_split(groups, 0.1, seed=seed)
    va_, te_ = group_split([groups[i] for i in rest], 0.5, seed=seed + 1)
    split = {"train": [keep[i] for i in tr], "val": [keep[rest[i]] for i in va_], "test": [keep[rest[i]] for i in te_]}
    side = {}
    for name, idx in (("train", tr), ("val", [rest[i] for i in va_]), ("test", [rest[i] for i in te_])):
        for i in idx:
            side[i] = name
    cross = sum(side[i] != side[j] for i, j in pairs)
    f = sorted(glob.glob(LATEX_DIR + "/*.parquet"))[0]
    return split, {"n_pool": len(U), "near_pairs": len(pairs), "groups": len(set(groups)), "cross": cross}, f


def read_rows(f, rows):
    """按行号读 parquet 里的图和公式（只读需要的 row group）。"""
    import io
    import pyarrow.parquet as pq
    from PIL import Image
    pf = pq.ParquetFile(f)
    starts = np.cumsum([0] + [pf.metadata.row_group(g).num_rows for g in range(pf.num_row_groups)])
    out, cache = [], {}
    for r in rows:
        g = int(np.searchsorted(starts, r, side="right") - 1)
        if g not in cache:
            cache = {g: pf.read_row_group(g).to_pylist()}
        x = cache[g][r - starts[g]]
        out.append((Image.open(io.BytesIO(x["image"]["bytes"])).convert("RGB"), x["latex_formula"]))
    return out


def formula_bucket(ref):
    from miniocr.engine.clean import latex_key
    n = len(latex_key(ref))
    return "≤20 字符" if n <= 20 else ("21–60 字符" if n <= 60 else "> 60 字符")


def formula_score(pred, ref):
    """选题 B 的逐样本分数：latex_key 归一化之后的 1 − NED（第 33 章）。完全相同另记。"""
    from miniocr.engine.clean import latex_key
    from miniocr.metrics import ned
    a, b = latex_key(pred), latex_key(ref)
    return max(0.0, 1 - ned(a, b)), float(a == b)


def part5_project_b(n_eval=50, n_pool=20000, max_new=160):
    """选题 B 的起点：数据划分（按近似重复分组）+ SmolVLM-256M 的零样本基线（E0），逐样本结果写进 runs/。
    训练（LoRA SFT、GRPO）照第 35 章第 6、7 部分改：把 line_data 换成 read_rows(f, split["train"])，把问题换成 FORMULA_Q。"""
    import time
    import torch
    from labs.ch35.real_tour import load_smol
    if "--quick" in sys.argv:
        n_eval, n_pool = 3, 2000
    test = "--test" in sys.argv
    section(f"5. 选题 B 的起点：公式数据的划分与 SmolVLM-256M 的零样本基线（{'test' if test else 'val'}，{n_eval} 条）")
    t0 = time.time()
    split, st, f = formula_split(n_pool)
    if split is None:
        print("  本机没有缓存 OleehyO/latex-formulas（见第 33 章第 4 部分），跳过")
        return
    print(f"  去重后取 {st['n_pool']:,d} 条；Jaccard ≥ 0.8 的近似重复 {st['near_pairs']:,d} 对，连成 {st['groups']:,d} 组；"
          f"训练 {len(split['train']):,d} / 验证 {len(split['val']):,d} / 测试 {len(split['test']):,d}；"
          f"跨两边的近似重复 {st['cross']} 对（{time.time() - t0:.0f} 秒）")
    with open(os.path.join(OUT, "formula_split.json"), "w") as fh:
        json.dump(split, fh)
    print(f"  划分写进 {OUT}/formula_split.json（行号；训练时用 read_rows 读图）")
    which = "test" if test else "val"
    man = manifest({"model": "SmolVLM-256M", "question": FORMULA_Q, "max_new": max_new, "longest": 512,
                    "split": which, "n_pool": n_pool}, [f], 0)
    if test:
        TestGuard(os.path.join(OUT, "test_log.json")).check("smol-zeroshot__s0", reason=arg("--reason"))
    p, m = load_smol()
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": FORMULA_Q}]}]
    pr = p.apply_chat_template(msgs, add_generation_prompt=True)
    recs, t0 = [], time.time()
    with torch.no_grad():
        for img, ref in read_rows(f, split[which][:n_eval]):
            inp = p(text=[pr], images=[[img]], return_tensors="pt", size={"longest_edge": 512})
            g = m.generate(**inp, max_new_tokens=max_new, do_sample=False)
            pred = p.tokenizer.decode(g[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            sc, ex = formula_score(pred, ref)
            recs.append({"task": "formula", "bucket": formula_bucket(ref), "ref": ref, "pred": pred, "score": sc, "exact": ex})
    print(f"  {len(recs)} 条用了 {time.time() - t0:.0f} 秒；完全相同（归一化后）{np.mean([r['exact'] for r in recs]):.1%}")
    for r in recs[:3]:
        print(f"    真值 {r['ref'][:60]!r}\n    预测 {r['pred'][:60]!r}   1 − NED {r['score']:.2f}")
    write_run("smol-zeroshot", 0, which, recs, man, {"exact": float(np.mean([r["exact"] for r in recs]))})


PARTS = {1: part1_demo, 2: part2_eval, 3: part3_report, 4: part4_project_a, 5: part5_project_b}

if __name__ == "__main__":
    want = [int(a) for a in sys.argv[1:] if a.isdigit() and sys.argv[sys.argv.index(a) - 1] not in ("--seed", "--n")] or [1]
    for k in want:
        PARTS[k]()
