"""第 22 章：不确定性、配对比较、分桶误差分析、泄漏近似检查与反复看 test 的模拟。

部分 1：统计工具与机制检查（不需要模型，秒级）。
部分 2：对 val 重新生成三种输出：第 18 章零样本、严格 3-shot、第 21 章 SFT adapter（需 --adapter）。
部分 3：读取部分 2 的预测，做 bootstrap、配对比较、分桶、错例表、泄漏近似检查与 test 窥视模拟。
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch

from miniocr.data.latex_corpus import corpus_available
from miniocr.data.postcorrect import build_split, check_no_leak, few_shot_pool, target_splits
from miniocr.eval.analysis import (FORMULA_BUCKETS, LENGTH_BUCKETS, bootstrap_mean, error_class,
                                   error_table, formula_bucket, length_bucket, near_overlap,
                                   paired_bootstrap_diff, peeking_simulation)
from miniocr.eval.postcorrect_eval import score_postcorrect
from miniocr.prompting import few_shot_strict, generate_greedy, render_chat, zero_shot
from miniocr.train.sft import autocast_context

METHODS = ('zero_shot', 'strict_3shot', 'sft')


def emit(**record):
    print(json.dumps(dict(source='本项目观察', utc=datetime.now(timezone.utc).isoformat(), **record),
                     ensure_ascii=False, default=str), flush=True)


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + '\n')


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def part1(args):
    """机制检查：区间、配对差、分桶、错例标签与窥视模拟，全部用手算可对照的小例子。"""
    toy = bootstrap_mean([0, 0, 1, 1], samples=2000, seed=0)
    same = paired_bootstrap_diff([1, 0, 1], [1, 0, 1], samples=200, seed=0)
    assert same['diff'] == 0 and same['interval'] == (0.0, 0.0)
    assert formula_bucket(r'\frac{a}{b}^2') == '分式'
    assert formula_bucket(r'x^{2}') == '上下标'
    assert formula_bucket(r'\begin{pmatrix}a\end{pmatrix}') == '矩阵/数组'
    assert length_bucket('x' * 59) == LENGTH_BUCKETS[1]
    cases = [
        (dict(target='x^{2}', source='x^{2}', pred='x^{2}', is_clean=True), 'correct'),
        (dict(target='x^{2}', source='x_{2}', pred='Corrected: x^{2}', is_clean=False), 'format'),
        (dict(target='x^{2}', source='x_{2}', pred='x_{2}', is_clean=False), 'unchanged'),
        (dict(target='a+b', source='a+b', pred='a-b', is_clean=True), 'over-correction'),
        (dict(target='x+y', source='x+y', pred='x+y+z', is_clean=True), 'over-correction'),
        (dict(target='x+y', source='x y', pred='x+y+z', is_clean=False), 'extra'),
        (dict(target='x+y+z', source='x+yz', pred='x+y', is_clean=False), 'missing'),
        (dict(target='x+y', source='x y', pred='x-y', is_clean=False), 'substitution'),
    ]
    labels = []
    for row, expected in cases:
        got = error_class(row['target'], row['source'], row['pred'], is_clean=row['is_clean'])
        labels.append(dict(expected=expected, got=got))
        assert got == expected, (row, got)
    assert 0 <= toy['interval'][0] <= toy['value'] <= toy['interval'][1] <= 1
    sim = peeking_simulation(trials=2000, seed=0)
    # 候选真值完全相同，若仍报告 test 上的最优值，偏差应为正且随 k 增大。
    optimism = [r['optimism'] for r in sim['rows']]
    assert optimism[-1] > optimism[0] > -0.02
    return dict(toy_bootstrap=toy, paired_identical=same, error_labels=labels, peeking=sim)


def load_model(args):
    from miniocr.models.qwen3 import load_qwen3
    model, tok = load_qwen3(device=args.device, dtype=torch.float32)
    return model, tok


def part2(args):
    """对 val 生成原始输出（不做任何抽取）；每种方法一份 JSONL，记录生成用时与口径。"""
    methods = args.methods or list(METHODS)
    if 'sft' in methods and args.adapter is None:
        raise ValueError('生成 sft 需要 --adapter（第 21 章选定 step 的 adapter 目录）')
    val = build_split('val', args.n, seed=args.seed)
    shots = few_shot_pool(3, seed=args.seed)
    write(args.out / 'few-shot-ids.json', [x.id for x in shots])
    model, tok = load_model(args)
    results = {}
    for name in methods:
        if name == 'sft':
            from peft import PeftModel
            model_run = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
        else:
            model_run = model
        if name == 'strict_3shot':
            prompts = [render_chat(tok, few_shot_strict(e.source, shots)) for e in val]
        else:
            prompts = [render_chat(tok, zero_shot(e.source)) for e in val]
        with autocast_context(args.device, args.dtype):
            gen = generate_greedy(model_run, tok, prompts, batch_size=args.batch_size,
                                  max_new_tokens=args.max_new_tokens)
        path = args.out / f'preds-{name}.jsonl'
        with path.open('w') as f:
            for e, text in zip(val, gen.raw_texts):
                f.write(json.dumps(dict(id=e.id, raw=text), ensure_ascii=False) + '\n')
        results[name] = dict(n=len(val), seconds=gen.seconds, capped=sum(gen.length_capped),
                             generated_tokens=sum(gen.generated_tokens), file=str(path))
        emit(generation=name, **results[name])
        if name == 'sft':
            # 释放 PEFT 包装，避免后续方法误用 adapter。
            model_run = model_run.unload() if hasattr(model_run, 'unload') else model_run
            del model_run
        release()
    summary_path = args.out / 'generation-summary.json'
    previous = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    write(summary_path, {**previous, **results})
    return results


def read_preds(path):
    return {d['id']: d['raw'] for d in map(json.loads, Path(path).read_text().splitlines())}


def part3(args):
    """只读部分 2 的预测做分析；所有调参与选择都只在 val 上，不接触 test。"""
    val = build_split('val', args.n, seed=args.seed)
    ids = [e.id for e in val]
    preds = {}
    for name in METHODS:
        path = args.out / f'preds-{name}.jsonl'
        if path.exists():
            p = read_preds(path)
            assert set(p) == set(ids), f'{name} 的 id 与 val 不一致'
            preds[name] = [p[i] for i in ids]
    assert preds, '没有可分析的预测文件，先运行部分 2'
    per_method, scored = {}, {}
    for name, raw in preds.items():
        s = score_postcorrect(val, raw, bootstrap_samples=args.bootstrap, bootstrap_seed=args.seed)
        scored[name] = s
        em = [int(r['em_norm']) for r in s['rows']]
        buckets = {}
        for label, key in (('formula', formula_bucket), ('length', length_bucket)):
            groups = {}
            for r, e in zip(s['rows'], val):
                groups.setdefault(key(e.target), []).append(r['em_norm'])
            buckets[label] = {k: dict(count=len(v), **bootstrap_mean(v, samples=args.bootstrap, seed=args.seed))
                              for k, v in sorted(groups.items())}
        per_method[name] = dict(summary={k: v for k, v in s['summary'].items() if k != 'rows'},
                                em_norm_bootstrap=bootstrap_mean(em, samples=args.bootstrap, seed=args.seed),
                                buckets=buckets,
                                error_table=error_table([dict(target=e.target, source=e.source, pred=r,
                                                              is_clean=e.is_clean) for e, r in zip(val, raw)]))
    paired = {}
    if 'sft' in scored:
        for base in ('zero_shot', 'strict_3shot'):
            if base in scored:
                a = [int(r['em_norm']) for r in scored[base]['rows']]
                b = [int(r['em_norm']) for r in scored['sft']['rows']]
                paired[f'sft_minus_{base}'] = paired_bootstrap_diff(a, b, samples=args.bootstrap, seed=args.seed)
    # 泄漏近似：只能检查本课程语料内部的 train/val 重叠（规范键与 MinHash 近重复）。
    train_pool = list(target_splits()['train'][:args.train_ref_n])
    val_targets = [e.target for e in val]
    exact = check_no_leak(dict(train=train_pool, val=val_targets))
    near = near_overlap(train_pool, val_targets, threshold=.8, seed=args.seed)
    leak = dict(train_reference=len(train_pool), exact=exact, near_threshold=.8,
                near_hits=near['hits'], near_checked=near['checked'],
                limitation='只检查本语料 train 参考；无法检查基座模型的预训练数据')
    result = dict(n=len(val), corpus_available=corpus_available(), methods=per_method,
                  paired=paired, leak=leak,
                  # 教学模拟固定假设真值 0.7，只说明窥视机制，不代表任何模型的真实准确率。
                  peeking=peeking_simulation(true_accuracy=.7, n_val=args.n, n_test=args.n,
                                             trials=2000, seed=args.seed),
                  buckets=dict(formula=FORMULA_BUCKETS, length=LENGTH_BUCKETS))
    write(args.out / 'analysis.json', result)
    emit(analysis=dict(methods={k: v['summary']['em_norm'] for k, v in per_method.items()},
                       paired=paired, leak=leak))
    return result


PARTS = {1: ('统计工具与机制检查', part1), 2: ('对 val 重新生成原始输出', part2),
         3: ('bootstrap、配对比较、分桶与错例分析', part3)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('parts', nargs='*', type=int)
    p.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    p.add_argument('--dtype', choices=('fp32', 'bf16'), default='fp32')
    p.add_argument('--n', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--max-new-tokens', type=int, default=192)
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--train-ref-n', type=int, default=20000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--adapter', type=Path, help='第 21 章选定 step 的 adapter 目录')
    p.add_argument('--methods', nargs='*', choices=METHODS)
    p.add_argument('--out', type=Path, default=Path('outputs/book/ch22/local'))
    args = p.parse_args(argv)
    if any(i not in PARTS for i in args.parts):
        p.error('未知部分号')
    if args.n <= 0 or args.batch_size <= 0 or args.max_new_tokens <= 0 or args.bootstrap < 0:
        p.error('规模参数必须为正（bootstrap 可为 0）')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA 不可用')
    if args.device == 'cpu' and torch.cuda.is_available():
        os.environ['ACCELERATE_USE_CPU'] = 'true'
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    versions = {k: importlib.metadata.version(k) for k in ('torch', 'transformers', 'peft', 'numpy')}
    env = dict(versions=versions, args=vars(args),
               device=torch.cuda.get_device_name() if args.device == 'cuda' else args.device,
               precision='基座 FP32 + autocast ' + args.dtype + '（与第 21 章 evaluate 一致）',
               generation=dict(do_sample=False, enable_thinking=False, max_new_tokens=args.max_new_tokens,
                               batch_size=args.batch_size),
               data=dict(version='postcorrect-v1', split='latex-v1 seed=0', noise_seed=args.seed,
                         val_ids_sha256=hashlib.sha256('\n'.join(
                             e.id for e in build_split('val', args.n, seed=args.seed)).encode()).hexdigest()))
    write(args.out / 'environment.json', env)
    for i in args.parts or [1]:
        start = time.perf_counter()
        emit(part=i, title=PARTS[i][0], event='start')
        result = PARTS[i][1](args)
        write(args.out / f'part{i}.json', result)
        emit(part=i, event='done', seconds=time.perf_counter() - start)


if __name__ == '__main__':
    main()
