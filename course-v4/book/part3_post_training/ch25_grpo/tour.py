"""第 25 章：组内优势、GRPO 损失、分层奖励与奖励投机；TRL GRPO 与有缺陷奖励对照；与继续 SFT 比较。

部分 1：CPU 机制检查（优势、损失、奖励探针）。
部分 2：GPU。TRL GRPOTrainer，起点为第 23 章合并的 SFT 模型；--reward layered|parse_only。
部分 3：GPU。继续 SFT 对照：同样的提示与步数，监督换成标准答案（TRL SFT + LoRA）。
部分 4：GPU。默认在 val 上生成；选型冻结后，预先登记的候选可各用 --final-test 评一次 test，不据此改设置。
本章只登记了继续 SFT；开发期不看 test 错例，最终报告定稿后才可解封（第 0 章 §0.10.1）。
部分 5：CPU。汇总各运行的 val 指标、奖励投机统计，与第 22 章 SFT 预测做配对比较。
"""
import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import time

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch

from miniocr.data.postcorrect import build_split
from miniocr.eval.analysis import bootstrap_mean, paired_bootstrap_diff
from miniocr.eval.postcorrect_eval import score_postcorrect
from miniocr.eval.rewards import hack_probes, layered_reward, parse_only_reward
from miniocr.prompting import render_chat, zero_shot
from miniocr.train.grpo import group_advantages, grpo_loss

REWARDS = {'layered': layered_reward, 'parse_only': parse_only_reward}


def emit(**record):
    print(json.dumps(dict(source='本项目观察', utc=datetime.now(timezone.utc).isoformat(), **record),
                     ensure_ascii=False, default=str), flush=True)


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + '\n')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def part1(args):
    """优势与损失的手算检查；奖励探针；裁剪方向检查。"""
    import math
    adv = group_advantages(torch.tensor([1., 0., 1., 0.]), 2, scale=False)
    assert torch.allclose(adv, torch.tensor([.5, -.5, .5, -.5]))
    zero = group_advantages(torch.tensor([1., 1., 1., 1.]), 4)
    assert torch.allclose(zero, torch.zeros(4))
    # 正优势在 ratio 超过 1+eps 时饱和：梯度应为 0。
    new = torch.tensor([[math.log(1.5)]], requires_grad=True)
    loss, diag = grpo_loss(new, torch.zeros(1, 1), new.detach(), torch.tensor([1.]), torch.ones(1, 1),
                           epsilon=.2, beta=0.)
    loss.backward()
    assert abs(new.grad.item()) < 1e-6 and abs(loss.item() + 1.2) < 1e-6
    # 负优势、ratio 降到 1-eps 以下时同样饱和。
    new2 = torch.tensor([[math.log(.5)]], requires_grad=True)
    loss2, _ = grpo_loss(new2, torch.zeros(1, 1), new2.detach(), torch.tensor([-1.]), torch.ones(1, 1),
                         epsilon=.2, beta=0.)
    loss2.backward()
    assert abs(new2.grad.item()) < 1e-6
    # 参考与当前相同时 KL 为 0。
    _, diag_same = grpo_loss(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2),
                             torch.tensor([0.]), torch.ones(1, 2), beta=.04)
    assert diag_same['kl_mean'] == 0
    src, tgt = r'\frac{a}{b}+c', r'\frac{a}{b}+d'
    probes = [dict(name=n, text=t, layered=layered_reward(t, src, tgt), parse_only=parse_only_reward(t, src, tgt))
              for n, t in hack_probes(src, tgt)]
    # 只看能否解析的奖励给 'single_symbol' 与 'copy_source' 满分，分层奖励不会。
    by_name = {p['name']: p for p in probes}
    assert by_name['empty']['layered'] == 0 and by_name['single_symbol']['parse_only'] == 1
    assert by_name['copy_source']['layered'] < 0.5 and by_name['gold_answer']['layered'] == 1
    return dict(advantages=dict(scaled_example=group_advantages(torch.tensor([1., 0.]), 2).tolist()),
                clip_saturation_grad=abs(new.grad.item()), probes=probes)


def make_policy_trainer(args, rows, out, reward_name):
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM
    from trl import GRPOConfig, GRPOTrainer
    base = AutoModelForCausalLM.from_pretrained(str(args.base), dtype=torch.float32, local_files_only=True)
    reward = REWARDS[reward_name]

    def reward_fn(completions, source, target, **kwargs):
        return [reward(c, s, t) for c, s, t in zip(completions, source, target)]
    config = GRPOConfig(output_dir=str(out), max_steps=args.steps,
                        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.accumulation,
                        num_generations=args.group, max_completion_length=args.max_completion,
                        temperature=1.0, learning_rate=args.lr, lr_scheduler_type='constant', warmup_steps=0,
                        beta=args.kl_beta, epsilon=.2, loss_type='grpo', scale_rewards='group',
                        bf16=args.dtype == 'bf16', use_cpu=args.device == 'cpu',
                        gradient_checkpointing=args.checkpointing,
                        gradient_checkpointing_kwargs={'use_reentrant': False},
                        logging_steps=1, save_strategy='no', report_to='none', disable_tqdm=True,
                        dataloader_pin_memory=False, seed=args.seed, data_seed=args.seed)
    peft_config = LoraConfig(r=args.rank, lora_alpha=2*args.rank, lora_dropout=0., bias='none',
                             target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM')
    return GRPOTrainer(model=base, reward_funcs=[reward_fn], args=config,
                       train_dataset=Dataset.from_list(rows), processing_class=args.tok,
                       peft_config=peft_config)


def prompt_rows(args):
    # 每个优化步使用新提示：总数 = steps × prompts，避免同一组提示被反复训练。
    total = args.steps * args.prompts
    rows = build_split('train', args.train_offset + total, seed=args.data_seed)[args.train_offset:]
    return [dict(prompt=render_chat(args.tok, zero_shot(e.source)), source=e.source, target=e.target) for e in rows]


def part2(args):
    """GRPO：每步 prompts×group 个采样，组内标准化优势；记录奖励、有效组比例、KL 与输出长度。"""
    from transformers import AutoTokenizer
    args.tok = AutoTokenizer.from_pretrained(str(args.base), local_files_only=True)
    run = args.out / f'grpo-{args.reward}'
    if (run / 'adapter').exists():
        raise ValueError(f'{run} 已有 adapter；请使用新的 --out')
    rows = prompt_rows(args)
    trainer = make_policy_trainer(args, rows, run / 'trainer', args.reward)
    write(run / 'train-config.json', trainer.args.to_dict())
    started = time.perf_counter()
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    output = trainer.train()
    seconds = time.perf_counter() - started
    history = trainer.state.log_history
    write(run / 'trainer-history.json', history)
    trainer.model.save_pretrained(run / 'adapter')
    result = dict(reward=args.reward, loss_type=trainer.args.loss_type,
                  steps=args.steps, prompts=len(rows), group=args.group,
                  max_completion_length=args.max_completion, kl_beta=args.kl_beta, lr=args.lr,
                  metrics=output.metrics, seconds=seconds,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
                  final_log=history[-1] if history else None)
    write(run / 'train-summary.json', result)
    emit(grpo_train=dict(reward=args.reward, seconds=seconds, metrics=output.metrics))
    del trainer; release()
    return result


def part3(args):
    """继续 SFT 对照：同样的提示、同样的步数，监督换成标准答案（不使用奖励）。"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from miniocr.data.sft import encode_sft
    from miniocr.train.trl_sft import make_trainer
    tok = AutoTokenizer.from_pretrained(str(args.base), local_files_only=True)
    run = args.out / 'sft-continue'
    if (run / 'adapter').exists():
        raise ValueError(f'{run} 已有 adapter；请使用新的 --out')
    train = build_split('train', args.train_offset + args.steps * args.prompts, seed=args.data_seed)[args.train_offset:]
    encoded = [x for e in train if (x := encode_sft(tok, e.source, e.target, max_length=args.max_length, id=e.id))]
    model = AutoModelForCausalLM.from_pretrained(str(args.base), dtype=torch.float32, local_files_only=True)
    trainer = make_trainer(model, tok, encoded, run / 'trainer', device=args.device,
                           dtype=args.dtype, steps=args.steps, batch_size=args.batch_size,
                           accumulation=args.accumulation, lr=args.lr, seed=args.seed, rank=args.rank,
                           checkpointing=args.checkpointing)
    started = time.perf_counter()
    output = trainer.train()
    write(run / 'trainer-history.json', trainer.state.log_history)
    trainer.model.save_pretrained(run / 'adapter')
    result = dict(steps=args.steps, examples=len(encoded), metrics=output.metrics,
                  seconds=time.perf_counter() - started)
    write(run / 'train-summary.json', result)
    emit(sft_continue=result)
    del trainer, model; release()
    return result


def part4(args):
    """默认 val；显式 final_test 评测指定的预先登记候选，每个运行只评一次并保留独立结果。"""
    split = 'test' if args.final_test else 'val'
    run = args.out / args.run
    if args.final_test and any((run / name).exists() for name in
                               ('test-metrics.json', 'preds-test.jsonl', 'test-started.json')):
        raise ValueError('该运行已启动过最终 test；不得重复评测')
    if args.final_test:
        write(run / 'test-started.json', dict(run=args.run, n=args.n, seed=args.seed,
              utc=datetime.now(timezone.utc).isoformat(), max_new_tokens=args.max_new_tokens))
    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from miniocr.prompting import generate_greedy
    from miniocr.train.sft import autocast_context
    tok = AutoTokenizer.from_pretrained(str(args.base), local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(str(args.base), dtype=torch.float32, local_files_only=True).to(args.device).eval()
    model = PeftModel.from_pretrained(base, run / 'adapter', is_trainable=False)
    val = build_split(split, args.n, seed=args.data_seed)
    prompts = [render_chat(tok, zero_shot(e.source)) for e in val]
    with autocast_context(args.device, args.dtype):
        gen = generate_greedy(model, tok, prompts, batch_size=args.gen_batch, max_new_tokens=args.max_new_tokens)
    with (run / f'preds-{split}.jsonl').open('w') as f:
        for e, text in zip(val, gen.raw_texts):
            f.write(json.dumps(dict(id=e.id, raw=text), ensure_ascii=False) + '\n')
    s = score_postcorrect(val, list(gen.raw_texts), bootstrap_samples=args.bootstrap, bootstrap_seed=args.seed)
    result = dict(run=args.run, split=split, n=len(val), seconds=gen.seconds, capped=sum(gen.length_capped),
                  generated_tokens=sum(gen.generated_tokens),
                  summary={k: v for k, v in s['summary'].items() if k != 'rows'})
    write(run / f'{split}-metrics.json', result)
    emit(**{f'{split}_eval': dict(run=args.run, em_norm=s['summary']['em_norm'], cer=s['summary']['cer'])})
    del model, base; release()
    return result


def part5(args):
    """CPU 汇总：奖励投机统计（能解析但不对、过短、复制输入）与配对比较。"""
    val = build_split('val', args.n, seed=args.data_seed)
    ids = [e.id for e in val]
    sft = {d['id']: d['raw'] for d in read_jsonl(args.sft_preds)}
    sft_scored = score_postcorrect(val, [sft[i] for i in ids], bootstrap_samples=args.bootstrap, bootstrap_seed=args.seed)
    a = [int(r['em_norm']) for r in sft_scored['rows']]
    runs = {}
    for name in ('grpo-layered', 'grpo-parse_only', 'sft-continue'):
        path = args.out / name / 'val-metrics.json'
        if not path.exists():
            continue
        preds = {d['id']: d['raw'] for d in read_jsonl(args.out / name / 'preds-val.jsonl')}
        scored = score_postcorrect(val, [preds[i] for i in ids], bootstrap_samples=args.bootstrap,
                                   bootstrap_seed=args.seed)
        b = [int(r['em_norm']) for r in scored['rows']]
        hacks = dict(copy_source=sum(1 for r, e in zip(scored['rows'], val) if r['pred'].strip() == e.source.strip()),
                     short_or_empty=sum(1 for r in scored['rows'] if len(r['pred'].strip()) < 4),
                     parses_not_exact=sum(1 for r in scored['rows'] if r['parses'] and not r['em_norm']))
        runs[name] = dict(em_norm=bootstrap_mean(b, samples=args.bootstrap, seed=args.seed),
                          cer=scored['summary']['cer'], hacks=hacks,
                          paired_vs_sft=paired_bootstrap_diff(a, b, samples=args.bootstrap, seed=args.seed))
    result = dict(n=len(val), sft_em_norm=bootstrap_mean(a, samples=args.bootstrap, seed=args.seed),
                  runs=runs, sft_hacks=dict(copy_source=sum(1 for r, e in zip(sft_scored['rows'], val) if r['pred'].strip() == e.source.strip())))
    write(args.out / 'comparison.json', result)
    emit(comparison=result)
    return result


PARTS = {1: ('机制检查：优势、GRPO 损失与奖励探针', part1), 2: ('GRPO 训练', part2),
         3: ('继续 SFT 对照训练', part3), 4: ('val 生成', part4), 5: ('汇总与奖励投机统计', part5)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('parts', nargs='*', type=int)
    p.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    p.add_argument('--dtype', choices=('fp32', 'bf16'), default='fp32')
    p.add_argument('--base', type=Path, help='第 23 章合并的 SFT 模型目录（sft-merged）')
    p.add_argument('--reward', choices=tuple(REWARDS), default='layered')
    p.add_argument('--run', default='grpo-layered', help='部分 4 的运行名')
    p.add_argument('--final-test', action='store_true', help='冻结设置后，显式部分 4 单次评 test')
    p.add_argument('--n', type=int, default=20)
    p.add_argument('--prompts', type=int, default=8, help='每步提示数（GRPO 每步 prompts×group 个采样）')
    p.add_argument('--train-offset', type=int, default=4096)
    p.add_argument('--group', type=int, default=8)
    p.add_argument('--steps', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--accumulation', type=int, default=1)
    p.add_argument('--lr', type=float, default=5e-6)
    p.add_argument('--kl-beta', type=float, default=.04)
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--max-length', type=int, default=384)
    p.add_argument('--max-completion', type=int, default=160)
    p.add_argument('--max-new-tokens', type=int, default=192)
    p.add_argument('--gen-batch', type=int, default=16)
    p.add_argument('--checkpointing', action='store_true')
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--data-seed', type=int, default=None, help='数据噪声种子；默认沿用 --seed，多训练种子比较时固定为 0')
    p.add_argument('--sft-preds', type=Path, help='第 22 章 val 的 SFT 预测 JSONL')
    p.add_argument('--out', type=Path, default=Path('outputs/book/ch25/local'))
    args = p.parse_args(argv)
    if args.data_seed is None:
        args.data_seed = args.seed
    if args.final_test and args.parts != [4]:
        p.error('--final-test 必须仅运行部分 4')
    if any(i not in PARTS for i in args.parts):
        p.error('未知部分号')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA 不可用')
    if args.device == 'cpu' and torch.cuda.is_available():
        os.environ['ACCELERATE_USE_CPU'] = 'true'
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    versions = {k: importlib.metadata.version(k) for k in ('torch', 'transformers', 'trl', 'peft')}
    invocation = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    write(args.out / f'environment-{invocation}.json', dict(versions=versions,
          args={k: str(v) for k, v in vars(args).items()},
          device=torch.cuda.get_device_name() if args.device == 'cuda' else args.device,
          precision='合并 SFT 基座 FP32 参数 + bf16 autocast；LoRA 为 FP32'))
    for i in args.parts or [1]:
        start = time.perf_counter()
        emit(part=i, title=PARTS[i][0], event='start')
        result = PARTS[i][1](args)
        write(args.out / (f'part{i}-test.json' if args.final_test else f'part{i}.json'), result)
        emit(part=i, event='done', seconds=time.perf_counter() - start)


if __name__ == '__main__':
    main()
