"""第 23 章：Bradley-Terry 与 DPO 的机制检查、偏好对构造、TRL DPO 训练与 β 扫描。

部分 1：CPU 机制检查（手算值、梯度方向、一步自写 DPO）。
部分 2：GPU。合并第 21 章选定的 SFT adapter 得到 SFT 策略，并对 train 的新提示各采样 K 个候选。
部分 3：CPU。按与标准答案的相似度构造偏好对，统计长度与格式偏好，检查分词边界。
部分 4：GPU。用 TRL DPOTrainer 在 SFT 合并模型上训练 LoRA-DPO（reference = 关闭 adapter 的 SFT），记录 reward/margin。
部分 5：GPU。在 val 上生成 DPO 输出（同一评测协议）。
部分 6：CPU。读取所有 β 的 val 结果，按固定规则选 β，与第 21 章 SFT 及第 22 章预测做配对比较。
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

from miniocr.data.postcorrect import build_split
from miniocr.eval.analysis import bootstrap_mean, paired_bootstrap_diff
from miniocr.eval.latex_norm import normalize_latex
from miniocr.eval.metrics import sample_error_rate
from miniocr.eval.postcorrect_eval import format_issues, score_postcorrect
from miniocr.prompting import render_chat, zero_shot
from miniocr.train.preference import bradley_terry_loss, dpo_loss

BETAS = (0.05, 0.1, 0.3, 1.0)


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
    """手算可对照的数值 + 梯度符号 + 一步自写 DPO（tiny 随机 Qwen，只验证机制，不代表效果）。"""
    import math
    loss0 = bradley_terry_loss(torch.tensor([0.]), torch.tensor([0.])).item()
    assert abs(loss0 - math.log(2)) < 1e-6
    # policy 相对 ref 的对数比：chosen +2，rejected -1；beta=0.5 → r_c=1, r_r=-0.5, margin=1.5
    loss, rc, rr = dpo_loss(torch.tensor([2.]), torch.tensor([-1.]), torch.tensor([0.]), torch.tensor([0.]), beta=.5)
    expected = math.log(1 + math.exp(-1.5))
    assert abs(loss.item() - expected) < 1e-6 and abs(rc.item() - 1) < 1e-6 and abs(rr.item() + .5) < 1e-6
    hand_value = loss.item()
    # 梯度方向：增大 chosen 的 log-prob 应降低损失。
    pc = torch.tensor([0.], requires_grad=True)
    l2, _, _ = dpo_loss(pc, torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.]), beta=.1)
    l2.sum().backward()
    assert pc.grad.item() < 0

    from transformers import Qwen3Config, Qwen3ForCausalLM
    from miniocr.train.preference import dpo_step_loss, response_logprob
    torch.manual_seed(args.seed)
    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, eos_token_id=1, pad_token_id=0)
    model = Qwen3ForCausalLM(cfg)
    ref_model = Qwen3ForCausalLM(cfg)
    ref_model.load_state_dict(model.state_dict())
    ref_model.eval()
    # 一个偏好对：同一提示 3 4，行 0 为 chosen（答案 5 1），行 1 为 rejected（答案 6 7）。
    # 批内 chosen/rejected 交错排列，dpo_step_loss 按偶/奇行取出两者。
    pair = dict(input_ids=torch.tensor([[3, 4, 5, 1], [3, 4, 6, 7]]),
                labels=torch.tensor([[-100, -100, 5, 1], [-100, -100, 6, 7]]),
                attention_mask=torch.ones(2, 4, dtype=torch.long))

    def ref_fn(b):
        with torch.no_grad():
            return response_logprob(ref_model(input_ids=b['input_ids'], attention_mask=b['attention_mask'],
                                              use_cache=False).logits, b['labels'])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(3):
        loss, chosen, rejected = dpo_step_loss(model, ref_fn, pair, beta=.5)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    # 第一步 policy=ref，loss 必为 log 2。
    assert abs(losses[0] - math.log(2)) < 1e-5, losses
    assert losses[-1] < losses[0]
    return dict(hand_checks=dict(bt_zero=loss0, dpo_expected=expected, dpo_value=hand_value),
                one_step_losses=losses, note='tiny 随机模型，只验证机制；效果见部分 4–6')


def sample_candidates(model, tok, prompts, *, k, batch_size, max_new_tokens, seed):
    """对每个提示采样 k 个候选（温度 1.0、无 top-p/top-k），返回每个提示的 k 个解码文本与长度。"""
    device = next(model.parameters()).device
    eos_ids = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, (list, tuple))
                  else [model.generation_config.eos_token_id]) | {tok.eos_token_id}
    old_side = tok.padding_side
    tok.padding_side = 'left'
    out = []
    torch.manual_seed(seed)
    started = time.perf_counter()
    try:
        model.eval()
        with torch.inference_mode():
            for off in range(0, len(prompts), batch_size):
                chunk = prompts[off:off+batch_size]
                enc = tok(chunk, padding=True, return_tensors='pt', add_special_tokens=False).to(device)
                gen = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                     num_return_sequences=k, max_new_tokens=max_new_tokens,
                                     pad_token_id=tok.pad_token_id, use_cache=True)
                width = enc['input_ids'].shape[1]
                for i in range(len(chunk)):
                    texts = []
                    for j in range(k):
                        ids = gen[i*k+j, width:].tolist()
                        end = next((t+1 for t, x in enumerate(ids) if x in eos_ids), len(ids))
                        texts.append(tok.decode(ids[:end], skip_special_tokens=True))
                    out.append(texts)
    finally:
        tok.padding_side = old_side
    return out, time.perf_counter() - started


def part2(args):
    """合并第 21 章 SFT adapter 为 SFT 策略并保存；对 train 第 2048 条之后的新提示采样候选。"""
    from peft import PeftModel
    from miniocr.models.qwen3 import load_qwen3
    from miniocr.train.sft import autocast_context
    if args.adapter is None:
        raise ValueError('需要 --adapter（第 21 章选定 step）')
    model, tok = load_qwen3(device=args.device, dtype=torch.float32)
    peft_model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    merged = peft_model.merge_and_unload()
    sft_dir = args.out / 'sft-merged'
    if args.device == 'cuda':
        merged = merged.to(torch.bfloat16)
    merged.save_pretrained(sft_dir)
    tok.save_pretrained(sft_dir)
    prompts_rows = build_split('train', args.train_offset + args.prompts, seed=args.seed)[args.train_offset:]
    prompts = [render_chat(tok, zero_shot(e.source)) for e in prompts_rows]
    with autocast_context(args.device, args.dtype):
        candidates, seconds = sample_candidates(merged, tok, prompts, k=args.k, batch_size=args.sample_batch,
                                                max_new_tokens=args.max_new_tokens, seed=args.seed)
    rows = []
    for e, texts in zip(prompts_rows, candidates):
        rows.append(dict(id=e.id, source=e.source, target=e.target, candidates=texts))
    with (args.out / 'candidates.jsonl').open('w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    result = dict(prompts=len(rows), k=args.k, seconds=seconds, sft_dir=str(sft_dir),
                  dtype_saved=str(merged.dtype), sampling=dict(temperature=1.0, top_p=1.0, top_k=0))
    emit(sampling=result)
    release()
    return result


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def part3(args):
    """按与标准答案的相似度构造偏好对；只用训练提示，不接触 val/test。"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(args.out / 'sft-merged'), local_files_only=True)
    rows = read_jsonl(args.out / 'candidates.jsonl')
    pairs, stats = [], dict(prompts=len(rows), no_gap=0, tokenization_mismatch=0, ties=0)
    for r in rows:
        errs = [sample_error_rate(r['target'], c) for c in r['candidates']]
        exact = [normalize_latex(r['target']) == normalize_latex(c) for c in r['candidates']]
        order = sorted(range(len(errs)), key=lambda i: (errs[i], len(r['candidates'][i])))
        best, worst = order[0], order[-1]
        if best == worst or errs[worst] - errs[best] < args.min_gap:
            stats['no_gap'] += 1
            continue
        if errs[worst] == errs[best]:
            stats['ties'] += 1
        chosen_text = r['candidates'][best].strip() + tok.eos_token
        rejected_text = r['candidates'][worst].strip() + tok.eos_token
        prompt = render_chat(tok, zero_shot(r['source']))
        prefix = tok.encode(prompt, add_special_tokens=False)
        full_c = tok.encode(prompt + chosen_text, add_special_tokens=False)
        full_r = tok.encode(prompt + rejected_text, add_special_tokens=False)
        if full_c[:len(prefix)] != prefix or full_r[:len(prefix)] != prefix:
            stats['tokenization_mismatch'] += 1
            continue
        pairs.append(dict(id=r['id'], prompt=prompt, chosen=chosen_text, rejected=rejected_text,
                          chosen_err=errs[best], rejected_err=errs[worst], chosen_exact=exact[best],
                          rejected_exact=exact[worst], chosen_len=len(r['candidates'][best]),
                          rejected_len=len(r['candidates'][worst]),
                          rejected_format=bool(format_issues(r['candidates'][worst]))))
    with (args.out / 'pairs.jsonl').open('w') as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    n = len(pairs)
    stats.update(pairs=n,
                 chosen_exact_rate=sum(p['chosen_exact'] for p in pairs) / n if n else None,
                 rejected_exact_rate=sum(p['rejected_exact'] for p in pairs) / n if n else None,
                 chosen_longer_rate=sum(p['chosen_len'] > p['rejected_len'] for p in pairs) / n if n else None,
                 mean_chosen_len=sum(p['chosen_len'] for p in pairs) / n if n else None,
                 mean_rejected_len=sum(p['rejected_len'] for p in pairs) / n if n else None,
                 rejected_format_rate=sum(p['rejected_format'] for p in pairs) / n if n else None,
                 min_gap=args.min_gap,
                 note='偏好只依据与标准答案的字符编辑率，长度与格式偏好需要单独看统计量')
    write(args.out / 'pair-stats.json', stats)
    emit(pairs=stats)
    return stats


def make_dpo_trainer(model, tok, rows, out, args, beta):
    from datasets import Dataset
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer
    config = DPOConfig(output_dir=str(out), max_steps=args.steps, per_device_train_batch_size=args.batch_size,
                       gradient_accumulation_steps=args.accumulation, learning_rate=args.lr,
                       lr_scheduler_type='constant', warmup_steps=0, beta=beta, max_length=args.max_length,
                       bf16=args.dtype == 'bf16', fp16=False, use_cpu=args.device == 'cpu',
                       gradient_checkpointing=args.checkpointing,
                       gradient_checkpointing_kwargs={'use_reentrant': False},
                       logging_steps=5, logging_first_step=True, save_strategy='no', report_to='none',
                       disable_tqdm=True, dataloader_pin_memory=False, seed=args.seed, data_seed=args.seed)
    peft_config = LoraConfig(r=args.rank, lora_alpha=2*args.rank, lora_dropout=0., bias='none',
                             target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM')
    return DPOTrainer(model=model, args=config, train_dataset=Dataset.from_list(rows),
                      processing_class=tok, peft_config=peft_config)


def part4(args):
    """TRL DPOTrainer：reference 为关闭 LoRA 的 SFT 合并模型；记录 reward、margin 与 TRL 报告的 loss。"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    beta = args.beta
    run = args.out / f'beta-{beta}'
    if (run / 'adapter').exists():
        raise ValueError(f'{run} 已有 adapter；请使用新的 --out')
    tok = AutoTokenizer.from_pretrained(str(args.out / 'sft-merged'), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(args.out / 'sft-merged'), dtype=torch.float32,
                                                 local_files_only=True)
    rows = [dict(prompt=p['prompt'], chosen=p['chosen'], rejected=p['rejected'])
            for p in read_jsonl(args.out / 'pairs.jsonl')]
    trainer = make_dpo_trainer(model, tok, rows, run / 'trainer', args, beta)
    started = time.perf_counter()
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    output = trainer.train()
    seconds = time.perf_counter() - started
    history = trainer.state.log_history
    write(run / 'trainer-history.json', history)
    trainer.model.save_pretrained(run / 'adapter')
    result = dict(beta=beta, steps=args.steps, pairs=len(rows), lr=args.lr, metrics=output.metrics,
                  seconds=seconds,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else None)
    write(run / 'train-summary.json', result)
    emit(dpo_train=result)
    del trainer, model; release()
    return result


def part5(args):
    """在 val 上生成 DPO 输出：SFT 合并模型 + 该 β 的 LoRA adapter。只看 val。"""
    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from miniocr.prompting import generate_greedy
    from miniocr.train.sft import autocast_context
    beta = args.beta
    run = args.out / f'beta-{beta}'
    tok = AutoTokenizer.from_pretrained(str(args.out / 'sft-merged'), local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(str(args.out / 'sft-merged'), dtype=torch.float32,
                                                local_files_only=True).to(args.device).eval()
    model = PeftModel.from_pretrained(base, run / 'adapter', is_trainable=False)
    val = build_split('val', args.n, seed=args.seed)
    prompts = [render_chat(tok, zero_shot(e.source)) for e in val]
    with autocast_context(args.device, args.dtype):
        gen = generate_greedy(model, tok, prompts, batch_size=args.gen_batch, max_new_tokens=args.max_new_tokens)
    with (run / 'preds-val.jsonl').open('w') as f:
        for e, text in zip(val, gen.raw_texts):
            f.write(json.dumps(dict(id=e.id, raw=text), ensure_ascii=False) + '\n')
    s = score_postcorrect(val, list(gen.raw_texts), bootstrap_samples=args.bootstrap, bootstrap_seed=args.seed)
    result = dict(beta=beta, n=len(val), seconds=gen.seconds, capped=sum(gen.length_capped),
                  summary={k: v for k, v in s['summary'].items() if k != 'rows'})
    write(run / 'val-metrics.json', result)
    emit(dpo_val=dict(beta=beta, em_norm=s['summary']['em_norm'], cer=s['summary']['cer']))
    del model, base; release()
    return result


def part6(args):
    """CPU 汇总：选 β 只看 val EM_norm（并列取较小 β）；与 SFT（第 22 章生成的 val 预测）配对比较。"""
    val = build_split('val', args.n, seed=args.seed)
    ids = [e.id for e in val]
    sft_path = args.sft_preds
    sft_map = {d['id']: d['raw'] for d in read_jsonl(sft_path)}
    sft_raw = [sft_map[i] for i in ids]
    sft_scored = score_postcorrect(val, sft_raw, bootstrap_samples=args.bootstrap, bootstrap_seed=args.seed)
    runs = {}
    for beta in BETAS:
        path = args.out / f'beta-{beta}' / 'val-metrics.json'
        if path.exists():
            runs[beta] = json.loads(path.read_text())
    assert runs, '没有 DPO 的 val 结果，先运行部分 4–5'
    chosen_beta = min(runs, key=lambda b: (-runs[b]['summary']['em_norm']['value'], b))
    a = [int(r['em_norm']) for r in sft_scored['rows']]
    per_beta = {}
    correctness = {}
    for beta, metrics in runs.items():
        rows = read_jsonl(args.out / f'beta-{beta}' / 'preds-val.jsonl')
        dpo_map = {d['id']: d['raw'] for d in rows}
        assert len(rows) == len(ids) and set(dpo_map) == set(ids), 'DPO 的 val ID 必须唯一且完全对齐'
        b = [int(normalize_latex(dpo_map[e.id]) == normalize_latex(e.target)) for e in val]
        correctness[beta] = b
        per_beta[str(beta)] = dict(em_norm=metrics['summary']['em_norm'], cer=metrics['summary']['cer'],
                                  paired_dpo_minus_sft=paired_bootstrap_diff(
                                      a, b, samples=args.bootstrap, seed=args.seed))
    b = correctness[chosen_beta]
    result = dict(n=len(val), selected_beta=chosen_beta,
                  selection_rule='val EM_norm 降序，并列取较小 β',
                  per_beta=per_beta,
                  sft_em_norm=bootstrap_mean(a, samples=args.bootstrap, seed=args.seed),
                  dpo_em_norm=bootstrap_mean(b, samples=args.bootstrap, seed=args.seed),
                  paired_dpo_minus_sft=paired_bootstrap_diff(a, b, samples=args.bootstrap, seed=args.seed))
    write(args.out / 'selection.json', result)
    emit(selection=result)
    return result


PARTS = {1: ('机制检查：BT/DPO 数值与一步自写 DPO', part1), 2: ('SFT 合并与候选采样', part2),
         3: ('偏好对构造与统计', part3), 4: ('TRL DPO 训练（β 扫描一项）', part4),
         5: ('val 生成（β 扫描一项）', part5), 6: ('选 β 与配对比较', part6)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('parts', nargs='*', type=int)
    p.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    p.add_argument('--dtype', choices=('fp32', 'bf16'), default='fp32')
    p.add_argument('--n', type=int, default=20, help='val 规模（部分 5、6）')
    p.add_argument('--prompts', type=int, default=8, help='采样提示数（部分 2）')
    p.add_argument('--train-offset', type=int, default=2048, help='采样提示从 train 的第几条开始')
    p.add_argument('--k', type=int, default=4)
    p.add_argument('--sample-batch', type=int, default=32)
    p.add_argument('--min-gap', type=float, default=.05)
    p.add_argument('--beta', type=float, default=.1)
    p.add_argument('--steps', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--accumulation', type=int, default=1)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--max-length', type=int, default=384)
    p.add_argument('--max-new-tokens', type=int, default=192)
    p.add_argument('--gen-batch', type=int, default=16)
    p.add_argument('--checkpointing', action='store_true')
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--adapter', type=Path, help='第 21 章选定 step 的 adapter 目录')
    p.add_argument('--sft-preds', type=Path, help='第 22 章 val 的 SFT 预测 JSONL')
    p.add_argument('--out', type=Path, default=Path('outputs/book/ch23/local'))
    args = p.parse_args(argv)
    if any(i not in PARTS for i in args.parts):
        p.error('未知部分号')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA 不可用')
    if args.device == 'cpu' and torch.cuda.is_available():
        os.environ['ACCELERATE_USE_CPU'] = 'true'
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    versions = {k: importlib.metadata.version(k) for k in ('torch', 'transformers', 'trl', 'peft')}
    env = dict(versions=versions, args={k: str(v) for k, v in vars(args).items()},
               device=torch.cuda.get_device_name() if args.device == 'cuda' else args.device,
               precision='SFT 合并 bf16（GPU）后作 DPO 的策略与 reference；LoRA 为 FP32 参数 + bf16 autocast')
    write(args.out / 'environment.json', env)
    for i in args.parts or [1]:
        start = time.perf_counter()
        emit(part=i, title=PARTS[i][0], event='start')
        result = PARTS[i][1](args)
        write(args.out / f'part{i}.json', result)
        emit(part=i, event='done', seconds=time.perf_counter() - start)


if __name__ == '__main__':
    main()
