"""第 34 章 VLM 可验证奖励（GRPO）。默认 1 为无权重机制实验；3–6 显式选择。

以第 33 章的 SFT/LoRA 适配器为起点：每张训练图采样 G 个候选，用 mathtext 可解析性、规范化字符串
和编辑距离给分层奖励，组内标准化得到优势，再用 token 级策略梯度 + KL(参考=SFT 起点) 更新。
对照是从同一起点、用同样的提示顺序继续做交叉熵 SFT。
"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch

from miniocr.data.vlm_sft import PROMPT, FormulaCollator
from miniocr.models.vlm_anatomy import local_snapshot, load_image_only_processor, MODELS
from miniocr.train.vlm_grpo import (PROBE_OUTPUTS, completion_mask, degenerate, group_advantages,
                                    grpo_loss, layered_reward, probe_rewards, token_logprobs)
from miniocr.train.vlm_lora import move_batch
from book.part4_vlm.ch33_vlm_lora.tour import emit, evaluate, reset, rows_for, save, peak, sync


def load_model(args, adapter, trainable):
    """加载基座与适配器；trainable=True 时 LoRA 参数转 fp32 以便稳定更新。"""
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText
    path = local_snapshot(args.model, args.model_cache)
    if path is None: raise FileNotFoundError(f'缺少本地 {MODELS[args.model]} 权重；不自动下载')
    processor = load_image_only_processor(path, args.model)
    processor.image_processor.size = {'shortest_edge': args.min_pixels, 'longest_edge': args.max_pixels}
    processor.tokenizer.padding_side = 'right'
    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float32
    base = AutoModelForImageTextToText.from_pretrained(path, local_files_only=True, dtype=dtype,
                attn_implementation='sdpa').to(args.device)
    model = PeftModel.from_pretrained(base, adapter, is_trainable=trainable)
    if trainable:
        for p in model.parameters():
            if p.requires_grad: p.data = p.data.float()
    emit(model=args.model, snapshot=str(path), adapter=str(adapter), trainable=trainable)
    return model, processor, dtype


def eos_ids(processor, model):
    ids = set()
    eos = model.generation_config.eos_token_id
    ids.update(eos if isinstance(eos, list) else [eos])
    if processor.tokenizer.eos_token_id is not None: ids.add(processor.tokenizer.eos_token_id)
    return sorted(i for i in ids if i is not None)


def image_and_prompt(args, processor, row):
    with Image.open(args.data/row['image']) as opened: im = opened.convert('RGB')
    messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': PROMPT}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return im, prompt


def sample_group(args, model, processor, dtype, row, group):
    """同一图片的 G 个候选：纯采样（top_k=0, top_p=1），使 log-prob 与采样分布一致。"""
    im, prompt = image_and_prompt(args, processor, row)
    batch = move_batch(processor(text=[prompt]*group, images=[im]*group, return_tensors='pt', padding=True),
                       args.device, dtype)
    model.eval()
    with torch.no_grad():  # 不用 inference_mode：其张量不能进入后续的 autograd 计算
        out = model.generate(**batch, max_new_tokens=args.train_max_new_tokens, do_sample=True,
                             temperature=1.0, top_k=0, top_p=1.0, use_cache=True,
                             pad_token_id=processor.tokenizer.pad_token_id)
    L = batch['input_ids'].shape[1]
    completion = out[:, L:]
    mask = completion_mask(completion, eos_ids(processor, model)).to(completion.device)
    return batch, completion, mask


def sequence_logprobs(model, batch, completion):
    """提示+候选一次前向；只保留 completion 位置的 logits（logits_to_keep 省显存）。"""
    T = completion.shape[1]
    ids = torch.cat([batch['input_ids'], completion], 1)
    att = torch.cat([batch['attention_mask'], torch.ones_like(completion)], 1)
    extras = {k: v for k, v in batch.items() if k not in ('input_ids', 'attention_mask')}
    if 'mm_token_type_ids' in extras:  # 多模态位置编码按完整序列对齐；候选文本位置类型为 0
        extras['mm_token_type_ids'] = torch.cat([extras['mm_token_type_ids'], torch.zeros_like(completion)], 1)
    out = model(input_ids=ids, attention_mask=att, use_cache=False, logits_to_keep=T+1, **extras)
    return token_logprobs(out.logits[:, :T], completion)


def make_scheduler(opt, steps):
    """线性预热再线性衰减；第一步学习率为 1/warmup 而不是 0，避免第一次更新被浪费。"""
    warm = min(10, max(1, steps // 10))
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s+1)/warm) * max(0.0, 1 - s/steps))


def row_order(args, rows):
    order = np.random.default_rng(args.seed).permutation(len(rows))
    return [rows[i] for i in order]


def train_grpo(args):
    rows = row_order(args, rows_for(args, 'train', args.train_n))
    if args.steps*args.prompts > len(rows): raise ValueError('训练图片不足；请增大 --train-n')
    policy, processor, dtype = load_model(args, args.adapter, trainable=True)
    ref, _, _ = load_model(args, args.adapter, trainable=False)
    ref.eval()
    for p in ref.parameters(): p.requires_grad_(False)
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    policy.enable_input_require_grads(); policy.train(); policy.config.use_cache = False
    params = [p for p in policy.parameters() if p.requires_grad]
    if not params: raise RuntimeError('没有可训练参数')
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = make_scheduler(opt, args.steps)
    out = args.out; out.mkdir(parents=True, exist_ok=True)
    log_path = out/'train_log.jsonl'; log_path.write_text('')
    reset(args); start = time.perf_counter(); probe_before = probe_summary(args, rows)
    for step in range(args.steps):
        began = time.perf_counter()
        batch_rows = [rows[(step*args.prompts + j) % len(rows)] for j in range(args.prompts)]
        groups = []
        for row in batch_rows:
            batch, completion, mask = sample_group(args, policy, processor, dtype, row, args.group)
            texts = processor.batch_decode(completion, skip_special_tokens=True)
            scored = [layered_reward(row['target'], t, reward_mode=args.reward_mode) for t in texts]
            groups.append(dict(row=row, batch=batch, completion=completion, mask=mask, scored=scored, texts=texts))
        rewards = torch.tensor([[s['reward'] for s in g['scored']] for g in groups], dtype=torch.float32)
        advantages, informative = group_advantages(rewards)
        total = sum(int(g['mask'][k].sum()) for gi, g in enumerate(groups) if bool(informative[gi])
                    for k in range(args.group))
        stats = dict(advantage_abs_mean=float(advantages.abs().mean()), step=step+1, reward_mean=float(rewards.mean()), informative=float(informative.float().mean()),
                     tier_fraction={t: sum(s['tier']==t for g in groups for s in g['scored'])/(args.prompts*args.group)
                                    for t in range(4)},
                     degenerate=sum(degenerate(t) for g in groups for t in g['texts'])/(args.prompts*args.group),
                     mean_tokens=float(np.mean([int(m.sum()) for g in groups for m in g['mask']])))
        opt.zero_grad(set_to_none=True)
        kl_values = []
        if total > 0:
            policy.train()
            for gi, g in enumerate(groups):
                if not bool(informative[gi]): continue
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.device == 'cuda' and args.dtype == 'bf16'):
                    pol = sequence_logprobs(policy, g['batch'], g['completion'])
                    with torch.no_grad():
                        ref_lp = sequence_logprobs(ref, g['batch'], g['completion'])
                    loss, s = grpo_loss(pol.float(), ref_lp.float(), g['mask'], advantages[gi].to(pol.device),
                                        beta=args.beta, normalizer=total)
                loss.backward(); kl_values.append(s['kl'])
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        sched.step()
        if step+1 in args.checkpoint_steps:
            policy.save_pretrained(out/f'adapter_step_{step+1}')
        stats.update(kl=float(np.mean(kl_values)) if kl_values else None, updated=total > 0,
                     seconds=time.perf_counter()-began, lr=sched.get_last_lr()[0])
        with log_path.open('a') as f: f.write(json.dumps(stats, ensure_ascii=False)+'\n')
        if step < 3 or (step+1) % 10 == 0:
            emit(step=step+1, reward_mean=stats['reward_mean'], informative=stats['informative'],
                 kl=stats['kl'], seconds=stats['seconds'])
    sync(args)
    probe_after = probe_summary(args, rows)
    policy.save_pretrained(out/'adapter')
    result = dict(config=vars(args), seconds=time.perf_counter()-start, probe_before=probe_before,
                  probe_after=probe_after, **peak(args))
    save(args, 'training.json', result)
    emit(来源='本项目观察', seconds=result['seconds'], peak_allocated_gib=result['peak_allocated_gib'])


def train_sft_control(args):
    """同一起点、同样的提示顺序，只做交叉熵继续训练；损失与 GRPO 使用同一学习率。"""
    rows = row_order(args, rows_for(args, 'train', args.train_n))
    if args.steps*args.prompts > len(rows): raise ValueError('训练图片不足；请增大 --train-n')
    model, processor, dtype = load_model(args, args.adapter, trainable=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads(); model.train(); model.config.use_cache = False
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = make_scheduler(opt, args.steps)
    collator = FormulaCollator(processor, args.data, args.max_tokens)
    out = args.out; out.mkdir(parents=True, exist_ok=True); log_path = out/'train_log.jsonl'; log_path.write_text('')
    reset(args); start = time.perf_counter()
    for step in range(args.steps):
        began = time.perf_counter()
        batch_rows = [rows[(step*args.prompts + j) % len(rows)] for j in range(args.prompts)]
        batch = move_batch(collator(batch_rows), args.device, dtype)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.device == 'cuda' and args.dtype == 'bf16'):
            loss = model(**batch).loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        rec = dict(step=step+1, loss=float(loss), seconds=time.perf_counter()-began, lr=sched.get_last_lr()[0])
        with log_path.open('a') as f: f.write(json.dumps(rec)+'\n')
        if step < 3 or (step+1) % 10 == 0: emit(step=step+1, loss=rec['loss'], seconds=rec['seconds'])
    sync(args); model.save_pretrained(out/'adapter')
    save(args, 'training.json', dict(config=vars(args), seconds=time.perf_counter()-start, **peak(args)))


def probe_summary(args, rows):
    """训练前后对固定退化输出的打分，以及取前 50 个 train 答案的 tier≥1 比例。"""
    targets = [r['target'] for r in rows[:50]]
    table = [probe_rewards(t) for t in targets]
    rewards = [{o: layered_reward(t, o, reward_mode=args.reward_mode)['reward'] for o in PROBE_OUTPUTS} for t in targets]
    return dict(n=len(targets), outputs=list(PROBE_OUTPUTS), reward_mode=args.reward_mode,
                mean_reward={o: float(np.mean([r[o] for r in rewards])) for o in PROBE_OUTPUTS},
                tier_at_least_1={o: sum(t[o] >= 1 for t in table)/len(table) for o in PROBE_OUTPUTS})


def part1(args):
    """奖励与投机探针：无权重，固定样例；展示分层与退化输出能否拿到奖励。"""
    # (标准答案, 接近但有错的输出, 只有解释文字的输出)：手工给定，便于读者核对分层边界。
    examples = [('\\frac{a}{b}', '\\frac{a}{c}', 'The fraction is shown above.'),
                ('x^{2}+y^{2}=r^{2}', 'x^{2}+y^{2}=r^{3}', 'This is the circle equation.'),
                ('\\sum_{i=1}^{n} i', '\\sum_{i=1}^{m} i', 'Explanation: a sum.'),
                ('\\int_0^1 f(x)\\,dx', '\\int_0^1 f(x)\\,dy', 'Because of the integral.')]
    table = []
    for target, close, prose in examples:
        for name, text in (('exact', target), ('close', close), ('prose', prose)):
            row = layered_reward(target, '$' + text + '$' if name != 'prose' else text)
            table.append(dict(target=target, case=name, tier=row['tier'], reward=row['reward']))
    probes = {t: probe_rewards(t) for t, _, _ in examples}
    save(args, 'reward_table.json', dict(table=table, probes=probes))
    for item in table: emit(**item)
    emit(probes=probes, note='退化输出若达到 tier≥1，就是可被钻的奖励漏洞')


def part2(args):
    """显存估算：只读 config，不载权重。真实峰值在训练 JSON 中以实测为准。"""
    from transformers import AutoConfig
    path = local_snapshot(args.model, args.model_cache)
    if path is None: raise FileNotFoundError(f'缺少 {MODELS[args.model]} 的 config；不下载')
    cfg = AutoConfig.from_pretrained(path, local_files_only=True)
    text = getattr(cfg, 'text_config', cfg)
    V = text.vocab_size; H = text.hidden_size; Lyr = text.num_hidden_layers
    T = args.train_max_new_tokens; P = args.prompts; G = args.group
    logits_bf16 = G*T*V*2; logp_fp32 = G*T*V*4
    per_seq_act = G*(T+400)*H*2*Lyr*2  # 粗略：每层激活 bf16，两倍余量；梯度检查点后会远小于此
    est = dict(vocab=V, hidden=H, layers=Lyr, prompts_per_step=P, group=G, max_new_tokens=T,
               logits_bf16_gib_per_group=logits_bf16/2**30, logprob_fp32_gib_per_group=logp_fp32/2**30,
               rough_activation_upper_gib_per_group=per_seq_act/2**30,
               note='上界式估算；梯度检查点、参考模型副本与优化器状态另计；以训练 JSON 的 peak 为准')
    save(args, 'memory_estimate.json', est)
    for k, v in est.items(): emit(**{k: v})


def part3(args):
    if args.arm != 'grpo' or not args.adapter: raise ValueError('部分 3 为 GRPO 训练，需要 --arm grpo --adapter')
    train_grpo(args)


def part4(args):
    if args.arm != 'sft' or not args.adapter: raise ValueError('部分 4 为同起点 SFT 对照，需要 --arm sft --adapter')
    train_sft_control(args)


def evaluate_val_batched(args, model, processor, dtype, rows):
    """左 padding 批量贪心 val；EOS 后 padding 不计入输出长度。"""
    import hashlib
    from miniocr.data.postcorrect import PostCorrectExample
    from miniocr.eval.postcorrect_eval import score_postcorrect, extract_formula
    processor.tokenizer.padding_side = 'left'
    model.eval(); start = time.perf_counter(); observations = []; predictions = []
    for offset in range(0, len(rows), args.eval_batch_size):
        chunk = rows[offset:offset+args.eval_batch_size]
        images, prompts = zip(*(image_and_prompt(args, processor, row) for row in chunk))
        batch = move_batch(processor(text=list(prompts), images=list(images), padding=True, return_tensors='pt'),
                           args.device, dtype)
        with torch.inference_mode():
            outputs = model.generate(**batch, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
        completions = outputs[:, batch['input_ids'].shape[1]:]
        masks = completion_mask(completions, eos_ids(processor, model))
        for row, tokens, mask in zip(chunk, completions, masks):
            tokens = tokens[:int(mask.sum())]
            raw = processor.decode(tokens, skip_special_tokens=True)
            pred = extract_formula(raw); predictions.append(pred)
            observations.append(dict(id=row['id'], generated_tokens=len(tokens),
                truncated=len(tokens) == args.max_new_tokens and int(tokens[-1]) not in eos_ids(processor, model),
                target=row['target'], raw=raw, pred=pred))
        emit(split='val', completed=len(observations), n=len(rows), seconds=time.perf_counter()-start)
    examples = [PostCorrectExample(r['id'], 'val', r['target'], r['target']) for r in rows]
    metrics = score_postcorrect(examples, predictions)['summary']; sync(args)
    return dict(model=args.model, model_id=MODELS[args.model], adapter=str(args.adapter), split='val',
        condition='correct', prompt=PROMPT, do_sample=False, enable_thinking=False,
        attn_implementation='sdpa', processor=processor.image_processor.to_dict(),
        sample_ids_sha256=hashlib.sha256(json.dumps([r['id'] for r in rows]).encode()).hexdigest(),
        metrics={k: metrics[k] for k in ('n', 'em_raw', 'em_norm', 'cer', 'parse_rate')},
        seconds=time.perf_counter()-start, truncated=sum(r['truncated'] for r in observations),
        observations=observations, config=vars(args), **peak(args))


def part5(args):
    """val 评测（只用于选择）。--adapter 为空时评测基座；与第 33 章 evaluate 口径一致。"""
    if args.split != 'val': raise ValueError('部分 5 只用于 val；test 请用部分 6')
    rows = rows_for(args, 'val', args.n); reset(args)
    model, processor, dtype = load_model(args, args.adapter, trainable=False)
    if args.eval_batch_size == 1:
        result, _ = evaluate(args, model, processor, dtype, rows, split='val')
    else:
        result = evaluate_val_batched(args, model, processor, dtype, rows)
    result['mode'] = args.name or 'adapter'
    save(args, 'val_result.json', result); emit(mode=result['mode'], split='val', metrics=result['metrics'])


def part6(args):
    """test 一次性评测：起点 SFT 与所选模型各评一次；lock 文件用排他创建。"""
    if args.split != 'test' or not args.final_test or not args.adapter or not args.compare_adapter:
        raise ValueError('test 需要 --split test --final-test --adapter <起点> --compare-adapter <所选>')
    with (args.data/'ch34_final_test.lock').open('x') as f:
        json.dump(dict(time=datetime.now(timezone.utc).isoformat(), config=vars(args)), f, default=str, indent=2)
    rows = rows_for(args, 'test', args.n); reset(args)
    model, processor, dtype = load_model(args, args.adapter, trainable=False)
    model.load_adapter(str(args.compare_adapter), adapter_name='chosen', is_trainable=False)
    for name, adapter in (('start_sft', 'default'), ('chosen', 'chosen')):
        model.set_adapter(adapter)
        result, _ = evaluate(args, model, processor, dtype, rows, split='test')
        result['mode'] = name; save(args, f'test_{name}.json', result); emit(mode=name, split='test', metrics=result['metrics'])


PARTS = {1: ('奖励分层与投机探针', part1), 2: ('显存估算', part2), 3: ('GRPO 训练', part3),
         4: ('同起点 SFT 对照', part4), 5: ('val 评测', part5), 6: ('test 一次性评测', part6)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('parts', type=int, nargs='*'); p.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    p.add_argument('--dtype', choices=('fp32', 'bf16'), default='fp32'); p.add_argument('--n', type=int, default=20)
    p.add_argument('--eval-batch-size', type=int, default=1)
    p.add_argument('--checkpoint-steps', type=int, nargs='*', default=[])
    p.add_argument('--train-n', type=int, default=2048); p.add_argument('--steps', type=int, default=5)
    p.add_argument('--prompts', type=int, default=1, help='每步的图片数（组数）')
    p.add_argument('--group', type=int, default=4, help='每张图的候选数 G')
    p.add_argument('--train-max-new-tokens', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-5); p.add_argument('--beta', type=float, default=0.04)
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--reward-mode', choices=('layered', 'no_parse_bonus'), default='layered')
    p.add_argument('--arm', choices=('grpo', 'sft'), default='grpo')
    p.add_argument('--model', choices=('qwen35', 'qwen3vl'), default='qwen35'); p.add_argument('--model-cache', type=Path)
    p.add_argument('--data', type=Path); p.add_argument('--out', type=Path, default=Path('outputs/book/ch34/local'))
    p.add_argument('--adapter', type=Path); p.add_argument('--compare-adapter', type=Path)
    p.add_argument('--name', type=str); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--split', choices=('val', 'test'), default='val'); p.add_argument('--final-test', action='store_true')
    p.add_argument('--min-pixels', type=int, default=16384); p.add_argument('--max-pixels', type=int, default=65536)
    p.add_argument('--max-new-tokens', type=int, default=256); p.add_argument('--max-tokens', type=int, default=2048)
    args = p.parse_args(argv)
    if any(x < 1 for x in (args.n, args.train_n, args.steps, args.prompts, args.group, args.train_max_new_tokens,
                           args.max_new_tokens, args.max_tokens, args.eval_batch_size)):
        p.error('规模必须为正')
    if args.group < 2: p.error('GRPO 组内标准化至少需要 2 个候选')
    if any(x not in PARTS for x in args.parts): p.error('部分号为 1–6')
    if args.device == 'cuda' and not torch.cuda.is_available(): p.error('CUDA 不可用')
    if any(x >= 2 for x in args.parts) and args.data is None: p.error('需要 --data（第 33 章落盘的数据目录）')
    torch.set_num_threads(2)
    start = time.perf_counter()
    emit(start_utc=datetime.now(timezone.utc).isoformat(), config=vars(args), torch=torch.__version__)
    args.out.mkdir(parents=True, exist_ok=True); save(args, 'config.json', vars(args))
    for number in args.parts or (1,):
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        title, fn = PARTS[number]; emit(部分=number, 标题=title); begun = time.perf_counter()
        fn(args); reset(args); emit(部分=number, seconds=time.perf_counter()-begun)
    timing = dict(end_utc=datetime.now(timezone.utc).isoformat(), seconds=time.perf_counter()-start)
    save(args, 'timing.json', timing); emit(**timing)


if __name__ == '__main__':
    main()
