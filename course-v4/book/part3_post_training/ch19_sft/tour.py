"""第 19 章：SFT 监督信号。默认用微型随机模型；--real 显式启用本地 Qwen3。"""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'

import numpy as np
import torch

from miniocr.data.latex_corpus import corpus_available, _parquet_paths
from miniocr.data.postcorrect import build_split, few_shot_pool, check_no_leak, target_splits
from miniocr.data.sft import encode_sft, collate_sft, VARIANTS
from miniocr.eval.postcorrect_eval import score_postcorrect, extract_formula, copy_baseline
from miniocr.models.gpt2 import GPT2, GPT2Config
from miniocr.models.qwen3 import find_local_qwen3, load_qwen3
from miniocr.prompting import zero_shot, few_shot, few_shot_strict, render_chat, generate_greedy
from miniocr.train.sft import causal_loss, train_sft, mean_loss, autocast_context

DEFAULT_N = 20
CONTEXT = {}


def emit(**values):
    print(json.dumps(dict(来源='本项目观察', **CONTEXT, **values), ensure_ascii=False, default=str), flush=True)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + '\n')


class ByteTokenizer:
    """CPU 教具：UTF-8 byte+4，不冒充真实 Qwen tokenizer；0 padding、1 EOS。"""
    pad_token_id, eos_token_id = 0, 1

    def encode(self, text, add_special_tokens=False):
        return [b+4 for b in text.encode('utf-8')]

    def decode(self, ids, skip_special_tokens=False):
        return bytes(i-4 for i in ids if i >= 4).decode('utf-8', errors='replace')

    def convert_ids_to_tokens(self, ids):
        return [('<pad>' if i == 0 else '<eos>' if i == 1 else repr(bytes([i-4]))) for i in ids]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        text = ''.join(f"<{m['role']}>\n{m['content']}\n" for m in messages)
        if add_generation_prompt:
            text += '<assistant>\n'
        return self.encode(text) if tokenize else text


class TinyLM(torch.nn.Module):
    """复用第 17 章因果 GPT；仅右 padding，真实 token 不会关注其后的 pad。"""
    def __init__(self):
        super().__init__()
        self.net = GPT2(GPT2Config(vocab_size=260, n_positions=512, n_embd=32, n_head=2, n_layer=1))
        for p in self.parameters():
            if p.ndim > 1:
                torch.nn.init.normal_(p, std=.02)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return SimpleNamespace(logits=self.net(input_ids))


def tokenizer(args):
    if not args.real:
        return ByteTokenizer()
    path = find_local_qwen3()
    if path is None:
        raise FileNotFoundError('--real 需要本地 Qwen3-0.6B 权重；不会下载')
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True)


def part1(args):
    tok = tokenizer(args)
    source, target = r'\frac{a}{b', r'\frac{a}{b}'
    prompt = render_chat(tok, zero_shot(source))
    ex = encode_sft(tok, source, target, max_length=1024)
    emit(模板=repr(prompt), 答案=target, prompt_tokens=ex.prompt_length, total_tokens=len(ex.input_ids),
         eos=tok.eos_token_id, tokenizer='Qwen3' if args.real else '教学 byte tokenizer')
    for i, (token, label) in enumerate(zip(ex.input_ids, ex.labels)):
        emit(position=i, token_id=token, token=tok.convert_ids_to_tokens([token])[0], label=label,
             loss=label != -100, predictor_position=i-1 if label != -100 else None)
    if args.out:
        write_json(args.out/'token_trace.json', dict(prompt=prompt, target=target, **asdict(ex)))


def part2(args):
    tok = tokenizer(args)
    examples = [encode_sft(tok, x, x, max_length=1024) for x in ('x', r'\frac{a+b}{c}')]
    batch = collate_sft(examples, tok.pad_token_id, multiple=8)
    # 用相同样本逐渐降低预算；超长整条拒收，不能截掉全部答案或伪造 EOS。
    length = len(examples[1].input_ids)
    emit(张量={k: v.tolist() for k, v in batch.items()}, padding策略='右 padding；按位置 mask',
         截断实验=[dict(max_length=m, accepted=encode_sft(tok, r'\frac{a+b}{c}', r'\frac{a+b}{c}', max_length=m) is not None)
                  for m in (length, length-1, examples[1].prompt_length)],
         策略='超预算整条拒收；训练统计拒收数，评测不丢长样本')


def part3(args):
    tok = ByteTokenizer()
    examples = [encode_sft(tok, 'x_2', 'x^2'), encode_sft(tok, 'a', 'a+b')]
    model = TinyLM().to(args.device)
    batch = {k: v.to(args.device) for k, v in collate_sft(examples, tok.pad_token_id).items()}
    logits = model(**{k: v for k, v in batch.items() if k != 'labels'}).logits
    loss = causal_loss(logits, batch['labels'])
    terms = [-logits[b, t-1].float().log_softmax(-1)[label] for b, row in enumerate(batch['labels'])
             for t, label in enumerate(row.tolist()) if t and label != -100]
    reference = torch.stack(terms).mean()
    loss.backward()
    emit(实现loss=float(loss.detach()), 逐token手算=float(reference.detach()),
         计算精度='fp32（本部分的手算对拍不使用 autocast）',
         绝对差=float((loss-reference).detach().abs()), supervised_tokens=len(terms),
         可训练参数=sum(p.numel() for p in model.parameters() if p.requires_grad),
         embedding_grad_norm=float(model.net.wte.weight.grad.norm()),
         数学结论='logits[t-1] 预测 labels[t]，只 shift 一次；提示虽然没有直接 CE，仍通过上下文得到梯度')


@torch.no_grad()
def tiny_generate(model, tok, source, budget=16):
    ids = torch.tensor([tok.encode(render_chat(tok, zero_shot(source)))], device=next(model.parameters()).device)
    generated = []
    model.eval()
    for _ in range(budget):
        token = model(ids).logits[:, -1].argmax(-1, keepdim=True)
        generated.append(token.item())
        if token.item() == tok.eos_token_id:
            break
        ids = torch.cat((ids, token), dim=1)
    return dict(text=tok.decode(generated), tokens=len(generated), eos=generated[-1] == tok.eos_token_id)


def part4(args):
    tok = ByteTokenizer()
    pairs = [('x_2', 'x^2'), ('y_2', 'y^2'), ('a_2', 'a^2'), ('b_2', 'b^2')]
    correct = [encode_sft(tok, s, t) for s, t in pairs]
    results = {}
    for variant in VARIANTS:
        torch.manual_seed(args.seed)
        model = TinyLM().to(args.device)
        train = [encode_sft(tok, s, t, variant=variant) for s, t in pairs]
        history = train_sft(model, train, 0, steps=args.toy_steps, batch_size=4, lr=.01, dtype=args.dtype, seed=args.seed)
        result = dict(first_train_loss=history[0]['loss'], last_train_loss=history[-1]['loss'],
                      training_dtype=args.dtype, generation_dtype='fp32',
                      correct_target_nll=mean_loss(model, correct, 0, dtype=args.dtype),
                      generations=[tiny_generate(model, tok, s) for s, _ in pairs], history=history)
        results[variant] = result
        emit(实验='微型模型故障演练（训练集记忆，非泛化成绩）', variant=variant,
             **{k: v for k, v in result.items() if k != 'history'})
    if args.out:
        write_json(args.out/'toy_faults.json', results)


def prepare(tok, examples, max_length, variant='correct'):
    encoded = [encode_sft(tok, e.source, e.target, max_length=max_length, variant=variant, id=e.id) for e in examples]
    kept = [e for e in encoded if e is not None]
    if not kept:
        raise ValueError('全部训练/teacher-forcing 样本超长；请增大 max-length')
    return kept, dict(requested=len(examples), accepted=len(kept), rejected=len(examples)-len(kept),
                      max_length=max_length, supervised_tokens=sum(sum(t != -100 for t in e.labels[1:]) for e in kept),
                      supervised_prompt_tokens=sum(sum(t != -100 for t in e.labels[1:e.prompt_length]) for e in kept),
                      supervised_terminal_eos=sum(e.labels[-1] == tok.eos_token_id for e in kept),
                      ids_sha256=hashlib.sha256('\n'.join(e.id for e in kept).encode()).hexdigest())


def evaluate_model(args, model, tok, examples, *, name, mode='zero_shot', save_rows=True):
    pool = few_shot_pool(3, args.seed) if mode != 'zero_shot' else []
    fn = dict(zero_shot=lambda s: zero_shot(s), few_shot=lambda s: few_shot(s, pool),
              few_shot_strict=lambda s: few_shot_strict(s, pool))[mode]
    prompts = [render_chat(tok, fn(e.source)) for e in examples]
    emit(阶段='开始生成', 实验=name, 实际n=len(examples), mode=mode, eval_batch_size=args.eval_batch_size)
    with autocast_context(args.device, args.dtype):
        generated = generate_greedy(model, tok, prompts, sources=[e.source for e in examples], batch_size=args.eval_batch_size)
    raw = score_postcorrect(examples, generated.texts)
    extracted = score_postcorrect(examples, [extract_formula(s) for s in generated.texts])
    summary = dict(raw=raw['summary'], extracted=extracted['summary'], generation_seconds=generated.seconds,
                   length_capped=sum(generated.length_capped), mean_generated_tokens=float(np.mean(generated.generated_tokens)),
                   ids_sha256=hashlib.sha256('\n'.join(e.id for e in examples).encode()).hexdigest(),
                   mode=mode, n=len(examples), split=examples[0].split,
                   generation_budget='min(256,2*batch_max_source_tokens+32)', eval_batch_size=args.eval_batch_size)
    emit(实验=name, 指标=summary)
    if args.out:
        write_json(args.out/f'{name}.json', summary)
        if save_rows:
            with (args.out/f'{name}.jsonl').open('w') as f:
                for i, ex in enumerate(examples):
                    row = dict(example=ex.to_dict(), prompt=prompts[i], raw=generated.raw_texts[i],
                               extracted=extracted['rows'][i]['pred'], score=raw['rows'][i],
                               generated_tokens=generated.generated_tokens[i], length_capped=generated.length_capped[i])
                    f.write(json.dumps(row, ensure_ascii=False)+'\n')
    return summary


def require_real(args):
    if not args.real:
        emit(状态='跳过真实模型部分；显式传 --real 启用。默认 CPU 教具不读取大权重。')
        return False
    if not corpus_available():
        raise FileNotFoundError('真实实验要求本地 cleaned_formulas；不允许把内置样例当评测结果')
    return True


def part5(args):
    if not require_real(args):
        return
    if not args.out:
        raise ValueError('真实训练须 --out 保存日志与配置')
    start = time.perf_counter()
    tok = tokenizer(args)
    train_examples = build_split('train', args.train_n, args.seed)
    val_examples = build_split('val', args.n, args.seed)
    train, stats = prepare(tok, train_examples, args.max_length, args.variant)
    if args.variant == 'correct':
        assert stats['supervised_prompt_tokens'] == 0
        assert stats['supervised_terminal_eos'] == stats['accepted']
    val, val_stats = prepare(tok, val_examples, args.max_length)
    leak = check_no_leak(dict(train=train_examples, val=val_examples))
    if not leak['ok']:
        raise ValueError('规范化目标泄漏')
    model, tok = load_qwen3(device=args.device, dtype=torch.float32)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != sum(p.numel() for p in model.parameters()):
        raise ValueError('本章应为全参训练')
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    initial = mean_loss(model, val, tok.pad_token_id, batch_size=args.batch_size, dtype=args.dtype)
    metadata = dict(config=vars(args), base_snapshot=str(find_local_qwen3()), train=stats, val_teacher_forcing=val_stats,
                    leak=leak, trainable_params=trainable, total_params=sum(p.numel() for p in model.parameters()),
                    precision='fp32 weights/AdamW; '+args.dtype+' autocast; fp32 CE', initial_val_nll=initial,
                    data_files=[str(p) for p in _parquet_paths()], started_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.out/'train_config.json', metadata)
    emit(训练前检查=metadata)
    best = None
    history_file = (args.out/'history.jsonl').open('w')
    def callback(step, record):
        nonlocal best
        history_file.write(json.dumps(record)+'\n')
        history_file.flush()
        if step == 1 or step % 10 == 0:
            emit(训练=record)
        if step == args.steps or (args.eval_every and step % args.eval_every == 0):
            nll = mean_loss(model, val, tok.pad_token_id, batch_size=args.batch_size, dtype=args.dtype)
            result = evaluate_model(args, model, tok, val_examples, name=f'val-step{step}')
            key = (result['raw']['em_norm']['value'], -result['raw']['cer']['value'])
            emit(step=step, 正确答案val_NLL=nll, selection_key=key)
            write_json(args.out/f'nll-step{step}.json', nll)
            if best is None or key > tuple(best['key']):
                best = dict(step=step, key=key, val_nll=nll, result=result)
                if args.save_checkpoint:
                    model.save_pretrained(args.out/'best')
                    tok.save_pretrained(args.out/'best')
                write_json(args.out/'selected.json', best)
            model.train()
    try:
        history = train_sft(model, train, tok.pad_token_id, steps=args.steps, batch_size=args.batch_size,
                            accumulation=args.accumulation, lr=args.lr, dtype=args.dtype, seed=args.seed, callback=callback)
    finally:
        history_file.close()
    metadata.update(selected_step=best['step'], train_seconds=sum(r['seconds'] for r in history),
                    total_seconds=time.perf_counter()-start,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
                    ended_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.out/'train_summary.json', metadata)
    emit(训练完成=metadata)


def part6(args):
    if not require_real(args):
        return
    model, tok = load_qwen3(device=args.device, dtype=torch.float32)
    examples = build_split('val', args.n, args.seed)
    copy = copy_baseline(examples)['summary']
    emit(基线='copy', 指标=copy)
    if args.out:
        write_json(args.out/'baseline-copy.json', copy)
    for mode in ('zero_shot', 'few_shot', 'few_shot_strict'):
        evaluate_model(args, model, tok, examples, name='baseline-'+mode, mode=mode)


def final_manifest(args):
    """测试前冻结 checkpoint 选择记录与完整生成配置；不允许重复评 test。"""
    if not args.final or not args.out:
        raise ValueError('test 只允许冻结后用 --final --out 显式评一次')
    selected = args.checkpoint.parent/'selected.json'
    if not selected.exists():
        raise ValueError('test 需要可追溯的 val 选择记录 selected.json')
    value = dict(checkpoint=str(args.checkpoint.resolve()), selected=json.loads(selected.read_text()),
                 config=vars(args), frozen_utc=datetime.now(timezone.utc).isoformat(),
                 selection_sha256=hashlib.sha256(selected.read_bytes()).hexdigest())
    # 先独占写入，再加载 test；失败也留下记录，不能无声重跑。
    path = args.out/'test_once.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=str)
    return value


def part7(args):
    if not require_real(args):
        return
    if args.checkpoint is None or not args.checkpoint.is_dir():
        raise ValueError('评估已训练模型须指定 --checkpoint 本地目录')
    if args.split == 'test':
        final_manifest(args)
    model, tok = load_qwen3(args.checkpoint, device=args.device, dtype=torch.float32)
    examples = build_split(args.split, args.n, args.seed)
    evaluate_model(args, model, tok, examples, name=f'final-{args.split}', save_rows=args.split != 'test')


def part8(args):
    tok = tokenizer(args)
    messages = zero_shot('x_2') + [dict(role='assistant', content='x^2'), dict(role='user', content='y_2')]
    rendered = render_chat(tok, messages)
    emit(多轮模板=repr(rendered), token_ids=tok.encode(rendered, add_special_tokens=False),
         说明='前一轮 assistant 是本轮上下文。本章只实现最后一轮答案监督；不能用本章单轮边界推断所有轮次 mask。')


PARTS = {1: ('模板与逐 token 监督', part1), 2: ('padding 与超长处理', part2),
         3: ('因果 shift 与手算 loss', part3), 4: ('微型模型四组故障对照', part4),
         5: ('Qwen3 自写全参 SFT', part5), 6: ('第 18 章同协议提示基线', part6),
         7: ('冻结 checkpoint 的 val/test 评估', part7), 8: ('多轮模板概览', part8)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parts', nargs='*', type=int)
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    parser.add_argument('--dtype', choices=('fp32', 'bf16'), default='fp32')
    parser.add_argument('--n', type=int, default=DEFAULT_N)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--real', action='store_true')
    parser.add_argument('--train-n', type=int, default=16)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--toy-steps', type=int, default=120)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--eval-batch-size', type=int, default=4)
    parser.add_argument('--accumulation', type=int, default=1)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--variant', choices=VARIANTS, default='correct')
    parser.add_argument('--eval-every', type=int, default=0)
    parser.add_argument('--save-checkpoint', action='store_true')
    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args(argv)
    if any(p not in PARTS for p in args.parts):
        parser.error('部分号必须为 1–8')
    if min(args.n, args.train_n, args.steps, args.toy_steps, args.batch_size, args.eval_batch_size, args.accumulation) < 1 or args.max_length < 2 or args.eval_every < 0:
        parser.error('规模参数须为正，max-length >= 2，eval-every >= 0')
    if not np.isfinite(args.lr) or args.lr <= 0:
        parser.error('lr 必须为正且有限')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA 不可用')
    if args.device == 'mps' and (not torch.backends.mps.is_available() or args.dtype == 'bf16'):
        parser.error('MPS 不可用或此循环不支持 MPS bf16')
    torch.set_num_threads(2)
    CONTEXT.update(device=args.device, device_name=torch.cuda.get_device_name() if args.device == 'cuda' else args.device,
                   dtype=args.dtype, n=args.n, threads=2)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    import transformers
    code_paths = [Path(__file__), Path('miniocr/data/sft.py'), Path('miniocr/train/sft.py'),
                  Path('miniocr/data/postcorrect.py'), Path('miniocr/eval/postcorrect_eval.py'), Path('miniocr/prompting.py')]
    provenance = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths}
    if args.out:
        write_json(args.out/'environment.json', dict(torch=torch.__version__, transformers=transformers.__version__,
                   code_sha256=provenance, **CONTEXT))
    emit(配置=vars(args), torch=torch.__version__, transformers=transformers.__version__, 默认n=DEFAULT_N,
         时间=datetime.now(timezone.utc).isoformat(), 数字口径='计时为实际 wall time；toy 为机制教具，真实实验同 postcorrect-v1')
    start = time.perf_counter()
    for p in args.parts or (1, 2, 3, 4, 8):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        title, fn = PARTS[p]
        emit(部分=p, 标题=title)
        begun = time.perf_counter()
        fn(args)
        emit(部分=p, 用时秒=time.perf_counter()-begun)
    emit(tour总用时秒=time.perf_counter()-start, 结束时间=datetime.now(timezone.utc).isoformat())


if __name__ == '__main__':
    main()
