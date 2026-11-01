"""第 33 章 VLM LoRA。默认 1/2 为无权重机制实验，3–6 显式选择。"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
from PIL import Image
import torch
from miniocr.data.vlm_sft import PROMPT, FormulaCollator, answer_labels
from miniocr.train.vlm_lora import adapter_spec, attach_lora, parameter_audit, move_batch
from miniocr.models.vlm_anatomy import local_snapshot, load_image_only_processor, MODELS


def emit(**value):
    print(' '.join(f'{k}={json.dumps(v, ensure_ascii=False, default=str)}' for k,v in value.items()), flush=True)


def save(args, name, value):
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def sync(args):
    if args.device == 'cuda': torch.cuda.synchronize()
    elif args.device == 'mps': torch.mps.synchronize()


def peak(args):
    return dict(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if args.device == 'cuda' else None,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if args.device == 'cuda' else None)


def reset(args):
    gc.collect()
    if args.device == 'cuda': torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


def rows_for(args, split, n):
    rows = json.loads((args.data/f'{split}.json').read_text())
    if len(rows) < n: raise ValueError(f'{split} 只有 {len(rows)} 条，需要 {n}')
    return rows[:n]


def align_generation_eos(model_key, model):
    """Qwen3.5 的模板以 im_end 结束；同时保留快照原有的 endoftext。"""
    if model_key == 'qwen35':
        model.generation_config.eos_token_id = [248046, 248044]


def load(args, adapter=None):
    from transformers import AutoModelForImageTextToText
    path = local_snapshot(args.model, args.model_cache)
    if path is None: raise FileNotFoundError(f'缺少本地 {MODELS[args.model]} 权重；不自动下载')
    processor = load_image_only_processor(path, args.model)
    processor.image_processor.size = {'shortest_edge': args.min_pixels, 'longest_edge': args.max_pixels}
    processor.tokenizer.padding_side = 'right'
    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(path, local_files_only=True, dtype=dtype,
                attn_implementation='sdpa').to(args.device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    align_generation_eos(args.model, model)
    emit(model=args.model, snapshot=str(path), adapter=adapter, dtype=args.dtype)
    return model, processor, dtype


def part1(args):
    ids = torch.tensor([[0, 11, 99, 12, 20, 21], [11, 99, 12, 22, 0, 0]], device=args.device)
    attention = torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]], device=args.device)
    labels = answer_labels(ids, attention, [3, 3], [99])
    logits = torch.randn(2, 6, 100, device=args.device, requires_grad=True)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1,100), labels[:,1:].reshape(-1))
    loss.backward()
    masked = labels[:,1:] == -100
    assert torch.count_nonzero(logits.grad[:, :-1][masked]) == 0
    result = dict(input_ids=ids.tolist(), labels=labels.tolist(), loss=float(loss.detach()), masked_gradient_zero=True)
    save(args, 'mask.json', result); emit(来源='本项目观察，随机机制探针', **result)


def toy_model():
    from torch import nn
    m = nn.Module(); m.model = nn.Module(); m.model.language_model = nn.Module()
    m.model.language_model.in_proj_qkv = nn.Linear(8, 24)
    m.model.language_model.q_proj = nn.Linear(8, 8)
    m.model.visual = nn.Module(); m.model.visual.qkv = nn.Linear(8, 24)
    m.model.visual.merger = nn.Sequential(nn.Linear(8, 8))
    return m


def part2(args):
    result = {s: adapter_spec(toy_model(), s) for s in ('language', 'connector', 'vision')}
    save(args, 'strategies.json', result); emit(来源='数学结构，缩小模块树', **result)


def part3(args):
    """一真实 batch 梯度审计及保存/新基座加载；随后 TRL 冒烟另用部分 4。"""
    reset(args); model, processor, dtype = load(args)
    model, spec = attach_lora(model, args.strategy, args.rank)
    audit = parameter_audit(model); save(args, 'parameters.json', dict(spec=spec, **audit))
    assert audit['groups']['language']['trainable'] > 0
    assert (audit['groups']['connector']['trainable'] > 0) == (args.strategy != 'language')
    assert (audit['groups']['vision']['trainable'] > 0) == (args.strategy == 'vision')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    batch = move_batch(FormulaCollator(processor, args.data, args.max_tokens)(rows_for(args,'train',args.batch_size)), args.device, dtype)
    model.eval()
    with model.disable_adapter(), torch.no_grad(): base_before = model(**batch,use_cache=False).logits.float().cpu()
    model.train(); start = time.perf_counter()
    loss = model(**batch, use_cache=False).loss
    assert torch.isfinite(loss)
    loss.backward()
    gradients = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None: raise AssertionError(f'没有梯度: {name}')
            if not torch.isfinite(p.grad).all(): raise AssertionError(f'梯度非有限: {name}')
            if p.grad.abs().max() > 0: gradients[name] = float(p.grad.float().norm())
        elif p.grad is not None: raise AssertionError(f'冻结参数产生梯度: {name}')
    assert gradients, '所有可训练参数的梯度均为零'
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    model.eval()
    with model.disable_adapter(), torch.no_grad(): base_after = model(**batch,use_cache=False).logits.float().cpu()
    disabled_error = float((base_before-base_after).abs().max())
    assert disabled_error == 0, '禁用 adapter 未恢复原基座，包括原始连接器'
    del base_before,base_after
    with torch.no_grad(): before = model(**batch, use_cache=False).logits.float().cpu()
    checkpoint = args.out/'adapter'; model.save_pretrained(checkpoint)
    memory = peak(args); del optimizer, model; reset(args)
    model, _, _ = load(args, checkpoint); model.eval()
    with torch.no_grad(): after = model(**batch, use_cache=False).logits.float().cpu()
    error = float((before-after).abs().max())
    assert error < (0.04 if args.dtype == 'bf16' else 1e-5), error
    sync(args)
    reload_memory = peak(args)
    memory = {k: max(v, reload_memory[k]) if v is not None else None for k,v in memory.items()}
    result = dict(loss=float(loss.detach()), supervised_tokens=int((batch['labels'] != -100).sum()),
        image_tokens_masked=bool((batch['labels'][batch['input_ids']==processor.image_token_id] == -100).all()),
        trainable=audit['trainable'], groups=audit['groups'], nonzero_gradient_parameters=len(gradients),
        gradient_norms=gradients, disabled_adapter_max_abs_error=disabled_error, reload_max_abs_error=error, seconds=time.perf_counter()-start, **memory)
    save(args,'smoke.json',result); emit(来源='本项目观察', **{k:v for k,v in result.items() if k != 'gradient_norms'})


def part4(args):
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    reset(args); model, processor, dtype = load(args)
    model, spec = attach_lora(model, args.strategy, args.rank)
    audit = parameter_audit(model); save(args,'parameters.json',dict(spec=spec, **audit))
    train_rows = rows_for(args, 'train', args.train_n)
    config = SFTConfig(output_dir=str(args.out/'trainer'), max_steps=args.steps,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.accum,
        learning_rate=args.lr, warmup_steps=min(20, max(1,args.steps//10)), lr_scheduler_type='cosine',
        bf16=args.dtype=='bf16', fp16=False, use_cpu=args.device=='cpu',
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant':False},
        max_length=None, packing=False, dataset_kwargs={'skip_prepare_dataset': True},
        remove_unused_columns=False, save_strategy='no', eval_strategy='no', logging_steps=max(1,min(10,args.steps//5)),
        logging_first_step=True, report_to='none', disable_tqdm=True, seed=args.seed,
        dataloader_num_workers=0, dataloader_pin_memory=args.device=='cuda', optim='adamw_torch',
        completion_only_loss=False, eos_token='<|im_end|>' if args.model == 'qwen35' else None)
    trainer = SFTTrainer(model=model, args=config, train_dataset=Dataset.from_list(train_rows),
        data_collator=FormulaCollator(processor,args.data,args.max_tokens), processing_class=processor)
    # TRL 初始化会按 tokenizer 对齐 EOS；随后恢复双停止 token。监督模板保持不变。
    align_generation_eos(args.model, trainer.model)
    start = time.perf_counter(); result = trainer.train(); sync(args)
    trainer.model.save_pretrained(args.out/'adapter')
    result = dict(metrics=result.metrics, history=trainer.state.log_history, seconds=time.perf_counter()-start,
        config=vars(args), trainable=audit['trainable'], groups=audit['groups'], **peak(args))
    save(args,'training.json',result); emit(来源='本项目观察', **{k:v for k,v in result.items() if k not in ('history','config')})


def trim_generated(tokens, eos_ids):
    """Remove batch padding after the first generated EOS, keeping EOS itself."""
    for i, token in enumerate(tokens.tolist()):
        if token in eos_ids:
            return tokens[:i+1]
    return tokens


def evaluate(args, model, processor, dtype, rows, condition='correct', split='val'):
    from miniocr.data.postcorrect import PostCorrectExample
    from miniocr.eval.postcorrect_eval import score_postcorrect, extract_formula
    predictions = []; observations = []; model.eval(); start = time.perf_counter()
    if condition == 'wrong' and len(rows)<2: raise ValueError('错图对照至少两条')
    align_generation_eos(args.model, model)
    generation_kwargs = {'eos_token_id': [248046, 248044]} if args.model == 'qwen35' else {}
    batch_size = getattr(args, 'eval_batch_size', 1)
    previous_padding = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = 'left'
    try:
        for offset in range(0, len(rows), batch_size):
            chunk = rows[offset:offset+batch_size]; images = []; prompts = []
            for i, row in enumerate(chunk, offset):
                with Image.open(args.data/row['image']) as opened: im = opened.convert('RGB')
                if condition == 'wrong':
                    with Image.open(args.data/rows[(i+1)%len(rows)]['image']) as opened: im = opened.convert('RGB').resize(im.size)
                elif condition == 'blank': im = Image.new('RGB',im.size,'white')
                messages = [{'role':'user','content':[{'type':'image'},{'type':'text','text':PROMPT}]}]
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                images.append(im); prompts.append(prompt)
            batch = move_batch(processor(text=prompts, images=images, padding=True, return_tensors='pt'),args.device,dtype)
            with torch.inference_mode(): output = model.generate(**batch, max_new_tokens=args.max_new_tokens,do_sample=False,use_cache=True,**generation_kwargs)
            eos = model.generation_config.eos_token_id; eos = eos if isinstance(eos,list) else [eos]
            for row, sequence in zip(chunk, output):
                tokens = sequence[batch['input_ids'].shape[1]:]
                tokens = trim_generated(tokens, eos)
                raw = processor.decode(tokens,skip_special_tokens=True)
                pred = extract_formula(raw); predictions.append(pred)
                obs = dict(id=row['id'],generated_tokens=len(tokens),truncated=len(tokens)==args.max_new_tokens and int(tokens[-1]) not in eos)
                if split == 'val': obs.update(target=row['target'],raw=raw,pred=pred)
                observations.append(obs)
            emit(condition=condition,split=split,completed=len(observations),n=len(rows),seconds=time.perf_counter()-start)
    finally:
        processor.tokenizer.padding_side = previous_padding
    examples = [PostCorrectExample(r['id'],split,r['target'],r['target']) for r in rows]
    metrics = score_postcorrect(examples,predictions)['summary']
    sync(args)
    result = dict(model=args.model,model_id=MODELS[args.model],snapshot=str(local_snapshot(args.model,args.model_cache)),
        prompt=PROMPT,formatted_prompt=prompt,enable_thinking=False,do_sample=False,attn_implementation='sdpa',
        processor=processor.image_processor.to_dict(),eos_token_id=model.generation_config.eos_token_id,
        adapter=str(args.adapter) if args.adapter else None, split=split,condition=condition,
        metrics={k:metrics[k] for k in ('n','em_raw','em_norm','cer','parse_rate')},
        sample_ids_sha256=hashlib.sha256(json.dumps([r['id'] for r in rows]).encode()).hexdigest(),
        seconds=time.perf_counter()-start, truncated=sum(r['truncated'] for r in observations),
        observations=observations,config=vars(args), **peak(args))
    return result, predictions


def part5(args):
    if args.split == 'test':
        if not args.final_test or not args.adapter: raise ValueError('test 必须 --final-test --adapter，冻结配置后基线/微调各评一次')
        # >>> 练习 33.3：全数据目录锁，换 out 也不能悄悄重评 test。
        with (args.data/'ch33_final_test.lock').open('x') as f:
            json.dump(dict(time=datetime.now(timezone.utc).isoformat(),config=vars(args)),f,default=str,indent=2)
    rows = rows_for(args,args.split,args.n)
    reset(args); model, processor, dtype = load(args,args.adapter)
    modes = ['baseline','finetuned'] if args.split=='test' else ['finetuned' if args.adapter else 'baseline']
    for mode in modes:
        with model.disable_adapter() if mode=='baseline' and args.adapter else nullcontext():
            result,_ = evaluate(args,model,processor,dtype,rows,split=args.split)
        result['mode'] = mode
        save(args,f'{args.split}_{mode}.json',result); emit(mode=mode,split=args.split,metrics=result['metrics'])


def part6(args):
    if args.split != 'val' or not args.adapter: raise ValueError('视觉依赖对照仅 val，必须 --adapter')
    rows = rows_for(args,'val',args.n); reset(args); model, processor, dtype = load(args,args.adapter)
    reference = None
    for condition in ('correct','wrong','blank'):
        result,predictions = evaluate(args,model,processor,dtype,rows,condition)
        if reference is None: reference = predictions
        result['changed_predictions'] = sum(a!=b for a,b in zip(reference,predictions))
        save(args,f'dependency_{condition}.json',result)
        emit(condition=condition,metrics=result['metrics'],changed=result['changed_predictions'])


PARTS = {1:('label mask',part1),2:('冻结策略与 DeltaNet 目标',part2),3:('真实梯度/保存/加载',part3),
         4:('TRL LoRA 训练',part4),5:('基线与微调评测',part5),6:('视觉依赖',part6)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('parts',type=int,nargs='*'); p.add_argument('--device',choices=('cpu','mps','cuda'),default='cpu')
    p.add_argument('--dtype',choices=('fp32','bf16'),default='fp32'); p.add_argument('--n',type=int,default=20)
    p.add_argument('--train-n',type=int,default=16); p.add_argument('--steps',type=int,default=5)
    p.add_argument('--batch-size',type=int,default=1); p.add_argument('--accum',type=int,default=1)
    p.add_argument('--rank',type=int,default=8); p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--strategy',choices=('language','connector','vision'),default='language')
    p.add_argument('--model',choices=('qwen35','qwen3vl'),default='qwen35'); p.add_argument('--model-cache',type=Path)
    p.add_argument('--data',type=Path); p.add_argument('--prepare-data',action='store_true')
    p.add_argument('--audit-data',action='store_true')
    p.add_argument('--out',type=Path,default=Path('outputs/book/ch33/local')); p.add_argument('--adapter',type=Path)
    p.add_argument('--seed',type=int,default=0); p.add_argument('--split',choices=('val','test'),default='val')
    p.add_argument('--final-test',action='store_true'); p.add_argument('--min-pixels',type=int,default=16384)
    p.add_argument('--max-pixels',type=int,default=65536); p.add_argument('--max-new-tokens',type=int,default=256)
    p.add_argument('--max-tokens',type=int,default=2048)
    p.add_argument('--eval-batch-size',type=int,default=1)
    args = p.parse_args(argv)
    if any(x<1 for x in (args.n,args.train_n,args.steps,args.batch_size,args.accum,args.rank,args.max_tokens,args.max_new_tokens,args.min_pixels,args.eval_batch_size)) or args.max_pixels<args.min_pixels: p.error('规模必须为正，像素上限不得小于下限')
    if any(x not in PARTS for x in args.parts): p.error('部分号为 1–6')
    if args.device=='cuda' and not torch.cuda.is_available(): p.error('CUDA 不可用')
    if args.device=='mps' and not torch.backends.mps.is_available(): p.error('MPS 不可用')
    if (args.prepare_data or args.audit_data or any(x>=3 for x in args.parts)) and args.data is None: p.error('需要 --data')
    if args.split=='test' and args.parts != [5]: p.error('test 仅可单独运行部分 5')
    torch.set_num_threads(2); start=time.perf_counter()
    emit(start_utc=datetime.now(timezone.utc).isoformat(),config=vars(args),torch=torch.__version__,
         device=torch.cuda.get_device_name() if args.device=='cuda' else args.device)
    save(args,'config.json',vars(args))
    if args.prepare_data:
        from miniocr.data.vlm_sft import prepare_images
        emit(data=prepare_images(args.data,args.train_n,args.n))
    elif args.audit_data:
        from miniocr.data.vlm_sft import audit_samples
        path = local_snapshot(args.model,args.model_cache)
        if path is None: raise FileNotFoundError('缺少 processor 缓存')
        processor = load_image_only_processor(path,args.model)
        processor.image_processor.size = {'shortest_edge':args.min_pixels,'longest_edge':args.max_pixels}
        result = audit_samples(processor,args.data,rows_for(args,'train',args.train_n),args.max_tokens)
        save(args,'data_audit.json',result); emit(**{k:v for k,v in result.items() if k!='rows'})
    else:
        for number in args.parts or (1,2):
            random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
            title, fn = PARTS[number]; emit(部分=number,标题=title); begun=time.perf_counter()
            fn(args); reset(args); emit(部分=number,seconds=time.perf_counter()-begun)
    timing = dict(end_utc=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-start)
    save(args,'timing.json',timing); emit(**timing)


if __name__ == '__main__': main()
