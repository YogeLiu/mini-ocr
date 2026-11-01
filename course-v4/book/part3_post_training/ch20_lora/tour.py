"""第 20 章：自写 LoRA、PEFT 对拍、真实 Qwen 合并、显存与后纠错训练。"""
import argparse
from contextlib import nullcontext
import copy
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch
from torch import nn

from miniocr.nn.lora import (LoRALinear, inject_lora, merge_lora, parameter_counts,
                            adapter_state, memory_estimate)
from miniocr.models.qwen3 import find_local_qwen3, load_qwen3
from miniocr.data.postcorrect import build_split, save_jsonl, load_jsonl, check_no_leak
from miniocr.data.latex_corpus import corpus_available
from miniocr.data.sft import encode_sft, collate_sft
from miniocr.train.sft import train_sft, mean_loss, causal_loss, autocast_context, synchronize
from miniocr.prompting import render_chat, zero_shot, generate_greedy
from miniocr.eval.postcorrect_eval import score_postcorrect, extract_formula, copy_baseline


def emit(**record):
    print(json.dumps(dict(source='本项目观察', utc=datetime.now(timezone.utc).isoformat(), **record),
                     ensure_ascii=False, default=str), flush=True)


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str)+'\n')


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def mechanism(device='cpu'):
    """B=0 时 A 的梯度为零；更新 B 后验证输入梯度和合并。"""
    torch.manual_seed(0)
    base = nn.Linear(11, 7).to(device)
    layer = LoRALinear(base, 4, 8)
    x = torch.randn(3, 5, 11, device=device, requires_grad=True)
    initial_error = (base(x)-layer(x)).abs().max().item()
    layer(x).square().mean().backward()
    a_grad, b_grad = layer.A.grad.norm().item(), layer.B.grad.norm().item()
    with torch.no_grad():
        layer.B.normal_(std=.05)
    expected = layer(x)
    actual = layer.merged()(x)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    assert initial_error == 0 and a_grad == 0 and b_grad > 0 and base.weight.grad is None
    return dict(initial_max_error=initial_error, first_A_grad_norm=a_grad, first_B_grad_norm=b_grad,
                merge_max_error=(actual-expected).abs().max().item(), **parameter_counts(layer))


def parity(device='cpu'):
    """与 PEFT 普通 Linear 的同配置前向和 A/B/输入梯度对拍；非零 B 防止空对拍。"""
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(1)
    base = nn.Linear(13, 9).to(device)
    ours = LoRALinear(copy.deepcopy(base), 4, 8)
    holder = nn.ModuleDict({'projection': copy.deepcopy(base)})
    wrapped = get_peft_model(holder, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.,
                              target_modules=['projection'], bias='none', init_lora_weights=True))
    theirs = wrapped.base_model.model['projection']
    with torch.no_grad():
        ours.B.normal_(std=.03)
        theirs.lora_A['default'].weight.copy_(ours.A)
        theirs.lora_B['default'].weight.copy_(ours.B)
    x = torch.randn(2, 3, 13, device=device, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    a, b = ours(x), theirs(y)
    a.square().sum().backward(); b.square().sum().backward()
    pairs = dict(forward=(a,b), A_grad=(ours.A.grad,theirs.lora_A['default'].weight.grad),
                 B_grad=(ours.B.grad,theirs.lora_B['default'].weight.grad), input_grad=(x.grad,y.grad))
    for u,v in pairs.values():
        torch.testing.assert_close(u,v,atol=2e-6,rtol=2e-6)
    return {key: (u-v).abs().max().item() for key,(u,v) in pairs.items()}


def part1(args):
    return mechanism(args.device)


def tiny_qwen():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8))


def part2(args):
    rows = []
    for rank in args.ranks:
        for alpha in (rank, 2*rank):
            for targets in target_sets(args):
                model = tiny_qwen()
                names = inject_lora(model, targets, rank, alpha)
                counts = parameter_counts(model)
                assert counts['formula'] == counts['trainable']
                rows.append(dict(rank=rank, alpha=alpha, targets=targets, modules=len(names), **counts))
    return dict(origin='数学公式与代码统计（小型 Qwen 架构，不是 0.6B 参数数）', rows=rows)


def part3(args):
    if importlib.util.find_spec('peft') is None:
        return dict(skipped='PEFT 未安装；没有自动安装')
    return parity(args.device)


def real_model(args):
    # 基座/adapter fp32；训练与生成按 --dtype autocast。合并校验固定 fp32。
    return load_qwen3(device=args.device, dtype=torch.float32)


def part4(args):
    if find_local_qwen3() is None:
        return dict(skipped='没有本地 Qwen3-0.6B 权重')
    model, tok = real_model(args)
    selected = None
    if args.checkpoint_dir is not None:
        selected = json.loads((args.checkpoint_dir/'selection.json').read_text())
    targets = selected['targets'] if selected else ('q_proj','v_proj')
    rank, alpha = (selected['rank'],selected['alpha']) if selected else (4,8)
    names = inject_lora(model, targets, rank, alpha)
    counts = parameter_counts(model)
    torch.manual_seed(args.seed)
    if selected:
        state = torch.load(args.checkpoint_dir/f"{selected['winner']}.pt",map_location=args.device,weights_only=True)
        assert set(state) == set(adapter_state(model))
        model.load_state_dict(state,strict=False)
    else:
        for module in model.modules():
            if isinstance(module, LoRALinear):
                with torch.no_grad():
                    module.B.normal_(std=.001)
    ids = tok('Correct OCR: x^{2}+y^{2}', return_tensors='pt').to(args.device)
    model.eval()
    with torch.no_grad():
        before = model(**ids).logits
        merge_lora(model)
        after = model(**ids).logits
    error = (before-after).abs().max().item()
    torch.testing.assert_close(before, after, atol=2e-4, rtol=2e-4)
    result = dict(modules=len(names), max_logit_error=error, dtype='fp32',
                  adapter=selected['winner'] if selected else 'random nonzero B', **counts)
    del model; release()
    return result


def part5(args):
    if find_local_qwen3() is None:
        return dict(skipped='没有本地 Qwen3 权重')
    rows=[]
    for batch, length in ((1,128),(2,256),(2,512)):
        model, tok = real_model(args)
        inject_lora(model, ('q_proj','v_proj'), 16, 32)
        estimate = memory_estimate(model,batch,length,activation_bytes=2 if args.dtype=='bf16' else 4)
        from miniocr.data.sft import SFTExample
        ex = SFTExample(tuple([100]*length), tuple([-100]*32+[100]*(length-32)),32)
        if args.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        tensors={k:v.to(args.device) for k,v in collate_sft([ex]*batch,tok.pad_token_id).items()}
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
        def allocated():
            synchronize(args.device)
            return torch.cuda.memory_allocated() if args.device=='cuda' else None
        stages=[]
        start=time.perf_counter()
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            before=allocated()
            with autocast_context(args.device,args.dtype):
                logits=model(input_ids=tensors['input_ids'],attention_mask=tensors['attention_mask'],use_cache=False).logits
                loss=causal_loss(logits,tensors['labels'])
            forward=allocated()
            loss.backward()
            backward=allocated()
            optimizer.step()
            stages.append(dict(before=before,after_forward=forward,after_backward=backward,
                               after_optimizer=allocated(),loss=loss.item()))
            del logits,loss
        state_bytes=sum(v.numel()*v.element_size() for state in optimizer.state.values()
                        for v in state.values() if torch.is_tensor(v))
        grad_bytes=sum(p.grad.numel()*p.grad.element_size() for p in model.parameters() if p.grad is not None)
        rows.append(dict(batch=batch,length=length,estimate=estimate,
                         peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None,
                         peak_reserved_bytes=torch.cuda.max_memory_reserved() if args.device=='cuda' else None,
                         measured_optimizer_state_bytes=state_bytes,measured_gradient_bytes=grad_bytes,
                         stages=stages,seconds=time.perf_counter()-start))
        del model,optimizer,tensors; release()
    return dict(rows=rows)


def target_sets(args):
    return [tuple(x.split(',')) for x in args.targets.split(';')]


def get_data(args, split, n):
    path=args.out/f'{split}-n{n}-seed{args.seed}.jsonl'
    if path.exists():
        return load_jsonl(path)
    examples=build_split(split,n,seed=args.seed)
    save_jsonl(examples,path)
    return examples


def encode(examples,tok,length):
    result=[encode_sft(tok,e.source,e.target,max_length=length,id=e.id) for e in examples]
    return [e for e in result if e is not None]


def evaluate_model(args,model,tok,examples,label):
    prompts=[render_chat(tok,zero_shot(e.source)) for e in examples]
    with autocast_context(args.device,args.dtype):
        gen=generate_greedy(model,tok,prompts,batch_size=args.eval_batch_size,
                            sources=[e.source for e in examples],max_new_tokens=args.max_new_tokens)
    raw=score_postcorrect(examples,gen.texts)
    extracted=score_postcorrect(examples,[extract_formula(s) for s in gen.texts])
    # test 只存汇总，避免随后查看逐条错例。
    if examples[0].split != 'test':
        write(args.out/f'{label}-predictions.json',dict(raw=raw['rows'],extracted=extracted['rows']))
    summary=dict(raw=raw['summary'],extracted=extracted['summary'],seconds=gen.seconds,
                 capped=sum(gen.length_capped),generated_tokens=sum(gen.generated_tokens))
    write(args.out/f'{label}-metrics.json',summary)
    emit(evaluation=label,em_norm=summary['extracted']['em_norm'],cer=summary['extracted']['cer'],seconds=gen.seconds)
    return summary


def part6(args):
    if find_local_qwen3() is None:
        return dict(skipped='没有本地 Qwen3 权重')
    train=get_data(args,'train',args.train_n)
    val=get_data(args,'val',args.n)
    assert check_no_leak(dict(train=train,val=val))['ok']
    model,tok=real_model(args)
    encoded=encode(train,tok,args.max_length)
    val_encoded=encode(val,tok,args.max_length)
    if not encoded or not val_encoded:
        raise ValueError('长度过滤后数据为空')
    first=collate_sft(encoded[:args.batch_size],tok.pad_token_id)
    mask_report=dict(train_requested=len(train),train_accepted=len(encoded),val_n=len(val),
                     val_loss_n=len(val_encoded),supervised_tokens=int((first['labels']!=-100).sum()),
                     prompt_mask_ok=all(all(x==-100 for x in e.labels[:e.prompt_length]) for e in encoded),
                     eos_ok=all(e.labels[-1]==tok.eos_token_id for e in encoded),
                     real_corpus=corpus_available())
    write(args.out/'mask-check.json',mask_report); emit(mask_check=mask_report)
    baseline=evaluate_model(args,model,tok,val,'baseline-val')
    write(args.out/'copy-val.json',copy_baseline(val)['summary'])
    del model; release()
    rows=[]
    for targets in target_sets(args):
        for rank in args.ranks:
            torch.manual_seed(args.seed)
            model,tok=real_model(args)
            inject_lora(model,targets,rank,args.alpha_ratio*rank)
            name=f'r{rank}-'+ '-'.join(targets)
            counts=parameter_counts(model)
            assert counts['trainable']==counts['formula']
            if args.device=='cuda':
                torch.cuda.reset_peak_memory_stats()
            estimate=memory_estimate(model,args.batch_size,args.max_length,
                                     activation_bytes=2 if args.dtype=='bf16' else 4)
            emit(training=name,counts=counts,memory_estimate=estimate)
            def callback(step,record):
                if step==1 or step%10==0 or step==args.steps:
                    emit(training=name,**record)
            history=train_sft(model,encoded,tok.pad_token_id,steps=args.steps,batch_size=args.batch_size,
                accumulation=args.accumulation,lr=args.lr,dtype=args.dtype,seed=args.seed,callback=callback)
            peak=torch.cuda.max_memory_allocated() if args.device=='cuda' else None
            reserved=torch.cuda.max_memory_reserved() if args.device=='cuda' else None
            torch.save(adapter_state(model),args.out/f'{name}.pt')
            write(args.out/f'{name}-history.json',history)
            nll=mean_loss(model,val_encoded,tok.pad_token_id,batch_size=args.batch_size,dtype=args.dtype)
            metrics=evaluate_model(args,model,tok,val,name+'-val')
            seconds=sum(x['seconds'] for x in history)
            row=dict(name=name,rank=rank,alpha=args.alpha_ratio*rank,targets=targets,counts=counts,
                val_nll=nll,metrics=metrics,training_seconds=seconds,peak_allocated_bytes=peak,
                peak_reserved_bytes=reserved,memory_estimate=estimate,
                examples_per_second=args.steps*args.batch_size*args.accumulation/seconds,
                supervised_tokens_per_second=sum(x['supervised_tokens'] for x in history)/seconds,
                first_loss=history[0]['loss'],last_loss=history[-1]['loss'])
            rows.append(row); write(args.out/'sweep.json',rows)
            del model; release()
    # 主指标 raw EM_norm；同分按 CER、rank、名字排序。规则事先固定。
    winner=min(rows,key=lambda r:(-r['metrics']['raw']['em_norm']['value'],
                                  r['metrics']['raw']['cer']['value'],r['rank'],r['name']))
    selection=dict(winner=winner['name'],rank=winner['rank'],alpha=winner['alpha'],targets=winner['targets'],
                   rule='val raw EM_norm desc, CER asc, rank asc, name asc',
                   args=vars(args),base_snapshot=str(find_local_qwen3()),
                   baseline_em=baseline['raw']['em_norm']['value'])
    write(args.out/'selection.json',selection)
    return dict(selection=selection,rows=rows)


def part7(args):
    return dict(bitsandbytes_available=importlib.util.find_spec('bitsandbytes') is not None,
                status='仅演示思路；未进行量化训练，也没有安装新包',
                idea='冻结的基座以4bit存储，计算时反量化；LoRA A/B及梯度、Adam状态保持高精度。',
                caveat='权重压缩不消除激活、logits、workspace和adapter状态；不能把4bit权重大小当训练峰值。')


def part8(args):
    if not args.final_test:
        return dict(skipped='最终设置冻结后显式 --final-test 才能评一次 test')
    selection=json.loads((args.out/'selection.json').read_text())
    if (args.out/'FINAL_TEST_STARTED.json').exists():
        raise ValueError('该实验目录已开始过 test；禁止反复评测')
    saved=selection['args']
    for key in ('dtype','seed','max_length','max_new_tokens','eval_batch_size','n'):
        if getattr(args,key)!=saved[key]:
            raise ValueError(f'最终评测参数必须与选择时一致：{key}')
    write(args.out/'FINAL_TEST_STARTED.json',selection)
    examples=get_data(args,'test',args.n)
    model,tok=real_model(args)
    inject_lora(model,selection['targets'],selection['rank'],selection['alpha'])
    state=torch.load(args.out/f"{selection['winner']}.pt",map_location=args.device,weights_only=True)
    assert set(state)==set(adapter_state(model))
    model.load_state_dict(state,strict=False)
    metrics=evaluate_model(args,model,tok,examples,'final-test')
    del model; release()
    return dict(selected=selection['winner'],test_only_once=True,metrics=metrics)


PARTS={1:('机制与零初始化',part1),2:('rank/alpha/目标模块参数公式',part2),
       3:('PEFT前向与梯度对拍',part3),4:('Qwen挂载与合并',part4),
       5:('显存先估后测',part5),6:('后纠错rank与目标模块扫描',part6),
       7:('QLoRA思路',part7),8:('冻结后的单次test',part8)}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('parts',nargs='*',type=int)
    p.add_argument('--device',choices=('cpu','mps','cuda'),default='cpu')
    p.add_argument('--dtype',choices=('fp32','bf16'),default='fp32')
    p.add_argument('--n',type=int,default=20)
    p.add_argument('--train-n',type=int,default=32)
    p.add_argument('--steps',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--accumulation',type=int,default=1)
    p.add_argument('--eval-batch-size',type=int,default=2)
    p.add_argument('--max-length',type=int,default=384)
    p.add_argument('--max-new-tokens',type=int,default=192)
    p.add_argument('--ranks',type=int,nargs='+',default=[4])
    p.add_argument('--targets',default='q_proj,v_proj',help='逗号分模块，分号分配置组')
    p.add_argument('--alpha-ratio',type=float,default=2.)
    p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--final-test',action='store_true')
    p.add_argument('--checkpoint-dir',type=Path,help='部分4可选：验证已选训练adapter的合并')
    p.add_argument('--out',type=Path,default=Path('outputs/book/ch20/local'))
    args=p.parse_args(argv)
    if any(i not in PARTS for i in args.parts) or min(args.n,args.train_n,args.steps,args.batch_size,
            args.accumulation,args.eval_batch_size,args.max_length,args.max_new_tokens,*args.ranks)<1:
        p.error('部分号/规模参数非法')
    if args.device=='cuda' and not torch.cuda.is_available(): p.error('CUDA不可用')
    if args.device=='mps' and (not torch.backends.mps.is_available() or args.dtype=='bf16'): p.error('MPS不可用或不支持本实验bf16')
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    import transformers
    paths = [Path(__file__), Path('miniocr/nn/lora.py'), Path('miniocr/data/sft.py'),
             Path('miniocr/train/sft.py'), Path('miniocr/data/postcorrect.py'),
             Path('miniocr/data/latex_corpus.py'), Path('miniocr/eval/postcorrect_eval.py'),
             Path('miniocr/prompting.py')]
    write(args.out/('environment-'+ '-'.join(map(str,args.parts or [1,2,3,7]))+'.json'),dict(torch=torch.__version__, transformers=transformers.__version__,
          code_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
          config=vars(args),base_snapshot=str(find_local_qwen3())))
    emit(config=vars(args),torch=torch.__version__,transformers=transformers.__version__,
         device_name=torch.cuda.get_device_name() if args.device=='cuda' else args.device,
         precision='fp32参数/Adam + '+args.dtype+'计算',base_snapshot=find_local_qwen3())
    # 默认只跑快速机制；真实模型与训练须指定部分号。
    for i in args.parts or [1,2,3,7]:
        torch.manual_seed(args.seed)
        title,fn=PARTS[i]; start=time.perf_counter(); emit(part=i,title=title,event='start')
        result=fn(args)
        write(args.out/f'part{i}.json',result)
        emit(part=i,result=result,seconds=time.perf_counter()-start)


if __name__=='__main__':
    main()
