"""第 21 章：TRL 检查清单、真实 LoRA SFT、val 阶段对照与单次 test。"""
import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import torch

from miniocr.data.postcorrect import build_split, save_jsonl, load_jsonl, check_no_leak, few_shot_pool
from miniocr.data.latex_corpus import corpus_available, _parquet_paths
from miniocr.data.sft import encode_sft, collate_sft
from miniocr.models.qwen3 import find_local_qwen3, load_qwen3
from miniocr.train.trl_sft import make_trainer, tiny_fixture, audit_batch, loss_step_parity
from miniocr.train.sft import mean_loss, autocast_context
from miniocr.eval.postcorrect_eval import score_postcorrect, extract_formula, copy_baseline
from miniocr.prompting import render_chat, zero_shot, few_shot_strict, generate_greedy


def emit(**record):
    print(json.dumps(dict(source='本项目观察',utc=datetime.now(timezone.utc).isoformat(),**record),
                     ensure_ascii=False,default=str),flush=True)


def write(path, obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n')


def release():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def data(args, split, n):
    root=args.data_dir or args.out/'data'
    root.mkdir(parents=True,exist_ok=True)
    path=root/f'{split}-n{n}-seed{args.seed}.jsonl'
    if path.exists(): return load_jsonl(path)
    rows=build_split(split,n,seed=args.seed)
    save_jsonl(rows,path)
    return rows


def encode(rows,tok,max_length):
    return [x for e in rows if (x:=encode_sft(tok,e.source,e.target,max_length=max_length,id=e.id)) is not None]


def counts(model):
    return dict(total=sum(p.numel() for p in model.parameters()),
                trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                names=[n for n,p in model.named_parameters() if p.requires_grad])


def evaluate(args,model,tok,rows,name,*,prompts=None):
    with autocast_context(args.device,args.dtype):
        gen=generate_greedy(model,tok,prompts or [render_chat(tok,zero_shot(e.source)) for e in rows],
            batch_size=args.eval_batch_size,max_new_tokens=args.max_new_tokens)
    raw=score_postcorrect(rows,gen.texts)
    extracted=score_postcorrect(rows,[extract_formula(s) for s in gen.texts])
    result=dict(raw=raw['summary'],extracted=extracted['summary'],seconds=gen.seconds,
                capped=sum(gen.length_capped),generated_tokens=sum(gen.generated_tokens))
    if rows[0].split!='test':
        write(args.out/f'{name}-predictions.json',dict(raw=raw['rows'],extracted=extracted['rows']))
    write(args.out/f'{name}-metrics.json',result)
    emit(evaluation=name,metrics=result)
    return result


def part1(args):
    model,tok,examples=tiny_fixture()
    trainer=make_trainer(model.to(args.device),tok,examples,args.out/'tiny',device=args.device)
    batch=trainer.data_collator(trainer.train_dataset.to_list())
    reference=collate_sft(examples,tok.pad_token_id)
    for k in reference: torch.testing.assert_close(batch[k],reference[k])
    audit=audit_batch(batch,tok,examples)
    # 不等长 microbatch 的分母应相加，而不是平均两个 mean loss。
    parity=loss_step_parity(trainer,[trainer.data_collator([row]) for row in trainer.train_dataset])
    del trainer,model; release()
    return dict(batch=audit,parity=parity)


def prepare(args):
    if find_local_qwen3() is None: raise FileNotFoundError('没有本地 Qwen3-0.6B 权重；不会下载')
    train,val=data(args,'train',args.train_n),data(args,'val',args.n)
    assert check_no_leak(dict(train=train,val=val))['ok']
    model,tok=load_qwen3(device=args.device,dtype=torch.float32)
    write(args.out/'base-generation-config.json',model.generation_config.to_dict())
    encoded=encode(train,tok,args.max_length)
    if not encoded: raise ValueError('长度过滤后训练集为空')
    manifest=dict(version='postcorrect-v1',split='latex-v1 seed=0',noise_seed=args.seed,
        real_corpus=corpus_available(),parquet=[str(p) for p in _parquet_paths()],
        train_requested=len(train),train_accepted=len(encoded),val_n=len(val),
        train_ids_sha256=hashlib.sha256('\n'.join(x.id for x in encoded).encode()).hexdigest(),
        val_ids_sha256=hashlib.sha256('\n'.join(x.id for x in val).encode()).hexdigest())
    write(args.out/'data-manifest.json',manifest)
    emit(data=manifest)
    return model,tok,encoded,val


def part2(args):
    model,tok,encoded,_=prepare(args)
    trainer=make_trainer(model,tok,encoded,args.out/'audit',device=args.device,rank=args.rank)
    # 真正的 train dataloader 第一批；其内容与手写 collator 单独对拍。
    batch=next(iter(trainer.get_train_dataloader()))
    audit=audit_batch(batch,tok,encoded)
    write(args.out/'first-batch.json',audit)
    for row in audit['rows']:
        for token in row: emit(token=token)
    reference=collate_sft(encoded[:2],tok.pad_token_id)
    actual=trainer.data_collator(trainer.train_dataset.select(range(min(2,len(encoded)))).to_list())
    for k in reference: torch.testing.assert_close(reference[k],actual[k])
    # 真实模型只取短样本，FP32 对拍控制显存。
    short=sorted(encoded,key=lambda e:len(e.input_ids))[:2]
    parity=loss_step_parity(trainer,[collate_sft([x],tok.pad_token_id) for x in short])
    result=dict(parity=parity,parameters=counts(trainer.model),batch=audit)
    assert all('lora_' in n for n in result['parameters']['names'])
    del trainer,model; release()
    return result


def part3(args):
    from transformers import TrainerCallback
    if (args.out/'selection.json').exists(): raise ValueError('已有选择结果，请使用新的 --out')
    model,tok,encoded,val=prepare(args)
    baseline=evaluate(args,model,tok,val,'baseline-val')
    if args.strong_baseline:
        shots=few_shot_pool(3,seed=args.seed)
        write(args.out/'few-shot-ids.json',[x.id for x in shots])
        evaluate(args,model,tok,val,'strict-3shot-val',
                 prompts=[render_chat(tok,few_shot_strict(e.source,shots)) for e in val])
    write(args.out/'copy-val.json',copy_baseline(val)['summary'])
    val_encoded=encode(val,tok,args.max_length)
    rows=[]
    started=time.perf_counter()
    class Stages(TrainerCallback):
        def on_step_end(self,training_args,state,control,**kwargs):
            if state.global_step % args.eval_every and state.global_step!=args.steps: return
            step=state.global_step
            model=kwargs['model']
            name=f'step-{step}'
            path=args.out/name
            model.save_pretrained(path)
            tok.save_pretrained(path)
            metrics=evaluate(args,model,tok,val,name+'-val')
            nll=mean_loss(model,val_encoded,tok.pad_token_id,batch_size=args.batch_size,dtype=args.dtype)
            # 同一固定 train 前缀的 NLL，避免拿不同 batch 的 loss 误判过拟合。
            train_nll=mean_loss(model,encoded[:min(128,len(encoded))],tok.pad_token_id,
                                batch_size=args.batch_size,dtype=args.dtype)
            row=dict(step=step,metrics=metrics,val_nll=nll,train_probe_nll=train_nll,
                     effective_epochs=step*args.batch_size*args.accumulation/len(encoded))
            rows.append(row); write(args.out/'stages.json',rows); emit(stage=row)
        def on_log(self,training_args,state,control,logs=None,**kwargs):
            emit(step=state.global_step,training=logs)
    trainer=make_trainer(model,tok,encoded,args.out/'trainer',device=args.device,dtype=args.dtype,
        steps=args.steps,batch_size=args.batch_size,accumulation=args.accumulation,lr=args.lr,
        seed=args.seed,rank=args.rank,checkpointing=args.checkpointing,callbacks=[Stages()])
    write(args.out/'sft-config.json',trainer.args.to_dict())
    write(args.out/'trainable-parameters.json',counts(trainer.model))
    audit=audit_batch(next(iter(trainer.get_train_dataloader())),tok,encoded)
    write(args.out/'training-first-batch.json',audit)
    emit(training_first_batch=audit,parameters=counts(trainer.model))
    if args.device=='cuda': torch.cuda.reset_peak_memory_stats()
    output=trainer.train()
    write(args.out/'trainer-history.json',trainer.state.log_history)
    result=dict(metrics=output.metrics,seconds=time.perf_counter()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved() if args.device=='cuda' else None)
    # 固定规则只看 val：raw EM_norm 最大、CER 最小、较早步数。
    winner=min(rows,key=lambda r:(-r['metrics']['raw']['em_norm']['value'],
                                 r['metrics']['raw']['cer']['value'],r['step']))
    selection=dict(step=winner['step'],rule='val raw EM_norm desc, CER asc, step asc',
                   args=vars(args),baseline=baseline,selected=winner,training=result)
    write(args.out/'selection.json',selection)
    del trainer,model; release()
    return selection


def selected(args):
    from peft import PeftModel
    selection=json.loads((args.out/'selection.json').read_text())
    saved=selection['args']
    for key in ('dtype','seed','n','max_length','max_new_tokens','eval_batch_size'):
        if getattr(args,key)!=saved[key]: raise ValueError(f'重载/最终评测需与选择配置一致：{key}')
    model,tok=load_qwen3(device=args.device,dtype=torch.float32)
    model=PeftModel.from_pretrained(model,args.out/f"step-{selection['step']}",is_trainable=False)
    return model,tok,selection


def part4(args):
    model,tok,selection=selected(args)
    val=data(args,'val',args.n)
    metrics=evaluate(args,model,tok,val,'reloaded-val')
    # 比较逐条输出，而不只比较汇总。
    before=json.loads((args.out/f"step-{selection['step']}-val-predictions.json").read_text())
    after=json.loads((args.out/'reloaded-val-predictions.json').read_text())
    assert before==after, '重载后的逐条预测不一致'
    result=dict(predictions_identical=True,step=selection['step'],metrics=metrics)
    write(args.out/'reload-verified.json',result)
    del model; release()
    return result


def part5(args):
    if not args.final_test: return dict(skipped='需要冻结配置后显式 --final-test')
    if not (args.out/'reload-verified.json').exists(): raise ValueError('先完成部分4重载验证')
    # 独占创建标记，进程失败也不自动重跑 test。
    marker=args.out/'FINAL_TEST_STARTED.json'
    if marker.exists(): raise ValueError('该目录已开始过 test；禁止反复评测')
    model,tok,selection=selected(args)
    with marker.open('x') as f: json.dump(dict(step=selection['step'],utc=datetime.now(timezone.utc).isoformat()),f)
    test=data(args,'test',args.n)
    train,val=data(args,'train',selection['args']['train_n']),data(args,'val',args.n)
    assert check_no_leak(dict(train=train,val=val,test=test))['ok']
    result=evaluate(args,model,tok,test,'final-test')
    del model; release()
    return dict(test_only_once=True,step=selection['step'],metrics=result)


def part6(args):
    """只读已完成的指标；生成 CSV 与可导出的曲线，不触发任何模型评测。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    stages=json.loads((args.out/'stages.json').read_text())
    baseline=json.loads((args.out/'baseline-val-metrics.json').read_text())
    records=[dict(method='zero-shot',split='val',step=0,n=baseline['raw']['n'],
                  em_norm=baseline['raw']['em_norm']['value'],cer=baseline['raw']['cer']['value'],
                  train_nll='',val_nll='')]
    if (args.out/'strict-3shot-val-metrics.json').exists():
        strong=json.loads((args.out/'strict-3shot-val-metrics.json').read_text())['raw']
        records.append(dict(method='strict-3shot',split='val',step=0,n=strong['n'],
            em_norm=strong['em_norm']['value'],cer=strong['cer']['value'],train_nll='',val_nll=''))
    if (args.out/'copy-val.json').exists():
        copy=json.loads((args.out/'copy-val.json').read_text())
        records.append(dict(method='copy',split='val',step=0,n=copy['n'],
            em_norm=copy['em_norm']['value'],cer=copy['cer']['value'],train_nll='',val_nll=''))
    for row in stages:
        records.append(dict(method='sft',split='val',step=row['step'],n=row['metrics']['raw']['n'],
            em_norm=row['metrics']['raw']['em_norm']['value'],cer=row['metrics']['raw']['cer']['value'],
            train_nll=row['train_probe_nll']['loss'],val_nll=row['val_nll']['loss']))
    if (args.out/'final-test-metrics.json').exists():
        test=json.loads((args.out/'final-test-metrics.json').read_text())
        selection=json.loads((args.out/'selection.json').read_text())
        records.append(dict(method='sft-selected',split='test-once',step=selection['step'],n=test['raw']['n'],
            em_norm=test['raw']['em_norm']['value'],cer=test['raw']['cer']['value'],train_nll='',val_nll=''))
    with (args.out/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    x=[r['step'] for r in stages]
    for key,label in [('train_probe_nll','fixed train probe'),('val_nll','validation')]:
        axes[0].plot(x,[r[key]['loss'] for r in stages],marker='o',label=label)
    axes[0].set(xlabel='optimizer step',ylabel='token-weighted NLL'); axes[0].legend()
    axes[1].plot(x,[r['metrics']['raw']['em_norm']['value'] for r in stages],marker='o',label='SFT val')
    axes[1].axhline(baseline['raw']['em_norm']['value'],linestyle='--',label='prompt baseline')
    if (args.out/'strict-3shot-val-metrics.json').exists():
        strong=json.loads((args.out/'strict-3shot-val-metrics.json').read_text())
        axes[1].axhline(strong['raw']['em_norm']['value'],linestyle=':',label='strict 3-shot baseline')
    axes[1].set(xlabel='optimizer step',ylabel='raw-output normalized exact match'); axes[1].legend()
    fig.suptitle('Project observation: rented RTX 4090; see environment JSON for precision/sample sizes')
    fig.tight_layout(); fig.savefig(args.out/'learning-curves.png',dpi=160); plt.close(fig)
    return dict(rows=records,artifacts=['results.csv','learning-curves.png'])


PARTS={1:('CPU机制与框架对拍',part1),2:('真实batch、参数、loss检查',part2),
       3:('LoRA SFT与val阶段对照',part3),4:('重载adapter复现val',part4),
       5:('冻结设置单次test',part5),6:('结果表与曲线',part6)}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('parts',nargs='*',type=int)
    p.add_argument('--device',choices=('cpu','mps','cuda'),default='cpu')
    p.add_argument('--dtype',choices=('fp32','bf16'),default='fp32')
    for name,default in [('n',20),('train-n',32),('steps',2),('batch-size',1),('accumulation',1),
                         ('eval-batch-size',2),('eval-every',100),('rank',4),('max-length',384),('max-new-tokens',192)]:
        p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--checkpointing',action='store_true')
    p.add_argument('--strong-baseline',action='store_true',help='额外评第18章严格格式3-shot提示基线')
    p.add_argument('--final-test',action='store_true')
    p.add_argument('--data-dir',type=Path)
    p.add_argument('--out',type=Path,default=Path('outputs/book/ch21/local'))
    args=p.parse_args(argv)
    if any(i not in PARTS for i in args.parts): p.error('未知部分号')
    if min(args.n,args.train_n,args.steps,args.batch_size,args.accumulation,args.eval_batch_size,
           args.eval_every,args.rank,args.max_length,args.max_new_tokens)<=0 or args.lr<=0: p.error('规模/lr 必须为正')
    if args.device=='cuda' and not torch.cuda.is_available(): p.error('CUDA 不可用')
    if args.device=='mps' and (not torch.backends.mps.is_available() or args.dtype!='fp32'): p.error('MPS 不可用或精度不支持')
    if args.device=='cpu' and torch.cuda.is_available(): os.environ['ACCELERATE_USE_CPU']='true'
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2); torch.manual_seed(args.seed)
    versions={key:importlib.metadata.version(key) for key in ('torch','transformers','trl','peft','accelerate','datasets')}
    paths=[Path(__file__),Path('miniocr/train/trl_sft.py'),Path('miniocr/data/sft.py'),
           Path('miniocr/train/sft.py'),Path('miniocr/data/postcorrect.py'),Path('miniocr/data/latex_corpus.py'),
           Path('miniocr/eval/postcorrect_eval.py'),Path('miniocr/prompting.py')]
    env=dict(versions=versions,args=vars(args),base_snapshot=str(find_local_qwen3()),
             code_sha256={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths},
             precision='FP32基座/adapter/Adam，'+args.dtype+'计算；loss对拍FP32',
             generation=dict(do_sample=False,enable_thinking=False,padding_side='left',
                             max_new_tokens=args.max_new_tokens,batch_size=args.eval_batch_size),
             device=torch.cuda.get_device_name() if args.device=='cuda' else args.device)
    write(args.out/('environment-'+ '-'.join(map(str,args.parts or [1]))+'.json'),env)
    emit(environment=env)
    for i in args.parts or [1]:
        start=time.perf_counter(); emit(part=i,title=PARTS[i][0],event='start')
        result=PARTS[i][1](args)
        write(args.out/f'part{i}.json',result)
        emit(part=i,event='done',seconds=time.perf_counter()-start)


if __name__=='__main__': main()
