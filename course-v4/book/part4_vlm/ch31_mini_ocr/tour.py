"""第31章：1 结构/mask，2 开源图文数据，3 从零训练，4 视觉对照/saliency，5 最终test。"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from miniocr.data.contrastive_pairs import prepare_pairs
from miniocr.data.postcorrect import PostCorrectExample
from miniocr.device import pick_device, synchronize
from miniocr.eval.postcorrect_eval import score_postcorrect
from miniocr.models.formula_ocr import FormulaOCR, targets_for, token_saliency


def write(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def amp(args):
    return torch.autocast('cuda',dtype=torch.bfloat16) if args.precision == 'bf16' else nullcontext()


def dataset(args, split):
    return prepare_pairs(args.data_dir,{split:getattr(args,split+'_n')},args.source,
                         args.max_length,allow_test=split == 'test' and args.final_test)[split]


def subset(data, n):
    return {k:v[:n] if k in ('images','ids','manifest') else v for k,v in data.items()}


def score(data, predictions):
    examples = [PostCorrectExample(r['id'],r['split'],r['formula'],r['formula']) for r in data['manifest']]
    result = score_postcorrect(examples,predictions)
    # OCR has no noisy textual source: omit correction-only rates.
    result['summary'] = {k:result['summary'][k] for k in ('n','em_raw','em_norm','cer','parse_rate')}
    return result


@torch.no_grad()
def evaluate(model, data, args, images=None):
    model.eval(); predictions = []
    x = data['images'] if images is None else images
    for start in range(0,len(x),args.batch_size):
        with amp(args):
            predictions.extend(model.generate(x[start:start+args.batch_size].to(args.device)))
    return score(data,predictions)


def part1(args):
    model = FormulaOCR().to(args.device)
    inputs,labels = targets_for(['x',r'\frac{1}{2}'],64)
    images = torch.rand(2,1,32,160,device=args.device)
    logits,loss = model(images,inputs.to(args.device),labels.to(args.device)); loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    return dict(parameters=sum(p.numel() for p in model.parameters()),visual_tokens=model.prefix_length,
                logits_shape=list(logits.shape),supervised_tokens=int(labels.ne(-100).sum()),
                loss=float(loss.detach()),all_gradients_finite=True,visual_labels_ignored=True,
                diagram='PNG → crop/letterbox 1×32×160 → ViT → MLP connector → [visual; BOS; bytes] → GPT → bytes/EOS')


def part2(args):
    datasets = prepare_pairs(args.data_dir,{'train':args.train_n,'val':args.val_n},args.source,args.max_length)
    keys = [{r['key_sha256'] for r in datasets[s]['manifest']} for s in ('train','val')]
    assert not keys[0]&keys[1]
    manifests = {s:d['manifest'] for s,d in datasets.items()}
    write(args.out/'manifest.json',manifests)
    fig,axes = plt.subplots(4,2,figsize=(10,6))
    for ax,x,r in zip(axes.flat,datasets['val']['images'],manifests['val']):
        ax.imshow(x[0],cmap='gray_r'); ax.set_title(r['formula'][:65],fontsize=7); ax.axis('off')
    fig.tight_layout(); fig.savefig(args.out/'pairs.png'); plt.close(fig)
    return dict(source=args.source,counts={s:len(d['images']) for s,d in datasets.items()},
                split='latex-v1 full corpus before length filtering' if args.source == 'open' else 'controlled rendered fixture',
                distinct_keys=True,overlap=0,max_utf8_bytes=args.max_length,shape=[1,32,160],
                selection='first eligible distinct normalized keys in fixed parquet order',
                manifest_sha256=hashlib.sha256(json.dumps(manifests,sort_keys=True).encode()).hexdigest())


def part3(args):
    args.run_dir.mkdir(parents=True,exist_ok=True)
    if (args.run_dir/'frozen.json').exists(): raise FileExistsError('use fresh run-dir')
    train,val = dataset(args,'train'),dataset(args,'val')
    assert not {r['key_sha256'] for r in train['manifest']}&{r['key_sha256'] for r in val['manifest']}
    model = FormulaOCR(args.width,args.vision_depth,args.decoder_depth,args.patch,args.max_length).to(args.device)
    inputs,labels = targets_for([r['formula'] for r in train['manifest']],args.max_length)
    # Caller-owned host tensors; only batches occupy GPU memory.
    opt = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    generator = torch.Generator().manual_seed(args.seed)
    best,rows = float('inf'),[]
    check = subset(val,min(args.n,len(val['images'])))
    baseline = evaluate(model,check,args)['summary']
    start = time.perf_counter()
    if args.device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    for step in range(1,args.steps+1):
        model.train(); opt.zero_grad(set_to_none=True)
        ix = torch.randint(len(inputs),(args.batch_size,),generator=generator)
        rate = args.lr*min(step/100,1)*(.1+.9*(1+math.cos(math.pi*step/args.steps))/2)
        for group in opt.param_groups: group['lr'] = rate
        with amp(args):
            _,loss = model(train['images'][ix].to(args.device),inputs[ix].to(args.device),labels[ix].to(args.device))
        assert torch.isfinite(loss)
        loss.backward()
        if step == 1:
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            print(json.dumps(dict(smoke_loss=float(loss.detach()),all_gradients_finite=True,
                trainable=sum(p.numel() for p in model.parameters()),supervised_tokens=int(labels[ix].ne(-100).sum()),
                padding_and_visual_ignored=True)),flush=True)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        if step == 1 or step%args.log_every == 0 or step == args.steps:
            synchronize(args.device)
            row = dict(step=step,loss=float(loss.detach()),lr=rate,grad_norm=float(norm),
                       seconds=time.perf_counter()-start)
            if step%args.eval_every == 0 or step == args.steps:
                metrics = evaluate(model,check,args)['summary']
                row.update(val=metrics)
                if metrics['cer']['value'] < best:
                    best = metrics['cer']['value']
                    torch.save(dict(config=model.config,model=model.state_dict(),step=step),args.run_dir/'best.pt')
            rows.append(row); print(json.dumps(row),flush=True); write(args.run_dir/'curves.json',rows)
    saved = torch.load(args.run_dir/'best.pt',map_location=args.device,weights_only=True)
    model.load_state_dict(saved['model'])
    validation = evaluate(model,val,args)
    write(args.run_dir/'validation.json',validation)
    frozen = dict(config=model.config,best_step=saved['step'],steps=args.steps,batch_size=args.batch_size,
                  train_n=args.train_n,val_n=args.val_n,test_n=args.test_n,max_length=args.max_length,
                  source=args.source,seed=args.seed,precision=args.precision,baseline=baseline,
                  selection=f'minimum normalized CER on fixed first {len(check["images"])} validation images; earliest tie',
                  val=validation['summary'],seconds=time.perf_counter()-start,
                  parameters=sum(p.numel() for p in model.parameters()),
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if args.device.type == 'cuda' else None,
                  checkpoint_sha256=hashlib.sha256((args.run_dir/'best.pt').read_bytes()).hexdigest())
    write(args.run_dir/'frozen.json',frozen)
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    axes[0].plot([r['step'] for r in rows],[r['loss'] for r in rows]); axes[0].set_ylabel('training token CE')
    v = [r for r in rows if 'val' in r]
    axes[1].plot([r['step'] for r in v],[r['val']['cer']['value'] for r in v]); axes[1].set_ylabel('validation normalized CER')
    for ax in axes: ax.set_xlabel('optimizer step'); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(args.run_dir/'learning_curves.png'); plt.close(fig)
    return frozen


def winner(args):
    frozen = json.loads((args.run_dir/'frozen.json').read_text())
    for key in ('source','train_n','val_n','test_n','max_length','precision'):
        if getattr(args,key) != frozen[key]: raise ValueError(f'{key} differs from frozen setup')
    if hashlib.sha256((args.run_dir/'best.pt').read_bytes()).hexdigest() != frozen['checkpoint_sha256']:
        raise ValueError('checkpoint changed after freezing')
    saved = torch.load(args.run_dir/'best.pt',map_location=args.device,weights_only=True)
    model = FormulaOCR(**saved['config']).to(args.device)
    model.load_state_dict(saved['model']); model.eval()
    return model,frozen


def part4(args):
    model,frozen = winner(args)
    data = subset(dataset(args,'val'),args.n)
    if len(data['images']) < 2: raise ValueError('image shuffle needs at least two images')
    images = data['images']; occluded = images.clone(); occluded[:,:,:,60:100] = 0
    conditions = dict(clean=images,occluded=occluded,shuffled=images.roll(1,0),blank=torch.zeros_like(images))
    results = {name:evaluate(model,data,args,x) for name,x in conditions.items()}
    write(args.out/'controls.json',results)
    examples = []
    for i,row in enumerate(results['clean']['rows']):
        examples.append(dict(id=row['id'],reference=row['target'],**{k:v['rows'][i]['pred'] for k,v in results.items()},
                             clean_correct=row['em_norm'],
                             unchanged_when_blank=row['pred']==results['blank']['rows'][i]['pred']))
    write(args.out/'error_cases.json',examples)
    inputs,labels = targets_for([data['manifest'][0]['formula']],args.max_length)
    inputs = inputs.to(args.device)
    positions = sorted(set([0,min(3,int(labels.ne(-100).sum())-1),int(labels.ne(-100).sum())-1]))
    fig,axes = plt.subplots(1,len(positions)+1,figsize=(14,3))
    axes[0].imshow(images[0,0],cmap='gray_r'); axes[0].set_title('input (val index 0)')
    maps = []
    for ax,position in zip(axes[1:],positions):
        sal = token_saliency(model,images[:1].to(args.device),inputs,position,int(labels[0,position])).cpu()
        maps.append(sal.numpy()); ax.imshow(images[0,0],cmap='gray_r')
        ax.imshow(sal,cmap='magma',alpha=.7); ax.set_title(f'target byte/EOS position {position}')
    for ax in axes: ax.axis('off')
    fig.tight_layout(); fig.savefig(args.out/'saliency.png',dpi=160); plt.close(fig)
    np.savez(args.out/'saliency.npz',maps=np.stack(maps),positions=positions)
    return dict(split='val',n=len(images),best_step=frozen['best_step'],
                metrics={k:v['summary'] for k,v in results.items()},saliency_precision='float32',
                saliency='absolute gradient of teacher-forced target token log probability; positions may be UTF-8 bytes',
                warning='occlusion/shuffle/blank are distribution shifts; effects are evidence of image dependence, not definitive per-error causes')


def part5(args):
    if not args.final_test: return dict(status='skipped; --final-test required, test images unread')
    model,frozen = winner(args)
    marker = args.run_dir/'TEST_STARTED.json'
    with marker.open('x') as f: json.dump(dict(utc=datetime.now(timezone.utc).isoformat()),f)
    data = dataset(args,'test')
    test_keys = {r['key_sha256'] for r in data['manifest']}
    for split in ('train','val'):
        assert not test_keys&{r['key_sha256'] for r in dataset(args,split)['manifest']}
    result = evaluate(model,data,args)
    # M2 协议：这里只导出 test 汇总；开发期不看 test 错例，最终报告定稿后才可解封。
    final = dict(metrics=result['summary'],best_step=frozen['best_step'],test_evaluated_once=True,
                 checkpoint_sha256=frozen['checkpoint_sha256'],test_keys_sha256=sorted(test_keys))
    write(args.run_dir/'final_test.json',final)
    return final


PARTS = {1:part1,2:part2,3:part3,4:part4,5:part5}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('parts',type=int,nargs='*',default=[1])
    p.add_argument('--device',choices=('cpu','mps','cuda'),default='cpu')
    p.add_argument('--source',choices=('open','rendered'),default='open')
    p.add_argument('--out',type=Path,default=Path('outputs/book/ch31/local'))
    p.add_argument('--run-dir',type=Path); p.add_argument('--data-dir',type=Path)
    for flag,default in [('n',20),('train-n',128),('val-n',20),('test-n',20),('steps',40),
                         ('batch-size',8),('width',32),('vision-depth',1),('decoder-depth',2),('patch',8),
                         ('max-length',64),('eval-every',20),('log-every',20),('threads',2),('seed',31)]:
        p.add_argument('--'+flag,type=int,default=default)
    p.add_argument('--lr',type=float,default=.0007)
    p.add_argument('--precision',choices=('fp32','bf16'),default='fp32')
    p.add_argument('--final-test',action='store_true')
    args = p.parse_args(argv)
    if not args.parts: args.parts = [1]
    if any(i not in PARTS for i in args.parts) or len(set(args.parts)) != len(args.parts): p.error('unique parts 1..5')
    if any(getattr(args,k)<1 for k in ('n','train_n','val_n','test_n','steps','batch_size','width',
                                      'vision_depth','decoder_depth','patch','max_length','eval_every','log_every','threads')):
        p.error('sizes must be positive')
    if args.width%8 or 32%args.patch or 160%args.patch or args.lr <= 0: p.error('invalid model dimensions/lr')
    if args.precision == 'bf16' and args.device != 'cuda': p.error('bf16 experiment requires CUDA; CPU/MPS use fp32')
    args.run_dir = args.run_dir or args.out
    args.data_dir = args.data_dir or args.out/'data'
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'started.json').exists(): p.error('use fresh --out')
    torch.set_num_threads(args.threads); args.device = pick_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if args.device.type == 'cuda': torch.cuda.set_per_process_memory_fraction(.35)
    meta = dict(start_utc=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),
                torch=torch.__version__,gpu=torch.cuda.get_device_name() if args.device.type == 'cuda' else None,
                settings={k:str(v) if isinstance(v,(Path,torch.device)) else v for k,v in vars(args).items()})
    write(args.out/'started.json',meta); print(json.dumps(meta),flush=True)
    results = {}
    for part in args.parts:
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        synchronize(args.device); start = time.perf_counter(); print(f'部分 {part}',flush=True)
        result = PARTS[part](args)
        synchronize(args.device); result['part_wall_seconds'] = time.perf_counter()-start
        write(args.out/f'part{part}.json',result); results[str(part)] = result
        print(json.dumps({k:v for k,v in result.items() if k != 'test_keys_sha256'},ensure_ascii=False),flush=True)
    meta['end_utc'] = datetime.now(timezone.utc).isoformat()
    write(args.out/'run.json',dict(metadata=meta,results=results)); print('DONE',flush=True)


if __name__ == '__main__': main()
