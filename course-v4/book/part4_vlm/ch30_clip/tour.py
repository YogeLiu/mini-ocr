"""第 30 章：1 损失/温度，2 图文数据，3 双塔消融，4 近重复，5 最终 test。

从 course-v4 运行 python -m book.part4_vlm.ch30_clip.tour 1 --device cpu。
远程 --out/--data-dir/--run-dir 全部放在本章运行目录，路径设置见 lab.md。
"""
import argparse
import csv
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
import torch.nn.functional as F

from miniocr.data.contrastive_pairs import prepare_pairs
from miniocr.device import pick_device, synchronize
from miniocr.eval.retrieval import retrieval_metrics
from miniocr.nn.contrastive import DualEncoder, clip_loss, siglip_loss


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def part1(args):
    rows = []
    base = torch.randn(16,16, device=args.device)
    for temperature in (.03,.07,.2,1.):
        for kind in ('clip','siglip'):
            raw = base.clone().requires_grad_()
            bias = torch.tensor(-math.log(15), device=args.device, requires_grad=True)
            scores = raw/temperature
            loss = clip_loss(scores) if kind == 'clip' else siglip_loss(scores,bias)
            labels = torch.arange(16, device=args.device)
            reference = ((F.cross_entropy(scores,labels)+F.cross_entropy(scores.T,labels))/2
                         if kind == 'clip' else F.binary_cross_entropy_with_logits(
                             scores+bias, torch.eye(16, device=args.device), reduction='sum')/16)
            error = float((loss-reference).abs().detach())
            assert error < 1e-4
            loss.backward()
            rows.append(dict(loss=kind, temperature=temperature, value=float(loss.detach()),
                             reference_error=error, gradient_norm=float(raw.grad.norm()),
                             bias_gradient=None if bias.grad is None else float(bias.grad)))
    write_csv(args.out/'loss_checks.csv', rows)
    return dict(dtype='float32', batch=16, rows=rows,
                normalization='CLIP mean symmetric CE; SigLIP sum of B² pairs / B')


def data_for(args, split):
    n = {'train':args.train_n, 'val':args.val_n, 'test':args.test_n}[split]
    return prepare_pairs(args.data_dir, {split:n}, args.source, args.max_length,
                         allow_test=split == 'test' and args.final_test)[split]


def part2(args):
    datasets = prepare_pairs(args.data_dir, {'train':args.train_n,'val':args.val_n},
                             args.source,args.max_length)
    keys = [{r['key_sha256'] for r in datasets[s]['manifest']} for s in ('train','val')]
    assert not keys[0] & keys[1]
    manifest = {s:d['manifest'] for s,d in datasets.items()}
    write_json(args.out/'data_manifest.json', manifest)
    fig, axes = plt.subplots(4, 2, figsize=(10,6))
    for ax,row,x in zip(axes.flat, datasets['val']['manifest'],datasets['val']['images']):
        ax.imshow(x[0],cmap='gray_r',vmin=0,vmax=1)
        ax.set_title(row['formula'][:55],fontsize=7); ax.axis('off')
    fig.tight_layout(); fig.savefig(args.out/'pairs.png',dpi=150); plt.close(fig)
    return dict(source=args.source, counts={s:len(d['manifest']) for s,d in datasets.items()},
                split='latex-v1, full corpus before filtering' if args.source == 'open' else 'controlled formula-key hash buckets',
                max_utf8_bytes=args.max_length, image_shape=[1,32,160], shared_keys=0,
                manifest_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
                note='fixed parquet-order distinct-key prefixes; short formulas; no test images decoded')


def model_for(config, device):
    net = DualEncoder(config['width'],config['max_length'],config['temperature'],config['batch_size']).to(device)
    net.bias.requires_grad_(config['loss'] == 'siglip')
    return net


@torch.no_grad()
def embeddings(net, data, device, batch=128):
    net.eval()
    images, texts = [], []
    for start in range(0,len(data['ids']),batch):
        images.append(net.encode_image(data['images'][start:start+batch].to(device)).cpu())
        texts.append(net.encode_text(data['ids'][start:start+batch].to(device)).cpu())
    return torch.cat(images), torch.cat(texts)


def evaluate(net, data, device):
    image, text = embeddings(net,data,device)
    return retrieval_metrics(image @ text.T)


def selection_score(metrics):
    return (metrics['i2t_r1']+metrics['t2i_r1'])/2


def data_digest(data):
    """Bind a frozen experiment to its actual images, texts and ordered keys."""
    digest = hashlib.sha256(json.dumps(data['manifest'],sort_keys=True).encode())
    digest.update(data['images'].numpy().tobytes())
    digest.update(data['ids'].numpy().tobytes())
    return digest.hexdigest()


def part3(args):
    if (args.run_dir/'frozen.json').exists():
        raise FileExistsError('use a fresh --run-dir; frozen experiment exists')
    args.run_dir.mkdir(parents=True,exist_ok=True)
    train, val = data_for(args,'train'), data_for(args,'val')
    assert not {r['key_sha256'] for r in train['manifest']} & {r['key_sha256'] for r in val['manifest']}
    if max(args.batches) > len(train['ids']):
        raise ValueError('batch exceeds distinct train keys')
    x, ids = train['images'].to(args.device), train['ids'].to(args.device)
    results, curves = [], []
    for kind in args.losses:
        for batch in args.batches:
            for temperature in args.temperatures:
                torch.manual_seed(args.seed)
                config = dict(loss=kind,batch_size=batch,temperature=temperature,width=args.width,
                              max_length=args.max_length,seed=args.seed,lr=args.lr)
                tag = f'{kind}_b{batch}_t{temperature:g}'
                net = model_for(config,args.device)
                opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=args.lr,weight_decay=.001)
                generator = torch.Generator().manual_seed(args.seed)
                baseline = evaluate(net,val,args.device)
                steps = math.ceil(args.budget_examples/batch) if args.budget_examples else args.steps
                interval = max(1,math.ceil(args.eval_examples/batch)) if args.eval_examples else args.eval_every
                best, best_step = -1., 0
                checkpoint = args.run_dir/f'{tag}.pt'
                synchronize(args.device); start = time.perf_counter()
                if args.device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                first_loss = None
                for step in range(1,steps+1):
                    # Distinct keys within a batch avoid false negative copies.
                    index = torch.randperm(len(x),generator=generator)[:batch].to(args.device)
                    net.train(); opt.zero_grad(set_to_none=True)
                    scores = net(x[index],ids[index])
                    loss = clip_loss(scores) if kind == 'clip' else siglip_loss(scores,net.bias)
                    assert torch.isfinite(loss)
                    loss.backward()
                    if step == 1:
                        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                                   for p in net.parameters() if p.requires_grad)
                        first_loss = float(loss.detach())
                        print(json.dumps(dict(smoke=tag,loss=first_loss,
                            trainable=sum(p.numel() for p in net.parameters() if p.requires_grad),
                            padding_excluded=True,positive_pairs=batch,negative_pairs=batch*(batch-1))),flush=True)
                    torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
                    opt.step()
                    if step == 1 or step % interval == 0 or step == steps:
                        metrics = evaluate(net,val,args.device)
                        score = selection_score(metrics)
                        row = dict(tag=tag,step=step,examples_seen=step*batch,loss=float(loss.detach()),
                                   bias=float(net.bias.detach()),val_mean_r1=score,
                                   **{k:v for k,v in metrics.items() if '_ci95' not in k})
                        curves.append(row); print(json.dumps(row),flush=True)
                        if score > best:
                            best,best_step = score,step
                            torch.save(dict(model=net.state_dict(),config=config,step=step),checkpoint)
                        write_csv(args.run_dir/'curves.csv',curves)
                synchronize(args.device); seconds = time.perf_counter()-start
                peak = torch.cuda.max_memory_allocated()/2**20 if args.device.type == 'cuda' else None
                net.load_state_dict(torch.load(checkpoint,map_location=args.device,weights_only=True)['model'])
                metrics = evaluate(net,val,args.device)
                training = evaluate(net, {k:(v[:min(512,len(v))] if k in ('images','ids') else v)
                                          for k,v in train.items()}, args.device)
                results.append(dict(tag=tag,config=config,steps=steps,examples_seen=steps*batch,
                                    best_step=best_step,first_loss=first_loss,baseline=baseline,val=metrics,
                                    parameters=sum(p.numel() for p in net.parameters()),
                                    trainable=sum(p.numel() for p in net.parameters() if p.requires_grad),
                                    train_subset=training,seconds=seconds,peak_allocated_mib=peak,
                                    checkpoint=checkpoint.name))
                write_json(args.run_dir/'progress.json',results)
                del net,opt
    winner = max(results,key=lambda r:selection_score(r['val']))
    frozen = dict(configs=results,winner=winner['tag'],checkpoint=winner['checkpoint'],source=args.source,
                  train_n=args.train_n,val_n=args.val_n,test_n=args.test_n,max_length=args.max_length,
                  data_dir=str(args.data_dir),selection='maximum val mean bidirectional R@1, earliest tie',
                  data_sha256=dict(train=data_digest(train),val=data_digest(val)),
                  precision='float32; TF32 disabled',test_evaluated=False)
    write_json(args.run_dir/'frozen.json',frozen)
    fig,axes = plt.subplots(1,2,figsize=(12,4))
    for result in results:
        rows = [r for r in curves if r['tag'] == result['tag']]
        for ax,field in zip(axes,('loss','val_mean_r1')):
            ax.plot([r['examples_seen'] for r in rows],[r[field] for r in rows],label=result['tag'])
            ax.set_xlabel('paired training examples seen'); ax.set_ylabel(field)
    for ax in axes: ax.legend(fontsize=6); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(args.run_dir/'learning_curves.png',dpi=150); plt.close(fig)
    return frozen


def load_winner(args):
    frozen = json.loads((args.run_dir/'frozen.json').read_text())
    for key in ('source','train_n','val_n','test_n','max_length'):
        if getattr(args,key) != frozen[key]:
            raise ValueError(f'{key} differs from frozen experiment')
    if 'data_sha256' in frozen:
        for split,digest in frozen['data_sha256'].items():
            if data_digest(data_for(args,split)) != digest:
                raise ValueError('data content differs from frozen experiment')
    saved = torch.load(args.run_dir/frozen['checkpoint'],map_location=args.device,weights_only=True)
    net = model_for(saved['config'],args.device)
    net.load_state_dict(saved['model']); net.eval()
    return frozen,net


def part4(args):
    frozen,net = load_winner(args)
    val = data_for(args,'val')
    image,text = embeddings(net,val,args.device)
    # Controlled near-duplicates: brightness and one-pixel non-wrapping translation.
    n = min(args.n,len(image))
    shifted = F.pad(val['images'][:n]*.85,(1,0,0,0))[:,:,:,:160]
    with torch.no_grad(): query = net.encode_image(shifted.to(args.device)).cpu()
    near = query @ image.T
    recovered = near.argmax(1).eq(torch.arange(n))
    natural = image @ image.T
    natural.fill_diagonal_(-float('inf'))
    nearest = natural.argmax(1)
    rows = [dict(query=val['manifest'][i]['id'],neighbor=val['manifest'][int(nearest[i])]['id'],
                 cosine=float(natural[i,nearest[i]]),
                 query_formula=val['manifest'][i]['formula'],
                 neighbor_formula=val['manifest'][int(nearest[i])]['formula']) for i in range(n)]
    write_json(args.out/'near_duplicates.json',rows)
    write_json(args.out/'retrieval_examples.json',[
        dict(query=val['manifest'][i]['id'],top5=[val['manifest'][j]['id'] for j in
             (image[i]@text.T).argsort(descending=True)[:5].tolist()]) for i in range(n)])
    fig,axes = plt.subplots(min(6,n),2,figsize=(10,2*min(6,n)),squeeze=False)
    for i,(left,right) in enumerate(axes):
        j = int(nearest[i])
        for ax,index in ((left,i),(right,j)):
            ax.imshow(val['images'][index,0],cmap='gray_r',vmin=0,vmax=1)
            ax.set_title(val['manifest'][index]['formula'][:65],fontsize=7); ax.axis('off')
        right.set_xlabel(f'cosine={float(natural[i,j]):.4f}')
    fig.tight_layout(); fig.savefig(args.out/'near_duplicates.png',dpi=150); plt.close(fig)
    return dict(winner=frozen['winner'],split='val',n=n,gallery_n=len(image),
                injected_near_duplicate_r1=float(recovered.float().mean()),
                injected_correct=int(recovered.sum()),
                caveat='injected label known; natural nearest neighbors are review candidates, not proven duplicates')


def part5(args):
    if not args.final_test:
        return dict(status='skipped: final-test flag required; test not read')
    frozen,net = load_winner(args)
    marker = args.run_dir/'TEST_STARTED.json'
    if marker.exists():
        raise FileExistsError('test already started; do not reevaluate or tune on test')
    write_json(marker,dict(winner=frozen['winner'],utc=datetime.now(timezone.utc).isoformat()))
    test = data_for(args,'test')
    for split in ('train','val'):
        other = data_for(args,split)
        assert not {r['key_sha256'] for r in test['manifest']} & {r['key_sha256'] for r in other['manifest']}
    # One encoder pass on test; all recalls and query bootstrap use these embeddings.
    metrics = evaluate(net,test,args.device)
    result = dict(winner=frozen['winner'],metrics=metrics,test_evaluated_once=True,
                  ci='1000 query bootstrap resamples; fixed gallery; single training seed',
                  test_key_sha256=[r['key_sha256'] for r in test['manifest']])
    write_json(args.run_dir/'final_test.json',result)
    return result


PARTS = {1:part1,2:part2,3:part3,4:part4,5:part5}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('parts',nargs='*',type=int)
    p.add_argument('--device',choices=('cpu','mps','cuda'),default='cpu')
    p.add_argument('--source',choices=('rendered','open'),default='rendered')
    p.add_argument('--out',type=Path,default=Path('outputs/book/ch30/local'))
    p.add_argument('--data-dir',type=Path)
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--n',type=int,default=20,help='val inference/near-duplicate query count')
    p.add_argument('--train-n',type=int,default=128)
    p.add_argument('--val-n',type=int,default=32)
    p.add_argument('--test-n',type=int,default=32)
    p.add_argument('--steps',type=int,default=80)
    p.add_argument('--budget-examples',type=int,default=0,help='if positive, ceil(examples/B) steps per config')
    p.add_argument('--batches',type=int,nargs='+',default=[16])
    p.add_argument('--temperatures',type=float,nargs='+',default=[.07])
    p.add_argument('--losses',nargs='+',choices=('clip','siglip'),default=['clip','siglip'])
    p.add_argument('--width',type=int,default=32)
    p.add_argument('--max-length',type=int,default=96)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--eval-every',type=int,default=40)
    p.add_argument('--eval-examples',type=int,default=0,help='positive: evaluate after this many paired examples')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=30)
    p.add_argument('--final-test',action='store_true')
    args = p.parse_args(argv)
    parts = args.parts or list(PARTS)
    if len(set(parts)) != len(parts) or any(i not in PARTS for i in parts):
        p.error('parts must be unique integers 1..5')
    if min(args.n,args.train_n,args.val_n,args.test_n,args.steps,args.width,args.max_length,
           args.eval_every,args.threads,*args.batches) < 1 or min(args.batches) < 2:
        p.error('positive sizes required and batch >= 2')
    if args.width%4 or min(args.temperatures) <= 0 or args.lr <= 0 or min(args.budget_examples,args.eval_examples) < 0:
        p.error('width must be divisible by 4; invalid temperature/lr/budget')
    if args.val_n < 2:
        p.error('val-n must be >= 2 for nearest-neighbor demonstration')
    if any(len(v) != len(set(v)) for v in (args.losses,args.batches,args.temperatures)):
        p.error('duplicate configuration')
    args.data_dir = args.data_dir or args.out/'data'
    args.run_dir = args.run_dir or args.out
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'started.json').exists():
        p.error('use a fresh --out to preserve run records')
    torch.set_num_threads(args.threads)
    args.device = pick_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.device.type == 'cuda': torch.cuda.set_per_process_memory_fraction(.2)
    meta = dict(start_utc=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),
                torch=torch.__version__,gpu=torch.cuda.get_device_name() if args.device.type == 'cuda' else None,
                precision='float32; TF32 disabled',
                settings={k:str(v) if isinstance(v,(Path,torch.device)) else v for k,v in vars(args).items()})
    write_json(args.out/'started.json',meta); print(json.dumps(meta,ensure_ascii=False),flush=True)
    results = {}
    for part in parts:
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        synchronize(args.device); start = time.perf_counter()
        print(f'部分 {part}',flush=True)
        result = PARTS[part](args)
        synchronize(args.device); result['part_wall_seconds'] = time.perf_counter()-start
        results[str(part)] = result
        write_json(args.out/f'part{part}.json',result)
        print(json.dumps({k:v for k,v in result.items() if k != 'test_key_sha256'},ensure_ascii=False),flush=True)
    meta['end_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(args.out/'run.json',dict(metadata=meta,results=results))
    print(f'DONE out={args.out}',flush=True)


if __name__ == '__main__':
    main()
