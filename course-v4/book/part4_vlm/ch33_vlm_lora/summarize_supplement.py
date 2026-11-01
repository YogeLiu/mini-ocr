"""Summarize measured val-only results with paired bootstrap differences."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from miniocr.data.postcorrect import PostCorrectExample
from miniocr.eval.postcorrect_eval import score_postcorrect


def scored(result):
    assert result['split'] == 'val'
    obs = result['observations']
    examples = [PostCorrectExample(r['id'],'val',r['target'],r['target']) for r in obs]
    return score_postcorrect(examples,[r['pred'] for r in obs],bootstrap_samples=0)['rows']


def paired(reference, candidate, repeats=10000):
    a,b = scored(reference),scored(candidate)
    assert [(r['id'],r['target']) for r in a] == [(r['id'],r['target']) for r in b]
    n=len(a); rng=np.random.default_rng(33)
    ae=np.array([r['em_norm'] for r in a],dtype=float); be=np.array([r['em_norm'] for r in b],dtype=float)
    ap=np.array([r['parses'] for r in a],dtype=float); bp=np.array([r['parses'] for r in b],dtype=float)
    ad=np.array([r['edit_distance'] for r in a]); bd=np.array([r['edit_distance'] for r in b])
    lengths=np.array([r['ref_length'] for r in a])
    samples=[]
    for _ in range(repeats):
        idx=rng.integers(0,n,n)
        samples.append([(be[idx]-ae[idx]).mean(),(bd[idx]-ad[idx]).sum()/lengths[idx].sum(),(bp[idx]-ap[idx]).mean()])
    intervals=np.quantile(samples,[.025,.975],axis=0)
    values=[(be-ae).mean(),(bd-ad).sum()/lengths.sum(),(bp-ap).mean()]
    return dict(n=n,repeats=repeats,seed=33,method='paired sample bootstrap, candidate minus reference, percentile 95%',
        metrics={key:dict(value=float(values[i]),interval=intervals[:,i].tolist()) for i,key in enumerate(['em_norm','cer','parse_rate'])},
        em_gained=int(((be==1)&(ae==0)).sum()),em_lost=int(((be==0)&(ae==1)).sum()))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); args=p.parse_args()
    root=args.root; out=root/'summary'; out.mkdir(exist_ok=True)
    results={}
    for name in ['baseline500','r32_val500','old_r16_val500']:
        files=list((root/name).glob('val_*.json'))
        if files: results[name]=json.loads(files[0].read_text())
    compact={name:dict(metrics=r['metrics'],seconds=r['seconds'],truncated=r['truncated'],
        sample_ids_sha256=r['sample_ids_sha256']) for name,r in results.items()}
    differences={}
    if 'baseline500' in results:
        differences={name:paired(results['baseline500'],r) for name,r in results.items() if name!='baseline500'}
    if 'old_r16_val500' in results and 'r32_val500' in results:
        differences['r32_minus_old_r16']=paired(results['old_r16_val500'],results['r32_val500'])
    expanded = {}
    for name,r in results.items():
        observations = r['observations'][100:]
        examples = [PostCorrectExample(o['id'],'val',o['target'],o['target']) for o in observations]
        summary = score_postcorrect(examples,[o['pred'] for o in observations])['summary']
        expanded[name] = {key:summary[key] for key in ['n','em_norm','cer','parse_rate']}
    expanded_paired = {}
    if 'baseline500' in results:
        reference = {**results['baseline500'],'observations':results['baseline500']['observations'][100:]}
        expanded_paired = {name:paired(reference,{**r,'observations':r['observations'][100:]})
            for name,r in results.items() if name!='baseline500'}
    payload = dict(results=compact,paired=differences,expanded_val400=expanded,expanded_val400_paired=expanded_paired)
    (out/'metrics.json').write_text(json.dumps(payload,indent=2))
    with (out/'metrics.csv').open('w') as f:
        w=csv.writer(f); w.writerow(['run','metric','value','lower95','upper95'])
        for name,r in results.items():
            for metric in ['em_norm','cer','parse_rate']:
                v=r['metrics'][metric]; w.writerow([name,metric,v['value'],*v['interval']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,1,figsize=(9,7),sharex=True)
    for path in sorted(root.glob('*/training.json')):
        train=json.loads(path.read_text()); history=[r for r in train['history'] if 'loss' in r]
        with (out/f'{path.parent.name}_curve.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=['step','epoch','loss','learning_rate','grad_norm','mean_token_accuracy'],extrasaction='ignore')
            w.writeheader(); w.writerows(history)
        for axis,key in zip(ax,['loss','learning_rate']):
            axis.plot([r['step'] for r in history],[r[key] for r in history],label=path.parent.name)
            axis.set_ylabel(key); axis.grid(alpha=.3)
    ax[0].legend(); ax[1].set_xlabel('Optimizer step'); fig.tight_layout(); fig.savefig(out/'training_curves.png',dpi=160)
    print(json.dumps(payload,indent=2))


if __name__=='__main__': main()
