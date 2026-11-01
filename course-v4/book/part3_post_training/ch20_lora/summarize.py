"""从下载的 ch20 实测 JSON 生成 CSV/PNG；不加载模型、不访问 test。"""
import argparse
import csv
import json
from pathlib import Path


def summarize(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    full=root/'full'
    rows=json.loads((full/'sweep.json').read_text())
    flattened=[]
    for r in rows:
        m=r['metrics']['raw']; extracted=r['metrics']['extracted']
        flattened.append(dict(name=r['name'],rank=r['rank'],alpha=r['alpha'],
            targets=','.join(r['targets']),trainable=r['counts']['trainable'],
            val_n=m['n'],em_norm=m['em_norm']['value'],em_ci_low=m['em_norm']['interval'][0],
            em_ci_high=m['em_norm']['interval'][1],cer=m['cer']['value'],
            extracted_em_norm=extracted['em_norm']['value'],val_nll=r['val_nll']['loss'],
            peak_allocated_gib=r['peak_allocated_bytes']/2**30,
            peak_reserved_gib=r['peak_reserved_bytes']/2**30,
            training_seconds=r['training_seconds'],examples_per_second=r['examples_per_second'],
            supervised_tokens_per_second=r['supervised_tokens_per_second'],
            first_loss=r['first_loss'],last_loss=r['last_loss']))
    with (root/'sweep.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=flattened[0]); writer.writeheader(); writer.writerows(flattened)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for r in rows:
        h=json.loads((full/(r['name']+'-history.json')).read_text())
        # 固定20步均值，首尾不补点；原始每步值保留在 JSON。
        bins=[h[i:i+20] for i in range(0,len(h),20)]
        label=f"r={r['rank']} "+('q/v' if len(r['targets'])==2 else 'q/k/v/o')
        axes[0].plot([b[-1]['step'] for b in bins],[sum(t['loss'] for t in b)/len(b) for b in bins],label=label)
    for size, label, marker in [(2,'q/v','o'),(4,'q/k/v/o','s')]:
        group=[r for r in rows if len(r['targets'])==size]
        ranks=[r['rank'] for r in group]
        axes[1].plot(ranks,[r['metrics']['raw']['em_norm']['value']*100 for r in group],marker=marker,label=label)
        axes[2].plot(ranks,[r['peak_allocated_bytes']/2**30 for r in group],marker=marker,label=label)
    base=json.loads((full/'baseline-val-metrics.json').read_text())
    copy=json.loads((full/'copy-val.json').read_text())
    axes[1].axhline(base['raw']['em_norm']['value']*100,color='gray',ls='--',label='prompt baseline')
    axes[1].axhline(copy['em_norm']['value']*100,color='black',ls=':',label='copy baseline')
    axes[0].set(xlabel='Optimizer steps',ylabel='Train NLL (20-step mean)')
    axes[1].set(xlabel='Rank',ylabel='Val raw EM_norm (%)')
    axes[2].set(xlabel='Rank',ylabel='Training peak allocated (GiB)')
    for ax in axes: ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Project observation: rented RTX 4090; FP32 weights/Adam + BF16 compute')
    fig.tight_layout(); fig.savefig(root/'sweep.png',dpi=160); plt.close(fig)
    mem=json.loads((full/'part5.json').read_text())['rows']
    fig,ax=plt.subplots(figsize=(7,4))
    names=[f"B={r['batch']}, L={r['length']}" for r in mem]
    ax.bar(names,[r['peak_allocated_bytes']/2**30 for r in mem],label='Measured peak allocated')
    proxy=[sum(v for k,v in r['estimate'].items() if k.endswith('_bytes'))/2**30 for r in mem]
    ax.plot(names,proxy,'ro--',label='Component estimate (not upper bound)')
    ax.set(ylabel='GiB',title='Qwen3-0.6B, q/v rank=16, 2 steps, BF16 autocast')
    ax.legend(); fig.tight_layout(); fig.savefig(root/'memory.png',dpi=160); plt.close(fig)
    return flattened


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('outputs/book/ch20/gpu'))
    summarize(p.parse_args().root)
