"""Summarize only supplementary val artifacts, including paired example uncertainty."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from miniocr.data.postcorrect import PostCorrectExample
from miniocr.eval.postcorrect_eval import score_postcorrect


def summarize(root):
    root = Path(root)
    data = json.loads((root/'data/val.json').read_text())
    assert len(data) == 500 and all(r['split'] == 'val' for r in data)
    examples = [PostCorrectExample(r['id'], 'val', r['target'], r['target']) for r in data]
    reference = score_postcorrect(examples, [r['target'] for r in data])['summary']
    (root/'val_reference_metrics.json').write_text(json.dumps({k:reference[k] for k in ('n','em_norm','cer','parse_rate')},indent=2))
    table, paired = [], []
    for seed in (35, 36):
        scores = {}
        for arm in ('original', 'distill'):
            directory = root/f'{arm}_{seed}'
            result = json.loads((directory/f'{arm}_val.json').read_text())
            train = json.loads((directory/'training.json').read_text())
            preds = json.loads((directory/f'{arm}_val_predictions.json').read_text())
            assert [r['id'] for r in preds] == [r['id'] for r in data]
            scores[arm] = score_postcorrect(examples, [r['pred'] for r in preds], bootstrap_samples=0)['rows']
            scored = scores[arm]
            reconstructed = dict(em_norm=sum(r['em_norm'] for r in scored)/500,
                parse_rate=sum(r['parses'] for r in scored)/500,
                cer=sum(r['edit_distance'] for r in scored)/sum(r['ref_length'] for r in scored))
            for metric, value in reconstructed.items():
                if not np.isclose(value, result['metrics'][metric]['value'], rtol=0, atol=1e-12):
                    raise ValueError(f'{arm}/{seed}/{metric}: scoring environment differs from evaluation')
            row = dict(arm=arm, seed=seed, n=500, steps=train['steps'], pool=train['n_pool'],
                       training_seconds=train['seconds'], eval_seconds=result['seconds'],
                       peak_allocated_gib=train['peak_allocated_gib'], truncated=result['truncated'])
            for name in ('em_norm', 'cer', 'parse_rate'):
                row[name] = result['metrics'][name]['value']
                row[name+'_low'], row[name+'_high'] = result['metrics'][name]['interval']
            table.append(row)
        rng = np.random.default_rng(35)
        ref_lengths = np.array([r['ref_length'] for r in scores['original']])
        deltas = {
            'em_norm': np.array([int(d['em_norm'])-int(o['em_norm']) for o,d in zip(scores['original'],scores['distill'])]),
            'parse_rate': np.array([int(d['parses'])-int(o['parses']) for o,d in zip(scores['original'],scores['distill'])]),
            'cer': np.array([d['edit_distance']-o['edit_distance'] for o,d in zip(scores['original'],scores['distill'])])}
        draws = {k:[] for k in deltas}
        for _ in range(2000):
            idx = rng.integers(0,500,500)
            for key, delta in deltas.items():
                draws[key].append(float(delta[idx].sum()/(ref_lengths[idx].sum() if key=='cer' else 500)))
        paired.append(dict(seed=seed, direction='distill minus original', bootstrap_samples=2000,
            bootstrap_seed=35, differences={k:float(v.sum()/(ref_lengths.sum() if k=='cer' else 500)) for k,v in deltas.items()},
            intervals={k:np.quantile(v,[.025,.975]).tolist() for k,v in draws.items()}))
    with (root/'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    result = dict(rows=table, paired=paired, uncertainty='Example bootstrap conditional on each seed; two seeds do not estimate population seed uncertainty.')
    (root/'summary.json').write_text(json.dumps(result,indent=2))
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,3,figsize=(12,4))
    for ax,metric in zip(axes,('em_norm','cer','parse_rate')):
        for i,row in enumerate(table):
            ax.errorbar(i,row[metric],yerr=[[max(0,row[metric]-row[metric+'_low'])],[max(0,row[metric+'_high']-row[metric])]],fmt='o')
        ax.set_xticks(range(4),[f"{r['arm']}\ns{r['seed']}" for r in table],fontsize=8)
        ax.set_title(metric+' (val n=500)')
    fig.tight_layout(); fig.savefig(root/'comparison.png'); plt.close(fig)
    return result

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('root',type=Path)
    summarize(p.parse_args().root)
