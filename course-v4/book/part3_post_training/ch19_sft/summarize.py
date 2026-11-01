"""从既有 JSON 生成 CSV 和图；不加载模型、不读 test 错例、不修改实验结果。"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path


def summarize(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = []
    names = [('copy', root/'baselines/baseline-copy.json')]
    names += [(m, root/f'baselines/baseline-{m}.json') for m in ('zero_shot', 'few_shot', 'few_shot_strict')]
    names += [('SFT val', root/'final/final-val.json'), ('SFT test (once)', root/'final/final-test.json')]
    names += [(f'fault {m}', root/f'fault_{m}/val-step80.json') for m in ('correct', 'shift', 'prompt', 'no_eos')]
    for name, path in names:
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        scored = {'raw': data} if name == 'copy' else {k: data[k] for k in ('raw', 'extracted')}
        for scoring, summary in scored.items():
            for metric in ('em_norm', 'cer', 'fix_rate', 'broken_rate', 'parse_rate'):
                v = summary[metric]
                rows.append(dict(name=name, scoring=scoring, split=data.get('split', 'val'), n=summary['n'],
                                 metric=metric, value=v['value'], successes=v.get('successes'), count=v.get('count'),
                                 ci_low=v['interval'][0], ci_high=v['interval'][1], source=str(path.relative_to(root))))
    with (root/'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    curves = [json.loads(line) for line in (root/'main/history.jsonl').read_text().splitlines()]
    initial = json.loads((root/'main/train_config.json').read_text())['initial_val_nll']['loss']
    nll = [(0, initial)]
    for path in sorted((root/'main').glob('nll-step*.json')):
        nll.append((int(path.stem.split('step')[1]), json.loads(path.read_text())['loss']))
    nll.sort()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot([r['step'] for r in curves], [r['loss'] for r in curves], alpha=.65, label='train token-weighted CE')
    axes[0].plot(*zip(*nll), marker='o', label='val correct-target NLL')
    axes[0].set(xlabel='optimizer step', ylabel='nats / supervised token', title='Qwen3 full SFT, RTX 4090, mixed bf16')
    axes[0].legend()
    points = []
    for path in (root/'main').glob('val-step*.json'):
        points.append((int(path.stem.split('step')[1]), json.loads(path.read_text())['raw']['em_norm']['value']))
    points.sort()
    axes[1].plot(*zip(*points), marker='o')
    axes[1].set(xlabel='optimizer step', ylabel='raw EM_norm', title='Selection val prefix n=128', ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(root/'train_curve.png', dpi=160)
    plt.close(fig)
    compare = [r for r in rows if r['scoring'] == 'raw' and r['metric'] == 'em_norm' and r['name'] in
               ('copy', 'zero_shot', 'few_shot', 'few_shot_strict', 'SFT val')]
    fig, ax = plt.subplots(figsize=(8, 4))
    values = [r['value'] for r in compare]
    ax.bar([r['name'] for r in compare], values, yerr=[[v-r['ci_low'] for v, r in zip(values, compare)],
                                                      [r['ci_high']-v for v, r in zip(values, compare)]], capsize=4)
    ax.set(ylabel='raw EM_norm, Wilson 95% CI', ylim=(0, 1), title='Same val prefix n=500; RTX 4090, mixed bf16')
    fig.tight_layout()
    fig.savefig(root/'val_comparison.png', dpi=160)
    plt.close(fig)
    faults = []
    for variant in ('correct', 'shift', 'prompt', 'no_eos'):
        folder = root/f'fault_{variant}'
        if folder.exists():
            loss = json.loads((folder/'nll-step80.json').read_text())['loss']
            scores = json.loads((folder/'val-step80.json').read_text())
            faults.append(dict(variant=variant, correct_target_nll=loss,
                               em_norm=scores['raw']['em_norm']['value'], length_capped=scores['length_capped'],
                               mean_generated_tokens=scores['mean_generated_tokens']))
    write = lambda path, value: path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    diagnostics = {}
    for path in [*sorted((root/'baselines').glob('*.jsonl')), root/'final/final-val.jsonl']:
        if path.exists():
            records = [json.loads(line) for line in path.read_text().splitlines()]
            diagnostics[path.stem] = dict(n=len(records),
                format_issues=Counter(k for r in records for k in r['score']['format_issues']),
                targets_with_final_punct=sum(r['example']['target'].rstrip().endswith(('.', ',', ';')) for r in records))
    write(root/'val_format_diagnostics.json', diagnostics)
    write(root/'fault_summary.json', faults)
    if faults:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, metric in zip(axes, ('correct_target_nll', 'em_norm', 'length_capped')):
            ax.bar([r['variant'] for r in faults], [r[metric] for r in faults])
            ax.set(title=metric, xlabel='80 updates, train prefix 1024, val 80')
            ax.tick_params(axis='x', rotation=20)
        fig.tight_layout()
        fig.savefig(root/'faults.png', dpi=160)
        plt.close(fig)
    times = []
    for path in sorted(root.glob('*/train_summary.json')):
        data = json.loads(path.read_text())
        times.append(dict(run=path.parent.name, train_seconds=data['train_seconds'], total_seconds=data['total_seconds'],
                          peak_allocated_GiB=data['peak_allocated_bytes']/2**30 if data['peak_allocated_bytes'] else None,
                          selected_step=data['selected_step']))
    write(root/'run_times.json', times)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/book/ch19/gpu'))
    summarize(parser.parse_args().root)
