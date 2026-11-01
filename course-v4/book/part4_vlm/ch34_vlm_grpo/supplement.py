"""汇总补充 val-only 对照，不读取 test、不写 selection。"""
import argparse
import csv
from datetime import datetime, timedelta
import json
from pathlib import Path

import numpy as np
from miniocr.data.postcorrect import PostCorrectExample
from miniocr.eval.postcorrect_eval import score_postcorrect


def scored(result):
    obs = result['observations']
    examples = [PostCorrectExample(r['id'], 'val', r['target'], r['target']) for r in obs]
    return score_postcorrect(examples, [r['pred'] for r in obs], bootstrap_samples=0)['rows']


def paired_delta(base, arm):
    """固定 seed=0，按相同样本索引 paired bootstrap，报告 arm−base。"""
    assert [r['id'] for r in base] == [r['id'] for r in arm]
    n = len(base); rng = np.random.default_rng(0)
    b = np.array([[r['em_norm'], r['edit_distance'], r['ref_length'], r['parses']] for r in base], float)
    a = np.array([[r['em_norm'], r['edit_distance'], r['ref_length'], r['parses']] for r in arm], float)
    def metrics(x):
        return np.array([x[:, 0].mean(), x[:, 1].sum()/x[:, 2].sum(), x[:, 3].mean()])
    point = metrics(a)-metrics(b)
    draws = []
    for _ in range(2000):
        idx = rng.integers(n, size=n); draws.append(metrics(a[idx])-metrics(b[idx]))
    intervals = np.quantile(draws, [.025, .975], axis=0).T
    return {k: dict(value=float(v), interval=ci.tolist()) for k, v, ci in
            zip(('em_norm', 'cer', 'parse_rate'), point, intervals)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args(argv); root = args.root
    names = ('start', 'layered', 'no_parse_bonus', 'layered150')
    results = {n: json.loads((root/f'val_{n}/val_result.json').read_text()) for n in names}
    assert all(r['split'] == 'val' and r['metrics']['n'] == 500 for r in results.values())
    assert len({r['sample_ids_sha256'] for r in results.values()}) == 1
    rows = {n: scored(r) for n, r in results.items()}
    comparison = {n: dict(metrics=r['metrics'], truncated=r['truncated'],
        mean_tokens=float(np.mean([o['generated_tokens'] for o in r['observations']]))) for n, r in results.items()}
    deltas = {n: paired_delta(rows['start'], rows[n]) for n in names[1:]}
    deltas['layered600_minus_150'] = paired_delta(rows['layered150'], rows['layered'])
    deltas['no_parse_bonus_minus_layered'] = paired_delta(rows['layered'], rows['no_parse_bonus'])
    curves = []; training = {}; changes = {}
    for n in ('layered', 'no_parse_bonus'):
        log = [json.loads(line) for line in (root/n/'train_log.jsonl').read_text().splitlines()]
        assert len(log) == 600
        t = json.loads((root/n/'training.json').read_text())
        def mean(key, seq):
            vals = [r[key] for r in seq if r.get(key) is not None]
            return float(np.mean(vals)) if vals else None
        fields = ('reward_mean', 'kl', 'mean_tokens', 'informative', 'degenerate', 'advantage_abs_mean')
        training[n] = dict(seconds=t['seconds'], peak_allocated_gib=t['peak_allocated_gib'],
            peak_reserved_gib=t['peak_reserved_gib'], updated=sum(r['updated'] for r in log),
            mean={k: mean(k, log) for k in fields},
            first50={k: mean(k, log[:50]) for k in fields}, last50={k: mean(k, log[-50:]) for k in fields},
            probe=t['probe_after'])
        curves.extend(dict(arm=n, step=r['step'], **{k: r.get(k) for k in fields}) for r in log)
        changes[n] = [dict(id=b['id'], target=b['target'], before=b['pred'], after=a['pred'],
            before_distance=b['edit_distance'], after_distance=a['edit_distance'],
            before_parse=b['parses'], after_parse=a['parses'],
            parse_gained_cer_worse=not b['parses'] and a['parses'] and a['edit_distance']>b['edit_distance'])
            for b, a in zip(rows['start'], rows[n]) if b['pred'] != a['pred']]
        comparison[n]['parse_gained_cer_worse_count'] = sum(r['parse_gained_cer_worse'] for r in changes[n])
    runtime = []
    for path in sorted(root.glob('*/timing.json')):
        t = json.loads(path.read_text()); end = datetime.fromisoformat(t['end_utc'])
        runtime.append(dict(run=path.parent.name, seconds=t['seconds'], end_utc=t['end_utc'],
                            start_utc=(end-timedelta(seconds=t['seconds'])).isoformat()))
    report = dict(comparison=comparison, paired_delta=deltas, training=training,
                  runtime=runtime, total_seconds=sum(t['seconds'] for t in runtime))
    (root/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (root/'val_changed_samples.json').write_text(json.dumps(changes, ensure_ascii=False, indent=2))
    with (root/'training_curves.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(curves[0])); writer.writeheader(); writer.writerows(curves)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for n in ('layered', 'no_parse_bonus'):
        log = [r for r in curves if r['arm'] == n]
        for ax, key in zip(axes, ('reward_mean', 'kl', 'mean_tokens')):
            values = np.array([r[key] if r[key] is not None else np.nan for r in log])
            smooth = [np.nanmean(values[max(0,i-19):i+1]) if np.isfinite(values[max(0,i-19):i+1]).any() else np.nan for i in range(len(values))]
            ax.plot(range(1, len(values)+1), smooth, label=n); ax.set_ylabel(key); ax.grid(alpha=.2)
    axes[0].legend(); axes[-1].set_xlabel('step (20-step moving mean; KL excludes non-updates)')
    fig.tight_layout(); fig.savefig(root/'training_curves.png', dpi=150); plt.close(fig)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
