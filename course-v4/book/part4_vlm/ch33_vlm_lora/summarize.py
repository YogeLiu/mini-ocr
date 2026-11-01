"""从已完成的 JSON 生成实测 CSV/图；不产生或补猜任何训练结果。"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


WORK_ROOT = os.environ.get('COURSE_WORK_ROOT', '/hy-tmp')

def summarize(root, out):
    """汇总独立进程的墙钟与峰值；图中数值只来自运行文件。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root, out = Path(root), Path(out); out.mkdir(parents=True,exist_ok=True)
    evaluations = []; trainings = []; timings = []
    for path in sorted(root.glob('*/*.json')):
        if path.parent.name in ('data','prepare','audit'): continue
        value = json.loads(path.read_text())
        if path.name == 'timing.json': timings.append(dict(run=path.parent.name,**value)); continue
        if path.name == 'training.json':
            cfg = value['config']; trainings.append(dict(run=path.parent.name,model=cfg['model'],strategy=cfg['strategy'],
                n=cfg['train_n'],steps=cfg['steps'],rank=cfg['rank'],batch_size=cfg['batch_size'],accum=cfg['accum'],
                loss=value['metrics']['train_loss'],seconds=value['seconds'],peak_gib=value['peak_allocated_gib'],
                trainable=value['trainable'],path=str(path),history=value['history'])); continue
        if not isinstance(value,dict) or 'metrics' not in value or 'split' not in value: continue
        metrics = value['metrics']; evaluations.append(dict(run=path.parent.name,file=path.name,
            model=value['model'],split=value['split'],condition=value['condition'],n=metrics['n'],
            em_raw=metrics['em_raw']['value'],em_norm=metrics['em_norm']['value'],cer=metrics['cer']['value'],
            parse_rate=metrics['parse_rate']['value'],seconds=value['seconds'],peak_gib=value['peak_allocated_gib'],
            truncated=value['truncated'],changed=value.get('changed_predictions'),path=str(path)))
    for name,rows in [('evaluation',evaluations),('training',[{k:v for k,v in r.items() if k!='history'} for r in trainings])]:
        if rows:
            with (out/f'{name}.csv').open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if trainings:
        fig,ax=plt.subplots(figsize=(8,4))
        for row in trainings:
            history=[r for r in row['history'] if 'loss' in r]
            ax.plot([r['step'] for r in history],[r['loss'] for r in history],label=row['run'])
        ax.set(xlabel='Optimizer step',ylabel='Logged training loss'); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out/'training_loss.png',dpi=160); plt.close(fig)
    selected=[r for r in evaluations if r['run'] in ('qwen35_baseline','qwen3vl_baseline','qwen35_main_val','qwen3vl_main_val','final_test')]
    if selected:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        labels=[r['run']+'\n'+r['file'].replace('.json','') for r in selected]
        for ax,key in zip(axes,['em_norm','cer']):
            ax.bar(range(len(selected)),[r[key] for r in selected]); ax.set_xticks(range(len(selected)),labels,rotation=60,ha='right',fontsize=7)
            ax.set_ylabel(key)
        fig.tight_layout(); fig.savefig(out/'comparison.png',dpi=160); plt.close(fig)
    dependency=[r for r in evaluations if r['run']=='visual_dependency']
    if dependency:
        fig,axes=plt.subplots(1,2,figsize=(8,3))
        for ax,key in zip(axes,['em_norm','cer']):
            ax.bar([r['condition'] for r in dependency],[r[key] for r in dependency]); ax.set_ylabel(key)
        fig.tight_layout(); fig.savefig(out/'visual_dependency.png',dpi=160); plt.close(fig)
    low_path=root/'qwen35_baseline/val_baseline.json'
    high_path=root/'qwen35_baseline_highpixels/val_baseline.json'
    if low_path.exists() and high_path.exists():
        from miniocr.data.postcorrect import PostCorrectExample
        from miniocr.eval.postcorrect_eval import score_postcorrect
        low=json.loads(low_path.read_text()); high=json.loads(high_path.read_text())
        observations=low['observations'][:high['metrics']['n']]
        assert [r['id'] for r in observations] == [r['id'] for r in high['observations']]
        examples=[PostCorrectExample(r['id'],'val',r['target'],r['target']) for r in observations]
        scores=score_postcorrect(examples,[r['pred'] for r in observations])['summary']
        comparison=dict(split='val',n=len(observations),selection='same first 40 val IDs',
            low_pixels={k:scores[k] for k in ('n','em_raw','em_norm','cer','parse_rate')},
            high_pixels=high['metrics'],low_source=str(low_path),high_source=str(high_path))
        (out/'preprocessing_comparison.json').write_text(json.dumps(comparison,indent=2))
    result=dict(evaluations=evaluations,trainings=trainings,timings=timings,
        gpu_task_wall_seconds=sum(r['seconds'] for r in timings),
        accounting='sum successful GPU tour wall times including CPU/load/eval; excludes prepare, idle, startup outside tour; shared GPU, not kernel-only')
    (out/'summary.json').write_text(json.dumps(result,indent=2)); return result


def export_artifacts(root, target, logs):
    """打包指标/日志/图，不复制数据图片、数据清单或模型权重。"""
    import tarfile
    root, target, logs = Path(root), Path(target), Path(logs)
    paths = list(root.glob('*.json')) + list(root.glob('*.jsonl'))
    excluded = {'data','hf','datasets','mpl'}
    for folder in root.iterdir():
        if folder.is_dir() and folder.name not in excluded:
            for suffix in ('*.json','*.png','*.csv'):
                paths.extend(folder.glob(suffix))
    paths += [p for p in (root/'data/metadata.json',root/'data/ch33_final_test.lock') if p.exists()]
    with tarfile.open(target,'w:gz') as archive:
        for path in sorted(set(paths)):
            archive.add(path,arcname='runs/'+str(path.relative_to(root)))
        for path in sorted(logs.glob('ch33_*.log')):
            archive.add(path,arcname='logs/'+path.name)
    return target


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--export',type=Path); p.add_argument('--logs',type=Path,default=Path(f'{WORK_ROOT}/logs'))
    a=p.parse_args();result=summarize(a.root,a.out)
    print(json.dumps({k:v for k,v in result.items() if k not in ('evaluations','trainings','timings')},indent=2))
    files=[Path(__file__).parent/f for f in ('tour.py','gpu_suite.py','summarize.py','test_ch33_mechanisms.py')]
    files += [Path('miniocr/data/vlm_sft.py'),Path('miniocr/train/vlm_lora.py')]
    (a.out/'code_sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
    if a.export: print(export_artifacts(a.root,a.export,a.logs))


if __name__=='__main__': main()
