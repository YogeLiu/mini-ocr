"""租卡串行实验编排；screen/main/final 分阶段，所有调参依据 val。

从 course-v4 运行 python -m book.part4_vlm.ch33_vlm_lora.gpu_suite screen。
本文件不下载、不安装、不修改系统。后台启动方式见 lab.md。
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

WORK_ROOT = os.environ.get('COURSE_WORK_ROOT', '/hy-tmp')

ROOT = Path(f'{WORK_ROOT}/runs/ch33')
LOGS = Path(f'{WORK_ROOT}/logs')
CACHE = f'{WORK_ROOT}/runs/ch32/hf/hub'


def run(name, *extra):
    """独立进程释放显存，失败立即终止阶段；保留命令和起止时间。"""
    if (ROOT/name/'timing.json').exists():
        old = json.loads((ROOT/name/'config.json').read_text())
        requested = list(map(str,extra)); i = 0; parts = []
        while i < len(requested):
            item = requested[i]
            if not item.startswith('--'): parts.append(int(item)); i += 1; continue
            key = item[2:].replace('-','_')
            value = True if key=='final_test' else requested[i+1]
            if str(old.get(key)) != str(value): raise ValueError(f'{name} 已完成且参数 {key} 不同；请使用新运行名')
            i += 1 if key=='final_test' else 2
        if old['parts'] != parts: raise ValueError(f'{name} 部分号不同')
        print(json.dumps(dict(name=name,resume='already completed; configuration matched')),flush=True); return
    required = 9000 if 'qwen3vl' in map(str,extra) or '--batch-size' in map(str,extra) else 5000
    while True:
        free = int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free >= required: break
        print(json.dumps(dict(waiting_for_memory=name,free_mib=free,required_mib=required,time=datetime.now(timezone.utc).isoformat())),flush=True)
        time.sleep(30)
    command = [sys.executable,'-u','-m','book.part4_vlm.ch33_vlm_lora.tour',*map(str,extra),
        '--device','cuda','--dtype','bf16','--model-cache',CACHE,'--data',str(ROOT/'data'),
        '--out',str(ROOT/name)]
    began = datetime.now(timezone.utc).isoformat(); start = time.perf_counter()
    print(json.dumps(dict(name=name,start=began,free_mib=free,command=command)),flush=True)
    with (LOGS/f'ch33_{name}.log').open('w') as stream:
        result = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    event = dict(name=name,start_utc=began,end_utc=datetime.now(timezone.utc).isoformat(),
                 seconds=time.perf_counter()-start,returncode=result.returncode,command=command)
    with (ROOT/'events.jsonl').open('a') as stream: stream.write(json.dumps(event)+'\n')
    print(json.dumps(event),flush=True)
    if result.returncode: raise RuntimeError(f'{name} failed; see {LOGS}/ch33_{name}.log')


def ranking(path):
    """EM_norm 最大，打平时 CER 最小；只接收 val 结果。"""
    result = json.loads(path.read_text())
    if result['split'] != 'val': raise ValueError('禁止基于 test 选择')
    scores = result['metrics']
    return scores['em_norm']['value'], -scores['cer']['value']


def screen():
    """三冻结策略同 128 样本/20 步；两个模型基线和预处理对照。"""
    run('qwen35_language_screen',5,'--n',40,'--adapter',ROOT/'qwen35_smoke/adapter')
    for strategy in ('connector','vision'):
        run(f'qwen35_{strategy}_smoke',3,4,'--strategy',strategy,'--rank',16,'--train-n',128,'--steps',20,'--accum',4)
        run(f'qwen35_{strategy}_screen',5,'--n',40,'--adapter',ROOT/f'qwen35_{strategy}_smoke/adapter')
    for model in ('qwen35','qwen3vl'):
        run(f'{model}_baseline',5,'--model',model,'--n',100)
    run('qwen35_baseline_highpixels',5,'--n',40,'--min-pixels',65536,'--max-pixels',262144)
    run('qwen3vl_smoke',3,4,'--model','qwen3vl','--rank',16,'--train-n',128,'--steps',20,'--accum',4)
    scores = {s: ranking(ROOT/f'qwen35_{s}_screen/val_finetuned.json') for s in ('language','connector','vision')}
    choice = max(scores,key=scores.get)
    (ROOT/'screen_selection.json').write_text(json.dumps(dict(strategy=choice,scores=scores,rule='val EM_norm max then CER min'),indent=2))


def main_runs(steps):
    """用 screening 选定策略从原基座重新训练；正式评估同一 100 条 val。"""
    choice = json.loads((ROOT/'screen_selection.json').read_text())['strategy']
    # 早期冒烟仅记录反向前后峰值；补测覆盖重载阶段，并验证 baseline 恢复。
    for strategy in ('language','connector'):
        run(f'qwen35_{strategy}_roundtrip',3,'--strategy',strategy,'--rank',16)
    for model in ('qwen35','qwen3vl'):
        strategy = choice if model == 'qwen35' else 'language'
        smoke_name = f'{model}_batch_smoke'
        run(smoke_name,4,'--model',model,'--strategy',strategy,'--rank',16,'--train-n',2048,
            '--steps',20,'--batch-size',4,'--accum',2)
        smoke = json.loads((ROOT/smoke_name/'training.json').read_text())
        if smoke['peak_allocated_gib'] > 20: raise RuntimeError('放大 batch 的峰值超过 20 GiB，需要减小 batch')
        estimate = smoke['metrics']['train_runtime']/20*steps
        print(json.dumps(dict(model=model,smoke_seconds=smoke['metrics']['train_runtime'],steps=steps,estimate_seconds=estimate)),flush=True)
        if estimate > 3600: raise RuntimeError('估算单次超过一小时，请根据已完成冒烟显式减小 --steps')
        run(f'{model}_main',4,'--model',model,'--strategy',strategy,'--rank',16,'--train-n',2048,
            '--steps',steps,'--batch-size',4,'--accum',2)
        run(f'{model}_main_val',5,'--model',model,'--n',100,'--adapter',ROOT/f'{model}_main/adapter')
    scores = {m: ranking(ROOT/f'{m}_main_val/val_finetuned.json') for m in ('qwen35','qwen3vl')}
    choice = max(scores,key=scores.get)
    (ROOT/'final_selection.json').write_text(json.dumps(dict(model=choice,adapter=str(ROOT/f'{choice}_main/adapter'),
        scores=scores,rule='val EM_norm max then CER min; freeze before test',n=100,max_new_tokens=256,
        min_pixels=16384,max_pixels=65536,rank=16,steps=steps,seed=0),indent=2))


def final():
    """先完成 val 视觉依赖，再锁定 test，基线和微调各评一次。"""
    choice = json.loads((ROOT/'final_selection.json').read_text())
    run('visual_dependency',6,'--model',choice['model'],'--adapter',choice['adapter'],'--n',100)
    run('final_test',5,'--model',choice['model'],'--adapter',choice['adapter'],'--n',100,'--split','test','--final-test')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('stage',choices=('screen','main','final'))
    p.add_argument('--steps',type=int,default=300); args=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True); LOGS.mkdir(parents=True,exist_ok=True)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HOME=str(ROOT/'hf'),HF_DATASETS_CACHE=str(ROOT/'datasets'),MPLCONFIGDIR=str(ROOT/'mpl'))
    if args.stage=='screen': screen()
    elif args.stage=='main': main_runs(args.steps)
    else: final()


if __name__=='__main__': main()
