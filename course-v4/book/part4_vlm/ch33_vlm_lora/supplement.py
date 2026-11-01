"""Independent val-only supplement; never invokes the frozen test protocol.

Run prepare-smoke first, then main after inspecting timing. All GPU subprocesses
share a 9000-second wall-clock budget and a 19 GiB allocator limit.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

WORK_ROOT = os.environ.get('COURSE_WORK_ROOT', '/hy-tmp')

ROOT = Path(f'{WORK_ROOT}/runs/ch33/supp')
LOGS = Path(f'{WORK_ROOT}/logs')
MODULE = 'book.part4_vlm.ch33_vlm_lora.tour'


def frozen_hashes():
    parent = ROOT.parent
    paths = list(parent.glob('*selection.json')) + list((parent/'final_test').rglob('*.json'))
    paths += list((parent/'data').glob('*.lock'))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def run(name, part, *extra, gpu=True):
    if part not in (3, 4, 5):
        raise ValueError('Only smoke, training, and val evaluation are allowed')
    if '--split' in extra or '--final-test' in extra:
        raise ValueError('Supplement fixes split=val')
    out = ROOT/name
    if out.exists():
        raise FileExistsError(out)
    events = [json.loads(line) for line in (ROOT/'events.jsonl').read_text().splitlines()] if (ROOT/'events.jsonl').exists() else []
    remaining = 9000 - sum(e['seconds'] for e in events if e['gpu'])
    if remaining <= 0:
        raise RuntimeError('GPU budget exhausted')
    free = int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    required = 9000 if part == 5 else 22000
    if free < required:
        raise RuntimeError(f'Need {required} MiB free before starting; available {free}')
    code = ('import torch; torch.cuda.set_per_process_memory_fraction(19*2**30/torch.cuda.get_device_properties(0).total_memory); '
            f'from {MODULE} import main; main()')
    command = [sys.executable, '-u', '-c', code, str(part), '--model','qwen3vl',
        '--device','cuda','--dtype','bf16','--model-cache',f'{WORK_ROOT}/runs/ch32/hf/hub',
        '--data',str(ROOT/'data'),'--out',str(out),'--split','val',*map(str,extra)]
    before = frozen_hashes()
    start = time.monotonic(); began = datetime.now(timezone.utc).isoformat()
    print(json.dumps(dict(name=name,start=began,command=command,remaining_seconds=remaining)),flush=True)
    status = -1
    try:
        with (LOGS/f'ch33_supp_{name}.log').open('x') as stream:
            result = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=remaining)
            status = result.returncode
    finally:
        event = dict(name=name,start_utc=began,end_utc=datetime.now(timezone.utc).isoformat(),
            seconds=time.monotonic()-start,returncode=status,gpu=gpu,command=command)
        with (ROOT/'events.jsonl').open('a') as stream: stream.write(json.dumps(event)+'\n')
        assert before == frozen_hashes(), 'Frozen artifacts changed'
        print(json.dumps(event),flush=True)
    if status:
        raise RuntimeError(f'{name} failed: {status}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare-smoke','main','old-val'])
    p.add_argument('--steps',type=int,default=1500)
    args = p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(ROOT/'mpl'),
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    if args.stage == 'prepare-smoke':
        (ROOT/'frozen_before.json').write_text(json.dumps(frozen_hashes(),indent=2))
        from miniocr.data.vlm_sft import prepare_images
        print(json.dumps(prepare_images(ROOT/'data',20000,500,include_test=False)),flush=True)
        run('smoke',4,'--rank',32,'--train-n',20000,'--steps',20,'--batch-size',16,'--accum',1,'--lr',1e-4)
        smoke = json.loads((ROOT/'smoke/training.json').read_text())
        print(json.dumps(dict(estimate_train_seconds=smoke['seconds']/20*args.steps,
            peak_allocated_gib=smoke['peak_allocated_gib'],peak_reserved_gib=smoke['peak_reserved_gib'])),flush=True)
    elif args.stage == 'main':
        run('train_r32',4,'--rank',32,'--train-n',20000,'--steps',args.steps,'--batch-size',16,'--accum',1,'--lr',1e-4)
        run('baseline500',5,'--n',500)
        run('r32_val500',5,'--n',500,'--adapter',ROOT/'train_r32/adapter')
    else:
        run('old_r16_val500',5,'--n',500,'--adapter',ROOT.parent/'qwen3vl_main/adapter')
    (ROOT/'frozen_after.json').write_text(json.dumps(frozen_hashes(),indent=2))


if __name__ == '__main__':
    main()
