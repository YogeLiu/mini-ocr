"""Val-only supplementary run; fixed two-seed comparison with a wall-time ceiling."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

WORK_ROOT = os.environ.get('COURSE_WORK_ROOT', '/hy-tmp')

ROOT = Path(f'{WORK_ROOT}/runs/ch35/supp')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    start = time.monotonic()
    # Allow for the completed smoke; subprocess timeouts target only our child.
    events = ROOT/"events.jsonl"
    used = sum(json.loads(line)["seconds"] for line in events.read_text().splitlines()) if args.resume and events.exists() else 0
    deadline = start + 8700 - used
    common = ['--val-only', '--device', 'cuda', '--dtype', 'bf16', '--model-cache',
              f'{WORK_ROOT}/runs/ch32/hf/hub', '--data', str(ROOT/'data')]
    tasks = [('teacher', ['2', '3', '--teacher-adapter', f'{WORK_ROOT}/runs/ch33/qwen3vl_main/adapter',
                         '--n', '5000', '--batch-size', '16', '--scale', '1', '--max-pixels', '65536'])]
    for seed in (35, 36):
        for arm in ('original', 'distill'):
            tasks.append((f'{arm}_{seed}', ['4', '--arm', arm, '--seed', str(seed), '--n-train', '5000',
                '--n', '500', '--batch-size', '16', '--steps', '3000', '--accum', '2', '--rank', '64', '--audit-n', '12']))
    os.environ.update(HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(ROOT/'mpl'))
    for name, extra in tasks:
        command = [sys.executable, '-u', '-m', 'book.part4_vlm.ch35_distill.tour', *extra, *common,
                   '--max-new-tokens', '192', '--teacher-dir', str(ROOT/'teacher'), '--out', str(ROOT/name)]
        if (ROOT/name).exists():
            if not args.resume or not (ROOT/name/'run_end.json').exists():
                raise FileExistsError(f"refusing to overwrite {ROOT/name}")
            old = json.loads((ROOT/name/'run_start.json').read_text())['config']
            options = command[4:]
            parts = []
            i = 0
            while i < len(options):
                token = options[i]
                if not token.startswith('--'):
                    parts.append(int(token)); i += 1; continue
                key = token[2:].replace('-', '_')
                value = True if key == 'val_only' else options[i+1]
                previous = old[key]
                matches = (float(previous) == float(value)) if isinstance(previous, (int, float)) else str(previous) == str(value)
                if not matches:
                    raise ValueError(f"resume config mismatch: {name}/{key}")
                i += 1 if key == 'val_only' else 2
            if old['parts'] != parts:
                raise ValueError(f"resume parts mismatch: {name}")
            print(json.dumps(dict(name=name, resume='completed; config matched')), flush=True)
            continue
        free = int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        required = 7000 if name == 'teacher' else 4000
        if free < required:
            raise RuntimeError(f'insufficient free GPU memory: {free} MiB')
        began = datetime.now(timezone.utc).isoformat()
        tick = time.monotonic()
        with open(f'{WORK_ROOT}/logs/ch35_supp_{name}.log', 'w') as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=max(1,deadline-tick))
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 'budget timeout'
        event = dict(name=name, start=began, end=datetime.now(timezone.utc).isoformat(),
                     seconds=time.monotonic()-tick, returncode=code, command=command)
        with (ROOT/'events.jsonl').open('a') as f:
            f.write(json.dumps(event)+'\n')
        print(json.dumps(event), flush=True)
        if code != 0:
            raise RuntimeError(f'{name}: {code}')

if __name__ == '__main__':
    main()
