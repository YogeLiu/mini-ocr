"""共享实例 GPU 采样；只读 nvidia-smi/ps，stop 文件出现即退出。"""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def monitor(root=Path('outputs/book/ch19/gpu')):
    """每十秒记录显存/利用率/进程；不配置系统，不控制其他进程。"""
    root.mkdir(parents=True, exist_ok=True)
    with (root/'gpu_monitor.jsonl').open('a') as f:
        while not (root/'monitor.stop').exists():
            commands = {
                'gpu': ['nvidia-smi', '--query-gpu=name,memory.used,memory.free,utilization.gpu', '--format=csv,noheader'],
                'processes': ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'],
                'ps': ['ps', '-eo', 'pid,args'],
            }
            row = dict(utc=datetime.now(timezone.utc).isoformat())
            for name, cmd in commands.items():
                output = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
                row[name] = [s for s in output.splitlines() if 'python' in s and '.tour ' in s] if name == 'ps' else output
            f.write(json.dumps(row)+'\n')
            f.flush()
            time.sleep(10)


if __name__ == '__main__':
    monitor()
