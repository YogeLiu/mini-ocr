"""A3 离线后纠错评测。默认 val；不下载数据/权重，不自动训练。

python -m assignments.a3.evaluate --data val.jsonl --config selected.json --baseline prompt.json
python -m assignments.a3.evaluate --freeze candidates.json
python -m assignments.a3.evaluate --data test.jsonl --config selected.json --split test --final-test --candidate lora-r4

配置 JSON：{"predictor":"your_package:predict", "kwargs":{}, "artifacts":{"本地权重路径":"sha256"},
"run_id":"...", "model_revision":"...", "prompt":"...", "generation":{}, "seed":0}
predict(sources: list[str], config: dict) -> list[str]：只收 source，不收答案；自行加载本地模型，
使用 config 中冻结的生成设置，返回未抽取/剥壳的原始解码文本。代码文件也应放 artifacts。
数据采用 miniocr.data.postcorrect.save_jsonl 格式，保留原 ID，不能重新枚举。
冻结清单：{"protocol":protocol_id(),"data_sha256":"数据所有者预先提供的 test 文件哈希",
"candidates":{"lora-r4":"selected.json 文件 sha256", "prompt":"prompt.json 文件 sha256"}}。
冻结不读 test。清单只写一次；开始记录先于读取 test、加载模型、生成和评分，失败也消耗该候选。
守卫只能防同一 ledger 目录的重复操作，不是全局访问控制；不得换目录/删记录重评。
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = Path(__file__).resolve().parent / 'test_ledger'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_id():
    files = ['assignments/a3/evaluate.py', 'miniocr/eval/postcorrect_eval.py',
             'miniocr/eval/latex_norm.py', 'miniocr/eval/metrics.py']
    return 'a3-text-v1:' + hashlib.sha256(''.join(digest(ROOT / p) for p in files).encode()).hexdigest()


def writable(path):
    path = Path(path)
    # outputs 是主仓库符号链接；也禁止经过其他符号链接写到工作树外。
    resolved = path.resolve()
    if 'outputs' in path.parts or 'outputs' in resolved.parts:
        raise ValueError('不能写 outputs')
    if not (resolved.is_relative_to(ROOT) or resolved.is_relative_to(Path('/tmp'))):
        raise ValueError('输出只允许当前副本或 /tmp')
    return path


def exclusive_json(path, value):
    path = writable(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2); f.flush(); os.fsync(f.fileno())


def freeze(manifest, ledger=DEFAULT_LEDGER):
    obj = json.loads(Path(manifest).read_text())
    if obj.get('protocol') != protocol_id() or not obj.get('candidates'):
        raise ValueError('协议不匹配或候选为空')
    for value in [obj.get('data_sha256', ''), *obj['candidates'].values()]:
        if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):
            raise ValueError('必须登记 SHA256')
    exclusive_json(Path(ledger)/'FROZEN.json',obj)


def start_test(config, candidate, ledger=DEFAULT_LEDGER):
    if not candidate or not re.fullmatch(r'[A-Za-z0-9_-]+',candidate):
        raise ValueError('候选名只能含字母数字下划线连字符')
    ledger=Path(ledger); frozen=json.loads((ledger/'FROZEN.json').read_text())
    config_hash=digest(config)
    if frozen['protocol'] != protocol_id() or frozen['candidates'].get(candidate) != config_hash:
        raise ValueError('候选未登记、配置已修改或协议已改变')
    # 按配置哈希锁定，给同一配置另取候选名也不能重评。
    exclusive_json(ledger/f'{config_hash}.STARTED.json',dict(candidate=candidate,config_sha256=config_hash,
        frozen_sha256=digest(ledger/'FROZEN.json'),started=datetime.now(timezone.utc).isoformat(),status='started'))
    return frozen


def predict(sources, config):
    cfg=json.loads(Path(config).read_text())
    required={'predictor','kwargs','artifacts','run_id','model_revision','prompt','generation','seed'}
    if not required <= cfg.keys(): raise ValueError('配置缺少可追溯字段')
    for path,sha in cfg['artifacts'].items():
        if digest(path) != sha: raise ValueError('代码/权重 artifact 已改变')
    module,name=cfg['predictor'].split(':')
    result=list(getattr(importlib.import_module(module),name)(sources,cfg))
    if len(result) != len(sources) or any(not isinstance(s,str) for s in result):
        raise ValueError('预测必须为等长字符串序列')
    return result


def panel(examples, predictions):
    from miniocr.eval.postcorrect_eval import score_postcorrect
    s=score_postcorrect(examples,predictions,bootstrap_samples=1000,bootstrap_seed=0)['summary']
    # 不导出逐例输出、错误类型分桶或 test 错例。
    return {k:s[k] for k in ('n','em_norm','cer','broken_rate','fix_rate')}


def evaluate(data, config, *, split='val', baseline=None, final_test=False, candidate=None, ledger=DEFAULT_LEDGER):
    if split not in ('val','test'): raise ValueError('split 只能为 val/test')
    frozen=None
    if split == 'test':
        if not final_test: raise ValueError('test 必须显式 --final-test；冻结后各候选一次，不据此改设置')
        if baseline is not None: raise ValueError('test 的提示基线须作为独立预登记候选')
        frozen=start_test(config,candidate,ledger)  # 必须在任何 test 数据访问之前
    elif final_test:
        raise ValueError('--final-test 只用于 test')
    elif baseline is None:
        raise ValueError('val 必须提供同批提示基线配置')
    if frozen and digest(data) != frozen['data_sha256']: raise ValueError('数据哈希不符')
    from miniocr.data.postcorrect import load_jsonl
    examples=load_jsonl(data)
    if not examples or any(e.split != split for e in examples) or len({e.id for e in examples}) != len(examples):
        raise ValueError('数据为空、split 不符或 ID 重复')
    sources=[e.source for e in examples]
    result=dict(protocol=protocol_id(),split=split,data_sha256=digest(data),config_sha256=digest(config),
                candidate=candidate,threshold='待校准',model=panel(examples,predict(sources,config)))
    if split == 'val':
        result['prompt_baseline']=panel(examples,predict(sources,baseline))
        result['baseline_config_sha256']=digest(baseline)
        result['copy_baseline']=panel(examples,sources)
        result['em_norm_gain_over_prompt']=result['model']['em_norm']['value']-result['prompt_baseline']['em_norm']['value']
    else:
        exclusive_json(Path(ledger)/f"{digest(config)}.RESULT.json",result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data'); p.add_argument('--config'); p.add_argument('--baseline')
    p.add_argument('--split',choices=['val','test'],default='val'); p.add_argument('--final-test',action='store_true')
    p.add_argument('--candidate'); p.add_argument('--freeze'); p.add_argument('--ledger',default=str(DEFAULT_LEDGER))
    a=p.parse_args(argv)
    if a.freeze:
        freeze(a.freeze,a.ledger); print('候选清单已冻结'); return
    if not a.data or not a.config: p.error('需要 --data 和 --config')
    print(json.dumps(evaluate(a.data,a.config,split=a.split,baseline=a.baseline,final_test=a.final_test,
                              candidate=a.candidate,ledger=a.ledger),ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
