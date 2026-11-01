"""A4 训练层评测（仅本地模型，不下载）。

python -m assignments.a4.evaluate --config run.json --data dev.jsonl
最终：另加 --split test --final-test --registry frozen.json --candidate <预登记 id>。

run.json: {"factory":"package.module:load", "checkpoint":"/absolute/model.pt",
 "device":"cpu", "settings":{...冻结的生成参数...}, "protocol_version":"你的协议版本"}。
factory(checkpoint, device, settings) 返回 predict(list[str])->list[str]；只向模型传 source。
JSONL 每行 {id, split:val|test, source, target, is_clean:bool}；is_clean 必须与原串相等一致。
registry: {"frozen":true, "test_sha256":"...", "candidates":{"id":{
 "config_sha256":"...", "checkpoint_sha256":"...", "factory_sha256":"..."}}}。
factory_sha256 是 factory 模块 .py 文件散列；其依赖/分词器版本须在 settings/报告记录。

在 --final-test 守卫处先持久化 started，再读 test；失败也消耗该候选机会。
首次启动冻结整个登记表；同检查点即使换候选名也拒绝重复。日志固定在本作业下。
只输出汇总，不输出逐样本、分桶或 test 探针。门槛未校准，不输出训练通过结论。
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from .tests._impl import adapters as A

LEDGER = Path(__file__).resolve().parent / 'final_test_runs'


def sha256(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def _append(path, record):
    import os
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
        f.flush()
        os.fsync(f.fileno())


def claim_final(registry, candidate, identity):
    """原子启动登记；调用前不可打开 test。进程崩溃也不退还机会。"""
    reg = json.loads(Path(registry).read_text())
    if reg.get('frozen') is not True or reg.get('candidates',{}).get(candidate) != identity:
        raise ValueError('必须先冻结并预登记候选及配置/检查点/工厂散列')
    if not isinstance(reg.get('test_sha256'),str) or len(reg['test_sha256']) != 64:
        raise ValueError('缺少预登记的测试集散列')
    LEDGER.mkdir(exist_ok=True)
    if LEDGER.is_symlink():
        raise ValueError('日志目录不得为符号链接')
    with (LEDGER/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        frozen = LEDGER/'registry.json'
        canonical = json.dumps(reg,sort_keys=True)
        if frozen.exists() and frozen.read_text() != canonical:
            raise ValueError('首次 final-test 后不能改登记表')
        log = LEDGER/'events.jsonl'
        records = [json.loads(s) for s in log.read_text().splitlines()] if log.exists() else []
        if any(r['event']=='started' and (r['candidate']==candidate or r['checkpoint_sha256']==identity['checkpoint_sha256']) for r in records):
            raise ValueError('同一候选/检查点已经启动过 test，拒绝重复')
        if not frozen.exists():
            with frozen.open('x') as f:
                f.write(canonical)
                f.flush()
                import os
                os.fsync(f.fileno())
        _append(log,dict(event='started',candidate=candidate,time=datetime.now(timezone.utc).isoformat(),**identity))
    return reg['test_sha256']


def reward_tools():
    from miniocr.eval.render import can_parse
    from miniocr.eval.latex_norm import normalize_latex
    from miniocr.eval.metrics import sample_error_rate
    from miniocr.eval.postcorrect_eval import format_issues
    return dict(parse=can_parse,normalize=normalize_latex,error_rate=sample_error_rate,format_issues=format_issues)


def summarize(rows, predictions, *, split):
    import numpy as np
    from miniocr.eval.postcorrect_eval import score_postcorrect
    if len(predictions)!=len(rows) or any(not isinstance(s,str) for s in predictions):
        raise ValueError('预测必须等长且全部为字符串')
    panel = score_postcorrect(rows,predictions)['summary']
    panel.pop('by_error_type',None)
    tools = reward_tools()
    rewards = [A.reward(p,e.source,e.target,**tools) for p,e in zip(predictions,rows)]
    rng=np.random.default_rng(0)
    means=[float(np.mean(np.asarray(rewards)[rng.integers(len(rows),size=len(rows))])) for _ in range(1000)]
    result=dict(protocol='a4-postcorrect-v1/raw+course-normalize',split=split,panel=panel,
                reward_mean=float(np.mean(rewards)),reward_interval=np.quantile(means,[.025,.975]).tolist(),
                threshold_status='待校准',training_pass=None)
    if split == 'dev':
        # 静态探针检查奖励规格；模型退化诊断单独报告，二者不能互相替代。
        probes={}
        for e in rows:
            for name,text in A.hack_probes(e.source,e.target):
                probes.setdefault(name,[]).append(A.reward(text,e.source,e.target,**tools))
        result['static_probe_rewards']={k:sum(v)/len(v) for k,v in probes.items()}
        result['gold_reward_mean']=sum(A.reward(e.target,e.source,e.target,**tools) for e in rows)/len(rows)
        result['model_diagnostics']=dict(empty_rate=sum(not p.strip() for p in predictions)/len(rows),
            wrong_single_symbol_rate=sum(len(p.strip())==1 and tools['normalize'](p)!=tools['normalize'](e.target) for p,e in zip(predictions,rows))/len(rows),
            copy_source_rate=sum(p==e.source for p,e in zip(predictions,rows))/len(rows),
            mean_char_length=sum(map(len,predictions))/len(rows))
        result['probe_status']='待校准：静态探针不证明模型通过；结合基线与主面板人工审查'
    return result


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',required=True)
    ap.add_argument('--data',required=True)
    ap.add_argument('--split',choices=['dev','test'],default='dev')
    ap.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    ap.add_argument('--final-test',action='store_true',help='选型已冻结；预登记候选各评一次，不据此改设置，开发期不看 test 错例')
    ap.add_argument('--registry')
    ap.add_argument('--candidate')
    args=ap.parse_args(argv)
    # 守卫必须早于 test 数据读取、模型加载和预测。
    if args.split=='test' and not (args.final_test and args.registry and args.candidate):
        raise ValueError('test 必须显式 --final-test、--registry 和 --candidate')
    if args.split=='dev' and args.final_test:
        raise ValueError('--final-test 只能用于 test')
    config=json.loads(Path(args.config).read_text())
    module,name=config['factory'].split(':')
    if config.get('device') != args.device:
        raise ValueError('设备必须与冻结配置的 device 相同')
    if not config.get('protocol_version') or not isinstance(config.get('settings'),dict):
        raise ValueError('配置必须包含协议版本与 settings')
    identity=dict(config_sha256=sha256(args.config),checkpoint_sha256=sha256(config['checkpoint']),
                  factory_sha256=sha256(importlib.util.find_spec(module).origin))
    expected=None
    if args.split=='test':
        expected=claim_final(args.registry,args.candidate,identity)
    try:
        data_hash=sha256(args.data)
        if expected is not None and data_hash!=expected:
            raise ValueError('测试数据与冻结散列不一致（已记录启动）')
        rows=[]
        for line in Path(args.data).read_text().splitlines():
            r=json.loads(line)
            if r['split'] != ('val' if args.split=='dev' else 'test') or not all(isinstance(r[k],str) for k in ('id','source','target')) or type(r['is_clean']) is not bool or r['is_clean']!=(r['source']==r['target']):
                raise ValueError('数据字段、划分或 is_clean 错误')
            rows.append(SimpleNamespace(**r,error_types=()))
        if not rows or len({r.id for r in rows})!=len(rows):
            raise ValueError('空集或重复 ID')
        predict=getattr(importlib.import_module(module),name)(config['checkpoint'],args.device,config['settings'])
        result=summarize(rows,predict([r.source for r in rows]),split=args.split)
        result.update(identity,data_sha256=data_hash,device=args.device,protocol_version=config['protocol_version'])
        if args.split=='test':
            _append(LEDGER/'events.jsonl',dict(event='completed',candidate=args.candidate,summary=result))
        print(json.dumps(result,ensure_ascii=False,allow_nan=False))
        return result
    except Exception:
        if args.split=='test':
            # 不写异常内容：模型/工具异常可能包含 test 原文。
            _append(LEDGER/'events.jsonl',dict(event='failed',candidate=args.candidate))
            raise RuntimeError('final-test 失败，已占用候选机会；日志不含逐样本内容') from None
        raise


if __name__=='__main__':
    main()
