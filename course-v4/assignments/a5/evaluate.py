"""A5 OCR 精简面板；使用已有本地原图缓存，不下载。

开发：--ckpt model.pt --data-dir /tmp/a5-data --n 20
冻结：--freeze-plan plan.json --run-dir /tmp/a5-final
终评：--run-dir /tmp/a5-final --candidate base --split test --final-test
CPU 冒烟：A_SOLUTIONS=1 ... --smoke --run-dir /tmp/a5-smoke

plan.json: {"candidates":{"base":"/absolute/model.pt"},"data_dir":"/absolute/cache",
"test_n":500,"device":"cpu","batch_size":8,"seed":0}
检查点：{"cfg":模型配置,"state_dict":模型权重}。候选/设置在 test 前一次冻结。
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import torch
from assignments.a5.tests._impl import adapters as A

ROOT = Path(__file__).resolve().parents[2]


def safe_path(path):
    path = Path(path).absolute()
    # 包括符号链接解析后的主仓库 outputs；该目录绝不能成为写入目标。
    forbidden = (ROOT/'outputs').resolve()
    if path.resolve().is_relative_to(forbidden):
        raise ValueError('outputs is shared; choose a private directory')
    return path


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def training_code():
    """记录可选训练源码目录；缺失与存在但为空是不同身份。"""
    directory = ROOT/'assignments/a5/training'
    return dict(status='present' if directory.is_dir() else 'missing',
                files={str(p.relative_to(directory)): digest(p)
                       for p in sorted(directory.rglob('*.py'))})


def code_digest():
    paths = [Path(__file__), *sorted((ROOT/'assignments/a5/mechanism').rglob('*.py')),
             ROOT/'assignments/a5/tests/adapters.py', *sorted((ROOT/'miniocr').rglob('*.py'))]
    if os.environ.get('A_SOLUTIONS') == '1':
        paths += sorted((ROOT/'assignments/solutions/a5').rglob('*.py'))
    payload = ''.join(str(p.relative_to(ROOT))+digest(p) for p in paths)
    payload += json.dumps(training_code(), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def write_new(path,obj):
    with safe_path(path).open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def data_snapshot(directory):
    """绑定已准备的 train/val 缓存与原始 parquet 快照元数据；不解码 test。"""
    from miniocr.data.latex_corpus import _parquet_paths
    caches = {s:digest(Path(directory)/f'{s}.pt') for s in ('train','val')
              if (Path(directory)/f'{s}.pt').exists()}
    sources = [{"path":str(p.resolve()),"size":p.stat().st_size,"mtime_ns":p.stat().st_mtime_ns}
               for p in _parquet_paths()]
    return dict(caches=caches,sources=sources)


def freeze(plan_path,run_dir):
    plan=json.loads(Path(plan_path).read_text())
    if not plan['candidates'] or plan['test_n']<1 or plan['batch_size']<1:
        raise ValueError('nonempty candidates and positive counts required')
    run_dir=safe_path(run_dir); run_dir.mkdir(parents=True,exist_ok=True)
    candidates={}
    for name,path in plan['candidates'].items():
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
            raise ValueError('candidate IDs use letters, digits, underscore, hyphen')
        p=Path(path).resolve()
        candidates[name]=dict(path=str(p),sha256=digest(p))
    if len({c['sha256'] for c in candidates.values()}) != len(candidates):
        raise ValueError('duplicate checkpoints under different candidate names')
    lengths = [torch.load(c['path'], map_location='cpu', weights_only=True)['cfg']['max_length']
               for c in candidates.values()]
    if any(type(length) is not int or length < 1 for length in lengths):
        raise ValueError('max_length must be a positive integer')
    if len(set(lengths)) != 1:
        raise ValueError('all candidates must have the same max_length')
    data_dir=safe_path(plan['data_dir']).resolve()
    frozen=dict(protocol='a5-ocr-v1/latex-v1',utc=datetime.now(timezone.utc).isoformat(),
                max_length=lengths[0],training_code=training_code(),candidates=candidates,data_dir=str(data_dir),test_n=plan['test_n'],
                batch_size=plan['batch_size'],device=plan['device'],seed=plan['seed'],
                data_snapshot=data_snapshot(data_dir),code_sha256=code_digest(),solutions=os.environ.get('A_SOLUTIONS')=='1',
                decoding='greedy; max_length+1; PAD/BOS forbidden',precision='float32',
                selection='fixed on val before test; no subsequent tuning')
    write_new(run_dir/'FROZEN.json',frozen)
    write_new(run_dir/'FROZEN.sha256.json',dict(sha256=digest(run_dir/'FROZEN.json')))
    if (data_dir/'test.pt').exists():
        write_new(run_dir/'TEST_DATA.sha256.json',dict(sha256=digest(data_dir/'test.pt')))
    return frozen


def begin_test(run_dir,candidate,final_test):
    # 开关和登记检查先于模型/数据加载。独占创建使并发及失败后的重试均拒绝。
    if not final_test: raise ValueError('test requires --final-test')
    run_dir=safe_path(run_dir)
    expected=json.loads((run_dir/'FROZEN.sha256.json').read_text())['sha256']
    if digest(run_dir/'FROZEN.json') != expected: raise ValueError('frozen registry changed')
    frozen=json.loads((run_dir/'FROZEN.json').read_text())
    if frozen['code_sha256'] != code_digest(): raise ValueError('evaluation/model code changed since freeze')
    if frozen['solutions'] != (os.environ.get('A_SOLUTIONS')=='1'): raise ValueError('implementation changed')
    if candidate not in frozen['candidates']: raise ValueError('candidate was not preregistered')
    if data_snapshot(frozen['data_dir']) != frozen['data_snapshot']:
        raise ValueError('training/validation cache or source snapshot changed')
    seal=run_dir/'TEST_DATA.sha256.json'
    if seal.exists() and digest(Path(frozen['data_dir'])/'test.pt') != json.loads(seal.read_text())['sha256']:
        raise ValueError('test cache changed')
    entry=frozen['candidates'][candidate]
    if digest(entry['path']) != entry['sha256']: raise ValueError('checkpoint changed')
    marker=run_dir/('TEST_STARTED_'+entry['sha256']+'.json')
    write_new(marker,dict(utc=datetime.now(timezone.utc).isoformat(),candidate=candidate,
                         checkpoint_sha256=entry['sha256'],registry_sha256=expected,
                         status='started; failure also consumes this candidate'))
    return frozen,entry


def load_model(path,device):
    ckpt=torch.load(path,map_location='cpu',weights_only=True)
    m=A.get_ocr(ckpt['cfg']).to(device)
    m.load_state_dict(ckpt['state_dict'])
    return m.eval()


def score(refs,preds,split):
    from miniocr.data.postcorrect import PostCorrectExample
    from miniocr.eval.postcorrect_eval import score_postcorrect
    examples=[PostCorrectExample(str(i),split,r,r) for i,r in enumerate(refs)]
    summary=score_postcorrect(examples,preds,bootstrap_samples=1000,bootstrap_seed=0)['summary']
    return {k:summary[k] for k in ('n','em_raw','em_norm','cer','parse_rate')}


def evaluate(model,data,batch_size,device):
    preds=[]
    start=time.perf_counter()
    with torch.no_grad():
        for i in range(0,len(data['images']),batch_size):
            preds.extend(model.generate(data['images'][i:i+batch_size].to(device)))
    refs=[r['formula'] for r in data['manifest']]
    result=score(refs,preds,data['split'])
    # 无逐例输出/分桶/错例统计；只返回总体数值及区间。
    return dict(metrics=result,seconds=time.perf_counter()-start,device=str(device),
                torch=torch.__version__,python=platform.python_version(),threshold='待校准',
                panel='ch00 §0.9.3 OCR精简版',normalizer='miniocr.eval.latex_norm; see code_sha256',
                renderer='matplotlib mathtext parse only; no render match',
                interval='Wilson 95%; CER percentile bootstrap n=1000 seed=0')


def smoke(run_dir):
    """只使用 train/val 的渲染冒烟数据；不能充当主任务结果或门槛。"""
    from miniocr.data.contrastive_pairs import prepare_pairs
    run_dir=safe_path(run_dir); run_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1); torch.manual_seed(0)
    cfg=dict(width=8,vision_depth=1,decoder_depth=1,patch=16,max_length=16)
    data=prepare_pairs(run_dir/'data',{'train':2,'val':2},source='rendered',max_length=16)
    m=A.get_ocr(cfg).cpu().train()
    x,y=A.targets_for([r['formula'] for r in data['train']['manifest']],16)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
    loss=m(data['train']['images'],x,y)[1]; loss.backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()):
        raise ValueError('missing/nonfinite training gradient')
    opt.step()
    path=run_dir/'smoke.pt'
    if path.exists(): raise FileExistsError('use fresh smoke directory')
    torch.save(dict(cfg=cfg,state_dict=m.state_dict()),path)
    loaded=load_model(path,'cpu')
    torch.testing.assert_close(m.eval()(data['train']['images'],x)[0],loaded(data['train']['images'],x)[0])
    result=evaluate(loaded,data['val'],2,'cpu')
    result.update(status='CPU smoke only; rendered train/val; not a training result',code_sha256=code_digest(),training_code=training_code())
    write_new(run_dir/'smoke.json',result)
    return {'status':result['status'],'checkpoint_roundtrip':True}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--final-test',action='store_true')
    p.add_argument('--freeze-plan'); p.add_argument('--run-dir')
    p.add_argument('--candidate'); p.add_argument('--ckpt'); p.add_argument('--data-dir')
    p.add_argument('--n',type=int,default=20); p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--device',default='cpu'); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args(argv)
    if a.smoke:
        if a.split!='val' or a.final_test: p.error('smoke never reads test')
        if not a.run_dir: p.error('--run-dir required')
        result=smoke(a.run_dir)
    elif a.freeze_plan:
        if not a.run_dir: p.error('--run-dir required')
        freeze(a.freeze_plan,a.run_dir); result={'status':'frozen; test unread'}
    else:
        if a.split=='test':
            if not a.run_dir: p.error('--run-dir required')
            f,e=begin_test(a.run_dir,a.candidate,a.final_test)
            ckpt,device,data_dir,n,batch,seed=e['path'],f['device'],f['data_dir'],f['test_n'],f['batch_size'],f['seed']
        else:
            if a.final_test: p.error('--final-test requires --split test')
            if not a.ckpt or not a.data_dir: p.error('--ckpt and --data-dir required')
            ckpt,device,data_dir,n,batch,seed=a.ckpt,a.device,a.data_dir,a.n,a.batch_size,a.seed
        if n<1 or batch<1: raise ValueError('positive counts required')
        torch.manual_seed(seed)
        model=load_model(ckpt,device)
        from miniocr.data.contrastive_pairs import prepare_pairs
        data=prepare_pairs(safe_path(data_dir),{a.split:n},source='open',
                           max_length=model.config['max_length'],allow_test=a.split=='test' and a.final_test)[a.split]
        if a.split=='test':
            seal=Path(a.run_dir)/'TEST_DATA.sha256.json'
            current=digest(Path(data_dir)/'test.pt')
            try:
                write_new(seal,dict(sha256=current))
            except FileExistsError:
                if json.loads(seal.read_text())['sha256'] != current:
                    raise ValueError('test data differs between candidates')
        result=evaluate(model,data,batch,device)
        result.update(utc=datetime.now(timezone.utc).isoformat(),checkpoint_sha256=digest(ckpt),code_sha256=code_digest(),training_code=training_code(),split=a.split,seed=seed)
        if a.split=='test': write_new(Path(a.run_dir)/('TEST_RESULT_'+e['sha256']+'.json'),result)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    return result


if __name__=='__main__': main()
