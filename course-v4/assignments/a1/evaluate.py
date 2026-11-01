"""A1 统一评测（symbol-classification-v1；门槛待校准）。

开发集：python -m assignments.a1.evaluate --ckpt run/model.pt
冻结：  python -m assignments.a1.evaluate --freeze candidates.json --run-dir run/final
最终：  python -m assignments.a1.evaluate --final-test --candidate mlp-s0 --run-dir run/final
冒烟：  python -m assignments.a1.evaluate --smoke --run-dir /tmp/a1-smoke

candidates.json 是 {"候选id": "检查点路径", ...}；相对路径相对于该 JSON。
先完成所有验证集选型，再一次性登记全部候选。最终评测从冻结文件取设备/批量/检查点，
启动记录先于测试数据访问，失败也消耗该候选的机会。同一研究必须保留同一 run-dir。
只加载自己生成的检查点。没有测试错例导出接口。数据生成不下载任何文件。
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from miniocr.data.symbols import (make_dataset, group_split, average_hash,
                                 cross_duplicates, nearest_pixel_distance)
from .tests._impl import adapters as A

PROTOCOL = 'a1-symbol-classification-v1'
# 只用于流程冒烟，不是三套训练配置，也不是效果/门槛建议。
SMOKE_DATA = dict(sources_per_class=2, per_source=2, seed=7, split_seed=8,
                  jitter=1.0, noise=.04, fresh_sources_per_class=1, fresh_seed=9)


def safe_path(path):
    """任何写入路径不得穿过 outputs 符号链接（包括以其他名称指向它的路径）。"""
    p = Path(path).absolute()
    blocked = (Path(__file__).resolve().parents[2]/'outputs').resolve()
    if p.resolve() == blocked or blocked in p.resolve().parents or 'outputs' in p.parts:
        raise ValueError('不得写入 outputs；请使用本副本的独立 run-dir 或 /tmp')
    return p


def sha256(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def write_json(path, obj, exclusive=False):
    path = safe_path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x' if exclusive else 'w',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.flush(); os.fsync(f.fileno())


def data_config(cfg):
    required = set(SMOKE_DATA)
    if set(cfg) != required:
        raise ValueError(f'data_cfg 必须恰含 {sorted(required)}')
    for k in ('sources_per_class','per_source','fresh_sources_per_class'):
        if type(cfg[k]) is not int or cfg[k] <= 0: raise ValueError(k)
    for k in ('seed','split_seed','fresh_seed'):
        if type(cfg[k]) is not int or cfg[k] < 0: raise ValueError(k)
    if cfg['seed'] == cfg['fresh_seed']: raise ValueError('全新来源需另一个生成种子')
    if any(not math.isfinite(cfg[k]) or cfg[k] < 0 for k in ('jitter','noise')):
        raise ValueError('invalid jitter/noise')
    return dict(cfg)


def dataset(cfg, split):
    """数据协议固定24类、原始[0,1]像素、来源8:1:1，split仅 train/val/test/fresh。
    make_dataset 为整体生成接口：生成池后只返回指定子集；不在此处计算模型预测。
    fresh 使用独立seed和不相交group_offset。test/fresh的调用由 final-test 入口控制。
    """
    cfg = data_config(cfg)
    if split not in {'train','val','test','fresh'}: raise ValueError('unknown split')
    fresh = split == 'fresh'
    d = make_dataset(sources_per_class=cfg['fresh_sources_per_class'] if fresh else cfg['sources_per_class'],
                     per_source=cfg['per_source'], seed=cfg['fresh_seed'] if fresh else cfg['seed'],
                     group_offset=24*cfg['sources_per_class'] if fresh else 0,
                     jitter=cfg['jitter'], noise=cfg['noise'])
    if not fresh:
        parts = group_split(d['groups'],fractions=(.8,.1,.1),seed=cfg['split_seed'])
        idx = parts[{'train':0,'val':1,'test':2}[split]]
        d = {k:v[idx] for k,v in d.items() if k in {'images','labels','groups'}}
    if not len(d['labels']): raise ValueError('empty split; increase sources')
    return {k:d[k] for k in ('images','labels','groups')}


def leakage_summary(train, heldout, pixel_threshold=.05, hash_threshold=2, chunk=64):
    """仅汇总近重复检出率；不返回索引、图片或模型错例。门限固定于选型之前。
    按query分块，避免一次构造全量距离矩阵；group disjoint是合成数据的真相参照。
    """
    if chunk <= 0 or pixel_threshold < 0 or hash_threshold < 0: raise ValueError('invalid audit parameters')
    a,b = train['images'],heldout['images']
    ha = average_hash(a)
    hash_hits = pixel_hits = 0
    for start in range(0,len(b),chunk):
        batch = b[start:start+chunk]
        _,_,flags = cross_duplicates(ha,average_hash(batch),max_dist=hash_threshold)
        _,dist = nearest_pixel_distance(a,batch)
        hash_hits += int(flags.sum()); pixel_hits += int((dist<=pixel_threshold).sum())
    return dict(n=len(b), shared_groups=len(set(train['groups']) & set(heldout['groups'])),
                hash_hit_rate=hash_hits/len(b),pixel_hit_rate=pixel_hits/len(b),
                hash_threshold=hash_threshold,pixel_threshold=pixel_threshold)


def grouped_interval(correct, groups, repeats=2000, seed=0):
    """来源cluster bootstrap 95% percentile CI；每次有放回抽同样数量的组，组内全部保留。
    返回图片加权准确率区间（不是逐图片bootstrap），不表达跨数据域的不确定性。
    """
    correct,groups = np.asarray(correct),np.asarray(groups)
    if correct.ndim != 1 or correct.shape != groups.shape or len(correct)==0 or repeats<=0:
        raise ValueError('invalid bootstrap inputs')
    unique,inverse = np.unique(groups,return_inverse=True)
    counts = np.bincount(inverse)
    hits = np.bincount(inverse,weights=correct)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        ix = rng.integers(len(unique),size=len(unique))
        values.append(hits[ix].sum()/counts[ix].sum())
    return [float(v) for v in np.quantile(values,[.025,.975])]


@torch.no_grad()
def metrics(model, data, device='cpu', batch_size=128):
    if batch_size <= 0: raise ValueError('batch_size must be positive')
    model.eval()
    total,correct = 0.,[]
    for start in range(0,len(data['labels']),batch_size):
        x = torch.as_tensor(data['images'][start:start+batch_size],device=device).flatten(1)
        y = torch.as_tensor(data['labels'][start:start+batch_size],device=device)
        z = model(x)
        total += float(A.run_cross_entropy(z,y))*len(y)
        correct.extend((z.argmax(-1)==y).cpu().tolist())
    if not correct: raise ValueError('empty evaluation set')
    return dict(n=len(correct),n_groups=len(np.unique(data['groups'])),
                loss=total/len(correct),accuracy=float(np.mean(correct)),
                accuracy_ci95=grouped_interval(correct,data['groups']),
                ci_method='source-cluster bootstrap percentile; repeats=2000; seed=0')


def load_model(path, device):
    ck = torch.load(path,map_location='cpu',weights_only=False)
    extra = ck['extra']
    cfg = extra['model_cfg']
    if cfg['input_dim'] != 1024 or cfg['n_classes'] != 24: raise ValueError('A1 requires 32x32 / 24 classes')
    data_config(extra['data_cfg'])
    model = A.get_classifier(**cfg)
    model.load_state_dict(ck['model'])
    return model.to(device),extra


def freeze(run_dir, candidates, device='cpu', batch_size=128):
    """一次性登记候选集合、内容哈希、实现源码哈希和评测设置；不访问测试数据。"""
    root = safe_path(run_dir)
    if not candidates or batch_size <= 0: raise ValueError('empty candidates / invalid batch')
    entries = {}
    seen = set()
    common_data = None
    for name,path in candidates.items():
        if not isinstance(name,str) or not name: raise ValueError('invalid candidate id')
        path = str(Path(path).resolve())
        _,extra = load_model(path,'cpu')
        if common_data is not None and extra['data_cfg'] != common_data:
            raise ValueError('all candidates must share data_cfg')
        common_data = extra['data_cfg']
        digest = sha256(path)
        if digest in seen: raise ValueError('duplicate checkpoint under different candidate ids')
        seen.add(digest)
        entries[name] = dict(checkpoint=path,sha256=digest,model_cfg=extra['model_cfg'])
    doc = dict(protocol=PROTOCOL,utc=datetime.now(timezone.utc).isoformat(),candidates=entries,
               data_cfg=common_data,device=device,batch_size=batch_size,implementation=implementation_hash(),
               selected_on='validation only; settings frozen',pixel_threshold=.05,hash_threshold=2)
    write_json(root/'frozen.json',doc,exclusive=True)
    return doc


def implementation_hash():
    """冻结 evaluator、接线与机制源码；参考模式也纳入参考 miniocr 文件。"""
    paths = [Path(__file__),Path(A.__file__),Path(__file__).parent/'tests/adapters.py',
             Path(__file__).resolve().parents[2]/'miniocr/data/symbols.py']
    paths += sorted((Path(A.__file__).parent/'mechanism').glob('**/*.py'))
    # 学生接线在 tests 下，机制目录在上一层。
    paths += sorted((Path(__file__).parent/'mechanism').glob('**/*.py'))
    if A.__name__.startswith('assignments.solutions.'):
        base = Path(__file__).resolve().parents[2]/'miniocr'
        paths += sorted(base.glob('**/*.py'))
    h = hashlib.sha256()
    for p in sorted(set(paths)): h.update(str(p).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def final_test(run_dir, candidate):
    root = safe_path(run_dir)
    frozen_bytes = (root/'frozen.json').read_bytes()
    frozen = json.loads(frozen_bytes)
    if frozen['protocol'] != PROTOCOL or frozen['implementation'] != implementation_hash():
        raise ValueError('protocol/implementation changed after freeze')
    entry = frozen['candidates'][candidate]
    if sha256(entry['checkpoint']) != entry['sha256']: raise ValueError('checkpoint changed after freeze')
    # 开关后的第一项持久副作用：O_EXCL 启动标记；失败也占用，不得删记录后重试。
    # 文件名按检查点内容而不是候选别名；同一个候选只能启动一次。
    marker = root / ('TEST_STARTED_'+entry['sha256']+'.json')
    write_json(marker,dict(candidate=candidate,checkpoint_sha256=entry['sha256'],
                          frozen_sha256=hashlib.sha256(frozen_bytes).hexdigest(),
                          utc=datetime.now(timezone.utc).isoformat(),protocol=PROTOCOL),exclusive=True)
    # 首次启动钉住整份预登记清单，后续候选不得追加/修改配置。
    seal = root/'REGISTRY_SHA256'
    digest = hashlib.sha256(frozen_bytes).hexdigest()
    try:
        with seal.open('x') as f: f.write(digest)
    except FileExistsError:
        if seal.read_text() != digest: raise ValueError('registry changed after first test')
    model,extra = load_model(entry['checkpoint'],frozen['device'])
    if extra['data_cfg'] != frozen['data_cfg']: raise ValueError('data config changed')
    train = dataset(frozen['data_cfg'],'train')
    result = dict(protocol=PROTOCOL,candidate=candidate,checkpoint_sha256=entry['sha256'])
    for split in ('test','fresh'):
        heldout = dataset(frozen['data_cfg'],split)
        result[split] = metrics(model,heldout,frozen['device'],frozen['batch_size'])
        result[split]['leakage'] = leakage_summary(train,heldout,frozen['pixel_threshold'],frozen['hash_threshold'])
    write_json(root/('TEST_RESULT_'+entry['sha256']+'.json'),result,exclusive=True)
    return result


def smoke(run_dir):
    """仅CPU，微型训练→保存→加载→验证评测；不生成fresh、不调用test评测。"""
    root = safe_path(run_dir); root.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(0); torch.set_num_threads(1)
    cfg = dict(input_dim=1024,hidden_dim=8,n_classes=24)
    train,val = dataset(SMOKE_DATA,'train'),dataset(SMOKE_DATA,'val')
    model = A.get_classifier(**cfg)
    opt = A.get_adamw(model.parameters(),lr=.001)
    generator = torch.Generator().manual_seed(0)
    history = []
    t0 = time.perf_counter()
    for step in range(3):
        model.train()
        idx = torch.randint(len(train['labels']),(8,),generator=generator)
        x = torch.from_numpy(train['images'][idx.numpy()]).flatten(1)
        y = torch.from_numpy(train['labels'][idx.numpy()])
        opt.lr = A.run_schedule(step,3,1,.001)
        opt.zero_grad(); loss = A.run_cross_entropy(model(x),y); loss.backward()
        norm = A.run_clip_grad_norm(model.parameters(),1.)
        opt.step()
        history.append(dict(step=step+1,lr=opt.lr,grad_norm=norm,
                            train=metrics(model,train),val=metrics(model,val)))
    path = str(root/'smoke.pt')
    A.save_checkpoint(path,model,opt,3,{'batch':generator},
                      dict(model_cfg=cfg,data_cfg=SMOKE_DATA,history=history,smoke_only=True))
    loaded,_ = load_model(path,'cpu')
    for k,v in model.state_dict().items(): torch.testing.assert_close(v,loaded.state_dict()[k],rtol=0,atol=0)
    result = dict(protocol=PROTOCOL,smoke_only=True,device='cpu',seconds=time.perf_counter()-t0,
                  checkpoint_sha256=sha256(path),history=history,val=metrics(loaded,val),
                  leakage=leakage_summary(train,val),threshold='待校准')
    write_json(root/'smoke.json',result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--ckpt'); mode.add_argument('--freeze'); mode.add_argument('--final-test',action='store_true')
    mode.add_argument('--smoke',action='store_true')
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--candidate')
    p.add_argument('--device',choices=['cpu','mps','cuda'],default=None)
    p.add_argument('--batch-size',type=int,default=None)
    args = p.parse_args(argv)
    if not args.ckpt and args.run_dir is None: p.error('此模式需要 --run-dir')
    if args.final_test:
        if not args.candidate or args.device is not None or args.batch_size is not None:
            p.error('--final-test 需要 --candidate；设备和批量只能使用冻结配置')
        result = final_test(args.run_dir,args.candidate)
    elif args.freeze:
        src = Path(args.freeze).resolve()
        candidates = {k:src.parent/v for k,v in json.loads(src.read_text()).items()}
        result = freeze(args.run_dir,candidates,args.device or 'cpu',128 if args.batch_size is None else args.batch_size)
    elif args.smoke:
        if args.device not in (None,'cpu'): p.error('冒烟只允许 CPU')
        result = smoke(args.run_dir)
    else:
        model,extra = load_model(args.ckpt,args.device or 'cpu')
        val = dataset(extra['data_cfg'],'val')
        result = dict(protocol=PROTOCOL,split='val',checkpoint_sha256=sha256(args.ckpt),
                      metrics=metrics(model,val,args.device or 'cpu',128 if args.batch_size is None else args.batch_size),
                      leakage=leakage_summary(dataset(extra['data_cfg'],'train'),val),threshold='待校准')
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    return result


if __name__ == '__main__':
    main()
