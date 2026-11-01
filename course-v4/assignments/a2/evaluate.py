"""A2 统一评测，CPU 默认；只读本地 latex-v1，默认 val 前 2000 条。

python -m assignments.a2.evaluate --tokenizer bpe.json --ckpt model.pt
先用 --register candidates.json --candidate baseline 登记每个冻结候选；全部登记后，
--final-test --candidate baseline 启动测试。第一次启动封存整个登记表；启动失败也占用次数。
登记表固定在 assignments/a2/final_test_registry.json；不可删除/另建以重评。
候选文件是 JSON，内容为 candidate_identity() 输出。--register 会写该文件供审阅。
检查点 {cfg,state_dict}；tokenizer {merges:[[hex,hex]],specials:[bos,eos]}。
仅汇总数字，不保存测试文本/逐条损失/错例。可编译率使用 mathtext 近似及 Wilson 区间。
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import fcntl

import torch
import torch.nn.functional as F
from miniocr.data.latex_corpus import corpus_available, standard_split, SAMPLE_FORMULAS
from miniocr.data.latex_lm import FormulaWindows
from .tests._impl import adapters as A

REGISTRY = Path(__file__).with_name('final_test_registry.json')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def corpus_identity():
    # 绑定完整缓存快照摘要；不解析/输出 test 文本或错例。
    from miniocr.data.latex_corpus import _parquet_paths
    return {str(p): digest(p) for p in _parquet_paths()}


def candidate_identity(args):
    return dict(checkpoint=digest(args.ckpt), tokenizer=digest(args.tokenizer),
                protocol='a2-latex-v1-sliding-bos-masked-v1', device=args.device,
                corpus=corpus_identity() if not args.smoke else 'builtin-smoke',
                samples=args.samples, max_new=args.max_new, seed=args.seed,
                evaluator=digest(__file__), implementation=digest(A.__file__),
                adapters=digest(Path(__file__).parent/'tests/adapters.py'),
                mechanism={str(p.relative_to(Path(__file__).parent)): digest(p)
                           for p in sorted(Path(__file__).with_name('mechanism').rglob('*.py'))})


def registry_action(path, name, identity, start=False):
    """锁内原子登记/启动；started 在读取 test 前 fsync，失败也不释放次数。"""
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
        raise ValueError('candidate 只允许字母数字下划线连字符')
    path = Path(path)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(path.read_text()) if path.exists() else {'frozen': False, 'candidates': {}}
        records = state['candidates']
        if start:
            if name not in records or records[name]['identity'] != identity:
                raise ValueError('候选未预登记或冻结配置/文件已经变化')
            if 'started' in records[name]:
                raise ValueError('此候选已启动过，拒绝重复执行')
            state['frozen'] = True
            records[name]['started'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        else:
            if state['frozen'] or name in records or any(r['identity'] == identity for r in records.values()):
                raise ValueError('登记已冻结或候选名已存在')
            records[name] = {'identity': identity, 'registered': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        tmp = path.with_suffix('.tmp')
        with tmp.open('w') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    return records[name]


def load_tokenizer(path):
    d = json.loads(Path(path).read_text())
    merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a,b in d['merges']]
    specials = d['specials']
    if specials != ['<|bos|>', '<|eos|>']:
        raise ValueError('特殊符号须依序为 BOS/EOS')
    vocab = {i: bytes([i]) for i in range(256)}
    for a,b in merges:
        vocab[len(vocab)] = a+b
    for s in specials:
        vocab[len(vocab)] = s.encode()
    return A.get_tokenizer(vocab, merges, specials), len(vocab)-2, len(vocab)-1, len(vocab)


@torch.no_grad()
def score(model, tok, texts, bos, eos, context, batch=32):
    """每个内容 token 与 EOS 恰好一次；逐目标滑窗、无跨公式上下文。"""
    n_bytes = sum(len(t.encode('utf-8')) for t in texts)
    if not texts or n_bytes == 0 or batch < 1:
        raise ValueError('非空公式/正 batch 必需')
    windows = FormulaWindows([tok.encode(t, allow_special=False) for t in texts], bos, eos, context)
    device = next(model.parameters()).device
    total = 0.
    for indices in torch.arange(len(windows)).split(batch):
        x,lengths,y = windows.batch(indices)
        logits = model(x.to(device))[0][torch.arange(len(x), device=device), (lengths-1).to(device)].float().clone()
        logits[:,bos] = -torch.inf
        total += F.cross_entropy(logits,y.to(device),reduction='sum').item()
    return dict(n_formulas=len(texts), n_bytes=n_bytes, n_targets=len(windows), nll_sum=total,
                nat_per_token=total/len(windows), bits_per_byte=total/n_bytes/math.log(2))


def compilation(model, tok, bos, eos, count, max_new, seed):
    from miniocr.eval.render import can_parse
    from miniocr.train.pretrain import wilson_interval
    device = next(model.parameters()).device
    rows = A.run_generate(model, torch.full((count,1),bos,device=device,dtype=torch.long),
                          max_new,eos,[bos],greedy=False,seed=seed).cpu().tolist()
    successes = ended = 0
    for row in rows:
        ids = row[1:]
        if eos in ids:
            ended += 1
            ids = ids[:ids.index(eos)]
        successes += bool(can_parse(tok.decode(ids)))
    return dict(renderer='matplotlib.mathtext (可编译率近似)', count=count, successes=successes,
                rate=successes/count, interval=wilson_interval(successes,count), ended_with_eos=ended)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokenizer',required=True); p.add_argument('--ckpt',required=True)
    p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    p.add_argument('--final-test',action='store_true'); p.add_argument('--candidate')
    p.add_argument('--register',metavar='REVIEW_JSON')
    p.add_argument('--samples',type=int,default=50); p.add_argument('--max-new',type=int,default=128)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--smoke',action='store_true',help='只用内置自写公式，不读真实语料；不得用于门槛')
    args=p.parse_args(argv)
    if args.samples<1 or args.max_new<1 or args.seed<0:
        p.error('samples/max-new 须正，seed 须非负')
    if args.final_test and (args.smoke or args.register or not args.candidate):
        p.error('--final-test 需要 candidate，不能和 smoke/register 混用')
    if not args.smoke and not corpus_available():
        p.error('缺本地真实语料；不下载，不把样例冒充正式评测')
    identity=candidate_identity(args)
    if args.register:
        if args.smoke:
            p.error('不能登记 smoke')
        registry_action(REGISTRY,args.candidate,identity)
        Path(args.register).write_text(json.dumps(identity,ensure_ascii=False,indent=2))
        return
    tok,bos,eos,vocab=load_tokenizer(args.tokenizer)
    ck=torch.load(args.ckpt,map_location='cpu',weights_only=False)
    if ck['cfg']['vocab_size'] != vocab or args.max_new>ck['cfg']['block_size']:
        p.error('词表不匹配或生成长度超过 block_size')
    model=A.get_gpt(ck['cfg']).to(args.device).eval(); model.load_state_dict(ck['state_dict'])
    # 显式开关处先持久化启动记录，随后才加载测试划分；崩溃也不得重评该候选。
    if args.final_test:
        registry_action(REGISTRY,args.candidate,identity,start=True)
    texts = SAMPLE_FORMULAS[:4] if args.smoke else standard_split()['test' if args.final_test else 'val'][:2000]
    result=dict(identity=identity,split='smoke' if args.smoke else 'test' if args.final_test else 'val',
                metrics=score(model,tok,texts,bos,eos,ck['cfg']['block_size']),
                compilation=compilation(model,tok,bos,eos,args.samples,args.max_new,args.seed),
                threshold='待校准')
    if args.final_test:
        dest=REGISTRY.with_name('final_test_'+args.candidate+'.json')
        with dest.open('x') as f:
            json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
