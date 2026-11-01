"""第 29 章：1 patch 对拍，2 token 预算，3 位置插值，4 公式网格，5 ViT/CNN。

从 course-v4 执行 python -m book.part4_vlm.ch29_vit.tour 1 2 --device cpu。
远程输出用 --out 指定到本章运行目录，不下载模型或数据。
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from miniocr.data.formula_images import load_formula_images
from miniocr.data.latex_corpus import _parquet_paths
from miniocr.data.symbols import make_dataset, group_split, count_shared_groups
from miniocr.device import pick_device, synchronize
from miniocr.nn.cnn import SymbolClassifier
from miniocr.nn.vit import patch_grid, patchify, patch_embed, interpolate_positions, TinyViT


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def csv_rows(path, rows):
    if rows:
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def part1(args):
    rows = []
    for shape, p in (((2, 3, 12, 20), 4), ((2, 1, 16, 24), 8), ((1, 2, 8, 12), 2)):
        originals = [torch.randn(*shape, device=args.device),
                     torch.randn(12, shape[1], p, p, device=args.device),
                     torch.randn(12, device=args.device)]
        inputs = [[v.clone().requires_grad_() for v in originals] for _ in range(2)]
        outputs = [patch_embed(*xs, method=m) for xs, m in zip(inputs, ('reshape', 'conv'))]
        dy = torch.randn_like(outputs[0])
        for out, xs in zip(outputs, inputs):
            (out * dy).sum().backward()
        row = dict(shape=list(shape), patch=p, tokens=outputs[0].shape[1],
                   forward_max_abs=float((outputs[0]-outputs[1]).abs().max().detach()))
        row.update({name: float((a.grad-b.grad).abs().max())
                    for name, a, b in zip(('dx_max_abs', 'dw_max_abs', 'db_max_abs'), *inputs)})
        assert max(row[k] for k in ('forward_max_abs', 'dx_max_abs', 'dw_max_abs', 'db_max_abs')) < 2e-4
        rows.append(row)
    return dict(dtype='float32', checks=rows, ordering='grid row-major; within patch C,pH,pW')


def part2(args):
    rows = []
    for h, w in ((32, 32), (64, 256), (128, 512), (65, 257)):
        for p in args.patches:
            gh, gw = patch_grid(h, w, p, pad=True)
            n = gh*gw
            rows.append(dict(height=h, width=w, patch=p, grid_h=gh, grid_w=gw,
                             pad_bottom=gh*p-h, pad_right=gw*p-w, spatial_tokens=n,
                             with_cls=n+1, attention_pairs=(n+1)**2,
                             score_mib_fp32_batch1_heads4=4*4*(n+1)**2/2**20))
    csv_rows(args.out / 'tokens.csv', rows)
    return dict(source='数学结论；右/下补白，计入 CLS；仅 attention score 张量，不是总显存', rows=rows)


def part3(args):
    old, new = (4, 6), (7, 10)
    pos = torch.randn(1, 25, 8, device=args.device, requires_grad=True)
    out = interpolate_positions(pos, old, new)
    out.square().mean().backward()
    assert torch.equal(pos[:, :1], out[:, :1]) and torch.isfinite(pos.grad).all()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for ax, value, grid in zip(axes, (pos, out), (old, new)):
        ax.imshow(value[0, 1:, 0].detach().cpu().reshape(grid), cmap='coolwarm')
        ax.set_title(f'position channel 0 / {grid}')
    fig.tight_layout()
    fig.savefig(args.out / 'position_interpolation.png', dpi=150)
    plt.close(fig)
    net = TinyViT(width=args.width, depth=args.depth, patch=args.patch).to(args.device)
    shapes = []
    for h, w in ((32, 32), (48, 64)):
        x = torch.randn(2, 1, h, w, device=args.device)
        shapes.append(dict(input=list(x.shape), output=list(net(x).shape),
                           spatial_tokens=(h//args.patch)*(w//args.patch)))
    return dict(dtype='float32', old_grid=old, new_grid=new, output_shape=list(out.shape),
                cls_max_abs=float((pos[:, :1]-out[:, :1]).abs().max().detach()),
                gradient_finite=bool(torch.isfinite(pos.grad).all()), model_shapes=shapes,
                caveat='插值使形状兼容，不保证分辨率改变后的准确率')


def patch_statistics(ink, p):
    """原像素网格补白；墨迹阈值只用于占用统计，不代表语义或 OCR 分数。"""
    h, w = ink.shape
    gh, gw = patch_grid(h, w, p, pad=True)
    x = F.pad(torch.as_tensor(ink)[None, None], (0, gw*p-w, 0, gh*p-h))
    patches = patchify(x, p)[0]
    occupied = (patches > .5).any(1)
    return dict(grid_h=gh, grid_w=gw, tokens=gh*gw, occupied=int(occupied.sum()),
                occupied_fraction=float(occupied.float().mean()))


def part4(args):
    print('读取完整 latex-v1 语料划分，随后取 val PNG', flush=True)
    examples = load_formula_images(args.n)
    if not examples:
        if args.require_data:
            raise FileNotFoundError('需要本地 cleaned_formulas PNG 缓存')
        return dict(status='skipped: no local PNG cache', n=0)
    rows, manifest = [], []
    for e in examples:
        manifest.append({k: v for k, v in e.items() if k != 'ink'})
        for p in args.patches:
            rows.append(dict(id=e['id'], key_sha256=e['key_sha256'], patch=p,
                             height=e['ink'].shape[0], width=e['ink'].shape[1],
                             fraction='\\frac' in e['formula'] or '\\dfrac' in e['formula'],
                             script='^' in e['formula'] or '_' in e['formula'],
                             **patch_statistics(e['ink'], p)))
    # 按标签挑展示样本；统计仍包含全部 n 张，不按视觉效果挑统计样本。
    selected = []
    for predicate in (lambda t: '\\frac' in t, lambda t: '^' in t, lambda t: '_' in t):
        e = next((v for v in examples if predicate(v['formula']) and v['id'] not in {s['id'] for s in selected}), None)
        if e is not None:
            selected.append(e)
    if not selected:
        selected = examples[:3]
    fig, axes = plt.subplots(len(selected), len(args.patches),
                             figsize=(4*len(args.patches), 3.5*len(selected)), squeeze=False)
    for i, e in enumerate(selected):
        h, w = e['ink'].shape
        for ax, p in zip(axes[i], args.patches):
            ax.imshow(e['ink'], cmap='gray_r', vmin=0, vmax=1)
            ax.set_xticks(np.arange(-.5, w, p), minor=True)
            ax.set_yticks(np.arange(-.5, h, p), minor=True)
            ax.grid(which='minor', color='tab:red', alpha=.6, linewidth=.5)
            ax.tick_params(which='minor', bottom=False, left=False)
            ax.set_title(f"{e['id'].split(':')[-1]} | p={p}, N={patch_statistics(e['ink'], p)['tokens']}")
            ax.set_xlabel('original pixel x'); ax.set_ylabel('original pixel y')
    fig.tight_layout()
    fig.savefig(args.out / 'formula_patches.png', dpi=180)
    plt.close(fig)
    write_json(args.out / 'image_manifest.json', manifest)
    csv_rows(args.out / 'patch_occupancy.csv', rows)
    summary = [dict(patch=p, n=len(examples),
                    mean_tokens=float(np.mean([r['tokens'] for r in rows if r['patch'] == p])),
                    mean_occupied_fraction=float(np.mean([r['occupied_fraction'] for r in rows if r['patch'] == p])))
               for p in args.patches]
    return dict(status='completed', n=len(examples), split='latex-v1 val, distinct normalized keys',
                source='OleehyO/latex-formulas cleaned_formulas', license='openrail',
                parquet_paths=[str(p) for p in _parquet_paths()], summary=summary,
                displayed_ids=[e['id'] for e in selected],
                caveat='原像素无缩放；frac/script 为公式标签，无符号框；占用率不是信息保留率或 OCR 准确率')


@torch.no_grad()
def evaluate(net, x, y, device, batch=256):
    net.eval()
    logits = torch.cat([net(x[i:i+batch].to(device)).cpu() for i in range(0, len(y), batch)])
    return dict(loss=float(F.cross_entropy(logits, y)), accuracy=float((logits.argmax(1)==y).float().mean()),
                correct=int((logits.argmax(1)==y).sum()), n=len(y)), logits


def shifted(x, pixels=4):
    """向右/下平移，零墨迹补白，禁止 torch.roll 的环绕。"""
    return F.pad(x, (pixels, 0, pixels, 0))[:, :, :x.shape[-2], :x.shape[-1]]


def train_subset(train, labels, groups, fraction):
    """每类取固定随机组前缀，低数据量集合嵌套于全量，不拆同来源组。"""
    rng = np.random.default_rng(29)
    selected = []
    for c in np.unique(labels):
        pool = np.unique(groups[train[labels[train] == c]])
        pool = rng.permutation(pool)
        selected.extend(pool[:max(1, int(len(pool)*fraction))])
    return train[np.isin(groups[train], selected)]


def build_model(kind, args):
    if kind == 'cnn':
        return SymbolClassifier('cnn', args.cnn_width)
    return TinyViT(width=args.width, depth=args.depth, patch=args.patch)


def part5(args):
    if (args.out / 'symbols.npz').exists():
        raise FileExistsError('use a fresh --out to preserve data and test protocol')
    print(f'controlled symbols sources/class={args.sources}, per_source={args.per_source}', flush=True)
    data = make_dataset(args.sources, args.per_source, seed=args.seed)
    tr, va, te = group_split(data['groups'], seed=0)
    assert count_shared_groups(data['groups'], tr, va, te) == 0
    np.savez_compressed(args.out / 'symbols.npz', **{k:data[k] for k in ('images', 'labels', 'groups')}, train=tr, val=va, test=te)
    digest = hashlib.sha256(data['images'].tobytes()+data['labels'].tobytes()+data['groups'].tobytes()).hexdigest()
    x, y = torch.from_numpy(data['images'])[:, None], torch.from_numpy(data['labels'])
    curves, results = [], []
    for fraction in args.fractions:
        indices = train_subset(tr, data['labels'], data['groups'], fraction)
        train_x, train_y = x[indices], y[indices]
        for seed in args.seeds:
            for kind in ('cnn', 'vit'):
                torch.manual_seed(seed)
                net = build_model(kind, args).to(args.device)
                parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
                opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                gen = torch.Generator().manual_seed(seed)
                tag = f'{kind}_f{fraction:g}_s{seed}'
                checkpoint = args.out / f'{tag}.pt'
                best, first_loss = float('inf'), None
                synchronize(args.device)
                start = time.perf_counter()
                if args.device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                for step in range(1, args.steps+1):
                    idx = torch.randint(len(indices), (args.batch_size,), generator=gen)
                    xb, yb = train_x[idx].to(args.device), train_y[idx].to(args.device)
                    net.train(); opt.zero_grad(set_to_none=True)
                    logits = net(xb)
                    assert logits.shape == (args.batch_size, 24) and 0 <= int(yb.min()) <= int(yb.max()) < 24
                    loss = F.cross_entropy(logits, yb)  # 每图一个有效类别；无 padding 标签、无因果 mask。
                    assert torch.isfinite(loss)
                    loss.backward()
                    if step == 1:
                        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
                        first_loss = float(loss.detach())
                        print(json.dumps(dict(smoke=tag, trainable=parameters, loss=first_loss,
                                              labels='all_valid', attention='bidirectional')), flush=True)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
                    opt.step()
                    if step == 1 or step % args.eval_every == 0 or step == args.steps:
                        val, _ = evaluate(net, x[va], y[va], args.device)
                        row = dict(tag=tag, model=kind, fraction=fraction, seed=seed, step=step,
                                   batch_loss=float(loss.detach()), val_loss=val['loss'], val_acc=val['accuracy'])
                        curves.append(row); print(json.dumps(row), flush=True)
                        if val['loss'] < best:
                            best, best_step = val['loss'], step
                            torch.save(dict(model=net.state_dict(), tag=tag, step=step, data_sha256=digest), checkpoint)
                synchronize(args.device)
                seconds = time.perf_counter()-start
                peak = torch.cuda.max_memory_allocated()/2**20 if args.device.type == 'cuda' else None
                saved = torch.load(checkpoint, map_location=args.device, weights_only=True)
                net.load_state_dict(saved['model'])
                val, _ = evaluate(net, x[va], y[va], args.device)
                train, _ = evaluate(net, train_x, train_y, args.device)
                shift, _ = evaluate(net, shifted(x[va]), y[va], args.device)
                resize = None
                if kind == 'vit':
                    resize, _ = evaluate(net, F.interpolate(x[va], size=(48,48), mode='bilinear', align_corners=False), y[va], args.device)
                results.append(dict(tag=tag, model=kind, fraction=fraction, seed=seed, n_train=len(indices),
                                    parameters=parameters, best_step=best_step, first_loss=first_loss,
                                    train=train, val=val, val_shift4=shift, val_resize48=resize,
                                    seconds=seconds, peak_allocated_mib=peak, checkpoint=checkpoint.name))
                write_json(args.out / 'progress.json', results)
                del net, opt
    csv_rows(args.out / 'learning_curves.csv', curves)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for r in results:
        rows = [v for v in curves if v['tag'] == r['tag']]
        axes[0].plot([v['step'] for v in rows], [v['val_loss'] for v in rows], label=r['tag'])
        axes[1].plot([v['step'] for v in rows], [v['val_acc'] for v in rows], label=r['tag'])
    for ax in axes:
        ax.set_xlabel('step'); ax.legend(fontsize=6); ax.grid(alpha=.2)
    axes[0].set_ylabel('validation CE'); axes[1].set_ylabel('validation accuracy')
    fig.tight_layout(); fig.savefig(args.out / 'learning_curves.png', dpi=150); plt.close(fig)
    result = dict(source='controlled ch05/ch28 symbols; 24 classes, font/pose/noise fixed across models',
                  data_sha256=digest, counts=dict(train=len(tr), val=len(va), test=len(te)), shared_groups=0,
                  selection='minimum validation CE per predeclared model/fraction/seed; equal steps, paired batches',
                  dtype='float32, TF32 disabled', configs=results, final_test=[])
    write_json(args.out / 'frozen_validation.json', result)
    # 所有配置和检查点冻结后才触碰测试图；每配置一次前向，同次计算区间。
    if args.final_test:
        for r in results:
            net = build_model(r['model'], args).to(args.device)
            net.load_state_dict(torch.load(args.out/r['checkpoint'], map_location=args.device, weights_only=True)['model'])
            metrics, logits = evaluate(net, x[te], y[te], args.device)
            ok = (logits.argmax(1) == y[te]).numpy()
            _, inv = np.unique(data['groups'][te], return_inverse=True)
            counts, successes = np.bincount(inv), np.bincount(inv, weights=ok)
            picks = np.random.default_rng(0).integers(len(counts), size=(1000, len(counts)))
            boot = successes[picks].sum(1)/counts[picks].sum(1)
            result['final_test'].append(dict(tag=r['tag'], **metrics,
                group_bootstrap_95=np.quantile(boot, [.025,.975]).tolist()))
            del net
        write_json(args.out / 'final_test.json', result['final_test'])
    write_json(args.out / 'training.json', result)
    return result


PARTS = {1:part1, 2:part2, 3:part3, 4:part4, 5:part5}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parts', nargs='*', type=int)
    parser.add_argument('--device', choices=('cpu','mps','cuda'), default='cpu')
    parser.add_argument('--n', type=int, default=20, help='real validation PNG count')
    parser.add_argument('--out', type=Path, default=Path('outputs/book/ch29/local'))
    parser.add_argument('--seed', type=int, default=29, help='data/mechanism seed')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--sources', type=int, default=8)
    parser.add_argument('--per-source', type=int, default=2)
    parser.add_argument('--steps', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--cnn-width', type=int, default=16)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--patch', type=int, choices=(2,4,8,16), default=4)
    parser.add_argument('--patches', type=int, nargs='+', default=[4,8,16,32])
    parser.add_argument('--fractions', type=float, nargs='+', default=[.25,1.])
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--weight-decay', type=float, default=.0001)
    parser.add_argument('--eval-every', type=int, default=40)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--require-data', action='store_true')
    parser.add_argument('--final-test', action='store_true', help='frozen final settings only, never smoke/tuning')
    args = parser.parse_args(argv)
    if any(p not in PARTS for p in args.parts) or len(set(args.parts)) != len(args.parts):
        parser.error('parts must be unique integers 1..5')
    if min(args.n,args.sources,args.per_source,args.steps,args.batch_size,args.width,args.cnn_width,args.depth,args.eval_every,args.threads,*args.patches) < 1:
        parser.error('sizes must be positive')
    if args.width % 4 or args.lr <= 0 or args.weight_decay < 0:
        parser.error('width must be divisible by 4; invalid optimizer')
    if any(not 0 < f <= 1 for f in args.fractions) or len(set(args.fractions)) != len(args.fractions) or len(set(args.seeds)) != len(args.seeds):
        parser.error('invalid fractions or duplicate seeds')
    args.out.mkdir(parents=True, exist_ok=True)
    selected = args.parts or list(PARTS)
    if (args.out/'run.json').exists() or any((args.out/f'part{p}.json').exists() for p in selected):
        parser.error('use a fresh --out')
    torch.set_num_threads(args.threads)
    args.device = pick_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.2)  # 48GB 卡本进程分配上限约 10GB。
    settings = {k:str(v) if isinstance(v,(Path,torch.device)) else v for k,v in vars(args).items()}
    meta = dict(start_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                torch=torch.__version__, numpy=np.__version__, settings=settings,
                gpu=torch.cuda.get_device_name() if args.device.type == 'cuda' else None,
                source='本项目观察；float32；部分 2 为数学结论')
    write_json(args.out/'started.json', meta)
    print(json.dumps(meta, ensure_ascii=False), flush=True)
    results = {}
    for part in selected:
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        synchronize(args.device); begin = time.perf_counter()
        print(f'部分 {part}', flush=True)
        result = PARTS[part](args)
        synchronize(args.device)
        result['part_wall_seconds'] = time.perf_counter()-begin
        results[str(part)] = result
        write_json(args.out/f'part{part}.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    meta['end_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(args.out/'run.json', dict(metadata=meta, results=results))
    print(f"DONE out={args.out} end={meta['end_utc']}", flush=True)


if __name__ == '__main__':
    main()
