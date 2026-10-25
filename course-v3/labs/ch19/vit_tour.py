"""第 19 章：视觉 Transformer。在 ShapesWorld 上训练 ViT，与 CNN 对比，并做位置编码、patch 大小、数据量、换分辨率的实验。
默认只用 CPU（GPU 被别的程序占用时也能跑）。

    python -m labs.ch19.vit_tour             # 全部 8 个部分（CPU 上约 1.5 小时，第 4 部分的 patch 4 最慢）
    python -m labs.ch19.vit_tour 3 6         # 只运行第 3、6 部分（第 6 部分用第 3 部分保存的模型）
    python -m labs.ch19.vit_tour --gpu       # 有空闲 GPU 时用 GPU
"""
import math
import os
import sys
import time

if "--gpu" not in sys.argv:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from miniocr.data.shapes import COLORS, draw_scene, sample_scene
from miniocr.models.cnn import conv_block
from miniocr.models.vit import PatchEmbed, ViT, ViTConfig, patchify
from miniocr.train.optim import warmup_cosine

DEV = "cuda" if "--gpu" in sys.argv and torch.cuda.is_available() else "cpu"
torch.set_num_threads(min(16, os.cpu_count() or 1))
OUT = "outputs/ch19"
os.makedirs(OUT, exist_ok=True)
COLOR_NAMES = list(COLORS)
STEPS, BS = 2000, 128
_C = {}


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ---------------------------------------------------------------------- 数据
def scenes(seed, n, H=64):
    """n 张 H×H 的场景图（uint8，(n, 3, H, H)）和两种标签：
    count = 物体个数 − 1（0–3），left = 最左边那个物体的颜色（0–3）。"""
    key = (seed, n, H)
    if key not in _C:
        rng = np.random.default_rng(seed)
        X = np.zeros((n, 3, H, H), np.uint8)
        count, left = np.zeros(n, np.int64), np.zeros(n, np.int64)
        for i in range(n):
            objs = sample_scene(rng, H, H, 1, 4)
            X[i] = draw_scene(objs, rng, H, H, noise=0.03).transpose(2, 0, 1)
            count[i] = len(objs) - 1
            left[i] = COLOR_NAMES.index(min(objs, key=lambda o: o["box"][0])["color"])
        _C[key] = (torch.from_numpy(X), {"count": torch.from_numpy(count), "left": torch.from_numpy(left)}, None)
    return _C[key][0], _C[key][1]


def to_float(xb):
    return (xb.to(DEV).float() / 127.5 - 1.0)


def train_data(task, n=16000):
    X, y = scenes(0, n)
    return X, y[task]


def test_data(task, H=64):
    X, y = scenes(1, 2000, H)
    return X, y[task]


# ---------------------------------------------------------------------- 模型
class CNN(nn.Module):
    """三个卷积块（3×3 卷积 → BN → ReLU → 2×2 池化），64 → 8；拉平后接线性层（保留位置信息）。"""

    def __init__(self, n_classes=4, w=32):
        super().__init__()
        self.features = nn.Sequential(conv_block(3, w), nn.MaxPool2d(2), conv_block(w, 2 * w), nn.MaxPool2d(2),
                                      conv_block(2 * w, 4 * w), nn.MaxPool2d(2))
        self.head = nn.Linear(4 * w * 64, n_classes)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def vit(**kw):
    cfg = dict(img_size=64, patch=8, d_model=128, n_layer=4, n_head=4, n_classes=4)
    cfg.update(kw)
    return ViT(ViTConfig(**cfg))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def accuracy(model, X, y, bs=500):
    model.eval()
    correct = 0
    for i in range(0, len(X), bs):
        correct += (model(to_float(X[i:i + bs])).argmax(-1).cpu() == y[i:i + bs]).sum().item()
    return correct / len(X)


def train(model, X, y, steps=STEPS, lr=1e-3, seed=0, log_every=0):
    """AdamW + 预热 + 余弦衰减，梯度裁剪 1.0。每步从训练集里随机抽 BS 张（有放回地按轮打乱）。返回用时（秒）。"""
    torch.manual_seed(seed)
    model.to(DEV).train()
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "pos" not in n and "cls" not in n]
    rest = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "pos" not in n and "cls" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": rest, "weight_decay": 0.0}], lr=lr)
    g = torch.Generator().manual_seed(seed)
    order, k = torch.randperm(len(X), generator=g), 0
    t0 = time.time()
    for step in range(steps):
        if k + BS > len(X):
            order, k = torch.randperm(len(X), generator=g), 0
        idx = order[k:k + BS]
        k += BS
        for gr in opt.param_groups:
            gr["lr"] = warmup_cosine(step, steps, 100, lr, lr * 0.05)
        loss = F.cross_entropy(model(to_float(X[idx])), y[idx].to(DEV))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log_every and (step + 1) % log_every == 0:
            print(f"      第 {step + 1:4d} 步  训练损失 {loss.item():.3f}")
    return time.time() - t0


_R = {}


def run(name, make, task, n=16000, key=None, **kw):
    """make() 造一个新模型（先 torch.manual_seed(0)，保证可重复）。key 相同的配置在同一次运行里只训练一次。"""
    if key is not None and (key, task, n) in _R:
        model, acc, sec = _R[(key, task, n)]
        print(f"    {name:<26} 参数 {n_params(model):>9,}  测试准确率 {acc:6.1%}  （与前面相同的配置，不重复训练）")
        return model
    torch.manual_seed(0)
    model = make()
    X, y = train_data(task, n)
    Xt, yt = test_data(task)
    sec = train(model, X, y, **kw)
    acc = accuracy(model, Xt, yt)
    print(f"    {name:<26} 参数 {n_params(model):>9,}  测试准确率 {acc:6.1%}  （{sec:4.0f} 秒）")
    if key is not None:
        _R[(key, task, n)] = (model, acc, sec)
    return model


# ---------------------------------------------------------------------- 1
def part1_patches():
    section("1. 把图切成 patch：图像变成一串 token")
    X, _ = scenes(1, 4)
    x = to_float(X[:1]).cpu()
    for p in (4, 8, 16):
        pt = patchify(x, p)
        print(f"  patch {p:2d}×{p:<2d}：64×64 的图 → {pt.shape[1]:4d} 个 token，每个 token 拉平后 3·{p}·{p} = {pt.shape[2]:4d} 维")
    torch.manual_seed(0)
    pe = PatchEmbed(3, 128, 8)
    y, hw = pe(x)
    lin = patchify(x, 8) @ pe.proj.weight.reshape(128, -1).T + pe.proj.bias
    print(f"  Conv2d(3, 128, kernel=8, stride=8) 的输出 {tuple(y.shape)}，网格 {hw}；"
          f"与'切块 + 线性层'的最大差 {(y - lin).abs().max().item():.1e}")
    print(f"  patch 嵌入的参数：3·8·8·128 + 128 = {sum(p.numel() for p in pe.parameters()):,}")
    print("  注意力的计算量与 token 数的平方成正比：")
    for p in (4, 8, 16):
        N = (64 // p) ** 2
        print(f"    patch {p:2d}：N = {N:4d}，注意力分数矩阵 N² = {N * N:>7,} 个元素（每头每层）")
    print("  真实模型：")
    for name, side, p, merge in (("ViT-B/16（224×224）", 224, 16, 1), ("SigLIP so400m/14（384×384）", 384, 14, 1),
                                 ("Qwen2.5-VL（patch 14，2×2 合并；1008×1008）", 1008, 14, 2),
                                 ("Qwen3-VL / Qwen3.5（patch 16，2×2 合并；1024×1024）", 1024, 16, 2)):
        n_patch = (side // p) ** 2
        print(f"    {name:<48} {n_patch:>5,} 个 patch，合并后 {n_patch // merge ** 2:>5,} 个视觉 token")


# ---------------------------------------------------------------------- 2
def part2_vit_vs_cnn():
    section("2. ViT 与 CNN：数物体、说出最左边物体的颜色")
    print(f"  训练 16000 张、测试 2000 张 64×64 的场景图（1–4 个物体）；每个模型 {STEPS} 步 × 批 {BS}")
    X, y = train_data("count")
    print(f"  两个任务的类别分布（训练集）：count {np.bincount(y.numpy(), minlength=4).tolist()}，"
          f"left {np.bincount(train_data('left')[1].numpy(), minlength=4).tolist()}")
    for task, desc in (("count", "数物体（4 类）"), ("left", "最左边物体的颜色（4 类）")):
        print(f"  任务：{desc}")
        run("CNN（宽 32，拉平）", CNN, task, key="cnn")
        run("ViT（patch 8，d 128，4 层）", vit, task, key="vit-learned-8")


# ---------------------------------------------------------------------- 3
def part3_position():
    section("3. 位置编码：没有位置信息，ViT 不知道谁在左边")
    print("  任务：最左边物体的颜色")
    for pos in ("none", "learned", "sincos", "rope2d"):
        m = run(f"位置 = {pos}", lambda: vit(pos=pos), "left", key=f"vit-{pos}-8")
        torch.save(m.state_dict(), f"{OUT}/vit_left_{pos}.pt")
    print("  任务：数物体（不需要知道位置）")
    run("位置 = none", lambda: vit(pos="none"), "count", key="vit-none-8")


# ---------------------------------------------------------------------- 4
def part4_patch_size():
    section("4. patch 大小：token 越多越细，也越贵")
    for p in (4, 8, 16):
        run(f"patch {p:2d}（{(64 // p) ** 2:3d} 个 token）", lambda: vit(patch=p), "left", key=f"vit-learned-{p}")


# ---------------------------------------------------------------------- 5
def part5_data():
    section("5. 数据量：归纳偏置少的 ViT 是不是更依赖数据")
    print(f"  每个模型都训练 {STEPS} 步（计算量相同），只改训练集大小；任务：最左边物体的颜色")
    for n in (1000, 4000, 16000):
        print(f"  训练集 {n} 张：")
        run("CNN", CNN, "left", n=n, key="cnn")
        run("ViT（learned）", vit, "left", n=n, key="vit-learned-8")


# ---------------------------------------------------------------------- 6
def part6_resolution():
    section("6. 换分辨率：在 64×64 上训练，在 96×96 上测试")
    print("  测试图是 96×96 的新场景（物体大小不变、画布更大）：patch 网格从 8×8 变成 12×12")
    Xt64, yt64 = test_data("left")
    Xt96, yt96 = test_data("left", H=96)
    for pos in ("learned", "sincos", "rope2d"):
        path = f"{OUT}/vit_left_{pos}.pt"
        if not os.path.exists(path):
            print(f"    （缺少 {path}，先运行第 3 部分）")
            return
        m = vit(pos=pos)
        m.load_state_dict(torch.load(path))
        m.to(DEV)
        how = {"learned": "8×8 的表双三次插值到 12×12", "sincos": "直接按 12×12 重新计算",
               "rope2d": "直接按 12×12 计算旋转角"}[pos]
        print(f"    {pos:<8} 64×64：{accuracy(m, Xt64, yt64):6.1%}   96×96：{accuracy(m, Xt96, yt96):6.1%}   （{how}）")
    print("  对照：同一批 96×96 测试图缩小到 64×64 后再测（物体变小了 1/3）：")
    small = F.interpolate(Xt96.float(), size=(64, 64), mode="area").round().clamp(0, 255).to(torch.uint8)
    for pos in ("learned", "rope2d"):
        m = vit(pos=pos)
        m.load_state_dict(torch.load(f"{OUT}/vit_left_{pos}.pt"))
        m.to(DEV)
        print(f"    {pos:<8} {accuracy(m, small, yt96):6.1%}")


# ---------------------------------------------------------------------- 7
def object_patches(objs, p=8, H=64):
    """每个物体覆盖的 patch 编号集合。"""
    g = H // p
    out = []
    for o in objs:
        x0, y0, x1, y1 = o["box"]
        out.append({(r * g + c) for r in range(y0 // p, (y1 - 1) // p + 1) for c in range(x0 // p, (x1 - 1) // p + 1)})
    return out


def part7_attention():
    section("7. CLS 在看哪里")
    path = f"{OUT}/vit_left_learned.pt"
    if not os.path.exists(path):
        print(f"  （缺少 {path}，先运行第 3 部分）")
        return
    m = vit()
    m.load_state_dict(torch.load(path))
    m.to(DEV).eval()
    pos = m.pos.detach().cpu()
    X, _ = test_data("left")
    with torch.no_grad():
        emb, _ = m.patch(to_float(X[:200]))
    print(f"  训练后：位置表的标准差 {pos.std().item():.3f}（初始 0.020），patch 嵌入输出的标准差 {emb.std().item():.3f}")
    P = F.normalize(pos, dim=-1).reshape(8, 8, -1)
    def sim(dr, dc):
        return (P[:8 - dr, :8 - dc] * P[dr:, dc:]).sum(-1).mean().item()
    rnd = (F.normalize(pos, dim=-1) @ F.normalize(pos, dim=-1).T)
    off = rnd[~torch.eye(64, dtype=torch.bool)].mean().item()
    print(f"  位置向量的余弦相似度：左右相邻 {sim(0, 1):.2f}，上下相邻 {sim(1, 0):.2f}，左右隔 4 格 {sim(0, 4):.2f}，"
          f"上下隔 4 格 {sim(4, 0):.2f}，任意两个位置平均 {off:.2f}")
    m.keep_attn()
    rng = np.random.default_rng(1)                                      # 与测试集相同的种子：重新生成场景以拿到物体框
    stats = {L: [0.0, 0.0, 0.0] for L in range(4)}                      # 最左物体、其他物体、背景（每个 patch 平均）
    shown, k3 = False, 0
    for i in range(500):
        objs = sample_scene(rng, 64, 64, 1, 4)
        img = draw_scene(objs, rng, 64, 64, noise=0.03)
        if len(objs) < 3:
            continue
        k3 += 1
        x = to_float(torch.from_numpy(img.transpose(2, 0, 1)[None].copy()))
        with torch.no_grad():
            pred = m(x).argmax(-1).item()
        cover = object_patches(objs)
        li = min(range(len(objs)), key=lambda j: objs[j]["box"][0])
        left_set = cover[li]
        other = set().union(*[c for j, c in enumerate(cover) if j != li]) - left_set
        bg = set(range(64)) - left_set - other
        for L, blk in enumerate(m.blocks):
            a = blk.attn.last_weights[0].mean(0)[0, 1:].cpu()           # 各头平均，CLS 对 64 个 patch 的权重
            for s, k in ((left_set, 0), (other, 1), (bg, 2)):
                stats[L][k] += a[list(s)].mean().item() if s else 0.0
        if not shown:
            shown = True
            print(f"  一张测试图：{', '.join(o['color'] + ' ' + o['shape'] + ' @x=' + str(o['box'][0]) for o in objs)}；"
                  f"预测 {COLOR_NAMES[pred]}")
            a = m.blocks[-1].attn.last_weights[0].mean(0)[0, 1:].reshape(8, 8).cpu()
            mark = {}
            for j, c in enumerate(cover):
                for q in c:
                    mark[q] = "L" if j == li else "o"
            print("  最后一层 CLS 对每个 patch 的注意力（×100，各头平均）    物体所在的 patch（L = 最左边的物体）")
            for r in range(8):
                row = " ".join(f"{100 * a[r, c].item():4.1f}" for c in range(8))
                print(f"    {row}      {' '.join(mark.get(r * 8 + c, '.') for c in range(8))}")
    print(f"  500 张测试图中物体 ≥ 3 个的 {k3} 张：CLS 对每个 patch 的平均注意力（×100；均匀分布时是 1.56）")
    print("     层      最左边的物体   其他物体   背景")
    for L in range(4):
        v = [100 * s / k3 for s in stats[L]]
        print(f"    第 {L + 1} 层     {v[0]:6.2f}       {v[1]:6.2f}     {v[2]:6.2f}")


# ---------------------------------------------------------------------- 8
def part8_no_position():
    section("8. 没有位置信息时，ViT 靠什么答对了一部分")
    paths = {pos: f"{OUT}/vit_left_{pos}.pt" for pos in ("none", "learned")}
    if not all(os.path.exists(p) for p in paths.values()):
        print("  （缺少第 3 部分保存的模型，先运行第 3 部分）")
        return
    Xt, yt = test_data("left")
    n_obj = scenes(1, 2000)[1]["count"] + 1
    rng = np.random.default_rng(1)                                      # 重新生成测试场景，拿到每个物体的颜色
    colors = []
    for _ in range(2000):
        objs = sample_scene(rng, 64, 64, 1, 4)
        draw_scene(objs, rng, 64, 64, noise=0.03)
        colors.append([COLOR_NAMES.index(o["color"]) for o in objs])
    # 不知道位置时最好的策略：猜出现次数最多的颜色；几种颜色并列最多时随机选一个，答对的概率是 1/并列数
    def best_guess(c, y):
        cnt = np.bincount(c, minlength=4)
        top = cnt == cnt.max()
        return top[y] / top.sum()
    best = np.array([best_guess(c, y) for c, y in zip(colors, yt.tolist())])
    preds = {}
    for pos, path in paths.items():
        m = vit(pos=pos)
        m.load_state_dict(torch.load(path))
        m.to(DEV).eval()
        with torch.no_grad():
            preds[pos] = torch.cat([m(to_float(Xt[i:i + 500])).argmax(-1).cpu() for i in range(0, 2000, 500)])
    print("     物体数   张数   无位置 ViT   learned ViT   '猜最多的颜色'的期望准确率")
    for k in (1, 2, 3, 4):
        sel = (n_obj == k)
        print(f"       {k}     {sel.sum().item():4d}     {(preds['none'][sel] == yt[sel]).float().mean():6.1%}"
              f"       {(preds['learned'][sel] == yt[sel]).float().mean():6.1%}         {best[sel.numpy()].mean():6.1%}")
    print(f"     全部    2000     {(preds['none'] == yt).float().mean():6.1%}       {(preds['learned'] == yt).float().mean():6.1%}"
          f"         {best.mean():6.1%}")


PARTS = {1: part1_patches, 2: part2_vit_vs_cnn, 3: part3_position, 4: part4_patch_size, 5: part5_data,
         6: part6_resolution, 7: part7_attention, 8: part8_no_position}

if __name__ == "__main__":
    want = [int(a) for a in sys.argv[1:] if a.isdigit()] or list(PARTS)
    for k in want:
        PARTS[k]()
    if want == list(PARTS):
        print("\n第 19 章的演示全部完成。")
