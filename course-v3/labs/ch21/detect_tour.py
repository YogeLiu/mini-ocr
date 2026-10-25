"""第 21 章：检测与版面分析。训练一个中心点检测器找文字行和版面区块，排阅读顺序，再接上第 13 章的 CRNN 做整页 OCR。
默认只用 CPU（GPU 被别的程序占用时也能跑）。

    python -m labs.ch21.detect_tour          # 全部 5 个部分（CPU 上约 1 小时；第 5 部分需要第 13 章保存的 outputs/ch13/crnn_lstm.pt）
    python -m labs.ch21.detect_tour 2 5      # 第 2 部分训练并保存检测器，第 5 部分读取它
    python -m labs.ch21.detect_tour --gpu    # 有空闲 GPU 时用 GPU
"""
import os
import sys
import time

if "--gpu" not in sys.argv:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from miniocr.boxes import average_precision, box_iou, nms, xy_cut, xy_cut_classic
from miniocr.data.page import sample_page
from miniocr.metrics import edit_distance
from miniocr.models.centernet import Detector, centernet_targets, decode, detector_loss
from miniocr.models.crnn import CRNN, ctc_decode, line_vocab
from miniocr.train.optim import warmup_cosine

DEV = "cuda" if "--gpu" in sys.argv and torch.cuda.is_available() else "cpu"
torch.set_num_threads(min(16, os.cpu_count() or 1))
OUT = "outputs/ch21"
os.makedirs(OUT, exist_ok=True)
CLASSES = ("line", "title", "text", "formula")       # 0：文字行（标题行和正文行）；1–3：三种版面区块
SCALE = 2                                            # 检测在半分辨率（192×288）上做
STEPS, BS = 2500, 8
_C = {}


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ---------------------------------------------------------------------- 数据
def page_targets(page):
    """一页的检测目标（半分辨率坐标）：所有标题行和正文行 → 类别 0；每个区块 → 类别 1–3。"""
    boxes, labels = [], []
    for b in page["blocks"]:
        if b["type"] != "formula":
            for l in b["lines"]:
                boxes.append(l["box"])
                labels.append(0)
        boxes.append(b["box"])
        labels.append(1 + ("title", "text", "formula").index(b["type"]))
    return torch.tensor(boxes, dtype=torch.float32) / SCALE, torch.tensor(labels)


def half(img):
    return np.array(Image.fromarray(img).resize((img.shape[1] // SCALE, img.shape[0] // SCALE), Image.BOX))


def pages(seed, n):
    key = (seed, n)
    if key not in _C:
        rng = np.random.default_rng(seed)
        ps = [sample_page(rng) for _ in range(n)]
        X = torch.from_numpy(np.stack([half(p["img"]) for p in ps]))[:, None]      # (n, 1, 192, 288) uint8
        T = [page_targets(p) for p in ps]
        _C[key] = (ps, X, T)
    return _C[key]


def to_float(xb):
    return (255.0 - xb.to(DEV).float()) / 255.0                                   # 墨迹 = 1


def batch_targets(T, idx, out_hw):
    ts = [centernet_targets(*T[i], len(CLASSES), out_hw) for i in idx]
    return tuple(torch.stack([t[k] for t in ts]).to(DEV) for k in range(4))


@torch.no_grad()
def detect(model, X, thr=0.3, bs=16):
    """返回每张图、每个类别的 (boxes, scores)，坐标已经换回全分辨率。"""
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        h, s, o = model(to_float(X[i:i + bs]))
        for b in range(len(h)):
            res = decode(h[b].cpu(), s[b].cpu(), o[b].cpu(), stride=4, thr=thr)
            out.append([(bx * SCALE, sc) for bx, sc in res])
    return out


def gt_boxes(page, cls):
    if cls == 0:
        bs = [l["box"] for b in page["blocks"] if b["type"] != "formula" for l in b["lines"]]
    else:
        bs = [b["box"] for b in page["blocks"] if b["type"] == CLASSES[cls]]
    return torch.tensor(bs, dtype=torch.float32).reshape(-1, 4)


# ---------------------------------------------------------------------- 1
def part1_boxes():
    section("1. 框、IoU、NMS")
    a = torch.tensor([[0.0, 0, 200, 16]])
    print("  一个 200×16 的文字行框，把它往下平移 k 个像素后与原框的 IoU：")
    for k in (1, 2, 4, 8):
        print(f"    下移 {k} 像素：IoU = {box_iou(a, a + torch.tensor([0.0, k, 0, k]))[0, 0].item():.3f}", end="")
        print(f"    （同样平移一个 100×100 的框：{box_iou(torch.tensor([[0.0, 0, 100, 100]]), torch.tensor([[0.0, k, 100, 100 + k]]))[0, 0].item():.3f}）")
    print("  细长的文字行对上下的误差极其敏感：偏 4 个像素 IoU 就只剩 0.6，所以文字检测常用 IoU ≥ 0.5 而不是更高的阈值")
    boxes = torch.tensor([[10.0, 10, 110, 30], [12, 11, 108, 31], [8, 9, 112, 29], [10, 50, 110, 70], [11, 52, 111, 71]])
    scores = torch.tensor([0.9, 0.75, 0.6, 0.85, 0.5])
    print(f"  5 个候选框（两行文字，各有几个重复的候选），分数 {[round(v, 2) for v in scores.tolist()]}")
    print(f"  IoU 矩阵：")
    for r in box_iou(boxes, boxes):
        print("    " + "  ".join(f"{v:.2f}" for v in r.tolist()))
    print(f"  NMS（阈值 0.5）保留：{nms(boxes, scores, 0.5).tolist()}（每行只留分数最高的那个）")


# ---------------------------------------------------------------------- 2
def part2_train():
    section("2. 训练中心点检测器")
    ps, X, T = pages(0, 3000)
    n_lines = sum((t[1] == 0).sum().item() for t in T)
    print(f"  训练 3000 页（半分辨率 192×288），平均每页 {n_lines / 3000:.1f} 个文字行、"
          f"{sum((t[1] > 0).sum().item() for t in T) / 3000:.1f} 个区块；单栏 {sum(p['columns'] == 1 for p in ps)} 页")
    torch.manual_seed(0)
    model = Detector(len(CLASSES)).to(DEV)
    print(f"  检测器 {sum(p.numel() for p in model.parameters()):,} 参数；输出 48×72 的网格（每格 = 原图 8×8 像素）；"
          f"{STEPS} 步 × 批 {BS}")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    for step in range(STEPS):
        idx = torch.randint(0, len(X), (BS,), generator=g).tolist()
        for gr in opt.param_groups:
            gr["lr"] = warmup_cosine(step, STEPS, 100, 2e-3, 1e-4)
        model.train()
        out = model(to_float(X[idx]))
        loss, parts = detector_loss(out, batch_targets(T, idx, out[0].shape[-2:]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if (step + 1) % 500 == 0:
            print(f"    第 {step + 1:4d} 步  损失 {loss.item():.3f}（热图 {parts['heat']:.3f}  宽高 {parts['size']:.3f}  "
                  f"偏移 {parts['offset']:.3f}）  {time.time() - t0:.0f} 秒")
    torch.save(model.state_dict(), f"{OUT}/detector.pt")
    evaluate(model)


def load_detector():
    if not os.path.exists(f"{OUT}/detector.pt"):
        sys.exit(f"  缺少 {OUT}/detector.pt：先运行 python -m labs.ch21.detect_tour 2")
    m = Detector(len(CLASSES))
    m.load_state_dict(torch.load(f"{OUT}/detector.pt"))
    return m.to(DEV)


def evaluate(model):
    ps, X, _ = pages(1, 200)
    dets = detect(model, X)
    print("  200 页测试集：")
    print("     类别        真实框   AP@0.5   AP@0.75  最终召回率  最终精确率")
    for c, name in enumerate(CLASSES):
        gts = [gt_boxes(p, c) for p in ps]
        preds = [d[c] for d in dets]
        r5, r75 = average_precision(preds, gts, 0.5), average_precision(preds, gts, 0.75)
        print(f"    {name:<10} {sum(len(g) for g in gts):6d}   {r5['ap']:6.1%}   {r75['ap']:6.1%}    {r5['recall']:6.1%}     "
              f"{r5['precision']:6.1%}")
    gts = [gt_boxes(p, 0) for p in ps]
    extra = 0
    for d in dets:
        b, s = d[0]
        extra += len(b) - len(nms(b, s, 0.5)) if len(b) else 0
    print(f"  热图峰值（3×3 最大池化）之后再做一次框的 NMS，又删掉了 {extra} 个重复的文字行框（共 "
          f"{sum(len(d[0][0]) for d in dets)} 个）")
    im = Image.fromarray(ps[0]["img"]).convert("RGB")
    dr = ImageDraw.Draw(im)
    colors = {0: (0, 90, 255), 1: (220, 0, 0), 2: (0, 160, 0), 3: (200, 0, 200)}
    for c in range(4):
        for bx in dets[0][c][0].tolist():
            dr.rectangle(bx, outline=colors[c], width=1 if c == 0 else 2)
    im.save(f"{OUT}/detect_example.png")
    print(f"  例图存到 {OUT}/detect_example.png（蓝：文字行，红：标题，绿：正文段落，紫：公式）")


# ---------------------------------------------------------------------- 3
def part3_errors():
    section("3. 错误分析：漏检和误检的是什么")
    model = load_detector()
    ps, X, _ = pages(1, 200)
    dets = detect(model, X)
    miss_w, hit_w, fp_iou = [], [], []
    for p, d in zip(ps, dets):
        g = gt_boxes(p, 0)
        b = d[0][0]
        if len(b) == 0:
            miss_w += (g[:, 2] - g[:, 0]).tolist()
            continue
        iou = box_iou(g, b)
        best = iou.max(1).values
        w = (g[:, 2] - g[:, 0])
        miss_w += w[best < 0.5].tolist()
        hit_w += w[best >= 0.5].tolist()
        fp_iou += iou.max(0).values[iou.max(0).values < 0.5].tolist()
    print(f"  文字行：漏检 {len(miss_w)} 个（IoU < 0.5），检到 {len(hit_w)} 个")
    if miss_w:
        print(f"    漏检的行平均宽 {np.mean(miss_w):.0f} 像素，检到的平均宽 {np.mean(hit_w):.0f} 像素；"
              f"漏检里宽度 < 60 像素的占 {np.mean(np.array(miss_w) < 60):.0%}（检到的里面占 {np.mean(np.array(hit_w) < 60):.0%}）")
    if fp_iou:
        fp = np.array(fp_iou)
        print(f"  误检 {len(fp)} 个：其中 {np.mean(fp > 0.1):.0%} 与某个真实行有部分重叠（IoU 0.1–0.5，多半是框的宽度估错了），"
              f"{np.mean(fp <= 0.1):.0%} 几乎不重叠")
    for thr in (0.1, 0.3, 0.5):
        d2 = detect(model, X, thr=thr)
        gts = [gt_boxes(p, 0) for p in ps]
        r = average_precision([d[0] for d in d2], gts, 0.5)
        print(f"    分数阈值 {thr}：召回率 {r['recall']:6.1%}  精确率 {r['precision']:6.1%}")


# ---------------------------------------------------------------------- 4
def part4_order():
    section("4. 阅读顺序")
    ps, _, _ = pages(1, 200)
    rng = np.random.default_rng(3)
    more = [sample_page(rng) for _ in range(800)]
    print("  用真实的区块框，1000 页中整页顺序完全正确的比例：")
    for cols in (1, 2):
        sub = [p for p in ps + more if p["columns"] == cols]
        res = []
        for f in (lambda b: sorted(range(len(b)), key=lambda i: (b[i][1], b[i][0])), xy_cut_classic, xy_cut):
            res.append(np.mean([f([x["box"] for x in p["blocks"]]) == list(range(len(p["blocks"]))) for p in sub]))
        print(f"    {'单栏' if cols == 1 else '双栏'}（{len(sub)} 页）：按 y 排序 {res[0]:6.1%}   经典 XY-cut {res[1]:6.1%}   "
              f"最宽空隙 XY-cut {res[2]:6.1%}")
    bad = None
    for p in more:
        b = [x["box"] for x in p["blocks"]]
        if p["columns"] == 2 and xy_cut(b) != list(range(len(b))):
            bad = p
            break
    if bad is not None:
        b = [x["box"] for x in bad["blocks"]]
        from miniocr.boxes import max_gap
        print(f"  一个失败的例子：正确顺序 {list(range(len(b)))}，XY-cut 给出 {xy_cut(b)}；"
              f"横向最宽空隙 {max_gap([(q[1], q[3]) for q in b])[0]} 像素，纵向（栏间）最宽空隙 {max_gap([(q[0], q[2]) for q in b])[0]} 像素")


# ---------------------------------------------------------------------- 5
V = line_vocab()


def crop_line(img, box):
    """按行框从整页裁出一行，变成 CRNN 的输入：高 32；框高超过 20 像素（标题）时按比例缩小；宽度补到 4 的倍数、至少 256。"""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    h = max(1, y1 - y0)
    strip = max(32, int(h * 1.6))
    cy = (y0 + y1) // 2
    top = max(0, cy - strip // 2)
    crop = img[top:top + strip, max(0, x0 - 4):min(img.shape[1], x1 + 4)]
    if crop.size == 0:
        return None
    if crop.shape[0] != 32:
        w = max(4, int(round(crop.shape[1] * 32 / crop.shape[0])))
        crop = np.array(Image.fromarray(crop).resize((w, 32), Image.BILINEAR))
    W = max(256, (crop.shape[1] + 3) // 4 * 4)
    canvas = np.full((32, W), 255, np.uint8)
    canvas[:, :crop.shape[1]] = crop
    return canvas


@torch.no_grad()
def recognize(crnn, crops):
    out = []
    for c in crops:
        x = torch.from_numpy((255.0 - c.astype(np.float32)) / 255.0)[None, None].to(DEV)
        out.append(ctc_decode(crnn(x), V)[0])
    return out


def assemble(lines, blocks, order_fn):
    """把检测到的行按阅读顺序排好：先对区块排序（order_fn），每一行归到中心落在其中的区块，块内按 y；
    不属于任何区块的行按 y 排在最后。"""
    border = order_fn([b.tolist() for b in blocks]) if len(blocks) else []
    used, out = set(), []
    for bi in border:
        x0, y0, x1, y1 = blocks[bi].tolist()
        mine = [i for i, l in enumerate(lines) if i not in used and x0 <= (l[0] + l[2]) / 2 <= x1 and y0 <= (l[1] + l[3]) / 2 <= y1]
        mine.sort(key=lambda i: lines[i][1])
        used.update(mine)
        out += mine
    out += sorted([i for i in range(len(lines)) if i not in used], key=lambda i: lines[i][1])
    return out


def page_text(page):
    return "\n".join(l["text"] for b in page["blocks"] if b["type"] != "formula" for l in b["lines"])


def part5_pipeline():
    section("5. 流水线 OCR：检测 → 排序 → 裁剪 → 识别")
    path = "outputs/ch13/crnn_lstm.pt"
    if not os.path.exists(path):
        print(f"  （缺少 {path}，先运行 python -m labs.ch13.ctc_tour 3）")
        return
    crnn = CRNN(len(V), rnn="lstm")
    crnn.load_state_dict(torch.load(path, map_location="cpu")["model"])
    crnn.to(DEV).eval()
    model = load_detector()
    ps, X, _ = pages(1, 200)
    ps, X = ps[:100], X[:100]
    dets = detect(model, X)
    rows = {"真实行框 + 真实顺序（只看识别）": [], "检测到的行 + 区块 XY-cut 排序": [], "检测到的行 + 按 y 排序": []}
    line_err, line_n = 0, 0
    t0 = time.time()
    for p, d in zip(ps, dets):
        ref = page_text(p)
        gl = [(l["box"], l["text"]) for b in p["blocks"] if b["type"] != "formula" for l in b["lines"]]
        hyp = recognize(crnn, [crop_line(p["img"], bx) for bx, _ in gl])
        for h, (_, t) in zip(hyp, gl):
            line_err += edit_distance(h, t)
            line_n += len(t)
        rows["真实行框 + 真实顺序（只看识别）"].append((edit_distance("\n".join(hyp), ref), len(ref)))
        lb, ls = d[0]
        keep = nms(lb, ls, 0.5) if len(lb) else torch.zeros(0, dtype=torch.long)
        lines = lb[keep]
        crops = [crop_line(p["img"], bx.tolist()) for bx in lines]
        ok = [i for i, c in enumerate(crops) if c is not None]
        lines, crops = lines[ok], [crops[i] for i in ok]
        texts = recognize(crnn, crops)
        blocks = torch.cat([d[c][0] for c in (1, 2, 3)]) if len(lines) else torch.zeros(0, 4)
        for name, fn in (("检测到的行 + 区块 XY-cut 排序", xy_cut),
                         ("检测到的行 + 按 y 排序", lambda b: sorted(range(len(b)), key=lambda i: (b[i][1], b[i][0])))):
            order = assemble(lines.tolist(), blocks, fn)
            rows[name].append((edit_distance("\n".join(texts[i] for i in order), ref), len(ref)))
    print(f"  100 页测试页（只算标题和正文，公式区块不识别），用时 {time.time() - t0:.0f} 秒")
    print(f"  按真实行框裁剪后逐行识别：字符错误率 {line_err / line_n:.2%}（第 13 章在单独的文字行上是 2.28%）")
    for name, v in rows.items():
        e, n = np.array(v).sum(0)
        print(f"    {name:<26} 整页 CER {e / n:6.2%}")
    print("  整页 CER 把三种错误都算在内：识别错的字符、漏检或多检的行、顺序错的行（顺序错一行，相当于删掉再插入一整行）")


PARTS = {1: part1_boxes, 2: part2_train, 3: part3_errors, 4: part4_order, 5: part5_pipeline}

if __name__ == "__main__":
    want = [int(a) for a in sys.argv[1:] if a.isdigit()] or list(PARTS)
    for k in want:
        PARTS[k]()
    if want == list(PARTS):
        print("\n第 21 章的演示全部完成。")
