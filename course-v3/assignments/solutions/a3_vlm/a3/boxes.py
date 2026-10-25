"""A3 参考实现：IoU、NMS、AP。框的约定 (x0, y0, x1, y1)，x1、y1 不包含。"""
import torch


def box_iou(a, b):
    """a (N, 4)、b (M, 4) → (N, M)。用广播一次算完。"""
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def nms(boxes, scores, iou_thr=0.5):
    """贪心：按分数从高到低，保留当前最高的框，删掉与它 IoU > iou_thr 的框。返回保留的下标（按分数从高到低）。"""
    order = scores.argsort(descending=True)
    keep = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        ious = box_iou(boxes[i][None], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return torch.tensor(keep, dtype=torch.long)


def average_precision(preds, gts, iou_thr=0.5):
    """preds：每张图 (boxes (P, 4), scores (P,))；gts：每张图 boxes (G, 4)。
    所有预测按分数排序，贪心匹配同一张图里 IoU 最大且未匹配的真实框；精确率改成"从这里往后的最大值"，
    AP = Σ 召回率增量 × 精确率。"""
    n_gt = sum(len(g) for g in gts)
    items = sorted(((float(s), img, k) for img, (b, sc) in enumerate(preds) for k, s in enumerate(sc)), key=lambda t: -t[0])
    matched = [torch.zeros(len(g), dtype=torch.bool) for g in gts]
    tp = []
    for _, img, k in items:
        g = gts[img]
        if len(g) == 0:
            tp.append(0)
            continue
        ious = box_iou(preds[img][0][k][None].float(), g.float())[0]
        ious[matched[img]] = -1
        j = int(ious.argmax())
        tp.append(int(ious[j] >= iou_thr))
        if tp[-1]:
            matched[img][j] = True
    if not tp or n_gt == 0:
        return 0.0
    tp = torch.tensor(tp, dtype=torch.float64)
    ctp = tp.cumsum(0)
    prec = ctp / torch.arange(1, len(tp) + 1)
    rec = ctp / n_gt
    prec = prec.flip(0).cummax(0).values.flip(0)
    drec = torch.cat([rec[:1], rec[1:] - rec[:-1]])
    return float((drec * prec).sum())
