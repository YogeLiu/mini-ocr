"""第 21 章：边界框的基本运算、检测的评测、阅读顺序。

    box_iou            两组框两两之间的交并比 IoU
    nms                非极大值抑制：同一个目标的多个重叠框只留分数最高的
    average_precision  检测的标准指标 AP：按分数排序、贪心匹配、算精确率–召回率曲线下的面积
    xy_cut_classic     经典的递归投影切分：先横切、再竖切
    xy_cut             改进的 XY-cut：每次沿最宽的空隙一刀切成两半

框的约定与 render.py 相同：(x0, y0, x1, y1)，x1、y1 不包含（宽 = x1 − x0）。
"""
import torch


# >>> 练习 21.1：IoU
def box_iou(a, b):
    """a (N, 4)、b (M, 4) 的 float 张量，返回 (N, M)：第 i 个框与第 j 个框的 交集面积 / 并集面积。
    交集：左上角取两者的较大值、右下角取较小值，宽高小于 0 时截成 0。用广播一次算完，不写循环。"""
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])              # (N, M, 2)
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter)
# <<<


# >>> 练习 21.2：NMS
def nms(boxes, scores, iou_thr=0.5):
    """贪心的非极大值抑制。按分数从高到低：取出当前最高的框保留下来，删掉所有与它 IoU > iou_thr 的框，重复。
    返回保留下来的框的下标（LongTensor），按分数从高到低。"""
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
# <<<


# >>> 练习 21.3：AP
def average_precision(preds, gts, iou_thr=0.5):
    """preds：每张图一个 (boxes (P, 4), scores (P,))；gts：每张图一个 boxes (G, 4)。
    步骤：
      1. 把所有图的预测放在一起，按分数从高到低排序；
      2. 依次处理每个预测：在它那张图里找 IoU 最大的、还没被匹配过的真实框；IoU ≥ iou_thr 就算 TP（并把那个真实框标成已匹配），
         否则算 FP（包括和已经匹配过的框重复检测）；
      3. 第 k 个预测处的 精确率 = 前 k 个里 TP 的比例，召回率 = 前 k 个里 TP 数 / 真实框总数；
      4. 把精确率改成"从这里往后的最大值"（让曲线单调不增），AP = Σ (召回率的增量 × 该处的精确率)。
    返回 dict(ap=..., precision=最终精确率, recall=最终召回率)。没有预测时 ap = 0。"""
    n_gt = sum(len(g) for g in gts)
    items = [(float(s), img, k) for img, (b, sc) in enumerate(preds) for k, s in enumerate(sc)]
    items.sort(key=lambda t: -t[0])
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
        if ious[j] >= iou_thr:
            matched[img][j] = True
            tp.append(1)
        else:
            tp.append(0)
    if not tp or n_gt == 0:
        return dict(ap=0.0, precision=0.0, recall=0.0)
    tp = torch.tensor(tp, dtype=torch.float)
    ctp = tp.cumsum(0)
    prec = ctp / torch.arange(1, len(tp) + 1)
    rec = ctp / n_gt
    prec_mono = prec.flip(0).cummax(0).values.flip(0)
    drec = torch.cat([rec[:1], rec[1:] - rec[:-1]])
    return dict(ap=float((drec * prec_mono).sum()), precision=float(prec[-1]), recall=float(rec[-1]))
# <<<


def _groups(intervals, min_gap):
    """一维区间列表 → 按"空隙 ≥ min_gap"分成的组（每组是区间下标的列表），组按坐标从小到大。"""
    order = sorted(range(len(intervals)), key=lambda i: intervals[i][0])
    groups, end = [], None
    for i in order:
        lo, hi = intervals[i]
        if end is None or lo - end >= min_gap:
            groups.append([i])
            end = hi
        else:
            groups[-1].append(i)
            end = max(end, hi)
    return groups


def max_gap(intervals):
    """一维区间列表投影到坐标轴上，返回 (最宽的空隙宽度, 空隙的起点)。没有空隙（区间连成一片）时返回 (0, None)。"""
    order = sorted(intervals)
    best, at, end = 0, None, order[0][1]
    for lo, hi in order[1:]:
        if lo - end > best:
            best, at = lo - end, end
        end = max(end, hi)
    return best, at


def xy_cut_classic(boxes, min_gap=8):
    """经典的 XY-cut：先横切（按 y 投影，在所有 ≥ min_gap 的空隙处切开，从上到下），切不开再竖切（从左到右），递归。"""
    if len(boxes) <= 1:
        return list(range(len(boxes)))
    for axis in (1, 0):
        gs = _groups([(b[axis], b[axis + 2]) for b in boxes], min_gap)
        if len(gs) > 1:
            out = []
            for g in gs:
                out += [g[i] for i in xy_cut_classic([boxes[i] for i in g], min_gap)]
            return out
    return sorted(range(len(boxes)), key=lambda i: (boxes[i][1], boxes[i][0]))


# >>> 练习 21.4：XY-cut（最宽空隙优先）
def xy_cut(boxes, min_gap=8):
    """boxes：(x0, y0, x1, y1) 的列表。返回阅读顺序（下标列表）。递归：
      1. 用 max_gap 分别求 y 方向（横切）和 x 方向（竖切）最宽的空隙；
      2. 两者都 < min_gap：切不开，按 (y0, x0) 排序返回；
      3. 否则沿空隙更宽的方向（相等时横切）在那个空隙处一刀切成两组：坐标在空隙之前的一组在前（上面 / 左边），
         其余的在后；对两组分别递归，再把组内顺序换回原来的下标、拼起来。"""
    n = len(boxes)
    if n <= 1:
        return list(range(n))
    gy, cy = max_gap([(b[1], b[3]) for b in boxes])
    gx, cx = max_gap([(b[0], b[2]) for b in boxes])
    if max(gy, gx) < min_gap:
        return sorted(range(n), key=lambda i: (boxes[i][1], boxes[i][0]))
    axis, c = (1, cy) if gy >= gx else (0, cx)
    first = [i for i in range(n) if boxes[i][axis + 2] <= c]
    second = [i for i in range(n) if boxes[i][axis + 2] > c]
    return ([first[i] for i in xy_cut([boxes[j] for j in first], min_gap)]
            + [second[i] for i in xy_cut([boxes[j] for j in second], min_gap)])
# <<<
