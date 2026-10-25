"""第 21 章：一个最小的"中心点"检测器（CenterNet 的思路）。

不预设锚框，而是在 stride 倍下采样的网格上预测三样东西：
    heat    (B, K, h, w)   每个类别一张热图：格子 (i, j) 是某个物体中心的概率（logits，过 sigmoid）
    size    (B, 2K, h, w)  在物体中心那一格预测它的 log 宽、log 高（以输入像素计），每个类别各一对
    offset  (B, 2, h, w)   中心落在格子里的小数部分（下采样丢掉的精度）

    Detector            编码器（5 次下采样到 1/32）+ 解码器（上采样回 1/4，带跳跃连接）+ 三个输出头
    centernet_targets   由真实框生成训练目标：高斯热图、中心处的宽高与偏移
    focal_loss          CenterNet 的热图损失：减轻"绝大多数格子都是背景"带来的不平衡
    detector_loss       focal_loss + 中心处宽高、偏移的 L1 损失
    decode              热图找峰值（3×3 最大池化）→ 取分数最高的 k 个 → 还原成框
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn import conv_block


class Detector(nn.Module):
    """输入 (B, 1, H, W)，H、W 是 32 的倍数；输出 stride = 4 的 (heat, size, offset)。

        stem   3×3 卷积、步长 2                         → 1/2,  宽 c0
        enc    4 级，每级 2×2 池化 + 两个卷积块          → 1/4, 1/8, 1/16, 1/32，宽 c1 … c4
        dec    3 级，每级 2 倍上采样、与同尺寸的编码器特征拼接（跳跃连接）、两个卷积块 → 1/16, 1/8, 1/4
        heads  在 1/4 的特征上：一个卷积块 + 1×1 卷积
    1/32 那一级的感受野有两百多像素，足够"看到"一整行文字，才能在行中心估计出整行的宽度。"""

    def __init__(self, n_classes, widths=(16, 32, 64, 96, 128)):
        super().__init__()
        self.K = n_classes
        c = widths
        self.stem = nn.Sequential(nn.Conv2d(1, c[0], 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c[0]), nn.ReLU())
        self.enc = nn.ModuleList([nn.Sequential(nn.MaxPool2d(2), conv_block(c[i], c[i + 1]), conv_block(c[i + 1], c[i + 1]))
                                  for i in range(4)])
        self.dec = nn.ModuleList([nn.Sequential(conv_block(c[i + 1] + c[i], c[i]), conv_block(c[i], c[i]))
                                  for i in (3, 2, 1)])
        head = lambda n_out: nn.Sequential(conv_block(c[1], 64), nn.Conv2d(64, n_out, 1))
        self.heat, self.size, self.offset = head(n_classes), head(2 * n_classes), head(2)
        nn.init.constant_(self.heat[-1].bias, -math.log((1 - 0.01) / 0.01))   # 初始时每格的物体概率 ≈ 0.01

    def forward(self, x):
        feats = []
        x = self.stem(x)
        for blk in self.enc:
            x = blk(x)
            feats.append(x)                                          # 1/4, 1/8, 1/16, 1/32
        y = feats[3]
        for blk, skip in zip(self.dec, (feats[2], feats[1], feats[0])):
            y = blk(torch.cat([F.interpolate(y, size=skip.shape[-2:], mode="nearest"), skip], 1))
        return self.heat(y), self.size(y), self.offset(y)


# >>> 练习 21.5：训练目标
def centernet_targets(boxes, labels, K, out_hw, stride=4):
    """一张图的目标。boxes (N, 4) 输入像素坐标，labels (N,) 类别编号。返回
        heat (K, h, w)    物体 n 的中心 c = ((x0+x1)/2, (y0+y1)/2) / stride，落在格子 (cy, cx) = floor(c)；
                          在它的类别通道上画一个各向异性的高斯 exp(−(dx²/2σx² + dy²/2σy²))，σx = max(0.5, 宽/stride/6)、
                          σy = max(0.5, 高/stride/6)，dx、dy 是格子到 (cx, cy) 的距离；多个物体重叠处取最大值；
                          中心那一格恰好是 1
        size (2K, h, w)   中心格的通道 (2k, 2k+1) = (log 宽, log 高)，其余为 0
        offset (2, h, w)  中心格 = c − floor(c)（x、y 两个分量）
        mask (K, h, w)    中心格为 True：只在这些格子上计算宽高和偏移的损失"""
    h, w = out_hw
    heat = torch.zeros(K, h, w)
    size = torch.zeros(2 * K, h, w)
    offset = torch.zeros(2, h, w)
    mask = torch.zeros(K, h, w, dtype=torch.bool)
    ys = torch.arange(h, dtype=torch.float32)[:, None]
    xs = torch.arange(w, dtype=torch.float32)[None, :]
    for (x0, y0, x1, y1), k in zip(boxes.tolist(), labels.tolist()):
        bw, bh = x1 - x0, y1 - y0
        c = torch.tensor([(x0 + x1) / 2 / stride, (y0 + y1) / 2 / stride])
        cx, cy = min(int(c[0]), w - 1), min(int(c[1]), h - 1)
        sx, sy = max(0.5, bw / stride / 6), max(0.5, bh / stride / 6)
        g = torch.exp(-((xs - cx) ** 2 / (2 * sx * sx) + (ys - cy) ** 2 / (2 * sy * sy)))
        heat[k] = torch.maximum(heat[k], g)
        size[2 * k, cy, cx], size[2 * k + 1, cy, cx] = math.log(bw), math.log(bh)
        offset[0, cy, cx], offset[1, cy, cx] = c[0] - cx, c[1] - cy
        mask[k, cy, cx] = True
    return heat, size, offset, mask
# <<<


def focal_loss(logits, target, alpha=2.0, beta=4.0):
    """CenterNet 的热图损失（"减罚"的 focal loss）。p = sigmoid(logits)，y = 高斯目标：
        y = 1 的格子：  −(1 − p)^α · log p
        其余格子：      −(1 − y)^β · p^α · log(1 − p)       离中心越近（y 越大）罚得越轻
    对所有格子求和，再除以 y = 1 的格子数。"""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    pos_loss = -((1 - p) ** alpha) * torch.log(p) * pos
    neg_loss = -((1 - target) ** beta) * (p ** alpha) * torch.log(1 - p) * (1 - pos)
    return (pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1)


def detector_loss(out, targets):
    """out = (heat, size, offset) 的模型输出；targets = 一批 centernet_targets 的结果堆叠成 (B, …) 的张量。"""
    heat, size, offset = out
    th, ts, to, tm = targets
    l_heat = focal_loss(heat, th)
    n = tm.sum().clamp(min=1)
    m2 = tm.repeat_interleave(2, dim=1)                              # (B, 2K, h, w)：每个类别的宽、高两个通道
    l_size = (F.l1_loss(size, ts, reduction="none") * m2).sum() / n
    m_any = tm.any(1, keepdim=True).expand_as(offset)
    l_off = (F.l1_loss(offset, to, reduction="none") * m_any).sum() / n
    return l_heat + l_size + l_off, dict(heat=l_heat.item(), size=l_size.item(), offset=l_off.item())


# >>> 练习 21.6：解码
@torch.no_grad()
def decode(heat, size, offset, stride=4, k=100, thr=0.3):
    """一张图的输出：heat (K, h, w) 的 logits，size (2K, h, w)，offset (2, h, w)。返回每个类别一个 (boxes (n, 4), scores (n,))。
    步骤：p = sigmoid(heat)；峰值 = 等于自己 3×3 邻域最大值的格子（F.max_pool2d，kernel 3，stride 1，padding 1）；
    在所有类别的峰值里取分数最高的 k 个、且分数 ≥ thr；对每个峰值 (k, i, j)：
        中心 = ((j + offset[0, i, j]) · stride, (i + offset[1, i, j]) · stride)，宽高 = exp(size[2k, i, j])、exp(size[2k+1, i, j])，
        框 = (cx − 宽/2, cy − 高/2, cx + 宽/2, cy + 高/2)。"""
    K, h, w = heat.shape
    p = torch.sigmoid(heat)
    peak = p * (F.max_pool2d(p[None], 3, stride=1, padding=1)[0] == p)
    scores, idx = peak.flatten().topk(min(k, peak.numel()))
    out = [([], []) for _ in range(K)]
    for s, ind in zip(scores.tolist(), idx.tolist()):
        if s < thr:
            break
        c, rem = divmod(ind, h * w)
        i, j = divmod(rem, w)
        cx = (j + offset[0, i, j].item()) * stride
        cy = (i + offset[1, i, j].item()) * stride
        bw, bh = math.exp(size[2 * c, i, j].item()), math.exp(size[2 * c + 1, i, j].item())
        out[c][0].append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
        out[c][1].append(s)
    return [(torch.tensor(b, dtype=torch.float32).reshape(-1, 4), torch.tensor(sc)) for b, sc in out]
# <<<
