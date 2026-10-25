"""第 12 章：卷积网络。

    SmallCNN     3 个"卷积块"（卷积 → BatchNorm → ReLU → 2×2 最大池化）+ 全局平均池化 + 线性层
    ResBlock     残差块：两个 3×3 卷积，输出与输入相加（第 11 章"归一化 + 残差"）
    DeepCNN      若干个卷积层堆起来，可选是否用残差块（用来比较深层网络有无残差的差别）
    random_translate   数据增强：每张图随机平移若干像素（空出来的地方补 0）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(c_in, c_out):
    """3×3 卷积（padding=1，边长不变）→ BatchNorm → ReLU。卷积后接 BatchNorm 时不需要偏置（会被减掉）。"""
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU())


# >>> 练习 12.5：一个小卷积网络
class SmallCNN(nn.Module):
    """输入 (B, 1, 32, 32)。结构（width = w）：
        conv_block(1, w)     → MaxPool2d(2)   32 → 16
        conv_block(w, 2w)    → MaxPool2d(2)   16 → 8
        conv_block(2w, 4w)   → MaxPool2d(2)    8 → 4
        全局平均池化（对每个通道的 4×4 求平均）→ (B, 4w) → nn.Linear(4w, n_classes)
    pool="flatten" 时不做全局平均池化，而是把 (4w, 4, 4) 拉平后接 nn.Linear(4w·16, n_classes)。
    """

    def __init__(self, n_classes=62, width=32, pool="avg"):
        super().__init__()
        w = width
        self.features = nn.Sequential(conv_block(1, w), nn.MaxPool2d(2),
                                      conv_block(w, 2 * w), nn.MaxPool2d(2),
                                      conv_block(2 * w, 4 * w), nn.MaxPool2d(2))
        self.pool = pool
        self.head = nn.Linear(4 * w * (1 if pool == "avg" else 16), n_classes)

    def forward(self, x):
        h = self.features(x)
        h = h.mean(dim=(2, 3)) if self.pool == "avg" else h.flatten(1)
        return self.head(h)
# <<<


# >>> 练习 12.6：残差块
class ResBlock(nn.Module):
    """y = ReLU(x + BN(conv(ReLU(BN(conv(x))))))，两个卷积都是 c→c、3×3、padding=1、无偏置。
    形状不变，所以可以直接相加。"""

    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        return F.relu(x + self.bn2(self.conv2(h)))
# <<<


class DeepCNN(nn.Module):
    """stem（1→c 的卷积块 + 2×2 池化，32 → 16）→ n_blocks 个"块"（每块两层卷积）→ 每隔 n_blocks//3 块做一次 2×2 池化
    → 全局平均池化 → 线性层。
    residual=True 时每块是 ResBlock；否则是两个普通的 conv_block（同样多的卷积层和参数，只差那个"+ x"）。"""

    def __init__(self, n_blocks=6, c=32, residual=True, n_classes=62):
        super().__init__()
        layers = [conv_block(1, c), nn.MaxPool2d(2)]
        every = max(1, n_blocks // 3)
        for i in range(n_blocks):
            layers.append(ResBlock(c) if residual else nn.Sequential(conv_block(c, c), conv_block(c, c)))
            if (i + 1) % every == 0 and i < 3 * every:
                layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(c, n_classes)

    def forward(self, x):
        return self.head(self.features(x).mean(dim=(2, 3)))


# >>> 练习 12.7：随机平移
def random_translate(x, max_shift, generator=None):
    """x (B, C, H, W)。每张图独立地在 [−max_shift, max_shift] 里随机取整数 (dy, dx)，把图平移，空出的地方补 0。
    返回新张量，不修改 x。做法：先四周补 max_shift 个 0，再为每张图裁出偏移后的 H×W 窗口。
    随机数用 torch.randint(..., generator=generator)（在 CPU 上生成），保证可重复。"""
    if max_shift == 0:
        return x.clone()
    B, C, H, W = x.shape
    m = max_shift
    xp = F.pad(x, (m, m, m, m))
    d = torch.randint(-m, m + 1, (B, 2), generator=generator)
    out = torch.empty_like(x)
    for b in range(B):
        dy, dx = int(d[b, 0]), int(d[b, 1])
        out[b] = xp[b, :, m - dy:m - dy + H, m - dx:m - dx + W]
    return out
# <<<
