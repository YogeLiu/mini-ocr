"""教学用小 ViT：显式 patch、二维位置表和双向多头注意力。

不调用现成 ViT/Transformer/MultiheadAttention；线性层与 autograd 使用 torch。
像素约定 NCHW，patch 内展平顺序 C,pH,pW；token 按网格行优先。
"""
import torch
from torch import nn
import torch.nn.functional as F


# >>> 练习 29.1：patchify 与卷积权重的对应
def patch_grid(height, width, patch, pad=False):
    """返回网格 (gh,gw)；默认拒绝余数，pad=True 表示右/下补白。"""
    if min(height, width, patch) < 1:
        raise ValueError("dimensions must be positive")
    if not pad and (height % patch or width % patch):
        raise ValueError("image dimensions must be divisible by patch")
    return ((height + patch - 1) // patch, (width + patch - 1) // patch)


def patchify(x, patch):
    """NCHW → N,(gh*gw),(C*p*p)，不隐式截掉边界。"""
    if x.ndim != 4:
        raise ValueError("expected NCHW")
    n, c, h, w = x.shape
    gh, gw = patch_grid(h, w, patch)
    return x.reshape(n, c, gh, patch, gw, patch).permute(0, 2, 4, 1, 3, 5).reshape(n, gh*gw, c*patch*patch)


def patch_embed(x, weight, bias=None, method="reshape"):
    """共享 D,C,p,p 权重，对比 reshape+linear 与 stride=p 卷积。"""
    if weight.ndim != 4 or weight.shape[-2] != weight.shape[-1]:
        raise ValueError("expected square D,C,p,p weights")
    p = weight.shape[-1]
    patch_grid(*x.shape[-2:], p)
    if method == "reshape":
        return F.linear(patchify(x, p), weight.flatten(1), bias)
    if method == "conv":
        return F.conv2d(x, weight, bias, stride=p).flatten(2).transpose(1, 2)
    raise ValueError("method must be reshape or conv")
# <<<


# >>> 练习 29.2：二维位置表插值，CLS 不参与空间插值
def interpolate_positions(pos, old_grid, new_grid, has_cls=True):
    """双三次、align_corners=False；显式传入长方形网格，不用 sqrt 猜形状。"""
    if len(old_grid) != 2 or len(new_grid) != 2 or min(*old_grid, *new_grid) < 1:
        raise ValueError("expected positive two-dimensional grids")
    prefix = int(has_cls)
    if pos.ndim != 3 or pos.shape[0] != 1 or pos.shape[1] != old_grid[0]*old_grid[1]+prefix:
        raise ValueError("position table does not match old grid")
    if tuple(old_grid) == tuple(new_grid):
        return pos
    spatial = pos[:, prefix:].reshape(1, *old_grid, pos.shape[-1]).permute(0, 3, 1, 2)
    spatial = F.interpolate(spatial, size=new_grid, mode="bicubic", align_corners=False)
    spatial = spatial.flatten(2).transpose(1, 2)
    return torch.cat((pos[:, :prefix], spatial), dim=1)
# <<<


# >>> 练习 29.3：双向注意力、pre-LN 残差与小 ViT
class VisionBlock(nn.Module):
    """pre-LN block；所有 patch/CLS 可相互注意，没有因果 mask。"""
    def __init__(self, width, heads=4):
        super().__init__()
        if width < 1 or heads < 1 or width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3*width), nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 4*width), nn.GELU(), nn.Linear(4*width, width))

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, t, 3, self.heads, d//self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        weights = ((q @ k.transpose(-2, -1)) / (d//self.heads)**.5).softmax(-1)
        y = (weights @ v).transpose(1, 2).reshape(b, t, d)
        x = x + self.proj(y)
        return x + self.mlp(self.norm2(x))


class TinyViT(nn.Module):
    """灰度符号分类；可变分辨率时插值位置表；输出未归一化 logits。"""
    def __init__(self, image_size=32, patch=4, width=64, depth=2, heads=4, classes=24):
        super().__init__()
        self.grid = patch_grid(image_size, image_size, patch)
        self.patch = patch
        self.weight = nn.Parameter(torch.empty(width, 1, patch, patch))
        self.bias = nn.Parameter(torch.zeros(width))
        nn.init.xavier_uniform_(self.weight.flatten(1))
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.pos = nn.Parameter(torch.empty(1, self.grid[0]*self.grid[1]+1, width))
        nn.init.normal_(self.pos, std=.02)
        self.blocks = nn.Sequential(*(VisionBlock(width, heads) for _ in range(depth)))
        self.norm, self.head = nn.LayerNorm(width), nn.Linear(width, classes)

    def forward(self, x):
        grid = patch_grid(*x.shape[-2:], self.patch)
        tokens = patch_embed(x, self.weight, self.bias)
        tokens = torch.cat((self.cls.expand(len(x), -1, -1), tokens), dim=1)
        tokens = tokens + interpolate_positions(self.pos, self.grid, grid)
        return self.head(self.norm(self.blocks(tokens))[:, 0])
# <<<
