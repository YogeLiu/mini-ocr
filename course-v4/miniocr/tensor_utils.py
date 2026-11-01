"""第 1 章：张量、形状与广播。全书反复用到的几个小工具。

    broadcast_shape        按广播规则算出两个形状相加后的形状（手写一遍规则，和 torch.broadcast_shapes 对拍）
    pairwise_sq_dists      两组向量两两之间的距离平方：用广播，(N, 1, D) − (1, M, D) → (N, M, D) → (N, M)
    pairwise_sq_dists_mm   同一件事的展开写法 ||a||² − 2a·b + ||b||²，不产生 (N, M, D) 的中间张量
    one_hot                整数编号 → one-hot 向量：(...,) → (..., V)
    image_to_batch         一张 (H, W, C) 的 uint8 图片 → (1, C, H, W) 的 float32 张量，取值 [0, 1]
    batch_to_image         反过来：(B, C, H, W) 里取第 b 张 → (H, W, C) 的 uint8

所有函数只依赖 torch，形状注释里大写字母是维度的名字。
"""
import torch


# >>> 练习 1.1：广播后的形状
def broadcast_shape(a, b):
    """a、b 是两个形状（tuple 或 torch.Size），返回广播后的形状（tuple）。不能广播时抛 ValueError。

    规则（从最右边一维开始，两两对齐）：
        1. 维数少的那个，在左边补 1，直到两个形状一样长；
        2. 每一维：相等 → 取这个值；有一个是 1 → 取另一个；否则不能广播。
    例：(3, 1) 和 (4,) → 先补成 (3, 1) 和 (1, 4) → (3, 4)。
    """
    a, b = tuple(a), tuple(b)
    n = max(len(a), len(b))
    a = (1,) * (n - len(a)) + a
    b = (1,) * (n - len(b)) + b
    out = []
    for x, y in zip(a, b):
        if x == y or y == 1:
            out.append(x)
        elif x == 1:
            out.append(y)
        else:
            raise ValueError(f"形状 {a} 和 {b} 不能广播：有一维是 {x} 对 {y}")
    return tuple(out)
# <<<


# >>> 练习 1.2：两两距离（广播）
def pairwise_sq_dists(A, B):
    """A (N, D)、B (M, D) → (N, M)，第 [i, j] 个元素是 ||A[i] − B[j]||²。

        A[:, None, :]    (N, 1, D)
        B[None, :, :]    (1, M, D)
        相减              (N, M, D)   广播：第 0 维 N 对 1，第 1 维 1 对 M
        平方后沿 D 求和    (N, M)
    中间张量有 N·M·D 个元素，N、M 很大时很占内存，这时改用 pairwise_sq_dists_mm。
    """
    diff = A[:, None, :] - B[None, :, :]       # (N, M, D)
    return (diff ** 2).sum(-1)                 # (N, M)
# <<<


# >>> 练习 1.3：两两距离（矩阵乘法）
def pairwise_sq_dists_mm(A, B):
    """和 pairwise_sq_dists 结果相同，但用 ||a − b||² = ||a||² − 2 a·b + ||b||²。

        (A ** 2).sum(-1)[:, None]    (N, 1)
        (B ** 2).sum(-1)[None, :]    (1, M)
        A @ B.T                      (N, M)
    三项广播相加得到 (N, M)。不产生 (N, M, D) 的中间张量：A ** 2、B ** 2 各有 N·D、M·D 个元素，
    其余是 (N, M)，所以额外内存是 O(N·D + M·D + N·M) 量级。
    浮点舍入可能让本该是 0 的结果变成很小的负数，所以最后用 clamp_min(0) 截掉。
    """
    a2 = (A ** 2).sum(-1)[:, None]             # (N, 1)
    b2 = (B ** 2).sum(-1)[None, :]             # (1, M)
    return (a2 - 2 * (A @ B.T) + b2).clamp_min(0)   # (N, M)
# <<<


# >>> 练习 1.4：one-hot
def one_hot(idx, num_classes, dtype=torch.float32):
    """idx 是整数张量，形状任意 (...)，取值在 [0, num_classes)。返回 (..., num_classes)，
    对应位置为 1，其余为 0。编号越界时抛 ValueError。

    用广播实现：idx[..., None] 形状 (..., 1)，torch.arange(V) 形状 (V,)，
    二者比较 == 得到 (..., V) 的 bool 张量，再转成 dtype。
    """
    if idx.numel() and (idx.min() < 0 or idx.max() >= num_classes):
        raise ValueError(f"编号必须在 [0, {num_classes}) 之内，实际范围 [{idx.min().item()}, {idx.max().item()}]")
    classes = torch.arange(num_classes, device=idx.device)    # (V,)
    return (idx[..., None] == classes).to(dtype)               # (..., V)
# <<<


# >>> 练习 1.5：图片和批量张量互转
def image_to_batch(img):
    """img：(H, W, C) 的 uint8 张量（图片文件读出来的常见格式）。
    返回 (1, C, H, W) 的 float32 张量，取值除以 255 落在 [0, 1]（PyTorch 卷积层要的格式）。

        (H, W, C) --permute(2, 0, 1)--> (C, H, W) --[None]--> (1, C, H, W)
    einops 写法：rearrange(img, "h w c -> 1 c h w")。
    """
    if img.dtype != torch.uint8 or img.dim() != 3:
        raise ValueError(f"需要 (H, W, C) 的 uint8 张量，实际 {tuple(img.shape)} {img.dtype}")
    x = img.permute(2, 0, 1)[None]             # (1, C, H, W)，此时还是 uint8，且不连续
    x = x.float() / 255                        # 新张量，但沿用了原来的内存顺序（通道仍挨着放）
    return x.contiguous()                      # 复制成标准的连续布局


def batch_to_image(x, b=0):
    """x：(B, C, H, W) 的浮点张量，取值 [0, 1]。返回第 b 张图片，(H, W, C) 的 uint8。

        x[b] (C, H, W) --permute(1, 2, 0)--> (H, W, C)
    einops 写法：rearrange(x[b], "c h w -> h w c")。
    """
    img = x[b].permute(1, 2, 0)                # (H, W, C)
    return (img.clamp(0, 1) * 255).round().to(torch.uint8)
# <<<
