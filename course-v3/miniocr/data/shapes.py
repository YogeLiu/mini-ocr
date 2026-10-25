"""ShapesWorld：一个用几何图形搭出来的"玩具自然场景"，用来学通用 VLM 的任务（第 2、20–23、30–32 章）。

一张场景图里有 1 到 max_objects 个物体，每个物体有四个属性：
    形状 shape：circle（圆）、square（正方形）、triangle（三角形）
    颜色 color：red、green、blue、yellow
    大小 size ：small、large
    位置       ：边界框 box = (x0, y0, x1, y1)，坐标约定与 render.py 相同
物体之间不重叠，所以"有几个""最左边的是什么""某个物体在哪"都有唯一的标准答案——
这让描述、问答、定位都能自动打分（可验证奖励，第 31 章）。

图像是 uint8 数组，形状 (H, W, 3)，三个通道依次是红、绿、蓝（RGB）。
"""
import numpy as np
from PIL import Image, ImageDraw

SHAPES = ("circle", "square", "triangle")
COLORS = {"red": (220, 40, 40), "green": (40, 170, 60), "blue": (40, 80, 220), "yellow": (230, 200, 30)}
SIZES = {"small": (8, 11), "large": (14, 18)}            # 边长（像素）的取值范围


def _overlaps(a, b, margin=2):
    return not (a[2] + margin <= b[0] or b[2] + margin <= a[0] or a[3] + margin <= b[1] or b[3] + margin <= a[1])


def sample_scene(rng, H=64, W=64, min_objects=1, max_objects=4):
    """随机摆放若干个互不重叠的物体，返回物体列表（还没有画）。"""
    n = int(rng.integers(min_objects, max_objects + 1))
    objs = []
    for _ in range(200):                                   # 最多尝试 200 次，放不下就少放几个
        if len(objs) == n:
            break
        size = str(rng.choice(list(SIZES)))
        lo, hi = SIZES[size]
        s = int(rng.integers(lo, hi + 1))
        x0, y0 = int(rng.integers(1, W - s - 1)), int(rng.integers(1, H - s - 1))
        box = (x0, y0, x0 + s, y0 + s)
        if any(_overlaps(box, o["box"]) for o in objs):
            continue
        objs.append({"shape": str(rng.choice(SHAPES)), "color": str(rng.choice(list(COLORS))),
                     "size": size, "box": box})
    return objs


def draw_scene(objs, rng=None, H=64, W=64, background=None, noise=0.0):
    """把物体画成 RGB 图。background 为 None 时，用随机的浅色背景（模拟光照不同）。"""
    if background is None:
        background = (245, 245, 245) if rng is None else tuple(int(v) for v in rng.integers(215, 256, size=3))
    im = Image.new("RGB", (W, H), background)
    d = ImageDraw.Draw(im)
    for o in objs:
        x0, y0, x1, y1 = o["box"]
        c = COLORS[o["color"]]
        if o["shape"] == "circle":
            d.ellipse((x0, y0, x1 - 1, y1 - 1), fill=c)
        elif o["shape"] == "square":
            d.rectangle((x0, y0, x1 - 1, y1 - 1), fill=c)
        else:
            d.polygon([((x0 + x1 - 1) / 2, y0), (x0, y1 - 1), (x1 - 1, y1 - 1)], fill=c)
    arr = np.array(im, dtype=np.uint8)
    if noise > 0 and rng is not None:
        arr = np.clip(arr.astype(np.float32) + rng.normal(0, noise * 255, arr.shape), 0, 255).astype(np.uint8)
    return arr


def describe(objs):
    """规范的图像描述：物体按从左到右（框的左边界）排序，例如
    'a large red circle, a small blue square'。描述只有一种写法，便于评测。"""
    order = sorted(objs, key=lambda o: (o["box"][0], o["box"][1]))
    return ", ".join(f"a {o['size']} {o['color']} {o['shape']}" for o in order)


def center(o):
    x0, y0, x1, y1 = o["box"]
    return (x0 + x1) / 2, (y0 + y1) / 2
