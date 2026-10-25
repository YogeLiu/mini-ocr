"""A5 参考实现：偏好数据。正确的描述是 chosen，在上面制造一处幻觉得到 rejected（RLHF-V 式的"只差几个词"）。"""
from miniocr.data.shapes import COLORS, SHAPES, SIZES, describe


def hallucinated_caption(objs, rng, kind="substitute"):
    """返回 (rejected 描述, 被改动或加入的物体 (size, color, shape))。候选 = 图里没有的 (颜色, 形状)，
    颜色按 COLORS 的顺序、形状按 SHAPES 的顺序两层循环；先抽候选（rng.integers），substitute 再抽要换的物体。
    substitute：换掉那个物体的颜色和形状（大小、框不变）后 describe；add：再抽一个大小，接在描述末尾 ", a {size} {color} {shape}"。"""
    present = {(o["color"], o["shape"]) for o in objs}
    absent = [(c, s) for c in COLORS for s in SHAPES if (c, s) not in present]
    c, s = absent[int(rng.integers(len(absent)))]
    if kind == "substitute":
        k = int(rng.integers(len(objs)))
        new = [dict(o, color=c, shape=s) if i == k else o for i, o in enumerate(objs)]
        return describe(new), (objs[k]["size"], c, s)
    if kind == "add":
        z = list(SIZES)[int(rng.integers(len(SIZES)))]
        return describe(objs) + f", a {z} {c} {s}", (z, c, s)
    raise ValueError(kind)
