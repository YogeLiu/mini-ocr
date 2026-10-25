"""第 30 章：偏好数据。对一张 ShapesWorld 的图，构造"正确的描述"和"几乎正确、但有一处幻觉"的描述。

RLHF-V（2023）一类的做法：让人把模型描述里的幻觉改掉，改之前的是 rejected、改之后的是 chosen。两者只差几个词，
偏好信号集中在出错的地方。我们用程序模拟这个过程：真值描述就是 chosen，在它上面制造一处幻觉得到 rejected。

    hallucinated_caption   substitute：把一个物体的颜色或形状换成图里没有的组合（长度几乎不变）
                           add：在描述末尾加一个图里没有的物体（变长）
"""
from .shapes import COLORS, SHAPES, SIZES, describe


# >>> 练习 30.4：制造一处幻觉
def hallucinated_caption(objs, rng, kind="substitute"):
    """objs：第 2 章 sample_scene 的物体列表。返回 (rejected 描述, 被改动或加入的物体 (size, color, shape))。
    substitute：随机挑一个物体（rng.integers），在"颜色、形状组合不在图里"的候选中随机选一个替换它的颜色和形状
                （大小不变），再用 describe 生成描述（物体顺序不变：只改颜色和形状，不改框）；
    add：随机选一个大小，以及一个不在图里的颜色、形状组合，接在 describe(objs) 的末尾（", a small red circle"）。
    候选的顺序：颜色按 COLORS 的顺序、形状按 SHAPES 的顺序两层循环，保证同一个 rng 得到同样的结果。"""
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
# <<<
