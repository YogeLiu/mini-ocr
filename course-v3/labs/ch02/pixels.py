"""第 2 章：图像就是数组。逐段运行，对照正文阅读。

    python -m labs.ch02.pixels            # 打印所有实验结果，图片保存到 outputs/ch02/
"""
import os
import string

import numpy as np
from PIL import Image

from miniocr.data.imgops import (crop, draw_boxes, letterbox, normalize_glyph, resize_area, resize_bilinear,
                                 resize_nearest, to_float, to_gray)
from miniocr.data.latex import parse, to_latex
from miniocr.data.render import ink_box, render_formula, render_glyph, save_grid, to_ascii
from miniocr.data.shapes import COLORS, describe, draw_scene, sample_scene

OUT = "outputs/ch02"
CHARSET = string.digits + string.ascii_letters          # 62 个字符


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def part1_tiny_image():
    section("1. 一张 5×5 的图就是一个 5×5 的整数表")
    img = np.full((5, 5), 255, dtype=np.uint8)           # 全白
    img[1:4, 2] = 0                                       # 第 1–3 行、第 2 列涂黑：一条竖线
    img[1, 1] = 0                                         # 左上方一点：像数字 1 的"头"
    print("数组：\n", img)
    print("形状", img.shape, "类型", img.dtype)
    print("字符画：\n" + to_ascii(img))
    print("img[1, 2] =", img[1, 2], "（第 1 行第 2 列，黑）；img[0, 0] =", img[0, 0], "（白）")


def part2_glyph():
    section("2. 渲染一个字符，看它的数字")
    rng = np.random.default_rng(0)
    img, box = render_glyph("A", rng)
    print(f"形状 {img.shape}，类型 {img.dtype}，最小值 {img.min()}，最大值 {img.max()}")
    print(f"墨迹像素（<128）有 {(img < 128).sum()} 个，占 {(img < 128).mean():.1%}")
    print("墨迹的外接框 (x0, y0, x1, y1) =", box)
    print(to_ascii(img))
    c = crop(img, box)
    print(f"裁剪后形状 {c.shape}  ← (行数, 列数) = (y1-y0, x1-x0) = ({box[3] - box[1]}, {box[2] - box[0]})")


def part3_color():
    section("3. 彩色图：每个像素 3 个数")
    rng = np.random.default_rng(1)
    objs = sample_scene(rng)
    img = draw_scene(objs, rng)
    print("场景描述:", describe(objs))
    print("形状", img.shape, " ← (高, 宽, 通道)")
    for o in objs:
        x0, y0, x1, y1 = o["box"]
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        print(f"  {o['color']:6s} {o['shape']:8s} 中心像素 img[{cy},{cx}] = {tuple(int(v) for v in img[cy, cx])}"
              f"   定义的颜色 {COLORS[o['color']]}")
    print("背景像素 img[0,0] =", tuple(int(v) for v in img[0, 0]))
    g = to_gray(img)
    print("转灰度后形状", g.shape)
    for name, rgb in COLORS.items():
        r, gg, b = rgb
        print(f"  {name:6s} {rgb} → 灰度 0.299×{r}+0.587×{gg}+0.114×{b} = {0.299 * r + 0.587 * gg + 0.114 * b:.1f}")
    save_grid([img], f"{OUT}/scene.png", scale=6, cols=1)
    save_grid([np.stack([g] * 3, -1)], f"{OUT}/scene_gray.png", scale=6, cols=1)
    chans = [np.stack([img[..., k]] * 3, -1) for k in range(3)]
    save_grid(chans, f"{OUT}/scene_channels.png", scale=4, cols=3)
    print(f"已保存 {OUT}/scene.png、scene_gray.png、scene_channels.png（三张分别是 R、G、B 通道）")


def part4_layouts():
    section("4. 形状的几种约定：HWC、CHW、BCHW")
    rng = np.random.default_rng(2)
    imgs = np.stack([draw_scene(sample_scene(rng), rng) for _ in range(8)])
    print("8 张彩色图叠起来：", imgs.shape, " ← (B, H, W, C)")
    chw = imgs.transpose(0, 3, 1, 2)
    print("换成通道在前：    ", chw.shape, " ← (B, C, H, W)，PyTorch 的卷积要这种")
    print("同一个数：imgs[3, 10, 20, 1] =", imgs[3, 10, 20, 1], "  chw[3, 1, 10, 20] =", chw[3, 1, 10, 20])
    print(f"占用内存：{imgs.nbytes} 字节 = 8×64×64×3 个 uint8；转成 float32 后是 {imgs.astype(np.float32).nbytes} 字节（4 倍）")


def part5_resize():
    section("5. 缩放：最近邻与双线性")
    rng = np.random.default_rng(3)
    img, _ = render_glyph("8", rng)
    for size in (16, 8):
        print(f"--- 缩小到 {size}×{size}：最近邻（左）与 双线性（右）")
        a, b = resize_nearest(img, size, size), resize_bilinear(img, size, size)
        for ra, rb in zip(to_ascii(a).split("\n"), to_ascii(b).split("\n")):
            print(ra + "   |   " + rb)
    up = resize_bilinear(resize_nearest(img, 8, 8), 32, 32)
    print(f"先缩到 8×8 再放大回 32×32，与原图的平均像素差 {np.abs(up.astype(int) - img).mean():.1f}（信息丢了就回不来）")


def nearest_template_accuracy(size, align, n_train=40, n_test=20, seed=0):
    """最近模板分类器：每类用 n_train 张图求平均作为"模板"；测试图与哪个模板最像（像素差的平方和最小），就判成哪类。

    align=False：直接把 32×32 的图缩到 size×size 再比较；
    align=True ：先做字符归一化（裁出墨迹框、保持比例放进 size×size 的格子），再比较。
    """
    rng = np.random.default_rng(seed)

    def prep(img):
        x = normalize_glyph(img, size) if align else resize_area(img, size, size)
        return to_float(x, invert=True)

    templates, tests = [], []
    for ch in CHARSET:
        templates.append(np.mean([prep(render_glyph(ch, rng)[0]) for _ in range(n_train)], axis=0))
        tests.append([prep(render_glyph(ch, rng)[0]) for _ in range(n_test)])
    T = np.stack(templates).reshape(len(CHARSET), -1)                    # (62, size*size)
    correct, confusions = 0, {}
    for k, group in enumerate(tests):
        X = np.stack(group).reshape(len(group), -1)                      # (n_test, size*size)
        d = ((X[:, None, :] - T[None, :, :]) ** 2).sum(-1)               # (n_test, 62)：到每个模板的距离
        pred = d.argmin(1)
        correct += (pred == k).sum()
        for p in pred[pred != k]:
            pair = (CHARSET[k], CHARSET[p])
            confusions[pair] = confusions.get(pair, 0) + 1
    return correct / (len(CHARSET) * n_test), sorted(confusions.items(), key=lambda kv: -kv[1])[:6]


def part6_resolution():
    section("6. 实验：最近模板分类器（62 类，随机猜的准确率是 1/62 = 1.6%）")
    print("预测先写在纸上：直接比较像素，准确率大约多少？先对齐再比较呢？分辨率降到 4×4 呢？")
    for align in (False, True):
        print("--- " + ("先做字符归一化（对齐位置和大小）" if align else "直接比较像素（不对齐）"))
        for size in (32, 16, 8, 4):
            acc, conf = nearest_template_accuracy(size, align)
            pairs = ", ".join(f"{a}→{b}×{n}" for (a, b), n in conf)
            print(f"  {size:2d}×{size:<2d}（{size * size:4d} 个像素）准确率 {acc:6.1%}   最常见的混淆: {pairs}")


def part7_aspect():
    section("7. 长宽比：硬压成正方形 vs 保持比例补边（letterbox）")
    rng = np.random.default_rng(4)
    tree = parse("\\frac{a+1}{b}=x^{2}")
    img, boxes = render_formula(tree, rng)
    print("原图", img.shape, "公式", to_latex(tree))
    squash = resize_area(img, 48, 48)
    lb, scale, (px, py) = letterbox(img, 48, 48)
    print(f"letterbox: scale={scale:.3f}，左右补 {px}、上下补 {py} 像素")
    print("--- 硬压成 48×48（字被横向挤扁）")
    print(to_ascii(resize_nearest(squash, 24, 48)))
    print("--- letterbox 成 48×48（比例不变，但字变得很小）")
    print(to_ascii(resize_nearest(lb, 24, 48)))
    save_grid([img], f"{OUT}/formula.png", scale=4, cols=1)
    save_grid([draw_boxes(img, [b for _, b in boxes])], f"{OUT}/formula_boxes.png", scale=4, cols=1)
    print(f"已保存 {OUT}/formula.png 与 formula_boxes.png（画出了每个符号的框）")


def part8_normalize():
    section("8. 归一化：从 0–255 的整数到模型吃的小数")
    img = np.array([[0, 64, 128, 255]], dtype=np.uint8)
    fmt = lambda a: "[" + ", ".join(f"{v:6.3f}" for v in np.ravel(a)) + "]"
    print("原始        ", img.tolist())
    print("/255        ", fmt(to_float(img)))
    print("反色 1-x    ", fmt(to_float(img, invert=True)), " ← 墨迹=1，背景=0")
    print("(x-0.5)/0.5 ", fmt(to_float(img, mean=0.5, std=0.5)), " ← Qwen3.5 的做法，范围 [-1, 1]")
    m, s = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    rgb = np.array([[[255, 255, 255]]], dtype=np.uint8)
    print("CLIP 的 mean/std 下，纯白像素 (255,255,255) 变成", fmt(to_float(rgb, m, s)))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    part1_tiny_image()
    part2_glyph()
    part3_color()
    part4_layouts()
    part5_resize()
    part6_resolution()
    part7_aspect()
    part8_normalize()
    print("\n第 2 章的演示全部完成。")
