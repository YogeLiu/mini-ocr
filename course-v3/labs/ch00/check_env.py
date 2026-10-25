"""第 0 章：检查环境是否搭好，并看一眼全书要用的两类数据。

    python -m labs.ch00.check_env
"""
import os
import platform
import sys


def main():
    print(f"Python {sys.version.split()[0]}（{platform.system()} {platform.machine()}）")
    if sys.version_info < (3, 10):
        print("  ✗ Python 版本太低，需要 3.10 或以上")
    import numpy as np
    import PIL
    print(f"NumPy {np.__version__}，Pillow {PIL.__version__}")
    try:
        import torch
        print(f"PyTorch {torch.__version__}")
        print(f"  CUDA 可用: {torch.cuda.is_available()}" +
              (f"（{torch.cuda.get_device_name(0)}，显存 {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB）"
               if torch.cuda.is_available() else ""))
        print(f"  MPS 可用: {torch.backends.mps.is_available()}")
        x = torch.randn(256, 256)
        dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        y = (x.to(dev) @ x.to(dev)).cpu()
        print(f"  在 {dev} 上做了一次 256×256 的矩阵乘法，结果与 CPU 相差 {float((y - x @ x).abs().max()):.2e}")
    except ImportError:
        print("PyTorch 还没有安装（第 0–8 章用不到，第 9 章之前装好即可）")

    from miniocr.data.latex import to_latex
    from miniocr.data.render import sample_formula, save_grid
    from miniocr.data.shapes import describe, draw_scene, sample_scene
    rng = np.random.default_rng(0)
    os.makedirs("outputs/ch00", exist_ok=True)
    scenes, texts = [], []
    for _ in range(4):
        objs = sample_scene(rng)
        scenes.append(draw_scene(objs, rng))
        texts.append(describe(objs))
    save_grid(scenes, "outputs/ch00/shapesworld.png", scale=4, cols=4)
    forms, labels = [], []
    for level in (1, 2, 3, 4):
        tree, img, _ = sample_formula(rng, level)
        forms.append(img)
        labels.append(to_latex(tree))
    save_grid(forms, "outputs/ch00/minidoc_formulas.png", scale=3, cols=2)
    print("\nShapesWorld 的 4 张场景图已保存到 outputs/ch00/shapesworld.png，它们的描述是：")
    for t in texts:
        print("  ", t)
    print("MiniDoc 的 4 张公式图已保存到 outputs/ch00/minidoc_formulas.png，它们的 LaTeX 是：")
    for t in labels:
        print("  ", t)
    print("\n环境检查完成。")


if __name__ == "__main__":
    main()
