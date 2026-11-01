"""第 0 章：渲染后比较。一个教学用的渲染器，基于 matplotlib 的 mathtext。

重要说明：本机没有安装 LaTeX，正式评测用哪个渲染器还没有选定（见第 0 章 0.7.3 节）。
这里用 matplotlib 自带的 mathtext 代替，它只支持 LaTeX 数学模式的一个子集：
    - 不支持 \\begin{pmatrix} 这类环境（矩阵没法渲染）；
    - 不认识 \\le、\\tfrac 等一些合法命令，也不接受 \\frac12 这种省略花括号的写法；
    - 字体、间距和真正的 LaTeX 不同。
所以本模块的"可解析率"只是"可编译率"的近似，"渲染一致率"也只在 mathtext 能解析的公式上有意义。
正文 0.7、0.8 节会用例子说明这两个近似在哪里出错。

    can_parse        mathtext 能不能解析这个公式（"可编译率"的近似）
    render           公式 → 灰度图（numpy 数组，0 是白、255 是墨迹）；解析失败抛 ValueError
    compare_images   两张灰度图是否一致：二值化 → 裁掉空白 → 补齐到同样大小 → 数不同的像素
    compare_render   一对（参考，预测）的可解析性和渲染一致性，返回字典
    save_comparison  把参考和预测的渲染结果上下拼成一张图，方便人看
"""
import logging
import warnings

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl                                    # noqa: E402
from matplotlib import font_manager                         # noqa: E402
from matplotlib.mathtext import MathTextParser              # noqa: E402

# mathtext 画大号括号、根号时会打印 "No TeX to Unicode mapping" 之类的日志，和结果无关，关掉。
logging.getLogger("matplotlib.mathtext").setLevel(logging.ERROR)
logging.getLogger("matplotlib._mathtext").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# 能显示汉字的字体，按顺序找第一个装了的。找不到时汉字会被画成方框，并给出警告。
CJK_FONTS = ["Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC", "Hiragino Sans GB", "Heiti SC",
             "Microsoft YaHei", "SimHei", "WenQuanYi Zen Hei", "AR PL UMing CN"]
DPI = 100
_parser = None
_rc = None


def _rc_params():
    """渲染时用的字体设置。正体（\\text、\\mathrm、数字）用中文字体，斜体用 DejaVu Sans，缺的符号回退到 Computer Modern。
    这组设置本身就会影响图片：换一台装了不同字体的机器，同一个公式渲染出来的像素可能不同。"""
    global _rc
    if _rc is None:
        names = {f.name for f in font_manager.fontManager.ttflist}
        cjk = next((n for n in CJK_FONTS if n in names), None)
        if cjk is None:
            warnings.warn("没有找到中文字体，汉字会被画成方框；两个都含汉字的公式的渲染比较不可信")
            _rc = {}
        else:
            _rc = {"mathtext.fontset": "custom", "mathtext.rm": cjk, "mathtext.it": "DejaVu Sans:italic",
                   "mathtext.bf": "DejaVu Sans:bold", "mathtext.sf": "DejaVu Sans",
                   "mathtext.tt": "DejaVu Sans Mono", "mathtext.fallback": "cm"}
    return _rc


def _parse(latex, dpi):
    global _parser
    if _parser is None:
        _parser = MathTextParser("agg")
    with mpl.rc_context(_rc_params()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _parser.parse(f"${latex}$", dpi=dpi)


def can_parse(latex):
    """mathtext 能解析返回 True，否则 False。空字符串算不能解析。"""
    if not latex.strip():
        return False
    try:
        _parse(latex, DPI)
        return True
    except (ValueError, RecursionError):
        # 模型可能生成深度嵌套公式；mathtext 递归上限也属于不可解析，不能中断整批评测。
        return False


def render(latex, dpi=DPI):
    """公式 → uint8 灰度图，形状 (高, 宽)，0 是白、255 是墨迹。不能解析时抛 ValueError。"""
    if not latex.strip():
        raise ValueError("空公式")
    return np.asarray(_parse(latex, dpi).image).copy()


# >>> 练习 0.7：比较两张渲染图
def compare_images(a, b, threshold=128):
    """比较两张灰度图（0 是白、255 是墨迹），返回不一致的程度 diff，在 [0, 1] 之间。

    1. 二值化：像素值 ≥ threshold 的算墨迹（True），其余算空白。
    2. 裁剪：各自裁到"包住所有墨迹的最小矩形"。整张图都没有墨迹时，当作 0×0 的空图。
    3. 补齐：两张图都在右边和下边补空白，补到同样的高和宽（取两者的最大值）。左上角对齐。
    4. diff = 只在一张图里是墨迹的像素数 / 至少在一张图里是墨迹的像素数（异或 / 或）。
       两张图都没有墨迹时 diff = 0.0。

    diff = 0 表示两张图逐像素一致。例：同一个公式渲染两次，diff == 0.0。
    """
    def crop(img):
        ink = img >= threshold
        rows, cols = np.flatnonzero(ink.any(1)), np.flatnonzero(ink.any(0))
        if len(rows) == 0:
            return np.zeros((0, 0), dtype=bool)
        return ink[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    A, B = crop(a), crop(b)
    H, W = max(A.shape[0], B.shape[0]), max(A.shape[1], B.shape[1])
    PA = np.zeros((H, W), dtype=bool)
    PB = np.zeros((H, W), dtype=bool)
    PA[:A.shape[0], :A.shape[1]] = A
    PB[:B.shape[0], :B.shape[1]] = B
    union = (PA | PB).sum()
    return float((PA ^ PB).sum() / union) if union else 0.0
# <<<


def compare_render(ref, pred, tol=0.0):
    """返回 {"ref_compiles", "compiles", "image_match", "pixel_diff"}。
    compiles 是预测能否被 mathtext 解析；参考或预测有一个不能解析时，image_match 和 pixel_diff 都是 None（"无法判断"，
    不是"不一致"）。tol 是允许的 diff 上限，默认 0：逐像素一致才算一致。"""
    rc, pc = can_parse(ref), can_parse(pred)
    out = {"ref_compiles": rc, "compiles": pc, "image_match": None, "pixel_diff": None}
    if rc and pc:
        d = compare_images(render(ref), render(pred))
        out["pixel_diff"] = d
        out["image_match"] = d <= tol
    return out


def save_comparison(pairs, path, dpi=150):
    """pairs 是 [(标题, 参考, 预测)]。每对画成一行：左边参考、右边预测，不能解析的写"mathtext 无法解析"。"""
    import matplotlib.pyplot as plt
    n = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(8, 0.9 * n + 0.6), squeeze=False)
    with mpl.rc_context(_rc_params()):
        for i, (title, ref, pred) in enumerate(pairs):
            for j, s in enumerate((ref, pred)):
                ax = axes[i][j]
                ax.axis("off")
                try:
                    ax.imshow(255 - render(s, dpi), cmap="gray", vmin=0, vmax=255)
                except ValueError:
                    ax.text(0.5, 0.5, "mathtext 无法解析", ha="center", va="center", fontsize=10,
                            family=_rc_params().get("mathtext.rm", "sans-serif"))
                if i == 0:
                    ax.set_title(["参考答案", "模型预测"][j], family=_rc_params().get("mathtext.rm", "sans-serif"))
            axes[i][0].text(-0.05, 0.5, title, transform=axes[i][0].transAxes, ha="right", va="center",
                            fontsize=9, family=_rc_params().get("mathtext.rm", "sans-serif"))
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
