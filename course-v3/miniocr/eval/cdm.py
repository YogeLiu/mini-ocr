"""公式的"渲染后比较"：简化版 CDM（Character Detection Matching，Wang 等，2024）。

字符串比较的问题：x^2 和 x^{2} 渲染出来一模一样，字符串却差两个字符；
反过来，x^{2} 和 x_{2} 只差一个字符，渲染出来 2 的位置完全不同。
CDM 的思路是比较"渲染出来的符号和它们的位置"：

  1. 把预测和参考都渲染出来，得到每个符号和它的位置（真实的 CDM 在图像上用颜色区分每个字符、找出框；
     我们的渲染器排版时就知道每个符号的位置，直接拿来用）；
  2. 把两边的符号配对：同一个符号、位置相近才算匹配；
  3. F1 = 2 · 匹配数 / (预测的符号数 + 参考的符号数)。

真实的 CDM 用匈牙利算法配对，再用 RANSAC 估计一个整体的仿射变换、剔除位置不一致的配对。
我们用一个更简单、但思路相同的办法：两边的符号都按横坐标排序，做"带二维约束的最长公共子序列"——
横向允许整体错位（前面多一个或少一个符号，后面的都会平移），纵向（基线以上多高）和字号必须一致。
"""
from ..data.latex import LatexError, parse
from ..data.render import font, layout


def symbols(latex, size=16):
    """LaTeX → [(符号, x 中心, y 中心, 字号)]，按 x 排序。y 相对于整个公式的基线（向下为正）。
    分数线记作 "frac"，根号的勾记作 "sqrt"。解析失败时抛出 LatexError（"渲染失败"）。"""
    box = layout(parse(latex), size)
    out = []
    for c in box.cmds:
        if c[0] == "t":
            _, x, y, ch, s = c
            out.append((ch, x + font(s).getlength(ch) / 2, y - 0.35 * s, s))
        elif c[0] == "l":
            _, x0, x1, y = c
            out.append(("frac", (x0 + x1) / 2, y, size))
        else:
            (x, y), *_ = c[1]
            out.append(("sqrt", x, y, size))
    return sorted(out, key=lambda t: (t[1], t[2]))


def _same(a, b, tol):
    """两个符号能否配对：同一个符号、纵向位置相差不超过 tol 个字号、字号之比在 0.75–1.33 之间。"""
    return (a[0] == b[0] and abs(a[2] - b[2]) <= tol * max(a[3], b[3])
            and 0.75 <= a[3] / b[3] <= 1.33)


# >>> 练习 25.4：带二维约束的最长公共子序列
def match_count(pred, ref, tol=0.25):
    """pred、ref 是 symbols() 的输出。返回能配对的符号数：
    在两个按 x 排好序的序列上做最长公共子序列（第 13 章编辑距离的同类动态规划），"相等"换成 _same(a, b, tol)。
    L[i][j] = pred[:i] 和 ref[:j] 的最大配对数：
        L[i][j] = max(L[i-1][j], L[i][j-1], L[i-1][j-1] + 1（若 pred[i-1] 与 ref[j-1] 能配对）)
    """
    n, m = len(pred), len(ref)
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        for j in range(1, m + 1):
            cur[j] = max(prev[j], cur[j - 1])
            if _same(pred[i - 1], ref[j - 1], tol):
                cur[j] = max(cur[j], prev[j - 1] + 1)
        prev = cur
    return prev[m]
# <<<


def cdm(pred_latex, ref_latex, tol=0.25):
    """简化版 CDM 的 F1，在 [0, 1] 之间。预测无法解析（渲染失败）时得 0。"""
    ref = symbols(ref_latex)
    try:
        pred = symbols(pred_latex)
    except LatexError:
        return 0.0
    if not pred and not ref:
        return 1.0
    return 2 * match_count(pred, ref, tol) / (len(pred) + len(ref))
