"""第 3 章：公式的语法树、LaTeX 的生成、解析与归一化。

一个公式有两种表示：
    语法树（tree）   计算机内部的结构，例如 ("frac", 分子, 分母)
    LaTeX 字符串     人和数据集使用的写法，例如 "\\frac{a}{b}"

同一棵树可以写成多种等价的 LaTeX（x^2 与 x^{2}、\\frac{a}{b} 与 {a \\over b}……）。
归一化 = 解析成树，再按唯一的"规范写法"写回去：
    normalize(s) = to_latex(parse(s))

树的节点（都是元组，第一个元素是节点类型）：
    ("sym", "x")                      一个符号：字母、数字、括号
    ("op", "+")                       一个运算符：+ - =
    ("row", [节点, ...])              从左到右排成一行
    ("script", 底, 下标, 上标)        带上下标；下标或上标可以是 None；下标、上标都是 row
    ("frac", 分子, 分母)              分式；分子、分母都是 row
    ("sqrt", 内容)                    根号；内容是 row
"""
import numpy as np

VARS = "xyzabcnk"
DIGITS = "0123456789"
OPS = "+-="
PARENS = "()"
SYMBOLS = VARS + DIGITS + PARENS


class LatexError(ValueError):
    """解析失败：字符串不是本书公式文法里的合法 LaTeX。"""


# ====================================================================== 树 -> LaTeX
def to_latex(node, rng=None):
    """语法树 -> LaTeX 字符串。

    rng 为 None：输出唯一的规范写法（上下标总带花括号、先下标后上标、分式用 \\frac、没有空格）。
    rng 不为 None：随机选一种等价写法，模拟真实数据集里写法不统一的情况。
    """
    kind = node[0]
    if kind in ("sym", "op"):
        if rng is not None and kind == "op" and rng.random() < 0.2:
            return " " + node[1] + " "                       # 运算符两边加空格，含义不变
        return node[1]
    if kind == "row":
        return "".join(to_latex(c, rng) for c in node[1])
    if kind == "script":
        _, base, sub, sup = node
        b = to_latex(base, rng)
        if base[0] == "row":                                  # 底是一组（比如 (x+1)^2 的括号部分以外的情况）要加花括号
            b = "{" + b + "}"
        parts = []
        if sub is not None:
            parts.append("_" + _script_arg(sub, rng))
        if sup is not None:
            parts.append("^" + _script_arg(sup, rng))
        if rng is not None and len(parts) == 2 and rng.random() < 0.5:
            parts.reverse()                                   # x_{i}^{2} 与 x^{2}_{i} 等价
        return b + "".join(parts)
    if kind == "frac":
        n, d = to_latex(node[1], rng), to_latex(node[2], rng)
        if rng is not None and rng.random() < 0.15:
            return "{" + n + " \\over " + d + "}"
        return "\\frac{" + n + "}{" + d + "}"
    if kind == "sqrt":
        return "\\sqrt{" + to_latex(node[1], rng) + "}"
    raise ValueError(f"未知的节点类型: {kind}")


def _script_arg(row, rng):
    s = to_latex(row, rng)
    single = len(row[1]) == 1 and row[1][0][0] == "sym"
    if rng is not None and single and rng.random() < 0.5:
        return s                                              # 只有一个字符时花括号可省略：x^2
    return "{" + s + "}"


# ====================================================================== LaTeX -> 树
def tokenize(s):
    """把 LaTeX 字符串切成记号：命令（\\frac、\\sqrt、\\over）、单个字符；空格丢掉。

    >>> tokenize("\\frac{a}{b} + x^2")
    ['\\\\frac', '{', 'a', '}', '{', 'b', '}', '+', 'x', '^', '2']
    """
    toks, i = [], 0
    while i < len(s):
        c = s[i]
        if c == " ":
            i += 1
        elif c == "\\":
            j = i + 1
            while j < len(s) and s[j].isalpha():
                j += 1
            if j == i + 1:
                raise LatexError(f"第 {i} 个字符的反斜杠后面没有命令名")
            toks.append(s[i:j])
            i = j
        else:
            toks.append(c)
            i += 1
    return toks


class _Parser:
    """递归下降解析器：每一种语法结构对应一个函数，函数之间互相调用。"""

    def __init__(self, toks):
        self.toks, self.i = toks, 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self, expect=None):
        t = self.peek()
        if t is None:
            raise LatexError("公式意外结束")
        if expect is not None and t != expect:
            raise LatexError(f"期望 {expect!r}，实际是 {t!r}")
        self.i += 1
        return t

    def row(self, inside_group=False):
        """一行：若干个"项"，直到遇到 } 或字符串结束。

        组内遇到 \\over 时，返回一个临时的 ("over", 左半, 右半) 节点，由调用者把它变成分式。
        """
        items = []
        while self.peek() is not None and self.peek() != "}":
            if self.peek() == "\\over":
                if not inside_group:
                    raise LatexError("\\over 只能出现在花括号里")
                self.take()
                den = self.row(inside_group=True)
                if den[0] == "over":
                    raise LatexError("一组花括号里只能有一个 \\over")
                return ("over", ("row", items), den)
            items.extend(self.item())
        return ("row", items)

    # >>> 练习 3.4：解析一个"项"（底 + 可能的上下标）
    def item(self):
        """一个项：先解析"底"（atom），再看后面有没有 _ 或 ^。返回一个列表（花括号组会被展开成多个项）。"""
        base = self.atom()
        sub = sup = None
        while self.peek() in ("_", "^"):
            mark = self.take()
            arg = self.script_arg()
            if mark == "_":
                if sub is not None:
                    raise LatexError("同一个底有两个下标")
                sub = arg
            else:
                if sup is not None:
                    raise LatexError("同一个底有两个上标")
                sup = arg
        if sub is None and sup is None:
            if base[0] == "row":                     # 单独的 {a+b}：花括号不改变含义，展开
                return list(base[1])
            return [base]
        if base[0] == "row" and len(base[1]) == 1:
            base = base[1][0]                        # {x}^2 与 x^2 相同
        return [("script", base, sub, sup)]
    # <<<

    def atom(self):
        t = self.peek()
        if t is None:
            raise LatexError("公式意外结束")
        if t == "{":
            return self.group()
        if t == "\\frac":
            self.take()
            return ("frac", self.group(), self.group())
        if t == "\\sqrt":
            self.take()
            return ("sqrt", self.group())
        if t in OPS:
            self.take()
            return ("op", t)
        if len(t) == 1 and t in SYMBOLS:
            self.take()
            return ("sym", t)
        raise LatexError(f"不认识的记号 {t!r}")

    def group(self):
        """{ 一行 }。{a \\over b} 在这里变成分式。"""
        self.take("{")
        r = self.row(inside_group=True)
        self.take("}")
        if r[0] == "over":
            return ("row", [("frac", r[1], r[2])])
        return r

    def script_arg(self):
        """上下标的内容：要么是 {一组}，要么是紧跟着的单个字符。"""
        if self.peek() == "{":
            return self.group()
        t = self.take()
        if len(t) == 1 and t in SYMBOLS:
            return ("row", [("sym", t)])
        raise LatexError(f"上下标后面不能直接跟 {t!r}")


def parse(s):
    """LaTeX 字符串 -> 语法树（一个 row）。不合法时抛出 LatexError。"""
    p = _Parser(tokenize(s))
    tree = p.row()
    if p.peek() is not None:
        raise LatexError(f"多余的 {p.peek()!r}")
    return _clean(tree)


def _clean(node):
    """去掉解析时留下的内部标记，保证树的形状与生成器产生的树完全一致。"""
    kind = node[0]
    if kind == "row":
        out = []
        for c in node[1]:
            c = _clean(c)
            out.extend(c[1] if c[0] == "row" else [c])
        return ("row", out)
    if kind == "frac":
        return ("frac", _clean(node[1]), _clean(node[2]))
    if kind == "sqrt":
        return ("sqrt", _clean(node[1]))
    if kind == "script":
        base = _clean(node[1]) if node[1][0] != "sym" else node[1]
        return ("script", base,
                None if node[2] is None else _clean(node[2]),
                None if node[3] is None else _clean(node[3]))
    return node


# >>> 练习 3.5：归一化
def normalize(s):
    """归一化 = 解析，再写成规范写法。不合法的字符串抛出 LatexError。"""
    return to_latex(parse(s))
# <<<


def is_valid(s):
    try:
        parse(s)
        return True
    except LatexError:
        return False


# ====================================================================== 随机生成公式
def _sym(rng, digit_p=0.4):
    pool = DIGITS[1:] if rng.random() < digit_p else VARS
    return ("sym", str(rng.choice(list(pool))))


def _small_row(rng, level, depth):
    """分子、分母、根号里的内容：1–3 个符号，可能带一个运算符。"""
    items = [_term(rng, level, depth + 1)]
    if rng.random() < 0.35:
        items += [("op", str(rng.choice(["+", "-"]))), _term(rng, level, depth + 1)]
    return ("row", items)


def _term(rng, level, depth=0):
    """一个项。level 越高，可用的结构越多；depth 限制嵌套深度。"""
    r = rng.random()
    if level >= 2 and depth < 2 and r < 0.30:
        kind = rng.choice(["sup", "sub", "both"], p=[0.5, 0.3, 0.2])
        base = _sym(rng, 0.2)
        sub = ("row", [_sym(rng)]) if kind in ("sub", "both") else None
        sup = ("row", [_sym(rng)]) if kind in ("sup", "both") else None
        if level >= 4 and sup is not None and rng.random() < 0.3:
            sup = ("row", [_sym(rng), ("op", "+"), _sym(rng)])
        return ("script", base, sub, sup)
    if level >= 3 and depth < 1 and r < 0.50:
        return ("frac", _small_row(rng, level, depth), _small_row(rng, level, depth))
    if level >= 4 and depth < 1 and r < 0.62:
        return ("sqrt", _small_row(rng, level, depth))
    if level >= 4 and depth < 1 and r < 0.70:
        inner = [_sym(rng), ("op", str(rng.choice(["+", "-"]))), _sym(rng)]
        return ("row", [("sym", "(")] + inner + [("sym", ")")])
    if rng.random() < 0.3:                                   # 系数乘变量，例如 3x（系数不用 1）
        return ("row", [("sym", str(rng.choice(list("23456789")))), _sym(rng, 0.0)])
    return _sym(rng)


_REQUIRED = {1: set(), 2: {"script"}, 3: {"frac"}, 4: {"sqrt", "paren", "nested"}}


def gen_formula(rng, level=3, max_terms=3):
    """随机生成一个公式的语法树。

    level 1：只有平铺的字符和运算符（例如 3x+y=7）
    level 2：至少有一个上下标
    level 3：至少有一个分式（也可能有上下标）
    level 4：至少有根号、括号或"分式里套上下标"这类嵌套之一
    等级越高越难，评测时按等级分桶（第 25 章）。
    """
    while True:
        n = int(rng.integers(1, max_terms + 1))
        items, used_eq = [], False
        for k in range(n):
            if k > 0:
                op = str(rng.choice(["+", "-", "="]))
                if op == "=":
                    op, used_eq = ("+" if used_eq else "="), True
                items.append(("op", op))
            items.append(_term(rng, level))
        tree = _clean(("row", items))                        # 展开嵌套的 row，与解析结果的形状一致
        need = _REQUIRED[level]
        if not need or structure_of(tree) & need:
            return tree


def leaves(node):
    """按从左到右的顺序列出所有符号和运算符（不含结构），例如 \\frac{a}{b} -> ['a', 'b']。"""
    kind = node[0]
    if kind in ("sym", "op"):
        return [node[1]]
    if kind == "row":
        return [x for c in node[1] for x in leaves(c)]
    if kind == "script":
        out = leaves(node[1])
        for part in node[2:]:
            if part is not None:
                out += leaves(part)
        return out
    if kind == "frac":
        return leaves(node[1]) + leaves(node[2])
    if kind == "sqrt":
        return leaves(node[1])
    raise ValueError(kind)


def structure_of(node, _depth=0):
    """公式里出现了哪些结构，用于分桶：{'script', 'frac', 'sqrt', 'paren', 'nested'} 的子集。
    'nested' 表示结构里还套着结构（例如分母里有上下标）。"""
    kind, out = node[0], set()
    if kind == "sym" and node[1] in PARENS:
        out.add("paren")
    if kind == "row":
        for c in node[1]:
            out |= structure_of(c, _depth)
        return out
    if kind in ("script", "frac", "sqrt"):
        out.add(kind)
        if _depth >= 1:
            out.add("nested")
        for c in node[1:]:
            if c is not None and not isinstance(c, str):
                out |= structure_of(c, _depth + 1)
    return out


def print_tree(node, indent=0):
    """把语法树画成缩进的文本，便于肉眼检查（第 3 章）。"""
    pad = "  " * indent
    kind = node[0]
    if kind in ("sym", "op"):
        print(f"{pad}{kind} {node[1]!r}")
        return
    print(f"{pad}{kind}")
    if kind == "row":
        for c in node[1]:
            print_tree(c, indent + 1)
    elif kind == "script":
        print_tree(node[1], indent + 1)
        for name, part in (("下标", node[2]), ("上标", node[3])):
            if part is not None:
                print(f"{pad}  [{name}]")
                print_tree(part, indent + 2)
    elif kind == "frac":
        print(f"{pad}  [分子]"); print_tree(node[1], indent + 2)
        print(f"{pad}  [分母]"); print_tree(node[2], indent + 2)
    elif kind == "sqrt":
        print_tree(node[1], indent + 1)


def demo_rng(seed=0):
    return np.random.default_rng(seed)
