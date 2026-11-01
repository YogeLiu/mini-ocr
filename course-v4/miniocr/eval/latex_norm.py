"""第 0 章：LaTeX 的分词与规范化。

同一个公式可以写成很多种 LaTeX：x^2 和 x^{2}、\\dfrac{1}{2} 和 \\frac12、\\left( 和 (……
直接比较字符串，会把"写法不同"误判成"识别错了"。规范化就是把这些写法统一成一种，再去比较。

    tokenize_latex     LaTeX 字符串 → token 列表（命令算一个 token，汉字一个字一个 token，空白丢掉）
    apply_token_rules  逐个 token 的规则：同义命令换成同一个名字，删掉只影响外观的命令（间距、括号尺寸）
    parse_latex        token 列表 → 一棵小语法树（只认花括号分组、上下标、带参数的命令）
    serialize          语法树 → 规范写法：参数一律带花括号、先下标后上标、去掉多余的花括号
    normalize_latex    以上四步串起来，返回用空格隔开的规范 token 串；解析失败时只做逐 token 规则
    normalized_tokens  同上，返回 token 列表（按 token 计算编辑距离时用）

规范写法的例子：
    normalize_latex("x^2+y^2")            == "x ^ { 2 } + y ^ { 2 }"
    normalize_latex("\\dfrac12")          == "\\frac { 1 } { 2 }"
    normalize_latex("\\left( a \\right)") == "( a )"
    normalize_latex("x^{2}_{i}")          == "x _ { i } ^ { 2 }"

这只是一个教学用的规范化器，只处理最常见的几类写法差异，盲区见正文 0.4.6 节。
"""

# ---------------------------------------------------------------------- 规则表
# 同义命令：左边换成右边。只收录"排出来的样子和含义都一样"的写法。
SYNONYMS = {
    "\\dfrac": "\\frac", "\\tfrac": "\\frac",
    "\\le": "\\leq", "\\ge": "\\geq", "\\ne": "\\neq",
    "\\to": "\\rightarrow", "\\gets": "\\leftarrow",
    "\\lbrace": "\\{", "\\rbrace": "\\}",
    "\\vert": "|", "\\lvert": "|", "\\rvert": "|",
    "\\land": "\\wedge", "\\lor": "\\vee",
}

# 只改变间距的命令：直接删掉。"~" 是不换行空格，也只影响间距。
SPACING = {"\\,", "\\;", "\\:", "\\!", "\\ ", "~", "\\quad", "\\qquad", "\\enspace", "\\thinspace"}

# 只改变括号大小或显示样式的命令：删掉命令本身，保留后面的括号。
SIZING = {"\\left", "\\right", "\\middle",
          "\\big", "\\Big", "\\bigg", "\\Bigg",
          "\\bigl", "\\Bigl", "\\biggl", "\\Biggl",
          "\\bigr", "\\Bigr", "\\biggr", "\\Biggr",
          "\\bigm", "\\Bigm", "\\biggm", "\\Biggm",
          "\\displaystyle", "\\textstyle", "\\limits", "\\nolimits"}

# 带参数的命令和它们的参数个数。\sqrt 还可以带一个方括号里的可选参数（开几次方）。
ARITY = {
    "\\frac": 2, "\\binom": 2,
    "\\sqrt": 1,
    "\\text": 1, "\\mathrm": 1, "\\mathbf": 1, "\\mathit": 1, "\\mathbb": 1, "\\mathcal": 1,
    "\\mathsf": 1, "\\mathtt": 1, "\\boldsymbol": 1, "\\operatorname": 1,
    "\\overline": 1, "\\underline": 1, "\\hat": 1, "\\bar": 1, "\\vec": 1, "\\tilde": 1,
    "\\dot": 1, "\\ddot": 1, "\\widehat": 1, "\\widetilde": 1, "\\overrightarrow": 1,
}


class LatexSyntaxError(ValueError):
    """解析失败：花括号不配对、上下标或命令缺参数等。"""


# >>> 练习 0.1：LaTeX 分词
def tokenize_latex(s):
    """把 LaTeX 字符串切成 token 列表。规则（按顺序判断）：

    1. 空白字符（空格、换行、制表符）丢掉，不产生 token。
    2. "\\begin{名字}" 和 "\\end{名字}" 整体算一个 token，例如 "\\begin{pmatrix}"。
       （花括号里的名字前后可能有空格，去掉空格后再拼，例如 "\\begin{ matrix }" → "\\begin{matrix}"。）
    3. 反斜杠后面跟一串英文字母：反斜杠连同这串字母算一个 token，例如 "\\frac"、"\\alpha"。
    4. 反斜杠后面跟一个非字母的字符：两个字符合起来算一个 token，例如 "\\,"、"\\{"、"\\\\"（换行）、"\\ "（空格命令）。
       字符串末尾单独一个反斜杠：算一个 token "\\"。
    5. 其他任何字符（字母、数字、运算符、括号、汉字……）各自算一个 token。

    例：tokenize_latex("\\frac{a}{b} + x^2") == ["\\frac", "{", "a", "}", "{", "b", "}", "+", "x", "^", "2"]
        tokenize_latex("S_{面积}")          == ["S", "_", "{", "面", "积", "}"]
    """
    toks, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
        elif c == "\\":
            j = i + 1
            while j < n and s[j].isascii() and s[j].isalpha():
                j += 1
            if j == i + 1:                                   # 反斜杠后面不是字母
                toks.append(s[i:i + 2])
                i += 2
                continue
            name = s[i:j]
            if name in ("\\begin", "\\end"):
                k = j
                while k < n and s[k].isspace():
                    k += 1
                if k < n and s[k] == "{":
                    close = s.find("}", k)
                    if close != -1:
                        toks.append(name + "{" + s[k + 1:close].strip() + "}")
                        i = close + 1
                        continue
            toks.append(name)
            i = j
        else:
            toks.append(c)
            i += 1
    return toks
# <<<


# >>> 练习 0.2：逐 token 的规范化规则
def apply_token_rules(tokens):
    """对 token 列表逐个应用三条规则，返回新的列表（不修改输入）：

    1. 同义命令换成同一个名字：查 SYNONYMS，例如 "\\dfrac" → "\\frac"、"\\le" → "\\leq"。
    2. 间距命令（SPACING 里的）直接删掉。
    3. 尺寸命令（SIZING 里的）删掉命令本身，保留后面的括号。
       特例：紧跟在 "\\left"、"\\right"、"\\middle" 后面的 "." 表示"不画括号"，要连同命令一起删掉。

    例：apply_token_rules(["\\left", "(", "a", "\\,", "b", "\\right", "."]) == ["(", "a", "b"]
    """
    out, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        if t in SPACING:
            i += 1
        elif t in SIZING:
            if t in ("\\left", "\\right", "\\middle") and i + 1 < len(tokens) and tokens[i + 1] == ".":
                i += 2
            else:
                i += 1
        else:
            out.append(SYNONYMS.get(t, t))
            i += 1
    return out
# <<<


# ---------------------------------------------------------------------- 语法树
# 节点有四种，都用元组表示：
#   ("tok", t)                       一个普通 token
#   ("group", [节点, ...])           一对花括号括起来的一组
#   ("cmd", 名字, 可选参数, [参数])  带参数的命令；可选参数是节点列表或 None，每个参数是一个节点
#   ("script", 底, 下标, 上标)        带上下标的东西；下标、上标是节点或 None


# >>> 练习 0.3：解析成语法树
def parse_latex(tokens):
    """token 列表（已经做过逐 token 规则）→ 节点列表。

    读入规则：
    - "{" 开始一个分组，读到配对的 "}" 为止，得到 ("group", [...])。多出来的 "}" 或缺少 "}" 都抛 LatexSyntaxError。
    - ARITY 里的命令：先看紧跟着的是不是 "["；是的话（只对 \\sqrt）一直读到配对的 "]"，作为可选参数（节点列表）。
      然后再读 ARITY[名字] 个"原子"作为参数。
    - 一个"原子"是：一个分组；或一个带参数的命令（连同它的参数）；或一个普通 token。
      需要原子却遇到 "}"、"^"、"_" 或已经读完，抛 LatexSyntaxError。
    - 读完一个原子后，如果后面跟着 "^" 或 "_"，就把它当作底，继续读上下标：
      "_" 后面读一个原子作下标，"^" 后面读一个原子作上标，两种可以按任意顺序各出现一次；
      同一种出现两次（例如 x^1^2）抛 LatexSyntaxError。结果是 ("script", 底, 下标, 上标)。
    - 开头就是 "^" 或 "_"（没有底），底记作 ("group", [])。

    例：parse_latex(["x", "^", "2"]) == [("script", ("tok", "x"), None, ("tok", "2"))]
    """
    pos = 0

    def atom():
        nonlocal pos
        if pos >= len(tokens) or tokens[pos] in ("}", "^", "_"):
            raise LatexSyntaxError(f"第 {pos} 个 token 处缺少一个参数")
        t = tokens[pos]
        if t == "{":
            pos += 1
            items = seq(inside_group=True)
            pos += 1                                         # 跳过 "}"
            return ("group", items)
        pos += 1
        if t in ARITY:
            opt = None
            if t == "\\sqrt" and pos < len(tokens) and tokens[pos] == "[":
                depth, start = 0, pos
                while pos < len(tokens):
                    if tokens[pos] == "[":
                        depth += 1
                    elif tokens[pos] == "]":
                        depth -= 1
                        if depth == 0:
                            break
                    pos += 1
                if pos >= len(tokens):
                    raise LatexSyntaxError("\\sqrt 的方括号没有配对")
                opt = parse_latex(tokens[start + 1:pos])
                pos += 1
            return ("cmd", t, opt, [atom() for _ in range(ARITY[t])])
        return ("tok", t)

    def with_scripts(base):
        nonlocal pos
        sub = sup = None
        while pos < len(tokens) and tokens[pos] in ("^", "_"):
            kind = tokens[pos]
            pos += 1
            if kind == "_":
                if sub is not None:
                    raise LatexSyntaxError("同一个底有两个下标")
                sub = atom()
            else:
                if sup is not None:
                    raise LatexSyntaxError("同一个底有两个上标")
                sup = atom()
        return base if sub is None and sup is None else ("script", base, sub, sup)

    def seq(inside_group):
        nonlocal pos
        items = []
        while pos < len(tokens):
            t = tokens[pos]
            if t == "}":
                if inside_group:
                    return items
                raise LatexSyntaxError(f"第 {pos} 个 token 是多出来的 }}")
            if t in ("^", "_"):
                items.append(with_scripts(("group", [])))
            else:
                items.append(with_scripts(atom()))
        if inside_group:
            raise LatexSyntaxError("花括号没有配对：缺少 }")
        return items

    return seq(inside_group=False)
# <<<


# >>> 练习 0.4：按规范写法写回
def serialize(nodes):
    """节点列表 → 规范 token 列表。规范写法：

    - 先"剥皮"：只套着一个元素的分组，等于这个元素本身（{x} 和 x 一样，{{a+b}} 和 {a+b} 一样），
      一层层剥掉，直到剩下的不再是单元素分组。下面每条规则都先剥皮再处理。
    - ("tok", t) → [t]
    - ("group", items)（剥皮后只剩多元素分组或空分组）：写成 ["{", ...items..., "}"]。空分组写成 ["{", "}"]。
    - ("cmd", 名字, 可选参数, 参数)：[名字]，有可选参数时接 ["[", ..., "]"]，然后每个参数都写成 ["{", ..., "}"]。
    - ("script", 底, 下标, 上标)：先写底，再写 ["_", "{", ..., "}"]（如果有下标），最后写 ["^", "{", ..., "}"]
      （如果有上标）。先下标、后上标。底剥皮后如果是多元素分组，保留它的花括号；
      如果本身就带上下标（例如 {x^2}^3 的底 x^2），也要保留一对花括号，否则会写成不合法的 x^{2}^{3}。
    - "参数写成 {…}"时，剥皮后如果是分组，只保留一层花括号：{ {a+b} } → { a + b }。

    例：serialize(parse_latex(["x", "^", "2", "_", "i"])) == ["x", "_", "{", "i", "}", "^", "{", "2", "}"]
    """
    def unwrap(node):                                        # 剥掉单元素分组：{{a}} → a，{{a+b}} → {a+b}
        while node[0] == "group" and len(node[1]) == 1:
            node = node[1][0]
        return node

    def ser(node):                                           # 一个节点 → token 列表
        node = unwrap(node)
        kind = node[0]
        if kind == "tok":
            return [node[1]]
        if kind == "group":                                  # 多元素或空分组
            return ["{"] + serialize(node[1]) + ["}"]
        if kind == "cmd":
            _, name, opt, args = node
            out = [name]
            if opt is not None:
                out += ["["] + serialize(opt) + ["]"]
            for a in args:
                out += braced(a)
            return out
        _, base, sub, sup = node                             # script
        base = unwrap(base)
        out = ["{"] + ser(base) + ["}"] if base[0] == "script" else ser(base)   # 底自己带上下标：保留花括号
        if sub is not None:
            out += ["_"] + braced(sub)
        if sup is not None:
            out += ["^"] + braced(sup)
        return out

    def braced(node):                                        # 参数写成 { … }；参数本身是分组时只保留一层
        node = unwrap(node)
        if node[0] == "group":
            return ["{"] + serialize(node[1]) + ["}"]
        return ["{"] + ser(node) + ["}"]

    out = []
    for node in nodes:
        out.extend(ser(node))
    return out
# <<<


def normalized_tokens(s):
    """规范化后的 token 列表。解析失败（花括号不配对等）时，退回到只做分词和逐 token 规则。"""
    toks = apply_token_rules(tokenize_latex(s))
    try:
        return serialize(parse_latex(toks))
    except LatexSyntaxError:
        return toks


def normalize_latex(s):
    """规范化后的字符串：规范 token 用一个空格隔开。两个公式规范化后相等，就认为"写法不同、含义相同"。"""
    return " ".join(normalized_tokens(s))


def is_well_formed(s):
    """能不能被 parse_latex 解析（花括号配对、上下标和命令都有参数）。这是一个比"能编译"弱得多的检查。"""
    try:
        parse_latex(apply_token_rules(tokenize_latex(s)))
        return True
    except LatexSyntaxError:
        return False
