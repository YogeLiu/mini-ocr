"""A4 参考实现：表格的 TEDS。HTML → 树（table → tr → td），Zhang–Shasha 树编辑距离，TEDS = 1 − 距离 / max(节点数)。"""
from html.parser import HTMLParser

from .metrics import ned


class Node:
    def __init__(self, tag, colspan=1, rowspan=1, text=""):
        self.tag, self.colspan, self.rowspan, self.text = tag, colspan, rowspan, text
        self.children = []


class _Parser(HTMLParser):
    KEEP = {"table", "thead", "tbody", "tr", "td", "th"}

    def __init__(self):
        super().__init__()
        self.root = Node("root")
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        if tag not in self.KEEP:
            return
        a = dict(attrs)
        n = Node("td" if tag == "th" else tag, int(a.get("colspan") or 1), int(a.get("rowspan") or 1))
        self.stack[-1].children.append(n)
        self.stack.append(n)

    def handle_endtag(self, tag):
        tag = "td" if tag == "th" else tag
        for k in range(len(self.stack) - 1, 0, -1):          # 容忍没闭合的标签：弹到对应的那一层
            if self.stack[k].tag == tag:
                del self.stack[k:]
                return

    def handle_data(self, data):
        if self.stack[-1].tag == "td":
            self.stack[-1].text += data.strip()


def parse_table(html):
    """第一个 <table> 的树；thead、tbody 被展开（它们的子节点直接挂到上一层）。没有 table 时返回空的 table 节点。"""
    p = _Parser()
    p.feed(html)
    p.close()

    def flat(n):
        kids = []
        for c in n.children:
            kids += flat(c).children if c.tag in ("thead", "tbody") else [flat(c)]
        n.children = kids
        return n

    tables = [c for c in flat(p.root).children if c.tag == "table"]
    return tables[0] if tables else Node("table")


def count_nodes(n):
    return 1 + sum(count_nodes(c) for c in n.children)


def _postorder(root):
    nodes, lml = [], []

    def go(n):
        first = None
        for c in n.children:
            k = go(c)
            first = k if first is None else first
        nodes.append(n)
        lml.append(len(nodes) - 1 if first is None else first)
        return lml[-1]

    go(root)
    seen, keyroots = set(), []
    for i in range(len(nodes) - 1, -1, -1):
        if lml[i] not in seen:
            seen.add(lml[i])
            keyroots.append(i)
    return nodes, lml, sorted(keyroots)


def tree_edit_distance(a, b, rename, ins=1.0, dele=1.0):
    """Zhang–Shasha：后序编号、最左叶子、关键根，对每对关键根算森林距离。"""
    A, la, ka = _postorder(a)
    B, lb, kb = _postorder(b)
    td = [[0.0] * len(B) for _ in A]
    for i in ka:
        for j in kb:
            li, lj = la[i], lb[j]
            nx, ny = i - li + 2, j - lj + 2
            fd = [[0.0] * ny for _ in range(nx)]
            for x in range(1, nx):
                fd[x][0] = fd[x - 1][0] + dele
            for y in range(1, ny):
                fd[0][y] = fd[0][y - 1] + ins
            for x in range(1, nx):
                i1 = li + x - 1
                for y in range(1, ny):
                    j1 = lj + y - 1
                    if la[i1] == li and lb[j1] == lj:
                        fd[x][y] = min(fd[x - 1][y] + dele, fd[x][y - 1] + ins, fd[x - 1][y - 1] + rename(A[i1], B[j1]))
                        td[i1][j1] = fd[x][y]
                    else:
                        fd[x][y] = min(fd[x - 1][y] + dele, fd[x][y - 1] + ins, fd[la[i1] - li][lb[j1] - lj] + td[i1][j1])
    return td[-1][-1]


def teds(pred_html, ref_html, structure_only=False):
    a, b = parse_table(pred_html), parse_table(ref_html)

    def rename(x, y):
        if x.tag != y.tag:
            return 1.0
        if x.tag != "td":
            return 0.0
        if (x.colspan, x.rowspan) != (y.colspan, y.rowspan):
            return 1.0
        return 0.0 if structure_only else ned(x.text, y.text)

    return 1.0 - tree_edit_distance(a, b, rename) / max(count_nodes(a), count_nodes(b))
