"""表格的 TEDS（Tree-Edit-Distance-based Similarity，Zhong 等，2020，PubTabNet）。

表格用 HTML 表示：<table><tr><td>…</td><td colspan="2">…</td></tr>…</table>。
把它看成一棵树（table → tr → td，td 带着 colspan、rowspan 和文字），两张表的差别 = 树的编辑距离：
把一棵树变成另一棵最少要多少次"删除一个节点、插入一个节点、改一个节点的标签"。

    TEDS = 1 − 编辑距离 / max(两棵树的节点数)

改标签的代价：两个都是 td 时，colspan/rowspan 不同记 1，相同则记文字的归一化编辑距离（0–1）；
否则标签相同记 0、不同记 1。只比结构（TEDS-S）时文字不计代价。
"""
from html.parser import HTMLParser

from ..metrics import ned


class Node:
    def __init__(self, tag, colspan=1, rowspan=1, text=""):
        self.tag, self.colspan, self.rowspan, self.text = tag, colspan, rowspan, text
        self.children = []

    def __repr__(self):
        return f"Node({self.tag}, {self.colspan}, {self.rowspan}, {self.text!r}, {len(self.children)} 个子节点)"


class _TableParser(HTMLParser):
    KEEP = {"table", "thead", "tbody", "tr", "td", "th"}

    def __init__(self):
        super().__init__()
        self.root = Node("root")
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        if tag not in self.KEEP:
            return
        a = dict(attrs)
        n = Node("td" if tag == "th" else tag, int(a.get("colspan", 1) or 1), int(a.get("rowspan", 1) or 1))
        self.stack[-1].children.append(n)
        self.stack.append(n)

    def handle_endtag(self, tag):
        tag = "td" if tag == "th" else tag
        for k in range(len(self.stack) - 1, 0, -1):                 # 容忍没闭合的标签：一直弹到对应的那一层
            if self.stack[k].tag == tag:
                del self.stack[k:]
                return

    def handle_data(self, data):
        if self.stack[-1].tag == "td":
            self.stack[-1].text += data.strip()


def parse_table(html):
    """HTML 字符串 → 树（根是 table 节点；thead/tbody 被去掉，只保留 table → tr → td）。
    残缺的 HTML 也尽量解析；完全没有 <table> 时返回一个空的 table 节点。"""
    p = _TableParser()
    p.feed(html)
    p.close()

    def flat(n):
        kids = []
        for c in n.children:
            kids += flat(c).children if c.tag in ("thead", "tbody") else [flat(c)]
        n.children = kids
        return n

    root = flat(p.root)
    tables = [c for c in root.children if c.tag == "table"]
    return tables[0] if tables else Node("table")


def _postorder(root):
    """后序遍历：返回节点列表、每个节点最左边的叶子的下标、关键根（keyroots）。"""
    nodes, lml = [], []

    def go(n):
        first = None
        for c in n.children:
            k = go(c)
            if first is None:
                first = k
        nodes.append(n)
        lml.append(len(nodes) - 1 if first is None else first)
        return lml[-1]

    go(root)
    seen, keyroots = set(), []
    for i in range(len(nodes) - 1, -1, -1):                            # 每个"最左叶子"只保留下标最大的那个节点
        if lml[i] not in seen:
            seen.add(lml[i])
            keyroots.append(i)
    return nodes, lml, sorted(keyroots)


# >>> 练习 25.5：树的编辑距离（Zhang–Shasha 算法）
def tree_edit_distance(a, b, rename, ins=1.0, dele=1.0):
    """两棵有序树的编辑距离（Zhang 和 Shasha，1989）。rename(x, y) 是把节点 x 改成 y 的代价。

    后序编号后，对每一对关键根 (i, j) 算"森林距离" fd：
      fd[x][y] = a 的节点 l(i)..l(i)+x−1 组成的森林 与 b 的 l(j)..l(j)+y−1 组成的森林 的距离；
      fd[x][0] = x · dele，fd[0][y] = y · ins；
      若两个森林的最后一个节点 i1、j1 的最左叶子正好是 l(i)、l(j)（各自都是一棵完整的树）：
          fd[x][y] = min(fd[x−1][y] + dele, fd[x][y−1] + ins, fd[x−1][y−1] + rename(i1, j1))，并记下 td[i1][j1] = fd[x][y]
      否则：
          fd[x][y] = min(fd[x−1][y] + dele, fd[x][y−1] + ins, fd[l(i1)−l(i)][l(j1)−l(j)] + td[i1][j1])
    返回 td[最后][最后]，即两棵整树的距离。
    """
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
                        fd[x][y] = min(fd[x - 1][y] + dele, fd[x][y - 1] + ins,
                                       fd[la[i1] - li][lb[j1] - lj] + td[i1][j1])
    return td[-1][-1]
# <<<


def count_nodes(n):
    return 1 + sum(count_nodes(c) for c in n.children)


def teds(pred_html, ref_html, structure_only=False):
    """两张 HTML 表格的 TEDS，在 [0, 1] 之间，1 表示完全相同。"""
    a, b = parse_table(pred_html), parse_table(ref_html)

    def rename(x, y):
        if x.tag != y.tag:
            return 1.0
        if x.tag != "td":
            return 0.0
        if (x.colspan, x.rowspan) != (y.colspan, y.rowspan):
            return 1.0
        return 0.0 if structure_only else ned(x.text, y.text)

    n = max(count_nodes(a), count_nodes(b))
    return 1.0 - tree_edit_distance(a, b, rename) / n
