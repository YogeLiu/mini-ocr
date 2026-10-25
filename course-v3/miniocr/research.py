"""第 36 章：做一次完整研究需要的工具。都是把前面各章的零件（第 25 章的 bootstrap、配对比较）组织成"可以写进报告"的形式。

    manifest        一次运行的身份证：配置、数据文件的内容哈希、种子 → 一个短哈希，结果表里每一行都带着它
    compare         两个系统在同一批样本上的配对比较，给出差、95% 区间和一句不夸大的结论
    seed_table      同一配置多个种子的均值、标准差、最小、最大；种子少于 3 个时标记出来
    bucket_report   按分桶（长度、类型……）报告每桶的样本数、均值和 95% 区间
    TestGuard       测试集只允许评测一次（方案冻结之后）；第二次评测要显式说明理由
"""
import hashlib
import json
from collections import defaultdict

import numpy as np

from .eval.stats import bootstrap_ci, paired_bootstrap


def file_hash(path, n=12):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


# >>> 练习 36.1：运行的身份证
def manifest(config, data_files=(), seed=0):
    """config：dict（可以嵌套，值都能被 JSON 序列化）；data_files：数据文件的路径；seed：随机种子。
    返回 dict：config、seed、data（{路径: file_hash(路径)}）、id。
    id = sha256(json.dumps({"config": config, "seed": seed, "data": data}, sort_keys=True)) 的前 12 位十六进制：
    配置里键的顺序不同、但内容相同时，id 相同；数据文件的内容变了，id 就变。"""
    data = {p: file_hash(p) for p in data_files}
    body = {"config": config, "seed": seed, "data": data}
    rid = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
    return dict(body, id=rid)
# <<<


# >>> 练习 36.2：配对比较与结论
def compare(a, b, n_boot=2000, seed=0, names=("A", "B")):
    """a、b：两个系统在同一批样本上的逐样本分数（同样长、同样顺序；0/1 的对错或 [0, 1] 的分数）。
    用 paired_bootstrap 得到差 a − b 的点估计和 95% 区间。结论：
        区间的下界 > 0：f"{A} 比 {B} 高"；上界 < 0：f"{A} 比 {B} 低"；否则："未观察到可靠差异"。
    返回 dict：mean_a、mean_b、diff、lo、hi、verdict。长度不同时抛出 ValueError（不是同一批样本，不能配对）。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError(f"两个系统的样本数不同（{len(a)} vs {len(b)}）：不是同一批样本，不能做配对比较")
    d, lo, hi, _ = paired_bootstrap(a, b, n_boot=n_boot, seed=seed)
    A, B = names
    verdict = f"{A} 比 {B} 高" if lo > 0 else (f"{A} 比 {B} 低" if hi < 0 else "未观察到可靠差异")
    return dict(mean_a=float(a.mean()), mean_b=float(b.mean()), diff=float(d), lo=lo, hi=hi, verdict=verdict)
# <<<


# >>> 练习 36.3：种子表
def seed_table(results):
    """results：[(配置名, 种子, 分数), …]。按配置名分组（按第一次出现的顺序），返回
    [(配置名, n, 均值, 标准差, 最小, 最大, 种子够不够), …]；标准差用 ddof=1（n = 1 时记为 nan）；
    种子够不够 = n ≥ 3。"""
    groups = defaultdict(list)
    order = []
    for name, _, score in results:
        if name not in groups:
            order.append(name)
        groups[name].append(score)
    out = []
    for name in order:
        s = np.asarray(groups[name], float)
        std = float(s.std(ddof=1)) if len(s) > 1 else float("nan")
        out.append((name, len(s), float(s.mean()), std, float(s.min()), float(s.max()), len(s) >= 3))
    return out
# <<<


# >>> 练习 36.4：分桶报告
def bucket_report(scores, buckets, n_boot=2000, seed=0):
    """scores：逐样本分数；buckets：每个样本的桶名（同样长）。按桶名排序，返回
    [(桶名, 样本数, 均值, 下界, 上界), …]，区间用 bootstrap_ci（每个桶单独重采样）。"""
    scores, buckets = np.asarray(scores, float), np.asarray(buckets)
    out = []
    for k in sorted(set(buckets.tolist())):
        s = scores[buckets == k]
        m, lo, hi = bootstrap_ci(s, n_boot=n_boot, seed=seed)
        out.append((k, len(s), float(m), lo, hi))
    return out
# <<<


class TestGuard:
    """测试集只评测一次。用法：
        guard = TestGuard("outputs/ch36/test_log.json")
        guard.check(run_id)            # 第一次：记录下来；同一个 run_id 第二次：抛出 RuntimeError
        guard.check(run_id, reason="修复了评测脚本的 bug，见 notes/…")   # 有理由时允许，并记录理由
    记录写在一个 JSON 文件里，报告里附上它：读者能看到测试集被用过几次、为什么。"""
    __test__ = False                                               # 名字以 Test 开头，告诉 pytest 这不是测试类

    def __init__(self, path):
        self.path = path
        try:
            with open(path) as f:
                self.log = json.load(f)
        except FileNotFoundError:
            self.log = []

    def check(self, run_id, reason=None):
        used = [e for e in self.log if e["run"] == run_id]
        if used and reason is None:
            raise RuntimeError(f"{run_id} 已经在测试集上评测过 {len(used)} 次；要再评测，请给出理由（reason=...）")
        self.log.append({"run": run_id, "n": len(used) + 1, "reason": reason})
        with open(self.path, "w") as f:
            json.dump(self.log, f, ensure_ascii=False, indent=1)


def md_table(header, rows):
    """简单的 Markdown 表格。"""
    line = lambda xs: "| " + " | ".join(str(x) for x in xs) + " |"
    return "\n".join([line(header), line(["---"] * len(header))] + [line(r) for r in rows])
