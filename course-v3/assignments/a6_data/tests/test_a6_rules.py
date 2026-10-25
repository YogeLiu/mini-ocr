"""A6 的限制：a6/ 里不能用现成的层、损失、优化器、注意力、插值（检查器见 assignments/_common/rules.py）。"""
from assignments._common.rules import check_dir

from ._impl import CODE_DIR


def test_no_builtin_layers_losses_optimizers():
    bad = check_dir(CODE_DIR)
    assert not bad, "a6/ 里用了被禁止的现成实现（见说明书第 1 节）：\n" + "\n".join(bad)



BANNED_IMPORTS = ("miniocr.engine", "miniocr.xiaomi", "miniocr.eval", "miniocr.metrics", "datasketch", "sklearn", "scipy",
                  "Levenshtein", "editdistance", "rapidfuzz", "jellyfish")
CHEAT = ("_truth", "planted")


def test_imports_and_no_peeking():
    """a6/ 从 miniocr 里只能用 miniocr.data；不能用现成的 MinHash、编辑距离、聚类库；
    不能碰数据池的真值（pool._truth、pool.planted）。读数据池用 assignments.a6_data.pool 的 load_pool、load_test。"""
    import ast
    import os
    bad = []
    for f in sorted(os.listdir(CODE_DIR)):
        if not f.endswith(".py"):
            continue
        for n in ast.walk(ast.parse(open(os.path.join(CODE_DIR, f), encoding="utf-8").read())):
            if isinstance(n, ast.Import):
                mods = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                mods = [n.module]
                bad += [f"{f}:{n.lineno} {a.name}" for a in n.names if a.name in CHEAT]
            else:
                mods = []
                if isinstance(n, ast.Attribute) and n.attr in CHEAT:
                    bad.append(f"{f}:{n.lineno} .{n.attr}")
            bad += [f"{f}:{n.lineno} {m}" for m in mods
                    if any(m == b or m.startswith(b + ".") for b in BANNED_IMPORTS)
                    or (m.startswith("miniocr") and not (m == "miniocr.data" or m.startswith("miniocr.data.")))]
    assert not bad, "a6/ 里用了不允许的模块或数据池的真值：\n" + "\n".join(bad)
