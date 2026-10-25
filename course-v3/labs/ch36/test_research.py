"""第 36 章的自动检查。"""
import math

import numpy as np
import pytest

from miniocr.research import TestGuard, bucket_report, compare, manifest, seed_table


def test_manifest(tmp_path):
    f = tmp_path / "d.txt"
    f.write_text("abc")
    a = manifest({"lr": 1e-4, "model": {"r": 16, "alpha": 32}}, [str(f)], seed=1)
    b = manifest({"model": {"alpha": 32, "r": 16}, "lr": 1e-4}, [str(f)], seed=1)
    assert a["id"] == b["id"] and len(a["id"]) == 12                     # 键的顺序不影响
    assert manifest({"lr": 1e-4, "model": {"r": 16, "alpha": 32}}, [str(f)], seed=2)["id"] != a["id"]
    f.write_text("abd")
    assert manifest({"lr": 1e-4, "model": {"r": 16, "alpha": 32}}, [str(f)], seed=1)["id"] != a["id"]


def test_compare():
    rng = np.random.default_rng(0)
    a = (rng.random(400) < 0.8).astype(float)
    b = a.copy()
    b[:60] = 0.0                                                            # b 在 60 个样本上更差
    r = compare(a, b, names=("新", "旧"))
    assert r["lo"] > 0 and r["verdict"] == "新 比 旧 高"
    assert compare(b, a)["verdict"] == "A 比 B 低"
    assert compare(a, a)["verdict"] == "未观察到可靠差异"
    with pytest.raises(ValueError):
        compare(a, a[:10])


def test_seed_table():
    rows = seed_table([("x", 0, 0.5), ("y", 0, 0.9), ("x", 1, 0.7), ("x", 2, 0.6)])
    assert rows[0][:2] == ("x", 3) and abs(rows[0][2] - 0.6) < 1e-12 and abs(rows[0][3] - 0.1) < 1e-12 and rows[0][6]
    assert rows[1][0] == "y" and rows[1][1] == 1 and math.isnan(rows[1][3]) and not rows[1][6]


def test_bucket_report():
    rows = bucket_report([1, 0, 1, 1, 0, 0], ["短", "短", "长", "长", "长", "中"])
    assert [r[0] for r in rows] == sorted(["短", "长", "中"])
    d = {r[0]: r for r in rows}
    assert d["长"][1] == 3 and abs(d["长"][2] - 2 / 3) < 1e-12 and d["中"][2] == 0.0


def test_test_guard(tmp_path):
    g = TestGuard(str(tmp_path / "log.json"))
    g.check("run1")
    with pytest.raises(RuntimeError):
        g.check("run1")
    g.check("run1", reason="修复了评测脚本")
    g2 = TestGuard(str(tmp_path / "log.json"))                             # 记录写在文件里，换一个进程也算数
    assert len(g2.log) == 2
    with pytest.raises(RuntimeError):
        g2.check("run1")
