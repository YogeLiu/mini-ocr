"""检查挖空工具本身：保留签名和文档字符串，函数体换成 raise。"""
from tools.make_student import strip_exercises

SRC = '''x = 1
# >>> 练习 9.9：示例
def f(a,
      b):
    """文档。"""
    return a + b


def g():
    return 2
# <<<
y = 2
'''


def test_strip_keeps_signature_and_doc():
    out, names = strip_exercises(SRC)
    assert names == ["练习 9.9：示例"]
    assert "def f(a,\n      b):\n    \"\"\"文档。\"\"\"\n    raise NotImplementedError('练习 9.9：示例')" in out
    assert "def g():\n    raise NotImplementedError" in out
    assert "return a + b" not in out and "return 2" not in out
    assert out.startswith("x = 1\n") and out.endswith("y = 2\n")


def test_strip_class_keeps_methods():
    src = '# >>> 练习 9.8：类\nclass A(B):\n    """类文档。"""\n\n    def __init__(self, n):\n        super().__init__()\n        self.n = n\n\n    @property\n    def m(self):\n        return self.n\n# <<<\n'
    out, _ = strip_exercises(src)
    compile(out, "x", "exec")
    assert out.startswith('class A(B):\n    """类文档。"""')
    assert "    def __init__(self, n):\n        raise NotImplementedError" in out
    assert "    @property\n    def m(self):" in out and "self.n = n" not in out
