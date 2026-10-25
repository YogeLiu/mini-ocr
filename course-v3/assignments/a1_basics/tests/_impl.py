"""选择被测的实现：默认是你的 adapters.py；设了环境变量 A_SOLUTIONS=1 时测参考实现（给维护者用）。"""
import importlib
import os

SOLUTIONS = bool(os.environ.get("A_SOLUTIONS"))
adapters = importlib.import_module("assignments.solutions.a1_basics.adapters" if SOLUTIONS
                                   else "assignments.a1_basics.tests.adapters")
CODE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "a1")
if SOLUTIONS:
    CODE_DIR = os.path.join(os.path.dirname(CODE_DIR), "..", "solutions", "a1_basics", "a1")
