"""选择被测的实现：默认是你的 adapters.py；设了环境变量 A_SOLUTIONS=1 时测参考实现（给维护者用）。"""
import importlib
import os

SOLUTIONS = os.environ.get("A_SOLUTIONS") == "1"
adapters = importlib.import_module("assignments.solutions.a4.adapters" if SOLUTIONS else "assignments.a4.tests.adapters")
ASSIGNMENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SOLUTIONS:
    ASSIGNMENT_DIR = os.path.join(os.path.dirname(ASSIGNMENT_DIR), "solutions", "a4")
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
