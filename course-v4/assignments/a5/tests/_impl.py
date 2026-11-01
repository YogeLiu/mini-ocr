"""默认测试学习者；仅 A_SOLUTIONS=1 选择维护者答案。"""
import importlib
import os
from pathlib import Path
SOLUTIONS = os.environ.get("A_SOLUTIONS") == "1"
ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
if SOLUTIONS:
    ASSIGNMENT_DIR = ASSIGNMENT_DIR.parent / "solutions" / "a5"
adapters = importlib.import_module("assignments.solutions.a5.adapters" if SOLUTIONS else "assignments.a5.tests.adapters")
