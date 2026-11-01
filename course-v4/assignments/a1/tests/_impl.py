"""默认调用学生接线；仅 A_SOLUTIONS=1 选择维护者参考解。"""
import importlib
import os
from pathlib import Path

SOLUTIONS = os.environ.get("A_SOLUTIONS") == "1"
adapters = importlib.import_module(
    "assignments.solutions.a1.adapters" if SOLUTIONS else "assignments.a1.tests.adapters")
ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
if SOLUTIONS:
    ASSIGNMENT_DIR = ASSIGNMENT_DIR.parent / "solutions" / "a1"
