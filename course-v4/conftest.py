# 让 pytest 在 course-v4/ 下能直接 import miniocr 和各阶段的 labs；测试一律在 CPU 上跑（正确性以 CPU 为准）
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
