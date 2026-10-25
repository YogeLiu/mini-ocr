# 让 pytest 在 course-v3/ 下能直接 import miniocr 和 labs
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
