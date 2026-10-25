"""生成"挖空版"练习目录：把参考实现里标记为练习的代码删掉，留下函数签名、文档字符串和测试。

标记方法（写在参考实现里）：

    # >>> 练习 3.5：归一化
    def normalize(s):
        \"\"\"文档字符串会保留。\"\"\"
        return to_latex(parse(s))          ← 这一部分会被替换
    # <<<

生成后变成：

    def normalize(s):
        \"\"\"文档字符串会保留。\"\"\"
        raise NotImplementedError("练习 3.5：归一化")

用法（在 course-v3/ 下）：
    python -m tools.make_student ../my-course          # 生成练习目录
    python -m tools.make_student --list                # 只列出所有练习
"""
import argparse
import os
import re
import shutil
import sys

START = re.compile(r"^(\s*)# >>> (练习[^\n]*)$")
END = re.compile(r"^\s*# <<<\s*$")
COPY = ["miniocr", "labs", "tools", "assignments", "conftest.py", "pytest.ini"]


def _docstring_end(lines, i):
    """lines[i] 若是文档字符串的开头，返回它结束那一行的下标；否则返回 None。"""
    s = lines[i].strip()
    for q in ('"""', "'''"):
        if s.startswith(q):
            if s.count(q) >= 2 and len(s) > 3:
                return i
            for j in range(i + 1, len(lines)):
                if q in lines[j]:
                    return j
    return None


def strip_exercises(text):
    """返回 (挖空后的文本, 练习名列表)。一个练习块里可以有多个函数，每个函数都只留签名和文档字符串。"""
    lines = text.split("\n")
    out, names, i = [], [], 0
    while i < len(lines):
        m = START.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        name = m.group(2)
        names.append(name)
        j = i + 1
        while j < len(lines) and not END.match(lines[j]):
            j += 1
        if j == len(lines):
            raise ValueError(f"练习块没有结束标记: {name}")
        block = lines[i + 1:j]
        out.extend(_hollow(block, name))
        i = j + 1
    return "\n".join(out), names


def _hollow(block, name):
    """把一个练习块挖空。块里的每个 def：保留 def 行（可能跨多行）和文档字符串，函数体换成 raise。
    class 行、类的文档字符串和装饰器原样保留，类里的每个方法按 def 处理。
    块里不在任何 def 之内的其他代码：整段换成 raise（缩进与该段第一行相同）。"""
    out, k = [], 0
    msg = f"raise NotImplementedError({name!r})"
    while k < len(block):
        line = block[k]
        stripped = line.lstrip()
        if stripped.startswith("def "):
            indent = len(line) - len(stripped)
            while not block[k].rstrip().endswith(":"):        # def 行可能跨多行
                out.append(block[k])
                k += 1
            out.append(block[k])
            k += 1
            if k < len(block):
                end = _docstring_end(block, k)
                if end is not None:
                    out.extend(block[k:end + 1])
                    k = end + 1
            out.append(" " * (indent + 4) + msg)
            while k < len(block) and (not block[k].strip() or len(block[k]) - len(block[k].lstrip()) > indent):
                k += 1                                            # 跳过函数体
            out.append("")
        elif stripped.startswith("class "):
            while not block[k].rstrip().endswith(":"):
                out.append(block[k])
                k += 1
            out.append(block[k])
            k += 1
            if k < len(block):
                end = _docstring_end(block, k)
                if end is not None:
                    out.extend(block[k:end + 1])
                    k = end + 1
            out.append("")
        elif stripped.startswith("@"):
            out.append(line)
            k += 1
        elif not stripped:
            k += 1
        else:
            indent = len(line) - len(stripped)
            out.append(" " * indent + msg)
            while k < len(block) and not block[k].lstrip().startswith(("def ", "class ", "@")):
                k += 1
    while out and not out[-1].strip():
        out.pop()
    return out


def iter_py(root):
    for d, _, files in os.walk(root):
        for f in sorted(files):
            if f.endswith(".py"):
                yield os.path.join(d, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dest", nargs="?")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if a.list:
        for p in iter_py(os.path.join(here, "miniocr")):
            _, names = strip_exercises(open(p, encoding="utf-8").read())
            for n in names:
                print(f"{os.path.relpath(p, here)}: {n}")
        return
    if not a.dest:
        ap.error("请给出目标目录，例如 ../my-course")
    if os.path.exists(a.dest):
        sys.exit(f"{a.dest} 已存在。为了不覆盖你写过的代码，请换一个目录名或先手动删除它。")
    for item in COPY:
        src = os.path.join(here, item)
        dst = os.path.join(a.dest, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "solutions"))   # 大作业的参考实现不复制
        else:
            shutil.copy2(src, dst)
    total = 0
    for p in iter_py(os.path.join(a.dest, "miniocr")):
        text, names = strip_exercises(open(p, encoding="utf-8").read())
        if names:
            open(p, "w", encoding="utf-8").write(text)
            total += len(names)
    print(f"已生成 {a.dest}，共挖空 {total} 个练习。进入该目录后运行 python -m pytest 查看哪些还没完成。")


if __name__ == "__main__":
    main()
