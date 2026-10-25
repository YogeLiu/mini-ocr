"""第 31 章：可验证奖励。每个任务一个打分函数，都在 [0, 1] 之间；格式不对时得 0（格式闸门）。

    well_formed     回答的格式对不对（数字、颜色词、yes/no、框或 none、……），和答案对不对无关
    task_reward     第 23 章九种任务的奖励：精确匹配、1 − NED、IoU
    doc_reward      文档 OCR 的奖励：文字 1 − NED、公式 CDM、表格 TEDS（第 25 章）

为什么奖励都放在 [0, 1]：多任务一起训练时（31.6 节），组内标准化只在同一道题的回答之间比较，
奖励的尺度不影响优势；但"格式错 = 0、完全对 = 1"让各任务的奖励含义一致，监控时可以直接比较。
"""
import torch

from ..boxes import box_iou
from ..data.instruct import parse_box
from ..data.shapes import COLORS
from ..metrics import ned

EXACT = ("count", "count_color", "left", "exists", "compare", "refer")


# >>> 练习 31.1：格式闸门
def well_formed(task, pred):
    """count、count_color：一个或多个数字（str.isdigit）；left：COLORS 里的一个颜色词；exists：yes 或 no；
    compare：the first / the second / the same；ground：parse_box 能解析，或者正好是 none；
    refer：以 "a " 开头；caption、text：非空。"""
    if task in ("count", "count_color"):
        return pred.isdigit()
    if task == "left":
        return pred in COLORS
    if task == "exists":
        return pred in ("yes", "no")
    if task == "compare":
        return pred in ("the first", "the second", "the same")
    if task == "ground":
        return pred == "none" or parse_box(pred) is not None
    if task == "refer":
        return pred.startswith("a ")
    return len(pred) > 0
# <<<


# >>> 练习 31.2：每个任务的奖励
def task_reward(task, pred, ref):
    """格式不对：0。否则：
        EXACT 里的任务：pred == ref 时 1，否则 0；
        ground：ref 是 none 时，pred 也是 none 得 1，否则 0；ref 是框时，pred 是 none 得 0，否则得两个框的 IoU；
        caption、text：max(0, 1 − NED(pred, ref))。"""
    if not well_formed(task, pred):
        return 0.0
    if task in EXACT:
        return float(pred == ref)
    if task == "ground":
        if ref == "none" or pred == "none":
            return float(pred == ref)
        a, b = parse_box(pred), parse_box(ref)
        return float(box_iou(torch.tensor([a], dtype=torch.float), torch.tensor([b], dtype=torch.float))[0, 0])
    return max(0.0, 1.0 - ned(pred, ref))
# <<<


def doc_reward(kind, pred, ref):
    """kind：text（1 − NED）、formula（CDM，渲染失败得 0）、table（TEDS，解析失败得 0）。"""
    if kind == "text":
        return max(0.0, 1.0 - ned(pred, ref))
    if kind == "formula":
        from ..eval.cdm import cdm
        return cdm(pred, ref)
    if kind == "table":
        from ..eval.teds import teds
        try:
            return teds(pred, ref)
        except Exception:
            return 0.0
    raise ValueError(kind)
