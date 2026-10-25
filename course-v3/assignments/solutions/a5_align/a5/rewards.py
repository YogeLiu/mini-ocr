"""A5 参考实现：可验证奖励，都在 [0, 1]；格式不对得 0（格式闸门）。十个任务：A3 的九个 + read。"""
import torch

from miniocr.data.instruct import parse_box
from miniocr.data.shapes import COLORS

from .boxes import box_iou
from .metrics import ned

EXACT = ("count", "count_color", "left", "exists", "compare", "refer")


def well_formed(task, pred):
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


def task_reward(task, pred, ref):
    """格式不对 0；EXACT 完全相同 1；ground：none 对 none 1，框对框 IoU，其余 0；caption、text、read：max(0, 1 − NED)。"""
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
