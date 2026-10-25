"""第 21 章的自动检查。"""
import math

import numpy as np
import torch

from miniocr.boxes import average_precision, box_iou, nms, xy_cut
from miniocr.data.page import all_lines, sample_page
from miniocr.models.centernet import centernet_targets, decode, focal_loss


def test_box_iou_by_hand():
    a = torch.tensor([[0.0, 0, 10, 10], [5, 5, 15, 15]])
    b = torch.tensor([[0.0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    iou = box_iou(a, b)
    assert iou.shape == (2, 3)
    assert math.isclose(iou[0, 0].item(), 1.0)
    assert math.isclose(iou[0, 1].item(), 50 / 150, rel_tol=1e-6)      # 交 5×10 = 50，并 100 + 100 − 50
    assert iou[0, 2].item() == 0.0 and iou[1, 2].item() == 0.0
    assert math.isclose(iou[1, 0].item(), 25 / 175, rel_tol=1e-6)


def test_nms():
    boxes = torch.tensor([[0.0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30], [0, 0, 9, 10]])
    scores = torch.tensor([0.8, 0.9, 0.7, 0.6])
    assert nms(boxes, scores, 0.5).tolist() == [1, 2]                   # 0、3 与 1 重叠太多被删掉
    assert nms(boxes, scores, 0.95).tolist() == [1, 0, 2, 3]            # 阈值很高时都保留，按分数排序


def test_average_precision():
    gts = [torch.tensor([[0.0, 0, 10, 10], [20, 20, 30, 30]]), torch.tensor([[0.0, 0, 10, 10]])]
    perfect = [(g.clone(), torch.ones(len(g))) for g in gts]
    assert math.isclose(average_precision(perfect, gts)["ap"], 1.0)
    preds = [(torch.tensor([[0.0, 0, 10, 10], [0, 0, 10, 10], [50, 50, 60, 60]]), torch.tensor([0.9, 0.8, 0.7])),
             (torch.tensor([[0.0, 0, 10, 10]]), torch.tensor([0.6]))]
    r = average_precision(preds, gts)
    # 排序：0.9 TP、0.8 FP（重复）、0.7 FP、0.6 TP；召回率 1/3、1/3、1/3、2/3；精确率 1、1/2、1/3、1/2
    # 单调化后 1、1/2、1/2、1/2；AP = 1/3·1 + 1/3·1/2 = 0.5
    assert math.isclose(r["ap"], 0.5, rel_tol=1e-6)
    assert math.isclose(r["recall"], 2 / 3, rel_tol=1e-6) and math.isclose(r["precision"], 0.5, rel_tol=1e-6)


def test_xy_cut_two_columns():
    boxes = [(10, 10, 300, 30),                                          # 标题（跨两栏）
             (10, 50, 140, 100), (160, 50, 300, 80), (10, 120, 140, 160), (160, 100, 300, 160)]
    assert xy_cut(boxes) == [0, 1, 3, 2, 4]                               # 标题 → 左栏 → 右栏
    rng = np.random.default_rng(0)
    ok = 0
    for _ in range(50):
        p = sample_page(rng, columns=2)
        b = [x["box"] for x in p["blocks"]]
        ok += xy_cut(b) == list(range(len(b)))
    assert ok >= 35


def test_targets_and_decode_roundtrip():
    boxes = torch.tensor([[8.0, 12, 72, 28], [100, 40, 140, 80]])
    labels = torch.tensor([0, 1])
    heat, size, off, mask = centernet_targets(boxes, labels, 2, (24, 40), stride=4)
    assert heat.shape == (2, 24, 40) and mask.sum() == 2
    assert heat[0, 5, 10] == 1.0 and heat[1, 15, 30] == 1.0              # 中心 (40, 20)/4 = (10, 5)；(120, 60)/4 = (30, 15)
    assert 0 < heat[0, 5, 12] < 1 and heat[0, 5, 12] > heat[0, 7, 10]    # 宽的框：横向衰减慢
    assert math.isclose(size[0, 5, 10].item(), math.log(64), rel_tol=1e-6)
    logits = torch.where(heat == 1, torch.tensor(5.0), torch.tensor(-5.0))
    res = decode(logits, size, off, stride=4, thr=0.5)
    assert torch.allclose(res[0][0], boxes[:1], atol=1e-4) and torch.allclose(res[1][0], boxes[1:], atol=1e-4)
    assert focal_loss(logits * 4, heat).item() < focal_loss(torch.zeros_like(heat), heat).item()


def test_page_annotations():
    p = sample_page(np.random.default_rng(1))
    assert p["img"].shape == (384, 576) and p["img"].dtype == np.uint8
    for b in p["blocks"]:
        x0, y0, x1, y1 = b["box"]
        for l in b["lines"]:
            lx0, ly0, lx1, ly1 = l["box"]
            assert x0 <= lx0 and y0 <= ly0 and lx1 <= x1 and ly1 <= y1    # 行在区块里
            assert (p["img"][ly0:ly1, lx0:lx1] < 128).any()               # 框里有墨迹
    assert len(all_lines(p)) == sum(len(b["lines"]) for b in p["blocks"])
