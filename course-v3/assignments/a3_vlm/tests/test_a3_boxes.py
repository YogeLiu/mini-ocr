"""A3 第三部分：IoU、NMS、AP。对照逐个算的纯 Python 版本和手算的例子。"""
import pytest
import torch

from ._impl import adapters as A


def _iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / u


def _rand_boxes(n, g):
    xy = torch.rand(n, 2, generator=g) * 50
    wh = torch.rand(n, 2, generator=g) * 20 + 1
    return torch.cat([xy, xy + wh], 1)


def test_box_iou_bruteforce():
    g = torch.Generator().manual_seed(0)
    a, b = _rand_boxes(7, g), _rand_boxes(5, g)
    out = A.run_box_iou(a, b)
    assert out.shape == (7, 5)
    ref = torch.tensor([[_iou(x.tolist(), y.tolist()) for y in b] for x in a])
    assert torch.allclose(out, ref, atol=1e-6)


def test_box_iou_cases():
    a = torch.tensor([[0.0, 0, 10, 10]])
    b = torch.tensor([[0.0, 0, 10, 10], [5, 0, 15, 10], [10, 10, 20, 20], [20, 20, 30, 30]])
    assert A.run_box_iou(a, b)[0].tolist() == pytest.approx([1.0, 1 / 3, 0.0, 0.0])


def _nms_ref(boxes, scores, thr):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    keep = []
    for i in order:
        if all(_iou(boxes[i], boxes[k]) <= thr for k in keep):
            keep.append(i)
    return keep


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("thr", [0.3, 0.5, 0.7])
def test_nms_bruteforce(seed, thr):
    g = torch.Generator().manual_seed(seed)
    boxes = _rand_boxes(40, g) * 0.5
    scores = torch.rand(40, generator=g)
    out = A.run_nms(boxes, scores, thr)
    assert out.dtype == torch.long
    assert out.tolist() == _nms_ref(boxes.tolist(), scores.tolist(), thr)


def test_ap_cases():
    gt = [torch.tensor([[0.0, 0, 10, 10], [20, 20, 30, 30]])]
    perfect = [(torch.tensor([[0.0, 0, 10, 10], [20, 20, 30, 30]]), torch.tensor([0.9, 0.8]))]
    assert A.run_average_precision(perfect, gt, 0.5) == pytest.approx(1.0)
    # 分数最高的是错的：精确率 0、1/2、2/3，召回在第 2、3 个时各涨 0.5 → AP = 0.5·(2/3) + 0.5·(2/3)
    one_fp = [(torch.tensor([[40.0, 40, 50, 50], [0, 0, 10, 10], [20, 20, 30, 30]]), torch.tensor([0.95, 0.9, 0.8]))]
    assert A.run_average_precision(one_fp, gt, 0.5) == pytest.approx(2 / 3)
    # 同一个真实框被预测两次：第二次算错
    dup = [(torch.tensor([[0.0, 0, 10, 10], [0, 0, 10, 10]]), torch.tensor([0.9, 0.8]))]
    assert A.run_average_precision(dup, gt, 0.5) == pytest.approx(0.5)


def test_ap_multi_image():
    gts = [torch.tensor([[0.0, 0, 10, 10]]), torch.tensor([[0.0, 0, 10, 10]]), torch.zeros(0, 4)]
    preds = [(torch.tensor([[0.0, 0, 10, 10]]), torch.tensor([0.5])),          # 对，分数低
             (torch.tensor([[1.0, 1, 11, 11]]), torch.tensor([0.9])),          # IoU 0.68：对
             (torch.tensor([[0.0, 0, 5, 5]]), torch.tensor([0.7]))]            # 第三张图没有物体：错
    # 排序：0.9 对（P=1, R=.5）、0.7 错、0.5 对（P=2/3, R=1）→ AP = 0.5·1 + 0.5·(2/3)
    assert A.run_average_precision(preds, gts, 0.5) == pytest.approx(0.5 + 1 / 3)
    assert A.run_average_precision(preds, gts, 0.75) == pytest.approx(0.5 / 3)      # 0.68 < 0.75：只有一个对，在第 3 位（P = 1/3）
