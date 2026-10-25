"""第 20 章的自动检查。"""
import math

import torch
import torch.nn.functional as F

from miniocr.data.corpus import text_vocab
from miniocr.models.clip import DualEncoder, TextEncoder, clip_loss, encode_texts, recall_at_k, siglip_loss, zero_shot
from miniocr.models.vit import ViT, ViTConfig


def test_clip_loss_by_hand():
    zi = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    zt = torch.tensor([[1.0, 0.0], [0.6, 0.8]])
    s = torch.tensor(2.0)
    L = s * zi @ zt.T                                                    # [[2, 1.2], [0, 1.6]]
    row = (-torch.log_softmax(L, 1).diagonal()).mean()
    col = (-torch.log_softmax(L, 0).diagonal()).mean()
    assert math.isclose(clip_loss(zi, zt, s).item(), ((row + col) / 2).item(), rel_tol=1e-6)
    r0 = -math.log(math.exp(2) / (math.exp(2) + math.exp(1.2)))
    r1 = -math.log(math.exp(1.6) / (math.exp(0) + math.exp(1.6)))
    c0 = -math.log(math.exp(2) / (math.exp(2) + math.exp(0)))
    c1 = -math.log(math.exp(1.6) / (math.exp(1.2) + math.exp(1.6)))
    assert math.isclose(clip_loss(zi, zt, s).item(), (r0 + r1 + c0 + c1) / 4, rel_tol=1e-6)
    perfect = torch.eye(4)
    assert clip_loss(perfect, perfect, torch.tensor(100.0)).item() < 1e-6
    assert math.isclose(clip_loss(perfect, perfect, torch.tensor(1e-6)).item(), math.log(4), rel_tol=1e-4)


def test_siglip_loss_by_hand():
    zi = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    zt = torch.tensor([[1.0, 0.0], [0.6, 0.8]])
    s, b = torch.tensor(2.0), torch.tensor(-1.0)
    L = s * zi @ zt.T + b                                                # [[1, 0.2], [-1, 0.6]]
    sig = lambda x: 1 / (1 + math.exp(-x))
    want = -(math.log(sig(1)) + math.log(sig(-0.2)) + math.log(sig(1)) + math.log(sig(0.6))) / 2
    assert math.isclose(siglip_loss(zi, zt, s, b).item(), want, rel_tol=1e-6)
    big = torch.eye(3) * 1.0                                             # 极端值不应溢出
    assert torch.isfinite(siglip_loss(big, big, torch.tensor(1e4), torch.tensor(-5e3)))


def test_zero_shot_and_recall():
    zi = F.normalize(torch.tensor([[1.0, 0.1], [0.1, 1.0], [0.9, 0.5]]), dim=-1)
    zt = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    assert zero_shot(zi, zt).tolist() == [0, 1, 0]
    sim = torch.tensor([[0.9, 0.5, 0.1], [0.8, 0.7, 0.0], [0.2, 0.3, 0.1]])
    assert math.isclose(recall_at_k(sim, 1), 1 / 3, rel_tol=1e-6)                    # 只有第 0 行的第一名是自己
    assert math.isclose(recall_at_k(sim, 2), 2 / 3, rel_tol=1e-6)
    assert recall_at_k(sim, 3) == 1.0
    same = torch.eye(3, dtype=torch.bool)
    same[1, 0] = True                                                    # 候选 0 和 1 的文字相同：查询 1 取到 0 也算对
    assert math.isclose(recall_at_k(sim, 1, same), 2 / 3, rel_tol=1e-6)


def test_text_encoder_ignores_padding():
    V = text_vocab()
    torch.manual_seed(0)
    enc = TextEncoder(len(V), d=32, n_layer=2, n_head=4).eval()
    a = encode_texts(V, ["a red circle"], 64)
    b = encode_texts(V, ["a red circle", "a much longer caption here"], 64)
    assert b.shape[1] > a.shape[1] and (b[0] == 0).any()
    assert torch.allclose(enc(a)[0], enc(b)[0], atol=1e-5)               # 补齐多少都不影响结果


def test_dual_encoder_runs_and_learns():
    V = text_vocab()
    torch.manual_seed(0)
    img = ViT(ViTConfig(img_size=16, patch=8, d_model=32, n_layer=1, n_head=4, pool="mean"))
    txt = TextEncoder(len(V), d=32, n_layer=1, n_head=4)
    for kind in ("clip", "siglip"):
        m = DualEncoder(img, txt, 32, 32, 16, loss=kind)
        x = torch.randn(4, 3, 16, 16)
        ids = encode_texts(V, ["red", "green", "blue", "yellow"], 16)
        opt = torch.optim.Adam(m.parameters(), 3e-3)
        first = None
        for _ in range(60):
            loss = m(x, ids)
            first = first if first is not None else loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert loss.item() < first * 0.5, kind
