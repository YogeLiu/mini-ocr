"""第 34 章的自动检查。"""
import numpy as np
import torch

from miniocr.models.gpt import GPT, GPTConfig
from miniocr.xiaomi.anchor import MaskHead, box_to_grid_mask, dice_loss, lm_hidden, ssa_loss
from miniocr.xiaomi.mixrl import kie_reward, mix_reward, parse_page, sampling_weights
from miniocr.xiaomi.mtp import MTPHead, mtp_loss


def test_box_to_grid_mask():
    m = box_to_grid_mask((0, 0, 16, 8))                       # 正好盖住第 0 行的前两个格子
    assert m.shape == (64,) and m[:2].tolist() == [1, 1] and m.sum() == 2
    m = box_to_grid_mask((4, 4, 12, 12))                      # 每个格子只盖住 1/4：都不到 0.3
    assert m.sum() == 1 and m[0] == 1                         # 退回 cover 最大的（并列取下标小的）
    m = box_to_grid_mask((4, 4, 12, 12), min_cover=0.25)
    assert sorted(torch.nonzero(m).flatten().tolist()) == [0, 1, 8, 9]
    m = box_to_grid_mask((56, 56, 64, 64))
    assert m[63] == 1 and m.sum() == 1


def test_dice_loss():
    t = torch.tensor([[1.0, 0, 0, 0]])
    assert dice_loss(torch.tensor([[20.0, -20, -20, -20]]), t) < 0.01
    assert dice_loss(torch.tensor([[-20.0, 20, 20, 20]]), t) > 0.7
    # 手算：p = 0.5 处处：dice = (2·0.5 + 1) / (2 + 1 + 1) = 0.5
    assert abs(dice_loss(torch.zeros(1, 4), t).item() - 0.5) < 1e-6


def test_mask_head_and_ssa():
    torch.manual_seed(0)
    h = MaskHead(16, 8)
    q, v = torch.randn(3, 16), torch.randn(3, 10, 16)
    lg = h(q, v)
    assert lg.shape == (3, 10)
    ref = torch.einsum("bk,bnk->bn", h.q(q), h.k(v)) / 8 ** 0.5 + h.b
    assert torch.allclose(lg, ref)
    tg = (torch.rand(3, 10) > 0.7).float()
    ntp = torch.tensor(2.0, requires_grad=True)
    tot, st = ssa_loss(ntp, lg, tg, torch.tensor([True, False, True]), lam_txt=1.0, lam_seg=0.5)
    sel = torch.tensor([0, 2])
    exp = 2.0 + 0.5 * (dice_loss(lg[sel], tg[sel]) + torch.nn.functional.binary_cross_entropy_with_logits(lg[sel], tg[sel]))
    assert torch.allclose(tot, exp) and set(st) == {"ntp", "dice", "ce"}
    tot, st = ssa_loss(ntp, lg, tg, torch.zeros(3, dtype=torch.bool))
    assert tot.item() == 2.0 and st["dice"] == 0.0
    tot.backward()


def test_mtp():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=20, block_size=32, d_model=16, n_layer=1, n_head=2)
    lm = GPT(cfg)
    ids = torch.randint(0, 20, (2, 9))
    logits, hidden = lm_hidden(lm, lm.tok(ids))
    assert torch.allclose(logits, lm(ids)[0], atol=1e-5)
    head = MTPHead(cfg)
    out = head(lm, hidden, lm.tok(ids))
    assert out.shape == (2, 8, 20)
    # 第 t 个输出不依赖第 t+2 个及以后的输入（因果）
    ids2 = ids.clone()
    ids2[:, 5:] = (ids2[:, 5:] + 1) % 20
    _, h2 = lm_hidden(lm, lm.tok(ids2))
    out2 = head(lm, h2, lm.tok(ids2))
    assert torch.allclose(out[:, :4], out2[:, :4], atol=1e-5)
    targets = torch.cat([ids[:, 1:], torch.full((2, 1), -100)], 1)
    loss = mtp_loss(out, targets)
    exp = torch.nn.functional.cross_entropy(out[:, :7].reshape(-1, 20), ids[:, 2:].reshape(-1))
    assert torch.allclose(loss, exp, atol=1e-5)
    assert mtp_loss(out, torch.full((2, 9), -100)).item() == 0.0


def test_kie_and_mix_reward():
    ref = {"name": "Li Lei", "date": {"issue": "2026-01-02", "expire": "2030-01-02"}}
    assert kie_reward('{"name": "Li Lei", "date": {"issue": "2026-01-02", "expire": "2030-01-02"}}', ref) == 1.0
    assert kie_reward('```json\n{"name": "Li Lei"}\n```', ref) == 1 / 3
    assert kie_reward('{"name": "Li Lei",', ref) == 0.0
    assert kie_reward('["Li Lei"]', ref) == 0.0
    assert abs(kie_reward('{"name": "Li Le"}', {"name": "Li Lei"}) - (1 - 1 / 6)) < 1e-9
    assert parse_page("[title] Hi\n\n[text] a b") == [("title", "Hi"), ("text", "a b")]
    assert parse_page("Hi\n\n[text] a") is None and parse_page("[figure] x") is None
    assert mix_reward("page", "[title] Hi\n\n[text] a b", [("title", "Hi"), ("text", "a b")]) == 1.0
    assert mix_reward("page", "Hi there", [("text", "Hi there")]) == 0.0
    assert mix_reward("text", "abcd", "abce") == 0.75
    assert mix_reward("text", "x" * 3000, "x") == 0.0
    assert mix_reward("vqa", "Paris", ["paris", "PARIS city"]) == 1.0
    assert mix_reward("kie", '{"a": "1"}', {"a": "1"}) == 1.0


def test_sampling_weights():
    a = [0.9, 0.9, 0.5, 1.0, 0.2]
    u = [0.2, 0.8, 0.2, 0.5, 0.5]
    lab = [0, 0, 0, 1, 1]
    p = sampling_weights(a, u, lab, eps=0.0)
    assert abs(p[:3].sum() - 1) < 1e-9 and abs(p[3:].sum() - 1) < 1e-9
    w = np.array([0.9 * 0.8, 0.9 * 0.2, 0.5 * 0.8])
    assert np.allclose(p[:3], w / w.sum())
    sharp = sampling_weights(a, u, lab, tau=0.25, eps=0.0)
    assert sharp[0] > p[0]                                      # 温度越低越集中到最"好"的样本
