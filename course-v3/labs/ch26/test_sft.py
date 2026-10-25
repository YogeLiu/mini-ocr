"""第 26 章的自动检查。"""
import functools

import torch

from miniocr.data.corpus import text_vocab
from miniocr.data.instruct import apply_chat_template
from miniocr.models.gpt import GPT, GPTConfig
from miniocr.models.vit import ViT, ViTConfig
from miniocr.models.vlm import VLM, MLPProjector
from miniocr.train.sft import (ROLE_ANSWER, ROLE_FORMAT, ROLE_IMAGE, ROLE_USER, chat_roles, embed_packed, pack,
                               packed_lm_forward, packed_mask, pad_weighted, sft_loss, sft_targets, steps_to)


@functools.cache
def vocab():
    return text_vocab()


MSGS = [{"role": "user", "content": "<image>Color?"}, {"role": "assistant", "content": "red"},
        {"role": "user", "content": "Count?"}, {"role": "assistant", "content": "3"}]


def test_chat_roles_matches_template():
    V = vocab()
    ids, roles = chat_roles(V, MSGS, 4)
    ref_ids, ref_tg = apply_chat_template(V, MSGS, 4)
    assert ids == ref_ids and len(roles) == len(ids)
    assert roles.count(ROLE_IMAGE) == 4 and roles[0] == ROLE_FORMAT
    assert "".join(V.itos[i] for i, r in zip(ids, roles) if r == ROLE_USER) == "Color?Count?"
    assert [i for i, r in zip(ids, roles) if r == ROLE_ANSWER] == V.encode("red") + [V.stoi["<eos>"]] + V.encode("3") + [V.stoi["<eos>"]]
    tg, w = sft_targets(ids, roles)                                    # 只算回答：与第 23 章的模板完全相同
    assert tg == ref_tg and all((x > 0) == (t != -100) for x, t in zip(w, tg))


def test_sft_targets_weights():
    V = vocab()
    ids, roles = chat_roles(V, MSGS, 4)
    tg, w = sft_targets(ids, roles, prompt_weight=0.1)
    for t in range(len(ids) - 1):
        r = roles[t + 1]
        assert w[t] == {ROLE_ANSWER: 1.0, ROLE_USER: 0.1, ROLE_FORMAT: 0.1, ROLE_IMAGE: 0.0}[r]
    tg2, w2 = sft_targets(ids, roles, eos_id=V.stoi["<eos>"], train_eos=False)
    assert sum(x > 0 for x in w2) == len(V.encode("red3"))            # 两个 <eos> 不再计算


def test_sft_loss_by_hand():
    # 两条样本：第 0 条 3 个计算损失的位置（每个 ce = ln 4），第 1 条 1 个位置（ce = ln 2）
    logits = torch.zeros(2, 3, 4)
    logits[1, 0] = torch.tensor([0.0, 0.0, -1e9, -1e9])               # 只在两个类之间均匀
    tg = torch.tensor([[0, 1, 2], [1, -100, -100]])
    import math
    tok = sft_loss(logits, tg, mode="token").item()
    smp = sft_loss(logits, tg, mode="sample").item()
    assert abs(tok - (3 * math.log(4) + math.log(2)) / 4) < 1e-5
    assert abs(smp - (math.log(4) + math.log(2)) / 2) < 1e-5
    w = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])               # 权重 0 的位置不计算；−100 的位置权重不起作用
    assert abs(sft_loss(logits, tg, w).item() - (math.log(4) + math.log(2)) / 2) < 1e-5
    z = torch.zeros(1, 2, 4, requires_grad=True)
    sft_loss(z, torch.full((1, 2), -100)).backward()                   # 没有任何计算损失的位置：返回 0、能求导


def test_pack_and_mask():
    ex = [([1] * 3, [5] * 3, [1.0] * 3), ([2] * 4, [6] * 4, [1.0] * 4), ([3] * 2, [7] * 2, [1.0] * 2), ([4] * 5, [8] * 5, [1.0] * 5)]
    p = pack(ex, 8, pad_id=0)
    assert p["order"] == [[0, 1], [2, 3]]           # 3 + 4 = 7；再放长 2 的会到 9 > 8，开新箱子；2 + 5 = 7 ≤ 8
    assert p["ids"].tolist() == [[1, 1, 1, 2, 2, 2, 2, 0], [3, 3, 4, 4, 4, 4, 4, 0]]
    assert p["seg"].tolist() == [[1, 1, 1, 2, 2, 2, 2, 0], [1, 1, 2, 2, 2, 2, 2, 0]]
    assert p["pos"].tolist() == [[0, 1, 2, 0, 1, 2, 3, 0], [0, 1, 0, 1, 2, 3, 4, 0]]
    assert p["targets"][0, 7].item() == -100 and p["weights"][0, 7].item() == 0.0
    m = packed_mask(p["seg"])
    assert m.shape == (2, 1, 8, 8)
    assert m[0, 0, 4].tolist() == [False, False, False, True, True, False, False, False]   # 第 2 条的第 1 个位置
    assert m[0, 0, 7].tolist() == [False] * 7 + [True]                                   # <pad> 只看自己


def tiny_lm():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=len(vocab()), block_size=64, n_layer=2, n_head=2, d_model=32)).eval()


def test_packed_equals_separate():
    lm = tiny_lm()
    g = torch.Generator().manual_seed(1)
    seqs = [torch.randint(0, len(vocab()), (n,), generator=g).tolist() for n in (5, 9, 3, 7)]
    p = pack([(s, s, [1.0] * len(s)) for s in seqs], 16, pad_id=0)
    with torch.no_grad():
        out = packed_lm_forward(lm, lm.tok(p["ids"]), p["seg"], p["pos"])
        for b, ks in enumerate(p["order"]):
            for k in ks:
                pos = (p["seg"][b] == ks.index(k) + 1).nonzero().flatten()
                ref = lm(torch.tensor([seqs[k]]))[0][0]
                assert torch.allclose(out[b, pos], ref, atol=1e-5)


def test_embed_packed_order():
    V = vocab()
    torch.manual_seed(0)
    vit = ViT(ViTConfig(img_size=16, patch=8, d_model=32, n_layer=1, n_head=4, pool="mean"))     # 2×2 = 4 个 patch
    lm = GPT(GPTConfig(vocab_size=len(V), block_size=64, n_layer=1, n_head=2, d_model=32))
    vlm = VLM(vit, MLPProjector(32, 32), lm, V.stoi["<img>"])
    exs = []
    for q in ("A", "BB", "C"):
        ids, roles = chat_roles(V, [{"role": "user", "content": "<image>" + q}, {"role": "assistant", "content": "x"}], 4)
        exs.append((ids, *sft_targets(ids, roles)))
    p = pack(exs, 40, V.stoi["<pad>"])
    imgs = torch.randn(3, 3, 16, 16)
    order = [k for ks in p["order"] for k in ks]
    emb = embed_packed(vlm, imgs[order], p["ids"])
    v = vlm.visual_tokens(imgs)
    for b, ks in enumerate(p["order"]):
        for j, k in enumerate(ks):
            where = ((p["seg"][b] == j + 1) & (p["ids"][b] == V.stoi["<img>"])).nonzero().flatten()
            assert torch.allclose(emb[b, where], v[k], atol=1e-6)


def test_pad_weighted_and_steps_to():
    ids, tg, w = pad_weighted([([1, 2], [2, -100], [1.0, 0.0]), ([3], [-100], [0.0])], pad_id=9)
    assert ids.tolist() == [[1, 2], [3, 9]] and tg.tolist() == [[2, -100], [-100, -100]] and w[1].tolist() == [0.0, 0.0]
    assert steps_to([(100, 0.1), (200, 0.6), (300, 0.9)], 0.5) == 200 and steps_to([(100, 0.1)], 0.5) is None
