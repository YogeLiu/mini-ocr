"""第 22 章的自动检查。"""
import functools
import torch

from miniocr.data.corpus import text_vocab
from miniocr.models.gpt import GPT, GPTConfig
from miniocr.models.vit import ViT, ViTConfig
from miniocr.models.vlm import (VLM, MergeProjector, MLPProjector, QFormer, build_example, merge_2x2, pad_batch,
                                set_trainable)


@functools.cache
def vocab():
    return text_vocab()        # 用到时再建（挖空版里练习 1.2 没做完时，只让用到它的测试失败）


def small_vlm(connector="mlp"):
    torch.manual_seed(0)
    vit = ViT(ViTConfig(img_size=16, patch=4, d_model=32, n_layer=1, n_head=4, pool="mean"))   # 4×4 = 16 个 patch
    lm = GPT(GPTConfig(vocab_size=len(vocab()), block_size=128, n_layer=2, n_head=2, d_model=32))
    conn = {"mlp": MLPProjector(32, 32), "merge": MergeProjector(32, 32), "qformer": QFormer(32, 32, n_query=6)}[connector]
    return VLM(vit, conn, lm, vocab().stoi["<img>"])


def test_merge_2x2():
    x = torch.arange(16).float().reshape(1, 16, 1)                       # 4×4 网格，值 = patch 编号
    y = merge_2x2(x, (4, 4))
    assert y.shape == (1, 4, 4)
    assert y[0].tolist() == [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]
    z = merge_2x2(torch.randn(2, 4 * 8, 3), (4, 8))                       # 长方形网格
    assert z.shape == (2, 8, 12)


def test_build_example_targets():
    ids, tg = build_example(vocab(), 3, "How many?", "2")
    assert len(ids) == len(tg)
    assert ids[0] == vocab().stoi["<bos>"] and ids[1:4] == [vocab().stoi["<img>"]] * 3 and ids[-1] == vocab().stoi["<eos>"]
    assert vocab().decode(ids) == "Q: How many?\nA: 2"
    sup = [(t, i) for t, i in enumerate(tg) if i != -100]
    assert [i for _, i in sup] == [vocab().stoi["2"], vocab().stoi["<eos>"]]          # 只监督答案和 <eos>
    assert ids[sup[0][0]] == vocab().stoi[" "] and ids[sup[0][0] + 1] == vocab().stoi["2"]   # 在 "A: " 的空格处预测答案
    a, b = pad_batch([build_example(vocab(), 3, "Hi", "x"), build_example(vocab(), 3, "Hello", "yz")], 0)
    assert a.shape == b.shape and (b[0, -2:] == -100).all()


def test_embed_places_visual_tokens():
    for kind, n in (("mlp", 16), ("merge", 4), ("qformer", 6)):
        m = small_vlm(kind)
        img = torch.randn(2, 3, 16, 16)
        ids, tg = pad_batch([build_example(vocab(), n, "q", "abc"), build_example(vocab(), n, "qq", "d")], vocab().stoi["<pad>"])
        emb = m.embed(img, ids)
        vis = m.visual_tokens(img)
        assert vis.shape == (2, n, 32)
        assert torch.allclose(emb[0, 1:1 + n], vis[0]) and torch.allclose(emb[1, 1:1 + n], vis[1])
        assert torch.allclose(emb[0, 0], m.lm.tok.weight[vocab().stoi["<bos>"]])
        logits, loss = m(img, ids, tg)
        assert logits.shape == (2, ids.shape[1], len(vocab())) and torch.isfinite(loss)


def test_set_trainable_and_overfit():
    m = small_vlm("mlp")
    n_conn = sum(p.numel() for p in m.connector.parameters())
    assert set_trainable(m, vision=False, lm=False) == n_conn
    img = torch.randn(2, 3, 16, 16)
    ids, tg = pad_batch([build_example(vocab(), 16, "c", "red"), build_example(vocab(), 16, "c", "blue")], vocab().stoi["<pad>"])
    set_trainable(m)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    for _ in range(150):
        loss = m(img, ids, tg)[1]
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
    prompt = torch.tensor([build_example(vocab(), 16, "c", "")[0][:-1]] * 2)     # 去掉 <eos>
    out = m.generate(img, prompt, max_new=6, eos=vocab().stoi["<eos>"])
    assert [vocab().decode(o.tolist()) for o in out] == ["red", "blue"]
