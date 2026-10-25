"""第 23 章的自动检查。"""
import functools
import numpy as np
import torch

from miniocr.data.corpus import text_vocab
from miniocr.data.instruct import TASKS, apply_chat_template, box_text, draw_text_scene, make_task, parse_box
from miniocr.data.shapes import draw_scene, sample_scene
from miniocr.models.gpt import GPT, GPTConfig
from miniocr.models.vit import ViT, ViTConfig
from miniocr.models.vlm import VLM, MLPProjector


@functools.cache
def vocab():
    return text_vocab()        # 用到时再建（挖空版里练习 1.2 没做完时，只让用到它的测试失败）


def test_chat_template_multi_turn():
    msgs = [{"role": "user", "content": "<image>Color?"}, {"role": "assistant", "content": "red"},
            {"role": "user", "content": "Count?"}, {"role": "assistant", "content": "3"}]
    ids, tg = apply_chat_template(vocab(), msgs, 4)
    assert len(ids) == len(tg) and ids[0] == vocab().stoi["<bos>"]
    assert ids.count(vocab().stoi["<img>"]) == 4 and ids.count(vocab().stoi["<eos>"]) == 2
    assert vocab().decode(ids) == "User: Color?\nAssistant: redUser: Count?\nAssistant: 3"
    sup = [t for t in tg if t != -100]
    assert sup == vocab().encode("red") + [vocab().stoi["<eos>"]] + vocab().encode("3") + [vocab().stoi["<eos>"]]   # 两轮回答都计算损失
    for t, v in enumerate(tg):                                            # 每个被监督的位置都在预测下一个 token
        if v != -100:
            assert ids[t + 1] == v
    two, _ = apply_chat_template(vocab(), [{"role": "user", "content": "A<image>B<image>?"}, {"role": "assistant", "content": "x"}], 3)
    assert two.count(vocab().stoi["<img>"]) == 6


def test_box_text_roundtrip():
    assert box_text((3, 14, 15, 92)) == "[3,14,15,92]"
    assert parse_box("[3,14,15,92]") == (3, 14, 15, 92)
    assert parse_box("none") is None and parse_box("[1,2,3]") is None and parse_box("[a,b,c,d]") is None


def test_tasks_are_answerable():
    rng = np.random.default_rng(0)
    for _ in range(50):
        objs = sample_scene(rng, 64, 64, 1, 3)
        img = draw_scene(objs, rng, 64, 64)
        img2, word, box = draw_text_scene(img, objs, rng)
        for t in TASKS:
            q, a = make_task(t, objs, rng, objs2=objs, word=word or "X")
            assert q.count("<image>") == (2 if t == "compare" else 1) and a
            if t == "ground" and a != "none":
                b = parse_box(a)
                assert any(tuple(o["box"]) == b for o in objs)
        if word:
            x0, y0, x1, y1 = box
            assert (img2[y0:y1, x0:x1].sum(-1) < 100).any()                 # 字真的写上去了
            assert all(x1 <= o["box"][0] or o["box"][2] <= x0 or y1 <= o["box"][1] or o["box"][3] <= y0 for o in objs)


def test_vlm_multi_image():
    torch.manual_seed(0)
    vit = ViT(ViTConfig(img_size=16, patch=8, d_model=32, n_layer=1, n_head=4, pool="mean"))   # 4 个 patch
    lm = GPT(GPTConfig(vocab_size=len(vocab()), block_size=64, n_layer=1, n_head=2, d_model=32))
    m = VLM(vit, MLPProjector(32, 32), lm, vocab().stoi["<img>"])
    ids, tg = apply_chat_template(vocab(), [{"role": "user", "content": "<image>,<image>"}, {"role": "assistant", "content": "a"}], 4)
    ids = torch.tensor([ids])
    imgs = torch.randn(1, 2, 3, 16, 16)
    emb = m.embed(imgs, ids)
    pos = (ids[0] == vocab().stoi["<img>"]).nonzero()[:, 0]
    v = m.visual_tokens(imgs[0])                                           # (2, 4, 32)
    assert torch.allclose(emb[0, pos[:4]], v[0]) and torch.allclose(emb[0, pos[4:]], v[1])
