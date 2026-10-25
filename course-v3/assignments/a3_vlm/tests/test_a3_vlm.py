"""A3 第四部分：2×2 合并、三种连接器、聊天模板、视觉 token 填充、带 KV cache 的生成。"""
import pytest
import torch

from ._impl import adapters as A


def test_merge_2x2_loops():
    B, h, w, d = 2, 4, 6, 3
    x = torch.randn(B, h * w, d)
    out = A.run_merge_2x2(x, (h, w))
    assert out.shape == (B, (h // 2) * (w // 2), 4 * d)
    for i in range(h // 2):
        for j in range(w // 2):
            n = i * (w // 2) + j
            parts = [x[:, (2 * i + a) * w + 2 * j + b] for a in (0, 1) for b in (0, 1)]   # 左上、右上、左下、右下
            assert torch.equal(out[:, n], torch.cat(parts, -1))


@pytest.mark.parametrize("kind,n_out", [("mlp", 16), ("merge", 4), ("qformer", 16)])
def test_connector_shapes(kind, n_out):
    torch.manual_seed(0)
    conn = A.get_connector(kind, 24, 40)
    y = conn(torch.randn(3, 16, 24), (4, 4))
    assert y.shape == (3, n_out, 40)
    y.sum().backward()
    assert all(p.grad is not None for p in conn.parameters() if p.requires_grad)


def test_qformer_ignores_patch_order_and_count():
    torch.manual_seed(0)
    conn = A.get_connector("qformer", 24, 40)
    x = torch.randn(2, 16, 24)
    perm = torch.randperm(16)
    assert torch.allclose(conn(x, (4, 4)), conn(x[:, perm], (4, 4)), atol=1e-5)
    assert conn(torch.randn(2, 36, 24), (6, 6)).shape == (2, 16, 40)        # patch 再多，输出个数不变


def test_chat_template():
    enc = lambda s: [ord(c) for c in s]                                      # noqa: E731
    bos, eos, img = 1, 2, 3
    msgs = [{"role": "user", "content": "<image>Q?"}, {"role": "assistant", "content": "ab"},
            {"role": "user", "content": "R"}, {"role": "assistant", "content": "c"}]
    ids, tg = A.run_chat_template(msgs, 2, enc, bos, eos, img)
    expect = ([bos] + enc("User: ") + [img, img] + enc("Q?\n") + enc("Assistant: ") + enc("ab") + [eos]
              + enc("User: R\n") + enc("Assistant: ") + enc("c") + [eos])
    assert ids == expect
    assert len(tg) == len(ids) and tg[-1] == -100
    sup = [t for t in range(len(ids) - 1) if tg[t] != -100]
    a1 = 1 + 6 + 2 + 3 + 11                                                  # 第一个回答 "a" 在 ids 里的下标
    a2 = a1 + 3 + 8 + 11                                                     # "c"
    assert sup == [a1 - 1, a1, a1 + 1, a2 - 1, a2]                           # 预测 a、b、eos、c、eos 的位置
    assert [tg[t] for t in sup] == enc("ab") + [eos] + enc("c") + [eos]


def test_scatter_visual():
    torch.manual_seed(0)
    emb = torch.randn(2, 6, 4)
    vis = torch.randn(2, 2, 4, requires_grad=True)
    mask = torch.tensor([[0, 1, 0, 1, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    before = emb.clone()
    out = A.run_scatter_visual(emb, vis, mask)
    assert torch.equal(emb, before)                                          # 不改输入
    assert torch.equal(out[0, 1], vis[0, 0]) and torch.equal(out[0, 3], vis[0, 1])
    assert torch.equal(out[1, 0], vis[1, 0]) and torch.equal(out[1, 1], vis[1, 1])
    assert torch.equal(out[~mask], emb[~mask])
    out.sum().backward()
    assert torch.equal(vis.grad, torch.ones_like(vis))
    with pytest.raises(ValueError):
        A.run_scatter_visual(emb, vis, torch.zeros(2, 6, dtype=torch.bool))


N_VIS = {"mlp": 16, "merge": 4, "qformer": 16}


def _prompt(n, B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    text = torch.randint(10, 50, (B, 5), generator=g)
    return torch.cat([text[:, :1], torch.full((B, n), 3), text[:, 1:]], 1)


@pytest.mark.parametrize("kind", ["mlp", "merge", "qformer"])
def test_vlm_forward(kind):
    torch.manual_seed(0)
    m = A.get_vlm(50, 3, kind)
    imgs = torch.randn(2, 3, 32, 32)
    ids = _prompt(N_VIS[kind])
    logits = A.run_vlm_logits(m, imgs, ids)
    assert logits.shape == (2, ids.shape[1], 50)
    # 换一张图，图之后的位置的 logits 要变（视觉 token 确实接进去了）；图之前的不变（因果）
    other = A.run_vlm_logits(m, torch.randn(2, 3, 32, 32), ids)
    assert torch.allclose(logits[:, 0], other[:, 0], atol=1e-5)
    assert not torch.allclose(logits[:, -1], other[:, -1], atol=1e-4)
    with pytest.raises(ValueError):
        A.run_vlm_logits(m, imgs, _prompt(N_VIS[kind] + 1))


@pytest.mark.parametrize("kind", ["merge", "qformer"])
def test_vlm_generate_matches_no_cache(kind):
    torch.manual_seed(0)
    m = A.get_vlm(50, 3, kind).eval()
    imgs = torch.randn(2, 3, 32, 32)
    ids = _prompt(N_VIS[kind])
    out = A.run_vlm_generate(m, imgs, ids, 8)
    assert out.shape == (2, 8)
    seq = ids
    with torch.no_grad():
        for _ in range(8):
            nxt = A.run_vlm_logits(m, imgs, seq)[:, -1].argmax(-1, keepdim=True)
            seq = torch.cat([seq, nxt], 1)
    assert torch.equal(out, seq[:, ids.shape[1]:])
