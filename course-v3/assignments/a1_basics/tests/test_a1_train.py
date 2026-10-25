"""A1 问题 train：可复现、能续训。只跑几步，在 CPU 上几秒内完成。"""
import numpy as np
import torch

from miniocr.data.tasks import glyph_arrays

from ._impl import adapters as A

_DATA = None


def _data():
    global _DATA
    if _DATA is None:
        X, y = glyph_arrays(np.random.default_rng(100), n_per_class=2, distort=1.0, noise=0.05, flat=False)
        Xv, yv = glyph_arrays(np.random.default_rng(101), n_per_class=1, distort=1.0, noise=0.05, flat=False)
        _DATA = (X[:, None], y, Xv[:, None], yv)
    return _DATA


def _cfg(**kw):
    c = dict(channels=(4, 8, 8), batch_size=16, max_lr=1e-2, min_lr=1e-3, warmup_iters=2, max_iters=6,
             lr_decay_iters=6, weight_decay=0.01, clip=1.0, label_smoothing=0.0, eval_every=3, seed=0,
             device="cpu", out_dir="")
    c.update(kw)
    return c


def _losses(h):
    return [r["loss"] for r in h]


def test_train_is_reproducible():
    h1, _ = A.run_train(_cfg(), _data())
    h2, _ = A.run_train(_cfg(), _data())
    assert [r["iter"] for r in h1] == list(range(1, 7))
    assert np.all(np.isfinite(_losses(h1)))
    assert _losses(h1) == _losses(h2)
    h3, _ = A.run_train(_cfg(seed=1), _data())
    assert _losses(h1) != _losses(h3)


def test_resume_matches_uninterrupted(tmp_path):
    full, m_full = A.run_train(_cfg(), _data())
    d = str(tmp_path / "run")
    first, _ = A.run_train(_cfg(max_iters=3, out_dir=d), _data())
    rest, m_rest = A.run_train(_cfg(out_dir=d), _data(), resume=f"{d}/ckpt.pt")
    assert [r["iter"] for r in rest] == [4, 5, 6]                    # 从第 4 步接着跑，不是从头
    np.testing.assert_allclose(_losses(first) + _losses(rest), _losses(full), rtol=1e-6)
    for p, q in zip(m_full.state_dict().values(), m_rest.state_dict().values()):
        torch.testing.assert_close(p, q)
