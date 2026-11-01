"""Paired image/text retrieval, with deterministic tie breaking and bootstrap CI."""
import numpy as np
import torch


# >>> 练习 30.3：双向 Recall@K；图库大小是评测协议的一部分
def retrieval_metrics(scores, ks=(1, 5, 10)):
    """One distinct positive per query; ties retain ascending gallery index.

    Scores are cosine similarities (temperature/bias cannot change their rank).
    Confidence intervals resample queries with a fixed gallery, not training runs.
    """
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1] or not scores.numel():
        raise ValueError('expected a nonempty paired square gallery')
    if any(k < 1 for k in ks):
        raise ValueError('K must be positive')
    scores = scores.detach().cpu()
    n = len(scores)
    result = {'gallery_n': n}
    for direction, matrix in (('i2t', scores), ('t2i', scores.T)):
        ranks = matrix.argsort(dim=1, descending=True, stable=True)
        hits = ranks.eq(torch.arange(n)[:, None])
        for k in ks:
            ok = hits[:, :min(k, n)].any(1).numpy()
            boot = np.random.default_rng(30).choice(ok, (1000, n)).mean(1)
            result[f'{direction}_r{k}'] = float(ok.mean())
            result[f'{direction}_r{k}_ci95'] = np.quantile(boot, [.025,.975]).tolist()
    return result
# <<<
