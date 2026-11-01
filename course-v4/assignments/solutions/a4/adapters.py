"""维护者参考答案：严格接口验证，张量实现独立于 miniocr。"""
import math
import torch


def _tensors(*xs, ndim=None, same=True):
    if any(not isinstance(x, torch.Tensor) or not x.is_floating_point() or not x.numel()
           or not torch.isfinite(x).all() or (ndim is not None and x.ndim != ndim) for x in xs):
        raise ValueError('需要非空有限浮点张量及正确维数')
    if any(x.device != xs[0].device or x.dtype != xs[0].dtype or
           (same and x.shape != xs[0].shape) for x in xs):
        raise ValueError('形状、设备或 dtype 不匹配')


def _scalar(x, lo, hi=math.inf, strict=False):
    if not isinstance(x, (float, int)) or not math.isfinite(x) or not (lo < x < hi if strict else lo <= x < hi):
        raise ValueError('标量超出定义域')


def bradley_terry_loss(chosen, rejected):
    _tensors(chosen, rejected, ndim=1)
    return torch.logaddexp(torch.zeros_like(chosen), rejected-chosen)


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=.1):
    _tensors(policy_chosen, policy_rejected, ref_chosen, ref_rejected, ndim=1)
    _scalar(beta, 0, strict=True)
    rc = beta*(policy_chosen-ref_chosen.detach())
    rr = beta*(policy_rejected-ref_rejected.detach())
    return bradley_terry_loss(rc, rr), rc, rr


def response_logprob(logits, labels):
    _tensors(logits, ndim=3)
    if labels.shape != logits.shape[:2] or labels.dtype != torch.long or labels.device != logits.device or logits.shape[1] < 2:
        raise ValueError('标签形状、类型、设备或序列长度错误')
    if ((labels != -100) & ((labels < 0) | (labels >= logits.shape[-1]))).any():
        raise ValueError('标签越界')
    x, y = logits[:, :-1], labels[:, 1:]
    lp = x - torch.logsumexp(x, -1, keepdim=True)
    return lp.gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1).masked_fill(y == -100, 0).sum(-1)


def group_advantages(rewards, group_size, *, scale=True, eps=1e-4):
    _tensors(rewards, ndim=1)
    _scalar(eps, 0, strict=True)
    if type(group_size) is not int or group_size < 1 or rewards.numel() % group_size:
        raise ValueError('非法组大小')
    r = rewards.reshape(-1, group_size)
    c = r-r.mean(-1, keepdim=True)
    return (c/(r.std(-1, correction=0, keepdim=True)+eps) if scale else c).reshape(-1)


def kl_estimators(log_ratio):
    _tensors(log_ratio)
    return dict(k1=-log_ratio, k2=log_ratio.square()/2, k3=torch.expm1(log_ratio)-log_ratio)


def grpo_loss(logp_new, logp_old, logp_ref, advantages, mask, *, epsilon=.2, beta=.04):
    _tensors(logp_new, logp_old, logp_ref, ndim=2)
    _tensors(logp_new, advantages, same=False)
    _scalar(epsilon, 0, 1, strict=True)
    _scalar(beta, 0)
    if advantages.shape != (logp_new.shape[0],) or mask.shape != logp_new.shape or mask.device != logp_new.device or not ((mask == 0) | (mask == 1)).all():
        raise ValueError('优势或 mask 错误')
    keep = mask.bool()
    # mask 先于 exp：被屏蔽的极端有限值不应溢出污染 loss/梯度。
    new = logp_new.masked_fill(~keep, 0)
    old = logp_old.detach().masked_fill(~keep, 0)
    ref = logp_ref.detach().masked_fill(~keep, 0)
    ratio = (new-old).exp()
    adv = advantages.detach()[:, None]
    surrogate = torch.minimum(ratio*adv, ratio.clamp(1-epsilon, 1+epsilon)*adv)
    kl = kl_estimators(ref-new)['k3']
    loss = ((-surrogate+beta*kl).masked_fill(~keep, 0).sum(-1)/keep.sum(-1).clamp(min=1)).mean()
    def avg(x):
        return float(x.detach().masked_fill(~keep, 0).sum()/keep.sum().clamp(min=1))
    return loss, dict(clip_fraction=avg(((ratio-1).abs()>epsilon).to(new.dtype)), kl_mean=avg(kl), ratio_mean=avg(ratio))


def reward(completion, source, target, *, parse, normalize, error_rate, format_issues, mode='layered'):
    if any(not isinstance(t, str) for t in (completion, source, target)) or mode not in ('layered', 'parse_only'):
        raise ValueError('字符串或奖励模式错误')
    if not completion.strip() or not parse(completion):
        return 0.
    if mode == 'parse_only':
        return 1.
    if format_issues(completion):
        return .1
    if normalize(completion) == normalize(target):
        return 1.
    base, pred = error_rate(target, source), error_rate(target, completion)
    if not all(math.isfinite(x) and x >= 0 for x in (base, pred)):
        raise ValueError('非法编辑率')
    return .1 if base == 0 else .1+.4*min(1., max(0., (base-pred)/base))


def hack_probes(source, target):
    if not isinstance(source, str) or not isinstance(target, str):
        raise ValueError('需要字符串')
    return [('empty', ''), ('single_symbol', 'x'), ('copy_source', source),
            ('copy_with_prose', f'Here is the corrected formula: {source}'),
            ('repeat_source', ' '.join([source]*3)), ('gold_answer', target)]
