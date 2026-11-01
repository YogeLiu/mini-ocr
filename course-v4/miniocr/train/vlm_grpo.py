"""第 34 章的 VLM 可验证奖励：分层奖励、组内优势、投机探针与 token 级 GRPO 损失。

只依赖 torch 与第 18、24 章已有的评测/机制函数；不加载模型、不下载权重。
调用方负责采样（每个图片 G 个候选）并在 policy 与参考模型上计算 token log-prob。
"""
import torch

from miniocr.eval.postcorrect_eval import extract_formula, score_postcorrect
from miniocr.train.policy_gradient import kl_estimators

# 分层奖励：不可解析 0；可解析 0.2；字符级距离比例不超过 0.2 为 0.5；规范化后完全一致为 1.0。
TIER_REWARDS = (0.0, 0.2, 0.5, 1.0)
CLOSE_RATIO = 0.2


def layered_reward(target, text, *, reward_mode="layered"):
    """从模型原始输出提取公式后打分；返回 reward、tier 与可审计的中间量。

    extract_formula 是启发式（第 18 章），所以 raw 输出与抽取结果都要保留。
    no_parse_bonus 只把 tier-1 奖励置零；层级判定与其余奖励保持不变，供消融对照。
    """
    if reward_mode not in ("layered", "no_parse_bonus"):
        raise ValueError(f"未知奖励模式: {reward_mode}")
    pred = extract_formula(text)
    row = score_postcorrect([_Example(target)], [pred], bootstrap_samples=0)['rows'][0]
    if not pred.strip() or not row['parses']:
        tier = 0
    elif row['em_norm']:
        tier = 3
    elif row['edit_distance'] <= CLOSE_RATIO * max(row['ref_length'], 1):
        tier = 2
    else:
        tier = 1
    return dict(reward=0.0 if reward_mode == "no_parse_bonus" and tier == 1 else TIER_REWARDS[tier], tier=tier, pred=pred,
                em_norm=bool(row['em_norm']), parses=bool(row['parses']),
                edit_distance=row['edit_distance'], ref_length=row['ref_length'])


class _Example:
    """只携带 target 的最小样本，满足 score_postcorrect 所需字段。"""
    def __init__(self, target):
        self.id = 'grpo'; self.split = 'train'; self.source = 'latex-formulas'
        self.target = target; self.is_clean = True; self.error_types = ()


# >>> 练习 34.1：组内相对优势。奖励全相同的组没有信号，必须排除而不是除以零。
def group_advantages(rewards, eps=1e-4):
    """rewards: [P, G]。返回 (advantages [P,G], informative [P] bool)。

    每组减去组均值再除以组标准差；标准差为 0 的组优势为 0 且 informative=False。
    """
    if rewards.ndim != 2:
        raise ValueError('rewards 必须是 [prompts, group] 二维张量')
    mean = rewards.mean(-1, keepdim=True)
    std = rewards.std(-1, unbiased=False, keepdim=True)
    informative = (std.squeeze(-1) > 0)
    advantages = (rewards - mean) / (std + eps)
    advantages = torch.where(informative[:, None], advantages, torch.zeros_like(advantages))
    return advantages, informative
# <<<


def completion_mask(tokens, eos_ids):
    """response 掩码：保留到第一个 EOS（含）为止；EOS 与 pad 共用 ID 时也正确。

    tokens: [B, T] 生成的 completion；eos_ids: 可迭代的 EOS token ID。
    """
    is_eos = torch.zeros_like(tokens, dtype=torch.bool)
    for eos in eos_ids:
        is_eos |= tokens == eos
    # 第一个 EOS 之前（不含）都是有效动作；第一个 EOS 本身也是动作。
    before = torch.cumsum(is_eos.long(), dim=-1) - is_eos.long()
    return (before == 0).long()


def token_logprobs(logits, targets):
    """logits [B,T,V] 与 targets [B,T] 对齐（已切片到 completion 位置），返回 [B,T] log-prob。"""
    if logits.shape[:-1] != targets.shape:
        raise ValueError('logits 与 targets 形状不一致')
    logp = logits.float().log_softmax(-1)
    return logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


# >>> 练习 34.2：token 级 GRPO 损失与 KL 惩罚的方向。
def grpo_loss(policy_lp, ref_lp, mask, advantages, beta=0.04, normalizer=None):
    """返回 (loss, stats)。

    policy_lp/ref_lp/mask: [B,T]；advantages: [B]，已停止梯度。
    KL 使用 k3 估计：采样来自 policy，log_ratio = log(ref) - log(policy)。
    normalizer 默认为 mask 的总 token 数；多组累积时由调用方传入全 batch 的 token 数。
    """
    mask = mask.to(policy_lp.dtype)
    total = mask.sum() if normalizer is None else torch.as_tensor(normalizer, dtype=policy_lp.dtype)
    if float(total) <= 0:
        raise ValueError('没有有效 completion token，无法计算损失')
    adv = advantages.detach().to(policy_lp.dtype)[:, None]
    pg = -(adv * policy_lp * mask).sum() / total
    k3 = kl_estimators((ref_lp.detach() - policy_lp).float())['k3'].to(policy_lp.dtype)
    kl = (k3 * mask).sum() / total
    loss = pg + beta * kl
    with torch.no_grad():
        stats = dict(pg=float(pg), kl=float((k3*mask).sum()/mask.sum().clamp_min(1)),
                     tokens=int(mask.sum()))
    return loss, stats
# <<<


PROBE_OUTPUTS = ('', 'x', '\\frac{a}{b}', '\\alpha', 'a+b=c')


# >>> 练习 34.3：投机探针。固定的退化输出若也能拿到奖励，说明奖励可被钻空子。
def probe_rewards(target):
    """对固定退化字符串打分；返回 {text: tier}，用于训练前后比较。"""
    return {text: layered_reward(target, text)['tier'] for text in PROBE_OUTPUTS}
# <<<


def degenerate(text):
    """粗略的退化标记：空、过短、或同一符号大量重复。仅用于统计，不参与奖励。"""
    s = extract_formula(text).strip()
    if len(s) < 3:
        return True
    tokens = s.split()
    return len(tokens) >= 8 and len(set(tokens)) <= 2
