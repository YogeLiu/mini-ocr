"""第 23 章：Bradley-Terry 与 DPO 损失的显式实现，用于机制教学、单元测试和 TRL 对拍。

约定：
- 序列 log-prob 是响应 token 的 log-prob 之和（不按长度平均），与 DPO 原文一致。
- reference 模型固定不更新；调用方负责在 no_grad 下计算 reference 的 log-prob。
- 损失对每个偏好对求值，返回逐对向量；批平均由调用方决定。
"""
import torch
import torch.nn.functional as F

from miniocr.train.policy_gradient import sequence_log_prob


# >>> 练习 23.1：Bradley-Terry 偏好似然与 DPO 的隐式奖励
def bradley_terry_loss(score_chosen, score_rejected):
    """逐对 -log sigmoid(s_c - s_r)：chosen 分数越高于 rejected，损失越小。

    s_c 与 s_r 可以是任意实数（奖励模型的标量输出）。P(c>r)=sigmoid(s_c-s_r)。
    """
    return -F.logsigmoid(score_chosen - score_rejected)


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    """DPO 的逐对损失与隐式奖励：r = beta*(log pi - log ref)，loss = -logsigmoid(r_c - r_r)。

    返回 (loss, reward_chosen, reward_rejected)，三者形状均为 [B]。
    所有 log-prob 为序列和（同一口径）；beta 必须为正。reference 的梯度应已被截断。
    """
    if not beta > 0:
        raise ValueError("beta 必须为正")
    for name, value in (("policy_chosen", policy_chosen), ("policy_rejected", policy_rejected),
                        ("ref_chosen", ref_chosen), ("ref_rejected", ref_rejected)):
        if value.dim() != 1:
            raise ValueError(f"{name} 必须是一维逐对向量")
    reward_chosen = beta * (policy_chosen - ref_chosen)
    reward_rejected = beta * (policy_rejected - ref_rejected)
    return bradley_terry_loss(reward_chosen, reward_rejected), reward_chosen, reward_rejected
# <<<


def response_logprob(logits, labels):
    """由因果 LM 的 logits [B,T,V] 与标签 [B,T]（提示与 padding 为 -100）求响应序列的 log-prob 和。

    按 next-token 对齐：第 t 个位置的 logits 预测第 t+1 个标签。这里复用第 24 章的
    sequence_log_prob，因此 mask 语义与 RL 部分一致。
    """
    targets = labels[:, 1:]
    mask = targets != -100
    return sequence_log_prob(logits[:, :-1].float(), targets, mask)


def dpo_step_loss(model, ref_logprob_fn, batch, beta=0.1):
    """用于手写一步 DPO：batch 含 input_ids/attention_mask/labels（chosen 与 rejected 交错成 2B 行）。

    ref_logprob_fn(batch) 返回 reference 的逐行序列 log-prob（形状 [2B]，应已 no_grad）。
    返回 (平均损失, 逐对 chosen 奖励, 逐对 rejected 奖励)。
    """
    logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                   use_cache=False).logits
    policy = response_logprob(logits, batch['labels'])
    ref = ref_logprob_fn(batch)
    pc, pr = policy[0::2], policy[1::2]
    rc, rr = ref[0::2], ref[1::2]
    loss, chosen, rejected = dpo_loss(pc, pr, rc, rr, beta=beta)
    return loss.mean(), chosen.detach(), rejected.detach()
