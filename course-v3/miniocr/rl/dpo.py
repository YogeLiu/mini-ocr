"""第 30 章：直接偏好优化（DPO）。

    optimal_policy     带 KL 约束的奖励最大化的解析解：π*(y) ∝ π_ref(y) · exp(R(y) / β)
    dpo_loss           −log σ(β · [(log π(y_w) − log π_ref(y_w)) − (log π(y_l) − log π_ref(y_l))])，可选加上 y_w 的 NLL
    pairs_from_groups  从"每题 G 个回答 + 奖励"里挑成对数据：每组奖励最高的当 chosen、最低的当 rejected

记号：y_w 是被偏好的回答（chosen，w = win），y_l 是不被偏好的回答（rejected，l = lose）。
pol_w、pol_l、ref_w、ref_l 都是 (B,) 的序列对数概率（第 27 章 seq_logprob 的 "sum"）；ref_* 不带梯度。
"""
import torch
import torch.nn.functional as F


# >>> 练习 30.1：最优策略
def optimal_policy(ref_logp, rewards, beta):
    """ref_logp、rewards：(K,)，K 个候选回答在参考模型下的对数概率和奖励。返回 (K,) 的 π*（概率，和为 1）：
    π* = softmax(ref_logp + rewards / beta)。"""
    return torch.softmax(ref_logp + rewards / beta, -1)
# <<<


# >>> 练习 30.2：DPO 的损失
def dpo_loss(pol_w, pol_l, ref_w, ref_l, beta=0.1, sft_weight=0.0, n_w=None):
    """返回 (平均损失, 统计)。
        隐含奖励 r_w = β (pol_w − ref_w)，r_l = β (pol_l − ref_l)
        损失 = −log σ(r_w − r_l)                   用 F.logsigmoid，不要写 log(sigmoid(…))（h 很负时会得到 log 0）
    sft_weight > 0 时再加上 sft_weight × (−pol_w / n_w)：chosen 回答每个 token 的平均负对数概率（n_w 是它的 token 数），
    用来防止 chosen 的概率一起下降（30.5 节）。
    统计（都不带梯度）：reward_w、reward_l（各自的平均）、margin（r_w − r_l 的平均）、acc（r_w > r_l 的比例）。"""
    rw, rl = beta * (pol_w - ref_w), beta * (pol_l - ref_l)
    loss = -F.logsigmoid(rw - rl)
    if sft_weight:
        loss = loss + sft_weight * (-pol_w / n_w)
    stats = dict(reward_w=rw.mean().item(), reward_l=rl.mean().item(), margin=(rw - rl).mean().item(),
                 acc=(rw > rl).float().mean().item())
    return loss.mean(), stats
# <<<


# >>> 练习 30.3：从组内采样构造成对数据
def pairs_from_groups(rewards, G, min_gap=0.0):
    """rewards (N·G,)，每连续 G 个是同一道题的回答。对每一组：奖励最高的下标当 chosen、最低的当 rejected
    （有并列时取第一个），只保留 最高 − 最低 > min_gap 的组。返回 [(chosen 的下标, rejected 的下标), …]，下标是全局的。"""
    r = rewards.float().view(-1, G)
    out = []
    for g in range(r.shape[0]):
        hi, lo = int(r[g].argmax()), int(r[g].argmin())
        if r[g, hi] - r[g, lo] > min_gap:
            out.append((g * G + hi, g * G + lo))
    return out
# <<<
