"""第 24 章的机制层：REINFORCE、KL 估计和 PPO surrogate。

仅依赖 torch，不包含数据读取、训练器或下载；调用方负责 on-policy 采样。
KL 的方向固定为 KL(p || q)，样本必须来自 p，log_ratio=log(q)-log(p)。
"""
import torch
import torch.nn.functional as F


# >>> 练习 24.1：策略梯度与不参与反传的基线
def reinforce_loss(log_prob, reward, baseline=0.0):
    """每回合的 log_prob（序列时先求和）乘 detached advantage 后取负均值。

基线必须与当前动作独立，例如历史 EMA 或给定状态的价值；detach 本身
不能消除“当前 batch 均值包含自身奖励”产生的统计偏差。
"""
    advantage = (reward - baseline).detach()
    return -(log_prob * advantage).mean()


def categorical_score_gradient(probs, actions, rewards, baseline=0.0):
    """返回每个样本对 softmax logits 的上升梯度 (R-b)(one_hot(a)-p)。"""
    score = F.one_hot(actions, probs.shape[-1]).to(probs.dtype) - probs
    return (rewards - baseline).unsqueeze(-1) * score
# <<<


# >>> 练习 24.2：三种 KL 估计器；注意采样分布和方向
def kl_estimators(log_ratio):
    """返回 k1/k2/k3 的逐样本值。k1、k3 无偏；k2 只是局部近似。

要求 p 和 q 有相同正支持。k3 的无偏性使用 E_p[q/p]=1；这里的
估计值用于数值演示，不保证直接反传就是某个 RL KL 正则的正确梯度。
"""
    return {"k1": -log_ratio, "k2": .5 * log_ratio.square(),
            "k3": torch.expm1(log_ratio) - log_ratio}
# <<<


# >>> 练习 24.3：PPO 的 min 与裁剪方向
def ppo_surrogate(ratio, advantage, epsilon=.2):
    """逐样本最大化目标；仅有利方向越界时饱和，不是两边一律零梯度。

ratio=pi_new(a|s)/pi_old(a|s)，advantage 来自旧策略并停止梯度。
调用方对返回值取负均值作为损失；ratio 须为非负概率比。
"""
    if not 0 < epsilon < 1:
        raise ValueError("epsilon 必须在 (0,1)")
    advantage = advantage.detach()
    return torch.minimum(ratio * advantage,
                         ratio.clamp(1-epsilon, 1+epsilon) * advantage)
# <<<


# >>> 练习 24.4：序列的 score function 与 response mask
def sequence_log_prob(logits, targets, response_mask):
    """对已对齐的 next-token logits/targets 求响应 token 的 log-prob 和。

logits 为 [B,T,V]，targets/mask 为 [B,T]。提示、padding 设 mask=False；
EOS 若是实际采样动作应为 True。先替换被屏蔽的 target，允许 padding=-100。
不按长度平均：终局奖励对应整条轨迹的 log 概率。
"""
    if logits.shape[:-1] != targets.shape or targets.shape != response_mask.shape:
        raise ValueError("logits/targets/mask 形状不匹配")
    safe = targets.masked_fill(~response_mask.bool(), 0)
    lp = logits.float().log_softmax(-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return lp.masked_fill(~response_mask.bool(), 0).sum(-1)
# <<<
