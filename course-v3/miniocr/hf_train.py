"""第 35 章：把我们的训练代码搬到真实模型（transformers / peft / trl）上需要的几个零件。

    answer_mask_by_prefix   用"提示（带生成提示）是完整序列的前缀"来定出回答的位置；不是前缀时报错
    lora_target_names       列出要加 LoRA 的线性层的完整名字（跳过视觉塔），交给 peft 的 target_modules
    hf_answer_logp          HF 模型的 logits → 回答部分的对数概率之和（第 27 章的 token_logprobs + seq_logprob）
    trl_reward              把第 31 章的 task_reward 包装成 TRL 的 reward_funcs 要求的函数签名
"""
import torch
import torch.nn as nn

from .rl.logprobs import seq_logprob, token_logprobs
from .rl.rewards import task_reward


# >>> 练习 35.1：用前缀定出回答的位置
def answer_mask_by_prefix(prompt_ids, full_ids):
    """prompt_ids：只有提示（apply_chat_template(..., add_generation_prompt=True)）的 token 列表；
    full_ids：提示 + 回答 + 结束符的 token 列表。
    要求 prompt_ids 正好是 full_ids 的前缀：否则抛出 ValueError，消息里写出第一个不同的位置。
    返回和 full_ids 等长的 0/1 列表：前缀的位置是 0，其余（回答和结束符）是 1。"""
    n = len(prompt_ids)
    if n > len(full_ids) or list(full_ids[:n]) != list(prompt_ids):
        k = next((i for i, (a, b) in enumerate(zip(prompt_ids, full_ids)) if a != b), min(n, len(full_ids)))
        raise ValueError(f"提示不是完整序列的前缀：第 {k} 个 token 起不同")
    return [0] * n + [1] * (len(full_ids) - n)
# <<<


# >>> 练习 35.2：LoRA 加在哪些层上
def lora_target_names(model, suffixes=("q_proj", "k_proj", "v_proj", "o_proj"), skip=("vision", "visual")):
    """遍历 model.named_modules()，返回所有满足下面三个条件的模块的完整名字（按遍历顺序）：
    是 nn.Linear；名字的最后一段（按 "." 切开）在 suffixes 里；名字的任何一段都不包含 skip 里的任一个子串。
    peft 的 LoraConfig(target_modules=...) 接受完整名字的列表，这样就能只给语言模型加 LoRA。"""
    out = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        parts = name.split(".")
        if parts[-1] in suffixes and not any(s in p for p in parts for s in skip):
            out.append(name)
    return out
# <<<


def hf_answer_logp(logits, ids, mask, reduce="sum"):
    """logits (B, T, V)：HF 模型在 ids (B, T) 上的输出；mask (B, T)：回答的位置为 1。
    位置 t 的 logits 预测的是 ids[t + 1]，所以对齐成 (B, T − 1)。返回 (B,) 的对数概率。"""
    lp = token_logprobs(logits, ids)                               # 已经对齐好：(B, T − 1)
    return seq_logprob(lp, mask[:, 1:].to(lp.dtype), reduce)


# >>> 练习 35.3：TRL 的奖励函数
def trl_reward(task):
    """返回一个函数 f(completions, ref, **kwargs) -> list[float]，供 GRPOTrainer(reward_funcs=[f]) 使用。
    TRL 把数据集里除 prompt 以外的列（这里是 ref）按样本传进来。completions 的每一项可能是字符串，
    也可能是对话格式 [{"role": "assistant", "content": ...}]（content 是字符串，或 [{"type": "text", "text": ...}] 的列表）。
    取出文字、strip() 之后，用 task_reward(task, 文字, ref) 打分。"""
    def text_of(c):
        if isinstance(c, str):
            return c
        content = c[-1]["content"]
        if isinstance(content, str):
            return content
        return "".join(p.get("text", "") for p in content if p.get("type") == "text")

    def f(completions, ref, **kwargs):
        return [task_reward(task, text_of(c).strip(), r) for c, r in zip(completions, ref)]
    f.__name__ = f"reward_{task}"
    return f
# <<<


def trainable_params(model):
    """(可训练的参数量, 总参数量)。"""
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tr, tot
