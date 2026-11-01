"""蒸馏机制：同词表 KL、仅答案监督的 mask、可审计的伪标签质检。

异构 VLM 的 tokenizer 不同，不能直接对齐其 logits。本模块的 KL 演示
限定同一词表；真实异构教师/学生实验使用序列蒸馏。
"""
import torch
from miniocr.eval.render import can_parse, compare_render
from miniocr.eval.postcorrect_eval import extract_formula


# >>> 练习 35.1：温度、方向、detach 与有效 token 分母。
def masked_kl(student, teacher, mask, temperature=2.0):
    """T² mean_token KL(teacher || student)，mask 不含 prompt/padding。

    logits 为 [B,L,V]，mask 为 [B,L]；空 mask 返回可求导的零。
    teacher 在这里 detach，防止误把教师一起训练。
    """
    if temperature <= 0 or student.shape != teacher.shape or mask.shape != student.shape[:-1]:
        raise ValueError("temperature/shape invalid; vocabulary must be identical")
    s = torch.log_softmax(student.float() / temperature, -1)
    t = torch.log_softmax(teacher.detach().float() / temperature, -1)
    terms = (t.exp() * (t-s)).sum(-1)
    weights = mask.to(terms.dtype)
    return (terms * weights).sum() / weights.sum().clamp_min(1) * temperature**2
# <<<


# >>> 练习 35.2：模型内部负责 causal shift；这里只 mask prompt。
def answer_labels(input_ids, prompt_ids, attention_mask=None):
    """单样本前缀校验后屏蔽 prompt，保留答案及 EOS；拒绝模板边界错位。"""
    p = prompt_ids.shape[-1]
    if input_ids.shape[0] != 1 or p >= input_ids.shape[-1]:
        raise ValueError("requires one example with a nonempty answer")
    if not torch.equal(input_ids[:, :p], prompt_ids):
        raise ValueError("chat template prompt is not a token prefix")
    labels = input_ids.clone()
    labels[:, :p] = -100
    if attention_mask is not None:
        labels[attention_mask == 0] = -100
    return labels
# <<<


# >>> 练习 35.3：质检不是正确性证明；未知渲染不能当失败或成功。
def quality_check(raw, reference=None, *, max_chars=384, render_tolerance=.65, truncated=False):
    """长度+mathtext 解析+可用时的训练参考渲染一致性。

    reference 只可为 train 标注。这里比较参考/预测的同渲染器图像，
    不是原始 PNG 的像素真值；mathtext 不支持的参考记 unknown 并保留。
    宽松容差有意保留部分教师错误，用于观察错误传递，不能称语义校验。
    """
    pred = extract_formula(raw)
    length_ok = 0 < len(pred) <= max_chars
    parses = can_parse(pred) if length_ok else False
    comparison = compare_render(reference, pred, tol=render_tolerance) if reference is not None and parses else {}
    match = comparison.get("image_match")
    accepted = length_ok and parses and not truncated and match is not False
    return dict(pred=pred, accepted=accepted, length_ok=length_ok, parses=parses,
                truncated=truncated, render_match=match, pixel_diff=comparison.get("pixel_diff"),
                render_reference="train label rendered with mathtext" if reference is not None else None)
# <<<
