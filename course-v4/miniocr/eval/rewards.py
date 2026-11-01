"""第 25 章：分层奖励、有缺陷的对照奖励，以及奖励投机探针。

分层奖励（正确版）：
- 无法解析或为空：0
- 可解析但有格式问题（解释、代码围栏、美元符、think 块）：0.1
- 可解析、格式干净且规范化完全一致：1.0
- 其余：0.1 + 0.4*gain，gain 为相对原串的编辑率改善比例（不改善或变差为 0）

有缺陷对照奖励（只看能否解析）：可解析即 1.0。它故意留下可被投机的口子，用来展示
策略会去利用奖励而不是任务；这不是可用的训练奖励。
"""
from miniocr.eval.latex_norm import normalize_latex
from miniocr.eval.metrics import sample_error_rate
from miniocr.eval.postcorrect_eval import format_issues
from miniocr.eval.render import can_parse


def layered_reward(completion, source, target):
    """返回 [0,1] 的分层奖励；source 为带噪输入，target 为标准答案。"""
    if not completion.strip() or not can_parse(completion):
        return 0.0
    if format_issues(completion):
        return 0.1
    if normalize_latex(target) == normalize_latex(completion):
        return 1.0
    base = sample_error_rate(target, source)
    if base <= 0:
        return 0.1
    pred = sample_error_rate(target, completion)
    gain = min(1.0, max(0.0, (base - pred) / base))
    return 0.1 + 0.4 * gain


def parse_only_reward(completion, source, target):
    """有缺陷对照：只看是否可解析，不看内容、格式或与输入的关系。"""
    return 1.0 if completion.strip() and can_parse(completion) else 0.0


def hack_probes(source, target):
    """固定的投机输出：空串、单字符、复制输入、加解释后复制、重复输入、标准答案。

    返回 [(名称, 文本)]；用于比较两种奖励对明显无用输出的打分。
    """
    return [
        ('empty', ''),
        ('single_symbol', 'x'),
        ('copy_source', source),
        ('copy_with_prose', f'Here is the corrected formula: {source}'),
        ('repeat_source', ' '.join([source] * 3)),
        ('gold_answer', target),
    ]
