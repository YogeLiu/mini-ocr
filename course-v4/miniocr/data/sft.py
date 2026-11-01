"""SFT 的单轮答案监督与右 padding；不改变 postcorrect-v1 数据身份。

模板采用第 18 章的非思考生成前缀，答案后直接追加 tokenizer.eos_token_id。
超长样本整条拒收，绝不把被截断的答案加上 EOS 当成完整答案。
"""
from dataclasses import dataclass

import torch

from miniocr.prompting import render_chat, zero_shot

IGNORE_INDEX = -100
VARIANTS = ("correct", "shift", "prompt", "no_eos")


@dataclass(frozen=True)
class SFTExample:
    """未 padding 的完整对话；prompt_length 是第一个答案 token 的位置。"""
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_length: int
    id: str = ""


# >>> 练习 19.1：渲染模板、定位答案边界并追加 EOS
def encode_sft(tokenizer, source, target, *, max_length=512, variant="correct", id=""):
    """返回 SFTExample，超长返回 None；空答案/非法模板边界显式报错。

    联合编码 prompt+target 并验证 prefix，不能假设分别 tokenize 后总能拼接。
    correct 的 labels 不提前 shift；shift 故障故意提前左移一次（训练再移一次）。
    prompt 故障监督提示；no_eos 故障删掉末尾 EOS。仅用于受控教学消融。
    """
    if variant not in VARIANTS or max_length < 2:
        raise ValueError("variant 非法或 max_length < 2")
    if not target.strip() or tokenizer.eos_token_id is None:
        raise ValueError("必须有非空答案及 EOS")
    prompt = render_chat(tokenizer, zero_shot(source))
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    ids = tokenizer.encode(prompt + target, add_special_tokens=False)
    if not prefix or ids[:len(prefix)] != prefix or len(ids) == len(prefix):
        raise ValueError("模板/答案的 token 边界不稳定，不能按字符数构造 mask")
    ids += [tokenizer.eos_token_id]
    if len(ids) > max_length:
        return None
    labels = [IGNORE_INDEX] * len(prefix) + ids[len(prefix):]
    if variant == "prompt":
        labels = ids.copy()
    elif variant == "shift":
        labels = labels[1:] + [IGNORE_INDEX]
    elif variant == "no_eos":
        ids, labels = ids[:-1], labels[:-1]
    return SFTExample(tuple(ids), tuple(labels), len(prefix), id)
# <<<


# >>> 练习 19.2：变长 batch 的右 padding 与三张量对齐
def collate_sft(examples, pad_token_id, *, multiple=1):
    """返回 CPU long 张量；padding 的 attention=0、labels=-100，按位置而非 ID 屏蔽。

    即使 pad_token_id == eos_token_id，真实 EOS 仍受监督。拒绝全 mask 的样本。
    """
    examples = list(examples)
    if not examples or multiple < 1 or pad_token_id is None:
        raise ValueError("需要非空 batch、pad_token_id 和正 multiple")
    width = (max(len(e.input_ids) for e in examples) + multiple - 1) // multiple * multiple
    ids = torch.full((len(examples), width), pad_token_id, dtype=torch.long)
    labels = torch.full_like(ids, IGNORE_INDEX)
    attention = torch.zeros_like(ids)
    for row, ex in enumerate(examples):
        n = len(ex.input_ids)
        if n < 2 or n != len(ex.labels) or not any(x != IGNORE_INDEX for x in ex.labels[1:]):
            raise ValueError("长度不匹配或没有可预测的监督 token")
        ids[row, :n] = torch.tensor(ex.input_ids)
        labels[row, :n] = torch.tensor(ex.labels)
        attention[row, :n] = 1
    return dict(input_ids=ids, attention_mask=attention, labels=labels)
# <<<
