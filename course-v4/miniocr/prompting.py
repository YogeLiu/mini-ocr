"""固定提示与离线批量贪心推理。提示示例必须来自 train；不训练或更新模型。"""
from dataclasses import dataclass
import time

import torch

ZERO_SHOT = "Correct OCR errors in the following LaTeX formula."
FEW_SHOT = ZERO_SHOT
STRICT = "Return only the corrected LaTeX formula. Do not explain. Do not add $ delimiters or Markdown code fences."
PARAPHRASES = (
    ZERO_SHOT,
    "Repair OCR mistakes in this LaTeX expression.",
    "Recover the correct LaTeX formula from this OCR transcription.",
)


def _validate_examples(examples):
    if any(ex.split != "train" for ex in examples):
        raise ValueError("少样本示例必须来自 train，不能使用 val/test")


def zero_shot(source, instruction=ZERO_SHOT):
    """零样本消息：system 固定任务，user 只含待改公式。"""
    return [{"role": "system", "content": instruction}, {"role": "user", "content": source}]


def few_shot(source, examples, k=3, instruction=FEW_SHOT):
    """前 k 个 train 示例作为 user/assistant 多轮对话；示例不足时报错。"""
    examples = list(examples)
    _validate_examples(examples)
    if k < 0 or len(examples) < k:
        raise ValueError("k 必须非负且示例数至少为 k")
    messages = [{"role": "system", "content": instruction}]
    for ex in examples[:k]:
        messages.extend([{"role": "user", "content": ex.source}, {"role": "assistant", "content": ex.target}])
    return messages + [{"role": "user", "content": source}]


def few_shot_strict(source, examples, k=3):
    """与 few_shot 同样的示例，附加只输出公式/不解释/不加 $ 的约束。"""
    return few_shot(source, examples, k, FEW_SHOT + " " + STRICT)


def render_chat(tokenizer, messages, enable_thinking=False):
    """使用 tokenizer 官方模板；False 在 Qwen3 assistant 开头预填空 think 块。"""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=enable_thinking)


def render_completion(source, examples=(), instruction=ZERO_SHOT):
    """不用 chat 模板，按 Noisy:/Corrected: 直接续写；示例仍只准来自 train。

    调用 generate_greedy(..., stop_at_newline=True) 生成到第一处换行即停止；完整原始生成
    另存在 GenerationResult.raw_texts。直接续写仍使用 chat 权重，不能当 Base 结果。
    """
    examples = list(examples)
    _validate_examples(examples)
    parts = [instruction]
    parts.extend(f"Noisy: {e.source}\nCorrected: {e.target}" for e in examples)
    parts.append(f"Noisy: {source}\nCorrected:")
    return "\n".join(parts)


@dataclass(frozen=True)
class GenerationResult:
    """texts 是可选换行截断后的输出；raw_texts 保留完整解码（都去特殊 token）。

    prompt_tokens 不含 padding；generated_tokens 包含 EOS、不含 EOS 后填充。
    seconds 含编码、生成、解码和设备同步，不含模型加载/评分；length_capped 标记
    未遇 EOS/换行停止且达到预算的输出。批次内共用最大预算，改变 batch_size 可能改变截断。
    从旧 JSONL 复用且未记录总用时时，seconds 为 None，不能由 token 数推算。
    """
    texts: tuple[str, ...]
    raw_texts: tuple[str, ...]
    prompt_tokens: tuple[int, ...]
    generated_tokens: tuple[int, ...]
    seconds: float | None
    length_capped: tuple[bool, ...]


def generate_greedy(model, tokenizer, prompts, max_new_tokens=None, batch_size=4, *, sources=None, stop_at_newline=False):
    """左 padding、do_sample=False 批量生成，返回 GenerationResult，不改 tokenizer 配置。

    sources 给出待改公式时默认每批预算 min(256, 2*该批最长源token数+32)；未给
    sources 时以提示长度作保守上界。显式 max_new_tokens 覆盖该预算。模板已经含
    特殊 token，编码始终 add_special_tokens=False，避免其他模型重复添加 BOS
    （Qwen3 bos_token 本来是 None）。不静默截上下文，超 config 上限时报错。
    stop_at_newline 按样本在首个含换行的生成 token 停止，texts 去掉该换行及其后缀；
    raw_texts 保留停止 token 的完整解码，generated_tokens 包含这个停止 token。
    """
    prompts = list(prompts)
    sources = prompts if sources is None else list(sources)
    if batch_size < 1 or (max_new_tokens is not None and max_new_tokens < 1):
        raise ValueError("batch_size 和 max_new_tokens 必须为正")
    if len(sources) != len(prompts):
        raise ValueError("sources 与 prompts 必须等长")
    device = next(model.parameters()).device

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    sync()
    start = time.perf_counter()
    texts, raw_texts, lengths, generated, capped = [], [], [], [], []
    old_side = tokenizer.padding_side
    was_training = model.training
    try:
        tokenizer.padding_side = "left"
        model.eval()
        with torch.inference_mode():
            for offset in range(0, len(prompts), batch_size):
                batch = prompts[offset:offset+batch_size]
                encoded = tokenizer(batch, padding=True, return_tensors="pt", add_special_tokens=False).to(device)
                budget = max_new_tokens or min(256, 2*max(len(tokenizer.encode(s, add_special_tokens=False))
                                                         for s in sources[offset:offset+batch_size])+32)
                width = encoded["input_ids"].shape[1]
                limit = getattr(model.config, "max_position_embeddings", None)
                if limit is not None and width + budget > limit:
                    raise ValueError(f"提示与生成预算超出上下文上限 {limit}")
                stop = _NewlineStop(tokenizer, width, len(batch)) if stop_at_newline else None
                stop_kwargs = {}
                if stop is not None:
                    from transformers import StoppingCriteriaList
                    stop_kwargs["stopping_criteria"] = StoppingCriteriaList([stop])
                output = model.generate(**encoded, max_new_tokens=budget, do_sample=False,
                                        temperature=None, top_p=None, top_k=None, use_cache=True,
                                        pad_token_id=tokenizer.pad_token_id, **stop_kwargs)
                eos = model.generation_config.eos_token_id
                eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
                for row, (ids, count) in enumerate(zip(output[:, width:].tolist(), encoded["attention_mask"].sum(1).tolist())):
                    end = next((i+1 for i, token in enumerate(ids) if token in eos_ids), len(ids))
                    newline_end = stop.ends[row] if stop is not None else None
                    if newline_end is not None:
                        end = min(end, newline_end)
                    active = ids[:end]
                    raw = tokenizer.decode(active, skip_special_tokens=True)
                    raw_texts.append(raw)
                    texts.append(raw.split("\n", 1)[0].strip() if stop_at_newline else raw)
                    lengths.append(int(count))
                    generated.append(end)
                    capped.append(end == budget and newline_end is None and (not active or active[-1] not in eos_ids))
    finally:
        tokenizer.padding_side = old_side
        model.train(was_training)
    sync()
    return GenerationResult(tuple(texts), tuple(raw_texts), tuple(lengths), tuple(generated),
                            time.perf_counter()-start, tuple(capped))


class _NewlineStop:
    """记录各行首次换行的实际 token 长度，避免把之后的批填充计入生成。"""
    def __init__(self, tokenizer, width, batch_size):
        self.tokenizer, self.width = tokenizer, width
        self.ends = [None] * batch_size

    def __call__(self, input_ids, scores, **kwargs):
        for row, ids in enumerate(input_ids[:, self.width:].tolist()):
            if self.ends[row] is None and "\n" in self.tokenizer.decode(ids, skip_special_tokens=True):
                self.ends[row] = len(ids)
        return torch.tensor([end is not None for end in self.ends], device=input_ids.device, dtype=torch.bool)
