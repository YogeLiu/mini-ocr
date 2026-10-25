"""A4 适配器的参考接法。"""
from __future__ import annotations

from .a4 import logprobs, lora, memory, metrics, sft, stats, teds
from .a4.model import GPT, GPTConfig
from .a4.nn import Linear


def run_edit_distance(a, b):
    return metrics.edit_distance(a, b)


def run_align_ops(pred, ref):
    return metrics.align_ops(pred, ref)


def run_cer(preds, refs):
    return metrics.cer(preds, refs)


def run_wer(preds, refs):
    return metrics.wer(preds, refs)


def run_ned(p, r):
    return metrics.ned(p, r)


def run_bleu(preds, refs, max_n):
    return metrics.bleu(preds, refs, max_n)


def run_anls(preds, answers, tau):
    return metrics.anls(preds, answers, tau)


def _to_tuple(n):
    return (n.tag, n.colspan, n.rowspan, n.text, [_to_tuple(c) for c in n.children])


def _to_node(t):
    n = teds.Node(t[0])
    n.children = [_to_node(c) for c in t[1]]
    return n


def run_parse_table(html):
    return _to_tuple(teds.parse_table(html))


def run_tree_edit_distance(a, b, rename):
    return teds.tree_edit_distance(_to_node(a), _to_node(b), lambda x, y: rename(x.tag, y.tag))


def run_teds(pred_html, ref_html, structure_only):
    return teds.teds(pred_html, ref_html, structure_only)


def run_bootstrap_ci(num, den, n_boot, alpha, seed):
    return stats.bootstrap_ci(num, den, n_boot, alpha, seed)


def run_paired_bootstrap(num_a, num_b, den, n_boot, alpha, seed):
    return stats.paired_bootstrap(num_a, num_b, den, n_boot, alpha, seed)


def run_mcnemar(correct_a, correct_b):
    return stats.mcnemar(correct_a, correct_b)


def run_reliability(conf, correct, n_bins):
    return stats.reliability(conf, correct, n_bins)


def run_ece(conf, correct, n_bins):
    return stats.ece(conf, correct, n_bins)


def run_auroc(scores, labels):
    return stats.auroc(scores, labels)


def run_token_logprobs(logits, ids):
    return logprobs.token_logprobs(logits, ids)


def run_token_entropy(logits):
    return logprobs.token_entropy(logits)


def run_seq_logprob(lp, mask, reduce):
    return logprobs.seq_logprob(lp, mask, reduce)


def run_labels_to_mask(targets):
    return logprobs.labels_to_mask(targets)


def run_mask_to_labels(ids, mask):
    return logprobs.mask_to_labels(ids, mask)


def run_mask_from_lengths(prompt_lens, total_lens, T):
    return logprobs.mask_from_lengths(prompt_lens, total_lens, T)


def run_chat_roles(messages, n_img, encode, bos, eos, img):
    return sft.chat_roles(messages, n_img, encode, bos, eos, img)


def run_sft_targets(ids, roles, prompt_weight, eos_id, train_eos):
    return sft.sft_targets(ids, roles, prompt_weight, eos_id=eos_id, train_eos=train_eos)


def run_sft_loss(logits, targets, weights, mode):
    return sft.sft_loss(logits, targets, weights, mode)


def run_pack(examples, max_len, pad_id):
    return sft.pack(examples, max_len, pad_id)


def run_packed_mask(seg):
    return sft.packed_mask(seg)


def get_lm(vocab_size, block_size, n_layer, n_head, d_model):
    return GPT(GPTConfig(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer, n_head=n_head, d_model=d_model))


def run_lm_logits(lm, ids, pos=None, mask=None):
    return lm(ids, pos=pos, mask=mask)[0]


def get_lora_linear(weight, bias, r, alpha):
    base = Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
    base.weight.data.copy_(weight)
    if bias is not None:
        base.bias.data.copy_(bias)
    return lora.LoRALinear(base, r, alpha)


def run_apply_lora(model, targets, r, alpha):
    return lora.apply_lora(model, targets, r, alpha)


def run_merge_lora(model):
    return lora.merge_lora(model)


def run_count_params(model):
    return lora.count_params(model)


def run_static_memory(n_params, n_trainable, param_dtype, optimizer, master_fp32):
    return memory.static_memory(n_params, n_trainable, param_dtype, optimizer, master_fp32)


def get_saved_tensors(model=None):
    return memory.SavedTensors(model)


def run_checkpoint_blocks(blocks):
    return memory.checkpoint_blocks(blocks)
