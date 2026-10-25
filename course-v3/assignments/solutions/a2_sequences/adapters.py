"""A2 参考实现的适配器（与 assignments/a2_sequences/tests/adapters.py 的接口相同）。"""
import torch

from .a2.ctc import ctc_loss, greedy_decode, prefix_beam_search
from .a2.generate import process_logits
from .a2.lm import accumulate_grads, cer, get_batch, nll_sum
from .a2.model import GPT, GPTConfig
from .a2.nn import RMSNorm, apply_rope, rope_tables, scaled_dot_product_attention
from .a2.rnn import CRNN, LSTMLayer
from .a2.tokenizer import Tokenizer, pretokenize, train_bpe


def run_pretokenize(text):
    return pretokenize(text)


def run_train_bpe(text, vocab_size, special_tokens):
    return train_bpe(text, vocab_size, special_tokens)


def get_tokenizer(vocab, merges, special_tokens):
    return Tokenizer(vocab, merges, special_tokens)


def run_ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction):
    return ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction=reduction)


def run_ctc_greedy(log_probs):
    return greedy_decode(log_probs)


def run_ctc_beam(log_probs, beam):
    return prefix_beam_search(log_probs, beam)


def run_lstm(d_in, hidden, weights, x, h0, c0):
    m = LSTMLayer(d_in, hidden)
    m.load_state_dict(weights)
    return m(x, (h0, c0))


def get_crnn(n_classes):
    return CRNN(n_classes)


def run_cer(preds, refs):
    return cer(preds, refs)


def run_rmsnorm(d, eps, weight, x):
    m = RMSNorm(d, eps)
    m.load_state_dict({"weight": weight})
    return m(x)


def run_rope(x, base, pos0):
    T, dh = x.shape[-2], x.shape[-1]
    cos, sin = rope_tables(pos0 + T, dh, base)
    return apply_rope(x, cos[pos0:], sin[pos0:])


def run_attention(q, k, v, mask):
    return scaled_dot_product_attention(q, k, v, mask)


def get_gpt(cfg):
    return GPT(GPTConfig(**cfg))


def run_gpt(cfg, state_dict, idx):
    m = get_gpt(cfg)
    m.load_state_dict(state_dict)
    return m.eval()(idx)[0]


def run_gpt_incremental(cfg, state_dict, idx, n_prefill):
    m = get_gpt(cfg)
    m.load_state_dict(state_dict)
    m.eval()
    caches = [{} for _ in m.blocks]
    out = [m(idx[:, :n_prefill], caches=caches)[0]]
    for t in range(n_prefill, idx.shape[1]):
        out.append(m(idx[:, t:t + 1], caches=caches, pos0=t)[0])
    return torch.cat(out, 1)


def run_process_logits(logits, temperature, top_k, top_p):
    return process_logits(logits, temperature, top_k, top_p)


def run_get_batch(ids, batch_size, context, generator):
    return get_batch(ids, batch_size, context, generator)


def run_accumulate_grads(model, x, y, micro_batches):
    return accumulate_grads(model, x, y, micro_batches)


def run_nll_sum(model, ids, context):
    return nll_sum(model, ids, context)
