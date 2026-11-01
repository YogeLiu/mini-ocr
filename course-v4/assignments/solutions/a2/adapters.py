"""A2 的参考实现：直接接到课程的参考代码 miniocr（第 6–16 章的练习答案）上。
学习者的版本不能 import miniocr 的模型与训练代码，必须在 a2/mechanism/ 里从零写（见 assignments/a2/handout.md）。"""
from __future__ import annotations

import math

import regex
import torch
import torch.nn.functional as F

from miniocr.generate import process_logits
from miniocr.models.gpt import GPT, GPTConfig, RMSNorm, apply_rope, rope_cache
from miniocr.nn.attention import attention
from miniocr.tokenizer import BPETokenizer, pretokenize, train_bpe
from miniocr.train.lm import get_batch


# ---------- 第一部分 ----------
def run_pretokenize(text):
    return pretokenize(text)


def _strip_specials(texts, specials):
    if not specials:
        return list(texts)
    pat = regex.compile("|".join(regex.escape(s) for s in sorted(specials, key=len, reverse=True)))
    return [piece for t in texts for piece in pat.split(t) if piece]


def run_train_bpe(texts, n_merges, special_tokens):
    if n_merges < 0 or any(not s for s in special_tokens) or len(set(special_tokens)) != len(special_tokens):
        raise ValueError("invalid BPE settings")
    merges = train_bpe(_strip_specials(texts, special_tokens), n_merges)
    tok = BPETokenizer(merges, special_tokens)
    return dict(tok.vocab), merges


class _Tok:
    def __init__(self, vocab, merges, specials):
        self.t = BPETokenizer(merges, specials)
        assert self.t.vocab == vocab, "vocab 与 merges、specials 推出的编号不一致"

    def encode(self, text, allow_special=True):
        return self.t.encode(text, allow_special=allow_special)

    def decode(self, ids):
        return self.t.decode(ids)

    def encode_iterable(self, iterable):
        for piece in iterable:
            yield from self.t.encode(piece)


def get_tokenizer(vocab, merges, special_tokens):
    return _Tok(vocab, merges, special_tokens)


# ---------- 第二部分 ----------
def run_rmsnorm(d, eps, weight, x):
    if d <= 0 or x.shape[-1] != d or weight.shape != (d,) or eps < 0:
        raise ValueError("RMSNorm shape/eps")
    xf = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * weight).to(x.dtype)


def run_rope(x, base, pos0):
    if x.ndim != 4 or pos0 < 0 or base <= 0:
        raise ValueError("RoPE shape/position/base")
    cos, sin = rope_cache(pos0 + x.shape[-2], x.shape[-1], base)
    return apply_rope(x, cos[pos0:], sin[pos0:])


def run_attention(q, k, v, mask):
    if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]:
        raise ValueError("attention dimensions")
    if mask is not None and (mask.dtype != torch.bool or not mask.any(-1).all()):
        raise ValueError("mask must be bool with a visible key per row")
    return attention(q, k, v, mask)[0]


def get_gpt(cfg):
    if min(cfg[k] for k in ("vocab_size", "block_size", "n_layer", "n_head", "d_model")) <= 0:
        raise ValueError("positive GPT dimensions required")
    if cfg["d_model"] % cfg["n_head"] or (cfg["d_model"] // cfg["n_head"]) % 2:
        raise ValueError("heads must divide width with even head dimension")
    return GPT(GPTConfig(**dict(cfg, attn_impl="manual")))


def _loaded(cfg, sd):
    m = get_gpt(cfg)
    m.load_state_dict(sd)
    return m.eval()


@torch.no_grad()
def run_gpt(cfg, state_dict, idx):
    return _loaded(cfg, state_dict)(idx)[0]


@torch.no_grad()
def run_gpt_incremental(cfg, state_dict, idx, n_prefill):
    if not 1 <= n_prefill <= idx.shape[1]:
        raise ValueError("invalid prefill")
    m = _loaded(cfg, state_dict)
    caches = m.new_caches()
    out = [m(idx[:, :n_prefill], caches=caches)[0]]
    for t in range(n_prefill, idx.shape[1]):
        out.append(m(idx[:, t:t + 1], caches=caches, pos0=t)[0])
    return torch.cat(out, 1)


def run_process_logits(logits, temperature, top_k, top_p):
    if not math.isfinite(temperature) or temperature <= 0 or top_k < 0 or not 0 < top_p <= 1:
        raise ValueError("invalid decoding settings")
    return process_logits(logits, temperature, top_k, top_p)


# ---------- 第三部分 ----------
def run_get_batch(ids, batch_size, context, generator):
    if ids.ndim != 1 or ids.dtype != torch.long or batch_size < 1 or not 1 <= context < len(ids):
        raise ValueError("invalid batch")
    return get_batch(ids, batch_size, context, generator)


def run_accumulate_grads(model, micro_batches):
    model.zero_grad(set_to_none=True)
    total = sum(int((y != -100).sum()) for _, y in micro_batches)
    if total == 0:
        raise ValueError("no valid targets")
    out = 0.0
    for x, y in micro_batches:
        logits = model(x)[0]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1),
                               ignore_index=-100, reduction="sum") / total
        loss.backward()
        out += loss.item()
    return out


@torch.no_grad()
def run_nll_sum(model, ids, context):
    if ids.ndim != 1 or len(ids) < 2 or context < 1:
        raise ValueError("invalid nll input")
    total = 0.0
    for s in range(0, len(ids) - 1, context):
        y = ids[s + 1:s + context + 1]
        x = ids[s:s + len(y)]
        logits = model(x[None])[0][0]
        total += F.cross_entropy(logits.float(), y, reduction="sum").item()
    return total


def run_swiglu(x, gate, up, down):
    if gate.shape != up.shape or gate.shape[1] != x.shape[-1] or down.shape != (x.shape[-1], gate.shape[0]):
        raise ValueError("SwiGLU dimensions")
    return (F.silu(x @ gate.T) * (x @ up.T)) @ down.T


def run_multihead(x, kv, n_heads, wq, wk, wv, wo, mask=None):
    d = x.shape[-1]
    if n_heads < 1 or d % n_heads or kv.shape[-1] != d or any(w.shape != (d,d) for w in (wq,wk,wv,wo)):
        raise ValueError("multihead dimensions")
    def split(t):
        return t.reshape(t.shape[0], t.shape[1], n_heads, d//n_heads).transpose(1,2)
    y = run_attention(split(x@wq.T), split(kv@wk.T), split(kv@wv.T), mask)
    return y.transpose(1,2).reshape(x.shape) @ wo.T


def run_generate(model, idx, max_new, eos, banned, greedy=True, seed=0, use_cache=True):
    from miniocr.generate import generate
    if max_new < 0 or idx.ndim != 2 or idx.shape[1] < 1:
        raise ValueError("invalid generation input")
    return generate(model, idx, max_new, eos=eos, banned=banned, greedy=greedy,
                    generator=torch.Generator(device=idx.device).manual_seed(seed), use_cache=use_cache)
