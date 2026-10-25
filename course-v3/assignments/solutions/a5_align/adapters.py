"""A5 适配器的参考接法。"""
from __future__ import annotations

from .a5 import dpo, grpo, mix, pg, prefs, rewards, rollout
from .a5.model import GPT, GPTConfig
from .a5.vision import ViT, ViTConfig
from .a5.vlm import VLM, MergeProjector


def run_group_advantages(r, G, std):
    return pg.group_advantages(r, G, std)


def run_loo_advantages(r, G):
    return pg.loo_advantages(r, G)


def run_zero_variance_fraction(r, G):
    return pg.zero_variance_fraction(r, G)


def run_kl_estimate(logp, ref_logp, kind):
    return pg.kl_estimate(logp, ref_logp, kind)


def run_clipped_objective(logp, old_logp, adv, eps_low, eps_high):
    return pg.clipped_objective(logp, old_logp, adv, eps_low, eps_high)


def run_optimal_policy(ref_logp, r, beta):
    return dpo.optimal_policy(ref_logp, r, beta)


def run_dpo_loss(pol_w, pol_l, ref_w, ref_l, beta, sft_weight, n_w):
    return dpo.dpo_loss(pol_w, pol_l, ref_w, ref_l, beta, sft_weight, n_w)


def run_pairs_from_groups(r, G, min_gap):
    return dpo.pairs_from_groups(r, G, min_gap)


def run_hallucinated_caption(objs, rng, kind):
    return prefs.hallucinated_caption(objs, rng, kind)


def run_aggregate(per_token, mask, mode, max_len):
    return grpo.aggregate(per_token, mask, mode, max_len)


def run_grpo_loss(logp, old_logp, adv, mask, ref_logp, beta, eps_low, eps_high, mode):
    return grpo.grpo_loss(logp, old_logp, adv, mask, ref_logp, beta, eps_low, eps_high, mode)


def run_overlong_penalty(length, max_len, cache):
    return grpo.overlong_penalty(length, max_len, cache)


def run_keep_groups(r, G):
    return grpo.keep_groups(r, G)


def run_well_formed(task, pred):
    return rewards.well_formed(task, pred)


def run_task_reward(task, pred, ref):
    return rewards.task_reward(task, pred, ref)


def get_vlm(vocab_size, img_id):
    vit = ViT(ViTConfig(image_size=32, patch=8, d=32, n_layer=1, n_head=2, pos="rope2d"))
    lm = GPT(GPTConfig(vocab_size=vocab_size, block_size=128, n_layer=2, n_head=2, d_model=32))
    return VLM(vit, MergeProjector(32, 32), lm, img_id)


def run_vlm_logits(model, images, ids):
    return model(images, ids)[0]


def run_sample(model, images, prompt, max_new, eos, temperature, generator, banned=()):
    return rollout.sample(model, images, prompt, max_new, eos, temperature, generator, banned)


def run_rollout(model, items, G, max_new, eos, temperature, generator):
    return rollout.rollout(model, items, G, max_new, eos, temperature, generator)


def run_make_batch(prompts, responses, finished, eos, pad):
    return rollout.make_batch(prompts, responses, finished, eos, pad)


def run_policy_logprobs(model, images, ids):
    return rollout.policy_logprobs(model, images, ids)


def run_pareto_front(points):
    return mix.pareto_front(points)


def run_mixed_batch(rng, B, ratio, make_new, make_old):
    return mix.mixed_batch(rng, B, ratio, make_new, make_old)


def run_rft_select(r, G, threshold, k):
    return mix.rft_select(r, G, threshold, k)


def run_probe_kl(logits_new, logits_base, mask):
    return mix.probe_kl(logits_new, logits_base, mask)
