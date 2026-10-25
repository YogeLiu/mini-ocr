"""A3 适配器的参考接法。"""
from __future__ import annotations

import torch

from .a3 import boxes, contrastive, qwen, vision, vlm
from .a3.model import GPT, GPTConfig
from .a3.nn import LayerNorm, gelu


def _load(module, weights):
    module.load_state_dict(weights)
    return module


def run_patchify(x, p):
    return vision.patchify(x, p)


def run_sincos_2d(h, w, d):
    return vision.sincos_2d(h, w, d)


def run_interpolate_pos(pos, old_hw, new_hw):
    return vision.interpolate_pos(pos, old_hw, new_hw)


def run_rope_2d(h, w, dh, base):
    return vision.rope_2d(h, w, dh, base)


def run_gelu(x):
    return gelu(x)


def run_layer_norm(d, weights, x):
    return _load(LayerNorm(d), weights)(x)


def run_vit_block(d, n_head, weights, x, key_mask=None, cos=None, sin=None):
    return _load(vision.ViTBlock(d, n_head), weights)(x, key_mask=key_mask, cos=cos, sin=sin)


def get_vit(image_size, patch, d, n_layer, n_head, pos):
    return vision.ViT(vision.ViTConfig(image_size=image_size, patch=patch, d=d, n_layer=n_layer, n_head=n_head, pos=pos))


def run_clip_loss(zi, zt, logit_scale):
    return contrastive.clip_loss(zi, zt, logit_scale)


def run_siglip_loss(zi, zt, logit_scale, logit_bias):
    return contrastive.siglip_loss(zi, zt, logit_scale, logit_bias)


def run_recall_at_k(sim, k, same=None):
    return contrastive.recall_at_k(sim, k, same)


def run_zero_shot(zi, class_zt):
    return contrastive.zero_shot(zi, class_zt)


def run_box_iou(a, b):
    return boxes.box_iou(a, b)


def run_nms(b, scores, iou_thr):
    return boxes.nms(b, scores, iou_thr)


def run_average_precision(preds, gts, iou_thr):
    return boxes.average_precision(preds, gts, iou_thr)


def run_merge_2x2(x, hw):
    return vlm.merge_2x2(x, hw)


def get_connector(kind, d_v, d):
    return {"mlp": vlm.MLPProjector, "merge": vlm.MergeProjector, "qformer": vlm.QFormer}[kind](d_v, d)


def run_chat_template(messages, n_img, encode, bos, eos, img):
    return vlm.chat_template(messages, n_img, encode, bos, eos, img)


def run_scatter_visual(emb, visual, mask):
    return vlm.scatter_visual(emb, visual, mask)


def get_vlm(vocab_size, img_id, connector):
    vit = get_vit(32, 8, 32, 1, 2, "rope2d")
    lm = GPT(GPTConfig(vocab_size=vocab_size, block_size=128, n_layer=2, n_head=2, d_model=32))
    return vlm.VLM(vit, get_connector(connector, 32, 32), lm, img_id)


def run_vlm_logits(model, images, ids):
    return model(images, ids)[0]


def run_vlm_generate(model, images, prompt, max_new):
    return model.generate(images, prompt, max_new)


def run_smart_resize(h, w, factor, min_pixels, max_pixels):
    return qwen.smart_resize(h, w, factor, min_pixels, max_pixels)


def run_qwen_patchify(x, patch, merge, temporal):
    return qwen.qwen_patchify(x, patch, merge, temporal)


def run_qwen_preprocess(img):
    return qwen.qwen_preprocess(img)


def run_expand_image_tokens(ids, grids, image_id, merge):
    return qwen.expand_image_tokens(ids, grids, image_id, merge)


def run_mrope_positions(ids, grids, image_id, merge):
    return qwen.mrope_positions(ids, grids, image_id, merge)
