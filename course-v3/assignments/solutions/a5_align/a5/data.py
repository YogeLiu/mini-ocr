"""A5 参考实现：字符级分词器、ShapesWorld 九个任务（同 A3）+ 读文字行（read），SFT 样本（ids、标签、权重）。"""
import numpy as np
import torch

from miniocr.data.instruct import TASKS, draw_text_scene, make_task, parse_box
from miniocr.data.render import render_line
from miniocr.data.shapes import draw_scene, sample_scene
from miniocr.data.text import sample_line_text

from .boxes import box_iou
from .sft import chat_roles, sft_targets
from .tokenizer import Tokenizer, train_bpe

SPECIALS = ["<pad>", "<bos>", "<eos>", "<img>"]
_V, _M = train_bpe("", len(SPECIALS) + 96, SPECIALS)
TOK = Tokenizer(_V, _M, SPECIALS)
PAD, BOS, EOS, IMG = range(4)
VOCAB_SIZE = len(_V)
ALL_TASKS = tuple(TASKS) + ("read",)
SECOND = ("count", "left", "exists", "count_color")
READ_Q = "<image>Read the text."
LINE_NOISE, LINE_DISTORT = 0.1, 0.5


def to_float(x):
    return x.float() / 127.5 - 1.0


def encode(s):
    return TOK.encode(s)


def scene(rng, text=False):
    objs = sample_scene(rng, 64, 64, 1, 3)
    img = draw_scene(objs, rng, 64, 64, noise=0.03)
    word = None
    if text:
        img, word, _ = draw_text_scene(img, objs, rng)
    return img.transpose(2, 0, 1), objs, word


def line(rng, noise=LINE_NOISE, distort=LINE_DISTORT):
    """(图像 uint8 (3, 32, 256)，文字)。灰度复制成 3 个通道。"""
    while True:
        t = sample_line_text(rng)
        try:
            g, _ = render_line(t, rng, noise=noise, distort=distort)
        except ValueError:
            continue
        return np.repeat(g[None], 3, 0), t


def example(task, rng, second=None, neg_ground=0.2):
    """(图像 uint8 (k, 3, H, W)，问答列表)。"""
    if task == "read":
        img, t = line(rng)
        return img[None], [(READ_Q, t)]
    if task == "compare":
        a, oa, _ = scene(rng)
        b, ob, _ = scene(rng)
        return np.stack([a, b]), [make_task("compare", oa, rng, objs2=ob)]
    while True:
        img, objs, word = scene(rng, text=(task == "text"))
        if task != "text" or word is not None:
            break
    pairs = [make_task(task, objs, rng, word=word, neg_ground=neg_ground)]
    if second is not None:
        q, a = make_task(second, objs, rng, neg_ground=neg_ground)
        pairs.append((q.replace("<image>", ""), a))
    return img[None], pairs


def convo(pairs):
    msgs = []
    for q, a in pairs:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    return msgs


def n_vis(connector, H, W, patch=8):
    g = (H // patch) * (W // patch)
    return {"mlp": g, "merge": g // 4, "qformer": 16}[connector]


def sft_example(rng, task, connector, prompt_weight=0.0, train_eos=True, multi_turn=0.3):
    """一条 SFT 样本：(图像 uint8 (k, 3, H, W), ids, targets, weights)。"""
    two = task not in ("compare", "text", "caption", "read") and rng.random() < multi_turn
    img, pairs = example(task, rng, second=SECOND[int(rng.integers(4))] if two else None)
    ids, roles = chat_roles(convo(pairs), n_vis(connector, *img.shape[2:]), encode, BOS, EOS, IMG)
    tg, w = sft_targets(ids, roles, prompt_weight, eos_id=EOS, train_eos=train_eos)
    return img, ids, tg, w


def prompt_ids(question, n_img):
    ids, _ = chat_roles([{"role": "user", "content": question}], n_img, encode, BOS, EOS, IMG)
    return ids + encode("Assistant: ")


def eval_set(seed, n_per_task, tasks=ALL_TASKS):
    """[(task, 图像 uint8 (k, 3, H, W), 问题, 答案)]，每个任务一个独立的随机数流。"""
    out = []
    for t in tasks:
        rng = np.random.default_rng(seed * 100 + list(ALL_TASKS).index(t))
        for _ in range(n_per_task):
            img, pairs = example(t, rng)
            out.append((t, img, pairs[0][0], pairs[0][1]))
    return out


def correct(task, pred, ref):
    """ground：都是 "none"，或 IoU ≥ 0.5；其余（包括 read）：去掉首尾空白后完全相同。"""
    if task != "ground" or ref == "none":
        return pred.strip() == ref
    b, r = parse_box(pred), parse_box(ref)
    if b is None:
        return False
    return box_iou(torch.tensor([b], dtype=torch.float), torch.tensor([r], dtype=torch.float))[0, 0].item() >= 0.5
