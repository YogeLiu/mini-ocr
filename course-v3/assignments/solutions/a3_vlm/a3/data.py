"""A3 参考实现：字符级分词器、ShapesWorld 的样本与批。图像生成、任务问答都来自 miniocr.data（数据生成器）。"""
import numpy as np
import torch

from miniocr.data.instruct import TASKS, draw_text_scene, make_task, parse_box
from miniocr.data.shapes import describe, draw_scene, sample_scene

from .boxes import box_iou
from .tokenizer import Tokenizer, train_bpe
from .vlm import chat_template, pad_batch

SPECIALS = ["<pad>", "<bos>", "<eos>", "<img>"]
_V, _M = train_bpe("", len(SPECIALS) + 96, SPECIALS)          # 不做合并：纯字符级，编号 0–3 是特殊符号
TOK = Tokenizer(_V, _M, SPECIALS)
PAD, BOS, EOS, IMG = range(4)
VOCAB_SIZE = len(_V)
SECOND = ("count", "left", "exists", "count_color")


def to_float(x):
    """uint8 (…, 3, H, W) → float，范围 [−1, 1]。"""
    return x.float() / 127.5 - 1.0


def scene(rng, lo=1, hi=3, text=False):
    objs = sample_scene(rng, 64, 64, lo, hi)
    img = draw_scene(objs, rng, 64, 64, noise=0.03)
    word = None
    if text:
        img, word, _ = draw_text_scene(img, objs, rng)
    return img.transpose(2, 0, 1), objs, word


def example(task, rng, second=None, neg_ground=0.2):
    """(图像 uint8 (k, 3, 64, 64)，问答列表)。compare 两张图，其余一张；second 给出时在同一张图上再问一轮。"""
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


def encode(s):
    return TOK.encode(s)


def sample_batch(rng, tasks, weights, batch_size, n_img, multi_turn=0.3):
    """整批同一个任务（按权重抽），可能是两轮对话。返回 (图像 uint8 (B, k, 3, 64, 64), ids, targets)。"""
    p = np.asarray(weights, dtype=float)
    task = tasks[int(rng.choice(len(tasks), p=p / p.sum()))]
    imgs, exs = [], []
    for _ in range(batch_size):
        two = task not in ("compare", "text", "caption") and rng.random() < multi_turn
        img, pairs = example(task, rng, second=SECOND[int(rng.integers(4))] if two else None)
        imgs.append(img)
        exs.append(chat_template(convo(pairs), n_img, encode, BOS, EOS, IMG))
    ids, tg = pad_batch(exs, PAD)
    return torch.from_numpy(np.stack(imgs)), ids, tg


def prompt_ids(question, n_img):
    ids, _ = chat_template([{"role": "user", "content": question}], n_img, encode, BOS, EOS, IMG)
    return ids + encode("Assistant: ")


def eval_set(seed, n_per_task, tasks=TASKS):
    """每个任务 n 个单轮问答。返回列表 [(task, 图像 uint8 (k, 3, 64, 64), 问题, 答案)]。"""
    out = []
    for t in tasks:
        rng = np.random.default_rng(seed * 100 + list(TASKS).index(t))
        for _ in range(n_per_task):
            img, pairs = example(t, rng)
            out.append((t, img, pairs[0][0], pairs[0][1]))
    return out


def correct(task, pred, ref):
    """定位：两边都是 "none"，或者解析出的框与真值 IoU ≥ 0.5；其余任务：完全相同。"""
    if task != "ground" or ref == "none":
        return pred.strip() == ref
    b, r = parse_box(pred), parse_box(ref)
    if b is None:
        return False
    return box_iou(torch.tensor([b], dtype=torch.float), torch.tensor([r], dtype=torch.float))[0, 0].item() >= 0.5


def caption_batch(rng, batch_size):
    """对比学习用：(图像 uint8 (B, 3, 64, 64)，描述文字的编号 (B, L)，用 PAD 补齐)。"""
    imgs, ids = [], []
    for _ in range(batch_size):
        img, objs, _ = scene(rng)
        imgs.append(img)
        ids.append(encode(describe(objs)))
    L = max(map(len, ids))
    t = torch.full((batch_size, L), PAD, dtype=torch.long)
    for i, s in enumerate(ids):
        t[i, :len(s)] = torch.tensor(s)
    return torch.from_numpy(np.stack(imgs)), t
