"""第 2 阶段的语料：中英平行句对（Hugging Face 上的 YogeLiu/zh-en-translation-dataset-600K，60 万对）。

我们不做翻译，只把中文句子和英文句子都当作"一行文本"，用来训练分词器（第 2.1 单元）和小语言模型（第 2.3–2.5 单元）。
数据集页面没有写许可证，这里只在本地学习时读取，不再分发（见 course-v4/data/LICENSES.md）。

    load_pairs     读前 n 对（或随机 n 对），返回 [(english, chinese), ...]
    split_pairs    按"句对"划分 train/dev/test：同一对的中英两句一定在同一个集合里
    to_lines       句对 → 文本行列表（lang="zh" / "en" / "both"）
    standard_split 课程统一的划分：开发/测试集固定取自前 22000 对；训练集可以多读，并去掉与开发/测试集有相同句子的句对
    SAMPLE_PAIRS   内置的 8 对短句：测试和没下载数据时用

划分为什么按句对而不是按行？中文句和它的英文译文说的是同一件事。如果中文进了训练集、英文进了验证集，
验证集上的"新"句子其实已经被模型以另一种语言见过——这是泄漏的一种（第 1.4 单元讲过 train/dev/test 的职责）。
"""
import glob
import os

import numpy as np

REPO = "YogeLiu/zh-en-translation-dataset-600K"

SAMPLE_PAIRS = [
    ("The model reads the image and writes the text.", "模型读入图像，写出文字。"),
    ("Attention lets every token look at every other token.", "注意力让每个词元都能看到其他所有词元。"),
    ("We split the data before we train.", "我们先划分数据，再开始训练。"),
    ("The loss went down, but the test score did not improve.", "损失降了，但测试分数没有提高。"),
    ("Count the dogs in the picture.", "数一数图中有几只狗。"),
    ("A cache stores the keys and values of earlier tokens.", "缓存保存了前面词元的键和值。"),
    ("Small models are cheap to train on a laptop.", "小模型在笔记本电脑上就能训练。"),
    ("Is there a cat on the sofa?", "沙发上有猫吗？"),
]


def _parquet_path():
    """先找本地 Hugging Face 缓存；没有就用 huggingface_hub 下载（约 100 MB）。"""
    root = os.path.expanduser(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub"))
    hits = sorted(glob.glob(os.path.join(root, "datasets--YogeLiu--zh-en-translation-dataset-600K",
                                         "snapshots", "*", "data", "*.parquet")))
    if hits:
        return hits[0]
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, "data/train-00000-of-00001.parquet", repo_type="dataset")


def load_pairs(n=None, seed=None):
    """n=None 读全部。seed 不为 None 时随机抽 n 对（固定种子，结果可复现），否则取前 n 对。
    会去掉首尾空白，丢掉任一边为空的句对。"""
    import pyarrow.parquet as pq
    t = pq.read_table(_parquet_path())
    en, zh = t.column("english").to_pylist(), t.column("chinese").to_pylist()
    idx = np.arange(len(en))
    if seed is not None:
        idx = np.random.default_rng(seed).permutation(len(en))
    if n is not None:
        idx = idx[:n]
    pairs = [((en[i] or "").strip(), (zh[i] or "").strip()) for i in idx]
    return [(e, z) for e, z in pairs if e and z]


def split_pairs(pairs, dev=0.01, test=0.01, seed=0):
    """按句对随机划分。返回 (train, dev, test) 三个句对列表。dev、test 可以是比例（< 1）或条数。"""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    n_dev = int(dev * len(pairs)) if dev < 1 else int(dev)
    n_test = int(test * len(pairs)) if test < 1 else int(test)
    pick = lambda ix: [pairs[i] for i in ix]
    return pick(order[n_dev + n_test:]), pick(order[:n_dev]), pick(order[n_dev:n_dev + n_test])


def to_lines(pairs, lang="both"):
    """lang="both" 时每对产生两行（先英文后中文）。"""
    out = []
    for e, z in pairs:
        if lang in ("en", "both"):
            out.append(e)
        if lang in ("zh", "both"):
            out.append(z)
    return out


BASE = 22000            # 2.1 单元起统一使用：load_pairs(BASE, seed=0) 中划出开发集、测试集各 1000 对


def standard_split(n_pairs=BASE, dev=1000, test=1000, pairs=None):
    """返回 (train, dev, test, n_removed)。开发/测试集永远从 load_pairs(BASE, seed=0) 里划出（与 n_pairs 无关），
    训练集 = 其余的句对 + 多读的句对，并去掉任一句与开发/测试集某句相同的句对（数据集里有重复句子）。
    多读的句对按内容排除前 BASE 对里已有的，所以 load_pairs 丢掉空句对造成的位置偏移不影响划分。
    pairs 给出时直接用它（例如 SAMPLE_PAIRS 的小实验），不读数据集。"""
    if pairs is None:
        base = load_pairs(BASE, seed=0)
        seen = set(base)
        extra = [p for p in load_pairs(n_pairs, seed=0) if p not in seen] if n_pairs is None or n_pairs > BASE else []
    else:
        base, extra = list(pairs), []
    tr, dv, ts = split_pairs(base, dev=dev, test=test, seed=0)
    held = {s for p in dv + ts for s in p}
    cand = tr + extra
    train = [p for p in cand if p[0] not in held and p[1] not in held]
    return train, dv, ts, len(cand) - len(train)
