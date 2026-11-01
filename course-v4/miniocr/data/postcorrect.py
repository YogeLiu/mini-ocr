"""postcorrect-v1：latex-v1 上的合成 OCR 噪声，供第 18–25 章共用。

划分固定沿用 latex-v1 seed=0；seed 仅控制噪声/示例选择。样本身份由
(version, seed, split, 原公式, 原划分序号) 的 SHA-256 决定；重新排序或取子集
时必须保留原序号，不能重新 enumerate。build_split 自动保持这一约定。
clean_fraction 是预留无错概率；抽中的类型不适用时跳过，不强行补噪，故实际
无错比例可能更高。错误抵消回原文时清空记录。extra_space 常被规范化消除，
因此 EM_raw 与 EM_norm 可以不同；这里的“有错”按原串定义，不代表语义有错。
不模拟图像、真实 OCR 错误分布或可辨识性；混淆后的公式可能仍然合法。
"""
from collections import Counter
from dataclasses import dataclass, field, asdict
from functools import lru_cache
import hashlib
from itertools import combinations
import json
from pathlib import Path
import random
import re
from types import MappingProxyType

from miniocr.data.latex_corpus import standard_split, normalization_key

VERSION = "postcorrect-v1"
ERROR_TYPES = ("confusable", "drop_brace", "command_typo", "missing_arg", "script_shift", "extra_space")
CONFUSABLE = {"l": "1", "1": "l", "O": "0", "0": "O", "o": "0", "I": "l",
              "S": "5", "5": "S", "B": "8", "8": "B", "\\times": "x", "\\cdot": ".",
              "-": "—", "—": "-"}
COMMAND_TYPOS = {"\\alpha": "\\aIpha", "\\beta": "\\bata", "\\left": "\\1eft", "\\sum": "\\surn"}
# 命令内部的 OCR 字母混淆；rn 与 m 是多字符混淆。
COMMAND_LETTERS = {"l": "I", "e": "c", "m": "rn", "o": "0", "i": "l", "a": "o", "rn": "m"}
_TOKEN = re.compile(r"\\[a-zA-Z]+|\\.|[^\\]", re.DOTALL)


@dataclass(frozen=True)
class NoiseConfig:
    """每类独立 Bernoulli 抽样，随机打乱施加顺序，最多 max_errors 处。

    rates 缺失的类型概率为零；复制并冻结传入映射，避免 DEFAULT_NOISE 被修改。
    默认候选错误期望 1.95 个，最多两处；没有可行候选时保留无错。
    """
    clean_fraction: float = .2
    rates: dict = field(default_factory=lambda: dict(zip(ERROR_TYPES, (.4, .3, .35, .3, .3, .3))))
    max_errors: int = 2

    def __post_init__(self):
        if not 0 <= self.clean_fraction <= 1 or not isinstance(self.max_errors, int) or self.max_errors < 0:
            raise ValueError("clean_fraction 须在 [0,1]，max_errors 须为非负整数")
        if set(self.rates) - set(ERROR_TYPES) or any(not 0 <= p <= 1 for p in self.rates.values()):
            raise ValueError("rates 必须使用 ERROR_TYPES 中的类型且概率在 [0,1]")
        object.__setattr__(self, "rates", MappingProxyType(dict(self.rates)))


DEFAULT_NOISE = NoiseConfig()


@dataclass(frozen=True)
class NoiseEdit:
    """一次局部编辑；position 是该次编辑前字符串的零起始字符位置，可依次重放。"""
    type: str
    position: int
    before: str
    after: str


@dataclass(frozen=True)
class PostCorrectExample:
    """稳定 ID、原划分、带噪 source、干净 target 与按施加顺序排列的 errors。

    is_clean 严格等价于 source==target 及 errors==()。构造时校验记录可重放，
    防止 JSONL 中的文本和标注失配；id 含 seed，不包含噪声配置，配置须另行记录。
    """
    id: str
    split: str
    source: str
    target: str
    errors: tuple[NoiseEdit, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "errors", tuple(self.errors))
        if self.split not in ("train", "val", "test"):
            raise ValueError("split 必须为 train/val/test")
        if (self.source == self.target) != (not self.errors):
            raise ValueError("无错样本必须 source==target 且 errors==()")
        replay = self.target
        for e in self.errors:
            if e.type not in ERROR_TYPES or not 0 <= e.position <= len(replay) or replay[e.position:e.position+len(e.before)] != e.before:
                raise ValueError("错误记录无法重放")
            replay = replay[:e.position] + e.after + replay[e.position+len(e.before):]
        if replay != self.source:
            raise ValueError("错误记录与 source 不一致")

    @property
    def error_types(self):
        """按施加顺序返回类型元组。"""
        return tuple(e.type for e in self.errors)

    @property
    def is_clean(self):
        """是否原串无错（不使用规范化判断）。"""
        return self.source == self.target

    def to_dict(self):
        """返回 JSON 可序列化字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        """从 to_dict/JSONL 字典恢复并校验。"""
        return cls(**{**value, "errors": tuple(NoiseEdit(**e) for e in value["errors"])})


def _group_end(text, pos):
    if pos >= len(text) or text[pos] != "{":
        return None
    depth = 0
    for m in _TOKEN.finditer(text, pos):
        depth += (m[0] == "{") - (m[0] == "}")
        if depth == 0:
            return m.end()
    return None


def _candidates(text, kind):
    result = []
    for m in _TOKEN.finditer(text):
        token, pos = m[0], m.start()
        if kind == "confusable" and token in CONFUSABLE:
            result.append((pos, token, CONFUSABLE[token]))
        elif kind == "drop_brace" and token in ("{", "}"):
            result.append((pos, token, ""))
        elif kind == "script_shift" and token in ("^", "_"):
            result.append((pos, token, "_" if token == "^" else "^"))
        elif kind == "command_typo" and re.fullmatch(r"\\[a-zA-Z]+", token):
            if token in COMMAND_TYPOS:
                result.append((pos, token, COMMAND_TYPOS[token]))
            else:
                for before, after in COMMAND_LETTERS.items():
                    for hit in re.finditer(before, token[1:]):
                        result.append((pos+1+hit.start(), before, after))
        elif kind == "missing_arg" and token in (r"\frac", r"\dfrac", r"\tfrac", r"\binom", r"\sqrt"):
            start = m.end()
            while start < len(text) and text[start].isspace():
                start += 1
            end = _group_end(text, start)
            if end is not None:
                if token != r"\sqrt":
                    start = end
                    while start < len(text) and text[start].isspace():
                        start += 1
                    end = _group_end(text, start)
                if end is not None:
                    result.append((start, text[start:end], ""))
        elif kind == "extra_space":
            # 只在 token 边界插空格，不拆命令名；即使原本有空格仍插入。
            result.append((m.end(), "", "  "))
    return result


def corrupt(text, rng, config=DEFAULT_NOISE):
    """用调用方 random.Random 返回 (带噪字符串, NoiseEdit 元组)，不改全局 RNG。

    先抽 clean_fraction，再独立抽各类；不可施加则跳过，最终无变化清空记录。
    missing_arg 只处理紧随命令的花括号参数（含嵌套），不覆盖所有 TeX 语法。
    """
    if rng.random() < config.clean_fraction:
        return text, ()
    source, edits = text, []
    kinds = [k for k in ERROR_TYPES if rng.random() < config.rates.get(k, 0)]
    rng.shuffle(kinds)
    for kind in kinds:
        if len(edits) >= config.max_errors:
            break
        choices = _candidates(source, kind)
        if choices:
            pos, before, after = rng.choice(choices)
            edits.append(NoiseEdit(kind, pos, before, after))
            source = source[:pos] + after + source[pos+len(before):]
    return (source, tuple(edits)) if source != text else (text, ())


def make_example(target, split, index, seed=0, config=DEFAULT_NOISE):
    """按原划分 index 构造独立样本；同身份不受生成顺序/子集大小影响。"""
    if index < 0:
        raise ValueError("index 必须非负")
    payload = json.dumps([VERSION, seed, split, target, index], ensure_ascii=False)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    source, errors = corrupt(target, random.Random(int(digest, 16)), config)
    return PostCorrectExample(f"{split}-{digest[:24]}", split, source, target, errors)


@lru_cache(maxsize=4)
def _splits(formulas, n_val, n_test):
    return standard_split(n_train=None, n_val=n_val, n_test=n_test, seed=0, formulas=formulas)


def target_splits(formulas=None, n_val=5000, n_test=5000):
    """返回 latex-v1 全部干净公式的只读元组映射；进程内缓存，不写磁盘。

    注入小语料时显式设置 n_val/n_test；seed=0 固定，噪声 seed 不改变划分。
    """
    data = _splits(None if formulas is None else tuple(formulas), n_val, n_test)
    return {name: tuple(values) for name, values in data.items()}


def build_split(split="val", n=None, seed=0, config=DEFAULT_NOISE, formulas=None, *, n_val=5000, n_test=5000):
    """先完整划分，再只对指定划分前 n 条加噪；n=None 表示该划分全部。"""
    if split not in ("train", "val", "test") or (n is not None and n < 0):
        raise ValueError("split 必须为 train/val/test，n 必须非负或 None")
    texts = _splits(None if formulas is None else tuple(formulas), n_val, n_test)[split]
    return [make_example(t, split, i, seed, config) for i, t in enumerate(texts[:n])]


def build_dataset(n_train=None, n_val=None, n_test=None, seed=0, config=DEFAULT_NOISE, formulas=None, *, split_n_val=5000, split_n_test=5000):
    """返回三划分样本；n_* 只截取已划分前缀，split_n_* 才控制留出规模。"""
    return {s: build_split(s, n, seed, config, formulas, n_val=split_n_val, n_test=split_n_test)
            for s, n in zip(("train", "val", "test"), (n_train, n_val, n_test))}


def few_shot_pool(k, seed=0, config=DEFAULT_NOISE, formulas=None, *, n_val=5000, n_test=5000):
    """从 train 的确定性随机排列中取前 k 条有错样本，不放回、不使用 val/test。

    只对扫描到的候选加噪；k 的较小值是较大值的前缀。无足够有错样本时报错。
    """
    if k < 0:
        raise ValueError("k 必须非负")
    texts = _splits(None if formulas is None else tuple(formulas), n_val, n_test)["train"]
    if k == 0:
        return []
    indices = list(range(len(texts)))
    random.Random(seed).shuffle(indices)
    result = []
    for i in indices:
        ex = make_example(texts[i], "train", i, seed, config)
        if not ex.is_clean:
            result.append(ex)
            if len(result) == k:
                return result
    raise ValueError(f"train 中有错示例不足 {k} 条")


def error_type_counts(examples):
    """每类出现的样本数；多类样本进入多个桶，包含零计数类型。"""
    counts = Counter({k: 0 for k in ERROR_TYPES})
    for ex in examples:
        counts.update(set(ex.error_types))
    return dict(counts)


def check_no_leak(dataset):
    """按 normalization_key(target) 检查交集，返回 ok 与各对不同键重叠数。

    值可为 PostCorrectExample 序列或干净公式序列，后者避免为泄漏检查构造全量噪声。
    只算一次每个划分的规范键，避免三次成对检查反复规范化大训练集。
    """
    keys = {s: {normalization_key(e if isinstance(e, str) else e.target) for e in values}
            for s, values in dataset.items()}
    overlaps = {f"{a}_{b}": len(keys[a] & keys[b]) for a, b in combinations(sorted(keys), 2)}
    return {"ok": not any(overlaps.values()), "overlaps": overlaps}


def save_jsonl(examples, path):
    """将样本写成 UTF-8 JSONL；调用方负责选择输出位置。"""
    with Path(path).open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")


def load_jsonl(path):
    """读取 UTF-8 JSONL 并恢复 frozen dataclass；跳过空行。"""
    with Path(path).open(encoding="utf-8") as f:
        return [PostCorrectExample.from_dict(json.loads(line)) for line in f if line.strip()]
