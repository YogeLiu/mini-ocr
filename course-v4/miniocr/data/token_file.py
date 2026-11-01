"""第 12 章：带分词器指纹的 uint16 token 文件；拒绝静默读取过期缓存。"""
import hashlib
import json
from pathlib import Path

import numpy as np

from miniocr.data.latex_lm import encode_formulas

FORMAT_VERSION = "miniocr-token-v1"


class StaleCacheError(ValueError):
    """缓存的分词器、数据版本或格式版本与当前请求不符。"""


def _digest(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tokenizer_digest(tokenizer):
    return _digest(dict(merges=[[a.hex(), b.hex()] for a, b in tokenizer.merges],
                        specials=list(tokenizer.specials), pattern=tokenizer.pattern))


def cache_key(tokenizer, data_version, extra=None):
    """哈希覆盖合并顺序、特殊符号顺序、预分词规则、数据版本及格式版本。

    extra 须能确定性地序列化为 JSON；数据版本由调用方维护，改变内容须升级。
    """
    return _digest(dict(tokenizer_hash=_tokenizer_digest(tokenizer), data_version=data_version,
                        format_version=FORMAT_VERSION, extra=extra))


def write_token_file(path, tokenizer, texts, data_version):
    """每条加 BOS/EOS，写入 <path>.bin 和 <path>.json；返回元数据。"""
    if len(tokenizer) >= 65536:
        raise ValueError("uint16 文件要求词表大小小于 65536")
    texts = list(texts)
    seqs, n_bytes = encode_formulas(tokenizer, texts)
    ids = np.fromiter((i for seq in seqs for i in seq), dtype="<u2")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(cache_key=cache_key(tokenizer, data_version), data_version=data_version,
                tokenizer_hash=_tokenizer_digest(tokenizer), format_version=FORMAT_VERSION,
                vocab_size=len(tokenizer), token_count=len(ids), formula_count=len(texts),
                dtype="uint16", byte_order="little", text_utf8_bytes=n_bytes)
    ids.tofile(str(path) + ".bin")
    Path(str(path) + ".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def load_token_file(path, tokenizer, data_version):
    """校验元数据后只读映射；空语料返回只读空数组（空文件不能 memmap）。"""
    meta = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
    mismatches = []
    for field, value, name in (
        ("tokenizer_hash", _tokenizer_digest(tokenizer), "分词器内容"),
        ("data_version", data_version, "数据版本"),
        ("format_version", FORMAT_VERSION, "格式版本"),
    ):
        if meta.get(field) != value:
            mismatches.append(name)
    if meta.get("cache_key") != cache_key(tokenizer, data_version) or mismatches:
        raise StaleCacheError("缓存不匹配：" + "、".join(mismatches or ["cache_key 摘要"]))
    if (meta.get("dtype") != "uint16" or meta.get("byte_order") != "little"
            or meta.get("vocab_size") != len(tokenizer)):
        raise ValueError("缓存 dtype、字节序或词表大小不一致")
    count = meta.get("token_count")
    if not isinstance(count, int) or count < 0:
        raise ValueError("元数据 token 数必须为非负整数")
    binary = Path(str(path) + ".bin")
    if binary.stat().st_size != count*2:
        raise ValueError("文件大小与元数据 token 数不一致")
    if count == 0:
        result = np.empty(0, dtype="<u2")
        result.flags.writeable = False
        return result
    return np.memmap(binary, dtype="<u2", mode="r", shape=(count,))
