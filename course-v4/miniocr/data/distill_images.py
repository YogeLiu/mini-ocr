"""第 35 章只读开源 PNG，完整 latex-v1 分组后取独立训练子集。

不修改已有数据接口；不下载数据，不将渲染图片当作原始数据。
"""
import hashlib
import io
import json
from pathlib import Path
from PIL import Image
from miniocr.data.latex_corpus import _parquet_paths, standard_split, normalization_key, clean_formula


def prepare_distill_images(out, n_train=256, n_teacher=192, n_eval=64, max_train_chars=256, *, val_only=False):
    """提取 base/teacher（均 train，键不重叠）及未按长度筛选的 val/test。

    训练长度上限只控制计算量。按 standard_split 顺序去规范化重复，
    取每个键的首条标准文本及 parquet 中对应的首张 PNG。
    val_only=True 时不物化 test 图片或 manifest；完整划分规则不变。
    """
    import pyarrow.parquet as pq
    paths = _parquet_paths()
    if not paths:
        raise FileNotFoundError("cleaned_formulas parquet missing; no synthetic fallback")
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    partitions = standard_split(n_train=None)
    corpus_rows = sum(map(len, partitions.values()))
    if val_only:
        partitions.pop("test", None)
    selected = {}
    for split, texts in partitions.items():
        seen, chosen = set(), []
        for text in texts:
            key = normalization_key(text)
            if key in seen or (split == "train" and len(text) > max_train_chars):
                continue
            seen.add(key); chosen.append(text)
            if len(chosen) == (n_train+n_teacher if split == "train" else n_eval):
                break
        selected[split] = chosen
    wanted = {t for values in selected.values() for t in values}
    found = {}
    for path in paths:
        row_number = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["latex_formula", "image"]):
            for item in batch.to_pylist():
                target = clean_formula(item["latex_formula"])
                if target in wanted:
                    key = normalization_key(target)
                    identity = hashlib.sha256(key.encode()).hexdigest()
                    payload = item["image"]["bytes"]
                    rgba = Image.open(io.BytesIO(payload)).convert("RGBA")
                    paper = Image.new("RGBA", rgba.size, "white"); paper.alpha_composite(rgba)
                    paper.convert("RGB").save(out / (identity+".png"))
                    found[target] = dict(id=identity, target=target, image=identity+".png",
                        parquet=str(path), row=row_number, image_sha256=hashlib.sha256(payload).hexdigest())
                    wanted.remove(target)
                row_number += 1
            if not wanted: break
        if not wanted: break
    if wanted: raise RuntimeError(f"missing {len(wanted)} images")
    counts = {}
    manifests = dict(base=selected["train"][:n_train], teacher=selected["train"][n_train:], val=selected["val"])
    if not val_only:
        manifests["test"] = selected["test"]
    for name, texts in manifests.items():
        rows = [dict(found[t], split="train" if name in ("base", "teacher") else name) for t in texts]
        (out/(name+".json")).write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        counts[name] = len(rows)
    expected = dict(base=n_train, teacher=n_teacher, val=n_eval, test=n_eval)
    if val_only:
        expected.pop("test")
    if counts != expected: raise ValueError(f"insufficient data: {counts} != {expected}")
    meta = dict(source="OleehyO/latex-formulas", config="cleaned_formulas", license="openrail",
        protocol="latex-v1", seed=0, counts=counts, corpus_rows=corpus_rows,
        max_train_chars=max_train_chars, paths=list(map(str, paths)), selection="unique normalized standard_split prefix")
    (out/"metadata.json").write_text(json.dumps(meta, indent=2))
    return meta
