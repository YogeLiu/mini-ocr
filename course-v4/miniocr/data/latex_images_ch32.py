"""为解剖实验抽取已缓存的开源公式图片，严格沿用完整 latex-v1 划分。

原始 parquet 只读；缓存写到调用者指定目录。先读取所有文本确定划分，
再按原始文件/行号取每个公式的首张图片，不下载、不回退到合成数据。
"""
from io import BytesIO
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageOps
from miniocr.data.latex_corpus import _parquet_paths, standard_split, clean_formula, normalization_key


def prepare_images(out, n=200):
    """保存 val/test 各前 n 条的原图与可渲染的配对图片、manifest 与来源元数据。

    test 图片仅用于最终冻结评测；不生成 test 渲染图，不读 test 错例。
    不按目标长度或可识别性筛选主评测；渲染失败明确记录、不替换公式。
    """
    import pyarrow.parquet as pq
    from miniocr.eval.render import render
    paths = _parquet_paths()
    if not paths:
        raise FileNotFoundError("需要已有 cleaned_formulas parquet 缓存")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    splits = standard_split(n_train=0)
    wanted = {s: list(dict.fromkeys(splits[s][:n])) for s in ("val", "test")}
    remaining = set(wanted["val"] + wanted["test"])
    found = {}
    for path in paths:
        row_number = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["latex_formula", "image"]):
            for raw, value in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
                target = clean_formula(raw)
                if target in remaining:
                    content = value["bytes"]
                    image = Image.open(BytesIO(content)).convert("RGB")
                    digest = hashlib.sha256((str(path.name)+":"+str(row_number)).encode()).hexdigest()[:20]
                    name = digest+".png"
                    image.save(out/name)
                    found[target] = dict(id=digest, target=target, image=name, width=image.width, height=image.height,
                                         parquet=path.name, row=row_number, image_sha256=hashlib.sha256(content).hexdigest(),
                                         normalization_key=normalization_key(target))
                    remaining.remove(target)
                row_number += 1
            if not remaining:
                break
        if not remaining:
            break
    if remaining:
        raise RuntimeError(f"未找到 {len(remaining)} 个目标的图片")
    for split, targets in wanted.items():
        rows = []
        for target in targets:
            row = {**found[target], "split": split, "rendered": None}
            if split == "val":
                try:
                    arr = render(target, dpi=160)
                    im = ImageOps.expand(Image.fromarray(255-arr).convert("RGB"), border=12, fill="white")
                    name = row["id"]+"_rendered.png"
                    im.save(out/name)
                    row["rendered"] = name
                except (ValueError, RecursionError) as exc:
                    row["render_error"] = str(exc)
            rows.append(row)
        (out/f"{split}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    meta = dict(protocol="latex-v1", seed=0, split_n_val=5000, split_n_test=5000,
                source="OleehyO/latex-formulas", config="cleaned_formulas", license="openrail",
                snapshots=[str(p) for p in paths], requested_n=n,
                unique_counts={k:len(v) for k,v in wanted.items()},
                selection="unique targets in standard_split prefix; first parquet occurrence")
    (out/"metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta
