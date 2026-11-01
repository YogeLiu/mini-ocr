"""只读已缓存的开源公式 PNG，先按完整语料的 latex-v1 划分，再选图。

不下载、不重渲染、不修改 postcorrect/评测接口；同规范键仅保留首张。
图片没有符号框标注，不能把整幅公式冒充单符号分类训练集。
"""
import hashlib
import io

import numpy as np
from PIL import Image

from miniocr.data.latex_corpus import _parquet_paths, clean_formula, normalization_key, standard_split


def load_formula_images(n=20, split="val"):
    """返回原始分辨率灰度墨迹图及可追踪 ID；只暴露 train/val，禁止读取 test 图。

    完整语料定划分，随后在 parquet 固定顺序中取 n 个不同键。这个子集不等同于
    standard_split 列表的前 n 行。不足 n 张就报错，缺缓存返回空列表供 CLI 显式跳过。
    """
    if split not in ("train", "val") or n < 1:
        raise ValueError("split must be train/val and n positive")
    paths = _parquet_paths()
    if not paths:
        return []
    import pyarrow.parquet as pq
    partitions = standard_split(n_train=None)
    keys = {normalization_key(t) for t in partitions[split]}
    seen, out = set(), []
    for path in paths:
        row = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=["image", "latex_formula"]):
            for item in batch.to_pylist():
                current_row, row = row, row + 1
                text = clean_formula(item["latex_formula"])
                key = normalization_key(text)
                if key not in keys or key in seen:
                    continue
                payload = item["image"]["bytes"]
                if not payload:
                    raise ValueError("expected embedded PNG bytes")
                with Image.open(io.BytesIO(payload)) as im:
                    # 透明处合成白纸，避免透明背景被转换成黑色。
                    rgba = im.convert("RGBA")
                    paper = Image.new("RGBA", rgba.size, "white")
                    paper.alpha_composite(rgba)
                    ink = 1 - np.asarray(paper.convert("L"), dtype=np.float32) / 255
                out.append(dict(id=f"{path.parent.parent.name}/{path.name}:{current_row}",
                                key_sha256=hashlib.sha256(key.encode()).hexdigest(),
                                split=split, formula=text, ink=ink))
                seen.add(key)
                if len(out) == n:
                    return out
    raise ValueError(f"requested {n} distinct {split} images, found {len(out)}")
