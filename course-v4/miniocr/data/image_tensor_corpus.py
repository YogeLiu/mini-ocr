"""第 27 章数据准备：开源原图为主，val 重渲染只用于控制变量。

完整语料先走 latex-v1，随后按规范键去重；不根据模型成绩筛选。
测试集只保存原图，禁止在开发实验读取预测或做错例分析。
"""
import hashlib
import io
import json
from pathlib import Path
from PIL import Image, ImageOps
from miniocr.data.latex_corpus import _parquet_paths, standard_split, normalization_key, clean_formula
from miniocr.eval.render import render


def distinct_prefix(texts, n):
    """按原顺序取 n 个规范键；返回的不同写法仍属于同一划分。"""
    seen, result = set(), []
    for text in texts:
        key = normalization_key(text)
        if key not in seen:
            seen.add(key)
            result.append(text)
            if len(result) == n:
                return result
    raise ValueError(f"need {n} distinct formulas, found {len(result)}")


def prepare_corpus(out, n=200, test_n=100, dpi=200):
    """保存 n 张原始 val、n 张可渲染 val（允许重叠）和 test_n 张原始 test。

    渲染失败完整计数，受控集不能冒充无偏开源评测。所有图片仍来自标准划分。
    manifest 存哈希、原文件行号、尺寸；原始缓存只读。
    """
    import pyarrow.parquet as pq
    paths = _parquet_paths()
    if not paths:
        raise FileNotFoundError("need existing cleaned_formulas parquet; no synthetic fallback")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/"manifest.json").exists():
        raise FileExistsError("use a new data directory to preserve the existing manifest")
    splits = standard_split(n_train=0)
    originals = distinct_prefix(splits["val"], n)
    tests = distinct_prefix(splits["test"], test_n)
    rendered, rejected, seen = {}, [], set()
    for text in splits["val"]:
        key = normalization_key(text)
        if key in seen:
            continue
        seen.add(key)
        try:
            image = ImageOps.expand(Image.fromarray(255-render(text, dpi=dpi)).convert("RGB"), border=12, fill="white")
        except ValueError as exc:
            rejected.append(dict(key_sha256=hashlib.sha256(key.encode()).hexdigest(), reason=str(exc)))
            continue
        name = hashlib.sha256(key.encode()).hexdigest()[:24]+"_rendered.png"
        image.save(out/name)
        rendered[text] = name
        if len(rendered) == n:
            break
    if len(rendered) != n:
        raise ValueError("not enough renderable val formulas")
    wanted = set(originals) | set(tests) | set(rendered)
    found = {}
    for path in paths:
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=["image", "latex_formula"]):
            for item in batch.to_pylist():
                row, offset = offset, offset+1
                text = clean_formula(item["latex_formula"])
                if text not in wanted:
                    continue
                payload = item["image"]["bytes"]
                with Image.open(io.BytesIO(payload)) as raw:
                    rgba = raw.convert("RGBA")
                    image = Image.new("RGBA", rgba.size, "white")
                    image.alpha_composite(rgba)
                    image = image.convert("RGB")
                key_hash = hashlib.sha256(normalization_key(text).encode()).hexdigest()
                name = key_hash[:24]+"_original.png"
                image.save(out/name)
                found[text] = dict(id=key_hash[:24], key_sha256=key_hash, formula=text, original=name,
                                   rendered=rendered.get(text), width=image.width, height=image.height,
                                   parquet=path.name, row=row, png_sha256=hashlib.sha256(payload).hexdigest())
                wanted.remove(text)
            if not wanted:
                break
        if not wanted:
            break
    if wanted:
        raise RuntimeError(f"missing {len(wanted)} original PNGs")
    groups = dict(val_original=[found[t] for t in originals], val_rendered=[found[t] for t in rendered],
                  test_original=[found[t] for t in tests])
    val_keys = {r["key_sha256"] for r in groups["val_original"]+groups["val_rendered"]}
    assert not val_keys & {r["key_sha256"] for r in groups["test_original"]}
    result = dict(source="OleehyO/latex-formulas", config="cleaned_formulas", license="openrail",
                  snapshots=[str(p) for p in paths], protocol="latex-v1", split_seed=0, dpi=dpi,
                  n=n, test_n=test_n, render_attempts=len(seen), render_failures=len(rejected),
                  selection="distinct normalized keys in full standard_split order; renderable subset only for controlled comparisons",
                  rejected=rejected, **groups)
    (out/"manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    return result
