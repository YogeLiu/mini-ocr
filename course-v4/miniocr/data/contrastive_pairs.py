"""Chapter 30 paired data: cached open PNGs or a controlled mathtext fixture.

Open PNGs retain the full-corpus latex-v1 split. Only distinct normalized keys,
nonempty images and formulas fitting the byte budget are kept. This defines a
short-formula subset, not a representative sample of the complete corpus.
"""
import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from miniocr.data.latex_corpus import (
    _parquet_paths, clean_formula, normalization_key, standard_split,
)


def image_tensor(ink, size=(32, 160)):
    """Crop whitespace then resize with preserved aspect ratio; zero-ink padding."""
    rows, cols = np.where(ink > .05)
    if not len(rows):
        raise ValueError('blank image')
    ink = ink[rows.min():rows.max()+1, cols.min():cols.max()+1]
    h, w = size
    scale = min((h-4)/ink.shape[0], (w-4)/ink.shape[1])
    resized = Image.fromarray((ink*255).clip(0,255).astype('uint8')).resize(
        (max(1, round(ink.shape[1]*scale)), max(1, round(ink.shape[0]*scale))), Image.Resampling.BILINEAR)
    array = np.zeros(size, dtype=np.float32)
    rh, rw = resized.height, resized.width
    top, left = (h-rh)//2, (w-rw)//2
    array[top:top+rh, left:left+rw] = np.asarray(resized)/255
    return torch.from_numpy(array)[None]


def encode_bytes(texts, max_length=96):
    """Lossless UTF-8 bytes, padding=0; reject rather than truncate."""
    ids = torch.zeros(len(texts), max_length, dtype=torch.long)
    for i, text in enumerate(texts):
        raw = text.encode('utf-8')
        if not 0 < len(raw) <= max_length:
            raise ValueError('empty or overlength formula')
        ids[i, :len(raw)] = torch.tensor([b+1 for b in raw])
    return ids


def _pack(rows, images, source, split, max_length):
    keys = [r['key_sha256'] for r in rows]
    assert len(set(keys)) == len(keys)
    return dict(images=torch.stack(images), ids=encode_bytes([r['formula'] for r in rows], max_length),
                manifest=rows, source=source, split=split, max_length=max_length)


def open_pairs(counts, max_length=96, allow_test=False):
    """Read fixed parquet-order prefixes after full-corpus standard_split.

    Test images are decoded only with explicit allow_test=True. No download,
    no writes to the existing shared Hugging Face cache.
    """
    if not counts or any(s not in ('train','val','test') or n < 1 for s,n in counts.items()):
        raise ValueError('invalid split/count')
    if 'test' in counts and not allow_test:
        raise ValueError('test requires explicit final evaluation')
    paths = _parquet_paths()
    if not paths:
        raise FileNotFoundError('cleaned_formulas parquet cache required; no fallback/download')
    import pyarrow.parquet as pq
    partitions = standard_split(n_train=None)
    # Text membership inherits normalized group allocation from the shared module.
    membership = {text:s for s in counts for text in partitions[s]
                  if 0 < len(text.encode('utf-8')) <= max_length}
    rows, images, seen = ({s:[] for s in counts}, {s:[] for s in counts}, set())
    for path in paths:
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=['image','latex_formula']):
            for item in batch.to_pylist():
                row_id, offset = offset, offset+1
                text = clean_formula(item['latex_formula'])
                split = membership.get(text)
                if split is None or len(rows[split]) >= counts[split]:
                    continue
                key = hashlib.sha256(normalization_key(text).encode()).hexdigest()
                if key in seen:
                    continue
                payload = item['image']['bytes']
                with Image.open(io.BytesIO(payload)) as im:
                    rgba = im.convert('RGBA')
                    paper = Image.new('RGBA', rgba.size, 'white')
                    paper.alpha_composite(rgba)
                    ink = 1-np.asarray(paper.convert('L'), dtype=np.float32)/255
                try:
                    x = image_tensor(ink)
                except ValueError:
                    continue
                rows[split].append(dict(id=f'{path.parent.parent.name}/{path.name}:{row_id}',
                                        formula=text, key_sha256=key, split=split))
                images[split].append(x); seen.add(key)
                if all(len(rows[s]) == n for s,n in counts.items()):
                    return {s:_pack(rows[s], images[s], 'open', s, max_length) for s in counts}
    raise ValueError(f'insufficient eligible formulas: { {s:len(r) for s,r in rows.items()} }')


def rendered_pairs(counts, max_length=96, allow_test=False):
    """Fixed-font controlled compositional fixture, separate from latex-v1.

    Each unique formula's hash assigns its split BEFORE sample-size selection.
    This fixture is for CPU smoke/control experiments, not corpus claims.
    """
    if 'test' in counts and not allow_test:
        raise ValueError('test requires explicit final evaluation')
    from miniocr.eval.render import render
    rows, images = {s:[] for s in counts}, {s:[] for s in counts}
    for a in range(1, 100):
        for b in range(1, 100):
            for template in (r'{a}x+{b}', r'\frac{{{a}}}{{{b}}}', r'x^{{{a}}}+{b}'):
                text = template.format(a=a,b=b)
                key = hashlib.sha256(normalization_key(text).encode()).hexdigest()
                bucket = int(key[:8],16) % 10
                split = 'val' if bucket == 0 else 'test' if bucket == 1 else 'train'
                if split not in counts or len(rows[split]) >= counts[split]:
                    continue
                rows[split].append(dict(id=f'rendered:{key}', formula=text, key_sha256=key, split=split))
                images[split].append(image_tensor(render(text).astype(np.float32)/255))
                if all(len(rows[s]) == n for s,n in counts.items()):
                    return {s:_pack(rows[s],images[s],'rendered',s,max_length) for s in counts}
    raise ValueError('requested controlled fixture exceeds capacity')


def prepare_pairs(directory, counts, source='rendered', max_length=96, allow_test=False):
    """Persist tensors/manifests under the caller-owned run directory, never source."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if source not in ('open','rendered'):
        raise ValueError('unknown source')
    missing = {}
    for split,n in counts.items():
        if split == 'test' and not allow_test:
            raise ValueError('test requires explicit final evaluation')
        path = directory/f'{split}.pt'
        if path.exists():
            data = torch.load(path, weights_only=True)
            if len(data['manifest']) != n or data['source'] != source or data['max_length'] != max_length:
                raise ValueError('cache configuration mismatch; use a fresh --data-dir')
        else:
            missing[split] = n
    if missing:
        maker = open_pairs if source == 'open' else rendered_pairs
        for split,data in maker(missing, max_length, allow_test).items():
            torch.save(data, directory/f'{split}.pt')
    return {s:torch.load(directory/f'{s}.pt', weights_only=True) for s in counts}
