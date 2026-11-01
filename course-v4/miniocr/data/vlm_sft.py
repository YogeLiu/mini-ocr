"""图文 SFT：原始开源 PNG、latex-v1 分组划分、严格的答案监督 mask。"""
from io import BytesIO
import hashlib
import json
from pathlib import Path
import torch
from PIL import Image
from miniocr.data.latex_corpus import _parquet_paths, standard_split, clean_formula, normalization_key

PROMPT = 'Transcribe the mathematical formula in this image into LaTeX. Output only the LaTeX formula, without explanation or markdown.'


def prepare_images(out, n_train=2048, n_eval=100, *, include_test=True):
    """完整语料先划分，取每个 split 的唯一公式前缀，保存首次出现的原图。"""
    import pyarrow.parquet as pq
    paths = _parquet_paths()
    if not paths:
        raise FileNotFoundError('需要已有 cleaned_formulas parquet，不下载、不使用内置样例')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    splits = standard_split(n_train=None)
    wanted = {s: list(dict.fromkeys(splits[s]))[:n_train if s == 'train' else n_eval]
              for s in (('train', 'val', 'test') if include_test else ('train', 'val'))}
    remaining = set(sum(wanted.values(), [])); found = {}
    source_count = sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
    for path in paths:
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=['latex_formula', 'image']):
            for raw, value in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
                target = clean_formula(raw)
                if target in remaining:
                    content = value['bytes']; im = Image.open(BytesIO(content)).convert('RGB')
                    identity = hashlib.sha256((path.name+':'+str(offset)).encode()).hexdigest()[:20]
                    im.save(out/(identity+'.png'))
                    found[target] = dict(id=identity, target=target, image=identity+'.png',
                        width=im.width, height=im.height, parquet=path.name, row=offset,
                        normalization_key=normalization_key(target), image_sha256=hashlib.sha256(content).hexdigest())
                    remaining.remove(target)
                offset += 1
            if not remaining: break
        if not remaining: break
    if remaining: raise RuntimeError(f'找不到 {len(remaining)} 张图片')
    keys = {s: {normalization_key(t) for t in ts} for s, ts in wanted.items()}
    assert all(not (keys[a] & keys[b]) for a in keys for b in keys if a != b)
    for split, ts in wanted.items():
        (out/f'{split}.json').write_text(json.dumps([{**found[t], 'split': split} for t in ts], ensure_ascii=False, indent=2))
    meta = dict(source='OleehyO/latex-formulas', config='cleaned_formulas', license='openrail',
                protocol='latex-v1', seed=0, source_rows=source_count, snapshots=[str(p) for p in paths],
                counts={s: len(ts) for s, ts in wanted.items()}, normalization_overlap=0,
                selection='unique formula prefix after full corpus standard_split; first image occurrence')
    (out/'metadata.json').write_text(json.dumps(meta, indent=2)); return meta


# >>> 练习 33.1：只监督答案；padding 按 attention_mask 而不是 token 值判定。
def answer_labels(input_ids, attention_mask, prefix_lengths, image_token_ids=()):
    """支持左右 padding；prefix_lengths 是不含 padding 的提示 token 数。"""
    if input_ids.shape != attention_mask.shape or input_ids.ndim != 2:
        raise ValueError('input_ids/attention_mask 必须是同形二维张量')
    lengths = torch.as_tensor(prefix_lengths, device=input_ids.device)
    if lengths.shape != (input_ids.shape[0],) or (lengths < 0).any():
        raise ValueError('每个样本需要非负 prefix 长度')
    position = attention_mask.long().cumsum(-1)
    labels = input_ids.clone()
    labels[(attention_mask == 0) | (position <= lengths[:, None])] = -100
    for token_id in image_token_ids: labels[input_ids == token_id] = -100
    if (labels != -100).sum(-1).min() == 0:
        raise ValueError('存在没有答案监督 token 的样本；禁止静默截断')
    return labels
# <<<


class FormulaCollator:
    """官方 chat template/视觉展开；验证 prompt 是完整样本的精确 token 前缀。"""
    def __init__(self, processor, root, max_tokens=2048):
        self.processor, self.root, self.max_tokens = processor, Path(root), max_tokens

    def __call__(self, rows):
        p = self.processor; images = []; full = []; prefixes = []
        for row in rows:
            with Image.open(self.root/row['image']) as im: images.append(im.convert('RGB'))
            messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': PROMPT}]}]
            prefixes.append(p.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))
            messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': row['target']}]})
            full.append(p.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False))
        batch = p(text=full, images=images, padding=True, return_tensors='pt')
        prompt = p(text=prefixes, images=images, padding=True, return_tensors='pt')
        lens = prompt['attention_mask'].sum(-1).tolist()
        for i, length in enumerate(lens):
            a = batch['input_ids'][i][batch['attention_mask'][i].bool()]
            b = prompt['input_ids'][i][prompt['attention_mask'][i].bool()]
            if not torch.equal(a[:length], b): raise ValueError('chat template 提示/完整样本前缀不一致')
            if len(a) > self.max_tokens: raise ValueError(f'样本 {rows[i]["id"]} 超过 max_tokens={self.max_tokens}，不截断视觉 token')
        special = [p.tokenizer.convert_tokens_to_ids(t) for t in ('<|image_pad|>', '<|vision_start|>', '<|vision_end|>')]
        batch['labels'] = answer_labels(batch['input_ids'], batch['attention_mask'], lens, special)
        return batch


def audit_samples(processor, root, rows, max_tokens=2048):
    """不载模型，预检全部训练样本的精确展开长度、视觉 mask 和答案监督。"""
    collator = FormulaCollator(processor, root, max_tokens)
    observations = []
    for row in rows:
        batch = collator([row])
        labels, ids = batch['labels'], batch['input_ids']
        assert (labels[ids == processor.image_token_id] == -100).all()
        observations.append(dict(id=row['id'],tokens=int(batch['attention_mask'].sum()),
            supervised=int((labels != -100).sum()),image_tokens=int((ids == processor.image_token_id).sum())))
    return dict(n=len(rows),max_tokens=max(r['tokens'] for r in observations),
                min_supervised=min(r['supervised'] for r in observations),rows=observations)
