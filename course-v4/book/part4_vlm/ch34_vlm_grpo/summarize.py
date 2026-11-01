"""汇总第 34 章的 val 结果并按规则选择（只用 val）。

用法：python -m book.part4_vlm.ch34_vlm_grpo.summarize --root <run 根目录>
读取 <root>/<name>/val_result.json（部分 5 的输出）与 <root>/<name>/train_log.jsonl，
写出 selection.json、comparison.csv；不会填补缺失结果。
"""
import argparse
import csv
import json
from pathlib import Path


def ranking(result):
    """EM_norm 最大，打平时 CER 最小；只接受 val。"""
    if result['split'] != 'val':
        raise ValueError('禁止基于 test 选择')
    m = result['metrics']
    return m['em_norm']['value'], -m['cer']['value']


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args(argv)
    rows, scores = [], {}
    for path in sorted(args.root.glob('*/val_result.json')):
        result = json.loads(path.read_text())
        name = path.parent.name
        m = result['metrics']
        rows.append(dict(name=name, n=m['n'], em_norm=m['em_norm']['value'], cer=m['cer']['value'],
                         parse_rate=m['parse_rate']['value'], truncated=result.get('truncated'),
                         peak_allocated_gib=result.get('peak_allocated_gib')))
        scores[name] = ranking(result)
    if not scores:
        raise FileNotFoundError('没有 val_result.json，无法选择')
    choice = max(scores, key=scores.get)
    (args.root/'selection.json').write_text(json.dumps(dict(
        selected=choice, scores=scores, rule='val EM_norm max then CER min; freeze before test'),
        indent=2, ensure_ascii=False))
    with (args.root/'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(dict(selected=choice, rows=rows), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
