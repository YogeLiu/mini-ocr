"""从已落盘的结果生成 CSV/图；选择阶段只读取 val，test 不能参与选择。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone


def summarize(root, select=False):
    root=Path(root)
    data=[]; validation={}
    for arm in ('original','distill'):
        result=json.loads((root/arm/(arm+'_val.json')).read_text())
        training=json.loads((root/arm/'training.json').read_text())
        m=result['metrics']; validation[arm]=m
        data.append(dict(arm=arm,split='val',n=m['n'],em_norm=m['em_norm']['value'],
            cer=m['cer']['value'],parse_rate=m['parse_rate']['value'],truncated=result['truncated'],
            training_seconds=training['seconds'],steps=training['steps'],n_pool=training['n_pool']))
    if select:
        # 最低 CER；同分时最高 EM，仍同分选 original，预先固定规则。
        chosen=min(validation,key=lambda arm:(validation[arm]['cer']['value'],
            -validation[arm]['em_norm']['value'],arm!='original'))
        artifact=root/chosen/'adapter/adapter_model.safetensors'
        result=dict(selected=chosen,rule='min val CER; tie max val EM; tie original',
            timestamp=datetime.now(timezone.utc).isoformat(),validation=validation,
            adapter_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
        with (root/'selection.json').open('x') as file:
            json.dump(result,file,indent=2)
        print(json.dumps(result)); return
    if (root/'final/final_test.json').exists():
        result=json.loads((root/'final/final_test.json').read_text()); m=result['metrics']
        chosen=json.loads((root/'selection.json').read_text())['selected']
        data.append(dict(arm=chosen,split='test-final-once',n=m['n'],em_norm=m['em_norm']['value'],
            cer=m['cer']['value'],parse_rate=m['parse_rate']['value'],truncated=result['truncated']))
    with (root/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3))
    for arm in validation:
        training=json.loads((root/arm/'training.json').read_text())
        curve=training['curve']; window=20
        xs=[r['step'] for r in curve]
        ys=[sum(r['loss'] for r in curve[max(0,i-window+1):i+1])/min(window,i+1) for i in range(len(curve))]
        axes[0].plot(xs,ys,label=arm)
    axes[0].set(xlabel='optimizer step',ylabel='answer CE (20-step mean)'); axes[0].legend()
    arms=list(validation)
    axes[1].bar(arms,[validation[a]['cer']['value'] for a in arms])
    axes[1].set(ylabel='validation normalized CER',title=f"{validation[arms[0]]['n']} images; lower is better")
    fig.tight_layout(); fig.savefig(root/'comparison.png'); plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--select',action='store_true')
    a=p.parse_args(); summarize(a.root,a.select)
