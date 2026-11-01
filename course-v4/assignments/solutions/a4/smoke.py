"""仅 CPU 的合成微型策略冒烟；不是课程训练配置或门槛校准。
A_SOLUTIONS=1 python -m assignments.solutions.a4.smoke --directory /tmp/a4-smoke
保存并重新加载 2×2 条件 logits，分别走 DPO 与 GRPO 的前向、反向、更新。
"""
import argparse
import json
from pathlib import Path
import torch
from . import adapters as A


def load(checkpoint,device,settings):
    weights=torch.load(checkpoint,map_location=device,weights_only=True)['weights']
    def predict(sources):
        return ['xy'[int(weights[int(s=='y')].argmax())] for s in sources]
    return predict


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--directory',required=True)
    args=ap.parse_args()
    out=Path(args.directory).resolve()
    workspace=Path(__file__).resolve().parents[3]
    if out.is_relative_to((workspace/'outputs').resolve()):
        raise ValueError('不能写 outputs')
    out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    from assignments.a4.evaluate import main as evaluate
    for method in ('dpo','grpo'):
        w=torch.nn.Parameter(torch.zeros(2,2,dtype=torch.float64))
        optimizer=torch.optim.SGD([w],lr=.05)
        initial=w.detach().clone()
        history=[]
        for step in range(3):
            lp=w.log_softmax(-1)
            if method=='dpo':
                loss=A.dpo_loss(lp[:,1],lp[:,0],torch.full((2,),-torch.log(torch.tensor(2.)).item(),dtype=w.dtype),
                                torch.full((2,),-torch.log(torch.tensor(2.)).item(),dtype=w.dtype))[0].mean()
            else:
                actions=torch.multinomial(lp.detach().exp(),4,replacement=True)
                selected=lp.gather(1,actions).reshape(-1,1)
                rewards=(actions==1).to(w.dtype).reshape(-1)
                adv=A.group_advantages(rewards,4)
                ref=torch.full_like(selected,-torch.log(torch.tensor(2.,dtype=w.dtype)).item())
                loss,_=A.grpo_loss(selected,selected.detach(),ref,adv,torch.ones_like(selected))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            assert torch.isfinite(w.grad).all()
            history.append(dict(step=step,loss=float(loss.detach())))
        assert not torch.equal(w.detach(),initial)
        ck=out/f'{method}.pt'; torch.save(dict(weights=w.detach()),ck)
        cfg=out/f'{method}.json'
        cfg.write_text(json.dumps(dict(factory='assignments.solutions.a4.smoke:load',checkpoint=str(ck),settings={},device='cpu',protocol_version='synthetic-smoke-only')))
        dev=out/'dev.jsonl'
        dev.write_text('\n'.join(json.dumps(dict(id=str(i),split='val',source=s,target='y',is_clean=s=='y')) for i,s in enumerate(['x','y']))+'\n')
        (out/f'{method}-history.json').write_text(json.dumps(history))
        result=evaluate(['--config',str(cfg),'--data',str(dev),'--device','cpu'])
        (out/f'{method}-summary.json').write_text(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
