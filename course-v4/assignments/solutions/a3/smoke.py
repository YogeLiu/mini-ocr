"""维护者 CPU 管线冒烟，不是训练配置/门槛/效果证据。
CUDA_VISIBLE_DEVICES='' python -m assignments.solutions.a3.smoke --out /tmp/a3-smoke
仅自造 train/val，随机 tiny Qwen3；两步训练、保存重载、生成和正式评测入口。
"""
import argparse
import json
from pathlib import Path
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from miniocr.data.sft import encode_sft
from miniocr.data.postcorrect import PostCorrectExample,NoiseEdit,save_jsonl
from miniocr.nn.lora import inject_lora
from miniocr.train.sft import train_sft
from assignments.a3 import evaluate as E


class Tokenizer:
    eos_token_id=2
    def apply_chat_template(self,messages,**kwargs): return 'P'+messages[-1]['content']+':'
    def encode(self,text,**kwargs): return [{'P':3,':':4,'x':5,'0':6,'O':7}.get(c,8) for c in text]


def model():
    m=Qwen3ForCausalLM(Qwen3Config(vocab_size=16,hidden_size=16,intermediate_size=24,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        max_position_embeddings=64,bos_token_id=1,eos_token_id=2,pad_token_id=0))
    inject_lora(m,rank=2,alpha=4)
    return m


def predict(sources,config):
    torch.set_num_threads(1)
    m=model(); m.load_state_dict(torch.load(config['kwargs']['checkpoint'],map_location='cpu',weights_only=True)); m.eval()
    tok=Tokenizer(); texts=[]
    with torch.no_grad():
        for source in sources:
            ids=torch.tensor([tok.encode('P'+source+':')])
            output=m.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=3,do_sample=False,
                              pad_token_id=0,eos_token_id=2)
            texts.append(''.join({5:'x',6:'0',7:'O'}.get(i,'?') for i in output[0,ids.shape[1]:].tolist() if i!=2))
    return texts


def prompt(sources,config):
    # 随机未训练基座，同样模板和生成预算；只验证提示基线接线。
    return predict(sources,config)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True); a=p.parse_args()
    out=E.writable(a.out); out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1); torch.manual_seed(3)
    m=model(); torch.save(m.state_dict(),out/'base.pt')
    examples=[encode_sft(Tokenizer(),'O','0'),encode_sft(Tokenizer(),'x','x')]
    history=train_sft(m,examples,0,steps=2,batch_size=1,accumulation=2,lr=.001)
    torch.save(m.state_dict(),out/'trained.pt')
    reload=model(); reload.load_state_dict(torch.load(out/'trained.pt',weights_only=True)); m.eval(); reload.eval()
    with torch.no_grad():
        ids=torch.tensor([[3,7,4]])
        torch.testing.assert_close(m(ids).logits,reload(ids).logits)
    save_jsonl([PostCorrectExample('smoke-val-1','val','O','0',(NoiseEdit('confusable',0,'0','O'),)),
                PostCorrectExample('smoke-val-2','val','x','x')],out/'val.jsonl')
    for name,ckpt in [('model','trained.pt'),('prompt','base.pt')]:
        cfg=dict(predictor='assignments.solutions.a3.smoke:predict',kwargs={'checkpoint':str(out/ckpt)},
                 artifacts={str(out/ckpt):E.digest(out/ckpt),str(Path(__file__).resolve()):E.digest(__file__)},
                 run_id='cpu-smoke-only',model_revision='tiny-random',prompt='P{source}:',generation={'max_new_tokens':3},seed=3)
        (out/f'{name}.json').write_text(json.dumps(cfg))
    (out/'training.json').write_text(json.dumps({'scope':'smoke only, no threshold evidence','history':history,'reload_verified':True}))
    E.main(['--data',str(out/'val.jsonl'),'--config',str(out/'model.json'),'--baseline',str(out/'prompt.json')])


if __name__=='__main__': main()
