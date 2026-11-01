"""第 35 章：1 KL 机制，2 教师序列生成，3 质检，4 学生 LoRA 对照，5 最终 test。"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
from PIL import Image
import torch
from miniocr.train.distill import masked_kl, answer_labels, quality_check
from miniocr.models.vlm_anatomy import local_snapshot, load_image_only_processor, MODELS
from miniocr.data.postcorrect import PostCorrectExample
from miniocr.eval.postcorrect_eval import score_postcorrect, extract_formula

PROMPT = "Read the mathematical formula in this image. Output only its LaTeX, without explanation or delimiters."


def emit(**kwargs):
    print(json.dumps(kwargs, ensure_ascii=False, default=str), flush=True)


def save(args, name, value):
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def rows(args, name, n=None):
    if args.data is None: raise ValueError("requires --data")
    data=json.loads((args.data/(name+".json")).read_text())
    if n is not None and len(data)<n:
        raise ValueError(f"requested {n} {name} rows, only {len(data)} available")
    return data if n is None else data[:n]


def load(args, key):
    from transformers import AutoModelForImageTextToText
    path = local_snapshot(key, args.model_cache)
    if path is None: raise FileNotFoundError(f"missing local weights for {MODELS[key]}")
    processor = load_image_only_processor(path, key)
    if key == "smol":
        processor.image_processor.do_image_splitting = False
        processor.image_processor.size = {"longest_edge": 512}
    else:
        processor.image_processor.size = {"shortest_edge": 65536, "longest_edge": args.max_pixels}
    model = AutoModelForImageTextToText.from_pretrained(path, local_files_only=True,
        dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float32,
        attn_implementation="sdpa").to(args.device)
    emit(model=key, snapshot=str(path), parameters=sum(p.numel() for p in model.parameters()))
    return model, processor, path


def image_for(args, row, scale=1.):
    image = Image.open(args.data/row["image"]).convert("RGB")
    if scale != 1:
        # 控制变量：对开源图下采样再恢复尺寸，不自己渲染训练图。
        size = image.size
        image = image.resize(tuple(max(1, round(v*scale)) for v in size), Image.Resampling.BILINEAR)
        image = image.resize(size, Image.Resampling.BILINEAR)
    return image


def encode(args, processor, row, target=None, scale=1.):
    messages = [{"role": "user", "content": [{"type":"image"}, {"type":"text", "text":PROMPT}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image = image_for(args, row, scale)
    prefix = processor(text=prompt, images=[image], return_tensors="pt")
    if target is None:
        result = prefix
    else:
        messages.append({"role":"assistant", "content":[{"type":"text", "text":target}]})
        full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        result = processor(text=full, images=[image], return_tensors="pt")
        result["labels"] = answer_labels(result["input_ids"], prefix["input_ids"], result["attention_mask"])
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    return {k:v.to(args.device, dtype=dtype if v.is_floating_point() else v.dtype) for k,v in result.items()}


@torch.inference_mode()
def predict(args, model, processor, row, scale=1.):
    inputs = encode(args, processor, row, scale=scale)
    result = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                            use_cache=True, pad_token_id=processor.tokenizer.pad_token_id)
    ids = result[0, inputs["input_ids"].shape[1]:]
    raw = processor.tokenizer.decode(ids, skip_special_tokens=True)
    return dict(raw=raw, pred=extract_formula(raw), tokens=len(ids), truncated=len(ids) >= args.max_new_tokens)


@torch.inference_mode()
def predict_batch(args, model, processor, data, scale=1.):
    """Left-padded greedy decoding; each image remains paired with its prompt."""
    if len(data) == 1:
        return [predict(args, model, processor, data[0], scale)]
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    old_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"
    try:
        inputs = processor(text=[prompt]*len(data), images=[[image_for(args, r, scale)] for r in data],
                           padding=True, return_tensors="pt")
    finally:
        processor.tokenizer.padding_side = old_side
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    inputs = {k:v.to(args.device, dtype=dtype if v.is_floating_point() else v.dtype) for k,v in inputs.items()}
    output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                            use_cache=True, pad_token_id=processor.tokenizer.pad_token_id)
    records = []
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else (eos or [])
    for ids in output[:, inputs["input_ids"].shape[1]:].tolist():
        end = next((i+1 for i,t in enumerate(ids) if t in eos), len(ids))
        ids = ids[:end]
        raw = processor.tokenizer.decode(ids, skip_special_tokens=True)
        records.append(dict(raw=raw, pred=extract_formula(raw), tokens=len(ids),
                            truncated=len(ids)>=args.max_new_tokens and ids[-1] not in eos))
    return records


def scores(data, predictions):
    examples = [PostCorrectExample(r["id"], r["split"], r["target"], r["target"]) for r in data]
    result = score_postcorrect(examples, predictions, bootstrap_samples=500)
    # postcorrect 的 clean/fix 指标不适用于图像 OCR，只复用共同的评测口径。
    return {k:result["summary"][k] for k in ("n", "em_raw", "em_norm", "cer", "parse_rate")}


def evaluate(args, model, processor, data, name):
    predictions, observations = [], []
    model.eval(); start=time.perf_counter()
    for i in range(0, len(data), args.batch_size):
        chunk=data[i:i+args.batch_size]
        for row, observation in zip(chunk, predict_batch(args, model, processor, chunk)):
            predictions.append(observation["pred"])
            observations.append(dict(id=row["id"], **observation))
        emit(evaluation=name, completed=len(predictions), total=len(data))
    result = dict(metrics=scores(data, predictions), seconds=time.perf_counter()-start,
                  truncated=sum(r["truncated"] for r in observations))
    save(args, name+".json", result)
    if not name.endswith("test"):
        save(args, name+"_predictions.json", observations)
    emit(evaluation=name, **result)
    return result


def part1(args):
    """同词表小模型拟合教师分布，明确不是异构 VLM logits 对齐。"""
    teacher = torch.randn(8, 6, 11, device=args.device, requires_grad=True)
    student = torch.nn.Parameter(torch.zeros_like(teacher))
    mask = torch.ones(8, 6, device=args.device); mask[:, :2] = 0
    opt = torch.optim.Adam([student], lr=.15)
    curve=[]
    for step in range(args.kl_steps):
        opt.zero_grad(); loss=masked_kl(student, teacher, mask, args.temperature)
        loss.backward(); opt.step(); curve.append(float(loss.detach()))
    assert teacher.grad is None and torch.isfinite(student.grad).all()
    result=dict(scope="same-vocabulary random logits mechanism only", initial=curve[0],
        final=float(masked_kl(student,teacher,mask,args.temperature).detach()), curve=curve,
        temperature=args.temperature, steps=args.kl_steps, teacher_has_gradient=teacher.grad is not None)
    save(args,"kl.json",result); emit(part=1,**{k:v for k,v in result.items() if k!="curve"})


def part2(args):
    model, processor, path = load(args,"qwen3vl"); model.eval()
    if args.teacher_adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.teacher_adapter).merge_and_unload()
        model.eval()
    data=rows(args,"teacher",args.n)
    start=time.perf_counter(); records=[]
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/"teacher_stream.jsonl").open("w") as stream:
        for i in range(0,len(data),args.batch_size):
            chunk=data[i:i+args.batch_size]
            for row,pred in zip(chunk,predict_batch(args,model,processor,chunk,args.scale)):
                record=dict(row, **pred, scale=args.scale)
                records.append(record); stream.write(json.dumps(record, ensure_ascii=False)+"\n"); stream.flush()
            emit(teacher_completed=len(records),total=len(data),seconds=time.perf_counter()-start)
    save(args,"teacher.json",dict(rows=records,metrics=scores(data,[r["pred"] for r in records]),
        snapshot=str(path),teacher_adapter=str(args.teacher_adapter),seconds=time.perf_counter()-start,scale=args.scale,prompt=PROMPT))
    del model; gc.collect()
    if args.device=="cuda": torch.cuda.empty_cache()


def part3(args):
    records=json.loads((args.teacher_dir/"teacher.json").read_text())["rows"]
    checked=[]
    for row in records:
        if row["split"]!="train": raise ValueError("pseudo labels must be train-only")
        qc=quality_check(row["raw"],row["target"],render_tolerance=args.render_tolerance,
                         truncated=row["truncated"])
        checked.append(dict(row,**qc))
    accepted=[r for r in checked if r["accepted"]]
    result=dict(total=len(checked),accepted=len(accepted),render_unknown=sum(r["render_match"] is None for r in checked),
        all_metrics=scores(checked,[r["pred"] for r in checked]),
        accepted_metrics=scores(accepted,[r["pred"] for r in accepted]),render_tolerance=args.render_tolerance,rows=checked)
    save(args,"quality.json",result)
    # 固定前缀抽样，包含通过与拒绝，不按教师是否正确挑选。
    sample=accepted[:6]+[r for r in checked if not r["accepted"]][:6]
    save(args,"audit_samples.json",sample)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(max(1,len(sample)),1,figsize=(12,max(2,len(sample)*1.8)),squeeze=False)
    for ax,row in zip(axes[:,0],sample):
        ax.imshow(image_for(args,row,row["scale"])); ax.axis("off")
        ax.set_title(f"{row['id'][:8]} accepted={row['accepted']}\nref: {row['target'][:100]}\npred: {row['pred'][:100]}", fontsize=8,parse_math=False)
    fig.tight_layout(); fig.savefig(args.out/"audit.png"); plt.close(fig)
    emit(part=3,**{k:v for k,v in result.items() if k!="rows"})


def training_pool(args):
    base=rows(args,"base",args.n_train)
    if any(r["split"]!="train" for r in base): raise ValueError("base labels must be train-only")
    if len({r["id"] for r in base})!=len(base): raise ValueError("duplicate base IDs")
    pool=[dict(r,label=r["target"],scale=1.,origin="original") for r in base]
    if args.arm=="distill":
        pseudo=json.loads((args.teacher_dir/"quality.json").read_text())["rows"]
        base_ids={r["id"] for r in base}
        for r in pseudo:
            if r["split"]!="train" or r["id"] in base_ids: raise ValueError("invalid pseudo-label split/overlap")
            if r["accepted"]: pool.append(dict(r,label=r["pred"],origin="teacher"))
        if len(pool)==len(base): raise ValueError("no accepted pseudo labels")
    return pool


def part4(args):
    from peft import LoraConfig, get_peft_model
    model,processor,path=load(args,"smol")
    config=LoraConfig(r=args.rank,lora_alpha=args.rank*2,lora_dropout=0.,
        target_modules=r".*text_model.*\.(q_proj|k_proj|v_proj|o_proj)$",task_type="CAUSAL_LM")
    model=get_peft_model(model,config)
    trainable=[p for p in model.parameters() if p.requires_grad]
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n and "text_model" in n for n in names)
    pool=training_pool(args)
    # 同一随机种子、固定优化步数/有效 batch；额外伪标签导致原始样本曝光减少，须如实报告。
    probe=encode(args,processor,pool[0],pool[0]["label"],pool[0]["scale"])
    mask_check=dict(prompt_tokens=int((probe["labels"]==-100).sum()),
                    answer_tokens=int((probe["labels"]!=-100).sum()))
    emit(mask_prompt=mask_check["prompt_tokens"],mask_answer=mask_check["answer_tokens"],
        trainable=sum(p.numel() for p in trainable),lora_tensors=len(names),first_modules=names[:4])
    assert (probe["labels"]!=-100).any()
    del probe
    opt=torch.optim.AdamW(trainable,lr=args.lr,weight_decay=.01)
    rng=random.Random(args.seed); order=[]; curve=[]; exposures={"original":0,"teacher":0}
    if args.device=="cuda": torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter(); model.train()
    for step in range(args.steps):
        opt.zero_grad(); losses=[]
        for _ in range(args.accum):
            if not order: order=list(range(len(pool))); rng.shuffle(order)
            row=pool[order.pop()]; exposures[row["origin"]]+=1
            batch=encode(args,processor,row,row["label"],row["scale"])
            loss=model(**batch,use_cache=False).loss
            if not torch.isfinite(loss): raise RuntimeError("non-finite loss")
            (loss/args.accum).backward(); losses.append(float(loss.detach()))
        grad=float(torch.nn.utils.clip_grad_norm_(trainable,1.))
        if not np.isfinite(grad) or grad==0: raise RuntimeError("invalid LoRA gradient")
        opt.step()
        curve.append(dict(step=step+1,loss=float(np.mean(losses)),grad_norm=grad))
        if (step+1)%10==0 or step==0: emit(arm=args.arm,**curve[-1],seconds=time.perf_counter()-start)
    train_seconds=time.perf_counter()-start
    model.save_pretrained(args.out/"adapter")
    result=dict(arm=args.arm,steps=args.steps,accum=args.accum,rank=args.rank,lr=args.lr,seed=args.seed,
        dtype=args.dtype,mask_check=mask_check,trainable=sum(p.numel() for p in trainable),trainable_names=names,
        n_pool=len(pool),exposures=exposures,curve=curve,seconds=train_seconds,snapshot=str(path),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if args.device=="cuda" else None,
        config=dict(prompt=PROMPT,image_size=512,image_splitting=False,max_new_tokens=args.max_new_tokens))
    save(args,"training.json",result)
    evaluate(args,model,processor,rows(args,"val",args.n),args.arm+"_val")
    # train 诊断：错误伪标签的完全复制率；不是留出集泛化指标。
    if (args.teacher_dir/"quality.json").exists():
        from miniocr.eval.latex_norm import normalize_latex
        wrong=[r for r in json.loads((args.teacher_dir/"quality.json").read_text())["rows"]
               if r["accepted"] and normalize_latex(r["pred"])!=normalize_latex(r["target"])]
        observations=[]
        for r in wrong[:args.audit_n]:
            p=predict(args,model,processor,r,r["scale"])
            observations.append(dict(id=r["id"],reference=r["target"],teacher=r["pred"],student=p["pred"],
                copies_teacher=normalize_latex(p["pred"])==normalize_latex(r["pred"]),
                correct=normalize_latex(p["pred"])==normalize_latex(r["target"])))
        save(args,"error_transfer.json",dict(scope="train diagnostic only",available_wrong=len(wrong),
            n=len(observations),copied=sum(r["copies_teacher"] for r in observations),rows=observations))
    import matplotlib.pyplot as plt
    plt.plot([r["step"] for r in curve],[r["loss"] for r in curve]); plt.xlabel("optimizer step"); plt.ylabel("answer CE")
    plt.tight_layout(); plt.savefig(args.out/"loss.png"); plt.close()


def part5(args):
    if not args.final_test or args.adapter is None: raise ValueError("requires --final-test and --adapter")
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/"final_test.lock").exists(): raise FileExistsError(args.out/"final_test.lock")
    root = args.adapter.resolve().parent.parent
    selection = json.loads((root/"selection.json").read_text())
    chosen = selection.get("selected")
    if chosen not in ("original", "distill") or args.adapter.resolve() != root/chosen/"adapter":
        raise ValueError("adapter does not match val selection")
    digest = hashlib.sha256((args.adapter/"adapter_model.safetensors").read_bytes()).hexdigest()
    if digest != selection.get("adapter_sha256"):
        raise ValueError("adapter SHA256 does not match val selection")
    # 一次性锁在加载 test 之前创建；失败保留记录，不静默重跑。
    with (args.out/"final_test.lock").open("x") as f: f.write(datetime.now(timezone.utc).isoformat())
    from peft import PeftModel
    model,processor,path=load(args,"smol")
    model=PeftModel.from_pretrained(model,args.adapter)
    save(args,"frozen.json",dict(adapter=str(args.adapter),snapshot=str(path),n=args.n,
        max_new_tokens=args.max_new_tokens,prompt=PROMPT,selected_using="val only",
        selection=str(root/"selection.json"),adapter_sha256=digest))
    evaluate(args,model,processor,rows(args,"test",args.n),"final_test")


PARTS={1:part1,2:part2,3:part3,4:part4,5:part5}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("parts",nargs="*",type=int)
    p.add_argument("--device",choices=("cpu","mps","cuda"),default="cpu")
    p.add_argument("--dtype",choices=("fp32","bf16"),default="fp32")
    p.add_argument("--n",type=int,default=20)
    p.add_argument("--n-train",type=int,default=20)
    p.add_argument("--n-teacher",type=int,default=20)
    p.add_argument("--steps",type=int,default=5)
    p.add_argument("--accum",type=int,default=1)
    p.add_argument("--rank",type=int,default=8)
    p.add_argument("--lr",type=float,default=2e-4)
    p.add_argument("--seed",type=int,default=35)
    p.add_argument("--max-new-tokens",type=int,default=192)
    p.add_argument("--max-pixels",type=int,default=131072)
    p.add_argument("--scale",type=float,default=.75)
    p.add_argument("--render-tolerance",type=float,default=.65)
    p.add_argument("--kl-steps",type=int,default=50)
    p.add_argument("--temperature",type=float,default=2.)
    p.add_argument("--audit-n",type=int,default=12)
    p.add_argument("--arm",choices=("original","distill"),default="original")
    p.add_argument("--model-cache",type=Path)
    p.add_argument("--data",type=Path)
    p.add_argument("--out",type=Path,default=Path("outputs/book/ch35/local"))
    p.add_argument("--teacher-dir",type=Path,default=Path("outputs/book/ch35/teacher"))
    p.add_argument("--adapter",type=Path)
    p.add_argument("--teacher-adapter",type=Path)
    p.add_argument("--batch-size",type=int,default=1)
    p.add_argument("--val-only",action="store_true")
    p.add_argument("--prepare-data",action="store_true")
    p.add_argument("--final-test",action="store_true")
    args=p.parse_args(argv)
    if args.val_only and (5 in args.parts or args.final_test): p.error("val-only prohibits test")
    if args.batch_size < 1: p.error("batch-size must be positive")
    if any(x not in PARTS for x in args.parts): p.error("parts must be 1–5")
    if min(args.n,args.n_train,args.n_teacher,args.steps,args.accum,args.rank,args.max_new_tokens,args.kl_steps)<=0: p.error("scales must be positive")
    if not 0<args.scale<=1 or not 0<=args.render_tolerance<=1: p.error("invalid quality/scale settings")
    torch.set_num_threads(2)
    emit(start=datetime.now(timezone.utc).isoformat(),config=vars(args),torch=torch.__version__)
    start=time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    import platform
    environment=dict(start=datetime.now(timezone.utc).isoformat(),config=vars(args),
        python=platform.python_version(),torch=torch.__version__,device=args.device,
        gpu=torch.cuda.get_device_name() if args.device=="cuda" else None)
    if args.device=="cuda": torch.cuda.reset_peak_memory_stats()
    save(args,"run_start.json",environment)
    if args.prepare_data:
        from miniocr.data.distill_images import prepare_distill_images
        if args.data is None: p.error("--prepare-data requires --data")
        emit(data=prepare_distill_images(args.data,args.n_train,args.n_teacher,args.n,val_only=args.val_only))
    else:
        for part in args.parts or [1]:
            random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
            emit(part=part); PARTS[part](args)
    ending=dict(end=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-start,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if args.device=="cuda" else None)
    save(args,"run_end.json",ending)
    emit(**ending)


if __name__=="__main__": main()
