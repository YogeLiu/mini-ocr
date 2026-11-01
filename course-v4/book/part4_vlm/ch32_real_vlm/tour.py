"""第 32 章：真实 VLM 解剖。默认仅运行无权重的机制实验，真实推理显式选择 4/5。"""
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
from miniocr.models.vlm_anatomy import (
    MODELS, local_snapshot, smart_resize, replica_processor, MLPConnector, QueryConnector, load_image_only_processor, expand_image_placeholder,
)

PROMPT = "Transcribe the mathematical formula in this image into LaTeX. Output only the LaTeX formula, without explanation or markdown."


def emit(**values):
    print(" ".join(f"{k}={json.dumps(v, ensure_ascii=False, default=str)}" for k, v in values.items()), flush=True)


def save(args, name, value):
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        destination = args.out/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def sync(args):
    if args.device == "cuda":
        torch.cuda.synchronize()
    elif args.device == "mps":
        torch.mps.synchronize()


def hf_image_processor(args):
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
    path = local_snapshot("qwen35", args.model_cache)
    kwargs = dict(size={"shortest_edge": args.min_pixels, "longest_edge": args.max_pixels})
    if path:
        return Qwen2VLImageProcessorPil.from_pretrained(path, local_files_only=True, **kwargs)
    return Qwen2VLImageProcessorPil(patch_size=16, temporal_patch_size=2, merge_size=2,
                                   image_mean=[.5]*3, image_std=[.5]*3, **kwargs)


def sample_images(args):
    if args.data and (args.data/"val.json").exists():
        rows = json.loads((args.data/"val.json").read_text())[:args.n]
        return [(r["id"], Image.open(args.data/r["image"]).convert("RGB")) for r in rows]
    rng = np.random.default_rng(args.seed)
    return [(f"synthetic-{h}x{w}", Image.fromarray(rng.integers(0, 256, (h,w,3), dtype=np.uint8)))
            for h,w in [(37, 239), (64,64), (333,177), (48,960)]]


def part1(args):
    processor = hf_image_processor(args)
    rows = []
    path = local_snapshot("qwen35", args.model_cache)
    full = load_image_only_processor(path,"qwen35") if path else None
    if full is not None: full.image_processor = processor
    for name, im in sample_images(args):
        ref = processor(images=im, return_tensors="np")
        own = replica_processor(im, patch_size=processor.patch_size, merge_size=processor.merge_size,
                                temporal_patch_size=processor.temporal_patch_size, mean=processor.image_mean,
                                std=processor.image_std, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
        a,b = own["pixel_values"], ref["pixel_values"]
        row = dict(id=name, original_hw=[im.height, im.width], resized_hw=own["resized_hw"],
                   shape=list(a.shape), grid=own["image_grid_thw"].tolist(),
                   grid_equal=np.array_equal(own["image_grid_thw"],ref["image_grid_thw"]),
                   unequal_elements=int(np.count_nonzero(a != b)), max_abs_error=float(np.max(np.abs(a-b))),
                   merged_tokens=int(np.prod(own["image_grid_thw"])//processor.merge_size**2))
        if not row["grid_equal"] or row["unequal_elements"]:
            raise AssertionError(row)
        if full is not None:
            prompt=full.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":PROMPT}]}],
                                            tokenize=False,add_generation_prompt=True,enable_thinking=False)
            official=full(text=prompt,images=[im],return_tensors="np")
            expanded=expand_image_placeholder(prompt,full.image_token,own["image_grid_thw"],processor.merge_size)
            replica=full.tokenizer(expanded,return_tensors="np")
            row["input_ids_equal"]=bool(np.array_equal(official["input_ids"],replica["input_ids"]))
            row["attention_mask_equal"]=bool(np.array_equal(official["attention_mask"],replica["attention_mask"]))
            row["input_tokens"]=int(replica["input_ids"].shape[1])
            assert row["input_ids_equal"] and row["attention_mask_equal"]
        rows.append(row)
    save(args,"processor.json", dict(backend=type(processor).__name__, dtype="float32", rows=rows))
    emit(来源="本项目观察", processor=type(processor).__name__, n=len(rows),
         unequal_elements=sum(r["unequal_elements"] for r in rows), max_abs_error=max(r["max_abs_error"] for r in rows))


def part2(args):
    rows = []
    im = sample_images(args)[0][1]
    for scale in (.5,1,2,4):
        h,w = max(1,int(im.height*scale)),max(1,int(im.width*scale))
        for budget in (65536,262144,1048576):
            rh,rw=smart_resize(h,w,min_pixels=65536,max_pixels=budget)
            rows.append(dict(scale=scale,budget=budget,input_hw=[h,w],resized_hw=[rh,rw],
                             patches=rh*rw//256,visual_tokens=rh*rw//1024))
    save(args,"resolution.json", rows)
    for r in rows: emit(来源="数学结论（尺寸规则）", **r)
    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for budget in (65536,262144,1048576):
            subset=[r for r in rows if r["budget"]==budget]
            plt.plot([r["scale"] for r in subset],[r["visual_tokens"] for r in subset],marker="o",label=str(budget))
        plt.xlabel("Input scale"); plt.ylabel("Merged visual tokens"); plt.legend(title="pixel budget")
        plt.tight_layout(); plt.savefig(args.out/"resolution.png"); plt.close()


def part3(args):
    rows=[]
    dtype=torch.bfloat16 if args.dtype=="bf16" else torch.float32
    for tokens in (16,64,256):
        for cls in (MLPConnector,QueryConnector):
            model=cls().to(args.device,dtype=dtype)
            x=torch.randn(2,tokens,32,device=args.device,dtype=dtype,requires_grad=True)
            y=model(x); y.float().square().mean().backward()
            assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
            row=dict(connector=cls.__name__,input=list(x.shape),output=list(y.shape),
                     parameters=sum(p.numel() for p in model.parameters()),input_grad_norm=float(x.grad.float().norm()))
            rows.append(row); emit(来源="本项目观察，随机结构探针，非训练结果",**row)
    save(args,"connectors.json",rows)


def load_model(args,key):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    path=local_snapshot(key,args.model_cache)
    if path is None:
        raise FileNotFoundError(f"缺少本地 {MODELS[key]} 权重；本脚本不会联网下载")
    processor=load_image_only_processor(path,key)
    if key != "smol":
        processor.image_processor.size={"shortest_edge":args.min_pixels,"longest_edge":args.max_pixels}
    dtype=torch.bfloat16 if args.dtype=="bf16" else torch.float32
    started=time.perf_counter()
    model=AutoModelForImageTextToText.from_pretrained(path,local_files_only=True,dtype=dtype,
                                                    attn_implementation="sdpa").eval()
    if args.device=="cuda":
        # 共享租用实例：为激活和其他任务预留显存。
        required_gib={"smol":2,"qwen35":3,"qwen3vl":6}[key]
        waiting=time.perf_counter()
        while torch.cuda.mem_get_info()[0] < required_gib*2**30:
            emit(model=key,waiting_for_free_gib=required_gib,
                 free_gib=torch.cuda.mem_get_info()[0]/2**30)
            if time.perf_counter()-waiting>1200:
                raise RuntimeError("等待显存超过 20 分钟；已完成的模型结果仍保留，请空闲后续跑")
            time.sleep(15)
    model=model.to(args.device)
    emit(model=key,snapshot=str(path),load_seconds=time.perf_counter()-started,
         processor=type(processor.image_processor).__name__,parameters=sum(p.numel() for p in model.parameters()))
    return model,processor,path


def inputs_for(args,processor,image):
    messages=[{"role":"user","content":[{"type":"image"},{"type":"text","text":PROMPT}]}]
    prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    inputs=processor(text=prompt,images=[image],return_tensors="pt")
    dtype=torch.bfloat16 if args.dtype=="bf16" else torch.float32
    inputs={k:v.to(device=args.device,dtype=dtype if v.is_floating_point() else v.dtype) for k,v in inputs.items()}
    return inputs,prompt


def selected_rows(args):
    if args.split=="test" and (not args.final_test or len(args.models)!=1 or args.source!="original"):
        raise ValueError("test 只允许 --final-test、单一冻结模型和 original 图片")
    rows=json.loads((args.data/f"{args.split}.json").read_text())
    if args.source in ("rendered","paired-original"):
        rows=[r for r in rows if r["rendered"]]
    if len(rows)<args.n:
        raise ValueError(f"可用 {len(rows)} 条，小于 n={args.n}；请扩大准备集或降低 n")
    return rows[:args.n]


def part4(args):
    from miniocr.data.postcorrect import PostCorrectExample
    from miniocr.eval.postcorrect_eval import score_postcorrect,extract_formula
    if args.data is None: raise ValueError("推理需要 --data（先 --prepare-data）")
    rows=selected_rows(args)
    digest=hashlib.sha256(json.dumps([r["id"] for r in rows]).encode()).hexdigest()
    for key in args.models:
        name=f"{key}_{args.split}_{args.source}"
        # test 先落锁；即便失败也不能静默重评。
        if args.split=="test":
            if args.out is None: raise ValueError("最终 test 必须 --out 留审计记录")
            args.out.mkdir(parents=True,exist_ok=True)
            with (args.out/f"{name}.lock").open("x") as f: f.write(datetime.now(timezone.utc).isoformat())
        model,processor,path=load_model(args,key)
        warm=Image.open(args.data/rows[0]["image"]).convert("RGB")
        warm_inputs,_=inputs_for(args,processor,warm)
        with torch.inference_mode(): model.generate(**warm_inputs,max_new_tokens=2,do_sample=False)
        del warm_inputs
        sync(args)
        if args.device=="cuda": torch.cuda.reset_peak_memory_stats()
        observations=[]; preds=[]
        begun=time.perf_counter()
        for i,row in enumerate(rows):
            im=Image.open(args.data/row["rendered" if args.source=="rendered" else "image"]).convert("RGB")
            sync(args); start=time.perf_counter()
            inputs,prompt=inputs_for(args,processor,im)
            sync(args); processed=time.perf_counter()
            image_id=getattr(processor,"image_token_id",model.config.image_token_id)
            image_tokens=int((inputs["input_ids"]==image_id).sum())
            with torch.inference_mode():
                output=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False,use_cache=True)
            sync(args); done=time.perf_counter()
            tokens=output[0,inputs["input_ids"].shape[1]:]
            raw=processor.decode(tokens,skip_special_tokens=True)
            pred=extract_formula(raw); preds.append(pred)
            eos=model.generation_config.eos_token_id
            eos=[] if eos is None else eos if isinstance(eos,list) else [eos]
            obs=dict(id=row["id"],input_tokens=int(inputs["input_ids"].shape[1]),visual_tokens=image_tokens,
                     generated_tokens=len(tokens),truncated=len(tokens)==args.max_new_tokens and int(tokens[-1]) not in eos,
                     preprocess_seconds=processed-start,generate_seconds=done-processed,
                     grid=inputs["image_grid_thw"].cpu().tolist() if "image_grid_thw" in inputs else None,
                     pixel_shape=list(inputs["pixel_values"].shape))
            if args.split=="val": obs.update(target=row["target"],raw=raw,pred=pred)
            observations.append(obs)
            if args.split=="val" and ((i+1)%10==0 or i+1==len(rows)):
                save(args,f"predictions/{name}.json",dict(model=key,snapshot=str(path),split=args.split,
                     source=args.source,sample_ids_sha256=digest,completed=i+1,requested_n=len(rows),
                     dtype=args.dtype,max_new_tokens=args.max_new_tokens,prompt=prompt,observations=observations))
            if (i+1)%10==0 or i==0: emit(model=key,completed=i+1,n=len(rows),elapsed_seconds=time.perf_counter()-begun)
            del inputs,output,tokens
        elapsed=time.perf_counter()-begun
        examples=[PostCorrectExample(r["id"],args.split,r["target"],r["target"]) for r in rows]
        scores=score_postcorrect(examples,preds)["summary"]
        # 后纠错专用 broken/fix 分类不适用于图像识别，仅复用公共指标。
        scores={k:scores[k] for k in ("n","em_raw","em_norm","cer","parse_rate")}
        result=dict(model=key,model_id=MODELS[key],snapshot=str(path),split=args.split,source=args.source,
                    sample_ids_sha256=digest,device=args.device,dtype=args.dtype,batch_size=1,training_steps=0,
                    seed=args.seed,prompt=PROMPT,formatted_prompt=prompt,enable_thinking=False,
                    max_new_tokens=args.max_new_tokens,do_sample=False,attn_implementation="sdpa",
                    min_pixels=args.min_pixels,max_pixels=args.max_pixels,
                    processor=processor.image_processor.to_dict(),metrics=scores,
                    elapsed_seconds=elapsed,generate_seconds=sum(r["generate_seconds"] for r in observations),
                    generated_tokens=sum(r["generated_tokens"] for r in observations),
                    mean_input_tokens=float(np.mean([r["input_tokens"] for r in observations])),
                    mean_visual_tokens=float(np.mean([r["visual_tokens"] for r in observations])),
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if args.device=="cuda" else None,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if args.device=="cuda" else None,
                    truncated=sum(r["truncated"] for r in observations),observations=observations)
        save(args,name+".json",result)
        emit(来源="本项目观察",model=key,split=args.split,source=args.source,metrics=scores,
             seconds=elapsed,peak_allocated_gib=result["peak_allocated_gib"],truncated=result["truncated"])
        del model,processor
        gc.collect()
        if args.device=="cuda": torch.cuda.empty_cache()


def shape_tree(value):
    if isinstance(value,torch.Tensor): return list(value.shape)
    if isinstance(value,dict): return {k:shape_tree(v) for k,v in value.items() if v is not None}
    if isinstance(value,(tuple,list)): return [shape_tree(v) for v in value]
    return str(type(value).__name__)


def part5(args):
    model,processor,path=load_model(args,"qwen35")
    image=sample_images(args)[0][1]
    inputs,prompt=inputs_for(args,processor,image)
    traces={"processor":shape_tree(inputs)}
    handles=[]
    for name in ("model.visual.patch_embed","model.visual.blocks.0","model.visual.merger","model.visual","model.language_model"):
        module=model.get_submodule(name)
        def hook(mod,pos,kwargs,output,name=name):
            traces[name]=dict(input=shape_tree(pos),kwargs=shape_tree({k:v for k,v in kwargs.items() if isinstance(v,torch.Tensor)}),output=shape_tree(output))
        handles.append(module.register_forward_hook(hook,with_kwargs=True))
    with torch.inference_mode(): output=model(**inputs,use_cache=False)
    for h in handles: h.remove()
    traces["logits"]=list(output.logits.shape)
    traces["config"]=model.config.to_dict()
    traces["snapshot"]=str(path)
    save(args,"dataflow.json",traces)
    emit(来源="本项目观察",dataflow={k:v for k,v in traces.items() if k!="config"})
    del output,model,processor,inputs
    gc.collect()
    if args.device=="cuda": torch.cuda.empty_cache()


def part6(args):
    notes={"scope":"只读模块说明；不下载 GLM-OCR/PaddleOCR-VL，未测其性能",
           "qwen35":"dynamic patches -> vision transformer -> spatial merger -> hybrid language decoder",
           "glm_ocr":"CogViT visual encoder -> cross-modal connector -> GLM language decoder; OCR-specialized",
           "paddleocr_vl":"NaViT-style dynamic-resolution visual encoder -> ERNIE-4.5 language model; document parsing",
           "sources":["https://huggingface.co/zai-org/GLM-OCR","https://huggingface.co/PaddlePaddle/PaddleOCR-VL",
                      "https://huggingface.co/Qwen/Qwen3.5-0.8B"]}
    save(args,"reading.json",notes); emit(来源="文档说明，非本项目性能观察",**notes)


PARTS={1:("processor 逐位对照",part1),2:("动态分辨率与 token 合并",part2),3:("MLP 与 query 连接器结构",part3),
       4:("真实模型零样本比较",part4),5:("Qwen3.5 数据流 hooks",part5),6:("OCR 模块阅读索引",part6)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts",nargs="*",type=int)
    parser.add_argument("--device",choices=("cpu","mps","cuda"),default="cpu")
    parser.add_argument("--dtype",choices=("fp32","bf16"),default="fp32")
    parser.add_argument("--n",type=int,default=20)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--data",type=Path)
    parser.add_argument("--prepare-data",action="store_true")
    parser.add_argument("--model-cache",type=Path)
    parser.add_argument("--models",nargs="+",choices=MODELS,default=list(MODELS))
    parser.add_argument("--min-pixels",type=int,default=65536)
    parser.add_argument("--max-pixels",type=int,default=262144)
    parser.add_argument("--max-new-tokens",type=int,default=256)
    parser.add_argument("--source",choices=("original","rendered","paired-original"),default="original")
    parser.add_argument("--split",choices=("val","test"),default="val")
    parser.add_argument("--final-test",action="store_true")
    args=parser.parse_args(argv)
    if args.n<1 or args.max_new_tokens<1 or any(p not in PARTS for p in args.parts): parser.error("n/token 上限须为正，部分号 1–6")
    if args.min_pixels<=0 or args.max_pixels<args.min_pixels: parser.error("像素预算无效")
    if args.device=="cuda" and not torch.cuda.is_available(): parser.error("CUDA 不可用")
    if args.device=="mps" and not torch.backends.mps.is_available(): parser.error("MPS 不可用")
    import transformers
    torch.set_num_threads(2)
    started=time.perf_counter()
    emit(时间=datetime.now(timezone.utc).isoformat(),torch=torch.__version__,transformers=transformers.__version__,
         device=torch.cuda.get_device_name() if args.device=="cuda" else args.device,dtype=args.dtype,n=args.n,
         batch_size=1,training_steps=0,默认n=20,seed=args.seed)
    if args.prepare_data:
        if not args.data: parser.error("--prepare-data 需要 --data")
        from miniocr.data.latex_images_ch32 import prepare_images
        emit(data=prepare_images(args.data,args.n))
    else:
        for p in args.parts or (1,2,3,6):
            random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
            title,fn=PARTS[p]; emit(部分=p,标题=title)
            begun=time.perf_counter(); fn(args); emit(部分=p,用时秒=time.perf_counter()-begun)
    emit(结束时间=datetime.now(timezone.utc).isoformat(),tour总用时秒=time.perf_counter()-started)


if __name__=="__main__": main()
