"""第 26 章：多模态问题的可观察证据。默认只运行轻量部分 1/2；3/4 显式推理。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
from PIL import Image, ImageOps
import torch

from miniocr.eval.metrics import score_pair, summarize
from miniocr.eval.postcorrect_eval import extract_formula
from miniocr.eval.render import render, compare_render
from miniocr.models.multimodal import VisualPrefix, CrossAttention, prefix_mask
from miniocr.models.vlm_anatomy import local_snapshot, load_image_only_processor

FORMULAS = (r"x^2+y^2=z^2", r"\frac{a+b}{\sqrt{x^2+1}}", r"\sum_{i=1}^{n} i = \frac{n(n+1)}{2}")
PROMPT = "Transcribe the mathematical formula in this image into LaTeX. Output only the LaTeX formula, without explanation or markdown."


def emit(**values):
    print(json.dumps(values, ensure_ascii=False, default=str), flush=True)


def save(args, name, value):
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str)+"\n", encoding="utf-8")


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def rendered_examples(args):
    """三张受控示例固定内容/DPI/边框；render 的墨迹值必须反转为黑字白底。"""
    folder = args.out/"images"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, formula in enumerate(FORMULAS, 1):
        image = ImageOps.expand(Image.fromarray(255-render(formula, dpi=args.dpi)).convert("RGB"),
                                border=16, fill="white")
        path = folder/f"formula_{i}.png"
        image.save(path)
        rows.append(dict(id=f"controlled-{i}", formula=formula, image=str(path.relative_to(args.out)),
                         width=image.width, height=image.height,
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest(), split="controlled-demo"))
    return rows


def part1(args):
    rows = rendered_examples(args)
    save(args, "rendered.json", dict(source="mathtext controlled examples", dpi=args.dpi, rows=rows))
    for row in rows:
        emit(origin="本项目观察", part=1, **row)


def part2(args):
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    vision = torch.randn(args.batch_size, args.visual_tokens, args.vision_dim,
                         device=args.device, dtype=dtype, requires_grad=True)
    text = torch.randn(args.batch_size, args.text_tokens, args.text_dim,
                       device=args.device, dtype=dtype, requires_grad=True)
    prefix = VisualPrefix(args.vision_dim, args.text_dim).to(args.device, dtype=dtype)
    cross = CrossAttention(args.vision_dim, args.text_dim).to(args.device, dtype=dtype)
    combined = prefix(vision, text)
    mask = prefix_mask(args.visual_tokens, args.text_tokens, args.device)
    output, weights = cross(vision, text)
    loss = combined.float().square().mean() + output.float().square().mean()
    loss.backward()
    gradients_ok = all(p.grad is not None and torch.isfinite(p.grad).all().item()
                       and p.grad.abs().sum().item() > 0 for p in [vision, text, *prefix.parameters(), *cross.parameters()])
    if not gradients_ok:
        raise AssertionError("连接器应有有限且非零的梯度")
    result = dict(vision=list(vision.shape), text=list(text.shape), prefix=list(combined.shape),
                  prefix_mask=list(mask.shape), query=list(text.shape),
                  key_value=[args.batch_size,args.visual_tokens,args.text_dim],
                  cross_weights=list(weights.shape), cross_output=list(output.shape),
                  weight_row_sum_max_error=(weights.float().sum(-1)-1).abs().max().item(),
                  gradients_finite_nonzero=gradients_ok, diagnostic_loss=loss.item(),
                  prefix_parameters=sum(p.numel() for p in prefix.parameters()),
                  cross_parameters=sum(p.numel() for p in cross.parameters()), training_steps=0)
    save(args,"shapes.json",result)
    emit(origin="本项目观察；随机张量，非 OCR 成绩", part=2, **result)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(9,3))
    axes[0].imshow(mask.cpu().numpy(), cmap="gray", vmin=0,vmax=1)
    axes[0].set_title("Prefix mask: white = visible")
    axes[0].set_xlabel("Key (vision then text)"); axes[0].set_ylabel("Query")
    axes[1].imshow(weights[0].detach().float().cpu().numpy(), aspect="auto")
    axes[1].set_title("Random cross-attention, NOT alignment")
    axes[1].set_xlabel("Visual key"); axes[1].set_ylabel("Text query")
    fig.tight_layout(); fig.savefig(args.out/"shapes.png"); plt.close(fig)


class Inference:
    """按需离线加载，单图生成；不把参考公式传给模型。"""
    def __init__(self, args):
        self.args = args
        self.model = self.processor = None
        self.metadata = None

    def load(self):
        if self.model is not None:
            return True
        path = local_snapshot("qwen35", self.args.model_cache)
        if path is None:
            emit(origin="待验证", status="skip", reason="缺少本地 Qwen3.5-0.8B 权重；不会自动下载")
            return False
        from transformers import AutoModelForImageTextToText
        start = time.perf_counter()
        self.processor = load_image_only_processor(path, "qwen35")
        self.processor.image_processor.size = dict(shortest_edge=self.args.min_pixels, longest_edge=self.args.max_pixels)
        dtype = torch.float32 if self.args.dtype == "fp32" else torch.bfloat16
        self.model = AutoModelForImageTextToText.from_pretrained(
            path, local_files_only=True, dtype=dtype, attn_implementation="sdpa").to(self.args.device).eval()
        self.model.requires_grad_(False)
        sync(self.args.device)
        self.metadata = dict(model="Qwen/Qwen3.5-0.8B", snapshot=str(path), revision=path.name,
                             load_seconds=time.perf_counter()-start,
                             parameters=sum(p.numel() for p in self.model.parameters()),
                             trainable_parameters=sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                             processor=type(self.processor.image_processor).__name__)
        emit(origin="本项目观察", **self.metadata)
        return True

    def predict(self, image):
        args, processor = self.args, self.processor
        messages = [{"role":"user", "content":[{"type":"image"},{"type":"text","text":PROMPT}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        sync(args.device)
        started = time.perf_counter()
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k:v.to(device=args.device, dtype=self.model.dtype if v.is_floating_point() else v.dtype)
                  for k,v in inputs.items()}
        grid = inputs["image_grid_thw"].tolist()
        visual_tokens = int(inputs["image_grid_thw"].prod().item()) // processor.image_processor.merge_size**2
        image_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        expanded = int((inputs["input_ids"] == image_id).sum().item())
        if visual_tokens != expanded:
            raise AssertionError("grid 合并计数应等于输入中的视觉占位 token 数")
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                          do_sample=False, use_cache=True)
        generated = outputs[0,inputs["input_ids"].shape[1]:].tolist()
        raw = processor.tokenizer.decode(generated, skip_special_tokens=True)
        sync(args.device)
        seconds = time.perf_counter()-started
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos,int) else (eos or [])
        return dict(raw=raw, extracted=extract_formula(raw), prompt=prompt,
                    input_tokens=inputs["input_ids"].shape[1], visual_tokens=visual_tokens,
                    non_image_input_tokens=inputs["input_ids"].shape[1]-visual_tokens,
                    generated_tokens=len(generated), generated_ids=generated, grid=grid,
                    pixel_values_shape=list(inputs["pixel_values"].shape), seconds=seconds,
                    length_capped=len(generated)>=args.max_new_tokens and generated[-1] not in eos)


def infer_rows(args, inference, rows, name):
    if not inference.load():
        save(args, name+".json", dict(status="skipped", reason="missing local weights"))
        return
    results = []
    # 逐条刷盘；中断时保留已完成输出，完整汇总仅在全部完成后写入。
    with (args.out/(name+".jsonl")).open("w",encoding="utf-8") as handle:
        for row in rows:
            with Image.open(args.out/row["image"]) as image:
                prediction = inference.predict(image.convert("RGB"))
            result = dict(**row, **prediction)
            result["raw_score"] = score_pair(row["formula"], result["raw"])
            result["score"] = score_pair(row["formula"], result["extracted"])
            if name == "controlled":
                result["render_score"] = compare_render(row["formula"], result["extracted"])
            handle.write(json.dumps(result, ensure_ascii=False)+"\n"); handle.flush()
            results.append(result)
            emit(origin="本项目观察", experiment=name, **result)
    report = dict(status="complete", model=inference.metadata, config=vars(args), training_steps=0,
                  raw_metrics=summarize([r["raw_score"] for r in results]),
                  extracted_metrics=summarize([r["score"] for r in results]),
                  generation_seconds=sum(r["seconds"] for r in results),
                  visual_tokens=[r["visual_tokens"] for r in results],
                  generated_tokens=sum(r["generated_tokens"] for r in results),
                  capped_count=sum(r["length_capped"] for r in results),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device=="cuda" else None,
                  peak_reserved_bytes=torch.cuda.max_memory_reserved() if args.device=="cuda" else None)
    save(args,name+".json",report)
    emit(origin="本项目观察", experiment=name, summary=report)


def part3(args, inference):
    infer_rows(args, inference, rendered_examples(args)[:args.n], "controlled")


def part4(args, inference):
    from miniocr.data.formula_images import load_formula_images
    if args.data_cache:
        os.environ["HF_HUB_CACHE"] = str(args.data_cache)
    samples = load_formula_images(args.n, split="val")
    if not samples:
        emit(origin="待验证", part=4, status="skip", reason="无 cleaned_formulas 本地缓存")
        save(args,"val.json",dict(status="skipped",reason="missing local corpus"))
        return
    rows = []
    folder=args.out/"val_images"; folder.mkdir(parents=True,exist_ok=True)
    for i,sample in enumerate(samples):
        image = Image.fromarray(np.rint((1-sample.pop("ink"))*255).astype(np.uint8)).convert("RGB")
        path=folder/f"{i:04d}.png"; image.save(path)
        rows.append(dict(**sample, image=str(path.relative_to(args.out)), width=image.width, height=image.height,
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    save(args,"val_manifest.json",dict(source="OleehyO/latex-formulas", config="cleaned_formulas",
         license="openrail", protocol="latex-v1", split="val", seed=0, n=len(rows),
         selection="first distinct normalized keys in parquet order after full corpus split",
         preprocessing="PNG -> alpha composite white -> grayscale -> RGB; original dimensions",rows=rows))
    infer_rows(args,inference,rows,"val")


PARTS={1:("三张受控公式图",part1),2:("视觉前缀与交叉注意力",part2),
       3:("Qwen3.5 三图零样本",part3),4:("开源原图 val 验证",part4)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts",type=int,nargs="*")
    parser.add_argument("--device",choices=("cpu","mps","cuda"),default="cpu")
    parser.add_argument("--dtype",choices=("fp32","bf16"),default="fp32")
    parser.add_argument("--n",type=int,default=20,help="真实推理上限；部分 3 最多三张，部分 4 为 val 样本数")
    parser.add_argument("--out",type=Path,default=Path("outputs/book/ch26/local"))
    parser.add_argument("--model-cache",type=Path)
    parser.add_argument("--data-cache",type=Path)
    for name,default in (("seed",0),("dpi",160),("batch-size",2),("visual-tokens",6),("text-tokens",5),
                         ("vision-dim",8),("text-dim",12),("min-pixels",65536),("max-pixels",262144),
                         ("max-new-tokens",256)):
        parser.add_argument("--"+name,type=int,default=default)
    args=parser.parse_args(argv)
    for name in ("n","dpi","batch_size","visual_tokens","text_tokens","vision_dim","text_dim","min_pixels","max_pixels","max_new_tokens"):
        if getattr(args,name)<1: parser.error(name+" 必须为正")
    if args.max_pixels<args.min_pixels: parser.error("max-pixels 必须 >= min-pixels")
    if any(p not in PARTS for p in args.parts): parser.error("部分号为 1–4")
    if args.device=="cuda" and not torch.cuda.is_available(): parser.error("CUDA 不可用")
    if args.device=="mps" and not torch.backends.mps.is_available(): parser.error("MPS 不可用")
    if args.dtype=="bf16" and args.device=="mps": parser.error("MPS 路径请用 fp32")
    os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    torch.set_num_threads(2)
    args.out.mkdir(parents=True,exist_ok=True)
    import transformers
    start=time.perf_counter()
    metadata=dict(start_utc=datetime.now(timezone.utc).isoformat(), config=vars(args), python=platform.python_version(),
                  torch=torch.__version__,transformers=transformers.__version__,
                  device_name=torch.cuda.get_device_name() if args.device=="cuda" else args.device,
                  origin="本项目观察", training_steps=0, test_evaluated=False, parts=args.parts or [1,2])
    if args.device=="cuda": torch.cuda.reset_peak_memory_stats()
    emit(**metadata)
    inference=Inference(args)
    for number in args.parts or [1,2]:
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        title,fn=PARTS[number]; emit(part=number,title=title)
        sync(args.device); begun=time.perf_counter()
        if number in (3,4): fn(args,inference)
        else: fn(args)
        sync(args.device); emit(part=number,seconds=time.perf_counter()-begun)
    metadata.update(end_utc=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-start)
    save(args,"run.json",metadata); emit(**metadata)


if __name__=="__main__":
    main()
