"""第 27 章实验：张量、缩放与零样本 OCR。默认只运行轻量部分 1/2。"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from importlib.metadata import version

import numpy as np
from PIL import Image, ImageOps
import torch

from miniocr.data.image_tensors import (image_to_chw, normalize, transform_image, pixel_metrics,
                                        fraction_bar_mask)
from miniocr.eval.render import render
from miniocr.eval.metrics import score_pair, summarize
from miniocr.eval.postcorrect_eval import extract_formula
from miniocr.models.vlm_anatomy import local_snapshot, load_image_only_processor, MODELS

PROMPT = "Transcribe the mathematical formula in this image into LaTeX. Output only the LaTeX formula, without explanation or markdown."
CONFIGS = {"nearest128": (128, "nearest", "square"), "bilinear128": (128, "bilinear", "square"),
           "area128": (128, "area", "square"), "letterbox128": (128, "bilinear", "letterbox"),
           "letterbox384": (384, "bilinear", "letterbox"), "native": None}


def emit(**row):
    print(json.dumps(row, ensure_ascii=False, default=str), flush=True)


def save(args, name, value):
    (args.out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str)+"\n")


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def part1(args):
    rng = np.random.default_rng(args.seed)
    hwc = rng.integers(0, 256, (args.height, args.width, 3), dtype=np.uint8)
    x = image_to_chw(Image.fromarray(hwc), args.device)
    z = normalize(x)
    recovered = normalize(z, inverse=True)
    result = dict(hwc_shape=list(hwc.shape), chw_shape=list(x.shape), chw_stride=list(x.stride()),
                  dtype=str(x.dtype), original_range=[int(hwc.min()), int(hwc.max())],
                  normalized_range=[float(z.min()), float(z.max())],
                  inverse_max_abs=float((recovered-x).abs().max()),
                  reshape_is_permute=bool(np.array_equal(hwc.reshape(3, args.height, args.width), hwc.transpose(2, 0, 1))),
                  training_steps=0)
    save(args, "tensor.json", result)
    emit(part=1, origin="本项目观察", **result)


def part2(args):
    import matplotlib.pyplot as plt
    formulas = [r"\frac{a_i+b^{2}}{c_j+d^{2}}", r"\frac{1+x^{n+1}}{1+x_i}",
                r"\frac{\alpha_{i}+\beta^{2}}{\gamma_j}"]
    rows = []
    fig, axes = plt.subplots(len(formulas), 1+len(args.sides)*4, figsize=(3+len(args.sides)*5, 7), squeeze=False)
    for i, formula in enumerate(formulas):
        original = ImageOps.expand(Image.fromarray(255-render(formula, dpi=args.dpi)).convert("RGB"), border=12, fill="white")
        mask = fraction_bar_mask(original)
        original.save(args.out/f"fraction_{i}_original.png")
        Image.fromarray((mask*255).astype(np.uint8)).save(args.out/f"fraction_{i}_mask.png")
        axes[i, 0].imshow(original); axes[i, 0].set_title("original")
        j = 1
        for side in args.sides:
            for method, geometry in [("nearest", "square"), ("bilinear", "square"), ("area", "square"), ("bilinear", "letterbox")]:
                processed, meta = transform_image(original, side, method, geometry, args.device)
                metrics = pixel_metrics(original, processed, meta, mask, args.device)
                row = dict(formula=formula, dpi=args.dpi, **meta, **metrics)
                rows.append(row)
                axes[i, j].imshow(processed); axes[i, j].set_title(f"{method}\n{geometry} {side}")
                j += 1
        for ax in axes[i]:
            ax.axis("off")
    fig.tight_layout(); fig.savefig(args.out/"resizing.png", dpi=140); plt.close(fig)
    save(args, "pixels.json", rows)
    emit(part=2, origin="本项目观察", comparisons=len(rows), fraction_line_recall_mean=float(np.mean([r["fraction_line_recall"] for r in rows])))


def prepare(args):
    from miniocr.data.image_tensor_corpus import prepare_corpus
    if args.data_cache:
        os.environ["HF_HUB_CACHE"] = str(args.data_cache)
    if args.data is None:
        raise ValueError("--prepare-data requires --data outside source tree on rented GPU")
    meta = prepare_corpus(args.data, args.n, args.test_n, args.dpi)
    emit(prepared=str(args.data), n=meta["n"], test_n=meta["test_n"], render_attempts=meta["render_attempts"], render_failures=meta["render_failures"])


class Inference:
    """离线、冻结参数、左侧补齐批推理；记录每图真实视觉占位符数量。"""
    def __init__(self, args, key):
        from transformers import AutoModelForImageTextToText
        self.args, self.key = args, key
        path = local_snapshot(key, args.model_cache)
        if path is None:
            raise FileNotFoundError(f"missing local {MODELS[key]}; no automatic downloads")
        self.processor = load_image_only_processor(path, key)
        self.processor.tokenizer.padding_side = "left"
        if key == "qwen35":
            self.processor.image_processor.size = dict(shortest_edge=1024, longest_edge=args.max_pixels)
        else:
            # Keep the pretrained 512-pixel crop and 64-token layout. Changing only
            # max_image_size would disagree with the tokenizer's image_seq_len.
            self.processor.image_processor.do_image_splitting = False
            self.processor.image_processor.size = dict(longest_edge=512)
        dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
        self.model = AutoModelForImageTextToText.from_pretrained(path, local_files_only=True, dtype=dtype,
                                                               attn_implementation="sdpa").to(args.device).eval()
        self.model.requires_grad_(False)
        self.meta = dict(model=key, model_id=MODELS[key], revision=path.name,
                         parameters=sum(p.numel() for p in self.model.parameters()),
                         trainable_parameters=sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                         tokenizer_pad_token_id=self.processor.tokenizer.pad_token_id,
                         generation_pad_token_id=self.model.generation_config.pad_token_id,
                         embedding_rows=self.model.get_input_embeddings().num_embeddings,
                         processor=self.processor.image_processor.to_dict())
        emit(**self.meta)

    def predict(self, image):
        return self.predict_batch([image])[0]

    def predict_batch(self, images):
        """每图一个提示词；attention_mask 排除左 padding，输出裁到首个 EOS。"""
        p, args = self.processor, self.args
        messages = [{"role":"user", "content":[{"type":"image"}, {"type":"text", "text":PROMPT}]}]
        prompt = p.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        sync(args.device); start = time.perf_counter()
        inputs = p(text=[prompt]*len(images), images=[[im] for im in images], return_tensors="pt", padding=True)
        inputs = {k:v.to(device=args.device, dtype=self.model.dtype if v.is_floating_point() else v.dtype) for k, v in inputs.items()}
        mask = inputs["attention_mask"]
        assert bool((mask[:, -1] == 1).all()) and bool((mask[:, 1:] >= mask[:, :-1]).all()), "decoder requires left padding"
        assert int(inputs["input_ids"].max()) < self.model.get_input_embeddings().num_embeddings
        image_id = getattr(p, "image_token_id", self.model.config.image_token_id)
        counts = (inputs["input_ids"] == image_id).sum(1).tolist()
        grids = inputs["image_grid_thw"].tolist() if "image_grid_thw" in inputs else [None]*len(images)
        for count, grid in zip(counts, grids):
            if grid is not None:
                assert count == int(np.prod(grid))//p.image_processor.merge_size**2
            if count <= 0:
                raise AssertionError("no image tokens in actual input")
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
        generated = output[:, inputs["input_ids"].shape[1]:].tolist()
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else (eos or [])
        sync(args.device)
        elapsed = time.perf_counter()-start
        results = []
        for i, tokens in enumerate(generated):
            tokens = trim_after_eos(tokens, eos)
            raw = p.decode(tokens, skip_special_tokens=True)
            results.append(dict(raw=raw, pred=extract_formula(raw), visual_tokens=counts[i], grid=grids[i],
                                batch_pixel_values_shape=list(inputs["pixel_values"].shape),
                                input_tokens=int(inputs["attention_mask"][i].sum()), padded_input_tokens=int(inputs["input_ids"].shape[1]),
                                generated_tokens=len(tokens), capped=len(tokens)>=args.max_new_tokens and tokens[-1] not in eos,
                                seconds=elapsed/len(images), batch_seconds=elapsed, batch_size=len(images),
                                formatted_prompt=prompt))
        return results


def trim_after_eos(tokens, eos):
    """保留首个 EOS，丢弃批生成补在其后的 pad；无 EOS 时保留全部。"""
    for i, token in enumerate(tokens):
        if token in eos:
            return tokens[:i+1]
    return tokens


def evaluate(args, source, split="val", frozen=None):
    if args.data is None:
        raise ValueError("inference requires --data; prepare with --prepare-data")
    manifest = json.loads((args.data/"manifest.json").read_text())
    rows = manifest[f"{split}_{source}"][:args.n]
    if len(rows) != args.n:
        raise ValueError("prepared data has fewer samples than --n")
    digest = hashlib.sha256(json.dumps([r["id"] for r in rows]).encode()).hexdigest()
    for key in args.models:
        if local_snapshot(key, args.model_cache) is None:
            emit(status="skip", model=key, reason="missing local weights")
            save(args, f"{key}_{split}_{source}_skip.json", dict(status="skip", reason="missing local weights"))
            continue
        runner = Inference(args, key)
        for config in args.configs:
            name = f"{key}_{split}_{source}_{config}"
            if split == "test":
                # One lock in the data directory survives --out changes. All settings
                # are validated against a val-only freeze before acquiring this lock.
                with (args.data/"FINAL_TEST.lock").open("x") as handle:
                    handle.write(json.dumps(frozen, default=str))
            observations, scores, raw_scores = [], [], []
            sync(args.device)
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with (args.out/(name+".jsonl")).open("w") as handle:
                for offset in range(0, len(rows), args.batch_size):
                    batch = rows[offset:offset+args.batch_size]
                    images, metadata, differences = [], [], []
                    for row in batch:
                        with Image.open(args.data/row[source]) as im:
                            image = im.convert("RGB")
                        meta = pixels = None
                        if CONFIGS[config] is not None:
                            side, method, geometry = CONFIGS[config]
                            original = image
                            image, meta = transform_image(image, side, method, geometry, args.device)
                            pixels = pixel_metrics(original, image, meta, device=args.device)
                        images.append(image); metadata.append(meta); differences.append(pixels)
                    predictions = runner.predict_batch(images)
                    for row, image, meta, pixels, prediction in zip(batch, images, metadata, differences, predictions):
                        score = score_pair(row["formula"], prediction["pred"])
                        raw_score = score_pair(row["formula"], prediction["raw"])
                        scores.append(score); raw_scores.append(raw_score)
                        obs = dict(id=row["id"], config=config, transform=meta, pixels=pixels,
                                   input_image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                                   input_hw=[image.height, image.width], **prediction)
                        if split == "val":
                            obs.update(ref=row["formula"], score=score, raw_score=raw_score)
                        else:
                            obs = {k:v for k, v in obs.items() if k not in ("raw", "pred", "formatted_prompt")}
                        handle.write(json.dumps(obs, ensure_ascii=False)+"\n"); handle.flush()
                        observations.append(obs)
                    emit(experiment=name, completed=offset+len(batch), n=len(rows), seconds=time.perf_counter()-started)
            result = dict(status="complete", origin="本项目观察", **runner.meta, split=split, source=source,
                          config=config, sample_ids_sha256=digest, metrics=summarize(scores), raw_metrics=summarize(raw_scores),
                          dtype=args.dtype, device=args.device, training_steps=0, batch_size=args.batch_size, seed=args.seed,
                          prompt=PROMPT, max_new_tokens=args.max_new_tokens, max_pixels=args.max_pixels,
                          do_sample=False, enable_thinking=False,
                          seconds=time.perf_counter()-started, generation_seconds=sum(o["seconds"] for o in observations),
                          mean_visual_tokens=float(np.mean([o["visual_tokens"] for o in observations])),
                          visual_tokens_range=[min(o["visual_tokens"] for o in observations), max(o["visual_tokens"] for o in observations)],
                          generated_tokens=sum(o["generated_tokens"] for o in observations), capped=sum(o["capped"] for o in observations),
                          peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if args.device=="cuda" else None,
                          peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if args.device=="cuda" else None)
            if observations[0]["pixels"] is not None:
                result["pixel_metrics_mean"] = {k:float(np.mean([o["pixels"][k] for o in observations]))
                                                for k in observations[0]["pixels"]}
            if split == "test":
                result["confidence_intervals"] = bootstrap_intervals(scores, args.seed)
                result["frozen"] = frozen
            save(args, name+".json", result)
            emit(experiment=name, metrics=result["metrics"], seconds=result["seconds"], visual_tokens=result["mean_visual_tokens"], capped=result["capped"])
        del runner
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()


def bootstrap_intervals(scores, seed, repeats=2000):
    """按公式重采样的 95% percentile 区间，CER 始终用总编辑数/总字符数。"""
    rng = np.random.default_rng(seed)
    em = np.array([s["em_norm"] for s in scores], dtype=float)
    ed = np.array([s["ed_char"] for s in scores])
    length = np.array([s["len_char"] for s in scores])
    idx = rng.integers(0, len(scores), (repeats, len(scores)))
    return dict(em_norm=np.quantile(em[idx].mean(1), [.025, .975]).tolist(),
                cer_char=np.quantile(ed[idx].sum(1)/np.maximum(1, length[idx].sum(1)), [.025, .975]).tolist(),
                method="paired formula bootstrap, percentile 95%", repeats=repeats, seed=seed)


def freeze(args):
    """仅看原图 val，按 EM 降序、CER 升序、视觉 token 升序冻结单一模型和设置。"""
    candidates = []
    for path in args.select_from.glob("*_val_original_*.json"):
        row = json.loads(path.read_text())
        if row.get("status") == "complete":
            candidates.append((row, path))
    if not candidates:
        raise ValueError("no completed original-image val results")
    if len({(r["sample_ids_sha256"], r["metrics"]["n"]) for r, _ in candidates}) != 1:
        raise ValueError("selection candidates must use identical val samples")
    winner, path = min(candidates, key=lambda pair: (-pair[0]["metrics"]["em_norm"], pair[0]["metrics"]["cer_char"], pair[0]["mean_visual_tokens"], pair[0]["model"], pair[0]["config"]))
    result = {k:winner[k] for k in ("model", "revision", "config", "dtype", "batch_size", "max_new_tokens", "max_pixels", "seed", "prompt", "sample_ids_sha256")}
    result.update(selection="val original only: max EM, min CER, min visual tokens, lexical tie break", val_result=str(path),
                  val_result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), test_n=args.test_n)
    with (args.data/"frozen.json").open("x") as handle:
        json.dump(result, handle, indent=2)
    emit(frozen=result)


def part5(args):
    frozen = json.loads((args.data/"frozen.json").read_text())
    if args.n != frozen["test_n"]:
        raise ValueError("test n must match frozen test_n")
    args.models, args.configs = [frozen["model"]], [frozen["config"]]
    for key in ("dtype", "batch_size", "max_new_tokens", "max_pixels", "seed"):
        setattr(args, key, frozen[key])
    path = local_snapshot(frozen["model"], args.model_cache)
    if path is None or path.name != frozen["revision"] or PROMPT != frozen["prompt"]:
        raise ValueError("frozen model revision/prompt mismatch")
    torch.manual_seed(args.seed)
    evaluate(args, "original", "test", frozen)


PARTS = {1:part1, 2:part2, 3:lambda a:evaluate(a, "rendered"), 4:lambda a:evaluate(a, "original"), 5:part5}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", type=int, nargs="*")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--out", type=Path, default=Path("outputs/book/ch27/local"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--data-cache", type=Path)
    parser.add_argument("--model-cache", type=Path)
    parser.add_argument("--models", choices=("qwen35", "smol"), nargs="+", default=["qwen35", "smol"])
    parser.add_argument("--configs", choices=tuple(CONFIGS), nargs="+", default=list(CONFIGS)[:-1])
    parser.add_argument("--sides", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--select-from", type=Path, help="freeze winner from original val summaries; no test evaluation")
    parser.add_argument("--final-test", action="store_true")
    for name, default in [("batch-size",1), ("n",20), ("test-n",100), ("max-new-tokens",128), ("max-pixels",262144),
                          ("dpi",200), ("seed",0), ("threads",2), ("height",48), ("width",96)]:
        parser.add_argument("--"+name, type=int, default=default)
    args = parser.parse_args(argv)
    if any(p not in PARTS for p in args.parts):
        parser.error("parts must be 1–5")
    if min(args.batch_size, args.n, args.test_n, args.max_new_tokens, args.dpi, args.threads, args.height, args.width, *args.sides) < 1 or args.max_pixels < 1024:
        parser.error("positive sizes required; max-pixels >= 1024")
    if 5 in args.parts and (not args.final_test or args.parts != [5]):
        parser.error("test requires part 5 alone and --final-test after val freeze")
    if args.device == "cpu" and args.dtype != "fp32":
        parser.error("CPU teaching runs use fp32")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    meta = dict(started=datetime.now(timezone.utc).isoformat(), config=vars(args), python=platform.python_version(),
                torch=torch.__version__, device_name=torch.cuda.get_device_name() if args.device=="cuda" else args.device,
                packages={name:version(name) for name in ("numpy", "Pillow", "matplotlib")},
                tour_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                training_steps=0, test_evaluated=5 in args.parts)
    save(args, "run.json", meta)
    begun = time.perf_counter()
    if args.prepare_data:
        prepare(args)
    if args.select_from:
        freeze(args)
    for number in args.parts or ([] if args.prepare_data or args.select_from else [1, 2]):
        emit(part=number, started=datetime.now(timezone.utc).isoformat())
        PARTS[number](args)
    sync(args.device)
    meta.update(finished=datetime.now(timezone.utc).isoformat(), seconds=time.perf_counter()-begun, status="complete")
    save(args, "run.json", meta)
    emit(completed=True, seconds=meta["seconds"])


if __name__ == "__main__":
    main()
