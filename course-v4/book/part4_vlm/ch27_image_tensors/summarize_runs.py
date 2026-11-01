"""把已跑完的第 27 章日志整理成 CSV、图与耗时证据；不生成或猜测实验结果。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from miniocr.eval.metrics import score_pair, summarize as summarize_scores


def audit(root):
    """复算完整 val 分数与同图哈希；test 仅检查数量和未泄露逐条预测。"""
    compared, checked = {}, []
    for folder in ("rendered", "original", "final"):
        for path in sorted((root/folder).glob("*.json")):
            result = json.loads(path.read_text())
            if "metrics" not in result or result.get("status") != "complete":
                continue
            lines = [json.loads(line) for line in path.with_suffix(".jsonl").read_text().splitlines()]
            ids = [r["id"] for r in lines]
            assert len(lines) == result["metrics"]["n"] == len(set(ids))
            assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == result["sample_ids_sha256"]
            assert sum(r["capped"] for r in lines) == result["capped"]
            if result["split"] == "val":
                recalculated = summarize_scores([score_pair(r["ref"], r["pred"]) for r in lines])
                assert recalculated == result["metrics"], path
                key = (result["source"], result["config"])
                fingerprints = [(r["id"], r["input_image_sha256"]) for r in lines]
                if key in compared:
                    assert compared[key] == fingerprints, "models must receive identical images"
                compared[key] = fingerprints
            else:
                assert not any(set(r) & {"pred", "raw", "ref", "score", "raw_score"} for r in lines)
            checked.append(dict(file=str(path.relative_to(root)), n=len(lines)))
    manifest = json.loads((root/"data/manifest.json").read_text())
    val = {r["key_sha256"] for name in ("val_original", "val_rendered") for r in manifest[name]}
    test = {r["key_sha256"] for r in manifest["test_original"]}
    assert not val & test
    report = dict(checked=checked, val_test_key_overlap=0, identical_image_comparisons=len(compared),
                  test_predictions_disclosed=False, status="passed")
    (root/"audit.json").write_text(json.dumps(report, indent=2)+"\n")


def summarize(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(root)
    audit(root)
    rows = []
    for folder in ("rendered", "original", "final"):
        for path in sorted((root/folder).glob("*.json")):
            result = json.loads(path.read_text())
            if result.get("status") != "complete" or "metrics" not in result:
                continue
            row = {k:result[k] for k in ("model", "source", "split", "config", "dtype", "training_steps", "batch_size", "seconds",
                                        "mean_visual_tokens", "capped", "peak_allocated_gib", "peak_reserved_gib")}
            row.update(result["metrics"])
            row["summary_path"] = str(path.relative_to(root))
            row["sample_ids_sha256"] = result["sample_ids_sha256"]
            row.update({"pixel_"+k:v for k, v in result.get("pixel_metrics_mean", {}).items()})
            rows.append(row)
    if rows:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with (root/"metrics.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader(); writer.writerows(rows)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, source in enumerate(("rendered", "original")):
        for model in ("qwen35", "smol"):
            selected = [r for r in rows if r["source"] == source and r["model"] == model and r["split"] == "val"]
            if not selected:
                continue
            names = [r["config"] for r in selected]
            axes[0, col].plot(names, [r["em_norm"] for r in selected], marker="o", label=model)
            axes[1, col].plot(names, [r["cer_char"] for r in selected], marker="o", label=model)
        for i, label in enumerate(("Normalized EM", "Corpus CER")):
            axes[i, col].set_title(f"{source} val: {label}")
            axes[i, col].tick_params(axis="x", labelrotation=25)
            axes[i, col].legend(); axes[i, col].grid(alpha=.2)
    fig.tight_layout(); fig.savefig(root/"recognition.png", dpi=160); plt.close(fig)
    timings = []
    for folder in ("mechanisms", "smoke", "smoke_batch", "smoke_native", "rendered", "original", "freeze", "final"):
        path = root/folder/"run.json"
        if path.exists():
            run = json.loads(path.read_text())
            timings.append(dict(stage=folder, status=run.get("status"), started=run["started"],
                                finished=run.get("finished"), seconds=run.get("seconds"),
                                device=run["config"]["device"]))
    result = dict(runs=timings, task_wall_seconds=sum(t["seconds"] or 0 for t in timings if t["device"] == "cuda"),
                  note="sum of CUDA-command wall times, includes CPU prep and concurrent-lane contention; not exclusive GPU kernel time")
    (root/"timing.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(completed_summaries=len(rows), **result), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
