"""离线汇总已完成的第 32 章结果；不生成预测、不调用模型、不读取 test 错例。"""
import argparse
import csv
import json
from pathlib import Path
from miniocr.data.postcorrect import PostCorrectExample
from miniocr.eval.postcorrect_eval import score_postcorrect


def summarize(root):
    """按同一 ID 队列配对原图/渲染图，并导出指标 CSV 与观察图。"""
    root=Path(root)
    rows=[]
    paired=[]
    originals=[]
    cohorts={}
    for path in sorted(root.glob("*/*_val_*.json")):
        if path.parent.name not in ("main","rendered"): continue
        result=json.loads(path.read_text())
        m=result["metrics"]
        signature=(m["n"],result["sample_ids_sha256"])
        source=result["source"]
        if source in cohorts and cohorts[source]!=signature:
            raise ValueError(f"{source} 的模型结果不是同一批样本")
        cohorts[source]=signature
        rows.append(dict(model=result["model"],source=result["source"],n=m["n"],
                         em_norm=m["em_norm"]["value"],cer=m["cer"]["value"],
                         input_tokens=result["mean_input_tokens"],visual_tokens=result["mean_visual_tokens"],
                         peak_allocated_gib=result["peak_allocated_gib"],peak_reserved_gib=result["peak_reserved_gib"],
                         seconds=result["elapsed_seconds"],
                         tokens_per_second=result["generated_tokens"]/result["generate_seconds"],
                         truncated=result["truncated"],path=str(path.relative_to(root))))
        if result["source"]=="original": originals.append(result)
        if result["source"]=="rendered":
            other=json.loads((root/"main"/f'{result["model"]}_val_original.json').read_text())
            lookup={r["id"]:r for r in other["observations"]}
            ids=[r["id"] for r in result["observations"]]
            selected=[lookup[i] for i in ids] # fail loudly if cohorts do not nest
            examples=[PostCorrectExample(r["id"],"val",r["target"],r["target"]) for r in selected]
            scores=score_postcorrect(examples,[r["pred"] for r in selected])["summary"]
            paired.append(dict(model=result["model"],n=len(ids),ids=ids,
                               original={k:scores[k] for k in ("em_raw","em_norm","cer")},
                               rendered={k:m[k] for k in ("em_raw","em_norm","cer")},
                               original_input_tokens=sum(r["input_tokens"] for r in selected)/len(ids),
                               rendered_input_tokens=result["mean_input_tokens"],
                               original_generate_seconds=sum(r["generate_seconds"] for r in selected),
                               rendered_generate_seconds=result["generate_seconds"]))
    if not rows: raise ValueError("没有完成的 main/rendered 结果")
    with (root/"metrics.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (root/"paired.json").write_text(json.dumps(paired,indent=2,ensure_ascii=False))
    if len(originals)==3:
        winner=max(originals,key=lambda r:(r["metrics"]["em_norm"]["value"],-r["metrics"]["cer"]["value"]))
        selection={k:winner[k] for k in ("model","model_id","snapshot","seed","dtype","batch_size","prompt",
                                         "max_new_tokens","min_pixels","max_pixels","attn_implementation")}
        selection.update(criterion="highest val original EM_norm, tie-break lower CER",val_n=winner["metrics"]["n"],
                         val_sample_ids_sha256=winner["sample_ids_sha256"],test_n=100)
        (root/"val_selection.json").write_text(json.dumps(selection,indent=2,ensure_ascii=False))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for source in ("original","rendered"):
        subset=[r for r in rows if r["source"]==source]
        if not subset: continue
        for ax,key in zip(axes,("em_norm","input_tokens","peak_allocated_gib")):
            ax.plot([r["model"] for r in subset],[r[key] for r in subset],marker="o",label=source)
            ax.set_title(key)
    cohorts = {r["source"]: r["n"] for r in rows}
    axes[0].legend(); fig.suptitle(f"RTX 4090 BF16; cohorts {cohorts}")
    fig.tight_layout(); fig.savefig(root/"comparison.png"); plt.close(fig)
    return rows


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    args=parser.parse_args()
    print(json.dumps(summarize(args.root),indent=2))
