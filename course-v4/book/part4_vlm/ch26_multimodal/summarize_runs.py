"""只读取已完成的 ch26 GPU 产物，导出 CSV、token 图与成本汇总；不运行模型。"""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

from miniocr.train.pretrain import wilson_interval


def summarize_runs(root):
    """root 包含 smoke/、main/；缺完成标志或预测不齐时拒绝汇总。"""
    root = Path(root)
    runs = {name:json.loads((root/name/"run.json").read_text()) for name in ("smoke","main")}
    experiments = []
    token_rows = []
    for run_name, name in (("smoke","controlled"),("main","controlled"),("main","val")):
        report=json.loads((root/run_name/(name+".json")).read_text())
        rows=[json.loads(line) for line in (root/run_name/(name+".jsonl")).read_text().splitlines()]
        if report["status"]!="complete" or len(rows)!=report["extracted_metrics"]["n"]:
            raise ValueError("仅汇总完整实验")
        correct=sum(row["score"]["em_norm"] for row in rows)
        low,high=wilson_interval(correct,len(rows))
        result=dict(run=run_name,experiment=name,n=len(rows),
                    raw_em=report["raw_metrics"]["em_raw"],raw_normalized_em=report["raw_metrics"]["em_norm"],
                    extracted_normalized_correct=correct,extracted_normalized_em=correct/len(rows),
                    em_wilson_low=low,em_wilson_high=high,
                    extracted_cer=report["extracted_metrics"]["cer_char"],
                    generation_seconds=report["generation_seconds"],generated_tokens=report["generated_tokens"],
                    capped_count=report["capped_count"],peak_allocated_gib=report["peak_allocated_bytes"]/2**30,
                    peak_reserved_gib=report["peak_reserved_bytes"]/2**30,
                    visual_tokens_min=min(r["visual_tokens"] for r in rows),
                    visual_tokens_max=max(r["visual_tokens"] for r in rows),
                    visual_tokens_mean=sum(r["visual_tokens"] for r in rows)/len(rows))
        experiments.append(result)
        token_rows.extend(dict(run=run_name,experiment=name,id=row["id"],
                               **{k:row[k] for k in ("visual_tokens","input_tokens","generated_tokens","seconds","length_capped")})
                          for row in rows)
    for filename, rows in (("summary.csv",experiments),("tokens.csv",token_rows)):
        with (root/filename).open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    intervals=sorted((datetime.fromisoformat(r["start_utc"]),datetime.fromisoformat(r["end_utc"])) for r in runs.values())
    if intervals[0][1]>intervals[1][0]:
        raise ValueError("本章两次运行发生重叠，应另外计算墙钟并集")
    report=dict(runs=runs,experiments=experiments,
                gpu_task_wall_seconds=sum(r["seconds"] for r in runs.values()),
                generation_seconds=sum(r["generation_seconds"] for r in experiments),
                timing="tour wall time incl. CPU preparation, excluding download and inter-run gap; shared GPU")
    (root/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for name in ("controlled","val"):
        rows=[r for r in token_rows if r["run"]=="main" and r["experiment"]==name]
        ax.scatter([r["visual_tokens"] for r in rows],[r["generated_tokens"] for r in rows],
                   label=name,alpha=.65,s=35)
    ax.set_xlabel("Merged visual tokens"); ax.set_ylabel("Generated tokens (incl. control/EOS)")
    ax.set_title("Qwen3.5-0.8B, RTX 4090, bf16, greedy")
    ax.legend(); fig.tight_layout(); fig.savefig(root/"tokens.png"); plt.close(fig)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    args=parser.parse_args()
    print(json.dumps(summarize_runs(args.root),ensure_ascii=False,indent=2))
