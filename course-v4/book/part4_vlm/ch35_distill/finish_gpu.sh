#!/usr/bin/env bash
# 从已完成的 teacher/quality.json 接续；所有写入位于本章产物和日志目录。
set -euo pipefail
root="${COURSE_WORK_ROOT:-/hy-tmp}"
cd "$root/course-v4"
run_root="$root/runs/ch35"
model_cache="${CH35_MODEL_CACHE:-$root/runs/ch32/hf/hub}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export MPLCONFIGDIR="$run_root/mpl"
mkdir -p "$root/logs"
[[ -f "$run_root/teacher/quality.json" ]]
nvidia-smi
for arm in original distill; do
  [[ ! -e "$run_root/$arm/training.json" ]] || { echo "Refusing to overwrite $arm"; exit 1; }
  python -u -m book.part4_vlm.ch35_distill.tour 4 \
    --device cuda --dtype bf16 --n 64 --n-train 256 --steps 600 --accum 4 --rank 16 \
    --lr 0.0002 --seed 35 --max-new-tokens 192 --arm "$arm" \
    --model-cache "$model_cache" --data "$run_root/data" \
    --teacher-dir "$run_root/teacher" --out "$run_root/$arm" \
    > "$root/logs/ch35_${arm}.log" 2>&1
done
python -m book.part4_vlm.ch35_distill.summarize --root "$run_root" --select
chosen=$(python -c "import json; print(json.load(open(\"$root/runs/ch35/selection.json\"))[\"selected\"])")
python -u -m book.part4_vlm.ch35_distill.tour 5 \
  --device cuda --dtype bf16 --n 64 --max-new-tokens 192 --final-test \
  --adapter "$run_root/$chosen/adapter" --model-cache "$model_cache" \
  --data "$run_root/data" --out "$run_root/final" > "$root/logs/ch35_final.log" 2>&1
python -m book.part4_vlm.ch35_distill.summarize --root "$run_root"
