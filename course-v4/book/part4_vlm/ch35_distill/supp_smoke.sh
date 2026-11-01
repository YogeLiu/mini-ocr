#!/bin/bash
set -euo pipefail
root="${COURSE_WORK_ROOT:-/hy-tmp}"
cd "$root/course-v4"
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$root/runs/ch35/supp/mpl"
common=(--val-only --device cuda --dtype bf16 --model-cache "$root/runs/ch32/hf/hub" --data "$root/runs/ch35/supp/data")
python -u -m book.part4_vlm.ch35_distill.tour --prepare-data --val-only --n-train 5000 --n-teacher 5000 --n 500 --data "$root/runs/ch35/supp/data" --out "$root/runs/ch35/supp/prep"
python -u -m book.part4_vlm.ch35_distill.tour 2 "${common[@]}" --teacher-adapter "$root/runs/ch33/qwen3vl_main/adapter" --n 48 --batch-size 16 --scale 1 --max-pixels 65536 --max-new-tokens 192 --out "$root/runs/ch35/supp/teacher_smoke"
python -u -m book.part4_vlm.ch35_distill.tour 4 "${common[@]}" --n-train 5000 --n 32 --batch-size 8 --steps 30 --accum 1 --rank 64 --max-new-tokens 192 --audit-n 0 --out "$root/runs/ch35/supp/student_smoke"
