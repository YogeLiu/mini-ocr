#!/bin/sh
# Run after preparing data and inspecting the completed 20-image smoke log.
# The caller backgrounds this whole script with nohup. No package installation.
set -eu
root="${COURSE_WORK_ROOT:-/hy-tmp}"
cd "$root/course-v4"
export PYTHONDONTWRITEBYTECODE=1 HF_HOME="$root/runs/ch27/hf"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="$root/runs/ch27/mpl" XDG_CACHE_HOME="$root/runs/ch27/cache" TMPDIR="$root/runs/ch27/tmp"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p "$root/logs"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv > "$root/runs/ch27/gpu_main_before.csv"
python -u -m book.part4_vlm.ch27_image_tensors.tour 3 --device cuda --dtype bf16 --batch-size 8 --n 200 \
    --configs native nearest128 bilinear128 area128 letterbox128 letterbox384 \
    --data "$root/runs/ch27/data" --model-cache "$root/runs/ch27/hf/hub" \
    --out "$root/runs/ch27/rendered" > "$root/logs/ch27_rendered.log" 2>&1
python -u -m book.part4_vlm.ch27_image_tensors.tour 4 --device cuda --dtype bf16 --batch-size 8 --n 200 \
    --configs native nearest128 bilinear128 area128 letterbox128 letterbox384 \
    --data "$root/runs/ch27/data" --model-cache "$root/runs/ch27/hf/hub" \
    --out "$root/runs/ch27/original" > "$root/logs/ch27_original.log" 2>&1
python -u -m book.part4_vlm.ch27_image_tensors.tour --select-from "$root/runs/ch27/original" --test-n 100 \
    --data "$root/runs/ch27/data" --out "$root/runs/ch27/freeze" > "$root/logs/ch27_freeze.log" 2>&1
python -u -m book.part4_vlm.ch27_image_tensors.tour 5 --final-test --device cuda --n 100 \
    --data "$root/runs/ch27/data" --model-cache "$root/runs/ch27/hf/hub" \
    --out "$root/runs/ch27/final" > "$root/logs/ch27_final.log" 2>&1
python -m book.part4_vlm.ch27_image_tensors.summarize_runs "$root/runs/ch27" > "$root/logs/ch27_summary.log" 2>&1
date -Iseconds > "$root/runs/ch27/DONE"
