#!/usr/bin/env bash
# 在已准备好的租用实例上，从 $COURSE_WORK_ROOT/course-v4（默认 /hy-tmp/course-v4） 运行；不会安装、下载或关机。
# 主实验及故障对照仅看 val；最终 test 另行显式执行。
set -euo pipefail
stage="${1:-all}"
case "$stage" in all|remaining|faults) ;; *) echo 'usage: run_gpu.sh [all|remaining|faults]'; exit 2 ;; esac
work_root="${COURSE_WORK_ROOT:-/hy-tmp}"
cd "$work_root/course-v4"
root=outputs/book/ch19/gpu
mkdir -p "$root" "$work_root/logs"
run() {
  local name="$1"
  shift
  local required_mib=10240
  case "$name" in main) required_mib=32768 ;; fault_*) required_mib=20480 ;; esac
  # 共享实例按可用显存排队；并发状态保存在 GPU monitor，计时不当独占性能。
  while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt "$required_mib" ]; do
    echo "${name}: 等待至少 ${required_mib} MiB 可用显存"
    sleep 20
  done
  date -u +%FT%TZ > "$root/${name}.started"
  python -u -m book.part3_post_training.ch19_sft.tour "$@" > "$work_root/logs/ch19_${name}.log" 2>&1
  date -u +%FT%TZ > "$root/${name}.ended"
}
if [ "$stage" = all ]; then
  run main 5 --real --device cuda --dtype bf16 --train-n 4096 --steps 512 \
  --batch-size 8 --accumulation 2 --eval-batch-size 16 --n 128 --eval-every 128 \
  --save-checkpoint --out "$root/main"
  run baselines 6 --real --device cuda --dtype bf16 --n 500 --eval-batch-size 16 --out "$root/baselines"
fi
if [ "$stage" != faults ]; then
  run final_val 7 --real --device cuda --dtype bf16 --n 500 --eval-batch-size 16 \
  --checkpoint "$root/main/best" --out "$root/final"
fi
for variant in correct shift prompt no_eos; do
  run "fault_${variant}" 5 --real --device cuda --dtype bf16 --train-n 1024 --steps 80 \
    --batch-size 2 --accumulation 4 --gradient-checkpointing --eval-batch-size 16 --n 80 --variant "$variant" \
    --out "$root/fault_${variant}"
done
echo complete > "$root/suite.exit"
