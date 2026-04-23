#!/usr/bin/env bash
# Spin until Lambda gives us ANY acceptable GPU, then launch.
# Usage:
#   scripts/lambda_retry.sh --task sft --config configs/experiments/unsloth-qwen3-4B/exp0001/config_cpt.yaml
# (all args are passed through to lambda_launch.py, except --instance-type
#  which is chosen per-iteration from the priority list below)

set -u

# Priority: prefer beefier GPUs so we don't OOM full-precision Qwen3-4B.
TYPES=(
    gpu_1x_a100_sxm4     # 40GB, baseline
    gpu_1x_h100_pcie     # 80GB
    gpu_1x_h100_sxm5     # 80GB
    gpu_2x_a6000         # 2x48GB
    gpu_2x_a100          # 2x40GB
    gpu_8x_a100          # 8x40GB (wasteful, only if desperate)
    gpu_1x_a10           # 24GB (will OOM without load_in_4bit=true)
)

SLEEP_SECS="${SLEEP_SECS:-60}"

while true; do
    for t in "${TYPES[@]}"; do
        echo "=== $(date +%H:%M:%S)  trying $t ==="
        if python scripts/lambda_launch.py "$@" --instance-type "$t"; then
            echo "=== launch finished on $t ==="
            exit 0
        fi
    done
    echo "=== no stock anywhere, sleeping ${SLEEP_SECS}s ==="
    sleep "$SLEEP_SECS"
done
