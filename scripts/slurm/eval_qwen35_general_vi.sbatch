#!/usr/bin/env bash
#SBATCH --job-name=qwen35-general-vi
#SBATCH --array=0-1
#SBATCH --account=phan
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/qwen35-general-vi-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
models=(qwen35-base qwen35-mixed)
model=${models[${SLURM_ARRAY_TASK_ID:?This job requires array task 0 or 1}]}
# Each array task evaluates exactly one checkpoint.
for arg in "$@"; do
    case "$arg" in
        --models|--models=*) echo "Checkpoint is selected by the array task; omit --models" >&2; exit 2 ;;
    esac
done
echo "Evaluating $model (array ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID})"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export HF_HOME=${HF_HOME:-"$PWD/.cache/huggingface"}
export PYTORCH_ALLOC_CONF=expandable_segments:True
scratch_dir=$(mktemp -d "/tmp/qwen35-general-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf "$scratch_dir"' EXIT
export UV_CACHE_DIR="$scratch_dir/uv-cache"
uv_bin=${UV_BIN:-/mmfs1/project/phan/tqn/.local/bin/uv}
"$uv_bin" venv --python "$PWD/.venv/bin/python" "$scratch_dir/venv"
"$uv_bin" pip install --python "$scratch_dir/venv/bin/python" --torch-backend cu128 -r requirements-baselines.txt
"$scratch_dir/venv/bin/python" -u scripts/eval_qwen35_general_vi.py \
    --models "$model" \
    --output-dir "EVAL/outputs/qwen35_general_vi/slurm-${SLURM_ARRAY_JOB_ID}/${model}" "$@"
