#!/usr/bin/env bash
#SBATCH --job-name=vm14k-qwen35-base
#SBATCH --account=phan
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/vm14k-qwen35-base-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export HF_HOME="$PWD/.cache/huggingface"
export PYTORCH_ALLOC_CONF=expandable_segments:True
output_dir="$PWD/EVAL/outputs/qwen35_base_vm14k_direct/slurm-${SLURM_JOB_ID}"
mkdir -p "$output_dir"
exec > >(tee "$output_dir/evaluation.log") 2>&1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
# Keep CUDA packages in this job's scratch space and preserve existing environments.
job_scratch=$(mktemp -d "/tmp/qwen35-base-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf "$job_scratch"' EXIT
export UV_CACHE_DIR="$job_scratch/uv-cache"
uv_bin="${UV_BIN:-/mmfs1/project/phan/tqn/.local/bin/uv}"
"$uv_bin" venv --python "$PWD/.venv/bin/python" "$job_scratch/venv"
"$uv_bin" pip install --python "$job_scratch/venv/bin/python" \
    --torch-backend cu128 -r requirements-baselines.txt
"$job_scratch/venv/bin/python" -u scripts/eval_medical_baselines.py \
    --model qwen35-base --dataset vm14k --generation-mode direct \
    --batch-size 4 --output-dir "$output_dir"
