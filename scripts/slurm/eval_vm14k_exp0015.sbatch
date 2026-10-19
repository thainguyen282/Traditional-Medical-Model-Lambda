#!/usr/bin/env bash
#SBATCH --job-name=vm14k-exp0015
#SBATCH --account=phan
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-vm14k-exp0015-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export HF_HOME="$PWD/cache/huggingface"
eval_python="$PWD/.venv-vllm-slurm/bin/python"
output_dir="$PWD/EVAL/outputs/vm14k_exp0015/slurm-${SLURM_JOB_ID}"
mkdir -p "$output_dir"
"$eval_python" -u scripts/eval_general_vi_vllm.py \
    --model exp0015 --benchmark vm14k --protocol thinking \
    --max-input-tokens 4096 --max-new-tokens 32768 --max-num-seqs 8 \
    --gpu-memory-utilization 0.70 --enforce-eager --output-dir "$output_dir" \
    2>&1 | tee "$output_dir/evaluation.log"
"$eval_python" scripts/eval_general_vi_vllm.py --export-comparison \
    --comparison-models exp0015 --protocol thinking --benchmarks vm14k \
    --output-dir "$output_dir"
