#!/usr/bin/env bash
#SBATCH --job-name=exp0015-general-vi
#SBATCH --account=phan
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-exp0015-%j.out
#SBATCH --error=slurm-exp0015-%j.out

set -euo pipefail
repo_dir=${REPO_DIR:-${SLURM_SUBMIT_DIR:?Submit this job from the repository root}}
cd "$repo_dir"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export HF_HOME=${HF_HOME:-"$repo_dir/cache/huggingface"}
eval_env=${VLLM_ENV:-"$repo_dir/.venv-vllm-slurm"}

# Separate environment; the training environment is never modified.
if [ ! -x "$eval_env/bin/python" ] || ! "$eval_env/bin/python" -c 'import vllm, yaml, antlr4, pebble; assert vllm.__version__.split("+")[0] == "0.31.0"' >/dev/null 2>&1; then
    if [ -x "$repo_dir/.venv/bin/python" ]; then
        export PATH="$repo_dir/.venv/bin:$PATH"
    fi
    bash scripts/setup_vllm_env.sh "$eval_env"
fi
source "$eval_env/bin/activate"
python -c 'import torch, vllm; assert torch.cuda.is_available(), "CUDA is unavailable"; assert vllm.__version__.split("+")[0] == "0.31.0", vllm.__version__; print("vLLM", vllm.__version__, "GPU", torch.cuda.get_device_name(0), flush=True)'
nvidia-smi

output_dir=${OUTPUT_DIR:-"$repo_dir/EVAL/outputs/general_vi_exp0015/slurm-${SLURM_JOB_ID}"}
mkdir -p "$output_dir"
python -u scripts/run_exp0015_general_vi.py \
    --output-dir "$output_dir" \
    --limit "${EVAL_LIMIT:-0}" \
    --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-32768}" \
    --max-num-seqs "${EVAL_BATCH_SIZE:-8}" \
    --gpu-memory-utilization "${EVAL_GPU_MEMORY_UTILIZATION:-0.70}" \
    "$@" 2>&1 | tee "$output_dir/evaluation.log"
