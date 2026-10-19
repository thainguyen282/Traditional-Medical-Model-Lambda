#!/usr/bin/env bash
#SBATCH --job-name=vm14k-thinking
#SBATCH --account=phan
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-vm14k-thinking-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export HF_HOME="$PWD/cache/huggingface"
eval_python="$PWD/.venv-vllm-slurm/bin/python"
output_dir="$PWD/EVAL/outputs/vm14k_thinking/slurm-${SLURM_JOB_ID}"
mkdir -p "$output_dir"
for model in qwen3-4b-thinking-2507 gemma4-e2b; do
    revision=768f209d9ea81521153ed38c47d515654e938aea
    if [ "$model" = gemma4-e2b ]; then
        revision=3e22461f65e89153144f8adb70e3b8c2cc9845a7
    fi
    "$eval_python" -u scripts/eval_general_vi_vllm.py \
        --model "$model" --revision "$revision" --benchmark vm14k --protocol thinking \
        --max-input-tokens 4096 --max-new-tokens 32768 --max-num-seqs 8 \
        --gpu-memory-utilization 0.70 --enforce-eager --output-dir "$output_dir"
done
"$eval_python" scripts/eval_general_vi_vllm.py --export-comparison \
    --protocol thinking --benchmarks vm14k --output-dir "$output_dir"
