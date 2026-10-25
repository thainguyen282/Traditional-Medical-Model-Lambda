#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
eval_env=${VLLM_ENV:-"$repo_dir/.venv-vllm-slurm"}
if [ ! -x "$eval_env/bin/python" ]; then
    echo "Missing environment: $eval_env. Run: bash scripts/setup_vllm_env.sh \"$eval_env\"" >&2
    exit 1
fi
export PATH="$eval_env/bin:$PATH"
export HF_HOME=${HF_HOME:-"$repo_dir/cache/huggingface"}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
# Keep the device assignment supplied by the current machine/job.
python -c 'import torch, vllm; assert vllm.__version__.split("+")[0] == "0.31.0"; assert torch.cuda.is_available(), "CUDA unavailable"; print("vLLM", vllm.__version__, "GPU", torch.cuda.get_device_name(0), flush=True)'
exec python -u scripts/run_cpt_general_vi.py "$@"
