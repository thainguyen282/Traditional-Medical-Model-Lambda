#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
eval_env=${1:-"$repo_dir/.venv-vllm"}
python -m venv --clear "$eval_env"
source "$eval_env/bin/activate"
"$eval_env/bin/python" -m pip install uv
vllm_wheel="https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0+cu129-cp38-abi3-manylinux_2_28_$(uname -m).whl"
"$eval_env/bin/uv" pip install --python "$eval_env/bin/python" \
  --torch-backend=cu129 \
  "$vllm_wheel" 'pyarrow==23.0.1' 'python-dotenv==1.2.1' 'ninja==1.13.2'
"$eval_env/bin/python" -c 'import torch; from vllm import LLM, SamplingParams; import vllm; assert torch.version.cuda.startswith("12.9"), torch.version.cuda; print("vLLM", vllm.__version__, "CUDA", torch.version.cuda, flush=True)'

# Fail during setup if the JIT build executable cannot be launched.
command -v ninja
ninja --version
