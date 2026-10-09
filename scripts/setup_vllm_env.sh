#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
eval_env=${1:-"$repo_dir/.venv-vllm"}
python -m venv "$eval_env"
"$eval_env/bin/python" -m pip install uv
"$eval_env/bin/uv" pip install --python "$eval_env/bin/python" \
  --torch-backend=auto \
  'vllm==0.31.0' 'pyarrow==23.0.1' 'python-dotenv==1.2.1'
"$eval_env/bin/python" -c 'import vllm, pyarrow, dotenv; print("vLLM", vllm.__version__)'
