#!/usr/bin/env bash
# A separate environment keeps Gemma 4 dependencies out of training.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
eval_env=${1:-"$repo_dir/.venv-baselines"}
python -m venv "$eval_env"
"$eval_env/bin/python" -m pip install -r "$repo_dir/requirements-baselines.txt"
"$eval_env/bin/python" "$repo_dir/scripts/eval_medical_baselines.py" \
    --vm14k-baselines --check-runtime
