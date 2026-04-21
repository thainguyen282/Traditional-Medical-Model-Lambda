#!/usr/bin/env bash
#
# Runs on a fresh Lambda Cloud instance. Driven by /home/ubuntu/job.env which
# lambda_launch.py scp'd in, containing:
#   TASK_TYPE      cpt | sft
#   CONFIG_LIST    newline-separated config paths (empty for cpt)
#   GIT_REPO       repo URL
#   GIT_BRANCH     branch
#   HF_TOKEN       write-enabled Hugging Face token
#   WANDB_KEY      optional
#   GH_TOKEN       optional, only needed for private repos
#   HF_CPT_REPO    target HF repo for CPT artifacts
#   HF_SFT_REPO    target HF repo for SFT artifacts
#   JOB_TAG        human-readable tag for logging
#
# On finish, writes one of:
#   /home/ubuntu/DONE_SUCCESS
#   /home/ubuntu/DONE_FAILED
# lambda_launch.py polls for these and then terminates the instance.

set -u -o pipefail

cd /home/ubuntu
exec > >(tee -a /home/ubuntu/train.log) 2>&1

# shellcheck disable=SC1091
source /home/ubuntu/job.env

REPO_DIR="/home/ubuntu/Traditional-Medical-Model"

fail() {
    echo "[bootstrap] FAIL: $*" >&2
    touch /home/ubuntu/DONE_FAILED
    exit 1
}

echo "=========================================================="
echo "[bootstrap] job=$JOB_TAG task=$TASK_TYPE  start=$(date -u +%FT%TZ)"
echo "[bootstrap] host=$(hostname)  gpus:"
nvidia-smi -L || true
echo "=========================================================="

# ------------------------------------------------------------------
# 1. System deps (Lambda images usually have CUDA + python; install tmux + git)
# ------------------------------------------------------------------
sudo apt-get update -y || fail "apt update"
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git git-lfs tmux jq python3-venv build-essential \
    || fail "apt install"
git lfs install || true

# ------------------------------------------------------------------
# 2. Clone repo
# ------------------------------------------------------------------
if [ ! -d "$REPO_DIR/.git" ]; then
    if [ -n "${GH_TOKEN:-}" ]; then
        auth_repo="${GIT_REPO/https:\/\//https:\/\/${GH_TOKEN}@}"
    else
        auth_repo="$GIT_REPO"
    fi
    git clone --branch "$GIT_BRANCH" "$auth_repo" "$REPO_DIR" \
        || fail "git clone"
fi
cd "$REPO_DIR" || fail "cd repo"

# ------------------------------------------------------------------
# 3. Python env. Lambda images ship with a system Python 3 + CUDA.
#    We install deps into a local venv to avoid touching the system.
# ------------------------------------------------------------------
if [ ! -d "$REPO_DIR/.venv" ]; then
    python3 -m venv --system-site-packages "$REPO_DIR/.venv" || fail "venv"
fi
# shellcheck disable=SC1091
source "$REPO_DIR/.venv/bin/activate"

pip install --upgrade pip wheel || fail "pip upgrade"
pip install -r requirements.txt || fail "pip install -r requirements.txt"
pip install "huggingface_hub[cli]" python-dotenv || fail "pip install hf_hub"

# ------------------------------------------------------------------
# 4. Credentials
# ------------------------------------------------------------------
export HF_TOKEN
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential \
    || fail "huggingface-cli login"

if [ -n "${WANDB_KEY:-}" ]; then
    wandb login --relogin "$WANDB_KEY" || echo "[bootstrap] wandb login failed (non-fatal)"
fi

# Mirror .env in the repo so main.py/CPT.py see the same vars.
cat > "$REPO_DIR/.env" <<EOF
ROOT=$REPO_DIR
WANDB_KEY=${WANDB_KEY:-}
HF_TOKEN=${HF_TOKEN}
EOF

# ------------------------------------------------------------------
# 5. Run training
# ------------------------------------------------------------------
run_cpt() {
    echo "[bootstrap] ===== CPT ====="
    cd "$REPO_DIR/CPT" || return 1
    python CPT.py || return $?
    cd "$REPO_DIR"

    echo "[bootstrap] pushing CPT artifact to HF: $HF_CPT_REPO"
    python /home/ubuntu/lambda_push_to_hf.py \
        --task cpt \
        --repo "$HF_CPT_REPO" \
        --src "$REPO_DIR/CPT/cpt_qwen3_medical" \
        --revision "$JOB_TAG" \
        || return $?
}

run_sft() {
    local config="$1"
    echo "[bootstrap] ===== SFT: $config ====="
    cd "$REPO_DIR"
    python main.py --config "$config" || return $?

    # main.py writes to group_outputs/outputs/{SFT|CPT_SFT}/<run>/model.
    # We let push_to_hf.py inspect the config to pick the right artifact.
    python /home/ubuntu/lambda_push_to_hf.py \
        --task sft \
        --repo "$HF_SFT_REPO" \
        --config "$config" \
        --revision "$JOB_TAG" \
        || return $?
}

status=0
if [ "$TASK_TYPE" = "cpt" ]; then
    run_cpt || status=$?
elif [ "$TASK_TYPE" = "sft" ]; then
    [ -n "${CONFIG_LIST:-}" ] || fail "CONFIG_LIST empty for sft task"
    while IFS= read -r cfg; do
        [ -z "$cfg" ] && continue
        run_sft "$cfg" || { status=$?; break; }
    done <<< "$CONFIG_LIST"
else
    fail "unknown TASK_TYPE=$TASK_TYPE"
fi

echo "=========================================================="
echo "[bootstrap] job=$JOB_TAG  end=$(date -u +%FT%TZ)  exit=$status"
echo "=========================================================="

if [ "$status" -eq 0 ]; then
    touch /home/ubuntu/DONE_SUCCESS
else
    touch /home/ubuntu/DONE_FAILED
fi
exit "$status"
