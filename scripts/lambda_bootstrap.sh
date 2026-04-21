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
#   DOCKER_IMAGE   optional; defaults to ghcr.io/venera-ai/traditional-medical-model:<GIT_BRANCH>
#
# Flow (Dockerized):
#   1. Ensure Docker + NVIDIA runtime (preinstalled on Lambda images).
#   2. git clone repo for configs, scripts, notebooks, data parquets.
#   3. docker pull the pre-baked deps image.
#   4. docker run training + eval + push_to_hf all inside the container,
#      with the repo bind-mounted at /workspace.
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
IMAGE_DEFAULT="ghcr.io/venera-ai/traditional-medical-model:${GIT_BRANCH:-main2}"
IMAGE="${DOCKER_IMAGE:-$IMAGE_DEFAULT}"

fail() {
    echo "[bootstrap] FAIL: $*" >&2
    touch /home/ubuntu/DONE_FAILED
    exit 1
}

echo "=========================================================="
echo "[bootstrap] job=$JOB_TAG task=$TASK_TYPE  start=$(date -u +%FT%TZ)"
echo "[bootstrap] host=$(hostname)"
echo "[bootstrap] image=$IMAGE"
nvidia-smi -L || true
echo "=========================================================="

# ------------------------------------------------------------------
# 1. Sanity: docker + nvidia runtime (preinstalled on Lambda images).
# ------------------------------------------------------------------
command -v docker >/dev/null 2>&1 || fail "docker not found on host"
docker info >/dev/null 2>&1 || fail "docker daemon not reachable (user not in docker group?)"

# ------------------------------------------------------------------
# 2. System deps needed OUTSIDE the container (just git for cloning).
# ------------------------------------------------------------------
if ! command -v git >/dev/null 2>&1; then
    sudo apt-get update -y || fail "apt update"
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y git git-lfs jq \
        || fail "apt install git"
    git lfs install || true
fi

# ------------------------------------------------------------------
# 3. Clone repo (configs, scripts, data, notebooks).
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
# 4. Docker pull the pre-baked image.
#    Public ghcr.io images need no login. If you flip the package to
#    private, add: echo "$GH_TOKEN" | docker login ghcr.io -u <user> --password-stdin
# ------------------------------------------------------------------
if [ -n "${GH_TOKEN:-}" ] && [ "${GHCR_LOGIN:-}" = "1" ]; then
    echo "$GH_TOKEN" | docker login ghcr.io \
        -u "${GHCR_USER:-$USER}" --password-stdin \
        || fail "docker login ghcr.io"
fi
docker pull "$IMAGE" || fail "docker pull $IMAGE"

# ------------------------------------------------------------------
# 5. Mirror .env in the repo so main.py/CPT.py see the same vars.
#    ROOT=/workspace because that's the container-side bind mount path.
# ------------------------------------------------------------------
cat > "$REPO_DIR/.env" <<EOF
ROOT=/workspace
WANDB_KEY=${WANDB_KEY:-}
HF_TOKEN=${HF_TOKEN}
EOF

# ------------------------------------------------------------------
# 6. Helper: run a command inside the training container.
#    - /workspace : bind-mounted repo (read/write so outputs land on host)
#    - /host      : /home/ubuntu on host (for train.log + scp'd helpers)
#    - --shm-size : dataloader workers need large /dev/shm
# ------------------------------------------------------------------
in_docker() {
    docker run --rm --gpus all \
        --shm-size=32g \
        --ipc=host \
        -v "$REPO_DIR":/workspace \
        -v /home/ubuntu:/host \
        -e HF_TOKEN \
        -e WANDB_API_KEY="${WANDB_KEY:-}" \
        -e HF_CPT_REPO -e HF_SFT_REPO \
        -e JOB_TAG -e ROOT=/workspace \
        --workdir /workspace \
        "$IMAGE" \
        bash -lc "$1"
}

# ------------------------------------------------------------------
# 7. Run training (+ auto-eval, + HF push) inside the container.
# ------------------------------------------------------------------
run_cpt() {
    echo "[bootstrap] ===== CPT ====="
    in_docker "cd /workspace/CPT && python CPT.py" || return $?

    echo "[bootstrap] pushing CPT artifact + eval outputs to HF: $HF_CPT_REPO"
    in_docker "python /workspace/scripts/lambda_push_to_hf.py \
        --task cpt \
        --repo '$HF_CPT_REPO' \
        --src /workspace/CPT/cpt_qwen3_medical \
        --revision '$JOB_TAG' \
        --eval-src /workspace/EVAL/outputs \
        --log-file /host/train.log" || return $?
}

run_sft() {
    local config="$1"
    echo "[bootstrap] ===== SFT: $config ====="
    in_docker "cd /workspace && python main.py --config '$config'" || return $?

    # main.py writes model to group_outputs/outputs/{SFT|CPT_SFT}/model and
    # Benchmarker JSONs to group_outputs/outputs/{SFT|CPT_SFT}/benchmark_evaluations.
    # Unsloth_QA_*.ipynb write to EVAL/outputs/{SFT_*|CPT_SFT_*}/.
    # push_to_hf.py auto-uploads benchmark_evaluations; we add EVAL/outputs.
    in_docker "python /workspace/scripts/lambda_push_to_hf.py \
        --task sft \
        --repo '$HF_SFT_REPO' \
        --config '$config' \
        --revision '$JOB_TAG' \
        --eval-src /workspace/EVAL/outputs \
        --log-file /host/train.log" || return $?
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
