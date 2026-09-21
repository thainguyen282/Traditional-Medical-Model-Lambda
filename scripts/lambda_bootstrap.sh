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
echo "----- nvidia-smi (host) -----"
nvidia-smi || true
echo "----- kernel/driver -----"
uname -r || true
modinfo nvidia 2>/dev/null | grep -E '^version:' || true
echo "=========================================================="

# ------------------------------------------------------------------
# 1. Sanity: docker + nvidia runtime (preinstalled on Lambda images).
#    Lambda ships docker but 'ubuntu' user isn't in the 'docker' group,
#    so we invoke docker via sudo throughout. Passwordless sudo is on.
# ------------------------------------------------------------------
command -v docker >/dev/null 2>&1 || fail "docker not found on host"
DOCKER="sudo docker"
$DOCKER info >/dev/null 2>&1 || fail "docker daemon not reachable even via sudo"

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

# If a custom inference script was scp'd to /home/ubuntu, move it into the
# cloned repo so subsequent steps can run it from /workspace/scripts/.
if [ -n "${INFERENCE_SCRIPT:-}" ]; then
    script_basename=$(basename "$INFERENCE_SCRIPT")
    if [ -f "/home/ubuntu/$script_basename" ]; then
        mkdir -p "$REPO_DIR/scripts"
        mv "/home/ubuntu/$script_basename" "$REPO_DIR/scripts/$script_basename" || \
            echo "[bootstrap] WARN: failed moving $script_basename into repo"
    fi
fi

# If lambda_push_to_hf.py was scp'd to /home/ubuntu, move it into the repo's
# scripts directory so later in-docker pushes can reference /workspace/scripts/.
if [ -f "/home/ubuntu/lambda_push_to_hf.py" ]; then
    mkdir -p "$REPO_DIR/scripts"
    mv "/home/ubuntu/lambda_push_to_hf.py" "$REPO_DIR/scripts/lambda_push_to_hf.py" || \
        echo "[bootstrap] WARN: failed moving lambda_push_to_hf.py into repo"
fi

# ------------------------------------------------------------------
# 4. Docker pull the pre-baked image.
#    Public ghcr.io images need no login. If you flip the package to
#    private, add: echo "$GH_TOKEN" | docker login ghcr.io -u <user> --password-stdin
# ------------------------------------------------------------------
if [ -n "${GH_TOKEN:-}" ] && [ "${GHCR_LOGIN:-}" = "1" ]; then
    echo "$GH_TOKEN" | $DOCKER login ghcr.io \
        -u "${GHCR_USER:-$USER}" --password-stdin \
        || fail "docker login ghcr.io"
fi
$DOCKER pull "$IMAGE" || fail "docker pull $IMAGE"

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
    $DOCKER run --rm --gpus all \
        --shm-size=32g \
        --ipc=host \
        -v "$REPO_DIR":/workspace \
        -v /home/ubuntu:/host \
        -e HF_TOKEN="$HF_TOKEN" \
        -e WANDB_API_KEY="${WANDB_KEY:-}" \
        -e HF_CPT_REPO="$HF_CPT_REPO" \
        -e HF_SFT_REPO="$HF_SFT_REPO" \
        -e JOB_TAG="$JOB_TAG" \
        -e ROOT=/workspace \
        --workdir /workspace \
        "$IMAGE" \
        bash -lc "$1"
}

# ------------------------------------------------------------------
# 7. Run training (+ auto-eval, + HF push) inside the container.
# ------------------------------------------------------------------
run_cpt() {
    # Step 1: train + push adapter to HF inside container #1.
    echo "[bootstrap] ===== CPT: train + push adapter ====="
    in_docker "cd /workspace/CPT && python CPT.py" || return $?

    # Step 2: run eval notebooks in SEPARATE docker runs (one fresh CUDA
    # context per notebook). CPT.py's training initialises CUDA, and any
    # subprocess forked from that python process inherits a corrupted NVML
    # state (torch.cuda.is_available=False). A new `docker run` is a new
    # process tree → clean CUDA init.
    # CPT.py pushes the adapter to `main`, so eval must load from main too.
    local HF_REVISION="main"
    echo "[bootstrap] ===== CPT: eval (fresh containers, model from HF) ====="
    for nb in Unsloth_QA_CPT_new.ipynb Unsloth_QA_CPT_VM14K.ipynb; do
        local stem="${nb%.ipynb}"
        echo "[bootstrap] -- eval: $nb"
        in_docker "cd /workspace/EVAL && \
            jupyter nbconvert --to script --output '$stem' '$nb'" \
            || { echo "[bootstrap] WARN: nbconvert failed for $nb"; continue; }
        in_docker "cd /workspace/EVAL && \
            export CPT_HF_REPO='$HF_CPT_REPO' && \
            export CPT_HF_REVISION='$HF_REVISION' && \
            python '$stem.py'" \
            || echo "[bootstrap] WARN: eval $nb exited non-zero, continuing"
    done

    # Step 3: push eval outputs + train.log to the same revision.
    echo "[bootstrap] ===== CPT: push eval outputs + log ====="
    in_docker "python /workspace/scripts/lambda_push_to_hf.py \
        --task cpt \
        --repo '$HF_CPT_REPO' \
        --src /workspace/CPT/cpt_qwen3_medical \
        --revision main \
        --skip-model \
        --eval-src /workspace/EVAL/outputs \
        --log-file /host/train.log" \
        || echo "[bootstrap] WARN: eval push failed (adapter is already on HF)"
}

run_sft() {
    local config="$1"
    # Step 1: train + Benchmarker (lm-eval runs in-process, not a subprocess
    # — it does not trip the fork-after-CUDA issue that notebooks do).
    echo "[bootstrap] ===== SFT train: $config ====="
    in_docker "cd /workspace && python main.py --config '$config'" || return $?

    # Step 2: push adapter (+ auto benchmark_evaluations) to HF so the
    # eval notebooks can load the exact pushed version from the Hub.
    echo "[bootstrap] ===== SFT push adapter ====="
    in_docker "python /workspace/scripts/lambda_push_to_hf.py \
        --task sft \
        --repo '$HF_SFT_REPO' \
        --config '$config' \
        --revision '$JOB_TAG'" \
        || return $?

    # Step 3: compute the full revision name + pick which notebooks to run.
    local HF_REVISION
    HF_REVISION=$(in_docker "python /workspace/scripts/lambda_push_to_hf.py \
        --task sft --repo '$HF_SFT_REPO' --config '$config' \
        --revision '$JOB_TAG' --print-revision" | tr -d '\r' | tail -n1)
    if [ -z "$HF_REVISION" ]; then
        echo "[bootstrap] WARN: could not derive HF revision; skipping eval."
        return 0
    fi
    echo "[bootstrap] SFT HF revision: $HF_REVISION"

    local is_cpt_sft
    is_cpt_sft=$(in_docker "python -c \"
import yaml
cfg = yaml.safe_load(open('$config'))
print('yes' if 'cpt' in cfg.get('model', {}).get('name', '').lower() else 'no')
\"" | tr -d '\r' | tail -n1)

    local official_vm14k_test
    official_vm14k_test=$(in_docker "python -c \"
import yaml
cfg = yaml.safe_load(open('$config'))
print('yes' if cfg.get('evaluation', {}).get('official_vm14k_test', False) else 'no')
\"" | tr -d '\r' | tail -n1)

    # Step 4: run each notebook in its OWN docker run — fresh CUDA context,
    # avoids the fork-after-training NVML breakage.
    echo "[bootstrap] ===== SFT eval (fresh containers, model from HF) ====="
    if [ "$official_vm14k_test" = "yes" ]; then
        echo "[bootstrap] -- eval: official VM14K test split (29 rows)"
        in_docker "rm -rf /workspace/EVAL/outputs && mkdir -p /workspace/EVAL/outputs" \
            || { echo "[bootstrap] WARN: clear eval outputs failed"; return 1; }
        in_docker "cd /workspace/EVAL && \
            export SFT_HF_REPO='$HF_SFT_REPO' && \
            export SFT_HF_REVISION='$HF_REVISION' && \
            python 'Unsloth_QA_CPT_SFT_VM14K_TEST.py'" \
            || echo "[bootstrap] WARN: official VM14K test exited non-zero, continuing"
    else
        local notebooks
        if [ "$is_cpt_sft" = "yes" ]; then
            notebooks="Unsloth_QA_CPT_SFT_new.ipynb Unsloth_QA_CPT_SFT_VM14K.ipynb"
        else
            notebooks="Unsloth_QA_SFT_new.ipynb Unsloth_QA_SFT_VM14K.ipynb"
        fi
        for nb in $notebooks; do
            local stem="${nb%.ipynb}"
            echo "[bootstrap] -- eval: $nb"
            in_docker "cd /workspace/EVAL && \
                jupyter nbconvert --to script --output '$stem' '$nb'" \
                || { echo "[bootstrap] WARN: nbconvert failed for $nb"; continue; }
            in_docker "cd /workspace/EVAL && \
                export SFT_HF_REPO='$HF_SFT_REPO' && \
                export SFT_HF_REVISION='$HF_REVISION' && \
                python '$stem.py'" \
                || echo "[bootstrap] WARN: eval $nb exited non-zero, continuing"
        done
    fi

    # Step 5: push eval outputs + train.log to the same revision.
    echo "[bootstrap] ===== SFT push eval + log ====="
    in_docker "python /workspace/scripts/lambda_push_to_hf.py \
        --task sft \
        --repo '$HF_SFT_REPO' \
        --config '$config' \
        --revision '$JOB_TAG' \
        --skip-model \
        --eval-src /workspace/EVAL/outputs \
        --log-file /host/train.log" \
        || echo "[bootstrap] WARN: eval push failed (adapter is already on HF)"
}

# Sync with inference.py / README.md "🎯 Model output". Format per row:
#   <display_name>|<hf_repo>|<hf_revision>|<variant>
# variant ∈ {goc, goc_test, cpt, sft, cpt_sft}: picks the matching
# Unsloth_QA_<variant>_* notebooks. cpt_sft_test instead runs only
# Unsloth_QA_CPT_SFT_VM14K_TEST.py against the official 29-row test split.
# All variants also select the right HF_REPO/HF_REVISION env vars.
EVAL_README_MODELS=(
    "Goc|unsloth/Qwen3-4B-Instruct-2507||goc"
    "CPT-only|huyennkai/Venera-CPT-TMModel|main|cpt"
    "CPT-SFT-multiplechoice|huyennkai/Venera-Traditional-Model|cpt-sft-unsloth-qwen3-4B-exp0001-tm-sft-1776945292|cpt_sft"
    "SFT-multiplechoice|huyennkai/Venera-Traditional-Model|sft-unsloth-qwen3-4B-exp0001-tm-sft-1776949731|sft"
    "SFT-2topics|huyennkai/Venera-Traditional-Model|sft-unsloth-qwen3-4B-exp0001-tm-sft-1776771092|sft"
    "CPT-SFT-2topics|huyennkai/Venera-Traditional-Model|cpt-sft-unsloth-qwen3-4B-exp0001-tm-sft-1776784126|cpt_sft"
)

run_eval() {
    # Re-run Unsloth_QA notebooks for each README model, push outputs to that
    # model's existing HF branch (Goc has no branch → eval-goc-<JOB_TAG>).
    # Each notebook runs in a fresh `docker run` (CUDA-fork-after-train issue
    # documented in run_cpt — same root cause; load+infer in one container,
    # then exit so the next notebook gets a clean CUDA init).
    #
    # If EVAL_ONLY env var is set
    # ("name|source_repo|revision|variant[|output_repo]"), override
    # the README list with just that single entry. Used to re-eval one model
    # after an extract_answer fix, without re-running all 6. output_repo is
    # optional; when present, load from source_repo and upload results there.
    local models_to_run=()
    if [ -n "${EVAL_ONLY:-}" ]; then
        echo "[bootstrap] EVAL_ONLY set — overriding 6-model loop"
        models_to_run=("$EVAL_ONLY")
    else
        models_to_run=("${EVAL_README_MODELS[@]}")
    fi
    local fail_count=0
    for entry in "${models_to_run[@]}"; do
        IFS='|' read -r display repo revision variant output_repo <<< "$entry"
        echo "[bootstrap] ===== EVAL $display ($variant) ====="
        echo "[bootstrap] repo=$repo revision=${revision:-<none>}"

        # Pick the evaluation files for this variant. Most variants run a
        # notebook pair; cpt_sft_test runs one checked-in Python script.
        local env_export
        local eval_files=()
        case "$variant" in
            goc)
                eval_files=("Unsloth_QA_goc_new.ipynb" "Unsloth_QA_goc_VM14K.ipynb")
                env_export=""    # Goc loads base model directly, no env override
                ;;
            goc_test)
                eval_files=("Unsloth_QA_CPT_SFT_VM14K_TEST.py")
                # The test script uses FastLanguageModel.from_pretrained, so
                # passing the clean base repo (without a PEFT revision) loads
                # the unadapted model through the same inference path.
                env_export="export SFT_HF_REPO='$repo' && unset SFT_HF_REVISION && export EVAL_OUTPUT_SUBDIR='GOC_VM14K_TEST' && export EVAL_MODEL_NAME='qwen3-4b-instruct-2507-clean' &&"
                ;;
            cpt)
                eval_files=("Unsloth_QA_CPT_new.ipynb" "Unsloth_QA_CPT_VM14K.ipynb")
                env_export="export CPT_HF_REPO='$repo' && export CPT_HF_REVISION='$revision' &&"
                ;;
            sft)
                eval_files=("Unsloth_QA_SFT_new.ipynb" "Unsloth_QA_SFT_VM14K.ipynb")
                env_export="export SFT_HF_REPO='$repo' && export SFT_HF_REVISION='$revision' &&"
                ;;
            cpt_sft)
                eval_files=("Unsloth_QA_CPT_SFT_new.ipynb" "Unsloth_QA_CPT_SFT_VM14K.ipynb")
                env_export="export SFT_HF_REPO='$repo' && export SFT_HF_REVISION='$revision' &&"
                ;;
            cpt_sft_test)
                eval_files=("Unsloth_QA_CPT_SFT_VM14K_TEST.py")
                env_export="export SFT_HF_REPO='$repo' && export SFT_HF_REVISION='$revision' &&"
                ;;
            *)
                echo "[bootstrap] WARN: unknown variant '$variant' for $display — skipping"
                continue
                ;;
        esac

        # Clear EVAL/outputs so notebooks for this model write into a clean
        # dir; otherwise responses.jsonl from the previous model would
        # append (we open in mode "a"). Run inside docker because the bind
        # mount belongs to root, not ubuntu.
        in_docker "rm -rf /workspace/EVAL/outputs && mkdir -p /workspace/EVAL/outputs" \
            || { echo "[bootstrap] WARN: clear outputs failed for $display"; ((fail_count++)); continue; }

        local nb_failed=0
        for eval_file in "${eval_files[@]}"; do
            local script_file="$eval_file"
            if [[ "$eval_file" = *.ipynb ]]; then
                local stem="${eval_file%.ipynb}"
                in_docker "cd /workspace/EVAL && jupyter nbconvert --to script --output '$stem' '$eval_file'" \
                    || { echo "[bootstrap] WARN: nbconvert failed for $eval_file"; nb_failed=1; continue; }
                script_file="$stem.py"
            fi
            in_docker "cd /workspace/EVAL && $env_export python '$script_file'" \
                || { echo "[bootstrap] WARN: $eval_file exit non-zero — continuing"; nb_failed=1; }
        done

        # Push to HF: existing branch for trained models, fresh branch for Goc.
        local target_repo target_revision
        if [ -n "$output_repo" ] && [ "$variant" = "goc_test" ]; then
            target_repo="$output_repo"
            target_revision="eval-goc-test-$JOB_TAG"
        elif [ -n "$output_repo" ]; then
            target_repo="$output_repo"
            target_revision="$revision"
        elif [ "$variant" = "goc" ] || [ "$variant" = "goc_test" ]; then
            target_repo="$HF_SFT_REPO"
            target_revision="eval-$variant-$JOB_TAG"
        else
            target_repo="$repo"
            target_revision="$revision"
        fi

        echo "[bootstrap] push eval outputs -> $target_repo @ $target_revision"
        in_docker "python /workspace/scripts/lambda_push_to_hf.py \
            --task eval \
            --repo '$target_repo' \
            --revision '$target_revision' \
            --eval-src /workspace/EVAL/outputs" \
            || { echo "[bootstrap] WARN: push failed for $display"; ((fail_count++)); }

        if [ "$nb_failed" = "1" ]; then
            ((fail_count++))
        fi
    done

    # Final log push to a dedicated revision so train.log isn't lost.
    echo "[bootstrap] ===== EVAL: push aggregated train.log ====="
    in_docker "python /workspace/scripts/lambda_push_to_hf.py \
        --task eval \
        --repo '$HF_SFT_REPO' \
        --revision 'eval-$JOB_TAG' \
        --log-file /host/train.log" \
        || echo "[bootstrap] WARN: log push failed"

    if [ "$fail_count" -gt 0 ]; then
        echo "[bootstrap] EVAL completed with $fail_count partial failures"
        return 1
    fi
    return 0
}

run_inference() {
    # inference.py load từng model trên HF revision của nó → generate 5 câu →
    # ghi EVAL/outputs/comparison/comparison_<ts>.{json,txt}. Không train gì,
    # không đụng CUDA fork (chỉ 1 docker run duy nhất).
    echo "[bootstrap] ===== inference compare ====="
    # Allow overriding the inference script via INFERENCE_SCRIPT env var
    # If a custom inference script was provided, call it. For the
    # `test_inference_max_tokens.py` helper (which expects `--model`),
    # default the `--model` to `HF_SFT_REPO` so the job can run without
    # additional CLI args.
    if [ -n "${INFERENCE_SCRIPT:-}" ]; then
        script_path="/workspace/${INFERENCE_SCRIPT}"
        script_basename=$(basename "${INFERENCE_SCRIPT}")
        if [ "$script_basename" = "test_inference_max_tokens.py" ]; then
            echo "[bootstrap] running custom inference script: $script_basename (auto --model from HF_SFT_REPO)"
            if [ -n "${INFERENCE_REVISION:-}" ]; then
                echo "[bootstrap] passing --revision ${INFERENCE_REVISION} to inference script"
                in_docker "cd /workspace && python '$script_path' --model '$HF_SFT_REPO' --revision '$INFERENCE_REVISION'" || return $?
            else
                in_docker "cd /workspace && python '$script_path' --model '$HF_SFT_REPO'" || return $?
            fi
        else
            in_docker "cd /workspace && python '$script_path'" || return $?
        fi
    else
        in_docker "cd /workspace && python inference.py" || return $?
    fi

    local HF_INF_REPO="${HF_INF_REPO:-$HF_SFT_REPO}"
    echo "[bootstrap] ===== push comparison + log to $HF_INF_REPO ====="
    in_docker "python /workspace/scripts/lambda_push_to_hf.py \
        --task inference \
        --repo '$HF_INF_REPO' \
        --revision '$JOB_TAG' \
        --eval-src /workspace/EVAL/outputs/comparison \
        --log-file /host/train.log" \
        || echo "[bootstrap] WARN: inference push failed"
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
elif [ "$TASK_TYPE" = "inference" ]; then
    run_inference || status=$?
elif [ "$TASK_TYPE" = "eval" ]; then
    run_eval || status=$?
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
