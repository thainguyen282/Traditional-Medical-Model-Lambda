# Lambda Cloud training runner

Train CPT / SFT jobs on a Lambda Cloud GPU instance, push artifacts + eval
outputs + train.log to HF Hub, then auto-terminate the instance.

Deps are baked into a pre-built Docker image (`ghcr.io/venera-ai/traditional-medical-model:<branch>`)
so instances don't `pip install` on every launch — they just `docker pull`.

## Files

| File | Chạy ở đâu | Vai trò |
|---|---|---|
| [lambda_launch.py](lambda_launch.py) | **máy local** | Tổng chỉ huy từ xa. Gọi Lambda API tạo instance, scp 2 script kia lên, khởi chạy, theo dõi log, cuối cùng gọi API terminate. |
| [lambda_bootstrap.sh](lambda_bootstrap.sh) | **instance Lambda (host)** | Ensure docker, git clone repo, `docker pull` image, `docker run` training+eval+push inside container. Ghi sentinel `DONE_SUCCESS`/`DONE_FAILED`. |
| [lambda_push_to_hf.py](lambda_push_to_hf.py) | **instance Lambda (container)** | Upload LoRA adapter + tokenizer + `benchmark_evaluations/` + `EVAL/outputs/` + `train.log` lên HF branch. |
| [../Dockerfile](../Dockerfile) | **GitHub Actions** | CUDA 12.8 + Python 3.10 + toàn bộ `requirements.txt`. Rebuild tự động khi `Dockerfile` hoặc `requirements.txt` thay đổi. |
| [../.github/workflows/docker.yml](../.github/workflows/docker.yml) | **GitHub Actions** | Build + push image lên `ghcr.io/venera-ai/traditional-medical-model:<branch>`. |

### Flow tổng quan

```
Máy bạn                          Lambda Cloud instance (host)            Container
────────                         ─────────────────────────               ─────────
lambda_launch.py ──API──▶ [tạo instance]
       │
       ├── scp ─▶  lambda_bootstrap.sh
       ├── scp ─▶  lambda_push_to_hf.py
       │
       └── ssh ─▶  bash lambda_bootstrap.sh
                         │
                         ├─ git clone repo
                         ├─ docker pull ghcr.io/...:<branch>
                         └─ docker run --gpus all  ─────────────▶   python CPT.py / main.py
                                                                    python lambda_push_to_hf.py ──▶ HF Hub

lambda_launch.py ──API──▶ [terminate instance]
```

### Tại sao split layer như vậy

- `lambda_launch.py` **local** — chỉ nó biết IP/SSH key.
- `lambda_bootstrap.sh` **host** — Bash tự nhiên hơn Python cho `apt`, `docker run`, `tmux`.
- `lambda_push_to_hf.py` **container** — có thể rerun độc lập nếu push fail (ssh vào host, `docker run --rm -v $PWD:/workspace $IMAGE python scripts/lambda_push_to_hf.py --revision <tag>`).
- **Docker image chỉ bake deps** (không bake code) — code luôn lấy từ `git clone` mới nhất, image chỉ rebuild khi `requirements.txt` đổi.

## One-time setup

1. Register your **SSH public key** in the Lambda Cloud dashboard (Settings → SSH keys). Remember the name you gave it.
2. Create a **Lambda API key** (Settings → API keys).
3. Create a **write-enabled Hugging Face token** (https://huggingface.co/settings/tokens).
4. If the GitHub repo is private, create a **GitHub PAT** with `repo` scope.
5. Fill in `.env` (see [.env.example](../.env.example)):
   ```dotenv
   LAMBDA_API_KEY=secret_...
   LAMBDA_SSH_KEY_NAME=my-laptop-key
   LAMBDA_SSH_PRIVATE_KEY=~/.ssh/id_ed25519
   HF_TOKEN=hf_...
   WANDB_KEY=...            # optional
   GH_TOKEN=ghp_...         # only for private repos
   ```
6. Install local deps:
   ```bash
   pip install requests python-dotenv
   ```
7. **Trigger the first Docker image build** so Lambda instances have something to pull:
   - Push any change touching `Dockerfile` / `requirements.txt` to `main2`, or
   - Go to GitHub → Actions → "Build & push training image" → Run workflow on `main2`.
   - The image will appear at `ghcr.io/venera-ai/traditional-medical-model:main2`.
   - It is **public by default** — if the repo is private, the package will also be private. In that case, either:
     - Make the package public under `Packages → Settings → Change visibility`, or
     - Pass `GHCR_LOGIN=1` and `GHCR_USER=<gh-username>` in env so bootstrap does `docker login` with `$GH_TOKEN`.

## Usage

### Check available instance types
```bash
python scripts/lambda_launch.py --task sft --list-types
```
> **Warning (Slurm / local VM timeout):** Keep `lambda_launch.py` alive until it
> prints `terminating instance`. If Slurm kills the job or the local VM stops,
> the remote GPU may keep running and billing. 
> After any timeout, node failure, or forced kill, check the Lambda dashboard
> and terminate the instance manually. Do not use `--keep-alive` unless intended.

### Run a single SFT experiment
```bash
python scripts/lambda_launch.py \
    --task sft \
    --config configs/experiments/unsloth-qwen3-4B/exp0001/config.yaml
```
- `main.py` decides CPT-SFT vs pure SFT based on `config.model.name` (substring
  `cpt` → CPT-SFT).
- The HF revision (branch) will be
  `cpt-sft-<model-dir>-<exp-dir>-<job-tag>` or `sft-<model-dir>-<exp-dir>-<job-tag>`
  under `huyennkai/Venera-Traditional-Model`.

### Run CPT
```bash
python scripts/lambda_launch.py --task cpt
```
Pushes to `huyennkai/Venera-CPT-TMModel` on branch `cpt-<job-tag>`.

### Run multiple SFT configs on the same instance (cheaper)
Create a text file with one config per line:
```txt
# scripts/lambda_batch.txt
configs/experiments/unsloth-qwen3-4B/exp0001/config.yaml
configs/experiments/unsloth-qwen3-4B/exp0002/config.yaml
```
Then:
```bash
python scripts/lambda_launch.py --task sft --config-list scripts/lambda_batch.txt
```

### Exp0013 and Exp0014: SFT only from the clean model

Both start from `unsloth/Qwen3-4B-Instruct-2507` without a CPT checkpoint
and retain the datasets and training hyperparameters of their CPT-SFT counterparts.

Exp0013 trains directly on the merged YHCT MCQs:
```bash
python scripts/lambda_launch.py --task sft \
    --config configs/experiments/unsloth-qwen3-4B/exp0013/config.yaml
```

Exp0014 first trains on QA-20K, then continues that SFT adapter on YHCT MCQs.
Run both stages in order on the same instance:
```bash
python scripts/lambda_launch.py --task sft \
    --config-list configs/experiments/unsloth-qwen3-4B/exp0014/configs_sft.txt
```
Stage 2 loads `group_outputs/exp0014/qa20k/SFT/model` from stage 1;
its final adapter is saved to `group_outputs/exp0014/yhct/SFT/model`.
The existing `config_cpt.yaml` files and `exp0014/configs.txt` remain the CPT-SFT variants.

### Baseline evaluation (MedReason, 564 traditional medicine questions)

For the Qwen3-4B / Gemma-4-E2B comparison without thinking, see
[VM14K baseline experiment](VM14K_BASELINES.md). It uses an isolated evaluation
environment and gates full evaluation on a four-question smoke run of both models.

```bash
.venv/bin/python scripts/lambda_launch.py --task eval \
    --baselines medreason \
    --eval-datasets vm14k \
    --eval-batch-size 4 \
    --git-repo https://github.com/thainguyen282/Traditional-Medical-Model-Lambda.git \
    --git-branch thai-dev \
    --region us-east-1
```

`--region us-east-1` selects Virginia, USA. The launcher checks capacity for the
requested GPU and stops before launching if that region is unavailable; it does
not fall back to another region. Omitting `--region` keeps the existing first
available region behavior.

If provisioning is slow, add `--boot-timeout 1200` to allow up to 20 minutes
for Lambda to report `active`. The default remains 600 seconds. An assigned IP
while status is still `booting` does not bypass this check. Timeout still follows
the normal cleanup rule, terminating unless `--keep-alive` was passed.

`--baselines` (alias `--medical-baselines`) opts into the standalone baseline
runner. It defaults to the 564-row `vm14k` subset, all rows (`--eval-limit 0`),
and answer-only decoding (`--eval-generation-mode direct`). For a short smoke
run, add `--eval-limit 4`. `--eval-generation-mode notebook` allows free generation
with the notebook's token limit. Other datasets are `new` and `vm14k-full`;
other model aliases are listed in `scripts/eval_medical_baselines.py`.
These evaluation flags require `--baselines` and cannot be combined with
`--eval-only` or training configs.

The launcher validates Git repository/branch access, local data, and model access
before provisioning. It uses
the existing single A100 default, region selection, Docker image, Docker helper,
SSH polling, and termination behavior. Models and datasets run sequentially.
Only the selected local evaluator/data/notebooks are copied to the instance,
so this baseline path needs no Git push or image rebuild. Keep your existing
`DOCKER_IMAGE` setting when reusing a known working image.

Results (`responses.jsonl`, `summary.json`, `infer_result.txt`, `run_config.json`)
are uploaded to `$HF_SFT_REPO` on branch `eval-medical-<job-tag>` under
`eval/<job-tag>/medreason/vm14k/`. The launcher also attempts a local backup to
`logs/<job-tag>/medical-eval.tar.gz` before termination, including failed runs. If setup failed before creating an output directory,
the downloader skips the missing results. A download failure returns a nonzero status and still follows the existing
cleanup rule: terminate unless `--keep-alive` was explicitly passed. If both
upload and download fail, remote results are lost on termination.

Without `--baselines`, the existing training, inference, and notebook evaluation
paths are unchanged. No extra dependency installs or GPU instances are added.

### Debug / inspect after finish
```bash
python scripts/lambda_launch.py --task sft --config ... --keep-alive
# After the run, SSH in manually with the printed IP to inspect files.
```

### Other flags
- `--instance-type gpu_1x_h100_pcie` — override default `gpu_2x_a100`.
- `--git-branch some-branch` — override default `main2`. Also determines the Docker image tag pulled (`ghcr.io/.../...:<branch>`).

### Override the Docker image
If you want to pin a specific image (e.g. a prior build after the Dockerfile changed), set `DOCKER_IMAGE` in your `.env`:
```dotenv
DOCKER_IMAGE=ghcr.io/venera-ai/traditional-medical-model:sha-abc1234
```
`lambda_launch.py` forwards `DOCKER_IMAGE`; bootstrap picks this up instead of the branch default.

To run `thai-dev` with the working `main` environment, set `DOCKER_IMAGE` to
exactly the same image tag or digest used by the successful `main` run, and
pass `--git-branch thai-dev` (plus `--git-repo` for the fork). The image supplies
dependencies; the selected Git branch supplies training and evaluation code
through the `/workspace` bind mount. Commit and push code changes to that remote
branch before launching. Restoring the Dockerfile locally does not change an
already-published `thai-dev` image.

## How it works (flow)

```
local                                        lambda cloud
─────                                        ────────────
lambda_launch.py
  ├─ POST /instance-operations/launch   ───▶ provisioning
  ├─ poll GET /instances/<id>           ◀── status=active, ip=...
  ├─ ssh-wait
  ├─ scp lambda_bootstrap.sh
  ├─ scp lambda_push_to_hf.py
  ├─ scp job.env (secrets + params)
  ├─ tmux new -s train 'bash bootstrap' ───▶  clone repo
  │                                          install deps
  │                                          run CPT.py or main.py
  │                                          lambda_push_to_hf.py --> HF Hub
  │                                          touch DONE_SUCCESS|DONE_FAILED
  ├─ poll (tail train.log + check sentinel)
  └─ POST /instance-operations/terminate ──▶ instance gone
```

## Failure modes & what to do

| Symptom | What to check |
|---|---|
| "no region has capacity for gpu_2x_a100" | Lambda is out of stock for that type. Retry later or use `--instance-type gpu_1x_a100_sxm4`. |
| SSH never comes up | `lambda_launch.py` waits 5 min. Check the Lambda dashboard — instance may be stuck provisioning. |
| Training hangs, you Ctrl+C locally | The remote job keeps running in tmux. The script proceeds to terminate anyway. Pass `--keep-alive` if you want to inspect. |
| `lambda_launch.py` crashes mid-run | **Check the Lambda dashboard and terminate the instance manually** — otherwise you keep paying. |
| HF push fails (token scope) | Ensure `HF_TOKEN` has write access to both target repos, and the repo namespace matches your account. |

## Cost guardrails
- Default flow always terminates the instance — even on `KeyboardInterrupt` and
  `DONE_FAILED`. The only path that *doesn't* terminate is `--keep-alive`.
- If the orchestrator itself dies (e.g. machine reboots), instances will keep
  running. Always double-check the Lambda dashboard after an unusual exit.
