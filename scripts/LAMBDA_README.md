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

### Evaluate MedGemma-1.5-4B and HuatuoGPT-o1-8B

Run both models sequentially on the usual Vietnamese MCQ benchmarks:

```bash
HF_SFT_REPO=Maverfrick/Venera-Traditional-Model \
python scripts/lambda_launch.py --task eval \
    --medical-baselines medgemma huatuo \
    --instance-type gpu_1x_a100_sxm4
```

Select just `huatuo` or `medgemma` to evaluate one model. Add `--eval-limit 4`
for a smoke run (four questions from each dataset); omit it for a full run.
`--eval-batch-size 4` is the default, with automatic reduction down to one on OOM.
`--eval-generation-mode direct` is now the default: generate only `Final Answer: X`,
without a reasoning section. Existing launch commands use it automatically.
Use `--eval-generation-mode notebook` to reproduce the previous free-generation protocol.
For direct GPU execution, the corresponding flag is `--generation-mode direct|notebook`.

Models: `google/medgemma-1.5-4b-it` and
`FreedomIntelligence/HuatuoGPT-o1-8B`. The HF account used by `HF_TOKEN` must
first accept the terms at <https://huggingface.co/google/medgemma-1.5-4b-it>.
The launcher checks access before allocating a GPU. Existing Lambda/SSH and
Docker configuration from the setup above still applies.

Protocol:
- `new`: all 105 rows of `EVAL/Data/tcm_mcq.jsonl`.
- `vm14k`: all 564 rows of `EVAL/Data/eastern_medicine_VM14K.jsonl`.
  This is the historical benchmark, not the separate 29-row official test split.
- The existing baseline notebooks supply data normalization, the base MCQ prompt,
  answer extraction, and correctness rules. Both modes use each model's own chat
  template and a 2048-token input limit.
- `direct` adds a no-explanation instruction and prefills the assistant's response
  with `Final Answer:`. Greedy decoding is constrained to the question's valid
  option labels followed by EOS, with a 16-token cap. Reasoning text is never
  generated and then stripped. Long inputs are truncated from the left to retain
  the assistant prefix.
- `notebook` preserves the original prompt and unrestricted greedy generation
  with a 250-token cap. Both models can emit reasoning in this mode; neither
  model's current chat template implements `enable_thinking=False`.
- Transformers loads both models in NF4 with BF16 compute. MedGemma uses
  `Gemma3ForConditionalGeneration` for text-only questions; Huatuo uses
  `AutoModelForCausalLM`. No training or model-weight upload is performed.
- Results record `generation_mode`, the effective prompt and generation settings,
  `assistant_prefix`, and `constrained_to_options`. Direct mode is a different
  benchmark protocol: valid formatting is enforced, so its format deviation rate
  is zero by construction and is not comparable to the old free-generation rate.
  Accuracy can also change. These constraints control output, not model weights
  or an internal reasoning switch. The existing answer parser remains unchanged.

The launcher copies the local evaluator, notebooks, and data to the instance,
so this setup can run before the new files are pushed to GitHub. Each model and
dataset runs in a fresh Docker process using the existing image dependencies.

Outputs include `responses.jsonl`, `summary.json` (accuracy, format deviation,
topic/difficulty breakdowns), `infer_result.txt`, and `run_config.json` with
model revision and input/protocol hashes. Plots and language-adjusted throughput
from the original notebooks are not generated by this runner.

Results are uploaded to `$HF_SFT_REPO` on `eval-medical-<job-tag>` under
`eval/<job-tag>/<model>/<dataset>/`. They are also downloaded locally to
`logs/<job-tag>/medical-eval.tar.gz` before termination, including when Hub upload
fails. If local backup fails or the launcher is interrupted before completion,
the instance remains running and billable so results can be recovered; terminate
it manually after recovery. These protections apply to `--medical-baselines`
runs only. Existing training cleanup behavior is unchanged.

Validate locally without a GPU:

```bash
python scripts/eval_medical_baselines.py --dry-run
python scripts/eval_medical_baselines.py --model huatuo --check-access
python -m unittest discover -s tests -p 'test_medical_baselines.py'
```

For direct GPU execution, use `python scripts/eval_medical_baselines.py
--model huatuo --output-dir EVAL/outputs/my-medical-run`. The runner refuses to
overwrite an existing model/dataset directory; choose a new output directory
for a repeat evaluation. Direct execution does not upload results.

### Additional open-source baselines

The same evaluator also supports these Apache-2.0 instruct checkpoints:

| Alias | Checkpoint |
|---|---|
| `qwen3-4b` | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) |
| `qwen2.5-7b` | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| `mistral-7b` | [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) |

These compare the clean Qwen3 Instruct checkpoint with two 7B general-purpose
baselines. All use `AutoModelForCausalLM`, their own chat templates, and the
existing NF4/direct-answer protocol. Qwen3-4B-Instruct-2507 is a non-thinking
checkpoint. This is evaluation only: there are no training epochs.

Run a smoke experiment (4 questions per dataset, 24 predictions total):

```bash
python scripts/lambda_launch.py --task eval \
    --baselines qwen3-4b qwen2.5-7b mistral-7b \
    --eval-limit 4 --instance-type gpu_1x_a100_sxm4
```

For the full experiment, omit `--eval-limit 4`: 105 + 564 questions per model,
2,007 predictions across the three models. `--baselines` and the original
`--medical-baselines` flag are aliases; either accepts all five model names.
The evaluator's `--model all` now includes all five baselines. Use an explicit
model selection when reproducing the original two-model run.

Validate data locally without allocating a GPU:

```bash
python scripts/eval_medical_baselines.py --model qwen3-4b --dry-run
python -m unittest discover -s tests -p 'test_medical_baselines.py'
```

Outputs, Hub uploads, local backup, and instance cleanup follow the medical
baseline workflow above. The three models run sequentially on a single GPU;
no Docker dependency changes are needed for the pinned Transformers 4.57.6.
Compare results only against runs with the same generation mode and dataset
hashes. The historical VM14K set has substantial overlap with local training
data (see `REPORT_EXP0013_0014.md`); it is not an independent held-out benchmark
for the fine-tuned models.

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
`lambda_launch.py` forwards all env vars; bootstrap picks this up instead of the branch default.

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
