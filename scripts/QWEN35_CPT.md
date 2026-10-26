# Qwen3.5 CPT: developer review and launch commands

These commands run from the repository root with the existing `.env` credentials and `.venv`. Each command allocates a paid Lambda instance. No Lambda job has been launched as part of the local verification.

## Prepared private datasets

| Condition | Dataset revision | Train rows | Train tokens | Validation rows / tokens |
|---|---|---:|---:|---:|
| TradMed | `Maverfrick/TradMed-CPT-Qwen3.5-2048@68ad96858975ce523de7468d8983829c84c153bc` | 12,391 | 25,058,749 | 914 / 1,854,248 |
| 80/20 | `Maverfrick/TradMed-CPT-80-20-Qwen3.5-2048@a6e7c0e5a51318526fe117cf230394f0765dc2e0` | 18,243 | 31,323,437 | 1,267 / 2,439,219 |

Both use the tokenizer from `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The medical rows and validation boundaries are identical between conditions. Mixed training tokens: medical 25,058,749; FineWeb-Edu 3,758,815; Dolma 1,252,937; OpenWebMath 626,468; license-filtered Stack-dedup Python 626,468. Original raw source checkpoints were reused and retokenized, preserving code indentation. No thinking traces or instruction SFT are added. The earlier Qwen3-tokenized upload is unchanged.

Each full job runs **one epoch of its own dataset**, as requested. The mixed run therefore processes approximately 25% more tokens. This is not a compute-matched comparison. Counts above are corpus tokens; the training summary records actual processed non-padding tokens. Packing keeps the remainder; a single-token remainder reuses one preceding context token with its label masked.

## Launch

Run the GPU smoke job first (two optimizer steps and two VM14K questions, 512 generated tokens). Smoke scores are diagnostic only. The smoke job verifies GPU dependencies, training, adapter loading, evaluation, uploads and cleanup.

```bash
DOCKER_IMAGE=ghcr.io/thainguyen282/traditional-medical-model-lambda:thai-dev \
.venv/bin/python scripts/lambda_launch.py \
  --task cpt \
  --config configs/experiments/qwen3.5-4B/smoke.yaml \
  --instance-type gpu_1x_a100_sxm4 \
  --git-repo https://github.com/thainguyen282/Traditional-Medical-Model-Lambda.git \
  --git-branch thai-dev \
  --boot-timeout 600
```

TradMed-only, one epoch:

```bash
DOCKER_IMAGE=ghcr.io/thainguyen282/traditional-medical-model-lambda:thai-dev \
.venv/bin/python scripts/lambda_launch.py \
  --task cpt \
  --config configs/experiments/qwen3.5-4B/exp001/config_cpt.yaml \
  --instance-type gpu_1x_a100_sxm4 \
  --git-repo https://github.com/thainguyen282/Traditional-Medical-Model-Lambda.git \
  --git-branch thai-dev \
  --boot-timeout 600
```

80/20, one epoch:

```bash
DOCKER_IMAGE=ghcr.io/thainguyen282/traditional-medical-model-lambda:thai-dev \
.venv/bin/python scripts/lambda_launch.py \
  --task cpt \
  --config configs/experiments/qwen3.5-4B/mixed.yaml \
  --instance-type gpu_1x_a100_sxm4 \
  --git-repo https://github.com/thainguyen282/Traditional-Medical-Model-Lambda.git \
  --git-branch thai-dev \
  --boot-timeout 600
```

`--keep-alive` is omitted: the launcher downloads the run backup before automatically terminating the instance. Exact local implementation/configuration files are bundled, so these changes do not require a Git push before launch. Repository access and pinned private dataset access are checked before allocation. Configs reference published prepared data; Lambda does not prepare it again.

## Changes for review

- `CPT/train_qwen35.py`, `CPT/qwen35_common.py`: isolated native Transformers/PEFT implementation. The restored `CPT/CPT.py` and its dependencies remain unchanged. Preserve rank/alpha 64, rsLoRA, batch 2 × accumulation 8 (effective batch 16), AdamW 8-bit, LR 3e-5, cosine schedule, 3% warmup, BF16 and 2,048-token packing. Qwen3.5 hybrid attention requires its additional linear-attention projection targets; vision weights remain frozen. Raw code whitespace is retained.
- `requirements-qwen35.txt`: pinned separate environment, installed in the Lambda container at `/workspace/.venv-qwen35`; no Unsloth dependency. Native PyTorch attention fallback is supported; optional fast linear-attention kernels are not installed. This can affect throughput.
- `scripts/lambda_launch.py`, `scripts/lambda_bootstrap.sh`: forward the selected config, verify and install an exact local source bundle, execute the dedicated trainer/evaluator, and download backups before cleanup. Bundle SHA and individual file hashes accompany the Git revision, including uncommitted local changes. Training, evaluation and required upload errors fail the job. Diagnostic upload is best effort on failure. Local temporary bundle/env files use repository `.cache/lambda`.
- `CPT/eval_qwen35_vm14k.py`: load the current run's local final adapter and pinned base only; bypass legacy multi-dataset evaluation. Evaluate the frozen 564-question traditional-medicine VM14K subset with native thinking enabled, temperature .6, top-p .95, top-k 20, seed 3407, input budget 2,048, generation budget 32,768. Invalid/truncated outputs are incorrect. Final answers are parsed after `</think>`.
- `configs/experiments/qwen3.5-4B/{tradmed,mixed,smoke}.yaml`: independent configs with pinned model/data commits. Full jobs evaluate final adapters only; validation loss uses up to 128 packed validation blocks and does not select checkpoints.
- `configs/data/cpt_mix_qwen35.yaml`, `scripts/prepare_cpt_mix.py`: standalone local data preparation and approved private dataset upload. Existing grouped splits, seed 42, exact/MinHash deduplication and VM14K question/exclusion filtering are retained. This is not proof of complete source-document or answer-paraphrase decontamination; that broader audit remains outstanding.

Adapters, tokenizer, resolved configuration, dataset preparation manifest, processed-token counts, dependency versions, VM14K predictions/summary and logs upload to private `Maverfrick/Qwen3.5-4B-TradMed-CPT`, on separate branches named `<experiment-name>-<JOB_TAG>`. This model repository is a configured destination for future training jobs; no trained model has been uploaded by the local checks. Local backups are `logs/<JOB_TAG>/qwen35-backup.tar.gz`.

## Verification and scope

Local native Qwen3.5 CPU test passed finite forward/backward loss, LoRA gradients, frozen vision, a Trainer optimizer step with gradient checkpointing, and cached generation using the pinned dependencies. This used a tiny randomly initialized architecture, not the full pretrained weights or GPU optimizer. Configuration/source-bundle/failure-propagation tests and 15 data-preparation tests passed. Uploaded private manifests match local hashes; medical rows match exactly across both datasets; the mixed data audit found no train/validation document overlap and no token-count errors.

A full A100 smoke job is still required before the full training runs. No VM14K training results, baseline comparison, paired bootstrap intervals or reasoning-retention claims are available yet. This update provides the requested two training configurations and commands; it does not run the original-model baseline or the previously rolled-back three-condition pilot. VM14K alone measures this medical subset and cannot establish general thinking/instruction retention.

## W&B logging

Qwen3.5 training reports loss, learning rate, gradient norm and scheduled validation loss to W&B. Runs use project `qwen35-cpt` by default, named `<experiment-name>-<JOB_TAG>`. The existing `.env` `WANDB_KEY` is forwarded as `WANDB_API_KEY`; optional `WANDB_PROJECT` and `WANDB_ENTITY` are also forwarded. Launch commands are unchanged. Smoke runs log training metrics but skip validation as configured.

`wandb_run.json` stores the run URL and ID beside the adapter artifacts. W&B local files stay in the run output directory and the context manager finishes/syncs the run after training (including failure). Final VM14K outputs remain in Hugging Face and the downloaded backup; this change logs Trainer metrics to W&B. No training or external logging run was launched during implementation.

## A100 batch configuration

The user-provided smoke log for `tm-cpt-1793022537` showed CUDA OOM during backward with batch 4 / accumulation 4: 37.96 GiB process usage on a 39.49 GiB GPU, with another 7.58 GiB allocation requested. One optimizer step succeeded before the failure, so that step alone did not establish a safe memory peak. W&B synced, diagnostics uploaded, and the launcher downloaded a backup and terminated the instance. This was a failed smoke run, with no final adapter or VM14K results.

TradMed (`exp001/config_cpt.yaml`), mixed and smoke now use `per_device_train_batch_size: 2` and `gradient_accumulation_steps: 8`. Effective batch remains 16 sequences (32,768 packed tokens) per optimizer step on one GPU. Epoch count, learning rate, sequence length and evaluation protocol are unchanged. The user's move of the TradMed configuration into `exp001` is preserved.

Qwen3.5 Lambda containers also default to `PYTORCH_ALLOC_CONF=expandable_segments:True` to help reduce allocator fragmentation. The launcher forwards an explicit environment override if supplied. This is supplementary to lowering the batch; it does not guarantee sufficient memory or a particular speedup. See [PyTorch allocator documentation](https://docs.pytorch.org/docs/stable/notes/cuda.html#environment-variables).

Rerun smoke before full training. Batch 2 / accumulation 8 has passed local config/wiring checks but has not been validated on the A100. If it still fails, batch 1 / accumulation 16 is the fallback. Existing running jobs retain their loaded configuration. Compare steady-state tokens/second to assess throughput; training batching does not accelerate the separate serial VM14K generation loop.
