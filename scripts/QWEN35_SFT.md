# Qwen3.5 mixed CPT → SFT on Lambda

This setup continues the existing mixed-CPT LoRA adapter once; it does not add
another adapter or restart from the untuned base. No instance is allocated by
setup or preflight.

## Pinned inputs and recipe

- Base: `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- CPT: `Maverfrick/Qwen3.5-4B-TradMed-CPT@cd864d7e87eacd1c4c9d01e0ba9a5a686ac7d7b1`.
- SFT: `Maverfrick/yhct-merged-vm14k-clean@60ba87821f4e58b183d2ca8f5c926ab2aa86d2f6`.
  All 10,344 rows; 5% validation split, seed 42, as in exp0015. Checksums and the
  published VM14K question/ID-cleaning audit are verified before allocation.
  The audit does not establish freedom from semantic/paraphrase contamination.
- Same answer-only medical prompt and targets `Final Answer: X`; native chat
  template with thinking disabled. Only answer/EOS tokens contribute to loss.
  Input limit 2048; left-truncate only the prompt, never the answer.
- Inherit rank/alpha/targets from CPT. NF4 double quantization, BF16 compute,
  SDPA attention, frozen base/vision. Native Transformers/PEFT, not Unsloth.
- Exp0015 recipe: 3 epochs, LR 5e-5, weight decay .01, 100 warmup steps, cosine,
  8-bit AdamW, seed 3407, NEFTune alpha 5, validation/save every 100 steps,
  keep two checkpoints and select the best validation-loss checkpoint.
- Per-device batch 1 × accumulation 64 preserves effective batch 64 from the
  prior batch 8 × accumulation 8 recipe; evaluation batch is reduced to 1.
  This batch choice still requires a real GPU smoke test.

## Preflight (no GPU, no allocation)

Run from the repository root with the existing `.env` credentials:

```bash
.venv/bin/python scripts/train_qwen35_sft.py \
  --config configs/experiments/qwen3.5-4B/sft/mixed.yaml --mode preflight
```

## Smoke launch (allocates a paid GPU)

```bash
DOCKER_IMAGE=ghcr.io/thainguyen282/traditional-medical-model-lambda:thai-dev \
.venv/bin/python scripts/lambda_launch.py --task sft \
  --config configs/experiments/qwen3.5-4B/sft/smoke.yaml \
  --instance-type gpu_1x_a100_sxm4 \
  --git-repo https://github.com/thainguyen282/Traditional-Medical-Model-Lambda.git \
  --git-branch thai-dev --boot-timeout 600
```

The smoke job runs two optimizer steps on a tiny data subset, saves checkpoints,
selects/saves the best adapter, uploads it to a separate private branch, verifies
its remote checksum, and runs four direct-answer VM14K questions. It does not
produce a full VM14K score. Proceed only after this job succeeds.

## Full launch (allocates a paid GPU)

Use the same command with:

```text
--config configs/experiments/qwen3.5-4B/sft/mixed.yaml
```

No Git push is needed: the launcher bundles and hashes the exact local code and
configuration, installs the pinned isolated Qwen3.5 environment on Lambda, and
checks input access before allocation. A missing-capacity error allocates no GPU.
Keep the launcher alive until backup and cleanup finish.

## Artifacts and lifecycle

1. Periodic checkpoints and the selected final adapter are saved under
   `group_outputs/qwen35_sft/<experiment-name>-<JOB_TAG>/` on Lambda.
2. Before evaluation starts, the adapter, tokenizer and training metadata upload
   to private `Maverfrick/Qwen3.5-4B-TradMed-CPT`, branch
   `<experiment-name>-<JOB_TAG>`; the uploaded weights checksum is verified.
   W&B uses project `qwen35-sft` by default.
3. A fresh process reuses the existing direct-answer VM14K evaluator: four smoke
   questions, then all 564 for full jobs. Thinking is disabled, greedy output is
   constrained to labels/EOS, maximum 16 tokens, NF4/BF16, batch 4, seed 3407.
   This applies the continued CPT-SFT adapter once to the pinned base. Evaluation
   question text/predictions are kept local, not uploaded automatically.
4. The launcher downloads `logs/<JOB_TAG>/sft-backup.tar.gz` before terminating
   the instance. If completion is uncertain or backup fails, it preserves the
   instance (billing continues). A Ctrl+C during an SFT job preserves the host.

The setup passed local input preflight and masking/loader/bundle/order tests.
The optional native CPU test additionally checks saved-adapter continuation:

```bash
QWEN35_NATIVE_TEST=1 .venv-qwen35-check/bin/python -m unittest discover \
  -s tests -p test_qwen35_native.py
```

These checks do not establish full-model GPU memory use, throughput, or SFT
accuracy; the paid smoke run is still required.
