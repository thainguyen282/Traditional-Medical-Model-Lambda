# CPT adapters for Qwen3 Thinking-2507 and Gemma 4 E2B

These are separate text-only LoRA adapters trained on the same
`Maverfrick/TradMed-CPT-53k-2048` train split, with its validation split held
out. They do not modify the existing `Venera-CPT-TMModel@main` adapter.

The Gemma model is `google/gemma-4-E2B-it`. There is no checkpoint named
Gemma 4 A2B in the model family; if a different model is intended, change the
model choice before starting a paid run.

## Lambda commands

From the repository root, start each job separately:

```bash
.venv/bin/python scripts/lambda_launch.py --task cpt \
  --cpt-new-model qwen3-4b-thinking-2507 \
  --git-repo https://github.com/thainguyen282/Traditional-Medical-Model-Lambda.git \
  --git-branch thai-dev

.venv/bin/python scripts/lambda_launch.py --task cpt \
  --cpt-new-model gemma4-e2b \
  --git-repo https://github.com/thainguyen282/Traditional-Medical-Model-Lambda.git \
  --git-branch thai-dev
```

The launcher checks model, dataset, Git branch, and target Hugging Face repo
access before provisioning. It copies this training script to the instance,
installs an isolated Transformers 5.5 environment, and trains one epoch with
NF4 QLoRA, sequence length 2048, microbatch 2, gradient accumulation 8, and
seed 3407. Training uses raw text without a chat template. The Gemma LoRA
targets only the language model projections; vision and audio weights stay
frozen.

Each successful job uploads its adapter, `run_config.json`, and `metrics.json`
to `HF_CPT_REPO` on the unique branch `<model-key>-<job-tag>`. The original
`main` branch is left untouched. The branch name appears in the job log.
For a paid two-step smoke trial, add `--cpt-max-steps 2` to a Lambda command;
it creates its own unique branch and should not be treated as a trained adapter.

## Check the plan without a GPU

```bash
python CPT/train_new_bases.py --model qwen3-4b-thinking-2507 --dry-run
python CPT/train_new_bases.py --model gemma4-e2b --dry-run
```

For a local CUDA training run, install `requirements-cpt-new.txt` in an
isolated environment and run `CPT/train_new_bases.py --model <model-key>`.
Add `--max-steps 2` for a short smoke test and a fresh `--output-dir` for
each invocation. Local runs save the adapter but upload only when both
`--push-repo` and a new `--push-revision` are supplied.
