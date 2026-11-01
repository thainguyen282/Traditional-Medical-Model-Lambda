# Qwen3.5 Vietnamese general evaluation

Run from the repository root. This compares the original `Qwen/Qwen3.5-4B`
with the **mixed CPT-only** adapter (no SFT), using the existing Vietnamese
loaders, prompts, and pinned PolyMath math scorer.

| Model | Repository | Immutable revision |
|---|---|---|
| Original | Qwen/Qwen3.5-4B | 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a |
| Mixed CPT | Maverfrick/Qwen3.5-4B-TradMed-CPT | cd864d7e87eacd1c4c9d01e0ba9a5a686ac7d7b1 |

The adapter's declared base revision and rank are validated before evaluation.
Both models load the same original base weights/tokenizer. This setup uses
NF4 double quantization with BF16 compute, native Transformers/PEFT,
thinking enabled, temperature 0.6, top-p 0.95, top-k 20, and batch size 1.
Each question resets the seed to 3407 + row index for both models. Prompt
budget is 4,096 tokens; output budget is 32,768. Overlong inputs fail rather
than being silently truncated. Only the answer after `</think>` is scored;
missing final answers and length-limited generations count as incorrect.
This is a matched internal comparison, not a reproduction of published scores.

| Benchmark | Vietnamese rows per model | Metric |
|---|---:|---|
| MLogiQA | 80 | Choice accuracy |
| INCLUDE | 550 | Choice accuracy |
| MT-AIME24 | 30 | Math accuracy |
| PolyMath | 500 | Math accuracy |

All 1,160 questions run for each model (2,320 total generations). Dataset
commits and source file hashes resolve once per invocation. Separate jobs
record their own manifests; compare dataset revisions and hashes before comparing scores. Actual row counts are logged during preflight.

## Check access without GPU weights

Use an environment with `requirements-baselines.txt` installed. Existing
`.env` credentials are loaded without printing them.

```bash
.venv/bin/python scripts/eval_qwen35_general_vi.py --preflight-only \
  --output-dir EVAL/outputs/qwen35_general_vi/preflight
```

## Slurm smoke and full run

One submission creates **two independent Slurm array jobs**: task 0 evaluates
`qwen35-base`, and task 1 evaluates `qwen35-mixed`. Each job evaluates all four
benchmarks for its checkpoint, with thinking enabled, and requests its own A100.
They can run concurrently when resources are available.

Each job installs the pinned inference dependencies in a temporary environment
on its allocated GPU node, preserving training environments. Ensure `logs/`
exists before submission. Arguments after the job filename go to the evaluator.

```bash
mkdir -p logs
sbatch scripts/slurm/eval_qwen35_general_vi.sbatch \
  --limit 1 --max-new-tokens 512
sbatch scripts/slurm/eval_qwen35_general_vi.sbatch
```

Inspect both smoke jobs before submitting the full evaluation. Its short output budget
checks loading/generation only; scores are diagnostic. Each full job requests
one A100 for up to 24 hours; long reasoning can still exceed this allocation.
The runner preserves scheduler CUDA device assignments. Nothing uploads
automatically.

For an existing GPU environment:

```bash
/path/to/eval-env/bin/python scripts/eval_qwen35_general_vi.py \
  --output-dir EVAL/outputs/qwen35_general_vi/full-unique
```

Use `--models qwen35-base` or `--models qwen35-mixed` to run one model, and
`--benchmarks mlogiqa include` to restrict tasks. Always use a new output path.
For Slurm, checkpoint selection comes from the array task; do not pass
`--models`. To submit only one checkpoint, use `sbatch --array=0` for the
original or `sbatch --array=1` for CPT, followed by the job script path.
Slurm results are grouped under
`EVAL/outputs/qwen35_general_vi/slurm-<array-job-id>/<model>/`.
Logs are `logs/qwen35-general-vi-<array-job-id>_<task-id>.out`.
Each job writes its own comparison files for its one checkpoint.
Each model/benchmark saves `run_config.json`, flushed `responses.jsonl`, and
`summary.json`, including invalid and truncation counts. Successful unlimited
runs export `comparison.csv` and `comparison.md`; limited runs never export
comparison tables. Failed benchmarks retain a failed manifest and partial
responses; no final comparison is produced.
