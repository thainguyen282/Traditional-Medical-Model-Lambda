# Local Thinking CPT-only evaluation

Run from the repository root:

```bash
bash scripts/run_cpt_general_vi_local.sh --preflight-only
bash scripts/run_cpt_general_vi_local.sh \
  --output-dir EVAL/outputs/general_vi_cpt/local-full-20261020
```

The launcher uses the installed `.venv-vllm-slurm` environment (vLLM 0.31.0),
repository Hugging Face cache, and the current `CUDA_VISIBLE_DEVICES` assignment.
On n0002, CUDA device 0 currently resolves to an A100 MIG 3g.40gb (40,192 MiB).
Keep your scheduler/device assignment; no GPU reconfiguration is needed.
Set `VLLM_ENV` to use a different installed evaluation environment.

For a background run:

```bash
mkdir -p logs
nohup bash scripts/run_cpt_general_vi_local.sh \
  --output-dir EVAL/outputs/general_vi_cpt/local-full-20261020 \
  > logs/cpt-general-vi-local-20261020.log 2>&1 &
```

Use a new output directory for every run. Existing benchmark outputs are refused.
A short GPU check with the same generation settings is:

```bash
bash scripts/run_cpt_general_vi_local.sh --benchmarks mlogiqa --limit 1 \
  --output-dir EVAL/outputs/general_vi_cpt/local-smoke-new
```

## Checkpoint and protocol

The model alias `qwen3-4b-thinking-cpt` applies exactly one CPT-only adapter:

- `Maverfrick/Venera-CPT-TMModel@e0589a27e7b66bcbd9cfd8b709c0785d18bf0bf5`
- Declared base: `unsloth/Qwen3-4B-Thinking-2507@fb56efbd0f60c5bcd531901fa5fe3ee3157b5135`

This is the CPT checkpoint used as input to exp0015 SFT. The final CPT-SFT
checkpoint is a separate model. The older alias `qwen3-4b-cpt` refers to an
Instruct checkpoint and should not be used for this Thinking CPT evaluation.

Defaults match the saved Qwen3-4B-Thinking-2507 general Vietnamese evaluation:
BF16 without quantization, thinking enabled, seed 3407, temperature 0.6,
top-p 0.95, top-k 20, input budget 4,096, output budget 32,768,
8 active sequences, GPU memory utilization 0.70, and eager execution.
The same existing prompts, row loaders, answer parsers, and pinned math scorer
are used. Invalid/truncated answers count as incorrect.

All four benchmarks run sequentially in separate processes: mLogiQA (80),
INCLUDE (550), MT-AIME24 (30), and PolyMath (500), totaling 1,160 questions.
The full run can take hours because of the long reasoning budget.
Preflight checks access, validates adapter base/rank, caches the adapter,
and validates each dataset. Dataset revisions and hashes are recorded at run
time. Setup validation on October 20 confirmed that all four dataset revisions,
file hashes, and row counts match the saved base run
`tm-eval-1792338556`; compare them again if running after dataset updates.
The historical baseline uses the official Qwen repository, while this adapter
uses its declared Unsloth base; the protocol is matched, but repository
provenance differs.

## Outputs

The output root contains per-benchmark logs. Under
`qwen3-4b-thinking-cpt/<benchmark>/` are `run_config.json`, `responses.jsonl`,
`summary.json`, and `infer_result.txt`. Successful full runs also produce
`comparison.csv` and `comparison.md` for the CPT model. Limited checks do not
export comparison tables. Nothing is uploaded automatically.

Follow progress with `tail -f <output-dir>/mlogiqa.log`, then the other
benchmark logs. The launcher reports failed benchmark names and exits nonzero
if any benchmark fails.
