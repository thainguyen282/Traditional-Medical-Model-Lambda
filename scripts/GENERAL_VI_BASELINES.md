# Vietnamese general benchmark baselines

The first Lambda run was interrupted after three complete Qwen benchmarks.
See [the October 6 results report](../REPORT_GENERAL_VI_20261006.md).

The runner evaluates Qwen3-4B-Instruct-2507, Gemma-4-E2B-it, and the
`huyennkai/Venera-CPT-TMModel@main` LoRA adapter on four Vietnamese splits.
The CPT manifest pins both adapter and base model commits. All checkpoints
run sequentially with NF4 quantization, BF16 compute, greedy decoding, and
automatic batch reduction on CUDA out of memory.

| Benchmark | Source and split | Rows | Score |
| --- | --- | ---: | --- |
| MLogiQA | `Qwen/P-MMEval`, `mlogiqa/test/vi.jsonl` | 80 | Choice accuracy |
| INCLUDE | `CohereLabs/include-base-44`, Vietnamese test | 550 | Choice accuracy |
| MT-AIME24 | `amphora/MCLM`, MT-AIME2024 test, `vi` column | 30 | Math accuracy |
| PolyMath | `Qwen/PolyMath`, `vi` low/medium/high/top | 500 | Math accuracy |

MT-AIME24 is not part of P-MMEval. The Qwen3 technical report identifies
MCLM as its source. Math answers must be placed in `\boxed{...}`. The runner
uses the first boxed answer and the `math_equal` function from the pinned
[PolyMath evaluator](https://github.com/QwenLM/PolyMath/blob/fbf4e41cae78687d6be7447dbea897357c06aaa7/eval/scripts.py).
MCLM does not supply an executable scorer; its answers use this same math
equality function. This protocol does not claim to reproduce Qwen3's
published numbers: the report does not specify its full scoring parser or
decoding settings.

Each output has `responses.jsonl`, `summary.json`, and `run_config.json`.
Manifests contain immutable dataset and model revisions, source SHA256 hashes,
row counts, scorer revision, and decoding settings. `comparison.csv` and
`comparison.md` are written only after all twelve full runs complete.

## Dataset check without a GPU

```bash
.venv-baselines/bin/python scripts/eval_general_vi.py --dry-run
```

This downloads and validates the Vietnamese evaluation data but does not
load model weights. Install the isolated environment first with
`bash scripts/setup_baseline_env.sh`.

## Local GPU smoke and full runs

```bash
.venv-baselines/bin/python scripts/eval_general_vi.py \
  --limit 4 --batch-size 1 --output-dir EVAL/outputs/general_vi_smoke
.venv-baselines/bin/python scripts/eval_general_vi.py \
  --batch-size 4 --output-dir EVAL/outputs/general_vi_full
```

Use new output directories for retries. A limited run does not produce a
comparison table.

## Lambda GPU

```bash
.venv/bin/python scripts/lambda_launch.py --task eval --general-vi \
  --eval-batch-size 4 \
  --git-repo https://github.com/thainguyen282/Traditional-Medical-Model-Lambda.git \
  --git-branch thai-dev
```

The launcher validates model access and datasets before provisioning, then
copies the exact local runner and requirements to the instance. It uploads
artifacts to the existing evaluation branch and downloads a local backup
before terminating. For a paid short trial, add `--eval-limit 4`.
The remote Git branch must contain compatible bootstrap code.
