# Vietnamese general benchmark baselines

Completed measured run: [October 9 MLogiQA report](../markdown/REPORT_MLOGIQA_20261009.md).

## All four benchmarks with vLLM

The vLLM backend supports MLogiQA, INCLUDE, MT-AIME24, and PolyMath.
Use the existing model aliases `qwen3-4b-thinking-2507` and `gemma4-e2b`
for `Qwen/Qwen3-4B-Thinking-2507` and `google/gemma-4-E2B-it`.
Each model/benchmark pair runs in a fresh process. Both models use BF16,
temperature 0.6, top-p 0.95, top-k 20, seed 3407, and the same output budget.
Math scoring uses the existing pinned PolyMath boxed-answer/equality scorer;
only the final answer after thinking is scored. Length-limited generations
count as invalid and incorrect.

```bash
.venv/bin/python scripts/lambda_launch.py --task eval --general-vi \
  --instance-type gpu_1x_a100_sxm4 \
  --eval-backend vllm --eval-protocol thinking \
  --eval-models qwen3-4b-thinking-2507 gemma4-e2b \
  --eval-datasets mlogiqa include mt-aime24 polymath \
  --eval-batch-size 8 --eval-gpu-memory-utilization 0.70 \
  --eval-enforce-eager --eval-max-new-tokens 32768 --eval-limit 0
```

This is eight evaluations over 1,160 Vietnamese questions per model.
The 32,768-token limit allows long math reasoning and can make the run lengthy.
Use `--eval-limit 2` for an initial GPU trial. Limited trials skip comparison
export. Full runs export comparison tables for exactly the requested datasets.
The launcher transfers local evaluator code and setup scripts, uploads results
and `train.log` to `HF_SFT_REPO`, and downloads the evaluation archive under
`logs/<job-tag>/medical-eval.tar.gz` before cleanup.

## Existing CPT adapter on MLogiQA with vLLM

`qwen3-4b-cpt` loads `huyennkai/Venera-CPT-TMModel@main` as a rank-64
rsLoRA adapter on its declared base, `unsloth/Qwen3-4B-Instruct-2507`.
The manifest records the resolved adapter and base commits. This is an
instruction checkpoint, so it requires `--eval-protocol instruct`; its
final answer does not require a closing thinking tag. Sampling and output
budget can match the thinking baselines, but this is not a controlled
before/after CPT comparison to Thinking-2507.

```bash
.venv/bin/python scripts/lambda_launch.py --task eval --general-vi \
  --instance-type gpu_1x_a100_sxm4 \
  --eval-backend vllm --eval-protocol instruct \
  --eval-models qwen3-4b-cpt --eval-datasets mlogiqa \
  --eval-batch-size 8 --eval-gpu-memory-utilization 0.60 \
  --eval-enforce-eager --eval-max-new-tokens 8192 --eval-limit 0 \
  --git-repo https://github.com/thainguyen282/Traditional-Medical-Model-Lambda.git \
  --git-branch thai-dev
```

## Faster MLogiQA thinking evaluation with vLLM

This runs the same 80 Vietnamese MLogiQA rows on
`Qwen/Qwen3-4B-Thinking-2507` and `google/gemma-4-E2B-it`. vLLM queues all
questions and uses continuous batching, with up to 8 active sequences on the
40 GB Lambda GPU. The
progress bar updates as individual answers finish. Both models use their
thinking chat templates and sampling at temperature 0.6, top-p 0.95, top-k
20. Each model runs in a separate process so GPU memory is released before
the next model loads.

```bash
.venv/bin/python scripts/lambda_launch.py --task eval --general-vi \
  --eval-backend vllm --eval-protocol thinking \
  --eval-models qwen3-4b-thinking-2507 gemma4-e2b \
  --eval-datasets mlogiqa --eval-batch-size 8 \
  --eval-gpu-memory-utilization 0.70 \
  --eval-max-new-tokens 4096 \
  --git-repo https://github.com/thainguyen282/Traditional-Medical-Model-Lambda.git \
  --git-branch thai-dev
```

The 4,096-token cap makes the run faster but differs from the earlier
32,768-token Transformers thinking protocol; length-limited answers count as
invalid and incorrect. For the same token cap as that run, set
`--eval-max-new-tokens 32768`, which can take much longer. vLLM uses BF16
model weights, while the earlier run used NF4, so scores are matched between
the two models in this vLLM run but may differ from the earlier run.

Full runs write `comparison.csv`, `comparison.md`, per-model `summary.json`,
`responses.jsonl`, and manifests. Lambda uploads the entire directory to
`HF_SFT_REPO` branch `eval-medical-<job-tag>` and downloads a backup under
`logs/<job-tag>/medical-eval.tar.gz`. For a four-question trial, add
`--eval-limit 4`; limited runs do not write a comparison table.

## MLogiQA Vietnamese thinking run (80 questions)

This focused run compares the dedicated `Qwen/Qwen3-4B-Thinking-2507`
checkpoint with `google/gemma-4-E2B-it`. Both use their thinking chat
templates. Qwen uses its recommended thinking sampling settings (temperature
0.6, top-p 0.95, top-k 20, up to 32,768 output tokens). Gemma receives the
same sampling settings for a matched run; it is not a model in the Qwen3
report. Batch size starts at 24 and halves automatically on CUDA OOM.

```bash
.venv/bin/python scripts/lambda_launch.py --task eval --general-vi \
  --eval-models qwen3-4b-thinking-2507 gemma4-e2b \
  --eval-datasets mlogiqa --eval-protocol thinking \
  --eval-batch-size 24 \
  --git-repo https://github.com/thainguyen282/Traditional-Medical-Model-Lambda.git \
  --git-branch thai-dev
```

The launcher downloads a `logs/<job-tag>/medical-eval.tar.gz` backup. Inside
are `qwen3-4b-thinking-2507/mlogiqa/summary.json` and
`gemma4-e2b/mlogiqa/summary.json`, plus all per-question responses and a
`comparison.csv`/`comparison.md`. The same files are uploaded to the
`HF_SFT_REPO` Hugging Face repository on branch `eval-medical-<job-tag>` under
`eval/<job-tag>/`. The
paper reports 63.7% for the earlier Qwen3-4B in thinking mode on Vietnamese
MLogiQA; 68.8% is the Qwen3-14B thinking score. The 2507 checkpoint was
released later, so its result is not directly comparable to the paper's row.

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

## Paper-style Qwen/CPT comparison

The `paper-style` profile evaluates the Qwen3-4B checkpoint named in
[Table 33 of the Qwen3 report](https://arxiv.org/html/2505.09388), the
later Qwen3-4B-Instruct-2507 baseline, and our CPT adapter on the same
Vietnamese rows. It samples 55 of 550 INCLUDE test rows using seed 3407,
and uses the paper's published non-thinking settings: temperature 0.7,
top-p 0.8, top-k 20, presence penalty 1.5, and up to 32,768 output tokens.
The custom presence penalty applies to tokens generated in the current
answer. Model and dataset commits, sampled row indices, and decoding
settings are stored in every run manifest.

The paper does not publish its exact INCLUDE row indices, prompts, or
full scorer configuration. This is a matched comparison among the three
specified checkpoints, **not** an exact reproduction of Table 33.
The 2507 and CPT checkpoints are not the paper's Qwen3-4B checkpoint.
The CPT adapter uses `unsloth/Qwen3-4B-Instruct-2507` as its pinned base;
the standalone 2507 baseline uses the official Qwen repository.

Check the selected data without a GPU:

```bash
.venv-baselines/bin/python scripts/eval_general_vi.py \
  --protocol paper-style --dry-run
```

Run a short local smoke test, then a full evaluation in a new directory:

```bash
.venv-baselines/bin/python scripts/eval_general_vi.py \
  --protocol paper-style --limit 4 --batch-size 1 \
  --output-dir EVAL/outputs/general_vi_paper_smoke
.venv-baselines/bin/python scripts/eval_general_vi.py \
  --protocol paper-style --batch-size 1 \
  --output-dir EVAL/outputs/general_vi_paper_full
```

Lambda launch (GPU provisioning is not part of adding this profile):

```bash
.venv/bin/python scripts/lambda_launch.py --task eval --general-vi \
  --eval-protocol paper-style --eval-batch-size 1 \
  --git-repo https://github.com/thainguyen282/Traditional-Medical-Model-Lambda.git \
  --git-branch thai-dev
```

The 32,768-token cap makes a full math run much longer than the legacy
profile. The new profile writes its own comparison table only when all
twelve runs are complete.
