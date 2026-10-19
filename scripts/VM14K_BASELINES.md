# VM14K baselines without thinking

This experiment evaluates `Qwen/Qwen3-4B-Instruct-2507` and
`google/gemma-4-E2B-it` on the frozen 564-row traditional-medicine subset.
Qwen Instruct-2507 is a non-thinking model. Gemma's processor receives
`enable_thinking=False`. Both use constrained answer-only MCQ evaluation:
the same question/options and instructions, each model's native chat template,
`Final Answer:` assistant prefill, and output restricted to option labels plus EOS.

| Setting | Both models |
| --- | --- |
| Dataset | `vm14k` (`eastern_medicine_VM14K.jsonl`), all 564 rows |
| Quantization | NF4, double quantization, BF16 compute |
| Attention | SDPA |
| Decoding | Greedy, 16 new tokens maximum |
| Input budget | 2,048 tokens |
| Batch size | 4, automatically reduced on CUDA OOM |
| Seed | 3407 |
| Metrics | Accuracy; invalid-answer and token-limit rates as diagnostics |

Invalid-answer rate is not a measure of unconstrained instruction following:
this protocol constrains output to valid labels. Empty/unparsed answers count
as invalid and incorrect. No ROUGE, BLEU, or judge calls are made.

## Local GPU

Create a **separate** environment; the training requirements remain unchanged:

```bash
bash scripts/setup_baseline_env.sh
.venv-baselines/bin/python scripts/eval_medical_baselines.py \
  --vm14k-baselines --check-runtime --check-access
.venv-baselines/bin/python scripts/eval_medical_baselines.py \
  --vm14k-baselines --batch-size 4 --output-dir EVAL/outputs/vm14k_baselines_run1
```

The second command checks dependencies and access without downloading weights.
The final command smoke-tests four questions on **both** models, then evaluates
all 564 questions sequentially. A smoke failure skips the full evaluation.
Models use the same resolved checkpoint revisions for smoke and full runs.
Choose a fresh output directory for each invocation; existing model directories
are never silently overwritten or resumed.

Dataset-only validation needs no GPU or model downloads:

```bash
.venv/bin/python scripts/eval_medical_baselines.py --vm14k-baselines --dry-run
```

For only a short trial, add `--limit 4` to the paired invocation. It produces
per-model diagnostics but never a final comparison table.

## Lambda GPU

Use your existing `.env`, Docker image, Git repository and region settings:

```bash
.venv/bin/python scripts/lambda_launch.py --task eval \
  --baselines qwen3-4b gemma4-e2b \
  --eval-datasets vm14k --eval-generation-mode direct \
  --eval-limit 0 --eval-batch-size 4 \
  --git-repo https://github.com/thainguyen282/Traditional-Medical-Model-Lambda.git \
  --git-branch thai-dev
```

The launcher copies the evaluator, selected data/protocol, environment setup
script and pinned evaluation requirements from this checkout. When Gemma is
selected, bootstrap installs the isolated environment on the instance; no
training image rebuild or Git push is needed. Provisioning still uses the
existing GPU/region flags and cleanup rules. If your desired GPU is unavailable,
use an explicit `--instance-type` and `--region` supported by `--list-types`.

The exact Qwen/Gemma pair with `vm14k` and `direct` selects the same paired
smoke-then-full workflow as the local command. Other baseline selections retain
the existing per-model workflow.

## Artifacts

Each model produces `responses.jsonl`, `summary.json`, `infer_result.txt`, and
`run_config.json`. The manifest records immutable model revisions, dataset and
protocol hashes, dependency versions, thinking settings, quantization, and
requested/effective batch size. Smoke artifacts live under `smoke/`.

The root `comparison.csv` and `comparison.md` contain two rows with model,
correct/564, accuracy percentage, invalid-answer percentage, and token-limit
percentage. Export requires both complete, comparable 564-row runs and verifies
response counts, row identity, summaries, and settings. Limited or failed runs
cannot be presented as final scores. Re-export without inference:

```bash
.venv/bin/python scripts/eval_medical_baselines.py \
  --export-comparison --output-dir EVAL/outputs/vm14k_baselines_run1
```

Lambda uploads the full output directory through its existing Hub workflow and
downloads a local backup to `logs/<job-tag>/medical-eval.tar.gz` before termination.

## Verified run: 2026-10-05

Job `tm-eval-1791208642` passed both four-question smoke checks and completed
both 564-question evaluations on one NVIDIA A10 in `us-east-1`. Effective batch
size remained 4. Both invalid-answer and token-limit rates were 0%.

| Model | Correct / 564 | Accuracy |
| --- | ---: | ---: |
| Qwen3-4B-Instruct-2507 | 264 / 564 | 46.81% |
| Gemma-4-E2B-it | 225 / 564 | 39.89% |

Pinned model revisions were `cdbee75f17c01a7cc42f958dc650907174af0554` (Qwen)
and `3e22461f65e89153144f8adb70e3b8c2cc9845a7` (Gemma).
[Uploaded artifacts](https://huggingface.co/Maverfrick/Venera-Traditional-Model/tree/eval-medical-tm-eval-1791208642/eval/tm-eval-1791208642)
include the CSV/Markdown comparison, complete predictions, manifests, and smoke
results. The local copy is under `EVAL/outputs/medical_baselines/tm-eval-1791208642/`.

Sources: [Qwen model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507),
[Gemma model card](https://huggingface.co/google/gemma-4-E2B-it),
[Transformers v5.5.0 Gemma support](https://github.com/huggingface/transformers/tree/v5.5.0/src/transformers/models/gemma4).

## Matched general-baseline thinking evaluation

Submit `sbatch scripts/slurm/eval_vm14k_thinking.sbatch` to evaluate both
Qwen3-4B-Thinking-2507 and Gemma4-E2B on the frozen 564 traditional questions.
The job pins the same model commits as general run `tm-eval-1792338556` and
uses the same evaluator: vLLM 0.31.0, BF16 without quantization, temperature
0.6, top-p 0.95, top-k 20, min-p 0, seed 3407, 4096 input tokens,
32768 output tokens, 8 active sequences, GPU memory fraction 0.70, eager mode.
The environment must already be installed at `.venv-vllm-slurm`.
Outputs go to `EVAL/outputs/vm14k_thinking/slurm-<job-id>/`.

The VM14K medical prompt and frozen row order are preserved. This is a new
thinking protocol, not a reproduction of the original notebook: that uses
250 output tokens and greedy decoding (`do_sample=False`; its temperature
0.7/top-p 0.8 have no effect under greedy decoding). The direct baseline
above instead constrains output to a letter, disables thinking, uses 16
output tokens, 2048 input tokens, and Transformers with NF4 weights.
The new protocol retains special tokens in saved generated responses and
scores only the final answer after the thinking boundary. Length-limited
responses and missing reasoning boundaries count as invalid and incorrect.
The shared general MCQ parser replaces the original notebook answer parser.
Scores across these protocols measure different generation conditions.

## CPT-SFT exp0015 with answer-only decoding

```bash
.venv/bin/python scripts/lambda_launch.py --task eval \
  --baselines exp0015 --eval-datasets vm14k --eval-generation-mode direct \
  --eval-batch-size 4 --eval-limit 0 --instance-type gpu_1x_a100_sxm4
```

This uses the same direct NF4/greedy protocol above. The final adapter is
`Maverfrick/Venera-Traditional-Model@06fc5e88287113fc29b1cdf236dd294260272daa`,
applied once on `unsloth/Qwen3-4B-Thinking-2507@fb56efbd0f60c5bcd531901fa5fe3ee3157b5135`.
The native template's open thinking block is explicitly closed before the
`Final Answer:` prefill; only option labels and EOS may be generated.
Saved responses have the form `Final Answer: A`. This forces answer-only
inference on a thinking-trained checkpoint. Four valid, terminated smoke
answers are required before the full 564-row evaluation. The manifest records
adapter/base revisions and the prompt closure. Lambda uses the existing
medical bundle, result upload/download, and automatic termination lifecycle.
