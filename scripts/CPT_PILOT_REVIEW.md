# Developer review: Lambda CPT mixture pilot

This change adds an opt-in Qwen3.5-4B adapter CPT experiment to the existing
`CPT/CPT.py` entrypoint and Lambda launcher. Evaluation is restricted to the
frozen **564-question traditional-medicine VM14K subset**. It does not establish
retention of general knowledge, reasoning, or instruction following.

## What changed

| Subsystem | Before | Pilot behavior |
|---|---|---|
| Training data | One raw-text medical dataset | Prepared, grouped, decontaminated tokens from five sources; source allocation is measured in tokens |
| Training budget | Dataset epochs | One epoch over an explicitly allocated stream: `ceil(D / 0.8)` tokens for both arms |
| Optimization | Legacy Unsloth LoRA CPT | Same default batch 2 × accumulation 8, LR 3e-5, rank/alpha 64, BF16; seed 42; one budget epoch |
| Packing | Trainer packs raw text | CPU preparation produces 2,048-token sequences with EOS document boundaries; final partial sequence is retained and padded with labels masked by position |
| Attention boundaries | Cross-document attention possible | Remains possible; EOS separates documents, without block-diagonal attention |
| Base model | Legacy Qwen3 defaults | Model and tokenizer pinned to Qwen/Qwen3.5-4B commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Dependencies | Shared image, Transformers 4.57.6 | Isolated `.venv-cpt-pilot`, Transformers 5.5.0 and pinned Unsloth/Zoo source revisions; transitive pins in `requirements-cpt-pilot.lock.txt`; original image requirements unchanged |
| Lambda source | Configured CPT uses cloned code | Exact code, acquisition/preparation configs and frozen VM14K protocol bundled; originals fetched and tokenized on Lambda |
| Adapter publication | Legacy CPT writes `main` | Pilot writes `cpt-pilot-<job-tag>` in `HF_CPT_REPO` |
| Evaluation | Legacy CPT runs two notebooks and warns on failure | Fresh container loads the exact local final adapter and pinned base; runs VM14K only; failure is fatal |
| Cleanup | Existing launcher termination | Pilot backs up generated source exports/tokens, adapter, run outputs and logs before default termination; backup failure marks job failed |

`CPT/CPT.py` dispatches only configurations with `pilot.enabled: true` to the
new trainer. Legacy CPT and SFT behavior remains on its existing path.
Language projection adapters exclude vision modules, embeddings and LM head.
The seven existing projection targets are retained; hybrid linear-attention
projections are not added to this experiment.

## Acquisition and preparation in the same Lambda job

The local launcher no longer requires `CPT/pilot_data/prepared/`. It validates
model/preparation config compatibility and checks source access before allocation.
The remote job invokes `prepare_cpt_data.py --fetch-sources` before CPT. Original
pinned HF datasets are fetched and normalized by `fetch_cpt_sources.py`, then the
existing cleaning, document splitting, deduplication, decontamination and
tokenization run on the same Lambda instance. A failure prevents training.

See [CPT_DATA_PREPARATION.md](CPT_DATA_PREPARATION.md) for source choices and access
requirements. The Stack v2 hydration requires AWS credentials and gated HF access;
there is no silent substitution. Smoke uses a smaller acquisition spec; full
mixed/domain-only runs use the same full spec. Generated export/token hashes are
recorded and backed up before termination.

Both full training arms use the same acquisition/preparation configuration. The
mixed budget is 80/12/4/2/2 percent (integer rounding recorded). With 29M cleaned
domain tokens, it is 36.25M tokens. Domain-only repeats medical tokens to match
total processed tokens. Report actual shares and corpus passes.

## Exact Lambda commands

Use existing `.env` credentials and the existing `--git-repo`, `--git-branch`,
`DOCKER_IMAGE` settings as appropriate. Each invocation starts a separate A100
40GB job. The launcher must remain alive for monitoring, backup and termination.
The new flags do not use the general-task evaluation path.

First run the two-step training + two-question VM14K smoke test:

```bash
python scripts/lambda_launch.py --task cpt \
  --config configs/experiments/qwen3.5-4B/cpt-pilot/smoke.yaml \
  --cpt-pilot-smoke --instance-type gpu_1x_a100_sxm4
```

Only after that succeeds, run the baseline and two full training arms:

```bash
python scripts/lambda_launch.py --task eval --cpt-pilot-baseline \
  --config configs/experiments/qwen3.5-4B/cpt-pilot/mixed.yaml \
  --instance-type gpu_1x_a100_sxm4
python scripts/lambda_launch.py --task cpt \
  --config configs/experiments/qwen3.5-4B/cpt-pilot/domain-only.yaml \
  --instance-type gpu_1x_a100_sxm4
python scripts/lambda_launch.py --task cpt \
  --config configs/experiments/qwen3.5-4B/cpt-pilot/mixed.yaml \
  --instance-type gpu_1x_a100_sxm4
```

The evaluation uses native thinking-enabled templates, unconstrained generation,
32,768 output tokens, temperature 0.6, top-p 0.95, top-k 20 and a fixed per-question
seed. Prompts exceeding 2,048 tokens fail rather than silently truncate. Score
only the final answer after `</think>`; missing boundaries, invalid outputs and
length-truncated responses are incorrect. Final checkpoints only; no VM14K
checkpoint selection and no preservation SFT.

## Artifacts and report

Each launch writes `logs/<job-tag>/source_manifest.json` and `run.json`, including
the artifact URL and instance ID. Before termination it downloads
`logs/<job-tag>/cpt-pilot.tar.gz`; the archive includes training artifacts,
`CPT/pilot_runs/<job-tag>/vm14k/` and `train.log`. HF branches contain the adapter
and tokenizer at the root and evaluation outputs under `pilot/vm14k/`, plus
resolved configs, token accounting, source hashes and installed dependency versions.
Evaluation/upload errors fail the job; diagnostic uploads are best effort.
Missing setup dependencies can prevent diagnostic upload, so the launcher also
attempts the local backup. Default cleanup still terminates after backup errors.

After extracting the three backups, generate the report with directories that
contain each run's `summary.json` and `predictions.jsonl`, in this exact order:

```bash
python scripts/cpt_pilot.py compare \
  --runs /path/to/original/vm14k /path/to/domain/vm14k /path/to/mixed/vm14k \
  --output vm14k-pilot-comparison.md
```

The command requires complete, compatible 564-row runs, aligns question IDs and
reference answers, verifies matching cleaned snapshots across training arms, and reports paired 95% bootstrap intervals using 10,000
resamples with seed 42. Smoke runs cannot enter the comparison.

| Condition | VM14K result | Artifact link |
|---|---|---|
| Original Qwen3.5-4B | Pending execution | Generated in baseline `run.json` |
| Domain-only CPT | Pending execution | Generated in domain-only `run.json` |
| 80/20 CPT | Pending execution | Generated in mixed `run.json` |

No scores or remote artifact links are claimed before execution.

## Verification status

CPU tests cover token budgets, deterministic packing, split separation, exact
and near deduplication, export integrity, final-answer scoring, paired report
compatibility, source bundles, config forwarding, failure propagation and cleanup.
The local launcher no longer requires a prepared corpus. Source-access checks
currently stop the launch because AWS credentials for The Stack v2 hydration
are absent. No instance has been allocated. GPU compatibility, direct acquisition
and full VM14K results remain unverified. The updated acquisition dependency set resolves against the validated training
pins; python-dateutil is pinned below 3 for botocore compatibility.
An existing unrelated `test_cpt_config.py` assertion expects batch size 8 while
the existing exp0015 config specifies 16; this change does not alter that config.
