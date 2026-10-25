# Fetch and prepare CPT data inside the Lambda training job

The default pilot now performs all dataset acquisition and preprocessing on the
same Lambda instance that runs training. No local corpus exports, tokenization,
GPU libraries, or data-preparation environment are required to launch it.

The order is: install the isolated environment → fetch original pinned sources →
filter/deduplicate/decontaminate/tokenize → CPT → VM14K-only evaluation → upload
artifacts → download backup → terminate (unless `--keep-alive` is selected).

## Access requirements

Configure the existing Lambda API key, SSH key and Hugging Face token in `.env`.
The HF account must have accepted access to `bigcode/the-stack-v2`.
The Stack v2 contains file IDs, not code text. Its documented content source is
`softwareheritage/content/<blob_id>` on AWS S3, so also provide:

```dotenv
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
# Include when using temporary AWS credentials:
AWS_SESSION_TOKEN=...
AWS_DEFAULT_REGION=us-east-1
```

These values are transferred in the launcher's protected job environment and
forwarded into the container. They are excluded from source bundles and saved
experiment configs. Missing AWS keys fail before allocating a GPU. Gated HF
file access is also checked before allocation; S3 authorization is exercised
when downloading code on Lambda. A valid AWS account/key alone does not prove
that S3 content access will succeed.

## Single launch

First run the short training smoke job:

```bash
python scripts/lambda_launch.py --task cpt \
  --config configs/experiments/qwen3.5-4B/cpt-pilot/smoke.yaml \
  --cpt-pilot-smoke --instance-type gpu_1x_a100_sxm4
```

It caps raw domain acquisition near 131,072 tokens per split, fetches enough
replay documents for validation, trains two steps, and evaluates two VM14K
questions. Acquisition and preparation can still take time. Smoke results are
excluded from the final comparison.

After it succeeds, run the full mixed pilot:

```bash
python scripts/lambda_launch.py --task cpt \
  --config configs/experiments/qwen3.5-4B/cpt-pilot/mixed.yaml \
  --instance-type gpu_1x_a100_sxm4
```

The domain-only configuration uses the same full acquisition/preparation spec.
The comparison command rejects differing cleaned snapshots between the full arms.
The original-model baseline skips fetching, preparation and training.

## Source selection and preprocessing

Training configs reference `pilot.preparation_config`. Full preparation uses
`configs/data/cpt-pilot/prepare.yaml`, which references `upstream.yaml`; smoke
uses `prepare-smoke.yaml` and `upstream-smoke.yaml`. All five source repositories
and tokenizer revisions are pinned. Default source selections are:

- TradMed-CPT: train and validation chunks, grouped by original `source` and
  ordered by `chunk_index`.
- FineWeb-Edu: deterministic `sample-10BT` stream, educational score >= 3.
- Dolma v1.7: interleaved books and wiki from the pinned URL listing. The external
  component contents are recorded through generated export hashes; a pinned URL
  listing alone is not a cryptographic pin of those external objects.
- OpenWebMath: deterministic upstream text stream.
- The Stack v2: Python files with `permissive` license type; every detected license
  must be in MIT/Apache-2.0/BSD-2-Clause/BSD-3-Clause. Vendor/generated code is
  excluded. License filtering occurs before S3 content retrieval.

Replay acquisition includes 25% token headroom before cleaning and continues
until at least two document groups map to validation and minimum document-count
requirements are met. A scan cap prevents unbounded traversal. Actual cleaned
source sizes and any repetition during training are recorded rather than assumed.

The preparer filters licenses/components, groups chunks, deduplicates within and
across sources using exact hashes plus MinHash/LSH with Jaccard verification,
checks VM14K question/long-option overlap and supplied source article IDs, and
uses the pinned tokenizer with EOS document boundaries. Preserve established
validation splits; unsplit sources receive a deterministic 1% group holdout.
Unknown VM14K article provenance and approximate near-duplicate detection remain
limitations. Add known article IDs to `excluded_source_ids` in the upstream spec.
No separate general-benchmark contamination audit is claimed.

## Generated artifacts and failure behavior

On Lambda, `CPT/pilot_data/` contains original-source exports, a generated source
manifest, and prepared per-bucket train/validation token files. `preparation.json`
records source revisions, hashes, token counts, filters, and preparation config.
Complete prepared snapshots are published only after successful preprocessing.
Existing snapshots/exports are not silently overwritten.

The launcher includes these generated artifacts in its pre-termination backup.
Training's resolved configuration records preparation provenance on the HF run
branch. A fetching/preparation failure prevents training and triggers diagnostic
upload plus local backup attempts. No local prepared directory is required.

Standalone preparation remains available for debugging with
`python scripts/prepare_cpt_data.py --config configs/data/cpt-pilot/prepare.yaml --fetch-sources`.
Without `--fetch-sources`, the script still supports curated JSONL manifests.
