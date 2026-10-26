# Standalone 80/20 CPT dataset preparation

This tool prepares a dataset locally and optionally uploads it to Hugging Face.
It does not import the CPT trainer, modify training/Lambda code, launch a cloud
instance, or mix instruction/SFT examples into the corpus. No GPU or AWS access is required by the default configuration.

## Environment

Use the repository-local `.venv` created with uv. If it does not exist, first run
`uv venv .venv`. Install the CPU-only preparation dependencies:

```bash
mkdir -p .cache/tmp .cache/uv
TMPDIR="$PWD/.cache/tmp" UV_CACHE_DIR="$PWD/.cache/uv" \
  uv pip install --python .venv/bin/python -r requirements-data-prep.txt
```

## Configure inputs

Edit `configs/data/cpt_mix.yaml`. The default allocation by training tokens is:

| Bucket | Share | Default input |
|---|---:|---|
| Medical domain | 80% | Existing TradMed-CPT HF train/validation splits |
| FineWeb-Edu | 12% | HF sample-10BT; educational score >= 3 |
| Dolma | 4% | Direct books/wiki streams from the pinned Dolma v1.7 URL listing |
| OpenWebMath | 2% | HF mathematical text |
| Licensed code | 2% | The Stack Dedup Python text directly from HF |

The default configuration downloads all inputs directly. The Stack Dedup requires
accepting its Hugging Face access conditions; its `content` field includes the
code, so no AWS credentials are used. All detected repository licenses must match
the allowlist. Python is selected by default. Source repo/path/revision and license
metadata are retained in `provenance`.

Dolma's pinned URL listing selects books and wiki components. The reader handles
HTTP/content gzip layers and interleaves components. A pinned listing alone does
not cryptographically pin the external objects; corpus output hashes record what
was processed.

Local files remain supported by changing `inputs` to `kind: local`. The examples
below describe those optional local formats.

Example Dolma JSONL row:

```json
{"document_id":"book-123","text":"Actual book text...","component":"books"}
```

Example JSONL row for the optional local-code alternative:

```json
{"document_id":"repo/path.py","text":"Actual source code...","license":"MIT","language":"Python"}
```

Code rows require an allowed SPDX license (or list of licenses) and language.
Dolma rows require a component in `general`, `books`, `encyclopedia`. Match the
config field names to the actual input metadata; do not invent license labels.
A TXT input is a single complete document and has no license/component metadata,
so code/Dolma normally need JSONL or Parquet.

For any source, `kind: local` uses `path`; `kind: hf` uses `repo`, optional
`config`, `revision`, and `split`. Optional `role: train` or `role: validation`
preserves established split membership. Without a role, original document groups
are deterministically assigned to a 1% validation holdout using seed 42.

Set `document_field` to a globally unique original article/file ID, shared by all its chunks;
set `chunk_field` to their ordering field. If there is no ID, complete documents
can use a content-hash fallback. Chunked rows without a group ID are rejected.
Groups appearing in both established splits are quarantined entirely to
validation. This can substantially reduce training data when old splits were
made by chunk rather than original document; inspect the output counts.

The tokenizer defaults to Qwen3-4B-Thinking-2507 to match the existing experiment.
**Change it to the exact tokenizer for the model you plan to train** (for example
Qwen3.5-4B). EOS must exist in that tokenizer. No model weights are loaded.

## Build and review locally

```bash
.venv/bin/python scripts/prepare_cpt_mix.py \
  --config configs/data/cpt_mix.yaml
```

The tool normalizes prose, preserves code indentation, filters sources, groups chunks, checks supplied VM14K
questions/known document IDs, deduplicates within/across sources using exact
hashes and approximate MinHash/LSH, then creates Unicode-safe text chunks no
longer than 2,048 tokens. EOS is appended at the actual document end, not every
intermediate chunk. Sequence packing/order remains the training loader's job.
Unknown benchmark source articles and paraphrased questions can remain;
this is not a guarantee of complete benchmark decontamination.

By default, cleaned medical training tokens determine the total: `D / 0.8`.
At D = 29M the allocation is approximately 29M / 4.35M / 1.45M / 0.725M / 0.725M.
General sources must have sufficient cleaned tokens; repetition is disabled by
default. A small final-token shortfall can occur at a text boundary; the report
records actual counts and shares. Set `total_tokens` and `allow_repetition` only
when intentionally preparing a different processed-token budget.

Outputs under `CPT/prepared_mix/`:

- `train.jsonl` and `validation.jsonl`, each with a raw `text` column compatible
  with the restored CPT loader and metadata `source`, `bucket`, `chunk_index`,
  and `tokens`.
- `manifest.json`: resolved inputs, tokenizer revision, configuration, hashes,
  actual token allocations, available corpus sizes and repetition ratios.

Existing output is never overwritten. Use a new `output_dir` for a new version.
Code indentation must also be preserved by the future training loader; the restored
legacy CPT cleaner currently collapses whitespace and has not been changed.

Collected document groups are checkpointed under `.cache/cpt-source-groups/` so
processing retries reuse pinned inputs. Source configuration, local file hashes,
tokenizer revision and collection settings form each checkpoint key.

Corpus counts precede trainer packing and epoch repetition; validation tokens
are excluded from training allocations. `tokens` is metadata, not a loss mask.

## Upload explicitly

Add `HF_TOKEN` to the repository `.env` or authenticate with Hugging Face. Upload
an already reviewed dataset without rebuilding:

```bash
.venv/bin/python scripts/prepare_cpt_mix.py \
  --upload-only CPT/prepared_mix \
  --push-to-hub YOUR_ACCOUNT/tradmed-cpt-mixed
```

Or prepare and upload in the same local command by adding `--push-to-hub` to the
build command. New repositories default to private; add `--public` explicitly
for a public dataset. The upload includes `train`, `validation` and the manifest.
No upload happens without `--push-to-hub`.

After upload, a future training configuration can point `dataset.name` to the
new HF dataset, with `text_field: text`, `train_split: train`, and
`validation_split: validation`. No training configuration is changed by this tool.
