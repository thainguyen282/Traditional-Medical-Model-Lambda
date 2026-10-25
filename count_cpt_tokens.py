#!/usr/bin/env python3
"""Disposable CPU-only CPT corpus counter; does not import the training script.

Run: python count_cpt_tokens.py
Delete afterward: rm count_cpt_tokens.py
Requires datasets, tokenizers, huggingface-hub, pyyaml, python-dotenv.
"""
import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs/experiments/unsloth-qwen3-4B/exp0015/config_cpt.yaml"


def clean_text(text):
    # Match CPT/CPT.py without importing it (importing it starts training).
    lines = [" ".join(line.split()) for line in text.strip().splitlines()]
    return "\n".join(line for line in lines if line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", help="Override the configured Hugging Face dataset ID")
    parser.add_argument("--tokenizer", help="Override the configured model/tokenizer")
    parser.add_argument("--splits", nargs="+", help="Default: configured train and validation splits")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    # Only the tokenizer is needed; avoid loading GPU/model frameworks.
    os.environ.setdefault("USE_TORCH", "0")
    os.environ.setdefault("USE_TF", "0")
    import yaml
    from datasets import load_dataset
    from dotenv import load_dotenv
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    load_dotenv(ROOT / ".env", override=False)
    config = yaml.safe_load(args.config.read_text())
    data = config.get("dataset", {})
    dataset_id = args.dataset or data.get("name", "Maverfrick/TradMed-CPT-53k-2048")
    tokenizer_id = args.tokenizer or config.get("model", {}).get(
        "name", "unsloth/Qwen3-4B-Instruct-2507")
    field = data.get("text_field", "text")
    # Explicit for exp0015. Other trainer versions can differ in their defaults.
    if "add_special_tokens" not in data:
        parser.error("Set dataset.add_special_tokens in the config to make counting explicit")
    special = data["add_special_tokens"]
    splits = args.splits or list(dict.fromkeys([
        data.get("train_split", "train"), data.get("validation_split", "validation")]))
    token = os.getenv("HF_TOKEN")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # This CPT config uses Qwen's fast tokenizer with no special tokens.
    # Read its serialized tokenizer directly, avoiding model-library imports.
    if special:
        parser.error("This counter supports add_special_tokens: false (as in exp0015)")
    tokenizer_path = Path(tokenizer_id)
    if tokenizer_path.is_dir():
        tokenizer_path = tokenizer_path / "tokenizer.json"
    elif not tokenizer_path.is_file():
        tokenizer_path = Path(hf_hub_download(tokenizer_id, "tokenizer.json", token=token))
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    print(json.dumps({"dataset": dataset_id, "tokenizer": tokenizer_id,
                      "text_field": field, "add_special_tokens": special,
                      "config": str(args.config)}, indent=2), flush=True)
    for split in splits:
        dataset = load_dataset(dataset_id, split=split, token=token, streaming=True)
        rows = kept = total = 0
        minimum = None
        maximum = 0
        for batch in dataset.iter(batch_size=args.batch_size):
            texts = batch[field]
            rows += len(texts)
            cleaned = [clean_text(text) for text in texts if text]
            cleaned = [text for text in cleaned if text]
            if not cleaned:
                continue
            encoded = tokenizer.encode_batch(cleaned, add_special_tokens=False)
            lengths = [len(item.ids) for item in encoded]
            total += sum(lengths)
            kept += len(lengths)
            minimum = min(lengths) if minimum is None else min(minimum, min(lengths))
            maximum = max(maximum, max(lengths))
            if rows % (args.batch_size * 50) == 0:
                print(f"[{split}] {rows:,} rows; {total:,} tokens", flush=True)
        print(json.dumps({"split": split, "rows": rows, "nonempty_rows": kept,
                          "tokens_one_pass": total, "million_tokens": total / 1e6,
                          "mean_tokens_per_nonempty_row": total / kept if kept else 0,
                          "min_tokens": minimum, "max_tokens": maximum}, indent=2), flush=True)
    print("Counts are cleaned corpus tokens for one pass, before trainer packing, "
          "remainder handling, or any trainer-inserted EOS tokens. "
          "Validation tokens are not training tokens.")


if __name__ == "__main__":
    main()
