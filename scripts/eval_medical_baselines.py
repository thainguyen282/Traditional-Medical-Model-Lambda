#!/usr/bin/env python3
"""Evaluate medical baselines using the existing Vietnamese MCQ protocol."""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import time


ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "medgemma": "google/medgemma-1.5-4b-it",
    "huatuo": "FreedomIntelligence/HuatuoGPT-o1-8B",
}
DATASETS = {
    "new": ("tcm_mcq.jsonl", "Unsloth_QA_goc_new.ipynb"),
    "vm14k": ("eastern_medicine_VM14K.jsonl", "Unsloth_QA_goc_VM14K.ipynb"),
}
PROTOCOL_NAMES = {
    "generation_config", "system_prompt", "subset_size", "load_dataset",
    "process_medical_topics", "extract_answer", "evaluate_answer",
    "_REASONING_TAG_RE", "_MARKDOWN_HEADING_RE", "_EXPLICIT_ANSWER_RE",
    "_CORRECT_OPTION_RE", "_ORDERED_FALLBACKS",
}


def load_protocol(notebook: Path) -> dict:
    """Reuse notebook definitions without executing model loading or inference."""
    nodes = []
    found = set()
    for cell in json.loads(notebook.read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            name = None
            if isinstance(node, ast.FunctionDef):
                name = node.name
            elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                if isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
            if name in PROTOCOL_NAMES:
                nodes.append(node)
                found.add(name)
    if found != PROTOCOL_NAMES:
        raise ValueError(f"Missing notebook protocol definitions: {PROTOCOL_NAMES - found}")
    namespace = {"json": json, "re": re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(notebook), "exec"), namespace)
    return namespace


def prepare_dataset(key: str, limit: int = 0) -> tuple[list[dict], dict, dict]:
    filename, notebook_name = DATASETS[key]
    data_path = ROOT / "EVAL" / "Data" / filename
    notebook = ROOT / "EVAL" / notebook_name
    protocol = load_protocol(notebook)
    samples = list(protocol["load_dataset"](data_path))[:protocol["subset_size"]]
    if limit:
        samples = samples[:limit]
    if not samples:
        raise ValueError(f"No evaluation samples in {data_path}")
    for sample in samples:
        if sample["answer"] not in sample["options"]:
            raise ValueError(f"Invalid reference answer in {data_path}")
    metadata = {
        "dataset": key, "dataset_file": str(data_path.relative_to(ROOT)),
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "protocol_notebook": str(notebook.relative_to(ROOT)),
        "protocol_sha256": hashlib.sha256(notebook.read_bytes()).hexdigest(),
        "samples": len(samples), "limit": limit,
        "system_prompt": protocol["system_prompt"],
        "generation_config": protocol["generation_config"],
        "max_input_tokens": 2048,
    }
    return samples, protocol, metadata


def check_access(models: list[str], token: str | None) -> dict[str, str]:
    import requests

    revisions = {}
    for key in models:
        repo = MODELS[key]
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        response = requests.get(
            f"https://huggingface.co/{repo}/resolve/main/config.json",
            headers=headers, timeout=30,
        )
        if not response.ok:
            raise RuntimeError(
                f"Cannot access {repo} (HTTP {response.status_code}). "
                f"Check HF_TOKEN and accept/request access at https://huggingface.co/{repo}."
            )
        revision = next((
            item.headers["X-Repo-Commit"] for item in [response, *response.history]
            if "X-Repo-Commit" in item.headers
        ), None)
        if not revision:
            raise RuntimeError(f"Could not resolve an immutable revision for {repo}")
        revisions[key] = revision
        print(f"[medical-eval] access OK: {repo}@{revision}", flush=True)
    return revisions


def build_prompt(sample: dict, system_prompt: str) -> str:
    options = "\n".join(f"{key}) {value}" for key, value in sample["options"].items())
    return f"{system_prompt}\nQuestion: {sample['question']}\nOptions:\n{options}\n"


def load_model(key: str, revision: str, token: str | None):
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        Gemma3ForConditionalGeneration,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the 4-bit baseline evaluation")
    repo = MODELS[key]
    loader = Gemma3ForConditionalGeneration if key == "medgemma" else AutoModelForCausalLM
    model = loader.from_pretrained(
        repo, revision=revision, token=token,
        torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision, token=token)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def generate_batch(model, tokenizer, samples: list[dict], protocol: dict) -> list[dict]:
    import torch

    texts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": build_prompt(sample, protocol["system_prompt"])}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    ) for sample in samples]
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True,
        max_length=2048, add_special_tokens=False,
    ).to("cuda")
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, **protocol["generation_config"],
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = outputs[:, inputs.input_ids.shape[1]:]
    eos_ids = model.generation_config.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    results = []
    for tokens in generated.tolist():
        stopped = any(token in eos_ids for token in tokens)
        results.append({
            "response": tokenizer.decode(tokens, skip_special_tokens=True),
            "hit_token_limit": not stopped and len(tokens) >= protocol["generation_config"]["max_new_tokens"],
        })
    return results


def score_response(sample: dict, prediction: dict, protocol: dict, index: int) -> dict:
    extracted = protocol["extract_answer"](prediction["response"].strip())
    deviation = extracted not in sample["options"] and extracted not in ["NONE", ""]
    return {
        "q_idx": index, "question": sample["question"], "options": sample["options"],
        "correct_answer": sample["answer"], "extracted_answer": extracted,
        "is_correct": not deviation and protocol["evaluate_answer"](extracted, sample["answer"]),
        "is_deviation": deviation,
        "topics": protocol["process_medical_topics"](sample["medical_topic"]),
        "difficulty": sample["difficulty_level"].strip().capitalize(),
        **prediction,
    }


def summarize(rows: list[dict]) -> dict:
    def metrics(items):
        count = len(items)
        correct = sum(row["is_correct"] for row in items)
        deviations = sum(row["is_deviation"] for row in items)
        return {
            "samples": count, "correct": correct, "accuracy": correct / count,
            "deviations": deviations, "deviation_rate": deviations / count,
            "token_limit_rate": sum(row["hit_token_limit"] for row in items) / count,
        }

    groups = {"by_topic": defaultdict(list), "by_difficulty": defaultdict(list)}
    for row in rows:
        groups["by_difficulty"][row["difficulty"]].append(row)
        for topic in row["topics"]:
            groups["by_topic"][topic].append(row)
    return {**metrics(rows), **{
        name: {key: metrics(items) for key, items in group.items()}
        for name, group in groups.items()
    }}


def evaluate(key: str, dataset: str, revision: str, args) -> None:
    import torch

    samples, protocol, metadata = prepare_dataset(dataset, args.limit)
    output = args.output_dir / key / dataset
    output.mkdir(parents=True, exist_ok=False)
    metadata.update(model=MODELS[key], revision=revision, seed=3407,
                    quantization="nf4-double-quant-bfloat16", backend="transformers",
                    status="running")
    manifest = output / "run_config.json"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    from transformers import set_seed
    set_seed(3407)
    model = tokenizer = None
    rows = []
    batch_size = args.batch_size
    started = time.monotonic()
    try:
        model, tokenizer = load_model(key, revision, os.getenv("HF_TOKEN"))
        started = time.monotonic()
        with (output / "responses.jsonl").open("w", encoding="utf-8") as handle:
            while len(rows) < len(samples):
                batch = samples[len(rows):len(rows) + batch_size]
                try:
                    predictions = generate_batch(model, tokenizer, batch, protocol)
                except torch.cuda.OutOfMemoryError:
                    if batch_size == 1:
                        raise
                    batch_size = max(1, batch_size // 2)
                    gc.collect()
                    torch.cuda.empty_cache()
                    print(f"[medical-eval] OOM; retrying with batch size {batch_size}", flush=True)
                    continue
                if len(predictions) != len(batch):
                    raise RuntimeError("Generation returned the wrong number of responses")
                for sample, prediction in zip(batch, predictions):
                    row = score_response(sample, prediction, protocol, len(rows) + 1)
                    rows.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"[medical-eval] {key}/{dataset}: {len(rows)}/{len(samples)}", flush=True)
        summary = {**metadata, **summarize(rows), "status": "complete",
                   "inference_time_s": time.monotonic() - started}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (output / "infer_result.txt").write_text(
            f"Model: {MODELS[key]}\nDataset: {dataset}\n"
            f"Total Questions Processed: {len(rows)}\n"
            f"Overall Accuracy: {summary['accuracy']:.2%}\n"
            f"Deviation Rate: {summary['deviation_rate']:.2%}\n"
            f"Token Limit Rate: {summary['token_limit_rate']:.2%}\n",
            encoding="utf-8",
        )
        metadata["status"] = "complete"
        manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    except Exception as error:
        metadata.update(status="failed", error=str(error), completed_samples=len(rows))
        manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()


def main() -> int:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*MODELS, "all"], default="all")
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--limit", type=int, default=0, help="0 uses the full notebook evaluation subset")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "EVAL/outputs/medical_baselines")
    parser.add_argument("--dry-run", action="store_true", help="Validate datasets/protocol without GPU or network")
    parser.add_argument("--check-access", action="store_true", help="Validate model access without loading weights")
    args = parser.parse_args()
    if args.limit < 0 or args.batch_size < 1:
        parser.error("--limit must be nonnegative and --batch-size must be positive")
    models = list(MODELS) if args.model == "all" else [args.model]
    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        _, _, metadata = prepare_dataset(dataset, args.limit)
        print(json.dumps(metadata, indent=2))
    if args.dry_run:
        return 0
    revisions = check_access(models, os.getenv("HF_TOKEN"))
    if args.check_access:
        return 0
    failures = 0
    for model in models:
        for dataset in datasets:
            try:
                evaluate(model, dataset, revisions[model], args)
            except Exception as error:
                failures += 1
                print(f"[medical-eval] FAILED {model}/{dataset}: {error}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
