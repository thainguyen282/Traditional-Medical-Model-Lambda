#!/usr/bin/env python3
"""Evaluate open-weight baselines using the existing Vietnamese MCQ protocol."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import copy
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import time


ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "medgemma": "google/medgemma-1.5-4b-it",
    "huatuo": "FreedomIntelligence/HuatuoGPT-o1-8B",
    "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507",
    "gemma4-e2b": "google/gemma-4-E2B-it",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "medreason": "UCSC-VLAA/MedReason-8B",
    "baichuan-m2-32b": "baichuan-inc/Baichuan-M2-32B",
}
DATASETS = {
    "new": ("tcm_mcq.jsonl", "Unsloth_QA_goc_new.ipynb"),
    "vm14k": ("eastern_medicine_VM14K.jsonl", "Unsloth_QA_goc_VM14K.ipynb"),
    "vm14k-full": ("Sampled-VM14K.jsonl", "Unsloth_QA_goc_VM14K.ipynb"),
}
DEFAULT_DATASETS = ["new", "vm14k-full"]
TRADITIONAL_SUBSET = "traditional_medicine_564"
ANSWER_PREFIX = "Final Answer:"
DIRECT_INSTRUCTION = "\nAnswer directly. Do not include thinking, reasoning, or an explanation.\n"
VM14K_SHA256 = "314006d439047ac9b98a890fe14ee1954025b28040c3e162ef71dc771af56574"
VM14K_BASELINES = ("qwen3-4b", "gemma4-e2b")

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


def prepare_dataset(key: str, limit: int = 0, generation_mode: str = "direct") -> tuple[list[dict], dict, dict]:
    filename, notebook_name = DATASETS[key]
    data_path = ROOT / "EVAL" / "Data" / filename
    notebook = ROOT / "EVAL" / notebook_name
    protocol = load_protocol(notebook)
    if generation_mode not in {"direct", "notebook"}:
        raise ValueError(f"Unknown generation mode: {generation_mode}")
    protocol["generation_mode"] = generation_mode
    if generation_mode == "direct":
        protocol["system_prompt"] += DIRECT_INSTRUCTION
        protocol["generation_config"] = {
            "max_new_tokens": 16, "do_sample": False,
            "num_beams": 1, "num_return_sequences": 1,
            "min_length": 0,
        }
    # Full evaluation must not silently inherit a notebook's sampling cap.
    samples = list(protocol["load_dataset"](data_path))
    total_samples = len(samples)
    if key == "vm14k" and (total_samples != 564 or
            hashlib.sha256(data_path.read_bytes()).hexdigest() != VM14K_SHA256):
        raise ValueError("VM14K must match the frozen 564-row traditional medicine subset")
    subset_metadata = {}
    if key == "vm14k-full":
        reference_path = ROOT / "EVAL" / "Data" / DATASETS["vm14k"][0]
        reference = list(protocol["load_dataset"](reference_path))
        traditional = [s for s in samples if "Eastern Medicine" in
                       protocol["process_medical_topics"](s["medical_topic"])]

        def canonical(rows):
            # Preserve repeated questions and option permutations, not just IDs.
            return Counter(json.dumps(s, sort_keys=True, ensure_ascii=False) for s in rows)

        if len(reference) != 564 or canonical(traditional) != canonical(reference):
            raise ValueError("Full VM14K traditional subset does not match the historical 564 rows")
        for sample in samples:
            sample["is_traditional_medicine"] = "Eastern Medicine" in protocol["process_medical_topics"](sample["medical_topic"])
        subset_metadata = {
            "traditional_subset_file": str(reference_path.relative_to(ROOT)),
            "traditional_subset_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            "traditional_subset_total": len(reference),
        }
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
        "samples": len(samples), "total_dataset_samples": total_samples, "limit": limit,
        **subset_metadata,
        "system_prompt": protocol["system_prompt"],
        "generation_config": protocol["generation_config"],
        "max_input_tokens": 2048,
        "generation_mode": generation_mode,
        "assistant_prefix": ANSWER_PREFIX if generation_mode == "direct" else "",
        "constrained_to_options": generation_mode == "direct",
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


def check_runtime(models: list[str]) -> dict:
    """Import inference dependencies before downloading weights or evaluating."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import bitsandbytes
    import accelerate

    if "gemma4-e2b" in models:
        try:
            from transformers import AutoProcessor, Gemma4ForConditionalGeneration
        except ImportError as error:
            raise RuntimeError(
                "Gemma 4 requires the isolated environment: "
                "bash scripts/setup_baseline_env.sh"
            ) from error
    return {name: version(name) for name in
            ("torch", "transformers", "accelerate", "bitsandbytes", "tokenizers", "huggingface-hub")}


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
    processor = None
    if key == "gemma4-e2b":
        from transformers import AutoProcessor, Gemma4ForConditionalGeneration
        loader = Gemma4ForConditionalGeneration
        processor = AutoProcessor.from_pretrained(repo, revision=revision, token=token)
    model = loader.from_pretrained(
        repo, revision=revision, token=token,
        torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
    tokenizer = (processor.tokenizer if processor is not None else
                 AutoTokenizer.from_pretrained(repo, revision=revision, token=token))
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer, processor


def answer_constraint(tokenizer, samples, input_length, eos_ids, max_new_tokens):
    """Allow only an option label followed by EOS, including multi-token labels."""
    if not eos_ids:
        raise ValueError("Answer-only generation requires an EOS token")
    tries = []
    for sample in samples:
        root = {}
        for label in sample["options"]:
            tokens = tokenizer.encode(" " + label, add_special_tokens=False)
            if not tokens or len(tokens) + 1 > max_new_tokens or set(tokens) & set(eos_ids):
                raise ValueError(f"Cannot constrain option label: {label!r}")
            node = root
            for token in tokens:
                node = node.setdefault(token, {})
            for eos in eos_ids:
                node[eos] = {}
        tries.append(root)

    def allowed_tokens(batch_id, input_ids):
        node = tries[batch_id]
        for token in input_ids[input_length:].tolist():
            # Finished rows can remain in a batch until the longest row ends.
            if token in eos_ids:
                return list(eos_ids)
            node = node[token]
        return list(node)

    return allowed_tokens


def generate_batch(model, tokenizer, samples: list[dict], protocol: dict, processor=None) -> list[dict]:
    import torch

    direct = protocol["generation_mode"] == "direct"
    formatter = processor if processor is not None else tokenizer
    texts = [formatter.apply_chat_template(
        [{"role": "user", "content": build_prompt(sample, protocol["system_prompt"])}],
        tokenize=False, add_generation_prompt=True,
        **({"enable_thinking": False} if direct else {}),
    ) + (ANSWER_PREFIX if direct else "") for sample in samples]
    if direct:
        # Preserve the assistant prefill if a question exceeds the input budget.
        tokenizer.truncation_side = "left"
    inputs = formatter(
        text=texts, return_tensors="pt", padding=True, truncation=True,
        max_length=2048, add_special_tokens=False,
    ).to("cuda")
    eos_ids = model.generation_config.eos_token_id
    eos_ids = [token for token in (eos_ids if isinstance(eos_ids, list) else [eos_ids])
               if token is not None]
    constraints = {}
    if direct:
        constraints["prefix_allowed_tokens_fn"] = answer_constraint(
            tokenizer, samples, inputs.input_ids.shape[1], eos_ids,
            protocol["generation_config"]["max_new_tokens"],
        )
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, **protocol["generation_config"], **constraints,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = outputs[:, inputs.input_ids.shape[1]:]
    results = []
    for tokens in generated.tolist():
        stopped = any(token in eos_ids for token in tokens)
        response = tokenizer.decode(tokens, skip_special_tokens=True)
        if direct:
            response = ANSWER_PREFIX + " " + response.strip()
        results.append({
            "response": response,
            "hit_token_limit": not stopped and len(tokens) >= protocol["generation_config"]["max_new_tokens"],
        })
    return results


def score_response(sample: dict, prediction: dict, protocol: dict, index: int) -> dict:
    extracted = protocol["extract_answer"](prediction["response"].strip())
    deviation = extracted not in sample["options"]
    return {
        "q_idx": index, "question": sample["question"], "options": sample["options"],
        "is_traditional_medicine": "Eastern Medicine" in protocol["process_medical_topics"](sample["medical_topic"]),
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
            "samples": count, "correct": correct, "accuracy": correct / count if count else None,
            "deviations": deviations, "deviation_rate": deviations / count if count else None,
            "token_limit_rate": sum(row["hit_token_limit"] for row in items) / count if count else None,
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


def write_metrics(output: Path, summary: dict) -> None:
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    def percentage(name):
        value = summary[name]
        return "N/A (no evaluated rows)" if value is None else f"{value:.2%}"

    (output / "infer_result.txt").write_text(
        f"Model: {summary['model']}\nDataset: {summary['dataset']}\n"
        f"Generation Mode: {summary['generation_mode']}\n"
        f"Total Questions Processed: {summary['samples']}\n"
        f"Overall Accuracy: {percentage('accuracy')}\n"
        f"Deviation Rate: {percentage('deviation_rate')}\n"
        f"Token Limit Rate: {percentage('token_limit_rate')}\n",
        encoding="utf-8",
    )


def save_traditional_subset(output: Path, rows: list[dict], metadata: dict) -> None:
    """Export the exact predictions from the parent run; never infer twice."""
    selected = [row for row in rows if row["is_traditional_medicine"]]
    subset_dir = output / TRADITIONAL_SUBSET
    subset_dir.mkdir(exist_ok=False)
    subset_metadata = {
        **metadata,
        "dataset": "vm14k",
        "dataset_file": metadata["traditional_subset_file"],
        "dataset_sha256": metadata["traditional_subset_sha256"],
        "parent_dataset": "vm14k-full",
        "parent_dataset_sha256": metadata["dataset_sha256"],
        "parent_evaluated_samples": len(rows),
        "total_dataset_samples": metadata["traditional_subset_total"],
        "samples": len(selected),
        "complete_subset": len(selected) == metadata["traditional_subset_total"],
        "selection": "Eastern Medicine topic; validated against historical 564 rows",
        "q_idx_scope": "1-based row index in parent dataset",
        "status": "complete",
    }
    with (subset_dir / "responses.jsonl").open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (subset_dir / "run_config.json").write_text(json.dumps(subset_metadata, indent=2) + "\n", encoding="utf-8")
    write_metrics(subset_dir, {**subset_metadata, **summarize(selected)})


def export_vm14k_comparison(output: Path) -> None:
    """Publish a table only when both complete, comparable 564-row runs exist."""
    samples, _, expected = prepare_dataset("vm14k")
    records = []
    common = None
    for key in VM14K_BASELINES:
        directory = output / key / "vm14k"
        metadata = json.loads((directory / "run_config.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        rows = [json.loads(line) for line in (directory / "responses.jsonl").read_text().splitlines()]
        required = {
            "status": "complete", "dataset": "vm14k", "model": MODELS[key],
            "samples": 564, "total_dataset_samples": 564, "limit": 0,
            "generation_mode": "direct", "thinking_enabled": False,
            "constrained_to_options": True, "dataset_sha256": VM14K_SHA256,
            "generation_config": expected["generation_config"],
        }
        if any(metadata.get(k) != v or summary.get(k) != v for k, v in required.items()):
            raise ValueError(f"{key}: comparison requires a complete VM14K no-thinking run")
        if len(rows) != 564 or not metadata.get("revision"):
            raise ValueError(f"{key}: incomplete responses or missing checkpoint revision")
        for index, (row, sample) in enumerate(zip(rows, samples), 1):
            if (row["q_idx"] != index or row["question"] != sample["question"] or
                    row["options"] != sample["options"] or row["correct_answer"] != sample["answer"]):
                raise ValueError(f"{key}: responses do not match the frozen VM14K row order")
        recalculated = summarize(rows)
        if any(summary.get(k) != recalculated[k] for k in
               ("samples", "correct", "accuracy", "deviations", "deviation_rate", "token_limit_rate")):
            raise ValueError(f"{key}: summary does not match responses")
        settings = {k: metadata[k] for k in (
            "protocol_sha256", "system_prompt", "generation_config", "max_input_tokens",
            "assistant_prefix", "quantization", "seed", "dependencies", "chat_template_kwargs",
        )}
        if common is not None and settings != common:
            raise ValueError("Baseline runs used different evaluation settings or dependencies")
        common = settings
        records.append({
            "model": MODELS[key], "correct/564": f"{summary['correct']}/564",
            "accuracy_pct": round(summary["accuracy"] * 100, 4),
            "invalid_answer_pct": round(summary["deviation_rate"] * 100, 4),
            "token_limit_pct": round(summary["token_limit_rate"] * 100, 4),
        })
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = [
        "# VM14K baselines: constrained answer-only MCQ evaluation", "",
        "All 564 questions; thinking disabled. Invalid-answer rate is a decoding diagnostic, "
        "because output is constrained to option labels.", "",
        "| Model | Correct/564 | Accuracy (%) | Invalid answers (%) | Token limit (%) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in records:
        lines.append(f"| {row['model']} | {row['correct/564']} | {row['accuracy_pct']:.2f} | "
                     f"{row['invalid_answer_pct']:.2f} | {row['token_limit_pct']:.2f} |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(key: str, dataset: str, revision: str, args) -> None:
    import torch

    dependencies = check_runtime([key])
    samples, protocol, metadata = prepare_dataset(dataset, args.limit, args.generation_mode)
    output = args.output_dir / key / dataset
    output.mkdir(parents=True, exist_ok=False)
    metadata.update(model=MODELS[key], revision=revision, seed=3407,
                    quantization="nf4-double-quant-bfloat16", backend="transformers",
                    status="running")
    metadata.update(
        dependencies=dependencies, requested_batch_size=args.batch_size,
        thinking_enabled=False if args.generation_mode == "direct" or key == "qwen3-4b" else None,
        chat_template_kwargs={"enable_thinking": False} if args.generation_mode == "direct" else {},
        protocol_label="constrained answer-only MCQ evaluation" if args.generation_mode == "direct" else "notebook generation",
    )
    manifest = output / "run_config.json"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    from transformers import set_seed
    set_seed(3407)
    model = tokenizer = processor = None
    rows = []
    batch_size = args.batch_size
    started = time.monotonic()
    try:
        model, tokenizer, processor = load_model(key, revision, os.getenv("HF_TOKEN"))
        started = time.monotonic()
        with (output / "responses.jsonl").open("w", encoding="utf-8") as handle:
            while len(rows) < len(samples):
                batch = samples[len(rows):len(rows) + batch_size]
                try:
                    predictions = generate_batch(model, tokenizer, batch, protocol, processor)
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
        metadata["effective_batch_size"] = batch_size
        summary = {**metadata, **summarize(rows), "status": "complete",
                   "inference_time_s": time.monotonic() - started}
        if dataset == "vm14k-full":
            save_traditional_subset(output, rows, metadata)
        write_metrics(output, summary)
        metadata["status"] = "complete"
        manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    except Exception as error:
        metadata.update(status="failed", error=str(error), completed_samples=len(rows))
        manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        del model, tokenizer, processor
        gc.collect()
        torch.cuda.empty_cache()


def main() -> int:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*MODELS, "all"], default="all")
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all",
                        help="all: new + vm14k-full (includes separate traditional subset export)")
    parser.add_argument("--limit", type=int, default=0, help="0 uses every row in the selected dataset")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--generation-mode", choices=["direct", "notebook"], default="direct",
                        help="direct: answer-only decoding; notebook: original free generation (may reason)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "EVAL/outputs/medical_baselines")
    parser.add_argument("--dry-run", action="store_true", help="Validate datasets/protocol without GPU or network")
    parser.add_argument("--check-access", action="store_true", help="Validate model access without loading weights")
    parser.add_argument("--check-runtime", action="store_true", help="Check inference dependencies without loading weights")
    parser.add_argument("--vm14k-baselines", action="store_true",
                        help="Qwen3-4B + Gemma4-E2B, direct VM14K only; smoke both before a full comparison")
    parser.add_argument("--export-comparison", action="store_true",
                        help="Export existing complete Qwen/Gemma VM14K runs from --output-dir; no inference")
    args = parser.parse_args()
    if args.limit < 0 or args.batch_size < 1:
        parser.error("--limit must be nonnegative and --batch-size must be positive")
    if args.export_comparison:
        export_vm14k_comparison(args.output_dir)
        return 0
    models = list(MODELS) if args.model == "all" else [args.model]
    datasets = DEFAULT_DATASETS if args.dataset == "all" else [args.dataset]
    if args.vm14k_baselines:
        if args.model != "all" or args.dataset != "all" or args.generation_mode != "direct":
            parser.error("--vm14k-baselines fixes both models, VM14K, and direct decoding; omit model/dataset overrides")
        models, datasets = list(VM14K_BASELINES), ["vm14k"]
    for dataset in datasets:
        _, _, metadata = prepare_dataset(dataset, args.limit, args.generation_mode)
        print(json.dumps(metadata, indent=2))
    if args.dry_run:
        return 0
    if args.check_runtime:
        print(json.dumps(check_runtime(models), indent=2))
        if not args.check_access:
            return 0
    revisions = check_access(models, os.getenv("HF_TOKEN"))
    if args.check_access:
        return 0
    check_runtime(models)
    if args.vm14k_baselines and args.limit == 0:
        smoke_args = copy.copy(args)
        smoke_args.limit = 4
        smoke_args.output_dir = args.output_dir / "smoke"
        for key in models:
            try:
                evaluate(key, "vm14k", revisions[key], smoke_args)
                smoke = json.loads((smoke_args.output_dir / key / "vm14k" / "summary.json").read_text())
                if smoke["samples"] != 4 or smoke["deviations"] or smoke["token_limit_rate"]:
                    raise RuntimeError("Smoke run did not produce four valid, terminated answers")
            except Exception as error:
                print(f"[medical-eval] SMOKE FAILED {key}: {error}; full evaluation skipped", flush=True)
                return 1
    failures = 0
    for model in models:
        for dataset in datasets:
            try:
                evaluate(model, dataset, revisions[model], args)
            except Exception as error:
                failures += 1
                print(f"[medical-eval] FAILED {model}/{dataset}: {error}", flush=True)
    if args.vm14k_baselines and args.limit == 0 and not failures:
        export_vm14k_comparison(args.output_dir)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
