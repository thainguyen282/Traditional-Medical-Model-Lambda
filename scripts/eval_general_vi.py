#!/usr/bin/env python3
"""Evaluate three model checkpoints on four Vietnamese general benchmarks."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tarfile
import time
from urllib.request import urlopen

from eval_medical_baselines import MODELS, check_access, check_runtime, load_model

ROOT = Path(__file__).resolve().parents[1]
MODEL_KEYS = ("qwen3-4b", "gemma4-e2b", "qwen3-4b-cpt")
PAPER_STYLE_MODELS = ("qwen3-4b-paper", "qwen3-4b", "qwen3-4b-cpt")
ALL_MODEL_KEYS = tuple(dict.fromkeys((*MODEL_KEYS, *PAPER_STYLE_MODELS,
                                      "qwen3-4b-thinking-2507")))
BENCHMARKS = ("mlogiqa", "include", "mt-aime24", "polymath")
PROTOCOLS = ("legacy", "paper-style", "thinking")
SAMPLE_SEED = 3407
SOURCES = {
    "mlogiqa": ("Qwen/P-MMEval", ("mlogiqa/test/vi.jsonl",)),
    "include": ("CohereLabs/include-base-44", ("Vietnamese/test-00000-of-00001.parquet",)),
    "mt-aime24": ("amphora/MCLM", ("mt-aime2024.parquet",)),
    "polymath": ("Qwen/PolyMath", tuple(f"vi/{level}.parquet" for level in ("low", "medium", "high", "top"))),
}
# Official PolyMath answer extraction and math_equal. The MCLM paper does not
# publish an executable evaluator, so the same pinned math scorer is used there.
SCORER_COMMIT = "fbf4e41cae78687d6be7447dbea897357c06aaa7"
SCORER_URL = f"https://github.com/QwenLM/PolyMath/archive/{SCORER_COMMIT}.tar.gz"


def resolve_models(models: list[str], token: str | None):
    """Resolve full models and the CPT adapter with its base checkpoint."""
    from huggingface_hub import HfApi, hf_hub_download
    full = [name for name in models if name != "qwen3-4b-cpt"]
    revisions = {name: {"revision": revision} for name, revision in check_access(full, token).items()}
    if "qwen3-4b-cpt" in models:
        api = HfApi(token=token)
        repo = MODELS["qwen3-4b-cpt"]
        revision = api.model_info(repo).sha
        path = hf_hub_download(repo, "adapter_config.json", revision=revision, token=token)
        base = json.loads(Path(path).read_text())["base_model_name_or_path"]
        base_revision = api.model_info(base).sha
        revisions["qwen3-4b-cpt"] = {"revision": revision,
                                     "base_model": base, "base_revision": base_revision}
    return revisions


def load_general_model(key: str, info: dict, token: str | None):
    if key != "qwen3-4b-cpt":
        return load_model(key, info["revision"], token)
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch

    base = info["base_model"]
    model = AutoModelForCausalLM.from_pretrained(
        base, revision=info["base_revision"], token=token,
        torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    model = PeftModel.from_pretrained(model, MODELS[key], revision=info["revision"], token=token)
    tokenizer = AutoTokenizer.from_pretrained(MODELS[key], revision=info["revision"], token=token)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer, None


def source_rows(name: str, token: str | None = None):
    """Pin the Hub revision and hash the exact source files, before any limit."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    repo, files = SOURCES[name]
    revision = HfApi(token=token).repo_info(repo, repo_type="dataset").sha
    rows, hashes = [], {}
    for filename in files:
        path = Path(hf_hub_download(repo_id=repo, filename=filename,
                                    repo_type="dataset", revision=revision, token=token))
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        if filename.endswith(".jsonl"):
            part = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        else:
            part = pq.read_table(path).to_pylist()
        for row in part:
            rows.append(normalize_row(name, row, filename))
    if not rows:
        raise ValueError(f"{name}: empty Vietnamese evaluation split")
    return rows, {"dataset_repo": repo, "dataset_revision": revision,
                  "source_files": hashes, "total_rows": len(rows), "language": "vi"}


def select_rows(name: str, rows: list[dict], metadata: dict, protocol: str):
    """Use a fixed 10% INCLUDE subset for the paper-style profile."""
    if protocol != "paper-style" or name != "include":
        return rows, {**metadata, "selection": "all", "source_indices": list(range(len(rows)))}
    import random
    count = len(rows) // 10
    indices = sorted(random.Random(SAMPLE_SEED).sample(range(len(rows)), count))
    return [rows[i] for i in indices], {
        **metadata, "selection": "10% without replacement, fixed seed",
        "selection_seed": SAMPLE_SEED, "source_indices": indices,
    }


def normalize_row(name: str, row: dict, filename: str) -> dict:
    if name == "mlogiqa":
        options = row["options"]
        question = row["context"] + "\n\n" + row["question"]
        answer = row["answer"]
    elif name == "include":
        options = [row[f"option_{letter}"] for letter in "abcd"]
        question = row["question"]
        answer = row["answer"]
    elif name == "mt-aime24":
        return {"question": row["vi"], "answer": str(row["answer"]),
                "options": None, "source_file": filename}
    elif name == "polymath":
        return {"question": row["question"], "answer": str(row["answer"]),
                "options": None, "source_file": filename, "level": filename.split("/")[1].split(".")[0]}
    else:
        raise ValueError(name)
    if not isinstance(answer, int) or not 0 <= answer < 4 or len(options) != 4:
        raise ValueError(f"{name}: invalid multiple choice row")
    return {"question": question, "answer": "ABCD"[answer],
            "options": options, "source_file": filename}


def get_math_scorer(cache_dir: Path):
    """Load the benchmark author's pinned scorer, including bundled parsers."""
    target = cache_dir / f"PolyMath-{SCORER_COMMIT}" / "eval"
    if not (target / "scripts.py").exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        data = urlopen(SCORER_URL, timeout=60).read()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            members = [m for m in archive.getmembers() if
                       m.name.startswith(f"PolyMath-{SCORER_COMMIT}/eval/") and m.isfile()]
            for member in members:
                path = cache_dir / member.name
                if not path.resolve().is_relative_to(cache_dir.resolve()):
                    raise ValueError("Unsafe scorer archive")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.extractfile(member).read())
    # Upstream's source tree omits this package initializer while scripts.py
    # imports EvaluatorMath from bundled.symeval.
    shim = target / "bundled" / "symeval" / "__init__.py"
    if not shim.exists():
        shim.write_text("from .core import EvaluatorMath\n")
    sys.path.insert(0, str(target))
    from scripts import math_equal
    return math_equal


def boxed(text: str) -> str | None:
    """Use the first complete boxed answer, matching PolyMath run_eval.py."""
    start = text.find(r"\boxed{")
    if start < 0:
        return None
    start += len(r"\boxed{")
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index]
    return None


def parse_choice(text: str) -> str | None:
    # Reasoning may contain tentative letters; score only the final answer.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = re.sub(r"<\|[^>]+\|>|</?[^>]+>", "", text).strip()
    matches = re.findall(r"(?:final answer|đáp án|answer)\s*[:：]?\s*\(?([A-D])\)?", text, re.I)
    if matches:
        return matches[-1].upper()
    # Some chat models put the choice on the first line, then continue with
    # an explanation despite the answer-only instruction.
    first_line = re.match(r"^\s*\(?([A-D])\)?(?:\s|[.。:)])", text, re.I)
    if first_line:
        return first_line.group(1).upper()
    stripped = text.strip()
    if re.fullmatch(r"\(?[A-D]\)?[.。]?", stripped, re.I):
        return re.search("[A-D]", stripped, re.I).group().upper()
    return None


def prompt(name: str, row: dict) -> str:
    if row["options"]:
        options = "\n".join(f"{letter}. {value}" for letter, value in zip("ABCD", row["options"]))
        return f"{row['question']}\n{options}\nTrả lời chỉ bằng chữ cái A, B, C hoặc D."
    return f"{row['question']}\nGiải bài toán và đặt đáp án cuối cùng trong \\boxed{{}}."


def decoding_config(name: str, protocol: str) -> dict:
    if protocol == "thinking":
        return {"do_sample": True, "temperature": 0.6, "top_p": 0.95,
                "top_k": 20, "max_input_tokens": 4096,
                "max_new_tokens": 32768, "thinking_enabled": True}
    if protocol == "paper-style":
        return {"do_sample": True, "temperature": 0.7, "top_p": 0.8,
                "top_k": 20, "presence_penalty": 1.5,
                "max_input_tokens": 4096, "max_new_tokens": 32768,
                "thinking_enabled": False}
    return {"do_sample": False, "max_input_tokens": 4096,
            "max_new_tokens": 64 if name in ("mlogiqa", "include") else 2048,
            "thinking_enabled": False if name in ("mlogiqa", "include") else None}


def generate(model, tokenizer, processor, name: str, rows: list[dict], config: dict,
             protocol: str):
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList

    formatter = processor or tokenizer
    disable_thinking = protocol == "paper-style" or (
        name in ("mlogiqa", "include") and processor is not None)
    if protocol == "thinking":
        disable_thinking = False
    texts = [formatter.apply_chat_template(
        [{"role": "user", "content": prompt(name, row)}],
        tokenize=False, add_generation_prompt=True,
        **({"enable_thinking": protocol == "thinking"} if protocol == "thinking" else
           {"enable_thinking": False} if disable_thinking else {}),
    ) for row in rows]
    inputs = formatter(text=texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=config["max_input_tokens"],
                       add_special_tokens=False).to("cuda")
    kwargs = {key: config[key] for key in ("do_sample", "max_new_tokens")}
    if config["do_sample"]:
        kwargs.update(temperature=config["temperature"], top_p=config["top_p"],
                      top_k=config["top_k"])

        class PresencePenalty(LogitsProcessor):
            def __init__(self, batch_size, vocabulary_size, prompt_length, penalty):
                self.prompt_length = prompt_length
                self.penalty = penalty
                self.seen = torch.zeros((batch_size, vocabulary_size), dtype=torch.bool,
                                        device=inputs.input_ids.device)

            def __call__(self, input_ids, scores):
                if input_ids.shape[1] > self.prompt_length:
                    self.seen.scatter_(1, input_ids[:, -1:], True)
                return scores - self.seen.to(scores.dtype) * self.penalty

        if "presence_penalty" in config:
            kwargs["logits_processor"] = LogitsProcessorList([PresencePenalty(
                len(rows), model.config.vocab_size, inputs.input_ids.shape[1],
                config["presence_penalty"])])
    with torch.inference_mode():
        output = model.generate(**inputs, **kwargs, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.batch_decode(output[:, inputs.input_ids.shape[1]:],
                                  skip_special_tokens=protocol != "thinking")


def score(name: str, row: dict, response: str, math_equal=None) -> dict:
    extracted = parse_choice(response) if row["options"] else boxed(response)
    correct = extracted == row["answer"] if row["options"] else (
        bool(math_equal(extracted, row["answer"])) if extracted and math_equal else False)
    return {**row, "response": response, "extracted_answer": extracted, "is_correct": correct}


def evaluate(model_key: str, name: str, rows: list[dict], dataset_meta: dict,
             model_info: dict, output: Path, batch_size: int, limit: int,
             protocol: str = "legacy", math_equal=None):
    import gc
    import torch
    from transformers import set_seed

    target = output / model_key / name
    target.mkdir(parents=True, exist_ok=False)
    config = decoding_config(name, protocol)
    metadata = {**dataset_meta, "benchmark": name, "model": MODELS[model_key],
                "model_revision": model_info["revision"],
                "base_model": model_info.get("base_model"),
                "base_revision": model_info.get("base_revision"),
                "evaluated_rows": len(rows), "limit": limit,
                "protocol": protocol,
                "seed": 3407, "batch_size": batch_size,
                "quantization": "nf4-double-quant-bfloat16",
                "decoding": config,
                "scorer": ("QwenLM/PolyMath eval/scripts.py math_equal + boxed extraction"
                           if name in ("mt-aime24", "polymath") else "exact choice accuracy"),
                "scorer_revision": SCORER_COMMIT if name in ("mt-aime24", "polymath") else None,
                "status": "running"}
    manifest = target / "run_config.json"
    manifest.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    set_seed(3407)
    model = tokenizer = processor = None
    predictions = []
    try:
        started = time.monotonic()
        print(f"[general-vi] loading {model_key} for {name} "
              f"({len(rows)} questions)", flush=True)
        model, tokenizer, processor = load_general_model(model_key, model_info, os.getenv("HF_TOKEN"))
        print(f"[general-vi] loaded {model_key} for {name} in "
              f"{time.monotonic() - started:.1f}s", flush=True)
        with (target / "responses.jsonl").open("w", encoding="utf-8") as handle:
            while len(predictions) < len(rows):
                batch = rows[len(predictions):len(predictions) + batch_size]
                try:
                    responses = generate(model, tokenizer, processor, name, batch,
                                         config, protocol)
                except torch.cuda.OutOfMemoryError:
                    if batch_size == 1:
                        raise
                    batch_size = max(1, batch_size // 2)
                    torch.cuda.empty_cache()
                    print(f"[general-vi] CUDA OOM; retrying {model_key}/{name} "
                          f"with batch size {batch_size}", flush=True)
                    continue
                for row, response in zip(batch, responses):
                    prediction = {"index": len(predictions), **score(name, row, response, math_equal)}
                    predictions.append(prediction)
                    handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"[general-vi] {model_key}/{name}: "
                      f"{len(predictions)}/{len(rows)} "
                      f"({time.monotonic() - started:.0f}s)", flush=True)
        summary = {**metadata, "status": "complete", "effective_batch_size": batch_size,
                   "correct": sum(item["is_correct"] for item in predictions),
                   "accuracy": sum(item["is_correct"] for item in predictions) / len(rows),
                   "invalid_answers": sum(item["extracted_answer"] is None for item in predictions)}
        metadata["status"] = "complete"
        manifest.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        (target / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        (target / "infer_result.txt").write_text(
            f"Model: {MODELS[model_key]}\nBenchmark: {name}\n"
            f"Correct: {summary['correct']}/{len(rows)}\n"
            f"Accuracy: {summary['accuracy']:.2%}\n"
            f"Invalid answers: {summary['invalid_answers']}\n")
        print(f"[general-vi] complete {model_key}/{name}: "
              f"{summary['correct']}/{len(rows)} ({summary['accuracy']:.2%})", flush=True)
    except Exception as error:
        metadata.update(status="failed", error=str(error), completed_rows=len(predictions))
        manifest.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        raise
    finally:
        del model, tokenizer, processor
        gc.collect()
        torch.cuda.empty_cache()


def export_comparison(output: Path, protocol: str = "legacy",
                      models: list[str] | tuple[str, ...] | None = None,
                      benchmarks: list[str] | tuple[str, ...] | None = None):
    records = []
    models = models or (PAPER_STYLE_MODELS if protocol == "paper-style" else MODEL_KEYS)
    benchmarks = benchmarks or BENCHMARKS
    for name in benchmarks:
        common = None
        for model_key in models:
            target = output / model_key / name
            config = json.loads((target / "run_config.json").read_text())
            summary = json.loads((target / "summary.json").read_text())
            rows = [json.loads(line) for line in (target / "responses.jsonl").read_text().splitlines()]
            if (config["status"] != "complete" or summary["status"] != "complete" or
                config["limit"] != 0 or len(rows) != config["selected_rows"] or
                config.get("protocol", "legacy") != protocol or
                summary["evaluated_rows"] != len(rows) or
                summary["correct"] != sum(row["is_correct"] for row in rows) or
                summary["accuracy"] != summary["correct"] / len(rows) or
                any(row["index"] != index for index, row in enumerate(rows))):
                raise ValueError(f"Incomplete or inconsistent run: {model_key}/{name}")
            identity = (config["dataset_revision"], config["source_files"],
                        config["source_indices"], config["scorer_revision"],
                        config["decoding"], config["selection"])
            if common is not None and identity != common:
                raise ValueError(f"Incomparable dataset/scorer revisions: {name}")
            common = identity
            records.append({"benchmark": name, "model": model_key, "correct": summary["correct"],
                            "total": len(rows), "accuracy": summary["accuracy"]})
    with (output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = [f"# Vietnamese general benchmark results ({protocol})", "",
             "Accuracy on selected Vietnamese splits; math uses the pinned PolyMath scorer.", "",
             "| Benchmark | Model | Correct / Total | Accuracy |", "| --- | --- | ---: | ---: |"]
    lines += [f"| {r['benchmark']} | {r['model']} | {r['correct']} / {r['total']} | {r['accuracy']:.2%} |"
              for r in records]
    (output / "comparison.md").write_text("\n".join(lines) + "\n")


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=ALL_MODEL_KEYS, default=None)
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument("--protocol", choices=PROTOCOLS, default="legacy")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "EVAL/outputs/general_vi")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--export-comparison", action="store_true")
    args = parser.parse_args()
    if args.limit < 0 or args.batch_size < 1:
        parser.error("--limit must be nonnegative and --batch-size positive")
    models = list(dict.fromkeys(args.models or (
        PAPER_STYLE_MODELS if args.protocol == "paper-style" else MODEL_KEYS)))
    if args.export_comparison:
        export_comparison(args.output_dir, args.protocol, models,
                          list(dict.fromkeys(args.benchmarks)))
        return 0
    datasets = {}
    for name in dict.fromkeys(args.benchmarks):
        rows, metadata = source_rows(name, os.getenv("HF_TOKEN"))
        rows, metadata = select_rows(name, rows, metadata, args.protocol)
        metadata["selected_rows"] = len(rows)
        datasets[name] = (rows[:args.limit] if args.limit else rows, metadata)
        print(json.dumps({"benchmark": name, "protocol": args.protocol,
                          "selected_rows": len(rows), "dataset_revision": metadata["dataset_revision"],
                          "selection": metadata["selection"]}, ensure_ascii=False), flush=True)
    if args.dry_run:
        return 0
    revisions = resolve_models(models, os.getenv("HF_TOKEN"))
    check_runtime(models)
    math_equal = (get_math_scorer(Path.home() / ".cache" / "poly_math_scorer")
                  if any(x in args.benchmarks for x in ("mt-aime24", "polymath")) else None)
    failures = 0
    for key in models:
        for name in dict.fromkeys(args.benchmarks):
            rows, metadata = datasets[name]
            try:
                evaluate(key, name, rows, metadata, revisions[key], args.output_dir,
                         args.batch_size, args.limit, args.protocol, math_equal)
            except Exception as error:
                failures += 1
                print(f"[general-vi] FAILED {key}/{name}: {error}", flush=True)
    if not failures and args.limit == 0:
        export_comparison(args.output_dir, args.protocol, models,
                          list(dict.fromkeys(args.benchmarks)))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
