#!/usr/bin/env python3
"""Evaluate Vietnamese general benchmarks with vLLM continuous batching and thinking."""
from __future__ import annotations

import argparse
import gc
from functools import lru_cache
import json
import os
from pathlib import Path
import time

from eval_general_vi import (BENCHMARKS, SCORER_COMMIT, boxed, export_comparison,
                             get_math_scorer, parse_choice, prompt as general_prompt,
                             source_rows as general_source_rows)
from eval_medical_baselines import (MODELS as BASELINE_MODELS, build_prompt, prepare_dataset)

MODELS = {**BASELINE_MODELS, "exp0015": "Maverfrick/Venera-Traditional-Model"}
EXP0015_REVISION = "06fc5e88287113fc29b1cdf236dd294260272daa"
EXP0015_BASE = "unsloth/Qwen3-4B-Thinking-2507"
EXP0015_BASE_REVISION = "fb56efbd0f60c5bcd531901fa5fe3ee3157b5135"
EXP0015_CPT_REVISION = "e0589a27e7b66bcbd9cfd8b709c0785d18bf0bf5"

ROOT = Path(__file__).resolve().parents[1]
THINKING_MODEL_KEYS = ("qwen3-4b-thinking-2507", "gemma4-e2b")
MODEL_KEYS = (*THINKING_MODEL_KEYS, "qwen3-4b-cpt", "exp0015")


EVALUATION_BENCHMARKS = (*BENCHMARKS, "vm14k")


@lru_cache(maxsize=1)
def vm14k_protocol():
    return prepare_dataset("vm14k", generation_mode="notebook")


def source_rows(benchmark, token):
    if benchmark != "vm14k":
        return general_source_rows(benchmark, token)
    rows, protocol, metadata = vm14k_protocol()
    return rows, {
        "dataset_repo": "local/VM14K-traditional",
        "dataset_revision": metadata["dataset_sha256"],
        "source_files": {metadata["dataset_file"]: metadata["dataset_sha256"]},
        "total_rows": len(rows), "language": "vi",
        "system_prompt": protocol["system_prompt"],
        "prompt_protocol_sha256": metadata["protocol_sha256"],
        "constrained_to_options": False,
    }


def prompt(benchmark, row):
    if benchmark != "vm14k":
        return general_prompt(benchmark, row)
    _, protocol, _ = vm14k_protocol()
    return build_prompt(row, protocol["system_prompt"])


def final_answer(model_key: str, raw: str, finish_reason: str | None, benchmark: str = "mlogiqa") -> str | None:
    if finish_reason == "length":
        return None
    if model_key == "qwen3-4b-cpt":
        return parse_choice(raw) if benchmark in ("mlogiqa", "include", "vm14k") else boxed(raw)
    if model_key == "gemma4-e2b":
        from vllm.reasoning.gemma4_utils import parse_thinking_output
        raw = parse_thinking_output(raw)["answer"] or ""
    elif "</think>" in raw:
        raw = raw.rsplit("</think>", 1)[1]
    else:
        return None
    return parse_choice(raw) if benchmark in ("mlogiqa", "include", "vm14k") else boxed(raw)


def run(args):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from vllm import LLM, SamplingParams
    import vllm

    token = os.getenv("HF_TOKEN")
    rows, dataset_meta = source_rows(args.benchmark, token)
    total_rows = len(rows)
    is_math = args.benchmark in ("mt-aime24", "polymath")
    math_equal = get_math_scorer(ROOT / "cache" / "general_vi") if is_math else None
    rows = rows[:args.limit] if args.limit else rows
    requested_revision = (getattr(args, "revision", None) or
                          (EXP0015_REVISION if args.model == "exp0015" else None))
    revision = HfApi(token=token).model_info(MODELS[args.model], revision=requested_revision).sha
    engine_model, engine_revision = MODELS[args.model], revision
    lora_request = None
    lora_kwargs = {}
    is_cpt = args.model == "qwen3-4b-cpt"
    is_adapter = args.model in ("qwen3-4b-cpt", "exp0015")
    if is_adapter:
        from vllm.lora.request import LoRARequest
        adapter_config = json.loads(Path(hf_hub_download(
            MODELS[args.model], "adapter_config.json", revision=revision, token=token)).read_text())
        engine_model = adapter_config["base_model_name_or_path"]
        expected_base = EXP0015_BASE if args.model == "exp0015" else "unsloth/Qwen3-4B-Instruct-2507"
        if engine_model != expected_base:
            raise ValueError(f"Unexpected CPT base: {engine_model}; verify its protocol before evaluation")
        engine_revision = HfApi(token=token).model_info(
            engine_model, revision=EXP0015_BASE_REVISION if args.model == "exp0015" else None).sha
        adapter_path = snapshot_download(MODELS[args.model], revision=revision, token=token,
                                        allow_patterns=["adapter_config.json", "adapter_model.safetensors"])
        lora_request = LoRARequest(args.model, 1, adapter_path)
        lora_kwargs = {"enable_lora": True, "max_loras": 1,
                       "max_lora_rank": adapter_config["r"]}
    target = args.output_dir / args.model / args.benchmark
    target.mkdir(parents=True, exist_ok=False)
    decoding = {"do_sample": True, "temperature": 0.6, "top_p": 0.95,
                "top_k": 20, "max_input_tokens": args.max_input_tokens,
                "max_new_tokens": args.max_new_tokens, "thinking_enabled": not is_cpt}
    manifest = {**dataset_meta, "selection": "all", "source_indices": list(range(total_rows)),
                "selected_rows": total_rows, "benchmark": args.benchmark, "model": MODELS[args.model],
                "model_revision": revision, "base_model": engine_model if is_adapter else None,
                "base_revision": engine_revision if is_adapter else None,
                "evaluated_rows": len(rows), "limit": args.limit, "protocol": f"{args.protocol}-vllm",
                "seed": 3407, "batch_size": args.max_num_seqs,
                "quantization": "none-bfloat16", "decoding": decoding,
                "scorer": "PolyMath math_equal + boxed extraction" if is_math else "exact choice accuracy",
                "scorer_revision": SCORER_COMMIT if is_math else None,
                "backend": "vllm", "vllm_version": vllm.__version__,
                "max_num_seqs": args.max_num_seqs,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "enforce_eager": args.enforce_eager,
                "language_model_only": args.model == "gemma4-e2b",
                "status": "running"}
    if args.model == "exp0015":
        manifest.update(cpt_source="Maverfrick/Venera-CPT-TMModel",
                        cpt_source_revision=EXP0015_CPT_REVISION,
                        adapter_composition="CPT adapter continued by SFT; apply final adapter once",
                        adapter_config=adapter_config)
    config_path = target / "run_config.json"
    config_path.write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.monotonic()
    try:
        print(f"[vllm] initializing {args.model}: max_num_seqs={args.max_num_seqs}, "
              f"gpu_memory_utilization={args.gpu_memory_utilization}, "
              f"enforce_eager={args.enforce_eager}, "
              f"language_model_only={manifest['language_model_only']}", flush=True)
        llm = LLM(model=engine_model, revision=engine_revision, dtype="bfloat16",
                  max_model_len=args.max_input_tokens + args.max_new_tokens,
                  max_num_seqs=args.max_num_seqs,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  enforce_eager=args.enforce_eager,
                  # MLogiQA contains text only; disable Gemma media inputs.
                  language_model_only=manifest["language_model_only"],
                  seed=3407, **lora_kwargs)
        print(f"[vllm] loaded {args.model} in {time.monotonic()-started:.1f}s; "
              f"queuing {len(rows)} {args.benchmark} questions", flush=True)
        sampling = SamplingParams(temperature=0.6, top_p=0.95, top_k=20,
                                  min_p=0, max_tokens=args.max_new_tokens, seed=3407)
        chats = [[{"role": "user", "content": prompt(args.benchmark, row)}] for row in rows]
        outputs = llm.chat(chats, sampling_params=sampling,
                           chat_template_kwargs={"enable_thinking": not is_cpt},
                           use_tqdm=True, **({"lora_request": lora_request} if is_adapter else {}))
        tokenizer = llm.get_tokenizer()
        predictions = []
        with (target / "responses.jsonl").open("w", encoding="utf-8") as handle:
            for index, (row, result) in enumerate(zip(rows, outputs)):
                generated = result.outputs[0]
                raw = tokenizer.decode(generated.token_ids, skip_special_tokens=False)
                answer = final_answer(args.model, raw, generated.finish_reason, args.benchmark)
                prediction = {"index": index, **row, "response": raw,
                              "extracted_answer": answer,
                              "is_correct": (bool(math_equal(answer, row["answer"])) if answer is not None and is_math
                                             else answer == row["answer"] if not is_math else False),
                              "finish_reason": generated.finish_reason,
                              "generated_tokens": len(generated.token_ids)}
                predictions.append(prediction)
                handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        if len(predictions) != len(rows):
            raise RuntimeError("vLLM returned fewer responses than questions")
        summary = {**manifest, "status": "complete", "effective_batch_size": args.max_num_seqs,
                   "correct": sum(p["is_correct"] for p in predictions),
                   "accuracy": sum(p["is_correct"] for p in predictions) / len(rows),
                   "invalid_answers": sum(p["extracted_answer"] is None for p in predictions),
                   "token_limit_hits": sum(p["finish_reason"] == "length" for p in predictions),
                   "elapsed_seconds": round(time.monotonic() - started, 2)}
        manifest["status"] = "complete"
        config_path.write_text(json.dumps(manifest, indent=2) + "\n")
        (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (target / "infer_result.txt").write_text(
            f"Model: {MODELS[args.model]}\nBenchmark: {args.benchmark} vi\n"
            f"Correct: {summary['correct']}/{len(rows)}\n"
            f"Accuracy: {summary['accuracy']:.2%}\n"
            f"Invalid answers: {summary['invalid_answers']}\n"
            f"Token limit hits: {summary['token_limit_hits']}\n")
        print(f"[vllm] complete {args.model}: {summary['correct']}/{len(rows)} "
              f"({summary['accuracy']:.2%}), {summary['elapsed_seconds']}s", flush=True)
        del llm
        gc.collect()
        return 0
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        config_path.write_text(json.dumps(manifest, indent=2) + "\n")
        raise


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_KEYS)
    parser.add_argument("--revision", help="Hub model/adapter revision (exp0015 defaults to its pinned commit)")
    parser.add_argument("--benchmark", choices=EVALUATION_BENCHMARKS, default="mlogiqa")
    parser.add_argument("--benchmarks", nargs="+", choices=EVALUATION_BENCHMARKS, default=list(BENCHMARKS),
                        help="Benchmarks included in comparison export")
    parser.add_argument("--protocol", choices=("thinking", "instruct"), default="thinking")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--comparison-models", nargs="+", choices=MODEL_KEYS, default=list(THINKING_MODEL_KEYS))
    parser.add_argument("--export-comparison", action="store_true")
    args = parser.parse_args()
    if args.limit < 0 or args.max_num_seqs < 1 or args.max_new_tokens < 1 or args.max_input_tokens < 1:
        parser.error("limit must be nonnegative; sequence and token counts must be positive")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("GPU memory utilization must be between 0 and 1")
    if args.export_comparison:
        if args.model or args.limit:
            parser.error("comparison export requires full runs of both models")
        export_comparison(args.output_dir, f"{args.protocol}-vllm", args.comparison_models, args.benchmarks)
        return 0
    if args.model is None:
        parser.error("--model is required unless exporting the comparison")
    if (args.model == "qwen3-4b-cpt") != (args.protocol == "instruct"):
        parser.error("CPT requires --protocol instruct; thinking baselines require --protocol thinking")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
