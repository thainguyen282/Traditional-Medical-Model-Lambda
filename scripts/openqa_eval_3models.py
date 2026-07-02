#!/usr/bin/env python3
"""
Open-QA (ROUGE) evaluation for THREE models, on one GPU, in one process:

  1. instruct  — unsloth/Qwen3-4B-Instruct-2507       (base, no adapter)
  2. cpt       — <HF_CPT_REPO>@main                    (CPT LoRA adapter)
  3. sft_cpt   — <HF_SFT_REPO>@<latest cpt-sft branch> (SFT-from-CPT adapter)

Adapted from Venera-AI/Trad-Chinese-Medical-Model
  data_pipeline/eval/Unsloth_OpenQA_Eval.ipynb

Eval set: held-out `test` split of New-TradMedInstruct-qa_20k (conversation
schema — same free-form QA the SFT was trained on). For each model we generate
an answer per question and score ROUGE-1/2/L against the reference answer.

Outputs land under $ROOT/EVAL/outputs/OpenQA_Eval/<model_key>/ and the whole
folder is pushed to <HF_SFT_REPO>@eval via scripts/lambda_push_to_hf.py.

Run on Lambda:
  python scripts/lambda_launch.py --task inference \
      --inference-script scripts/openqa_eval_3models.py \
      --instance-type gpu_1x_a100_sxm4 --git-branch main
"""
import os
import sys
import gc
import time
import json
import subprocess

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from rouge_score import rouge_scorer

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
ROOT = os.environ.get("ROOT", os.getcwd())
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_CPT_REPO = os.environ.get("HF_CPT_REPO", "huyennkai/Venera-CPT-TMModel")
HF_SFT_REPO = os.environ.get("HF_SFT_REPO", "huyennkai/Venera-Traditional-Model")
# Latest SFT-from-CPT branch (override via SFT_HF_REVISION if a newer run exists).
SFT_CPT_REVISION = os.environ.get(
    "SFT_HF_REVISION",
    "cpt-sft-unsloth-qwen3-4B-exp0007-tm-sft-1782355488",
)

EVAL_DATASET = "venera-ai/New-TradMedInstruct-qa_20k"
EVAL_SPLIT = "test"
# Cap #questions for a quick run; 0/empty = full split (986 rows).
SUBSET = int(os.environ.get("OPENQA_SUBSET", "0") or 0) or None

MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 24  # halved automatically on OOM
GEN_CONFIG = dict(
    max_new_tokens=250,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    do_sample=False,  # greedy — deterministic, matches the notebook
)

OUT_ROOT = os.path.join(ROOT, "EVAL", "outputs", "OpenQA_Eval")
os.makedirs(OUT_ROOT, exist_ok=True)

MODELS = [
    # (key, path, revision)
    ("instruct", "unsloth/Qwen3-4B-Instruct-2507", None),
    ("cpt", HF_CPT_REPO, "main"),
    ("sft_cpt", HF_SFT_REPO, SFT_CPT_REVISION),
]

_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_eval_set():
    ds = load_dataset(EVAL_DATASET, split=EVAL_SPLIT, token=HF_TOKEN)
    out = []
    for ex in ds:
        conv = ex.get("conversation") or []
        user_msg = assistant_msg = None
        for turn in conv:
            if turn.get("role") == "user":
                user_msg = turn.get("content")
            elif turn.get("role") == "assistant":
                assistant_msg = turn.get("content")
        if user_msg and assistant_msg:
            out.append({"question": user_msg, "answer": assistant_msg})
    if SUBSET:
        out = out[:SUBSET]
    return out


# --------------------------------------------------------------------------
# Inference (one model)
# --------------------------------------------------------------------------
def run_model(key, path, revision, eval_set):
    print(f"\n{'='*70}\n[openqa] Loading {key}: {path}" +
          (f"@{revision}" if revision else "") + f"\n{'='*70}", flush=True)

    kwargs = dict(
        model_name=path,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        attn_implementation="sdpa",
    )
    if revision:
        kwargs["revision"] = revision
    model, tokenizer = FastLanguageModel.from_pretrained(**kwargs)
    FastLanguageModel.for_inference(model)
    # Left padding is required for correct batched decoder generation.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def infer_batch(questions):
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for q in questions
        ]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_SEQ_LENGTH,
        ).to("cuda")
        with torch.inference_mode():
            out = model.generate(**inputs, **GEN_CONFIG)
        gen = [
            tokenizer.decode(out[i][inputs.input_ids.shape[1]:],
                             skip_special_tokens=True)
            for i in range(len(out))
        ]
        del inputs, out
        torch.cuda.empty_cache()
        return gen

    # ---- generate with adaptive batch (OOM -> halve) ----
    preds = []
    t0 = time.time()
    i, bs = 0, BATCH_SIZE
    n = len(eval_set)
    while i < n:
        batch_qs = [s["question"] for s in eval_set[i:i + bs]]
        try:
            preds.extend(infer_batch(batch_qs))
            i += len(batch_qs)
            if i % 120 == 0:
                print(f"  [{key}] {i}/{n}", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect()
            bs = max(4, bs // 2)
            print(f"  [{key}] OOM -> batch_size={bs}", flush=True)
            if bs < 4:
                raise
    total_time = time.time() - t0

    # ---- score ROUGE ----
    out_dir = os.path.join(OUT_ROOT, key)
    os.makedirs(out_dir, exist_ok=True)
    resp_path = os.path.join(out_dir, "responses.jsonl")
    r1 = r2 = rl = 0.0
    empty = 0
    with open(resp_path, "w", encoding="utf-8") as f:
        for idx, (s, pred) in enumerate(zip(eval_set, preds), 1):
            pred = (pred or "").strip()
            ref = s["answer"].strip()
            sc = _scorer.score(ref, pred)
            a, b, c = (sc["rouge1"].fmeasure, sc["rouge2"].fmeasure,
                       sc["rougeL"].fmeasure)
            r1 += a; r2 += b; rl += c
            is_empty = (not pred) or (len(pred.split()) < 3)
            empty += int(is_empty)
            f.write(json.dumps({
                "q_idx": idx, "question": s["question"], "reference": ref,
                "prediction": pred, "rouge1": round(a, 4),
                "rouge2": round(b, 4), "rougeL": round(c, 4),
                "empty": is_empty,
            }, ensure_ascii=False) + "\n")

    m = max(1, len(preds))
    result = {
        "model": key, "path": path, "revision": revision,
        "samples": len(preds),
        "rouge1": round(r1 / m, 4), "rouge2": round(r2 / m, 4),
        "rougeL": round(rl / m, 4),
        "empty_rate": round(empty / m, 4),
        "inference_time_s": round(total_time, 1),
    }
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[openqa] {key} -> {result}", flush=True)

    del model, tokenizer
    torch.cuda.empty_cache(); gc.collect()
    return result


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    if not HF_TOKEN:
        print("[openqa] ERROR: HF_TOKEN not set", file=sys.stderr)
        return 1

    eval_set = load_eval_set()
    print(f"[openqa] eval set: {len(eval_set)} QA pairs "
          f"({EVAL_DATASET}:{EVAL_SPLIT})", flush=True)

    results = []
    for key, path, rev in MODELS:
        try:
            results.append(run_model(key, path, rev, eval_set))
        except Exception as e:  # one bad model shouldn't sink the rest
            print(f"[openqa] WARN: {key} failed: {e}", flush=True)
            results.append({"model": key, "error": str(e)})

    # aggregated comparison table
    with open(os.path.join(OUT_ROOT, "summary_all.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n[openqa] ===== SUMMARY =====")
    for r in results:
        print(" ", r, flush=True)

    # ---- push to <HF_SFT_REPO>@eval ----
    here = os.path.dirname(os.path.abspath(__file__))
    push = os.path.join(here, "lambda_push_to_hf.py")
    print(f"\n[openqa] pushing results to {HF_SFT_REPO}@eval", flush=True)
    rc = subprocess.run([
        sys.executable, push,
        "--task", "eval",
        "--repo", HF_SFT_REPO,
        "--revision", "eval",
        "--eval-src", OUT_ROOT,
    ]).returncode
    if rc != 0:
        print(f"[openqa] WARN: push failed (exit {rc})", flush=True)
        return rc
    print(f"[openqa] done. Results at {HF_SFT_REPO}@eval:eval/OpenQA_Eval/",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
