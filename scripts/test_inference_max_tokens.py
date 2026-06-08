import argparse
import json
import csv
import os
from datetime import datetime
import torch
import re
from unsloth import FastLanguageModel
import subprocess
import sys


# Reuse reasoning-strip helper from inference.py to remove leaked tags
_REASONING_TAGS = ("think", "tool_call", "tool_use", "reasoning", "thought",
                   "scratchpad")
_BLOCK_RE = re.compile(
    r"<(?P<t>" + "|".join(_REASONING_TAGS) + r")\b[^>]*>.*?</(?P=t)>",
    re.IGNORECASE | re.DOTALL,
)
_STRAY_TAG_RE = re.compile(
    r"</?(?:" + "|".join(_REASONING_TAGS) + r")\b[^>]*>",
    re.IGNORECASE,
)


def strip_reasoning(text: str) -> str:
    if not text:
        return text
    text = _BLOCK_RE.sub("", text)
    text = _STRAY_TAG_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_model(model_path, revision=None, load_in_4bit=True):
    kwargs = dict(
        model_name=model_path,
        load_in_4bit=load_in_4bit,
        attn_implementation="sdpa",
    )
    if revision:
        kwargs["revision"] = revision
    model, tokenizer = FastLanguageModel.from_pretrained(**kwargs)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_raw(model, tokenizer, prompt: str,
                 max_new_tokens=8096,
                 temperature=0.7, top_p=0.9, top_k=40,
                 repetition_penalty=1.05,
                 strip_tags=True):
    # Tokenize prompt directly (no chat template / system prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            repetition_penalty=repetition_penalty,
        )

    generated_text = tokenizer.decode(
        outputs[0][input_len:],
        skip_special_tokens=True,
    )
    if strip_tags:
        generated_text = strip_reasoning(generated_text)
    return generated_text


DEFAULT_QUESTIONS = [
    "Huyệt Hợp Cốc có vị trí ở đâu và thường được sử dụng để khu phong nhiệt trong trường hợp nào?",
    "Bài thuốc kiện vận tỳ vị có tác dụng táo thấp như thế nào và bạch truật đóng vai trò gì trong phương?",
    "Chứng lạnh tay chân liên quan thế nào đến khí huyết trong Đông y, và bài dẫn huyết có thể cải thiện ra sao?",
    "Vì sao sốt lại là cơ chế tự vệ có lợi cho cơ thể theo Đông y?",
    "Bấm huyệt Thập Chỉ Đạo là gì trong Đông y và có tác dụng điều trị như thế nào?",
]


def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(output_dir, f"raw_inference_{timestamp}")

    json_file = f"{base}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # CSV
    csv_file = f"{base}.csv"
    model_names = sorted({r["model"] for r in results})
    # pivot table: question -> {model: response}
    questions = [r["question"] for r in results if r["model"] == results[0]["model"]]
    table = {q: {m: "" for m in model_names} for q in questions}
    for r in results:
        table[r["question"]][r["model"]] = r["response"]

    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Question"] + model_names)
        for q in questions:
            w.writerow([q] + [table[q][m] for m in model_names])

    md_file = f"{base}.md"
    PREVIEW_LEN = 400

    def md_cell(text: str) -> str:
        if len(text) > PREVIEW_LEN:
            text = text[:PREVIEW_LEN].rstrip() + "…"
        return text.replace("|", "\\|").replace("\n", "<br>")

    with open(md_file, "w", encoding="utf-8") as f:
        f.write(f"# Raw inference results — {timestamp}\n\n")
        f.write(f"Responses trimmed to {PREVIEW_LEN} chars; full text in `{os.path.basename(csv_file)}` / `{os.path.basename(json_file)}`.\n\n")
        f.write("| # | Question | " + " | ".join(model_names) + " |\n")
        f.write("|---|---|" + "|".join(["---"] * len(model_names)) + "|\n")
        for i, q in enumerate(questions, 1):
            row = [str(i), md_cell(q)] + [md_cell(table[q][m]) for m in model_names]
            f.write("| " + " | ".join(row) + " |\n")

    print(f"Results saved to: {json_file}, {csv_file}, {md_file}")
    return base  # return base path without extension


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Model path or HF repo id")
    p.add_argument("--revision", default=None)
    p.add_argument("--questions-file", default=None, help="One question per line (UTF-8)")
    p.add_argument("--output-dir", default="EVAL/outputs/raw_max_tokens")
    p.add_argument("--max-new-tokens", type=int, default=8096)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--push-to-hf-repo", default=None,
                   help="HF repo id to push eval outputs to (optional)")
    p.add_argument("--push-to-hf-revision", default=None,
                   help="HF revision/branch name to push eval outputs to (required if --push-to-hf-repo is set)")
    args = p.parse_args()

    if args.questions_file:
        with open(args.questions_file, "r", encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip()]
    else:
        questions = DEFAULT_QUESTIONS

    models_config = {args.model: {"path": args.model, "revision": args.revision}}

    results = []
    for model_name, cfg in models_config.items():
        model_path = cfg["path"]
        revision = cfg.get("revision")

        print(f"Loading model: {model_name} -> {model_path} {('@'+revision) if revision else ''}")
        model, tokenizer = load_model(model_path, revision=revision)

        for i, q in enumerate(questions, 1):
            print(f"\nQuestion {i}/{len(questions)}: {q[:120]}")
            response = generate_raw(
                model, tokenizer, q,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                strip_tags=True,
            )
            print(f"Response length: {len(response)} chars")

            results.append({
                "model": model_name,
                "model_path": model_path,
                "revision": revision,
                "question": q,
                "response": response,
            })

        del model, tokenizer
        torch.cuda.empty_cache()

    base = save_results(results, args.output_dir)

    # Optional: push outputs to Hugging Face using existing helper script.
    if args.push_to_hf_repo:
        if not args.push_to_hf_revision:
            print("--push-to-hf-revision is required when --push-to-hf-repo is set", file=sys.stderr)
            sys.exit(1)
        # The push script expects --task inference --repo <repo> --revision <rev> --eval-src <dir>
        push_cmd = [
            sys.executable, os.path.join(os.path.dirname(__file__), "lambda_push_to_hf.py"),
            "--task", "inference",
            "--repo", args.push_to_hf_repo,
            "--revision", args.push_to_hf_revision,
            "--eval-src", os.path.join(args.output_dir),
        ]
        env = os.environ.copy()
        print("Pushing results to HF:", " ".join(push_cmd))
        r = subprocess.run(push_cmd, env=env)
        if r.returncode != 0:
            print("Push to HF failed", file=sys.stderr)
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
