import torch
import json
import os
import csv
import re
from datetime import datetime
from unsloth import FastLanguageModel


# =========================================================================
# Post-processing: strip reasoning / tool-call artifacts that leak from the
# base model's Qwen3 chat scaffolding when SFT pushes it into an MCQ-only
# register. Removed both paired blocks (<tag>…</tag>) and any stray opening
# or closing tags that leak alone.
# =========================================================================
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


SYSTEM_PROMPT = (
    "Bạn là chuyên gia Y học cổ truyền. Trả lời câu hỏi bằng văn bản tự nhiên "
    "tiếng Việt, cấu trúc rõ ràng, dẫn giải cơ chế và nêu ví dụ minh họa khi "
    "phù hợp. TUYỆT ĐỐI KHÔNG dùng <tool_call>, <think>, <reasoning> hay bất "
    "kỳ thẻ hệ thống nào. Trả lời trực tiếp, đầy đủ, không viết tắt."
)


def generate_response(model, tokenizer, messages,
                      max_new_tokens=1536,
                      min_new_tokens=256,
                      temperature=0.7, top_p=0.9, top_k=40,
                      repetition_penalty=1.05,
                      use_system_prompt=True,
                      strip_tags=True):
    # Prepend system prompt if caller hasn't already supplied one. Helps keep
    # SFT-tuned models (which overfit to MCQ brevity) from truncating.
    if use_system_prompt and (not messages or messages[0].get("role") != "system"):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            repetition_penalty=repetition_penalty,
        )

    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    if strip_tags:
        generated_text = strip_reasoning(generated_text)
    return generated_text


def compare_models(models_config, questions, output_dir="EVAL/outputs/comparison"):
    os.makedirs(output_dir, exist_ok=True)

    results = []

    for model_name, cfg in models_config.items():
        # Support both old flat str form and new {path, revision} form.
        if isinstance(cfg, str):
            model_path, revision = cfg, None
        else:
            model_path, revision = cfg["path"], cfg.get("revision")

        print(f"\n{'='*60}")
        print(f"Loading model: {model_name}")
        print(f"Path: {model_path}" + (f"  @ {revision}" if revision else ""))
        print(f"{'='*60}")

        model, tokenizer = load_model(model_path, revision=revision)

        for i, question in enumerate(questions):
            print(f"\n--- Question {i+1}: {question[:80]}... ---")
            messages = [{"role": "user", "content": question}]
            response = generate_response(model, tokenizer, messages)
            print(f"[{model_name}]: {response[:200]}...")

            results.append({
                "model": model_name,
                "model_path": model_path,
                "revision": revision,
                "question": question,
                "response": response,
            })

        # Free GPU memory before loading next model
        del model, tokenizer
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Save results as a table (rows = questions, cols = models).
    # 3 formats:
    #   - .json : structured, per-cell (model, question, response, revision)
    #   - .csv  : full-text table, open trong Excel / Google Sheet
    #   - .md   : Markdown table, responses trimmed cho readable preview
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(output_dir, f"comparison_{timestamp}")

    json_file = f"{base}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Pivot: table[question] = {model_name: response}
    model_names = list(models_config.keys())
    table = {q: {m: "" for m in model_names} for q in questions}
    for r in results:
        table[r["question"]][r["model"]] = r["response"]

    # --- CSV (full text, each cell = full response) -------------------
    csv_file = f"{base}.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Question"] + model_names)
        for q in questions:
            w.writerow([q] + [table[q][m] for m in model_names])

    # --- Markdown (GitHub/HF-friendly preview) ------------------------
    PREVIEW_LEN = 400

    def md_cell(text: str) -> str:
        """Escape pipes, collapse newlines to <br>, trim for readability."""
        if len(text) > PREVIEW_LEN:
            text = text[:PREVIEW_LEN].rstrip() + "…"
        return text.replace("|", "\\|").replace("\n", "<br>")

    md_file = f"{base}.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(f"# Inference comparison — {timestamp}\n\n")
        f.write(f"Responses trimmed to {PREVIEW_LEN} chars; full text in "
                f"`{os.path.basename(csv_file)}` / `{os.path.basename(json_file)}`.\n\n")
        f.write("| # | Question | " + " | ".join(model_names) + " |\n")
        f.write("|---|---|" + "|".join(["---"] * len(model_names)) + "|\n")
        for i, q in enumerate(questions, 1):
            row = [str(i), md_cell(q)] + [md_cell(table[q][m]) for m in model_names]
            f.write("| " + " | ".join(row) + " |\n")

    print(f"\n\nResults saved to:\n  - {json_file}\n  - {csv_file}\n  - {md_file}")
    return results


if __name__ == "__main__":
    # 6 models: Instruct baseline + 5 finetuned variants from README
    # (see README.md "🎯 Model output" section).
    VENERA_SFT = "huyennkai/Venera-Traditional-Model"
    VENERA_CPT = "huyennkai/Venera-CPT-TMModel"

    # Focused: chỉ so sánh base vs CPT-only@main để check riêng CPT adapter mới.
    # (Bỏ các bản SFT — xem git history nếu cần lại bảng so sánh 6 model.)
    models_config = {
        "Qwen3-4B-Instruct-goc": {
            "path": "unsloth/Qwen3-4B-Instruct-2507",
        },
        "CPT-only": {
            "path": VENERA_CPT,
            # CPT adapter giờ push thẳng vào branch `main` (xem CPT.py).
            "revision": "main",
        },
    }

    print(f"Models to compare: {list(models_config.keys())}")

    # 5 câu đại diện: 2 câu CPT-flavored (kiến thức nền YHCT), 3 câu SFT-flavored
    # (QA / ứng dụng lâm sàng) — đủ để so sánh định tính giữa các model.
    questions = [
        "Huyệt Hợp Cốc có vị trí ở đâu và thường được sử dụng để khu phong nhiệt trong trường hợp nào?",
        "Bài thuốc kiện vận tỳ vị có tác dụng táo thấp như thế nào và bạch truật đóng vai trò gì trong phương?",
        "Chứng lạnh tay chân liên quan thế nào đến khí huyết trong Đông y, và bài dẫn huyết có thể cải thiện ra sao?",
        "Vì sao sốt lại là cơ chế tự vệ có lợi cho cơ thể theo Đông y?",
        "Bấm huyệt Thập Chỉ Đạo là gì trong Đông y và có tác dụng điều trị như thế nào?",
    ]

    compare_models(models_config, questions)
