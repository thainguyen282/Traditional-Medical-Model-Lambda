import torch
import json
import os
import csv
from datetime import datetime
from unsloth import FastLanguageModel


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


def generate_response(model, tokenizer, messages, max_new_tokens=512, temperature=0.7, top_p=0.8, top_k=20):
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
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
        )

    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
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

    models_config = {
        "Qwen3-4B-Instruct-goc": {
            "path": "unsloth/Qwen3-4B-Instruct-2507",
        },
        # --- Hiện tại (dataset Maverfrick/TradMedInstruct-multiplechoice) ---
        "CPT-only": {
            "path": VENERA_CPT,
            "revision": "cpt-tm-cpt-1776775114",
        },
        "CPT-SFT (multiplechoice)": {
            "path": VENERA_SFT,
            "revision": "cpt-sft-unsloth-qwen3-4B-exp0001-tm-sft-1776945292",
        },
        "SFT-only (multiplechoice, regression)": {
            "path": VENERA_SFT,
            "revision": "sft-unsloth-qwen3-4B-exp0001-tm-sft-1776949731",
        },
        # --- Historical (dataset Maverfrick/TradMedInstruct-2topics-3546) ---
        "SFT-only (2topics)": {
            "path": VENERA_SFT,
            "revision": "sft-unsloth-qwen3-4B-exp0001-tm-sft-1776771092",
        },
        "CPT-SFT (2topics)": {
            "path": VENERA_SFT,
            "revision": "cpt-sft-unsloth-qwen3-4B-exp0001-tm-sft-1776784126",
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
