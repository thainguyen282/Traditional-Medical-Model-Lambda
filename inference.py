import torch
import json
import os
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

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"comparison_{timestamp}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Save readable text report
    report_file = os.path.join(output_dir, f"comparison_{timestamp}.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        for i, question in enumerate(questions):
            f.write(f"\n{'='*80}\n")
            f.write(f"Câu hỏi {i+1}: {question}\n")
            f.write(f"{'='*80}\n")
            for r in results:
                if r["question"] == question:
                    f.write(f"\n[{r['model']}]:\n{r['response']}\n")
                    f.write(f"{'-'*40}\n")

    print(f"\n\nResults saved to:\n  - {output_file}\n  - {report_file}")
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
