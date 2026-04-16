import torch
import json
import os
from datetime import datetime
from unsloth import FastLanguageModel


def load_model(model_path, load_in_4bit=True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        load_in_4bit=load_in_4bit,
        attn_implementation="sdpa",
    )
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

    for model_name, model_path in models_config.items():
        print(f"\n{'='*60}")
        print(f"Loading model: {model_name}")
        print(f"Path: {model_path}")
        print(f"{'='*60}")

        model, tokenizer = load_model(model_path)

        for i, question in enumerate(questions):
            print(f"\n--- Question {i+1}: {question[:80]}... ---")
            messages = [{"role": "user", "content": question}]
            response = generate_response(model, tokenizer, messages)
            print(f"[{model_name}]: {response[:200]}...")

            results.append({
                "model": model_name,
                "model_path": model_path,
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
    models_config = {
        "Qwen3-4B-Goc": "unsloth/Qwen3-4B-Instruct-2507",
        # CPT mới (Tier 1, raw-text completion, alpha=32) — load adapter trực tiếp
        "Qwen3-4B-CPT": "CPT/cpt_qwen3_medical",
        "Qwen3-4B-CPT-SFT": "group_outputs/outputs/CPT_SFT/model",
        # SFT-only chưa có model local. Bỏ comment khi train xong:
        # "Qwen3-4B-SFT": "group_outputs/outputs/SFT/model",
    }

    # Bỏ qua các path không tồn tại (skip thay vì crash)
    models_config = {
        name: path for name, path in models_config.items()
        if path.startswith("unsloth/") or os.path.exists(path)
    }
    print(f"Models to compare: {list(models_config.keys())}")

    questions = [
        # === Từ CPT data (sách Đông y cổ truyền) ===
        # CPT: kiến thức về huyệt vị và kinh lạc
        "Huyệt Hợp Cốc có vị trí ở đâu và thường được sử dụng để khu phong nhiệt trong trường hợp nào?",
        # CPT: kinh lạc - hệ thống kinh Tam Tiêu
        "Hệ thống kinh Tam Tiêu Thủ Thiếu Dương có đường đi như thế nào và ý nghĩa lâm sàng là gì?",
        # CPT: bài thuốc cổ phương - Bình Vị Tán / kiện tỳ
        "Bài thuốc kiện vận tỳ vị có tác dụng táo thấp như thế nào và bạch truật đóng vai trò gì trong phương?",
        # CPT: châm cứu - thủ pháp châm
        "Phương pháp châm hành khí đao pháp được thực hiện như thế nào và có tác dụng thông quan tiết gì?",
        # CPT: mạch chẩn - biểu chứng lý chứng
        "Mạch phù chủ biểu chứng, mạch trầm chủ lý chứng được giải thích như thế nào trong tứ chẩn Đông y?",

        # === Từ SFT data (Maverfrick/TradMedInstructData) ===
        # SFT: nhiệt độ cơ thể và ung thư
        "Tại sao nhiệt độ cơ thể thấp lại có thể làm giảm hiệu quả điều trị ung thư theo Đông y?",
        # SFT: khí huyết và chứng lạnh tay chân
        "Chứng lạnh tay chân liên quan thế nào đến khí huyết trong Đông y, và bài dẫn huyết có thể cải thiện ra sao?",
        # SFT: sốt là cơ chế tự vệ
        "Vì sao sốt lại là cơ chế tự vệ có lợi cho cơ thể theo Đông y?",
        # SFT: viêm xoang và luân nhiệt
        "Khi bị viêm xoang, liệu pháp luân nhiệt nóng-lạnh có tác dụng gì theo Đông y?",
        # SFT: bấm huyệt Thập Chỉ Đạo
        "Bấm huyệt Thập Chỉ Đạo là gì trong Đông y và có tác dụng điều trị như thế nào?",
    ]

    compare_models(models_config, questions)
