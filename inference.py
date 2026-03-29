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
        "Qwen3-4B-CPT": "CPT/cpt_qwen3_medical_merged",
        "Qwen3-4B-CPT-SFT": "group_outputs/outputs/29_3_exp/model",
    }

    questions = [
        # Lý luận cơ bản YHCT
        "Trình bày học thuyết Âm Dương trong Y học cổ truyền và ứng dụng trong chẩn đoán bệnh.",
        "Ngũ hành tương sinh tương khắc là gì? Cho ví dụ ứng dụng trong điều trị.",
        # Chẩn đoán
        "Tứ chẩn trong Đông y gồm những gì? Giải thích từng phương pháp.",
        "Mạch Huyền thường gặp trong bệnh lý gì? Cách xử trí theo YHCT?",
        # Bài thuốc
        "Thành phần và công dụng của bài thuốc Tứ Vật Thang là gì?",
        "Bài thuốc Lục Vị Địa Hoàng Hoàn dùng trong trường hợp nào?",
        # Châm cứu
        "Huyệt Hợp Cốc có tác dụng gì và thường dùng trong điều trị bệnh gì?",
        "Trình bày đường đi của kinh Phế thủ thái âm và các huyệt quan trọng.",
    ]

    compare_models(models_config, questions)
