import json
import time
import uuid
from typing import Dict, List
import os

from google import genai
from google.genai import types


from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH)


API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("❌ GOOGLE_API_KEY not found in environment")


# ================= CONFIG =================
# Using the new model name format
MODEL_NAME = "gemini-2.5-flash"
INPUT_FILE = "synthesize_extracted_text.jsonl"
OUT_INSTRUCTION = "processed/dong_y_instruction.jsonl"
OUT_QA = "processed/dong_y_qa.jsonl"
SLEEP = 1.2

# ================= INIT CLIENT =================
client = genai.Client(api_key=API_KEY)

# ================= LLM CALL =================
def call_llm(prompt: str) -> str:
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                top_p=0.95,
                max_output_tokens=2048,
            ),
        )
        return response.text.strip()
    except Exception as e:
        print(f"❌ LLM error: {e}")
        return ""

# ================= STEP 1: ANALYSIS =================
def analyze_text(text: str) -> str:
    prompt = f"""
Bạn là một chuyên gia Đông y cổ truyền, trả lời bằng tiếng Việt, văn phong học thuật.

Phân tích đoạn văn sau theo các bước:
1. Tóm tắt nội dung chính
2. Giải thích các khái niệm Đông y xuất hiện
3. Lập luận bệnh lý (tạng phủ, âm dương, hàn nhiệt nếu có)
4. Ý nghĩa trong chẩn đoán và điều trị

Đoạn văn:
{text}
"""
    return call_llm(prompt)

# ================= STEP 2A: INSTRUCTION DATA =================
def generate_instruction(text: str, analysis: str) -> Dict:
    prompt = f"""
Dựa trên nội dung phân tích sau, hãy tạo MỘT instruction-following sample chất lượng cao cho mô hình Đông y.

Yêu cầu:
- Instruction rõ ràng, mang tính học thuật Đông y
- Output có cấu trúc, lập luận mạch lạc
- Không chép lại nguyên văn đoạn gốc

[Phân tích]
{analysis}
"""
    output = call_llm(prompt)

    return {
        "id": str(uuid.uuid4()),
        "difficulty": "medium",
        "instruction": "Phân tích và giải thích đoạn văn theo lý luận Đông y cổ truyền.",
        "input": text,
        "output": output,
    }

# ================= STEP 2B: QA DATA =================
def generate_qa(text: str, analysis: str) -> List[Dict]:
    prompt = f"""
Dựa trên nội dung sau, hãy tạo bộ câu hỏi – trả lời (QA) theo Đông y.

Yêu cầu:
1. Easy QA: 1 câu
2. Medium QA: 1 câu
3. Hard QA (multi-hop): 1 câu
4. Negative QA: 1 câu hỏi có giả định SAI, kèm giải thích

⚠️ CHỈ trả về JSON hợp lệ, không thêm văn bản ngoài JSON.

Format:
[
  {{
    "difficulty": "easy|medium|hard",
    "type": "factual|reasoning|multi-hop|negative",
    "question": "...",
    "answer": "...",
    "is_negative": true|false
  }}
]

[Nội dung]
{analysis}
"""
    raw = call_llm(prompt)
    if not raw:
        return []

    # Clean up potential markdown code blocks
    raw = raw.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    try:
        qa_list = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"⚠️ QA JSON parse failed: {e}")
        print(f"Raw response: {raw[:200]}...")
        return []

    results = []
    for qa in qa_list:
        results.append({
            "id": str(uuid.uuid4()),
            "difficulty": qa.get("difficulty"),
            "type": qa.get("type"),
            "question": qa.get("question"),
            "context": text,
            "answer": qa.get("answer"),
            "is_negative": qa.get("is_negative", False),
        })

    return results

# ================= MAIN =================
def main():
    os.makedirs("processed", exist_ok=True)

    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file '{INPUT_FILE}' not found!")
        return

    processed_count = 0
    error_count = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUT_INSTRUCTION, "w", encoding="utf-8") as fout_ins, \
         open(OUT_QA, "w", encoding="utf-8") as fout_qa:

        for line_idx, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⚠️ Line {line_idx}: Invalid JSON - {e}")
                error_count += 1
                continue

            text = rec.get("text", "").strip()
            if not text:
                print(f"⚠️ Line {line_idx}: Empty text field")
                continue

            print(f"▶ Processing sample {line_idx}")

            # 1. Analyze
            analysis = analyze_text(text)
            time.sleep(SLEEP)

            if not analysis:
                print(f"⚠️ Line {line_idx}: Analysis failed")
                error_count += 1
                continue

            # 2A. Instruction
            ins_sample = generate_instruction(text, analysis)
            fout_ins.write(json.dumps(ins_sample, ensure_ascii=False) + "\n")
            fout_ins.flush()  # Ensure data is written
            time.sleep(SLEEP)

            # 2B. QA
            qa_samples = generate_qa(text, analysis)
            for qa in qa_samples:
                fout_qa.write(json.dumps(qa, ensure_ascii=False) + "\n")
            fout_qa.flush()  # Ensure data is written
            time.sleep(SLEEP)

            processed_count += 1
            print(f"✓ Sample {line_idx} completed ({processed_count} processed, {error_count} errors)")

    print(f"\n✅ Dataset generation done!")
    print(f"   Processed: {processed_count} samples")
    print(f"   Errors: {error_count} samples")
    print(f"   Output files:")
    print(f"   - {OUT_INSTRUCTION}")
    print(f"   - {OUT_QA}")

if __name__ == "__main__":
    main()