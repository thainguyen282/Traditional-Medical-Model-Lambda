import json
import time
import uuid
import os
from typing import List, Tuple

from google import genai
from google.genai import types
from dotenv import load_dotenv

# ================= PATH & ENV =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("❌ GOOGLE_API_KEY not found")

# ================= CONFIG =================
MODEL_NAME = "gemini-2.5-flash"
INPUT_FILE = "synthesize_extracted_text.jsonl"
OUT_CONVERSATION = "proccessed/dong_y_conversation.jsonl"
SLEEP = 1.2

client = genai.Client(api_key=API_KEY)

# ================= LLM CALL =================
def call_llm(prompt: str) -> str:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.25,
            top_p=0.95,
            max_output_tokens=1024,
        ),
    )
    return response.text.strip()

# ================= STEP 1: ANALYSIS =================
def analyze_text(text: str) -> str:
    prompt = f"""
Bạn là một trợ lý AI chuyên phân tích học thuật Đông y,
mục tiêu là trích xuất **lập luận y học** phục vụ huấn luyện mô hình reasoning.

YÊU CẦU CHUNG:
- Phân tích sâu, có diễn giải logic
- Trình bày rõ mối quan hệ nguyên nhân → cơ chế → hệ quả
- Không bổ sung kiến thức ngoài văn bản

Hãy phân tích đoạn văn theo ĐÚNG cấu trúc:

I. Nội dung cốt lõi
II. Thuật ngữ và khái niệm Đông y
III. Lý luận bệnh lý và sinh lý
IV. Hàm ý cho chẩn đoán và điều trị

[ĐOẠN VĂN]
{text}
"""
    return call_llm(prompt)

# ================= STEP 2: SINGLE QA =================
def generate_single_qa(analysis: str, qa_type: str) -> Tuple[str, str]:
    prompt = f"""
Bạn là trợ lý AI Đông y học thuật.

Từ phân tích dưới đây, hãy tạo 1 CẶP hỏi–đáp loại: {qa_type}

YÊU CẦU:
- Chỉ dùng thông tin trong phân tích
- Không thêm kiến thức ngoài
- Không xưng danh
- Trả lời có lập luận rõ ràng
- Trả về đúng 2 dòng:

User: ...
Assistant: ...

[PHÂN TÍCH]
{analysis}
"""
    out = call_llm(prompt)

    user, assistant = "", ""
    for line in out.splitlines():
        if line.startswith("User:"):
            user = line.replace("User:", "").strip()
        elif line.startswith("Assistant:"):
            assistant = line.replace("Assistant:", "").strip()

    return user, assistant

# ================= MAIN =================
def main():
    os.makedirs("proccessed", exist_ok=True)

    processed = 0
    total_convs = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUT_CONVERSATION, "w", encoding="utf-8") as fout:

        for idx, line in enumerate(fin, 1):
            rec = json.loads(line)
            text = rec.get("text", "").strip()
            if not text:
                continue

            print(f"▶ Processing sample {idx}")

            analysis = analyze_text(text)
            time.sleep(SLEEP)

            qa_types = [
                "Factual (Easy)",
                "Reasoning (Medium)",
                "Synthesis (Hard)",
                "Clinical-oriented (Medium)",
            ]

            for qa_type in qa_types:
                user, assistant = generate_single_qa(analysis, qa_type)
                time.sleep(SLEEP)

                if not user or not assistant:
                    continue

                fout.write(json.dumps({
                    "id": str(uuid.uuid4()),
                    "conversation": [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": assistant},
                    ]
                }, ensure_ascii=False) + "\n")

                total_convs += 1

            processed += 1
            print(f"✓ Sample {idx} done | Total convs: {total_convs}")

    print("\n✅ DONE")
    print(f"Processed texts: {processed}")
    print(f"Total conversations: {total_convs}")
    print(f"Avg conv / text: {total_convs / max(1, processed):.2f}")

if __name__ == "__main__":
    main()
