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
OUT_CONVERSATION = "processed/dong_y_conversation.jsonl"
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
Bạn là chuyên gia Đông y cổ truyền.

Phân tích đoạn văn sau theo cấu trúc CỐ ĐỊNH:

I. Tóm tắt nội dung (2–3 câu)
II. Khái niệm Đông y xuất hiện (liệt kê + giải thích ngắn)
III. Lý luận bệnh lý (tạng phủ, âm dương, hàn nhiệt nếu có)
IV. Ý nghĩa trong chẩn đoán và điều trị

⚠️ Không thêm kiến thức ngoài đoạn văn.

[ĐOẠN VĂN]
{text}
"""

    return call_llm(prompt)

# ================= STEP 2A: INSTRUCTION DATA =================
def generate_instruction(text: str, analysis: str) -> Dict:
    prompt = f"""
Dựa trên phân tích sau, hãy tạo MỘT instruction–answer cho mô hình Đông y.

Yêu cầu:
- Instruction mang tính học thuật, rõ nhiệm vụ
- Output có 3 phần:
  1. Tóm tắt ý chính
  2. Phân tích lý luận Đông y
  3. Ứng dụng trong chẩn đoán/điều trị
- Không sao chép nguyên văn đoạn gốc
- Không thêm kiến thức ngoài phân tích

[PHÂN TÍCH]
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
Bạn là chuyên gia xây dựng dataset Question–Answer cho mô hình Đông y.

⚠️ QUY TẮC BẮT BUỘC:
- Chỉ sử dụng thông tin CÓ TRONG nội dung cung cấp
- Không thêm kiến thức Đông y bên ngoài
- Câu trả lời phải ngắn gọn, chính xác, học thuật
- Hard QA phải yêu cầu suy luận từ ÍT NHẤT 2 ý trong nội dung
- Negative QA phải có giả định SAI và câu trả lời phải chỉ ra vì sao sai, dựa trên nội dung

Hãy tạo đúng 4 QA:
1. Easy (factual, trả lời trực tiếp)
2. Medium (reasoning, giải thích ngắn)
3. Hard (multi-hop reasoning)
4. Negative (giả định sai + bác bỏ)

⚠️ CHỈ trả về JSON hợp lệ, KHÔNG markdown, KHÔNG giải thích thêm.

FORMAT CHÍNH XÁC:
[
  {{
    "difficulty": "easy",
    "type": "factual",
    "question": "...",
    "answer": "...",
    "is_negative": false
  }},
  {{
    "difficulty": "medium",
    "type": "reasoning",
    "question": "...",
    "answer": "...",
    "is_negative": false
  }},
  {{
    "difficulty": "hard",
    "type": "multi-hop",
    "question": "...",
    "answer": "...",
    "is_negative": false
  }},
  {{
    "difficulty": "medium",
    "type": "negative",
    "question": "...",
    "answer": "...",
    "is_negative": true
  }}
]

[NỘI DUNG]
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

# ================= STEP 2C: CONVERSATION DATA =================
def generate_conversations(text: str, analysis: str) -> List[Dict]:
    """
    Tạo nhiều conversation samples từ một đoạn văn Đông y
    Mỗi conversation có format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    prompt = f"""
Bạn là chuyên gia Đông y cổ truyền. Từ phân tích dưới đây, hãy tạo 4 CẶP HỘI THOẠI (user-assistant) về Đông y.

⚠️ QUY TẮC BẮT BUỘC:
- CHỈ sử dụng thông tin CÓ TRONG phân tích
- KHÔNG thêm kiến thức Đông y bên ngoài
- Mỗi cặp hội thoại phải có độ khó khác nhau
- Câu trả lời của assistant phải chính xác, học thuật, ngắn gọn

HÃY TẠO 4 CẶP HỘI THOẠI:
1. **Factual** (Easy): Hỏi về thông tin cụ thể trong nội dung
2. **Reasoning** (Medium): Hỏi giải thích lý luận Đông y
3. **Analysis** (Hard): Yêu cầu phân tích nhiều yếu tố kết hợp
4. **Clinical** (Medium): Hỏi về ứng dụng lâm sàng

⚠️ CHỈ trả về JSON hợp lệ, KHÔNG markdown, KHÔNG giải thích thêm.

FORMAT CHÍNH XÁC:
[
  {{
    "user": "Câu hỏi người dùng 1",
    "assistant": "Câu trả lời chuyên gia 1"
  }},
  {{
    "user": "Câu hỏi người dùng 2",
    "assistant": "Câu trả lời chuyên gia 2"
  }},
  {{
    "user": "Câu hỏi người dùng 3",
    "assistant": "Câu trả lời chuyên gia 3"
  }},
  {{
    "user": "Câu hỏi người dùng 4",
    "assistant": "Câu trả lời chuyên gia 4"
  }}
]

[PHÂN TÍCH]
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
        conversations = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"⚠️ Conversation JSON parse failed: {e}")
        print(f"Raw response: {raw[:200]}...")
        return []

    results = []
    for conv in conversations:
        # Chuyển đổi sang format chuẩn cho training
        results.append({
            "id": str(uuid.uuid4()),
            "conversation": [
                {
                    "role": "user",
                    "content": conv.get("user", "")
                },
                {
                    "role": "assistant",
                    "content": conv.get("assistant", "")
                }
            ]
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
    total_conversations = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUT_INSTRUCTION, "w", encoding="utf-8") as fout_ins, \
         open(OUT_QA, "w", encoding="utf-8") as fout_qa, \
         open(OUT_CONVERSATION, "w", encoding="utf-8") as fout_conv:

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
            fout_ins.flush()
            time.sleep(SLEEP)

            # 2B. QA
            qa_samples = generate_qa(text, analysis)
            for qa in qa_samples:
                fout_qa.write(json.dumps(qa, ensure_ascii=False) + "\n")
            fout_qa.flush()
            time.sleep(SLEEP)

            # 2C. Conversation (NEW!)
            conv_samples = generate_conversations(text, analysis)
            if conv_samples:
                for conv in conv_samples:
                    fout_conv.write(json.dumps(conv, ensure_ascii=False) + "\n")
                    total_conversations += 1
                fout_conv.flush()
            else:
                print(f"⚠️ Line {line_idx}: Conversation generation failed")
            time.sleep(SLEEP)

            processed_count += 1
            print(f"✓ Sample {line_idx} completed - Generated {len(conv_samples)} conversations")
            print(f"   Total: {processed_count} processed, {error_count} errors, {total_conversations} conversations")

    print(f"\n✅ Dataset generation done!")
    print(f"   Processed: {processed_count} samples")
    print(f"   Errors: {error_count} samples")
    print(f"   Total conversations: {total_conversations}")
    print(f"   Output files:")
    print(f"   - {OUT_INSTRUCTION}")
    print(f"   - {OUT_QA}")
    print(f"   - {OUT_CONVERSATION}")
    print(f"\n📊 Stats: ~{total_conversations/max(1,processed_count):.1f} conversations per input sample")

if __name__ == "__main__":
    main()