import os
import json
import uuid
import random
import time
import re
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import google.generativeai as genai

# ======================
# CONFIG
# ======================
PARQUET_FILES = [
    "train-00000-of-00001.parquet",
    "test-00000-of-00001.parquet",
]

OUTPUT_FILE = "tcm_mcq.jsonl"
MAX_SAMPLES_PER_FILE = 200
MAX_CONTEXT_LENGTH = 1000

# ======================
# LOAD ENV + MULTI KEYS
# ======================
load_dotenv()

API_KEYS = os.getenv("GEMINI_API_KEYS", "").split(",")

if not API_KEYS or API_KEYS == [""]:
    raise ValueError("❌ No API keys found in GEMINI_API_KEYS")

key_index = 0
key_fail_count = {k: 0 for k in API_KEYS}
key_cooldown = {k: 0 for k in API_KEYS}


def get_next_key():
    global key_index

    for _ in range(len(API_KEYS)):
        key = API_KEYS[key_index]
        key_index = (key_index + 1) % len(API_KEYS)

        if time.time() < key_cooldown[key]:
            continue

        return key

    return None


def get_model():
    key = get_next_key()

    if not key:
        print("⛔ All API keys exhausted. Sleeping 30s...")
        time.sleep(30)
        return get_model()

    genai.configure(api_key=key)
    return genai.GenerativeModel("gemini-2.5-flash"), key


# ======================
# RATE LIMIT GLOBAL
# ======================
LAST_CALL = 0
MIN_INTERVAL = 1.5


def rate_limit():
    global LAST_CALL
    now = time.time()
    wait = MIN_INTERVAL - (now - LAST_CALL)
    if wait > 0:
        time.sleep(wait)
    LAST_CALL = time.time()


# ======================
# LOAD EXISTING
# ======================
def load_existing_questions(file_path):
    questions = set()

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    q = json.loads(line)["question"]
                    questions.add(normalize_text(q))
                except:
                    continue

    return questions


# ======================
# HELPERS
# ======================
def generate_id():
    return uuid.uuid4().hex


def normalize_text(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def shuffle_options(options, correct_index):
    indexed = list(enumerate(options))
    random.shuffle(indexed)

    new_options = [opt for _, opt in indexed]
    option_map = [orig_idx for orig_idx, _ in indexed]

    new_correct_index = option_map.index(correct_index)

    return new_options, option_map, new_correct_index


def clean_json(text):
    text = re.sub(r"```json|```", "", text)
    start = text.find("{")
    end = text.rfind("}") + 1
    return text[start:end]


def safe_parse(text):
    try:
        text = clean_json(text)
        return json.loads(text)
    except:
        return None


def is_bad_question(q):
    q = q.lower()
    blacklist = [
        "đoạn văn", "nội dung sau", "dưới đây",
        "theo đoạn", "dựa vào", "cho đoạn"
    ]
    return any(x in q for x in blacklist)


# ======================
# GEMINI CALL (MULTI KEY)
# ======================
def call_gemini(context, retries=5):
    prompt = f"""
Tạo 1 câu hỏi trắc nghiệm y học cổ truyền từ kiến thức dưới đây.

YÊU CẦU:
- Câu hỏi phải TỰ ĐỨNG ĐỘC LẬP
- KHÔNG dùng các cụm như:
  "theo đoạn văn", "dựa vào nội dung sau", "cho đoạn sau"
- Giống đề thi thật

- Có 4 đáp án
- Chỉ 1 đáp án đúng
- Không giải thích
- Tiếng Việt

Format JSON:
{{
"question": "...",
"options": ["...", "...", "...", "..."],
"answer_text": "đáp án đúng"
}}

Kiến thức:
{context}
"""

    delay = 1

    for attempt in range(retries):
        model, key = get_model()

        try:
            rate_limit()

            res = model.generate_content(prompt)

            if not res.text:
                raise ValueError("Empty response")

            data = safe_parse(res.text)

            if not data:
                raise ValueError("JSON parse failed")

            if (
                "question" in data and
                "options" in data and
                "answer_text" in data and
                len(data["options"]) == 4
            ):
                return data

            raise ValueError("Invalid schema")

        except Exception as e:
            err = str(e)
            print(f"[Retry {attempt+1}] Key={key[:8]}... Error:", err[:120])

            key_fail_count[key] += 1

            # quota / rate limit
            if "quota" in err.lower() or "429" in err:
                key_cooldown[key] = time.time() + 60
                print(f"⏳ Cooldown 60s: {key[:8]}...")

            # fail nhiều → cooldown lâu hơn
            if key_fail_count[key] >= 3:
                key_cooldown[key] = time.time() + 120
                key_fail_count[key] = 0

            time.sleep(delay)
            delay = min(delay * 2, 10)

    return None


# ======================
# PROCESS ANSWER
# ======================
def process_answer(options, answer_text):
    if answer_text not in options:
        return None

    correct_index = options.index(answer_text)

    new_options, option_map, new_index = shuffle_options(
        options, correct_index
    )

    return new_options, option_map, new_index, answer_text


def infer_topic(text):
    text = text.lower()
    topics = ["Eastern Medicine"]

    if "phế" in text:
        topics.append("Pulmonology")
    if "thận" in text:
        topics.append("Nephrology")
    if "tâm" in text:
        topics.append("Cardiology")
    if "thần kinh" in text:
        topics.append("Neurology")

    return list(set(topics))


def infer_difficulty(text):
    l = len(text)
    if l < 80:
        return "Easy"
    elif l < 200:
        return "Moderate"
    return "Hard"


# ======================
# MAIN PROCESS
# ======================
def process_parquet(file_path, f, existing_questions):
    df = pd.read_parquet(file_path)
    df = df.sample(min(MAX_SAMPLES_PER_FILE, len(df)))

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            context = " ".join([str(v) for v in row.values])
            context = context[:MAX_CONTEXT_LENGTH]

            data = call_gemini(context)
            if not data:
                print("❌ Skip: Gemini failed")
                continue

            question = data["question"]
            options = data["options"]
            answer_text = data["answer_text"]

            if is_bad_question(question):
                print("⚠️ Skip: bad question")
                continue

            norm_q = normalize_text(question)
            if norm_q in existing_questions:
                print("⚠️ Skip: duplicate")
                continue

            processed = process_answer(options, answer_text)
            if not processed:
                print("⚠️ Skip: answer mismatch")
                continue

            options, option_map, new_idx, answer_text = processed

            sample = {
                "id": generate_id(),
                "difficulty_level": infer_difficulty(question),
                "medical_topic": infer_topic(question),
                "question": question,
                "options": options,
                "option_map": option_map,
                "answer": answer_text,
                "answer_index": new_idx
            }

            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            f.flush()

            existing_questions.add(norm_q)

            # random delay chống detect
            time.sleep(random.uniform(1.5, 3.5))

        except Exception as e:
            print("🔥 Unexpected error:", str(e)[:100])
            continue


def main():
    existing_questions = load_existing_questions(OUTPUT_FILE)
    print(f"Loaded {len(existing_questions)} existing questions")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for file in PARQUET_FILES:
            print(f"\n🚀 Processing {file}...")
            process_parquet(file, f, existing_questions)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()