import os
import io
import json
import time
import hashlib
import random
import requests
import argparse
import pandas as pd
import google.generativeai as genai
from pathlib import Path

# ─── CONFIG ──────────────────────────────────────────────────────────────────
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
GEMINI_MODEL    = "gemini-2.5-flash"
OUTPUT_FILE     = "tcm_mcq_dataset.jsonl"

HF_REPO         = "Maverfrick/Continued-Pretraining-Vietnamese-TCM"

# Map split name → tên file parquet trong repo (thư mục data/)
PARQUET_FILES   = {
    "train": "train-00000-of-00001.parquet",
    "test":  "test-00000-of-00001.parquet",
}

MAX_ROWS        = 200
SLEEP_BETWEEN   = 1.5
TEXT_COLUMN     = "text"

MEDICAL_TOPICS  = [
    "Eastern Medicine", "General Medicine", "Cardiology", "Pulmonology",
    "Nephrology", "Gastroenterology", "Neurology", "Hematology",
    "Ophthalmology", "Dermatology", "Gynecology", "Pediatrics",
    "Orthopedics", "Geriatrics", "Pharmacology",
]

# ─── PROMPT ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Bạn là chuyên gia Y học cổ truyền Việt Nam. Nhiệm vụ là tạo câu hỏi trắc nghiệm từ đoạn văn bản y học.

Yêu cầu:
- Tạo đúng 1 câu hỏi trắc nghiệm 4 đáp án từ đoạn văn bản
- Câu hỏi phải liên quan trực tiếp đến nội dung trong đoạn văn
- Có đúng 1 đáp án đúng, 3 đáp án sai nhưng hợp lý (không quá lộ liễu)
- Đáp án sai phải liên quan đến chủ đề y học để tránh dễ đoán
- Độ khó: Easy (khái niệm cơ bản), Moderate (ứng dụng lâm sàng), Hard (phân tích/chẩn đoán phức tạp)

Trả về JSON thuần túy (không có markdown), theo schema:
{
  "difficulty_level": "Easy|Moderate|Hard",
  "medical_topic": ["topic1"],
  "question": "...",
  "options": ["...", "...", "...", "..."],
  "answer": "A|B|C|D"
}

Chủ đề hợp lệ: Eastern Medicine, General Medicine, Cardiology, Pulmonology, Nephrology,
Gastroenterology, Neurology, Hematology, Ophthalmology, Dermatology, Gynecology,
Pediatrics, Orthopedics, Geriatrics, Pharmacology

QUAN TRỌNG: Chỉ trả về JSON, không thêm text nào khác."""


# ─── PARQUET LOADER ───────────────────────────────────────────────────────────

def hf_parquet_url(filename: str) -> str:
    return f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/data/{filename}"


def load_parquet_from_url(url: str, hf_token: str | None = None) -> pd.DataFrame:
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    print(f"⬇️  Downloading: {url}")
    resp = requests.get(url, headers=headers, timeout=180, stream=True)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    buf   = io.BytesIO()
    downloaded = 0
    for chunk in resp.iter_content(chunk_size=256 * 1024):
        buf.write(chunk)
        downloaded += len(chunk)
        if total:
            print(
                f"\r   {downloaded/1e6:.1f} / {total/1e6:.1f} MB "
                f"({downloaded/total*100:.0f}%)",
                end="", flush=True,
            )
    print()
    buf.seek(0)
    return pd.read_parquet(buf)


def load_texts(parquet_arg: str, split: str, hf_token: str | None = None) -> list[str]:
    """
    Load texts từ parquet. Thứ tự ưu tiên:
      1. File local tồn tại  → đọc trực tiếp
      2. URL (bắt đầu http)  → download
      3. Split name / filename → tự build HF URL
    """
    local = Path(parquet_arg)
    if local.exists():
        print(f"📂 Reading local parquet: {local}")
        df = pd.read_parquet(local)
    elif parquet_arg.startswith("http"):
        df = load_parquet_from_url(parquet_arg, hf_token)
    else:
        # parquet_arg có thể là split name ("train"/"test") hoặc tên file parquet
        filename = PARQUET_FILES.get(parquet_arg) \
                or PARQUET_FILES.get(split) \
                or parquet_arg
        url = hf_parquet_url(filename)
        df  = load_parquet_from_url(url, hf_token)

    print(f"✅ Loaded {len(df)} rows | columns: {list(df.columns)}")

    # Detect cột text
    col = TEXT_COLUMN
    if col not in df.columns:
        str_cols = [c for c in df.columns if df[c].dtype == object]
        if str_cols:
            col = str_cols[0]
            print(f"⚠️  Cột '{TEXT_COLUMN}' không tồn tại, dùng '{col}'")
        else:
            raise ValueError(f"Không tìm thấy cột text trong {list(df.columns)}")

    return df[col].dropna().astype(str).tolist()


# ─── MCQ HELPERS ─────────────────────────────────────────────────────────────

def make_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def parse_gemini_response(raw: str) -> dict | None:
    raw = raw.strip()
    # Strip markdown code fence nếu có
    if raw.startswith("```"):
        lines = raw.splitlines()
        end   = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        raw   = "\n".join(lines[1:end])
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        s = raw.find("{")
        e = raw.rfind("}") + 1
        if s != -1 and e > s:
            try:
                return json.loads(raw[s:e])
            except json.JSONDecodeError:
                pass
    return None


def build_mcq_record(gj: dict, source_text: str) -> dict:
    """Chuyển Gemini JSON → format JSONL mong muốn (với shuffle đáp án)"""
    raw_opts = gj.get("options", [])

    # Strip prefix A./B./C./D.
    clean = []
    for opt in raw_opts:
        for p in ("A. ", "B. ", "C. ", "D. ", "A.", "B.", "C.", "D."):
            if opt.startswith(p):
                opt = opt[len(p):]
                break
        clean.append(opt.strip())

    answer_letter = gj.get("answer", "A").strip().upper()
    correct_idx   = {"A": 0, "B": 1, "C": 2, "D": 3}.get(answer_letter, 0)

    # Shuffle giữ track đáp án đúng
    indices           = list(range(len(clean)))
    random.shuffle(indices)
    shuffled_opts     = [clean[i] for i in indices]
    new_answer_idx    = indices.index(correct_idx)
    new_answer_letter = "ABCD"[new_answer_idx]

    # Validate topics
    raw_topics   = gj.get("medical_topic", ["Eastern Medicine"])
    if isinstance(raw_topics, str):
        raw_topics = [raw_topics]
    valid_topics = [t for t in raw_topics if t in MEDICAL_TOPICS] or ["Eastern Medicine"]

    return {
        "id":               make_id(source_text[:100]),
        "difficulty_level": gj.get("difficulty_level", "Moderate"),
        "medical_topic":    valid_topics,
        "question":         gj.get("question", "").strip(),
        "options":          shuffled_opts,
        "option_map":       indices,
        "answer":           new_answer_letter,
        "answer_index":     new_answer_idx,
    }


def generate_mcq(model: genai.GenerativeModel, text: str) -> dict | None:
    prompt = f"Đoạn văn bản y học:\n\n{text[:3000]}\n\nHãy tạo 1 câu hỏi trắc nghiệm từ đoạn trên."
    try:
        resp = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=512,
            ),
        )
        parsed = parse_gemini_response(resp.text)
        if parsed:
            return build_mcq_record(parsed, text)
        print(f"  ⚠️  Parse lỗi: {resp.text[:120]}...")
    except Exception as e:
        print(f"  ❌ Gemini error: {e}")
    return None


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate TCM MCQ dataset with Gemini")
    ap.add_argument(
        "--parquet", default="train",
        help=(
            "Nguồn dữ liệu parquet. Có thể là:\n"
            "  - 'train' hoặc 'test'  (auto-download từ HuggingFace)\n"
            "  - Path file local      (vd: ./train-00000-of-00001.parquet)\n"
            "  - URL trực tiếp        (vd: https://...parquet)"
        ),
    )
    ap.add_argument("--split",     default="train",       help="Tên split khi tự build URL (train|test)")
    ap.add_argument("--hf-token",  default=None,          help="HuggingFace token (dataset private)")
    ap.add_argument("--max-rows",  type=int, default=MAX_ROWS, help="Số rows tối đa (default: 200, 0=toàn bộ)")
    ap.add_argument("--output",    default=OUTPUT_FILE,   help="File JSONL đầu ra")
    ap.add_argument("--api-key",   default=GEMINI_API_KEY,help="Gemini API key")
    ap.add_argument("--sleep",     type=float, default=SLEEP_BETWEEN, help="Giây chờ giữa Gemini calls")
    ap.add_argument("--resume",    action="store_true",   help="Tiếp tục từ lần chạy trước")
    args = ap.parse_args()

    if args.api_key == "YOUR_GEMINI_API_KEY_HERE":
        print("❌  Set GEMINI_API_KEY hoặc dùng --api-key <key>")
        return

    # Setup Gemini
    genai.configure(api_key=args.api_key)
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=SYSTEM_PROMPT)

    # Load parquet → list of texts
    try:
        texts = load_texts(args.parquet, args.split, args.hf_token)
    except Exception as e:
        print(f"❌  Không load được parquet: {e}")
        return

    # Giới hạn rows
    limit = args.max_rows if args.max_rows and args.max_rows > 0 else len(texts)
    texts = texts[:limit]

    # Resume: đọc IDs đã có
    existing_ids = set()
    output_path  = Path(args.output)
    if args.resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    existing_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
        print(f"📂 Resume: đã có {len(existing_ids)} records")

    success = failed = 0
    print(f"\n🚀 Xử lý {len(texts)} rows → {args.output}\n")

    with open(output_path, "a" if args.resume else "w", encoding="utf-8") as out_f:
        for i, text in enumerate(texts):
            rid = make_id(text[:100])
            if rid in existing_ids:
                print(f"  ⏭️  Row {i} đã có, bỏ qua")
                continue

            print(f"  🔄 [{i+1}/{len(texts)}]", end=" ")
            record = generate_mcq(model, text)

            if record:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                success += 1
                print(f"✅ [{record['difficulty_level']}] {record['question'][:70]}...")
            else:
                failed += 1
                print("❌ Bỏ qua")

            time.sleep(args.sleep)

    print(f"\n🎉 Hoàn thành!")
    print(f"   ✅ Thành công : {success}")
    print(f"   ❌ Thất bại   : {failed}")
    print(f"   📄 Output     : {output_path.resolve()}")


if __name__ == "__main__":
    main()