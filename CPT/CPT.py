import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth import UnslothTrainer, UnslothTrainingArguments

MODEL_NAME = "unsloth/Qwen3-4B-Instruct-2507"
OUTPUT_DIR = "cpt_qwen3_medical"
MAX_SEQ_LENGTH = 2048

# =========================
# LOAD MODEL
# =========================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=256,
    lora_dropout=0,
    bias="none",
    use_rslora=True,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# =========================
# LOAD DATASET
# =========================
dataset = load_dataset(
    "parquet",
    data_files={
        "train": "train-00000-of-00001.parquet",
    },
)["train"]

print("Columns:", dataset.column_names)
print("Sample:", dataset[0])

# =========================
# HELPER: CLEAN + SIMPLIFY TEXT
# =========================
def clean_text(text):
    text = text.strip()
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text

def simple_explanation(text):
    """
    Fake 'explanation' bằng cách:
    - rút gọn
    - giữ ý chính
    """
    text = clean_text(text)

    # cắt bớt để tránh quá dài + giống hệt input
    if len(text) > 300:
        text = text[:300] + "..."

    return text

# =========================
# FORMAT → CHAT TEMPLATE
# =========================
def format_text(example):
    raw_text = example.get("text", "").strip()

    if not raw_text:
        return {"text": ""}

    explanation = simple_explanation(raw_text)

    messages = [
        {
            "role": "system",
            "content": "Bạn là bác sĩ AI, giải thích kiến thức y học cổ truyền một cách dễ hiểu."
        },
        {
            "role": "user",
            "content": raw_text
        },
        {
            "role": "assistant",
            "content": explanation
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    return {"text": text}

dataset = dataset.map(format_text)

# =========================
# TRAIN
# =========================
training_args = UnslothTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=200,
    max_steps=2000,
    learning_rate=5e-5,
    fp16=False,
    bf16=True,
    logging_steps=1,
    optim="adamw_8bit",
    lr_scheduler_type="cosine",
    save_steps=40,
    save_total_limit=2,
)

trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=training_args,
)

trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"LoRA adapter saved to {OUTPUT_DIR} (no merge — SFT sẽ continue-train adapter này)")

print("Training complete.")

# =========================
# FREE GPU MEMORY BEFORE EVAL
# =========================
del model, trainer, tokenizer, dataset
torch.cuda.empty_cache()
import gc
gc.collect()

# =========================
# AUTO-RUN EVAL NOTEBOOKS
# =========================
import subprocess, sys

EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "EVAL")
notebooks = [
    "Unsloth_QA_CPT_new.ipynb",
    "Unsloth_QA_CPT_VM14K.ipynb",
]

total_notebooks = len(notebooks)
print(f"\nStarting eval for CPT ({total_notebooks} notebooks)")

for i, nb in enumerate(notebooks, 1):
    nb_path = os.path.join(EVAL_DIR, nb)
    print(f"\n{'='*60}")
    print(f"[{i}/{total_notebooks}] Running: {nb}")
    print(f"{'='*60}")
    result = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=-1",
            "--output", nb,
            nb_path,
        ],
        cwd=EVAL_DIR,
    )
    if result.returncode == 0:
        print(f"[{i}/{total_notebooks}] Done: {nb}")
    else:
        print(f"[{i}/{total_notebooks}] FAILED: {nb} (exit code {result.returncode})")

print(f"\nAll eval notebooks finished ({total_notebooks}/{total_notebooks})")