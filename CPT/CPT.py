import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth import UnslothTrainer, UnslothTrainingArguments

MODEL_NAME = "unsloth/Qwen3-4B-Instruct-2507"
OUTPUT_DIR = "cpt_qwen3_medical"
MAX_SEQ_LENGTH = 2048
EVAL_DATASET_ID = "Maverfrick/Continued-Pretraining-Vietnamese-TCM"

# =========================
# LOAD MODEL
# =========================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
)

# Full CPT mode: train cả embed_tokens + lm_head (LR 10x nhỏ hơn main LR,
# set qua UnslothTrainingArguments.embedding_learning_rate). Cần thiết vì
# dataset có nhiều thuật ngữ y học cổ truyền Việt mà base model chưa phủ kín.
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    # alpha=r với rsLoRA → scale nhất quán với SFT, tránh shock khi continue-train.
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_rslora=True,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "embed_tokens", "lm_head",
    ],
)

# =========================
# HELPER: CLEAN TEXT
# =========================
def clean_text(text):
    text = text.strip()
    # Giữ xuống dòng để bảo toàn cấu trúc đoạn, chỉ chuẩn hoá khoảng trắng thừa.
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)

# CPT = continual pre-training: train next-token prediction trên text y học thuần.
# KHÔNG bọc chat template, KHÔNG fake user/assistant turn — tránh dạy model hành vi
# "trả lời bằng cách copy lại câu hỏi" gây format collapse ở bước SFT.
def format_text(example):
    raw_text = example.get("text", "")
    if not raw_text:
        return {"text": ""}
    return {"text": clean_text(raw_text) + tokenizer.eos_token}

# =========================
# LOAD TRAIN DATASET (local parquet)
# =========================
train_dataset = load_dataset(
    "parquet",
    data_files={"train": "train-00000-of-00001.parquet"},
)["train"]
print("Train cols:", train_dataset.column_names, "rows:", len(train_dataset))

train_dataset = train_dataset.map(format_text)
train_dataset = train_dataset.filter(lambda x: len(x["text"]) > 0)

# =========================
# LOAD EVAL DATASET (HF — held-out TCM text)
# =========================
# Dùng split "test" của cùng source data để monitor CPT có học được domain
# hay không. eval_loss giảm = học được; tăng nhanh = overfit hoặc format collapse.
eval_dataset = load_dataset(EVAL_DATASET_ID, split="test")
print("Eval cols:", eval_dataset.column_names, "rows:", len(eval_dataset))
eval_dataset = eval_dataset.map(format_text)
eval_dataset = eval_dataset.filter(lambda x: len(x["text"]) > 0)

# =========================
# TRAIN
# =========================
training_args = UnslothTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    # warmup_ratio thay cho warmup_steps — tự scale theo tổng steps.
    warmup_ratio=0.03,
    num_train_epochs=2,
    # CPT trên model Instruct → LR thấp hơn SFT-from-base để giữ alignment.
    learning_rate=3e-5,
    # 10× nhỏ hơn main LR: embedding dễ drift, LR cao sẽ phá vocab.
    embedding_learning_rate=3e-6,
    fp16=False,
    bf16=True,
    logging_steps=10,
    optim="adamw_8bit",
    lr_scheduler_type="cosine",
    save_steps=500,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=200,
    seed=3407,
)

trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    # Pack short docs thành sequence đủ max_seq_length → bỏ hết padding,
    # compute không bị phí cho token <pad>.
    packing=True,
    args=training_args,
)

trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"LoRA adapter saved to {OUTPUT_DIR} (no merge — SFT sẽ continue-train adapter này)")

print("Training complete.")

# =========================
# FREE GPU MEMORY BEFORE PUSH + EVAL
# =========================
del model, trainer, tokenizer, train_dataset, eval_dataset
torch.cuda.empty_cache()
import gc
gc.collect()

# =========================
# PUSH ADAPTER TO HF (so eval loads the exact published version)
# =========================
import subprocess, sys, shutil, time

HERE = os.path.dirname(os.path.abspath(__file__))
PUSH_SCRIPT = os.path.abspath(os.path.join(HERE, "..", "scripts", "lambda_push_to_hf.py"))
EVAL_DIR = os.path.abspath(os.path.join(HERE, "..", "EVAL"))
EVAL_OUTPUTS_DIR = os.path.join(EVAL_DIR, "outputs")
HOST_LOG = "/host/train.log"  # bind-mounted from Lambda host

HF_CPT_REPO = os.environ.get("HF_CPT_REPO", "huyennkai/Venera-CPT-TMModel")
JOB_TAG = os.environ.get("JOB_TAG") or f"local-{int(time.time())}"
# Must mirror lambda_push_to_hf.py's naming: `cpt-<slugified_tag>`.
HF_REVISION = f"cpt-{JOB_TAG.replace('_', '-')}"

print(f"\n[CPT.py] pushing adapter to {HF_CPT_REPO}@{HF_REVISION}")
rc = subprocess.run([
    sys.executable, PUSH_SCRIPT,
    "--task", "cpt",
    "--repo", HF_CPT_REPO,
    "--src", OUTPUT_DIR,
    "--revision", JOB_TAG,
], cwd=HERE).returncode
if rc != 0:
    print(f"[CPT.py] ERROR: model push failed (exit {rc}); aborting before eval.")
    sys.exit(rc)

# =========================
# CLEAN PRIOR EVAL OUTPUTS (so we only push this run's results)
# Wipe everything under EVAL/outputs/ — if the repo accidentally ships
# older results (tracked by mistake), they won't leak into this run's HF
# revision. The notebooks recreate the subdirs they need.
# =========================
if os.path.isdir(EVAL_OUTPUTS_DIR):
    for entry in os.listdir(EVAL_OUTPUTS_DIR):
        p = os.path.join(EVAL_OUTPUTS_DIR, entry)
        if os.path.isdir(p):
            shutil.rmtree(p)
        else:
            os.remove(p)

print(f"\n[CPT.py] done. Adapter on HF: {HF_CPT_REPO}@{HF_REVISION}")
print(f"[CPT.py] Eval will run in a separate docker run from bootstrap.sh — "
      f"this avoids CUDA/NVML corruption caused by forking after training.")