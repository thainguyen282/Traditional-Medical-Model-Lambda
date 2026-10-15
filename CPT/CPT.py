import argparse
import json
import os
import yaml
from pathlib import Path
# Resolve and validate YAML before importing GPU libraries or accessing the Hub.
REPO_ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Continual pretraining with optional YAML config")
parser.add_argument("--config", type=Path)
parser.add_argument("--print-config", action="store_true", help="Print overrides and exit without loading a GPU")
cli = parser.parse_args()
config = {}
if cli.config:
    config_path = cli.config if cli.config.is_absolute() else REPO_ROOT / cli.config
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        parser.error("CPT config must be a YAML mapping")
    for section in ("model", "dataset", "unsloth", "training", "lora"):
        if not isinstance(config.get(section, {}), dict):
            parser.error(f"{section} must be a mapping")
    if config.get("dataset", {}).get("add_chat_template", False):
        parser.error("CPT requires raw text: add_chat_template must be false")
if cli.print_config:
    print(json.dumps(config, indent=2))
    raise SystemExit(0)
model_config = config.get("model", {})
data_config = config.get("dataset", {})
unsloth_config = config.get("unsloth", {})

from dotenv import load_dotenv
from huggingface_hub import HfApi, get_token
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth import UnslothTrainer, UnslothTrainingArguments

MODEL_NAME = model_config.get("name", "unsloth/Qwen3-4B-Instruct-2507")
OUTPUT_DIR = "cpt_qwen3_medical"
MAX_SEQ_LENGTH = model_config.get("max_seq_length", 2048)
# Dataset đã pre-chunk ~2048 tokens và có sẵn split train/validation.
DATASET_ID = data_config.get("name", "Maverfrick/TradMed-CPT-53k-2048")
TEXT_FIELD = data_config.get("text_field", "text")

# Load credentials for direct runs as well as Docker-provided environments.
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
HF_TOKEN = get_token()
if not HF_TOKEN:
    raise RuntimeError(
        f"{DATASET_ID} is private. Set HF_TOKEN in the environment or repository "
        ".env, or log in with Hugging Face before running CPT."
    )
try:
    HfApi(token=HF_TOKEN).dataset_info(DATASET_ID)
except Exception:
    raise RuntimeError(
        f"Cannot access private dataset {DATASET_ID}. Check that HF_TOKEN is "
        "valid and has read access to this venera-ai dataset; also check Hub "
        "connectivity. Browser login does not authenticate the training process."
    ) from None

# =========================
# LOAD MODEL
# =========================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=unsloth_config.get("load_in_4bit", False),
)

# LoRA chỉ trên linear projections — KHÔNG train embed_tokens / lm_head.
# Dataset mới TradMed-CPT-53k-2048 ~ 15–30M tokens (51.4k chunk × ~600–900 tok),
# vẫn DƯỚI ngưỡng 50M để dịch chuyển embedding một cách có ích, và việc đụng
# input/output representation trên 1 model đã được Instruct align rất dễ gây
# format collapse. Chỉ thêm "embed_tokens" + "lm_head" vào list dưới đây nếu
# phát hiện thuật ngữ Hán-Việt/TCM bị tokenize vỡ vụn (embedding_learning_rate
# trong training_args sẽ tự kích hoạt).
lora_kwargs = dict(
    # r=64: data domain lớn hơn ~30–60× so với run cũ → cần nhiều capacity hơn
    # r=32 để hấp thụ kiến thức TCM.
    r=64,
    # alpha=r với rsLoRA → scale nhất quán với SFT, tránh shock khi continue-train.
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_rslora=True,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

lora_kwargs.update(config.get("lora", {}))
if "gradient_checkpointing" in unsloth_config:
    lora_kwargs["use_gradient_checkpointing"] = unsloth_config["gradient_checkpointing"]
model = FastLanguageModel.get_peft_model(model, **lora_kwargs)

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
#
# Dataset đã pre-chunk: nhiều chunk (chunk_index 0..N) thuộc cùng một `source`
# là các đoạn LIÊN TỤC của cùng tài liệu. KHÔNG append eos cho mọi chunk — làm
# vậy sẽ chèn eos giữa các đoạn liền nhau, dạy model "tài liệu chỉ dài 2048 tok"
# và cắt đứt ngữ cảnh xuyên chunk. Để packing=True nối các chunk lại; eos chỉ
# nên xuất hiện ở ranh giới tài liệu thật (ngoài phạm vi xử lý ở đây).
def format_text(example):
    raw_text = example.get(TEXT_FIELD, "")
    if not raw_text:
        return {TEXT_FIELD: ""}
    return {TEXT_FIELD: clean_text(raw_text)}

# =========================
# LOAD DATASET (HF — đã có sẵn split train/validation)
# =========================
train_dataset = load_dataset(DATASET_ID, split=data_config.get("train_split", "train"), token=HF_TOKEN)
print("Train cols:", train_dataset.column_names, "rows:", len(train_dataset))
train_dataset = train_dataset.map(format_text)
train_dataset = train_dataset.filter(lambda x: len(x[TEXT_FIELD]) > 0)

# validation split held-out → monitor CPT có học được domain hay không.
# eval_loss giảm = học được; tăng nhanh = overfit hoặc format collapse.
eval_dataset = load_dataset(DATASET_ID, split=data_config.get("validation_split", "validation"), token=HF_TOKEN)
print("Eval cols:", eval_dataset.column_names, "rows:", len(eval_dataset))
eval_dataset = eval_dataset.map(format_text)
eval_dataset = eval_dataset.filter(lambda x: len(x[TEXT_FIELD]) > 0)

# =========================
# TRAIN
# =========================
training_kwargs = dict(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    # eff batch = 2 × 8 = 16: ổn định gradient hơn trên data lớn (51.4k chunk).
    gradient_accumulation_steps=8,
    # warmup_ratio thay cho warmup_steps — tự scale theo tổng steps.
    warmup_ratio=0.03,
    # 1 epoch: data giờ lớn ~30–60× run cũ → một lượt là đủ; cày nhiều epoch
    # trên Instruct model dễ format collapse. Nâng lên 2 nếu eval_loss vẫn giảm
    # đều ở cuối epoch 1.
    num_train_epochs=3,
    # CPT trên model Instruct → LR thấp hơn SFT-from-base để giữ alignment.
    learning_rate=3e-5,
    # Inert khi target_modules không có embed_tokens/lm_head — giữ sẵn để
    # khi nào bật Full CPT mode thì đã có giá trị hợp lý (10× < main LR).
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

training_kwargs.update(config.get("training", {}))
# Configured relative outputs are rooted at the repo, regardless of launch cwd.
if cli.config:
    output = Path(training_kwargs["output_dir"])
    training_kwargs["output_dir"] = str(output if output.is_absolute() else REPO_ROOT / output)
OUTPUT_DIR = training_kwargs["output_dir"]
print("[CPT] model:", MODEL_NAME, "dataset:", DATASET_ID, flush=True)
print("[CPT] effective training arguments:", json.dumps(training_kwargs, sort_keys=True), flush=True)
training_args = UnslothTrainingArguments(**training_kwargs)

trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field=TEXT_FIELD,
    max_seq_length=MAX_SEQ_LENGTH,
    # Pack short docs thành sequence đủ max_seq_length → bỏ hết padding,
    # compute không bị phí cho token <pad>.
    packing=data_config.get("packing", True),
    args=training_args,
    **({"dataset_kwargs": {"add_special_tokens": data_config["add_special_tokens"]}}
       if "add_special_tokens" in data_config else {}),
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

HF_CPT_REPO = os.environ.get("HF_CPT_REPO", "Maverfrick/Venera-CPT-TMModel")
JOB_TAG = os.environ.get("JOB_TAG") or f"local-{int(time.time())}"
# Push thẳng adapter vào branch `main` (lambda_push_to_hf.py dùng "main"
# verbatim). JOB_TAG vẫn được nhúng vào commit message để truy vết run nào.
HF_REVISION = "main"

print(f"\n[CPT.py] pushing adapter to {HF_CPT_REPO}@{HF_REVISION}")
rc = subprocess.run([
    sys.executable, PUSH_SCRIPT,
    "--task", "cpt",
    "--repo", HF_CPT_REPO,
    "--src", OUTPUT_DIR,
    "--revision", HF_REVISION,
    "--commit-message", f"Upload CPT adapter ({JOB_TAG})",
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