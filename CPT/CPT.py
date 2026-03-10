import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth import UnslothTrainer, UnslothTrainingArguments


MODEL_NAME = "unsloth/Qwen3-4B-Instruct-2507"
DATASET_PATH = "test_data.jsonl"
OUTPUT_DIR = "cpt_qwen3_medical"

MAX_SEQ_LENGTH = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=256,
    lora_alpha=256,
    lora_dropout=0,
    bias="none",
    use_rslora=True,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "embed_tokens",
        "lm_head",
    ],
)


dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train"
)


def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )

dataset = dataset.map(tokenize, batched=True)


training_args = UnslothTrainingArguments(
    
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,

    warmup_steps=2,
    max_steps=20,

    learning_rate=2e-4,
    embedding_learning_rate=2e-5,

    fp16=False,
    bf16=True,

    logging_steps=1,

    optim="adamw_8bit",
    lr_scheduler_type="cosine",

    save_steps=10,
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

print("Training complete.")