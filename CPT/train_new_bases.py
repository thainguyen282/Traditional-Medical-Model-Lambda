#!/usr/bin/env python3
"""Text-only continual pretraining with separate LoRA adapters for new bases."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

MODELS = {
    "qwen3-4b-thinking-2507": "Qwen/Qwen3-4B-Thinking-2507",
    "gemma4-e2b": "google/gemma-4-E2B-it",
}
DATASET = "Maverfrick/TradMed-CPT-53k-2048"
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Positive value for a short training smoke test")
    parser.add_argument("--push-repo", default=None,
                        help="Upload adapter to this existing Hugging Face model repo")
    parser.add_argument("--push-revision", default=None,
                        help="New, unique branch name; required with --push-repo")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = Path("CPT/outputs") / args.model
    if args.epochs <= 0 or args.batch_size < 1 or args.gradient_accumulation < 1:
        parser.error("epochs, batch size, and gradient accumulation must be positive")
    if args.max_seq_length < 2 or args.max_steps == 0:
        parser.error("invalid max sequence length or max steps")
    if bool(args.push_repo) != bool(args.push_revision):
        parser.error("--push-repo and --push-revision must be given together")
    if args.push_revision == "main":
        parser.error("use a unique branch, not main, for a new CPT adapter")
    if args.output_dir.exists() and not args.dry_run:
        parser.error(f"output directory already exists: {args.output_dir}")
    return args


def main():
    args = parse_args()
    plan = {"model": MODELS[args.model], "dataset": args.dataset,
            "output_dir": str(args.output_dir), "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "max_seq_length": args.max_seq_length, "max_steps": args.max_steps,
            "push_repo": args.push_repo, "push_revision": args.push_revision,
            "method": "text-only causal LM, NF4 QLoRA, no chat template"}
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return 0

    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, Trainer, TrainingArguments)

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CPT requires a CUDA GPU with BF16 support")
    token = os.getenv("HF_TOKEN")
    api = HfApi(token=token)
    model_revision = api.model_info(MODELS[args.model]).sha
    dataset_revision = api.repo_info(args.dataset, repo_type="dataset").sha
    plan.update(model_revision=model_revision, dataset_revision=dataset_revision,
                seed=3407, lora_r=64, lora_alpha=64)

    tokenizer = AutoTokenizer.from_pretrained(MODELS[args.model],
                                               revision=model_revision, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    raw_train = load_dataset(args.dataset, split="train", revision=dataset_revision, token=token)
    raw_eval = load_dataset(args.dataset, split="validation", revision=dataset_revision, token=token)
    if "text" not in raw_train.column_names or "text" not in raw_eval.column_names:
        raise ValueError("CPT dataset requires a text column in train and validation")

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True,
                         max_length=args.max_seq_length, add_special_tokens=False)

    train = raw_train.filter(lambda row: bool(row["text"] and row["text"].strip()))
    valid = raw_eval.filter(lambda row: bool(row["text"] and row["text"].strip()))
    train = train.map(tokenize, batched=True, remove_columns=train.column_names)
    valid = valid.map(tokenize, batched=True, remove_columns=valid.column_names)
    train = train.filter(lambda row: len(row["input_ids"]) >= 2)
    valid = valid.filter(lambda row: len(row["input_ids"]) >= 2)
    if len(train) == 0 or len(valid) == 0:
        raise ValueError("CPT dataset has an empty tokenized train or validation split")
    plan.update(train_rows=len(train), validation_rows=len(valid))

    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_use_double_quant=True,
                                      bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        MODELS[args.model], revision=model_revision, token=token,
        dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", quantization_config=quantization)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model,
                                            use_gradient_checkpointing=True)
    targets = (list(PROJECTIONS) if args.model.startswith("qwen") else
               r".*language_model.*\.(?:" + "|".join(PROJECTIONS) + r")")
    model = get_peft_model(model, LoraConfig(
        r=64, lora_alpha=64, lora_dropout=0, bias="none", use_rslora=True,
        target_modules=targets, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    def collate(examples):
        padded = tokenizer.pad(examples, padding=True, return_tensors="pt")
        labels = padded["input_ids"].clone()
        labels[padded["attention_mask"] == 0] = -100
        padded["labels"] = labels
        return padded

    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "run_config.json").write_text(json.dumps(plan, indent=2) + "\n")
    settings = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        learning_rate=3e-5, warmup_ratio=0.03,
        lr_scheduler_type="cosine", optim="adamw_8bit", bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps", eval_steps=200,
        save_strategy="steps", save_steps=500, save_total_limit=2,
        logging_steps=10, report_to="none", seed=3407,
        remove_unused_columns=False)
    trainer = Trainer(model=model, args=settings, train_dataset=train,
                      eval_dataset=valid, data_collator=collate)
    result = trainer.train()
    adapter_dir = args.output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    metrics = {"train": result.metrics, "validation": trainer.evaluate()}
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Adapter saved: {adapter_dir}", flush=True)

    if args.push_repo:
        api.repo_info(args.push_repo, repo_type="model")
        api.create_branch(args.push_repo, branch=args.push_revision, exist_ok=False)
        api.upload_folder(folder_path=str(adapter_dir), repo_id=args.push_repo,
                          revision=args.push_revision,
                          commit_message=f"CPT {args.model} ({model_revision[:12]})")
        api.upload_file(path_or_fileobj=str(args.output_dir / "run_config.json"),
                        path_in_repo="run_config.json", repo_id=args.push_repo,
                        revision=args.push_revision)
        api.upload_file(path_or_fileobj=str(args.output_dir / "metrics.json"),
                        path_in_repo="metrics.json", repo_id=args.push_repo,
                        revision=args.push_revision)
        print(f"Uploaded: https://huggingface.co/{args.push_repo}/tree/{args.push_revision}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
