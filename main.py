from unsloth import FastLanguageModel
import os
import yaml
import torch
import wandb
from datetime import datetime
from typing import Dict, Any
import argparse
from pathlib import Path
from functools import partial
from huggingface_hub import login
from dotenv import load_dotenv
from transformers import TrainingArguments
from base_trainer import MultiDatasetTrainer
from tools.format import formatting_func
from logging_wandb import CustomWandbCallback
from datasets import Dataset, concatenate_datasets
from uuid import uuid4

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set")
login(token=HF_TOKEN)

os.environ["TRITON_DISABLE_COMPILE"] = "1"
os.environ["TRITON_USE_PYTORCH_ALLOCATOR"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def merge_configs(defaults, model_config):
    """Merge defaults with model-specific config (model overrides defaults)"""
    merged = {**defaults}  # Shallow copy
    for key, value in model_config.items():
        if isinstance(value, dict) and key in merged:
            merged[key] = {**merged[key], **value}  # Recursive merge for nested dicts
        else:
            merged[key] = value
    return merged


def create_training_args(config: Dict[str, Any]) -> TrainingArguments:
    """Create training arguments from config"""
    train_config = config.get("training", {})

    return TrainingArguments(
        output_dir=config.get("output_dir", "./outputs"),
        per_device_train_batch_size=train_config.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=train_config.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 4),
        num_train_epochs=train_config.get("num_train_epochs", 3),
        learning_rate=float(train_config.get("learning_rate", 2e-4)),
        weight_decay=float(train_config.get("weight_decay", 0.01)),
        warmup_steps=train_config.get("warmup_steps", 100),
        lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),
        optim=train_config.get("optim", "adamw_8bit"),
        eval_strategy=train_config.get("eval_strategy", "steps"),
        eval_steps=train_config.get("eval_steps", 50),
        save_strategy=train_config.get("save_strategy", "steps"),
        save_steps=train_config.get("save_steps", 100),
        logging_steps=train_config.get("logging_steps", 10),
        seed=train_config.get("seed", 3407),
        report_to=train_config.get("report_to", "wandb"),
        dataloader_num_workers=4,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )


def create_lora_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create LoRA configuration from config"""
    lora_config = config.get("lora", {})

    return {
        "r": lora_config.get("r", 32),
        "lora_alpha": lora_config.get("lora_alpha", 32),
        "lora_dropout": lora_config.get("lora_dropout", 0.1),
        "target_modules": lora_config.get(
            "target_modules",
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
        "bias": lora_config.get("bias", "none"),
        "use_rslora": lora_config.get("use_rslora", False),
    }


def load_combined_datasets(dataset_configs):
    combine_train_datasets = []
    combine_val_datasets = []
    train_idx_ladder = []
    val_idx_ladder = []
    cur_train_idx = 0
    cur_val_idx = 0
    for id, dataset_config in enumerate(dataset_configs):
        tag = dataset_config.get("tag")
        if tag == "fine_tome":
            print("Loading FineTome dataset")
            from data.dataset.finetome import load_fine_tome_ds

            train_dataset, val_dataset = load_fine_tome_ds(dataset_config, id)
        elif tag == "dummy":
            print("Loading Dummy dataset")
            from data.dataset.dummy import load_dummy_ds

            train_dataset, val_dataset = load_dummy_ds(dataset_config)
        # for i in range(len(train_dataset)):
        #     print("Sample", i, ":", train_dataset[i])
        #     if i == 3:
        #         break
        if train_dataset is not None:
            print(f" Size train {tag}", len(train_dataset))
            combine_train_datasets.append(train_dataset)
            cur_train_idx += len(train_dataset)
            train_idx_ladder.append((cur_train_idx, tag))
        if val_dataset is not None:
            print(f" Size val {tag}", len(val_dataset))
            cur_val_idx += len(val_dataset)
            val_idx_ladder.append((cur_val_idx, tag))
            combine_val_datasets.append(val_dataset)

    if len(combine_train_datasets) > 0:
        combine_train_datasets = concatenate_datasets(combine_train_datasets)
    if len(combine_val_datasets) > 0:
        combine_val_datasets = concatenate_datasets(combine_val_datasets)
    return (
        combine_train_datasets,
        combine_val_datasets,
        train_idx_ladder,
        val_idx_ladder,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="experiments/configs/experiment_light.yaml"
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        default=True,
        help="Run hyperparameter sweep (default: True)",
    )

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    id = uuid4().hex[:8]  # Generate a unique ID for this run
    if args.sweep:
        print("Running hyperparameter sweeps for all models")
        sweep_config = config.get("sweep", {})

        for experiment in config["experiments"]:
            experiment_config = merge_configs(config.get("defaults", {}), experiment)
            model_name = experiment_config["name"]

            sweep_id = wandb.sweep(
                sweep_config, project=f"VeneraMedicalM1", entity="nguyenduc150202"
            )

            print(f"Starting sweep for model: {model_name}")
            wandb.agent(
                sweep_id,
                function=lambda: train_with_config(experiment_config, run_id=id),
                count=5,
            )
    else:
        print("Running single experiment with merged config")
        for experiment in config["experiments"]:
            train_with_config(
                merge_configs(config.get("defaults", {}), experiment), run_id=id
            )


def train_with_config(experiment_config: Dict[str, Any], run_id: None):
    """Train model with given configuration"""
    print("experiment config", experiment_config)
    run = wandb.init(
        project=experiment_config.get("monitoring", {}).get(
            "project_name", "VeneraMedicalM1"
        ),
        name=f'{experiment_config.get("name", "experiment")}-{run_id}',
        config=experiment_config,
        tags=[
            experiment_config.get("name"),
            experiment_config.get("model", "unknown_model"),
            "sweep_run",
        ],
        group=experiment_config.get("name"),
    )
    rows = []
    for section, value in experiment_config.items():
        if isinstance(value, dict):
            for key, val in value.items():
                rows.append({"section": section, "key": key, "value": str(val)})
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                for key, val in item.items():
                    rows.append(
                        {"section": f"{section}[{idx}]", "key": key, "value": str(val)}
                    )
        else:
            rows.append({"section": "root", "key": section, "value": str(value)})

    config_table = wandb.Table(
        columns=["section", "key", "value"], data=[list(row.values()) for row in rows]
    )
    wandb.log({"Run Config Table": config_table})

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=experiment_config["model"],
            max_seq_length=experiment_config["max_seq_length"],
            load_in_4bit=True,
            attn_implementation="sdpa",
        )

        lora_config = create_lora_config(experiment_config)
        model = FastLanguageModel.get_peft_model(model, **lora_config)

        train_dataset, val_dataset, train_idx_ladder, val_idx_ladder = (
            load_combined_datasets(experiment_config["dataset"])
        )

        training_args = create_training_args(experiment_config)

        trainer = MultiDatasetTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            train_idx_ladder=train_idx_ladder,
            val_idx_ladder=val_idx_ladder,
            formatting_func=partial(formatting_func, tokenizer=tokenizer),
        )

        trainer.add_callback(CustomWandbCallback())

        trainer.train()
        output_dir = Path(experiment_config["output_dir"]) / datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        trainer.save_model(output_dir)
        final_metrics = trainer.evaluate()
        wandb.log(final_metrics)

    except Exception as e:
        wandb.log({"error": str(e)})
        raise
    finally:
        wandb.finish()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
