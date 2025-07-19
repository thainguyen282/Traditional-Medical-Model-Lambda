from unsloth import FastLanguageModel
import os
import yaml
import torch
import wandb
from datetime import datetime
from typing import Dict, Any, Optional
import argparse
from pathlib import Path
from functools import partial
from huggingface_hub import login
from dotenv import load_dotenv
from transformers import TrainingArguments
from base_trainer import MultiDatasetTrainer
from collator import MultiDatasetCollator
from tools.format import formatting_func
from callbacks.logging_wandb import CustomWandbCallback
from data.dataset.finetome import load_fine_tome_ds
from datasets import Dataset, concatenate_datasets
import json
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
    merged = {**defaults}  
    for key, value in model_config.items():
        if isinstance(value, dict) and key in merged:
            merged[key] = {**merged[key], **value}  
        else:
            merged[key] = value
    return merged


def apply_sweep_config(experiment_config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply wandb sweep parameters to experiment config"""
    if not wandb.config:
        return experiment_config
    
    config = experiment_config.copy()
    
    sweep_params = {
        'learning_rate': ('training', 'learning_rate'),
        'lora_r': ('lora', 'r'),
        'lora_alpha': ('lora', 'lora_alpha'),
        'gradient_accumulation_steps': ('training', 'gradient_accumulation_steps'),
        'per_device_train_batch_size': ('training', 'per_device_train_batch_size'),
        'weight_decay': ('training', 'weight_decay'),
        'warmup_steps': ('training', 'warmup_steps'),
        'max_seq_length': (None, 'max_seq_length'),  
    }
    
    for param_name, (section, key) in sweep_params.items():
        if hasattr(wandb.config, param_name):
            value = getattr(wandb.config, param_name)
            
            if section is None:  
                config[key] = value
            else:  
                if section not in config:
                    config[section] = {}
                config[section][key] = value
            
            print(f"Applied sweep parameter: {param_name} = {value}")
    
    return config


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
        save_steps=train_config.get("save_steps", 50),
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
    """Load and combine multiple datasets"""
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
        elif tag=="med_sample":
            print("Load med sample")
            from data.dataset.med_sample import load_med_sample
            train_dataset,val_dataset=load_med_sample(dataset_config)
        # # Debug: print sample data
        # if train_dataset:
        #     for i in range(min(4, len(train_dataset))):
        #         print(f"Sample {i}:", train_dataset[i])
        
        if train_dataset is not None:
            print(f"Size train {tag}: {len(train_dataset)}")
            combine_train_datasets.append(train_dataset)
            cur_train_idx += len(train_dataset)
            train_idx_ladder.append((cur_train_idx, tag))
        
        if val_dataset is not None:
            print(f"Size val {tag}: {len(val_dataset)}")
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


def train_with_config(experiment_config: Dict[str, Any],sweep_id: Optional[str] = None):
    """Train model with given configuration"""
    
    
    try:
        run = wandb.init(
        project=experiment_config.get("monitoring", {}).get("project_name", "VeneraMedicalM1"),
        name=experiment_config.get("name", "experiment"),
        config=experiment_config,
         group=experiment_config.get("model"), 
          job_type="sweep" if sweep_id else "baseline",
        tags=[
            datetime.now().strftime("%Y-%m-%d"),             
            *(["sweep"] if sweep_id else []),
            *([f"sweep_{sweep_id[:6]}"] if sweep_id else []),
            
            "lora",
            "hyperparameter_tuning"
        ],
    )

        config = apply_sweep_config(experiment_config)
        
        wandb.config["experiment"]=  config

        print(yaml.dump(config, default_flow_style=False))

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config["model"],
            max_seq_length=config["max_seq_length"],
            load_in_4bit=True,
            attn_implementation="sdpa",
        )

        lora_config = create_lora_config(config)
        model = FastLanguageModel.get_peft_model(model, **lora_config)

        train_dataset, val_dataset, train_idx_ladder, val_idx_ladder = (
            load_combined_datasets(config["dataset"])
        )

        training_args = create_training_args(config)
        output_ckp=training_args.output_dir
        if sweep_id:
            output_ckp=os.path.join(output_ckp,f"sweep_{sweep_id}")
        output_ckp= os.path.join(output_ckp,f"run_{wandb.run.id}")
        os.makedirs(output_ckp, exist_ok=True)
        print("output_ckp",output_ckp)
        meta_data_file=os.path.join(output_ckp,"meta.json")
        with open(meta_data_file, "w") as f:  
            json.dump(experiment_config, f, indent=4) 
        training_args.output_dir=output_ckp
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
                
        final_metrics = trainer.evaluate()
        wandb.log(final_metrics)
        
        print(f"Training completed successfully. Model saved to {output_ckp}")

    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        wandb.log({"error": str(e)})
        raise
    finally:
        wandb.finish()
        torch.cuda.empty_cache()


def run_sweep(config: Dict[str, Any],test=False,sweep_count=3):
    """Run hyperparameter sweep for all experiments"""
    sweep_config = config.get("sweep", {})
    
    if not sweep_config:
        raise ValueError("No sweep configuration found in config file")
    if test:
        print("Test mode, sweep count =1")
        config["experiments"] =[ config["experiments"][0] ]
        config["defaults"]["dataset"]=[{
            "tag": "fine_tome",
        "path": "mlabonne/FineTome-100k",
        "train_samples": 10,
        "weight": 0.7
        }]
        sweep_count=1
    print("Sweep count",sweep_count)
    for experiment in config["experiments"]:
        experiment_config = merge_configs(config.get("defaults", {}), experiment)
        model_name = experiment_config["name"].replace("/", "_")
        
        print(f"\n{'='*60}")
        print(f"Starting sweep for model: {model_name}")
        print(f"{'='*60}")
        
        sweep_id = wandb.sweep(
            sweep_config, 
            project=f"{experiment_config.get('monitoring', {}).get('project_name', 'VeneraMedicalM1')}-sweeps",
            entity=os.getenv("WANDB_ENTITY")  
        )
        
        print(f"Sweep ID: {sweep_id}")
        
        def sweep_train(sweep_id):
            return train_with_config(experiment_config, sweep_id=sweep_id)
        
        wandb.agent(
            sweep_id,
            function=partial(sweep_train,sweep_id=sweep_id),
            count=sweep_count,  
        )
        
        print(f"Completed sweep for {model_name}")


def run_single_experiments(config: Dict[str, Any]):
    """Run single experiments without sweep"""
    for experiment in config["experiments"]:
        experiment_config = merge_configs(config.get("defaults", {}), experiment)
        print(f"\n{'='*60}")
        print(f"Training model: {experiment_config['name']}")
        print(f"{'='*60}")
        train_with_config(experiment_config)


def main():
    parser = argparse.ArgumentParser(description="Train language models with optional hyperparameter sweeps")
    parser.add_argument(
        "--config", 
        type=str, 
        default="experiments/configs/experiment_light.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--sweep", 
        
        action="store_true", 
        help="Run hyperparameter sweep"
    )
    parser.add_argument(
        "--test", 
        
        action="store_true", 
        help="test pipeline with minimal config"
    )
    parser.add_argument(
        "--sweep-count", 
        type=int, 
        default=5,
        help="Number of sweep trials per model"
    )
    
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.sweep:
        print("Running hyperparameter sweeps for all models")
        run_sweep(config,test=args.test,sweep_count=args.sweep_count )
    else:
        print("Running single experiments for all models")
        run_single_experiments(config)


if __name__ == "__main__":
    main()