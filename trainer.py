import os

os.environ["TRITON_DISABLE_COMPILE"] = "1"
os.environ["TRITON_USE_PYTORCH_ALLOCATOR"] = "1"
os.environ["TRITON_DISABLE"] = "1"
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
import pandas as pd
from datasets import Dataset, concatenate_datasets, get_dataset_config_names
from trl import SFTTrainer, SFTConfig
import argparse
import yaml
from typing import List
from datetime import datetime
import torch
import wandb
from logging_wandb import CustomWandbCallback
from wandb.sdk.wandb_run import Run
import gc
from huggingface_hub import login
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)
repo_id = "venera-ai/VeneraMedicalM1"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Medical LLM Training with Enhanced Parameters"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="/root/VeneraMedicalM1/experiments/configs/experiment_light.yaml",
        help="config path",
    )
    return parser.parse_args()


def load_combined_datasets(dataset_configs: List):
    combine_datasets = []
    for dataset_config in dataset_configs:
        try:
            dataset_name = dataset_config["name"]
            configs = get_dataset_config_names(dataset_name)
            config = "default" if "default" in configs else configs[0]

            dataset = load_dataset(dataset_name, name=config, split="train")
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            try:
                dataset = load_dataset(dataset_name)["train"]
            except:
                raise ValueError(
                    f"Could not load dataset {dataset_name} with any method"
                )
        combine_datasets.append(dataset)
    combine_datasets = concatenate_datasets(combine_datasets)
    return combine_datasets


def main():

    cmd_args = parse_arguments()
    now = datetime.now()
    date_time_string = now.strftime("%d-%m-%Y--%H-%M-%S")
    with open(cmd_args.config_path, "r") as file:
        config = yaml.safe_load(file)
    for experiment in config["experiments"]:
        print("EXPERIMENT ", experiment)
        output_dir=os.makedirs( os.path.join(experiment["output_dir"],date_time_string), exist_ok=True)
        print("output dir",output_dir)
        branch_name = f"{experiment['model']}-{date_time_string}"
        print("branch name",branch_name)
        
        run = wandb.init(
            project=f"VeneraMedicalM1 {date_time_string}",
            name=f"{experiment['name']}",
            reinit=True,
            config=experiment,
        )
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=experiment["model"],
            max_seq_length=experiment[
                "max_seq_length"
            ],  # Context length - can be longer, but uses more memory
            load_in_4bit=True,  # 4bit uses much less memory
            load_in_8bit=False,  # A bit more accurate, uses 2x memory
            full_finetuning=False,  # We have full finetuning now!
            #  attn_implementation="flash_attention_2"  #
            attn_implementation="sdpa",
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=32,  # Choose any number > 0! Suggested 8, 16, 32, 64, 128
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=32,  # Best to choose alpha = rank or rank*2
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",  # Supports any, but = "none" is optimized
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=3407,
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
        )

        combined_dataset = load_combined_datasets(experiment["dataset"])
        n_samples = experiment["n_samples"]
        sampled_data = [
            combined_dataset[i] for i in range(min(n_samples, len(combined_dataset)))
        ]
        combined_dataset = Dataset.from_list(sampled_data).shuffle(seed=3407)

        def formatting_func(examples):
            if "conversations" in examples:
                conversations = examples["conversations"]

                if isinstance(conversations, list) and len(conversations) > 0:
                    if isinstance(conversations[0], list):
                        texts = []
                        for conversation in conversations:
                            messages = [
                                {
                                    "role": (
                                        "user"
                                        if msg["from"] == "human"
                                        else "assistant"
                                    ),
                                    "content": msg["value"],
                                }
                                for msg in conversation
                            ]
                            text = tokenizer.apply_chat_template(
                                messages, tokenize=False, add_generation_prompt=False
                            )
                            texts.append(text)
                        return texts
                    else:
                        messages = [
                            {
                                "role": (
                                    "user" if msg["from"] == "human" else "assistant"
                                ),
                                "content": msg["value"],
                            }
                            for msg in conversations
                        ]
                        text = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=False
                        )
                        return [text]

            return [""]

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=combined_dataset,
            output_dir=output_dir,
            eval_dataset=None,
            formatting_func=formatting_func,
            args=SFTConfig(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,  # Use GA to mimic batch size!
                warmup_steps=5,
                # num_train_epochs = 1, # Set this for 1 full training run.
                # max_steps = 30,
                max_steps=4,
                learning_rate=2e-4,  # Reduce to 2e-5 for long training runs
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                report_to="wandb",  # Use this for WandB etc
            ),
        )
        trainer.add_callback(CustomWandbCallback())
        trainer_stats = trainer.train()
                
        api = HfApi() 
        trainer.save_model("./final_model")
        api.create_branch(repo_id=repo_id, branch=branch_name, exist_ok=True)
        api.upload_folder(
            folder_path="./final_model",
            repo_id=repo_id,
            revision=branch_name,  
            commit_message=f"Training {now.isoformat()} - {experiment['name']}"
        )
        if run is not None:
            run.finish()
            torch.cuda.empty_cache()  
            gc.collect()
if __name__ == "__main__":
    main()
