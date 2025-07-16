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
from data.finetome import load_fine_tome_ds
from base_trainer import MultiDatasetTrainer
from collator import MultiDatasetCollator
from transformers import TrainingArguments
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
    combine_train_datasets = []
    combine_val_datasets = []
    # dataset_id_maps = {} 
    cur_train_idx = 0
    train_idx_ladder=[]
    cur_val_idx = 0
    val_idx_ladder=[]
    for id, dataset_config in enumerate(dataset_configs):
        tag= dataset_config.get("tag")
        if tag == "fine_tome":
            print("Loading FineTome dataset")
            from data.finetome import load_fine_tome_ds

            train_dataset, val_dataset = load_fine_tome_ds(dataset_config, id)
            print(" fine tome train set",len(train_dataset))
            print(" fine tome val set",len(val_dataset))
        elif tag == "dummy":
            print("Loading Dummy dataset")
            from data.finetome import load_dummy_ds

            train_dataset, val_dataset = load_dummy_ds(
                train_samples=dataset_config.get("train_samples", 4),
                val_samples=dataset_config.get("val_samples", 3),
            )
            print(" Dummy train set",len(train_dataset))
            print(" Dummy val set",len(val_dataset))
            
        if train_dataset is not None:
            combine_train_datasets.append(train_dataset)
            cur_train_idx += len(train_dataset)
            train_idx_ladder.append((cur_train_idx,tag))
        if val_dataset is not None:
            cur_val_idx += len(val_dataset)
            val_idx_ladder.append((cur_val_idx,tag))
            combine_val_datasets.append(val_dataset)
            
    if len(combine_train_datasets) > 0:
        combine_train_datasets = concatenate_datasets(combine_train_datasets)
    if len(combine_val_datasets) > 0:
        combine_val_datasets = concatenate_datasets(combine_val_datasets)
    return combine_train_datasets, combine_val_datasets, train_idx_ladder, val_idx_ladder



def main():

    cmd_args = parse_arguments()
    now = datetime.now()
    date_time_string = now.strftime("%d-%m-%Y--%H-%M-%S")
    with open(cmd_args.config_path, "r") as file:
        config = yaml.safe_load(file)

    for experiment in config["experiments"]:
        experiment_tag = experiment["name"]
        run = wandb.init(
            project=f"VeneraMedicalM1 ",
            name=f"{experiment_tag} {date_time_string}",
            reinit=True,
            config=config,
        )
        run.config.update({"experiment": experiment})
        print("EXPERIMENT ", experiment)
        output_dir = os.path.join(experiment["output_dir"], date_time_string)
        os.makedirs(output_dir, exist_ok=True)
        print("Output dir", output_dir)
        branch_slug = f"{experiment['model'].replace('/', '-')}"
        branch_name = f"{branch_slug}-{date_time_string}"
        print("Branch name", branch_name)

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

        combined_train_dataset, combined_val_dataset, train_idx_ladder,val_idx_ladder = (
            load_combined_datasets(experiment["dataset"])
        )
        print("len of combined train dataset", len(combined_train_dataset))
        print("len of combined val dataset", len(combined_val_dataset))
        def formatting_func(examples):   
            output=[]
            if "conversations" in examples:
                conversations = examples["conversations"]
                if isinstance(conversations, list) and len(conversations) > 0:
                    if isinstance(conversations[0], list):
                        for i,conversation in enumerate(conversations):
                            messages = [
                                {
                                    "role": "user" if msg["from"] == "human" else "assistant",
                                    "content": msg["value"]
                                }
                                for msg in conversation
                            ]
                            text = tokenizer.apply_chat_template(
                                messages, 
                                tokenize=False, 
                                add_generation_prompt=False
                            )
                            output.append(text)
                    else:
                        messages = [
                            {
                                "role": "user" if msg["from"] == "human" else "assistant",
                                "content": msg["value"]
                            }
                            for msg in conversations
                        ]
                        text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False
                        )
                        output.append(text)
            
            return output
    
        training_args = TrainingArguments(
            output_dir="./output",  
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            eval_strategy="steps",  
            eval_steps=2,                # Evaluate every 2 training steps
            do_eval=True,             
            logging_steps=4,           
            num_train_epochs=1,
            learning_rate=2e-4,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            warmup_steps=5,
            seed=3407,
            report_to="wandb",         
            save_strategy="steps",   
            save_steps=10           
        )

        trainer = MultiDatasetTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,          # MUST pass TrainingArguments (not SFTConfig)
            train_dataset=combined_train_dataset,
            eval_dataset=combined_val_dataset,  # REQUIRED for evaluation
            dataloader_shuffle=False,
            formatting_func=formatting_func,
            train_idx_ladder=train_idx_ladder,
            val_idx_ladder=val_idx_ladder,
            data_collator=MultiDatasetCollator(tokenizer=tokenizer, mlm=False)
        )

        # trainer = MultiDatasetTrainer(
        #     model=model,
        #     tokenizer=tokenizer,
        #     train_dataset=combined_train_dataset,
        #     output_dir=output_dir,
        #     eval_dataset=combined_val_dataset,
        #     dataloader_shuffle=False,  
        #     formatting_func=formatting_func,
        #     train_idx_ladder=train_idx_ladder,
        #     val_idx_ladder=val_idx_ladder,
        #      evaluation_strategy="steps",  
        #         eval_steps=2,              
        #         do_eval=True,
        #     args=SFTConfig(
        #         per_device_train_batch_size=2,
        #         gradient_accumulation_steps=4,  # Use GA to mimic batch size!
        #         warmup_steps=5,
        #         num_train_epochs = 1, # Set this for 1 full training run.
        #         # max_steps = 60,
        #         # max_steps=4,
        #         learning_rate=2e-4,  # Reduce to 2e-5 for long training runs
        #         logging_steps=4,
        #         optim="adamw_8bit",
        #         weight_decay=0.01,
        #         lr_scheduler_type="linear",
        #         seed=3407,
               
        #         report_to="wandb"
        #     ),
        #     data_collator=MultiDatasetCollator(tokenizer=tokenizer, mlm=False),
        # )
        trainer.add_callback(CustomWandbCallback())
        trainer_stats = trainer.train()

        api = HfApi()
        trainer.save_model("./final_model")
        # api.create_branch(repo_id=repo_id, branch=branch_name, exist_ok=True)
        # api.upload_folder(
        #     folder_path="./final_model",
        #     repo_id=repo_id,
        #     revision=branch_name,
        #     commit_message=f"Training {now.isoformat()} - {experiment['name']}",
        # )
        if run is not None:
            run.finish()
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    main()
