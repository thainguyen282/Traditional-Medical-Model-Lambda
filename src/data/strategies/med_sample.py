import json
from datasets import Dataset
from typing import Tuple
from src.data.strategies.base import BaseDatasetStrategy
class MedSampleDatasetStrategy(BaseDatasetStrategy):
    def load(self, config: dict) -> Tuple[Dataset, Dataset]:
        train_path = config["train_path"]
        val_path = config["val_path"]
        with open(train_path, 'r') as f:
            train_dataset = [json.loads(line) for line in f]
        with open(val_path, 'r') as f:
            val_dataset = [json.loads(line) for line in f]
        if config["train_samples"]==-1:
           train_samples=len(train_dataset)
        else:
            train_samples=min(config["train_samples"],len(train_dataset))
        if config["val_samples"]==-1:
            val_samples=len(val_dataset)
        else:
            val_samples=min(config["val_samples"],len(val_dataset))     
        
        train_dataset = train_dataset.shuffle(seed=42).select(range(train_samples))
        val_dataset = val_dataset.shuffle(seed=42).select(range(val_samples))
       
        return train_dataset, val_dataset

if __name__ == "__main__":
    pass
    # import os 
    # import sys
    # sys.path.append("/root/VeneraMedicalM1")
    # from unsloth import FastLanguageModel

    # from training.format import formatting_func
    # model, tokenizer = FastLanguageModel.from_pretrained(
    #         model_name="unsloth/Qwen3-1.7B",
    #         max_seq_length=2048,
    #         load_in_4bit=True,
    #         attn_implementation="sdpa",
    #     )
    # config = {
    #     "path": "sample_training_data_v1.jsonl",
    #     "train_samples": 200,
    #     "weight":0.7
    # }
    # train_set,val_set=load_med_sample(config)
    # for item in val_set:
    #     new_item=formatting_func(item,tokenizer)
    #     if not len(new_item):
    #         raise("find non valid",item)
    #         # print("find non valid",item)
    # for item in train_set:
    #     print("item",item)
    #     new_item=formatting_func(item,tokenizer)
    #     if not len(new_item):
    #         raise("find non valid",item)
    
    
        