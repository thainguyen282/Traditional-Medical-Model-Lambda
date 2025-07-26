import json
from datasets import Dataset
from typing import Tuple
import sys
import os
import pickle

sys.path.append("/root/VeneraMedicalM1")
from src.data.strategies.base import BaseDatasetStrategy

def preformat(dataset):
    df = dataset.to_pandas()
    print("len df", len(df))

    def strip_fields(convs):
        return [
            {
                "role": str(c.get("role", "")),
                "content": str(c.get("content", ""))
            }
            for c in convs
        ]

    df["conversations"]=df["conversation"].apply(strip_fields)
    return Dataset.from_pandas(df[["conversations"]])

class MedSampleDatasetStrategy(BaseDatasetStrategy):
 
    def load(self, config: dict) -> Tuple[Dataset, Dataset]:
        train_path = config["train_path"]
        train_samples = config["train_samples"]
        split_ratio = config["train_val_split"]
        
        # Generate a unique cache key based on config
        cache_key = f"{train_path}_{train_samples}_{split_ratio}"
        cache_file = f"valset_cache_{hash(cache_key)}.pkl"
        
        # Try loading cached valset
        if os.path.exists(cache_file):
            print("Loading cached valset...")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                return cached_data['train'], cached_data['val']
        
        # Load and process fresh data
        with open(train_path, 'r') as f:
            train_dataset = Dataset.from_list([json.loads(line) for line in f])
        
        if train_samples == -1:
            train_samples = len(train_dataset)
        else:
            train_samples = min(train_samples, len(train_dataset))
        
        # Create consistent split
        train_dataset = train_dataset.shuffle(seed=42).select(range(train_samples))
        dataset = train_dataset.train_test_split(
            test_size=split_ratio,
            seed=42  # Fixed seed ensures same valset each time
        )
        
        train_set, val_set = preformat(dataset["train"]), preformat(dataset["test"])
        
        with open(cache_file, 'wb') as f:
            pickle.dump({'train': train_set, 'val': val_set}, f)
        
        return train_set, val_set


if __name__ == "__main__":
    config = {
        "train_path": "VeneraMedicalM1-dataset/1.8/final_data_v1.8.jsonl",
        # "val_path": "/root/VeneraMedicalM1/training_data_v5.jsonl",
        "train_samples": -1,
        "train_val_split": 0.2
    }
    strategy = MedSampleDatasetStrategy()
    train_dataset, val_dataset = strategy.load(config)
    print("len trainset",len(train_dataset),"len valset",len(val_dataset))
    # for i in range(len(train_dataset)):
    #     print(train_dataset[i])
    #     if i==2:
    #         break
    # for i in range(len(val_dataset)):
    #     print("____________________",val_dataset[i])
    #     if i==2:
    #         break