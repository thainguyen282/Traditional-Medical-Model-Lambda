import json
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download
from typing import Tuple
import sys
import os
import pickle
import yaml
from dotenv import load_dotenv
load_dotenv()
ROOT_DIR = os.getenv("ROOT")
sys.path.append(ROOT_DIR)
from src.data.strategies.base import BaseDatasetStrategy

def preformat(dataset):
    df = dataset.to_pandas()

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

class VeneraMedDatasetStrategy(BaseDatasetStrategy):
    def __init__(self):
        self.cache_dir = os.path.join(ROOT_DIR, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        super().__init__()
    def load(self, config: dict) -> Tuple[Dataset, Dataset]:
        yaml_path=os.path.join(ROOT_DIR,config["path"])
        with open(yaml_path, 'r') as f:
            base_config = yaml.safe_load(f)
        train_path = os.path.join(ROOT_DIR, base_config["train_path"])
        train_samples = config["train_samples"]
        split_ratio = config["train_val_split"]
        
        cache_key = f"{train_path}_{train_samples}_{split_ratio}"
        cache_file = os.path.join(self.cache_dir,  f"valset_cache_{hash(cache_key)}.pkl")
        
        if os.path.exists(cache_file):
            print("Loading cached valset...")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                return cached_data['train'], cached_data['val']
        
        # Try loading from local path first, then from Hugging Face if not found
        if not os.path.exists(train_path):
            print(f"Local file not found at {train_path}, downloading from Hugging Face...")
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(train_path), exist_ok=True)
                # Download file directly from HF Hub
                hf_file_path = hf_hub_download(
                    repo_id="venera-ai/VeneraMedicalM1-dataset",
                    filename="3.0/training_data_v3.6c.jsonl",
                    repo_type="dataset"
                )
                # Load from downloaded file
                with open(hf_file_path, 'r') as f:
                    train_dataset = Dataset.from_list([json.loads(line) for line in f])
            except Exception as e:
                print(f"Failed to download from Hugging Face: {e}")
                print("Please ensure internet connection and HF_TOKEN is set if needed.")
                raise
        else:
            with open(train_path, 'r') as f:
                train_dataset = Dataset.from_list([json.loads(line) for line in f])
        
        if train_samples == -1:
            train_samples = len(train_dataset)
        else:
            train_samples = min(train_samples, len(train_dataset))
        
        train_dataset = train_dataset.shuffle(seed=42).select(range(train_samples))
        dataset = train_dataset.train_test_split(
            test_size=split_ratio,
            seed=42  
        )
        
        train_set, val_set = preformat(dataset["train"]), preformat(dataset["test"])
        
        with open(cache_file, 'wb') as f:
            pickle.dump({'train': train_set, 'val': val_set}, f)
        
        return train_set, val_set


if __name__ == "__main__":
    config = {
        "path": "configs/dataset/venera_med_m1.yaml",
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