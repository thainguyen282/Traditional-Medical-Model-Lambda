from typing import Tuple
from datasets import Dataset
from .base_strategy import BaseDatasetStrategy
from datasets import load_dataset

def preformat(data):
    """
    Recursively updates dictionary keys:
    - Changes "from" → "role"
    - Changes "value" → "content"
    Works with nested dictionaries and lists.
    """
    if "conversations" in data:
        data["conversations"] = preformat(data["conversations"])
        return data
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            new_key = "role" if key == "from" else ("content" if key == "value" else key)
            new_dict[new_key] = preformat(value)
        return new_dict
    elif isinstance(data, list):
        return [preformat(item) for item in data]
    else:
        return data


class FineTomeDatasetStrategy(BaseDatasetStrategy):
    def load(self, config: dict) -> Tuple[Dataset, Dataset]:
        
        dataset_dict = load_dataset(config["path"])
        train_dataset = dataset_dict[list(dataset_dict.keys())[0]]  # Get first split
        
        train_dataset = train_dataset.add_column("dataset_id", [dataset_id] * len(train_dataset))
        if config["train_samples"]==-1:
            n_total_samples= len(train_dataset)
        else:
            n_total_samples = min(config["train_samples"], len(train_dataset))
        train_dataset = train_dataset.shuffle(seed=42).select(range(n_total_samples))
        train_dataset = train_dataset.map(
            lambda x: preformat(x),
            batched=False,
            desc="Preformatting keys"
        )
        split_ratio = 1- config["weight"]
        dataset = train_dataset.train_test_split(
            test_size=split_ratio, 
            seed=42
        )
        train_set, val_set = dataset["train"], dataset["test"]
    
        return train_set, val_set