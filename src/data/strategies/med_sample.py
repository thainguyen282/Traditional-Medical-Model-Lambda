import json
from datasets import Dataset
from typing import Tuple
import sys
import pandas as pd

sys.path.append("/root/VeneraMedicalM1")
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

    df["conversations"] = df["conversations"].apply(strip_fields)
    return Dataset.from_pandas(df[["conversations"]])

class MedSampleDatasetStrategy(BaseDatasetStrategy):
    def load(self, config: dict) -> Tuple[Dataset, Dataset]:
        train_path = config["train_path"]
        val_path = config["val_path"]

        with open(train_path, 'r') as f:
            train_dataset = [json.loads(line) for line in f]
        with open(val_path, 'r') as f:
            val_dataset = [json.loads(line) for line in f]

        train_dataset = Dataset.from_list(train_dataset)
        val_dataset = Dataset.from_list(val_dataset)

        train_samples = len(train_dataset) if config["train_samples"] == -1 else min(config["train_samples"], len(train_dataset))
        val_samples = len(val_dataset) if config["val_samples"] == -1 else min(config["val_samples"], len(val_dataset))

        train_dataset = train_dataset.shuffle(seed=42).select(range(train_samples))
        val_dataset = val_dataset.shuffle(seed=42).select(range(val_samples))

        train_dataset = preformat(train_dataset)
        val_dataset = preformat(val_dataset)


        return train_dataset, val_dataset


if __name__ == "__main__":
    config = {
        "train_path": "/root/VeneraMedicalM1/training_data_v3.jsonl",
        "val_path": "/root/VeneraMedicalM1/training_data_v5.jsonl",
        "train_samples": 20,
        "val_samples": 20
    }
    strategy = MedSampleDatasetStrategy()
    train_dataset, val_dataset = strategy.load(config)
