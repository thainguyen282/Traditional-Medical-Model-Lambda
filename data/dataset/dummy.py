from datasets import load_dataset, Dataset
from typing import Tuple, Dict, Any
import pandas as pd

def load_dummy_ds(dataset_config) -> Tuple[Dataset, Dataset]:
    """Load dummy dataset for testing"""
    import random
    
    def generate_sample():
        return {
            "conversations": [
                {"role": "human", "content": f"Question {random.randint(1, 1000)}?"},
                {"role": "user", "content": f"Answer {random.randint(1, 1000)}."}
            ]
        }
    # train_samples: int = 100, val_samples: int = 50
    train_samples = dataset_config.get("train_samples", 100)
    train_samples=int(train_samples*dataset_config.get("weight", 0.5))
    val_samples = int(train_samples * (1 - dataset_config.get("weight", 0.5)))
    
    train_data = [generate_sample() for _ in range(train_samples)]
    val_data = [generate_sample() for _ in range(val_samples)]
    
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    return train_dataset, val_dataset