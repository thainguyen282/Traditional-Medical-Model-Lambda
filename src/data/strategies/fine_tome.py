from typing import Tuple
from datasets import Dataset
import sys 

sys.path.append("/root/VeneraMedicalM1")
from src.data.strategies.base import BaseDatasetStrategy
from datasets import load_dataset

def preformat(data):
    
    outputs={
        "conversations":[]
    }
    out_conversations=outputs["conversations"]
    for content_data in data["conversations"]:
        sample={}
        if content_data["from"] == "human":
            sample["role"] = "user"
        elif content_data["from"] == "gpt":
            sample["role"] = "assistant"
        elif content_data["from"] == "system":
            sample["role"] = "system"
        sample["content"] = content_data["value"]
        out_conversations.append(sample)
    return outputs
      


class FineTomeDatasetStrategy(BaseDatasetStrategy):
    def load(self, config: dict) -> Tuple[Dataset, Dataset]:
        data_source="mlabonne/FineTome-100k"
        dataset_dict = load_dataset(data_source)
        train_dataset = dataset_dict[list(dataset_dict.keys())[0]]  # Get first split
        
        if config["train_samples"]==-1:
            n_total_samples= len(train_dataset)
        else:
            n_total_samples = min(config["train_samples"], len(train_dataset))
        train_dataset = train_dataset.shuffle(seed=42).select(range(n_total_samples))
        train_dataset = train_dataset.map(
            lambda x: preformat(x),
            batched=False,
            desc="Preformatting keys",
            remove_columns=train_dataset.column_names,
        )
        split_ratio = 1- config["train_val_split"]
        dataset = train_dataset.train_test_split(
            test_size=split_ratio, 
            seed=42
        )
        train_set, val_set = dataset["train"], dataset["test"]
    
        return train_set, val_set

if __name__ == "__main__":
    config={
        "train_samples":10,
        "train_val_split":0.2
    }
    strategy = FineTomeDatasetStrategy()
    train_dataset, val_dataset = strategy.load(config)
    for i in range(len(train_dataset)):
        print(train_dataset[i])
        if i==2:
            break
    pass