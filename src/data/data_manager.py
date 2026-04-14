import sys
from dotenv import load_dotenv
load_dotenv()
import os
ROOT=os.getenv("ROOT")
sys.path.append(ROOT)
from src.data.dataset_factory import DatasetFactory
from datasets import concatenate_datasets
import yaml

class DatasetManager:
    def __init__(
        self,
    ):
        self.library = {}

  
    def store_datasets(self, config):
        for dataset_config in config:
            path=dataset_config["path"]
            with open(path, "r") as f:
                base_config = yaml.safe_load(f)
            tag = base_config["name"]
            if tag in self.library:
                raise ValueError(
                    f"Dataset with tag {tag} already exists in the library"
                )
            merged_config = {**base_config, **dataset_config}
            train_set, val_set = DatasetFactory.create_datasets(tag, merged_config)
            print("---------------processed dataset-----------------",tag)
            self.library[tag] = {"train": train_set, "val": val_set, "tag": tag}

    def validate_datasets(self, n_samples):
        for tag in self.library:
            train_dataset = self.library[tag]["train"]
            val_dataset = self.library[tag]["val"]

            for ds_name, ds in [("train", train_dataset), ("val", val_dataset)]:
                for i in range(min(len(ds), n_samples)):
                    sample = ds[i]
                    convs = sample.get("conversations") or sample.get("messages")
                    if not convs or not isinstance(convs, list):
                        print(f"Error: {tag} {ds_name}[{i}] missing conversations field")
                        return False
                    for msg in convs:
                        if "role" not in msg or "content" not in msg:
                            print(f"Error: {tag} {ds_name}[{i}] message missing role/content: {msg}")
                            return False
        return True

    def load_combined_datasets(self):
        combine_train_datasets = []
        combine_val_datasets = []
        train_idx_ladder = []
        val_idx_ladder = []
        cur_train_idx = 0
        cur_val_idx = 0
        for tag in self.library:
            train_dataset = self.library[tag]["train"]
            val_dataset = self.library[tag]["val"]
            if train_dataset is not None:
                print(f"Train size ({tag}): {len(train_dataset)}")
                combine_train_datasets.append(train_dataset)
                cur_train_idx += len(train_dataset)
                train_idx_ladder.append((cur_train_idx, tag))

            if val_dataset is not None:
                print(f"Validation size ({tag}): {len(val_dataset)}")
                combine_val_datasets.append(val_dataset)
                cur_val_idx += len(val_dataset)
                val_idx_ladder.append((cur_val_idx, tag))
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


if __name__ == "__main__":
    from unsloth import FastLanguageModel
    from format import format_conversations
    from functools import partial

    config = [
        # {"tag": "fine_tome", "train_samples": 100, "train_val_split": 0.7},
        {
            "path": "configs/dataset/venera_med_m1.yaml",
            "train_samples": 10,
            "train_val_split": 0.2,
        },
    ]
    dataset_manager = DatasetManager()
    dataset_manager.store_datasets(config)
    n_samples = 4
    model_names = ["unsloth/Qwen3-0.6B", "unsloth/Qwen3-1.7B"]
    for model_name in model_names:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            load_in_4bit=True,
            attn_implementation="sdpa",
        )
        output = dataset_manager.validate_datasets(
            n_samples, partial(format_conversations, tokenizer=tokenizer)
        )
        print(f"Validation result for {model_name}: {output}")
    combine_train_datasets,combine_val_datasets,train_idx_ladder, val_idx_ladder=dataset_manager.load_combined_datasets()
