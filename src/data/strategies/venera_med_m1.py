import json
import os
import sys
import pickle
import yaml
from typing import Tuple

from datasets import Dataset
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = os.getenv("ROOT")
if ROOT_DIR is None:
    raise EnvironmentError("ROOT environment variable is not set")

sys.path.append(ROOT_DIR)

from src.data.strategies.base import BaseDatasetStrategy


# =========================
# Utils
# =========================

def preformat(dataset: Dataset) -> Dataset:
    """
    Chuẩn hoá dataset về format `conversations`
    Phù hợp cho SFTTrainer / Unsloth
    """

    # Nếu đã đúng format thì thôi
    if "conversations" in dataset.column_names:
        return dataset

    if "conversation" not in dataset.column_names:
        raise ValueError(
            f"Dataset phải có cột 'conversation' hoặc 'conversations'. "
            f"Hiện có: {dataset.column_names}"
        )

    dataset = dataset.rename_column("conversation", "conversations")

    def validate(example):
        convs = example["conversations"]

        if not isinstance(convs, list):
            raise ValueError(f"`conversations` phải là list, nhận được {type(convs)}")

        for msg in convs:
            if not isinstance(msg, dict):
                raise ValueError(f"Message phải là dict, nhận được {type(msg)}")
            if "role" not in msg or "content" not in msg:
                raise ValueError(f"Message thiếu role/content: {msg}")

        return example

    return dataset.map(validate, desc="Validating conversations format")


# =========================
# Dataset Strategy
# =========================

class VeneraMedDatasetStrategy(BaseDatasetStrategy):
    def __init__(self):
        super().__init__()
        self.cache_dir = os.path.join(ROOT_DIR, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def load(self, config: dict) -> Tuple[Dataset, Dataset]:
        """
        config lấy trực tiếp từ YAML:
        datasets:
          - path
            train_samples
            train_val_split
        """

        # ---- Load dataset YAML ----
        yaml_path = os.path.join(ROOT_DIR, config["path"])
        with open(yaml_path, "r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f)

        train_path = os.path.join(ROOT_DIR, base_config["train_path"])
        train_samples = config.get("train_samples", -1)
        split_ratio = config.get("train_val_split", 0.2)

        # ---- Cache ----
        cache_key = f"{train_path}_{train_samples}_{split_ratio}"
        cache_file = os.path.join(
            self.cache_dir, f"venera_med_cache_{hash(cache_key)}.pkl"
        )

        if os.path.exists(cache_file):
            print("✅ Loading dataset from cache")
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
                return cached["train"], cached["val"]

        # ---- Load dataset ----
        if os.path.exists(train_path):
            print(f"📂 Loading local dataset: {train_path}")
            with open(train_path, "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f]
        else:
            print("⚠️ Local dataset not found, downloading from HF Hub...")
            repo_id = base_config.get(
                "repo_id", "venera-ai/VeneraMedicalM1-dataset"
            )
            filename = base_config.get(
                "filename", "3.0/training_data_v3.6c.jsonl"
            )

            hf_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
            )

            print(f"⬇️ Downloaded dataset to: {hf_path}")
            with open(hf_path, "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f]

        dataset = Dataset.from_list(records)

        print("📊 Dataset columns:", dataset.column_names)
        print("🔎 First sample:", dataset[0])

        # ---- Sampling ----
        if train_samples != -1:
            train_samples = min(train_samples, len(dataset))
            dataset = dataset.shuffle(seed=42).select(range(train_samples))

        # ---- Train / Val split ----
        split = dataset.train_test_split(
            test_size=split_ratio,
            seed=42,
        )

        train_set = preformat(split["train"])
        val_set = preformat(split["test"])

        # ---- Cache result ----
        with open(cache_file, "wb") as f:
            pickle.dump({"train": train_set, "val": val_set}, f)

        print(f"✅ Train: {len(train_set)} | Val: {len(val_set)}")
        return train_set, val_set


# =========================
# Debug
# =========================

if __name__ == "__main__":
    config = {
        "path": "configs/dataset/venera_med_m1.yaml",
        "train_samples": -1,
        "train_val_split": 0.2,
    }

    strategy = VeneraMedDatasetStrategy()
    train_ds, val_ds = strategy.load(config)

    print("Train size:", len(train_ds))
    print("Val size:", len(val_ds))
    print("Train sample:", train_ds[0])
    print("Val sample:", val_ds[0])