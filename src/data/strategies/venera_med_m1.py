import os
import sys
import pickle
import yaml
from typing import Tuple

from datasets import Dataset, load_dataset
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

def convert_to_conversations(example):
    """
    Convert TradMedD format -> conversations format
    Giả định dataset có dạng:
    {
        "question": "...",
        "answer": "..."
    }
    """

    if "conversations" in example:
        return example

    if "question" in example and "answer" in example:
        return {
            "conversations": [
                {"role": "user", "content": example["question"]},
                {"role": "assistant", "content": example["answer"]},
            ]
        }

    raise ValueError(
        f"Không nhận diện được format dataset. Keys hiện có: {list(example.keys())}"
    )


def preformat(dataset: Dataset) -> Dataset:
    """
    Chuẩn hoá dataset về format conversations
    """

    if "conversations" not in dataset.column_names:
        dataset = dataset.map(convert_to_conversations)

    def validate(example):
        convs = example["conversations"]

        if not isinstance(convs, list):
            raise ValueError("`conversations` phải là list")

        for msg in convs:
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

        train_samples = config.get("train_samples", -1)
        split_ratio = config.get("train_val_split", 0.2)

        # ---- Cache ----
        cache_key = f"TradMedD_{train_samples}_{split_ratio}"
        cache_file = os.path.join(
            self.cache_dir, f"tradmedd_cache_{hash(cache_key)}.pkl"
        )

        if os.path.exists(cache_file):
            print("✅ Loading dataset from cache")
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
                return cached["train"], cached["val"]

        # ---- Load dataset from HF ----
        print("⬇️ Loading dataset from Hugging Face Hub...")

        dataset = load_dataset("Maverfrick/TradMedD")

        # Nếu có nhiều split
        if "train" in dataset:
            dataset = dataset["train"]
        else:
            # lấy split đầu tiên nếu không có train
            dataset = list(dataset.values())[0]

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

        # ---- Cache ----
        with open(cache_file, "wb") as f:
            pickle.dump({"train": train_set, "val": val_set}, f)

        print(f"✅ Train: {len(train_set)} | Val: {len(val_set)}")
        return train_set, val_set


# =========================
# Debug
# =========================

if __name__ == "__main__":
    config = {
        "train_samples": -1,
        "train_val_split": 0.2,
    }

    strategy = VeneraMedDatasetStrategy()
    train_ds, val_ds = strategy.load(config)

    print("Train size:", len(train_ds))
    print("Val size:", len(val_ds))
    print("Train sample:", train_ds[0])
    print("Val sample:", val_ds[0])