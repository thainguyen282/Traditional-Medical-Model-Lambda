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
    Convert dataset format -> conversations format
    Supports:
    - TradMedInstructData format with "messages" field
    - New-TradMedInstruct-qa_20k format with singular "conversation" field
    - TradMedD format with "question" and "answer" fields
    - Direct "conversations" field
    """

    if "conversations" in example:
        return example

    # TradMedInstructData format: messages -> conversations
    if "messages" in example:
        return {
            "conversations": example["messages"]
        }

    # New-TradMedInstruct-qa_20k: cột `conversation` (số ít) đã đúng format
    # list[{role, content}] — chỉ cần map sang key `conversations`.
    if "conversation" in example:
        return {
            "conversations": example["conversation"]
        }

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
        # Optional: mix in a fraction of open-chat data (FineTome-100k) so the
        # model doesn't collapse onto the MCQ "letter + phân tích" pattern and
        # keep emitting Qwen3 tool_call scaffolding on open-ended prompts.
        # Set in exp yaml: chat_mix_ratio: 0.1  # 10% of final dataset
        chat_mix_ratio = float(config.get("chat_mix_ratio", 0.0))

        # ---- Data source(s) ----
        # `data` accepts either a single string (HF repo or URL) or a LIST of
        # strings — when list, all sources are concat'd full (no reweighting).
        # Backward-compatible: existing string configs unchanged.
        # Phải biết source TRƯỚC khi tính cache key để không lẫn cache.
        data_source = config.get("data", "Maverfrick/TradMedInstruct-multiplechoice")
        # ######################## VALIDATION SOURCE ########################
        validation_source = config.get("validation_source")
        if isinstance(data_source, str):
            sources = [data_source]
        else:
            sources = list(data_source)

        def _normalize(s: str) -> str:
            return s.replace("https://huggingface.co/datasets/", "")

        sources = [_normalize(s) for s in sources]
        if validation_source:
            validation_source = _normalize(validation_source)
            if validation_source not in sources:
                raise ValueError(
                    f"validation_source '{validation_source}' is not present in data: {sources}"
                )
        # ###################### END VALIDATION SOURCE ######################

        # ---- Cache (key includes sources, validation source, and chat mix) ----
        safe_src = "__".join(s.replace("/", "_") for s in sources)
        cache_key = (
            f"{safe_src}_{train_samples}_{split_ratio}_mix{chat_mix_ratio}"
            f"_valsrc{validation_source}"
        )
        cache_file = os.path.join(
            self.cache_dir, f"tradmedd_cache_{hash(cache_key)}.pkl"
        )

        if os.path.exists(cache_file):
            print(f"✅ Loading dataset from cache ({sources})")
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
                return cached["train"], cached["val"]

        # Split every source by default. If validation_source is configured,
        # only that source contributes validation rows; all other sources are
        # used entirely for training.
        train_parts = []
        val_parts = []
        for src in sources:
            print(f"⬇️ Loading dataset from Hugging Face Hub: {src}")
            ds = load_dataset(src)
            if "train" in ds:
                ds = ds["train"]
            else:
                ds = list(ds.values())[0]
            print(f"   columns={ds.column_names}  rows={len(ds)}")

            # `train_samples` applies PER-SOURCE for mixes (caps each source
            # before concat). For single-source it matches old behavior.
            if train_samples != -1:
                n = min(train_samples, len(ds))
                ds = ds.shuffle(seed=42).select(range(n))

            # ################### VALIDATION-ONLY SPLIT ####################
            if validation_source is None or src == validation_source:
                split = ds.train_test_split(test_size=split_ratio, seed=42)
                train_part = preformat(split["train"])
                val_part = preformat(split["test"])
                train_parts.append(train_part)
                val_parts.append(val_part)
                split_mode = "train + validation"
                val_rows = len(val_part)
            else:
                train_part = preformat(ds)
                train_parts.append(train_part)
                split_mode = "training only"
                val_rows = 0

            # ####################### DEBUG OUTPUT ########################
            print("#############################")
            print("DATASET SPLIT DEBUG")
            print(f"source: {src}")
            print(f"mode: {split_mode}")
            print(f"train rows: {len(train_part)}")
            print(f"validation rows: {val_rows}")
            print("#############################")
            # ##################### END DEBUG OUTPUT ######################
            # ################# END VALIDATION-ONLY SPLIT ################

        if len(train_parts) > 1:
            from datasets import concatenate_datasets
            train_set = concatenate_datasets(train_parts).shuffle(seed=42)
            val_set = (
                concatenate_datasets(val_parts) if len(val_parts) > 1 else val_parts[0]
            )
            print(f"🔀 Concatenated {len(sources)} sources: "
                  f"train={len(train_set)} val={len(val_set)}")
        else:
            train_set = train_parts[0]
            val_set = val_parts[0]

        # ---- Optional chat mix (FineTome-100k) -------------------------
        # Mục tiêu: đẩy vào SFT khoảng `chat_mix_ratio` tổng samples là
        # general-chat long-form. Giúp model không collapse vào MCQ-brevity
        # và không regress về Qwen3 tool-call scaffolding trên prompt mở.
        if chat_mix_ratio and chat_mix_ratio > 0:
            from datasets import concatenate_datasets

            from src.data.strategies.fine_tome import preformat as finetome_preformat
            n_target = int(len(train_set) * chat_mix_ratio / (1.0 - chat_mix_ratio))
            print(f"🔀 Mixing {n_target} chat samples from mlabonne/FineTome-100k "
                  f"(ratio={chat_mix_ratio:.0%})")

            chat_ds = load_dataset("mlabonne/FineTome-100k")
            chat_ds = chat_ds[list(chat_ds.keys())[0]]
            chat_ds = chat_ds.shuffle(seed=42).select(range(min(n_target, len(chat_ds))))
            chat_ds = chat_ds.map(
                finetome_preformat,
                batched=False,
                desc="FineTome -> conversations",
                remove_columns=chat_ds.column_names,
            )
            train_set = concatenate_datasets([train_set, chat_ds]).shuffle(seed=42)

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