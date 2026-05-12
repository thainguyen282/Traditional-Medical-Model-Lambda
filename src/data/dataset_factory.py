from typing import Tuple, List
from datasets import Dataset, concatenate_datasets
import sys
import os 
from dotenv import load_dotenv
load_dotenv()
ROOT_DIR = os.getenv("ROOT")
sys.path.append(ROOT_DIR)
from src.data.strategies.fine_tome import FineTomeDatasetStrategy
from src.data.strategies.venera_med_m1 import  VeneraMedDatasetStrategy


class DatasetFactory:
    STRATEGIES = {
        "fine_tome": FineTomeDatasetStrategy,
        "venera_med_m1": VeneraMedDatasetStrategy,
        # Same strategy as venera_med_m1 — separate tag so the library can hold
        # both (single-source mc) and (multi-source mc + 2topics) without
        # tag collisions when an experiment ever lists both datasets.
        "venera_med_mixed": VeneraMedDatasetStrategy,
        "venera_med_2topics": VeneraMedDatasetStrategy,
    }

    @classmethod
    def create_datasets(cls, tag: str, config: dict) -> Tuple[Dataset, Dataset]:
        """Create train and validation datasets using the specified strategy"""
        strategy_class = cls.STRATEGIES.get(tag)
        if not strategy_class:
            raise ValueError(f"Unknown dataset tag: {tag}")

        strategy = strategy_class()
        return strategy.load(config)

if __name__ == "__main__":
    ds_config = {
        "tag": "fine_tome",
        "path": "mlabonne/FineTome-100k",
        "train_samples": 30,
        "weight": 0.7,
    }
    train_ds, val_ds, train_idx, val_idx = DatasetFactory.load_combined_datasets(
        [ds_config]
    )
