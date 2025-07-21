from typing import Tuple, List
from datasets import Dataset, concatenate_datasets
import sys

sys.path.append("/root/VeneraMedicalM1")
from src.data.strategies.fine_tome import FineTomeDatasetStrategy
from src.data.strategies.med_sample import  MedSampleDatasetStrategy


class DatasetFactory:
    STRATEGIES = {
        "fine_tome": FineTomeDatasetStrategy,
        "med_sample": MedSampleDatasetStrategy,
    }

    @classmethod
    def create_datasets(cls, tag: str, config: dict) -> Tuple[Dataset, Dataset]:
        """Create train and validation datasets using the specified strategy"""
        strategy_class = cls.STRATEGIES.get(tag)
        if not strategy_class:
            raise ValueError(f"Unknown dataset tag: {tag}")

        strategy = strategy_class()
        return strategy.load(config)

    @classmethod
    def load_combined_datasets(
        cls, dataset_configs: List[dict]
    ) -> Tuple[List[Dataset], List[Dataset], List[int], List[int]]:
        """Load and combine multiple datasets"""
        combine_train_datasets = []
        combine_val_datasets = []
        train_idx_ladder = []
        val_idx_ladder = []
        cur_train_idx = 0
        cur_val_idx = 0

        for dataset_config in dataset_configs:
            tag = dataset_config.get("tag")
            print("Loading dataset:", tag)
            train_dataset, val_dataset = cls.create_datasets(tag, dataset_config)

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
    ds_config = {
        "tag": "fine_tome",
        "path": "mlabonne/FineTome-100k",
        "train_samples": 30,
        "weight": 0.7,
    }
    train_ds, val_ds, train_idx, val_idx = DatasetFactory.load_combined_datasets(
        [ds_config]
    )
