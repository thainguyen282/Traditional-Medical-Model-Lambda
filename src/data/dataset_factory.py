from typing import Tuple
from datasets import Dataset
from .strategies import (
    # DummyDatasetStrategy,
    FineTomeDatasetStrategy,
    # MedSampleDatasetStrategy
)

class DatasetFactory:
    STRATEGIES = {
        # "dummy": DummyDatasetStrategy,
        "fine_tome": FineTomeDatasetStrategy,
        # "med_sample": MedSampleDatasetStrategy
    }
    
    @classmethod
    def create_datasets(cls, tag: str, config: dict) -> Tuple[Dataset, Dataset]:
        strategy_class = cls.STRATEGIES.get(tag)
        if not strategy_class:
            raise ValueError(f"Unknown dataset tag: {tag}")
        
        strategy = strategy_class()
        return strategy.load(config)