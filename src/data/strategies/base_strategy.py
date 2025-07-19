from abc import ABC, abstractmethod
from typing import Tuple
from datasets import Dataset

class BaseDatasetStrategy(ABC):
    @abstractmethod
    def load(self, config: dict) -> Tuple[Dataset, Dataset]:
        """Load and return (train_dataset, val_dataset)"""
        pass