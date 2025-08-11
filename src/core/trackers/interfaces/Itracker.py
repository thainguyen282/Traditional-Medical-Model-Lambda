from abc import ABC, abstractmethod
import wandb
from uuid import uuid4

class ITracker(ABC):
    @abstractmethod
    def init(self, **kwargs):
        pass
    @abstractmethod
    def save_config(self, config: dict):
        pass
    
    @abstractmethod
    def log(self, metrics: dict):
        pass
    
    @abstractmethod
    def finish(self):
        pass
    
    @abstractmethod
    def get_run_id(self):
        pass
