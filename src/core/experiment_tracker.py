from abc import ABC, abstractmethod
import wandb
from uuid import uuid4

class ExperimentTracker(ABC):
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

class WandbTracker(ExperimentTracker):
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.run = None
    
    def init(self, **kwargs):
        if self.enabled:
            print("WANDB: config:", kwargs)
            self.run = wandb.init(**kwargs)
        return self
    def get_run_id(self):
        if self.enabled and self.run:
            return self.run.id
        print("Wandb not initialized, returning hashed run id")
        return str(uuid4())[:8]
    def save_config(self, config: dict):
        if self.enabled and self.run:
            wandb.config.update(config)
    def log(self, metrics: dict):
        if self.enabled and self.run:
            wandb.log(metrics)
    
    def finish(self):
        if self.enabled and self.run:
            print("Call wandb finish")
            wandb.finish()
            self.run = None

class DummyTracker(ExperimentTracker):
    def init(self, **kwargs):
        print(f"[TRACKER] Initialized with config: {kwargs}")
        return self
    
    def get_run_id(self):
        return str(uuid4())[:8]
    
    def log(self, metrics: dict):
        print(f"[TRACKER] Logging metrics: {metrics}")
    def save_config(self, config: dict):
        print(f"[TRACKER] Saving config: {config}")
        self.config = config
    def finish(self):
        print("[TRACKER] Finished")