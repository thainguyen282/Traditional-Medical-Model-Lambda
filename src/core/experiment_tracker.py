from abc import ABC, abstractmethod
import wandb

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

class WandbTracker(ExperimentTracker):
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.run = None
    
    def init(self, **kwargs):
        if self.enabled:
            self.run = wandb.init(**kwargs)
        return self
    def save_config(self, config: dict):
        if self.enabled and self.run:
            wandb.config.update(config)
    def log(self, metrics: dict):
        if self.enabled and self.run:
            wandb.log(metrics)
    
    def finish(self):
        if self.enabled and self.run:
            wandb.finish()
            self.run = None

class DummyTracker(ExperimentTracker):
    def init(self, **kwargs):
        print(f"[TRACKER] Initialized with config: {kwargs}")
        return self
    
    def log(self, metrics: dict):
        print(f"[TRACKER] Logging metrics: {metrics}")
    def save_config(self, config: dict):
        print(f"[TRACKER] Saving config: {config}")
        self.config = config
    def finish(self):
        print("[TRACKER] Finished")