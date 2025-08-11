from src.core.trackers.interfaces.Itracker import ITracker
import wandb
from uuid import uuid4

class WandbTracker(ITracker):
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
