from src.core.trackers.interfaces.Itracker import ITracker
from uuid import uuid4

class DummyTracker(ITracker):
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