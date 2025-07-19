from typing import Dict, Any, Optional
import wandb
from ..core import ExperimentTracker, ConfigManager
from .train_command import TrainCommand

class SweepCommand:
    def __init__(
        self,
        config: Dict[str, Any],
        tracker: ExperimentTracker,
        sweep_count: int = 5,
        test: bool = False
    ):
        self.config = config
        self.tracker = tracker
        self.sweep_count = sweep_count
        self.test = test
    
    def execute(self):
        sweep_config = self.config.get("sweep", {})
        if not sweep_config:
            raise ValueError("No sweep configuration found")
        
        if self.test:
            self._setup_test_config()
        
        sweep_id = wandb.sweep(
            sweep_config,
            project=f"{self.config.get('monitoring', {}).get('project_name')}-sweeps"
        )
        
        wandb.agent(
            sweep_id,
            function=self._sweep_train,
            count=self.sweep_count
        )
    
    def _sweep_train(self):
        command = TrainCommand(self.config, self.tracker, sweep_id=wandb.run.sweep_id)
        command.execute()
    
    def _setup_test_config(self):
        """Simplify config for testing"""
        self.config["experiments"] = [self.config["experiments"][0]]
        self.config["defaults"]["dataset"] = [{
            "tag": "fine_tome",
            "path": "mlabonne/FineTome-100k",
            "train_samples": 10,
            "weight": 1.0
        }]
        self.sweep_count = 1