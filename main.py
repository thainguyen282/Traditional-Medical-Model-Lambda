import argparse
from typing import Dict, Any
from pathlib import Path

from src.commands import TrainCommand, SweepCommand
from src.core import ConfigManager, WandbTracker

def parse_args():
    parser = argparse.ArgumentParser(description="Train language models")
    parser.add_argument("--config", type=str, default="configs/experiment_dark.yaml")
    parser.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep")
    parser.add_argument("--test", action="store_true", help="Test pipeline with minimal config")
    parser.add_argument("--sweep-count", type=int, default=5, help="Number of sweep trials")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable WandB logging")
    return parser.parse_args()

def main():
    args = parse_args()
    
    config_manager = ConfigManager(args.config)
    tracker = WandbTracker(enabled=not args.disable_wandb)
    
    raw_config = config_manager.load()
    config = config_manager.validate(raw_config)
    print("Config:", config)
    
    if args.sweep:
        command = SweepCommand(config, tracker, sweep_count=args.sweep_count, test=args.test)
    else:
        command = TrainCommand(config, tracker)
    
    command.execute()

if __name__ == "__main__":
    main()