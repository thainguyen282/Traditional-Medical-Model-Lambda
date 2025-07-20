import argparse
from typing import Dict, Any
from pathlib import Path

from src.commands import TrainCommand, SweepCommand
from src.core import ConfigManager, WandbTracker

def parse_args():
    parser = argparse.ArgumentParser(description="Train language models")
    
    parser.add_argument("--config", type=str, default="configs/experiments/qwen-0.6B/exp0001/config.yaml")
    parser.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep")
    parser.add_argument("--test", action="store_true", help="Test pipeline with minimal config")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable WandB logging")
    
    args, unknown = parser.parse_known_args()
    
    allowed_overrides = {
        'training.batch_size': int,
        'adapter.lora_rank': int,
        'context_length': int
    }
    
    override_args = {}
    for arg in unknown:
        if arg.startswith("--") and '=' in arg:
            key = arg[2:].split('=')[0]
            if key in allowed_overrides:
                try:
                    override_args[key] = allowed_overrides[key](arg.split('=')[1])
                except ValueError:
                    print(f"Warning: Invalid value for {key}, skipping")
    
    return args, override_args

def main():
    args, override_args = parse_args()
    print("Override args:", override_args)
    
    tracker = WandbTracker(enabled=not args.disable_wandb)
    config_manager = ConfigManager()
    
    try:
        config = config_manager.load(args.config, override_args)
    except ValueError as e:
        print(f"Config validation error: {e}")
        return
        
    print("Validated config:", config)
    

    command = TrainCommand(config=config,tracker= tracker)
    
    command.execute()
if __name__ == "__main__":
    main()