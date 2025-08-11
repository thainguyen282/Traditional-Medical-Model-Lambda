import argparse
from src.commands import TrainCommand
# from src.core.experiment_tracker import WandbTracker,DummyTracker
from src.core.trackers.wandb_tracker import WandbTracker
from src.core.trackers.dummy_tracker import DummyTracker
from src.core.trackers.interfaces.Itracker import ITracker
from src.core.config_manager import ConfigManager
from src.core.loggers.term_logger import TerminalLogger
from src.core.loggers.dual_logger import DualLogger
from src.core.loggers.interfaces.Ilogger import ILogger
def parse_args():
    parser = argparse.ArgumentParser(description="Train language models")
    
    parser.add_argument("--config", type=str, default="configs/experiments/unsloth-qwen3-0.6B/exp0001/config.yaml")
    parser.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep")
    parser.add_argument("--test", action="store_true", help="Test pipeline with minimal config")
    parser.add_argument("--dry_run_config",type="str",default="configs/experiments/dry_run.yaml")
    parser.add_argument("--tracker",  type=str, default="wandb", help="tracker to save metrics. Accepted values: wandb, dummy")
    parser.add_argument("--logger",  type=str, default="dual", help="logger to save training output. Accepted values: dual(log to both console and file), terminal (terminal)")
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
    if args.test:
        print("Run test mode with minimal set up")
        args.config=args.dry_run_config
        args.tracker="dummy"
        args.logger="terminal"
        
    tracker:ITracker
    logger:ILogger
    if args.logger == "dual":
        logger = DualLogger()
    elif args.logger == "terminal":
        logger = TerminalLogger()
    else:
        raise ValueError(f"Unknown logger: {args.logger}")
    
    if args.tracker == "wandb":
        tracker = WandbTracker()
        override_args["training.report_to"] = "wandb"
        
    elif args.tracker == "dummy":
        tracker=DummyTracker()
        override_args["training.report_to"] = "none"
    else:
        raise ValueError(f"Unknown tracker: {args.tracker}")
    
    config_manager = ConfigManager()
    try:
        config = config_manager.load(args.config, override_args)
    except ValueError as e:
        print(f"Config validation error: {e}")
        return
    tracker.save_config(config)
        
    print("Validated config:", config)

    command = TrainCommand(config=config,tracker= tracker,logger=logger)
    
    command.execute()
if __name__ == "__main__":
    main()