import os
import yaml
import shutil
import argparse
import logging
from pathlib import Path
from typing import List, Dict
from benchmark import benchmark
def setup_logging(output_dir: str) -> logging.Logger:
    """Configure logging to both file and console"""
    log_file = Path(output_dir) / "benchmark.log"
    print("LOG FILE",log_file)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def validate_config(config: Dict) -> None:
    """Validate model configuration"""
    required_keys = {"name", "tags", "max_length", "batch_size"}
    if not all(key in config for key in required_keys):
        raise ValueError(f"Config missing required keys: {required_keys - set(config.keys())}")

def main(data_config_path: str, base_outdir: str) -> None:
    configs = [
        {
            "name": "unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit",
            "tags": ["unsloth", "baseline"],
            "max_length": 2048,
            "batch_size": 8
        },
        {
            "name": "unsloth/Qwen3-4B-unsloth-bnb-4bit",
            "tags": ["unsloth", "baseline"],
            "max_length": 2048,
            "batch_size": 8
        },
        {
            "name": "unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
            "tags": ["unsloth", "baseline"],
            "max_length": 2048,
            "batch_size": 8
        }
    ]

    try:
        with open(data_config_path, "r") as f:
            data_config = yaml.safe_load(f).get("datasets", {})
    except Exception as e:
        raise IOError(f"Failed to load data config: {str(e)}")

    slug = Path(data_config_path).stem  
    output_dir = Path(base_outdir) / slug
    
    try:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
    except Exception as e:
        raise IOError(f"Failed to prepare output directory: {str(e)}")

    logger = setup_logging(output_dir)
    logger.info(f"Starting benchmark for {len(configs)} models")

    target_config_path = output_dir / "config.yaml"
    try:
        with open(target_config_path, "w") as f:
            yaml.dump({"datasets": data_config}, f)
    except Exception as e:
        logger.error(f"Failed to save config: {str(e)}")
        raise

    for config in configs:
        try:
            validate_config(config)
            model_alias = config["name"].replace("/", "_")
            baseline_outdir = output_dir / model_alias
            baseline_outdir.mkdir(exist_ok=True)
            logger.info(f"Benchmarking {config['name']}")
            benchmark(
                data_config,
                config["name"],
                config["tags"],
                str(baseline_outdir),
                config["max_length"],
                config["batch_size"],
            )
        except Exception as e:
            logger.error(f"Failed to benchmark {config.get('name', 'unknown')}: {str(e)}")
            continue

    logger.info("Benchmarking completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline benchmark models")
    parser.add_argument(
        "--data_config_path",
        type=str,
        default="configs/evals/white.yaml",
        help="Path to data configuration YAML"
    )
    parser.add_argument(
        "--base_outdir",
        type=str,
        default="group_outputs/benchmark/baseline",
        help="Base output directory"
    )
    
    args = parser.parse_args()
    
    try:
        main(args.data_config_path, args.base_outdir)
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
        raise