import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
import yaml
from src.evaluation.benchmark import benchmark
from uuid import uuid4

class TrainingEvaluator:
    def __init__(
        self,
        base_line_output_dir: str = "baseline",
        eval_config_path:str="",
        tracker=None,
    ):
        self.base_line_output_dir = base_line_output_dir
        self.tracker = tracker
        if len(eval_config_path):
            self.eval_config_path=eval_config_path
        else:
            self.eval_config_path = "configs/evals/medical_mmlu.yaml"

    def set_eval_config_path(self, eval_config_path):
        self.eval_config_path = eval_config_path

    def is_exist_baseline(self, compare_train_config: dict):
        #  check if existing baseline experiement with given train_config and val_config has been performed or not
        compare_eval_config_path = self.eval_config_path
        with open(compare_eval_config_path) as f:
            compare_eval_config = yaml.safe_load(f)
        baseline_dirs = os.listdir(self.base_line_output_dir)
        for baseline_dir in baseline_dirs:
            baseline_dir_path = os.path.join(self.base_line_output_dir, baseline_dir)
            eval_config_path = os.path.join(baseline_dir_path, "eval_config.yaml")
            train_config_path = os.path.join(baseline_dir_path, "train_config.yaml")
            if os.path.exists(eval_config_path) and os.path.exists(train_config_path):
                with open(eval_config_path) as f:
                    eval_config = yaml.safe_load(f)
                with open(train_config_path) as f:
                    train_config = yaml.safe_load(f)
                if (
                    compare_train_config == train_config
                    and compare_eval_config == eval_config
                ):
                    print(
                        f"Baseline experiment with train_config path {train_config_path} and eval_config {eval_config_path} already exists"
                    )
                    return True
        return False

    def run_final_evaluation(
        self,
        output_dir,
        train_config,
        model_path=None,
        model=None,
        tokenizer=None,
        trained=False,
        tags=None,
        max_length=2048,
        batch_size=8,
    ) -> Dict:
        print(f"Running final benchmark evaluation with tags {tags}")
        with open(self.eval_config_path) as f:
            eval_config = yaml.safe_load(f)
        if not trained:
            if self.is_exist_baseline(train_config):
                return

            output_dir = os.path.join(self.base_line_output_dir, str(uuid4()))
            print("Run baseline benchmark on output dir", output_dir)
            os.makedirs(output_dir, exist_ok=True)
            eval_config_path = os.path.join(output_dir, "eval_config.yaml")
            train_config_path = os.path.join(output_dir, "train_config.yaml")
        else:
            eval_config_path = os.path.join(output_dir, "eval_config.yaml")
            train_config_path = os.path.join(output_dir, "train_config.yaml")
        with open(eval_config_path, "w") as f:
            yaml.dump(eval_config, f)
        with open(train_config_path, "w") as f:
            yaml.dump(train_config, f)

        benchmark(
            config=eval_config,
            model_path=model_path,
            model=model,
            tokenizer=tokenizer,
            tags=tags,
            output_dir=output_dir,
            max_length=max_length,
            batch_size=batch_size,
        )
