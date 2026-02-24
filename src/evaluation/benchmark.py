from dotenv import load_dotenv
load_dotenv()

import sys
import os
sys.path.append(os.getenv("ROOT", ""))

import yaml
from uuid import uuid4
from datetime import datetime

from src.evaluation.evaluators.unsloth_adapter import UnslothAdapter
from src.core.loggers.term_logger import TerminalLogger
from src.core.loggers.interfaces.Ilogger import ILogger
from src.evaluation.evaluators.intefaces.Ilm_evaluator import IlmEvaluator
from src.core.trackers.interfaces.Itracker import ITracker
from src.core.trackers.dummy_tracker import DummyTracker
from src.utils.timer import Timer


class Benchmarker:
    def __init__(
        self,
        baseline_dir="group_outputs/baseline/",
        history_path="group_outputs/benchmark_history.log",
        tracker: ITracker = None,
        logger: ILogger = None,
    ):
        os.makedirs(baseline_dir, exist_ok=True)

        if not os.path.exists(history_path):
            with open(history_path, "w", encoding="utf-8") as f:
                f.write("")

        self.history_path = history_path
        self.baseline_dir = baseline_dir
        self.logger = logger or TerminalLogger()
        self.tracker = tracker or DummyTracker()

    # ==========================================================
    # FIND EXISTING BASELINE
    # ==========================================================

    def find_exist_baseline(self, compare_eval_config: dict, compare_train_config: dict):
        baseline_subdirs = os.listdir(self.baseline_dir)

        for baseline_subdir in baseline_subdirs:
            baseline_dir_path = os.path.join(self.baseline_dir, baseline_subdir)
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
                    self.logger.info(
                        f"Baseline already exists at {baseline_dir_path}"
                    )
                    return baseline_dir_path

        return None

    # ==========================================================
    # CORE BENCHMARK
    # ==========================================================

    def benchmark(
        self,
        config,
        model_path=None,
        model=None,
        tokenizer=None,
        tags=None,
        output_dir: str = "./outputs",
        max_length=2048,
        batch_size=8,
    ):
        tags = tags or []

        for dataset in config:
            dataset_name = dataset["name"]

            # Choose evaluator
            if "unsloth" in tags:
                evaluator: IlmEvaluator = UnslothAdapter(self.logger)
            else:
                raise ValueError(f"Not implemented tags {tags} for benchmark")

            tasks = []
            if dataset.get("subtasks"):
                tasks = [item["name"] for item in dataset["subtasks"]]

            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{dataset_name}.json")

            dataset_config = dataset.get("config", {})

            evaluator.load(
                model_path=model_path,
                model=model,
                tokenizer=tokenizer,
                max_length=max_length,
                batch_size=batch_size,
            )

            self.logger.info(
                f"Running benchmark on dataset {dataset_name} → {output_path}"
            )

            evaluator.run_evaluate(
                tasks=tasks,
                num_fewshot=dataset_config.get("num_fewshot", 5),
                limit=dataset_config.get("limit", None),
                random_seed=42,
                save_ouput_path=output_path,
                print_output=True,
            )

    # ==========================================================
    # RUN MULTIPLE
    # ==========================================================

    def run_multiple(self, configs):
        multiple_id = uuid4().hex
        self.logger.info("Running multiple benchmarks")

        for config in configs:

            benchmark_tags = config.get("benchmark_tags", [])
            eval_config = config.get("eval_config")
            train_config = config.get("train_config")
            model_path = config.get("model_path")
            model = config.get("model")
            tokenizer = config.get("tokenizer")
            tags = config.get("tags") or ["unsloth"]
            output_dir = config.get("output_dir")
            max_length = config.get("max_length", 2048)
            batch_size = config.get("batch_size", 8)
            trained = config.get("trained", True)
            rerun = config.get("rerun", True)

            self.logger.info(f"Running benchmark with tags {benchmark_tags}")

            Timer.reset()
            start_time = Timer.start()

            run_output_path = self.run(
                eval_config=eval_config,
                train_config=train_config,
                model_path=model_path,
                model=model,
                tokenizer=tokenizer,
                tags=tags,
                output_dir=output_dir,
                max_length=max_length,
                batch_size=batch_size,
                trained=trained,
                rerun=rerun,
            )

            end_time = Timer.end()
            duration = Timer.duration()

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = (
                f"Ran benchmark group id {multiple_id} "
                f"with tags {benchmark_tags}, "
                f"on {run_output_path}. "
                f"Start: {start_time}, End: {end_time}, Duration: {duration}"
            )

            self.logger.info(message)

            with open(self.history_path, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} - {message}\n")

    # ==========================================================
    # SINGLE RUN
    # ==========================================================

    def run(
        self,
        eval_config: str | dict,
        train_config: str | dict,
        model_path=None,
        model=None,
        tokenizer=None,
        tags=None,
        output_dir: str = "./outputs",
        max_length=2048,
        batch_size=8,
        trained=True,
        rerun=False,
    ):
        tags = tags or []

        # Load YAML if needed
        if isinstance(eval_config, str):
            with open(eval_config, "r") as f:
                eval_config = yaml.safe_load(f)

        if isinstance(train_config, str):
            with open(train_config, "r") as f:
                train_config = yaml.safe_load(f)

        # Baseline mode
        if not trained:
            if not rerun:
                exist_baseline_path = self.find_exist_baseline(
                    eval_config, train_config
                )
                if exist_baseline_path:
                    return exist_baseline_path

            output_dir = os.path.join(self.baseline_dir, str(uuid4()))
            self.logger.info(f"Run baseline benchmark on {output_dir}")

        else:
            self.logger.info(f"Run benchmark on trained model → {output_dir}")

        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "eval_config.yaml"), "w") as f:
            yaml.dump(eval_config, f)

        with open(os.path.join(output_dir, "train_config.yaml"), "w") as f:
            yaml.dump(train_config, f)

        self.benchmark(
            eval_config,
            model_path=model_path,
            model=model,
            tokenizer=tokenizer,
            tags=tags,
            output_dir=output_dir,
            max_length=max_length,
            batch_size=batch_size,
        )

        return output_dir