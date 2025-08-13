from unsloth import FastLanguageModel
from typing import Dict, Any, Optional
import torch
import os
import copy
from functools import partial
from src.core.loggers.interfaces.Ilogger import ILogger
from src.core.trackers.interfaces.Itracker import ITracker
from src.training.argument_builder import ArgumentBuilder
from src.training.lora_config import LoraConfigBuilder
from src.training.med_trainer import MedTrainer
from src.core.loggers.term_logger import TerminalLogger

from src.data.data_manager import DatasetManager
from src.data.format import format_conversations
from unsloth.chat_templates import get_chat_template
from src.evaluation.benchmark import Benchmarker


class TrainCommand:
    def __init__(
        self,
        config: Dict[str, Any],
        tracker: Optional[ITracker] = None,
        logger: Optional[ILogger] = None,
        output_dir: str = "./outputs",
        base_config: Optional[Dict[str, Any]] = None,
        sweep_id: Optional[str] = None,
    ):
        self.base_config = base_config or {}
        self.config = config
        self.tracker = tracker
        self.logger=logger or TerminalLogger()
        self.output_dir = output_dir
        self.validate_samples = 5

    def execute(self):
        try:
            if self.tracker:
                self._init_tracking()
            if self.config["lora"]:
                self.lora_config = LoraConfigBuilder(self.config["lora"])
            self.run_id = self.tracker.get_run_id()
            print("---------------RUN ID:---------------", self.run_id)
            
            # [ For benchmark in later step
            non_trained_model, _ = self._load_model()
            cloned_config = copy.deepcopy(self.config)
            # ]
            
            model, tokenizer = self._load_model()
            self.compiled_formatting_func = partial(
                format_conversations, tokenizer=tokenizer
            )
            self.data_manager = DatasetManager()
            self.data_manager.store_datasets(self.config["datasets"])

            if not self.data_manager.validate_datasets(
                self.validate_samples, self.compiled_formatting_func
            ):
                raise ValueError("Dataset validation failed")

            train_dataset, val_dataset, train_ladder, val_ladder = (
                self.data_manager.load_combined_datasets()
            )
            train_config = self.config.get("training", {})
            output_dir = train_config.get("output_dir", "./outputs")
            output_dir = os.path.join(output_dir, self.run_id)

            print("---------------OUTPUT DIR:---------------", output_dir)

            model_output_dir = os.path.join(output_dir, "model")
            evaluate_output_dir = os.path.join(output_dir, "benchmark_evaluations")
            train_config["output_dir"] = model_output_dir
            self.training_args = ArgumentBuilder.build(train_config=train_config)

            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(model_output_dir, exist_ok=True)
            os.makedirs(evaluate_output_dir, exist_ok=True)
            os.makedirs(evaluate_output_dir, exist_ok=True)

            print("cloned_config",cloned_config, cloned_config.get("benchmark", {}).get("path", ""),)
            self.benchmark_evaluator = Benchmarker(
                logger=self.logger,
                tracker=self.tracker,
            )

            trainer = MedTrainer(
                model=model,
                tokenizer=tokenizer,
                args=self.training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                train_idx_ladder=train_ladder,
                val_idx_ladder=val_ladder,
                formatting_func=self.compiled_formatting_func,
                run_id=self.run_id,
                tracker=self.tracker,
                benchmark_evaluator=self.benchmark_evaluator,
            )

            trainer.train()
            final_metrics = trainer.evaluate()
            model_name=self.config["model"]["name"]
            eval_config = self.config.get("benchmark", self.base_config.get("benchmark", {})).get("path", "")
            self.logger.info(f"Benchmark evaluation path: {eval_config} ")
            self.benchmark_evaluator.run_multiple(
                configs=[
                    {
                        "benchmark_tags": ["Baseline",model_name],
                        "eval_config": eval_config,
                        "train_config": cloned_config,
                        "model_path": None,
                        "model": non_trained_model,
                        "tokenizer": tokenizer,
                        "tags": self.config.get("monitoring", {}).get("tags"),
                        "output_dir": evaluate_output_dir,
                        "max_length": self.config.get("model", {}).get("max_length", 2048),
                        "batch_size": self.base_config.get("benchmark", {}).get("batch_size", 8),
                        "trained": False,
                        "rerun": True,
                    },
                    {
                        "benchmark_tags":["Trained",model_name],
                        "eval_config": eval_config,
                        "train_config": cloned_config,
                        "model_path": None,
                        "model": non_trained_model,
                        "tokenizer": tokenizer,
                        "tags": self.config.get("monitoring", {}).get("tags"),
                        "output_dir": evaluate_output_dir,
                        "max_length": self.config.get("model", {}).get("max_length", 2048),
                        "batch_size": self.base_config.get("benchmark", {}).get("batch_size", 8),
                        "trained": True,
                        "rerun": True,
                    }
                    
                ]
            )
         

           
        except Exception as e:
            self.logger.error(str(e))
            raise
        finally:
            self._cleanup()

    def _init_tracking(self):
        return self.tracker.init(
            project=self.config.get("monitoring", {}).get("project"),
            name=self.config.get("name"),
            config=self.config,
            group=self.config.get("model").get("name"),
            tags=self.config.get("monitoring", {}).get("tags"),
        )

    def _load_model(self):
        if self.config["unsloth"]:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config["model"]["name"],
                max_seq_length=self.config["model"]["max_seq_length"],
                load_in_4bit=self.config["unsloth"]["load_in_4bit"],
                attn_implementation=self.config["unsloth"]["attn_implementation"],
            )
            model_name = self.config["model"]["name"]
            print("---------------MODEL NAME:---------------", model_name)
            if "qwen3" in model_name.lower():
                print("Using qwen3 tokenizer")
                tokenizer = get_chat_template(
                    tokenizer,
                    chat_template="qwen-3",
                )
            if self.lora_config:
                model = FastLanguageModel.get_peft_model(
                    model, **self.lora_config.get_config_dict()
                )
            return model, tokenizer
        else:
            raise NotImplementedError

    def _cleanup(self):
        self.tracker.finish()
        torch.cuda.empty_cache()
