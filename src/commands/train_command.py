from unsloth import FastLanguageModel
from typing import Dict, Any, Optional
import torch
import os
import copy
from functools import partial
import gc

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
    ):
        self.base_config = base_config or {}
        self.config = config
        self.tracker = tracker
        self.logger = logger or TerminalLogger()
        self.output_dir = output_dir
        self.validate_samples = 5

        self.lora_config = (
            LoraConfigBuilder(self.config["lora"])
            if self.config.get("lora")
            else None
        )

    # ==========================================================
    # MAIN EXECUTION
    # ==========================================================

    def execute(self):
        trainer = None
        model = None
        baseline_model = None

        try:
            if self.tracker:
                self._init_tracking()

            self.run_id = self.tracker.get_run_id()
            print("---------------RUN ID:---------------", self.run_id)

            # --------------------------------------------------
            # Load training model
            # --------------------------------------------------
            model, tokenizer = self._load_training_model()

            formatting_func = partial(
                format_conversations,
                tokenizer=tokenizer,
            )

            # --------------------------------------------------
            # Dataset
            # --------------------------------------------------
            data_manager = DatasetManager()
            data_manager.store_datasets(self.config["datasets"])

            if not data_manager.validate_datasets(
                self.validate_samples,
                formatting_func,
            ):
                raise ValueError("Dataset validation failed")

            train_dataset, val_dataset, train_ladder, val_ladder = (
                data_manager.load_combined_datasets()
            )

            # --------------------------------------------------
            # Training args
            # --------------------------------------------------
            train_config = self.config.get("training", {})

            output_dir = os.path.join(
                train_config.get("output_dir", "./outputs"),
                self.run_id,
            )

            model_output_dir = os.path.join(output_dir, "model")
            eval_output_dir = os.path.join(output_dir, "benchmark_evaluations")

            train_config["output_dir"] = model_output_dir

            training_args = ArgumentBuilder.build(train_config=train_config)

            os.makedirs(model_output_dir, exist_ok=True)
            os.makedirs(eval_output_dir, exist_ok=True)

            benchmark_evaluator = Benchmarker(
                logger=self.logger,
                tracker=self.tracker,
            )

            trainer = MedTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                train_idx_ladder=train_ladder,
                val_idx_ladder=val_ladder,
                formatting_func=formatting_func,
                run_id=self.run_id,
                tracker=self.tracker,
                benchmark_evaluator=benchmark_evaluator,
            )

            # --------------------------------------------------
            # TRAIN
            # --------------------------------------------------
            # trainer.train()
            trainer.train(
                resume_from_checkpoint="CPT/cpt_qwen3_medical/checkpoint-1240"
            )
            # --------------------------------------------------
            # HARD MEMORY CLEANUP
            # --------------------------------------------------
            if trainer:
                trainer.model = None

            del trainer
            del model
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            torch.cuda.synchronize()

            # --------------------------------------------------
            # Load fresh baseline (NO LORA)
            # --------------------------------------------------
            with torch.no_grad():
                baseline_model, baseline_tokenizer = self._load_baseline_model()

                baseline_model.eval()
                for param in baseline_model.parameters():
                    param.requires_grad = False

                eval_config = self.config.get("benchmark", {}).get("path", "")

                benchmark_evaluator.run_multiple(
                    configs=[
                        {
                            "benchmark_tags": ["Baseline"],
                            "eval_config": eval_config,
                            "train_config": copy.deepcopy(self.config),
                            "model": baseline_model,
                            "tokenizer": baseline_tokenizer,
                            "output_dir": eval_output_dir,
                            "trained": False,
                            "rerun": True,
                        }
                    ]
                )

            # --------------------------------------------------
            # Cleanup baseline
            # --------------------------------------------------
            del baseline_model
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        except Exception as e:
            self.logger.error(str(e))
            raise

        finally:
            self._cleanup()

    # ==========================================================
    # LOAD TRAINING MODEL (WITH LORA)
    # ==========================================================

    def _load_training_model(self):

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config["model"]["name"],
            max_seq_length=self.config["model"]["max_seq_length"],
            load_in_4bit=self.config["unsloth"]["load_in_4bit"],
            attn_implementation=self.config["unsloth"]["attn_implementation"],
            device_map="auto",
        )

        model_name = self.config["model"]["name"]

        if "qwen3" in model_name.lower():
            tokenizer = get_chat_template(
                tokenizer,
                chat_template="qwen-3",
            )

        if self.lora_config:
            model = FastLanguageModel.get_peft_model(
                model,
                **self.lora_config.get_config_dict(),
            )

        return model, tokenizer

    # ==========================================================
    # LOAD BASELINE MODEL (NO LORA)
    # ==========================================================

    def _load_baseline_model(self):

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config["model"]["name"],
            max_seq_length=self.config["model"]["max_seq_length"],
            load_in_4bit=self.config["unsloth"]["load_in_4bit"],
            attn_implementation=self.config["unsloth"]["attn_implementation"],
            device_map="auto",
        )

        model_name = self.config["model"]["name"]

        if "qwen3" in model_name.lower():
            tokenizer = get_chat_template(
                tokenizer,
                chat_template="qwen-3",
            )

        return model, tokenizer

    # ==========================================================
    # TRACKING
    # ==========================================================

    def _init_tracking(self):
        return self.tracker.init(
            project=self.config.get("monitoring", {}).get("project"),
            name=self.config.get("name"),
            config=self.config,
            group=self.config.get("model").get("name"),
            tags=self.config.get("monitoring", {}).get("tags"),
        )

    # ==========================================================
    # CLEANUP
    # ==========================================================

    def _cleanup(self):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()