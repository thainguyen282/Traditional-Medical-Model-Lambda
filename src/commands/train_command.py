from unsloth import FastLanguageModel
from typing import Dict, Any, Optional
import torch
import os
from functools import partial
from src.core.experiment_tracker import ExperimentTracker
from src.training.argument_builder import ArgumentBuilder
from src.training.lora_config import  LoraConfigBuilder
from src.training.multi_ds_trainer import MedTrainer
from src.training.monitor import TrainingEvaluator
from src.data.data_manager import DatasetManager
from src.callbacks.benchmark_callback import FinalEvalCallback
from src.utils.error_handler import log_error
from src.data.format import format_conversations
from unsloth.chat_templates import get_chat_template

class TrainCommand:
    def __init__(
        self,
        config: Dict[str, Any],
        tracker: ExperimentTracker,
        output_dir: str = "./outputs",
        sweep_id: Optional[str] = None
    ):
        self.config = config
        self.tracker = tracker
        self.output_dir = output_dir
        self.validate_samples=5
    
    def execute(self):
        try:
            if self.tracker:
                self._init_tracking()
            if self.config["lora"]:
                self.lora_config = LoraConfigBuilder(self.config["lora"])
            self.run_id= self.tracker.get_run_id()
            print("---------------RUN ID:---------------", self.run_id)
            
            model, tokenizer = self._load_model()
            self.compiled_formatting_func = partial(format_conversations, tokenizer=tokenizer)
            self.data_manager = DatasetManager()
            self.data_manager.store_datasets(self.config["datasets"])
            
            if not self.data_manager.validate_datasets(self.validate_samples, self.compiled_formatting_func):
                raise ValueError("Dataset validation failed")
            
            train_dataset, val_dataset, train_ladder, val_ladder = self.data_manager.load_combined_datasets()
            train_config=self.config.get("training", {})
            output_dir=train_config.get("output_dir", "./outputs")
            output_dir=os.path.join(output_dir,self.run_id)
            os.makedirs(output_dir, exist_ok=True)
            
            print("---------------OUTPUT DIR:---------------", self.output_dir)
            
            model_output_dir=os.path.join(output_dir,"model")
            evaluate_output_dir=os.path.join(output_dir,"benchmark_evaluations")
            os.makedirs(model_output_dir, exist_ok=True)
            os.makedirs(evaluate_output_dir, exist_ok=True)
            train_config["output_dir"] = model_output_dir
            self.training_args = ArgumentBuilder.build(train_config=train_config)
            os.makedirs(evaluate_output_dir, exist_ok=True)
            
            self.benchmark_evaluator=TrainingEvaluator(
                output_dir=evaluate_output_dir,
                model=model,
                tokenizer=tokenizer,
                tags= self.config.get("monitoring", {}).get("tags"),
                tracker=self.tracker
            )

            trainer=MedTrainer(
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
            callbacks=[FinalEvalCallback(self.benchmark_evaluator)]
        )
            trainer.train()
            final_metrics = trainer.evaluate()
            self.tracker.log(final_metrics)
            return trainer
            
        except Exception as e:
            log_error(e, self.tracker)
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
                attn_implementation=self.config["unsloth"]["attn_implementation"]
            )
            model_name=self.config["model"]["name"]
            print("---------------MODEL NAME:---------------", model_name)
            if "qwen3" in model_name.lower():
                print("Using qwen3 tokenizer")
                tokenizer = get_chat_template(
                    tokenizer,
                    chat_template = "qwen-3",
                )
            if self.lora_config:
                model = FastLanguageModel.get_peft_model(model, **self.lora_config.get_config_dict())
            return model, tokenizer
        else:
            raise NotImplementedError
            
    def _cleanup(self):
        self.tracker.finish()
        torch.cuda.empty_cache()
    