from unsloth import FastLanguageModel
from typing import Dict, Any, Optional
from pathlib import Path
import torch
from transformers import TrainingArguments
from functools import partial
from ..core import ExperimentTracker
from ..training import (
    ArgumentBuilder,
    LoraConfigBuilder,
    MultiDatasetTrainer,
)
from src.data.data_manager import DatasetManager
# from src.data.dataset_factory import DatasetFactory
from ..utils.error_handler import log_error
from src.data.format import formatting_func
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
                run = self._init_tracking()
            if self.config["lora"]:
                self.lora_config = LoraConfigBuilder(self.config["lora"])
                print("Lora config:", self.lora_config.get_config_dict)
            self.run_id= self.tracker.get_run_id()
            print("Run id:", self.run_id)
            
            model, tokenizer = self._load_model()
            
            compiled_formatting_func = partial(formatting_func, tokenizer=tokenizer)
            self.data_manager = DatasetManager()
            self.data_manager.store_datasets(self.config["datasets"])
            
            if not self.data_manager.validate_datasets(self.validate_samples, compiled_formatting_func):
                raise ValueError("Dataset validation failed")
            
            train_dataset, val_dataset, train_ladder, val_ladder = self.data_manager.load_combined_datasets()
            self.training_args = self._build_training_args()
            self.output_dir= self.training_args.output_dir
            
            trainer = self._build_trainer(
                model, tokenizer, 
                train_dataset, val_dataset,
                train_ladder, val_ladder
            )
            
            self._run_training(trainer)
            
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
            if self.lora_config:
                model = FastLanguageModel.get_peft_model(model, **self.lora_config.get_config_dict())
            return model, tokenizer
        else:
            raise NotImplementedError
        
   
    def _build_training_args(self):
        return ArgumentBuilder.build(self.config,self.run_id)
    
    def _build_trainer(self, model, tokenizer, train_ds, val_ds, train_ladder, val_ladder):
        return MultiDatasetTrainer(
            model=model,
            tokenizer=tokenizer,
            args=self.training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            train_idx_ladder=train_ladder,
            val_idx_ladder=val_ladder,
            formatting_func=partial(formatting_func, tokenizer=tokenizer),
            run_id=self.run_id,
            output_dir=self.output_dir,
            tracker=self.tracker
        )
    
    def _run_training(self, trainer):
        trainer.train()
        final_metrics = trainer.evaluate()
        self.tracker.log(final_metrics)
    
    def _cleanup(self):
        self.tracker.finish()
        torch.cuda.empty_cache()
    