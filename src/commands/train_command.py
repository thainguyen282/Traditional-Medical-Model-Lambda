from unsloth import FastLanguageModel
from typing import Dict, Any, Optional
from pathlib import Path
import torch
from transformers import TrainingArguments

from ..core import ExperimentTracker
from ..training import (
    ArgumentBuilder,
    LoraConfigBuilder,
    MultiDatasetTrainer
)
from ..data import DatasetFactory
from ..utils.error_handler import log_error
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
        self.sweep_id = sweep_id
    
    def execute(self):
        try:
            run = self._init_tracking()
            
            model, tokenizer = self._load_model()
            
            train_dataset, val_dataset, train_ladder, val_ladder = self._load_datasets()
            
            training_args = self._build_training_args()
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
            project=self.config.get("monitoring", {}).get("project_name"),
            name=self.config.get("name"),
            config=self.config,
            group=self.config.get("model"),
            tags=self.config.get("tags"),
        )
    
    def _load_model(self):
        if self.config["unsloth"]:
            return FastLanguageModel.from_pretrained(
                model_name=self.config["model"]["name"],
                max_seq_length=self.config["model"]["max_seq_length"],
                load_in_4bit=self.config["unsloth"]["load_in_4bit"],
                attn_implementation=self.config["unsloth"]["attn_implementation"]
            )
        else:
            raise NotImplementedError
        
    def _load_datasets(self):
        return DatasetFactory.load_combined_datasets(self.config["datasets"])
    
    def _build_training_args(self):
        return ArgumentBuilder.build(self.config)
    
    def _build_trainer(self, model, tokenizer, train_ds, val_ds, train_ladder, val_ladder):
        return MultiDatasetTrainer(
            model=model,
            tokenizer=tokenizer,
            args=self.training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            train_idx_ladder=train_ladder,
            val_idx_ladder=val_ladder,
            formatting_func=self._formatting_func
        )
    
    def _run_training(self, trainer):
        trainer.train()
        final_metrics = trainer.evaluate()
        self.tracker.log(final_metrics)
    
    def _cleanup(self):
        self.tracker.finish()
        torch.cuda.empty_cache()
    
    @staticmethod
    def _formatting_func(examples, tokenizer):
        # Your formatting implementation
        pass