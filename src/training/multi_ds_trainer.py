from transformers import Trainer
from tqdm.auto import tqdm
import torch
from trl import SFTTrainer, SFTConfig
import wandb  
import os
import json

class MultiDatasetTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_idx_ladder = kwargs.pop("train_idx_ladder", [])
        self.eval_idx_ladder = kwargs.pop("val_idx_ladder", [])
        self.dataset_names = kwargs.pop("dataset_names", {})
        self.tracker = kwargs.pop("tracker", None)
        self.output_dir= kwargs.pop("output_dir", "temp")
        self.error_log_path=os.path.join(self.output_dir, "training_errors.jsonl")
        self.skipped_samples = 0
        self.error_log = []
        
    def _log_error(self, error_type, error_msg, example=None, dataset_tag=None):
        """Logs errors to both file and W&B with structured metadata."""
        import json
        from datetime import datetime
        
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": self.state.global_step,
            "type": error_type,
            "message": str(error_msg),
            "dataset": dataset_tag,
            "example": example if example else None
        }
        
        with open(self.error_log_path, "a") as f:
            f.write(json.dumps(error_entry) + "\n")
        
        if self.tracker and (self.state.global_step % 10 == 0):  # Log every 10 steps
            self.tracker.log({
                "error/type": error_type,
                "error/message": error_msg[:200], 
                "error/dataset": dataset_tag,
                "error/count": self.skipped_samples,
            })

    def compute_loss(self, model, inputs, return_outputs=False):
        """Override to skip bad samples during training"""
        try:
            return super().compute_loss(model, inputs, return_outputs)
        except Exception as e:
            batch_idx = self.state.global_step * self.args.per_device_train_batch_size
            dataset_tag = self._get_dataset_from_index(batch_idx, self.train_idx_ladder)
            
            error_msg = f"Step {self.state.global_step}: {str(e)}"
            self._log_error(error_msg, inputs, dataset_tag)
            self.skipped_samples += 1
            
            return torch.tensor(0.0, requires_grad=True)

    def prediction_step(self, model, inputs, **kwargs):
        try:
            return super().prediction_step(model, inputs, **kwargs)
        except Exception as e:
            batch_idx = self.eval_step_counter * self.args.per_device_eval_batch_size
            dataset_tag = self._get_dataset_from_index(batch_idx, self.eval_idx_ladder)
            
            self._log_error(f"Eval error: {str(e)}", inputs, dataset_tag)
            return (None, None, None)  

    def log(self, logs, **kwargs):
        """Enhanced logging with error stats"""
        if torch.cuda.is_available():
            logs.update({
                "gpu_mem_alloc_gb": torch.cuda.memory_allocated() / 1e9,
                "gpu_mem_reserved_gb": torch.cuda.memory_reserved() / 1e9,
            })
        
        logs["skipped_samples"] = self.skipped_samples
        if self.error_log:
            logs["last_error"] = self.error_log[-1][:100]  # Truncate long errors
        
        super().log(logs, **kwargs)

""""
Explain 
global_step = 10

gradient_accumulation_steps = 2

per_device_train_batch_size = 8

world_size = 2 (2 GPUs)

len(dataset) = 1000
absolute_batch_idx = global_step * gradient_accumulation_steps
total_batch_size = per_device_train_batch_size * world_size
start_idx = absolute_batch_idx * total_batch_size
 end_idx = start_idx + total_batch_size
"""
