from transformers import TrainerCallback
import wandb
import torch
import psutil
from time import time 
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class CustomWandbCallback(TrainerCallback):
    def __init__(self):
        self.start_time = time()
        self.step_count = 0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
            
        essential = {
            "epoch": state.epoch,
            "step": state.global_step,
            "lr": logs.get("learning_rate", 0),
            
            "loss": logs.get("loss", None),
            "eval_loss": logs.get("eval_loss", None),
            
            "samples/sec": self._calc_throughput(args, state),
            
            "gpu_mem_MB": self._get_gpu_memory() if torch.cuda.is_available() else None
        }
        
        wandb.log({k: v for k, v in essential.items() if v is not None})

    def _calc_throughput(self, args, state):
        elapsed = time() - self.start_time
        samples_processed = state.global_step * args.train_batch_size
        return samples_processed / elapsed if elapsed > 0 else 0
    
    def _get_gpu_memory(self):
        return torch.cuda.memory_allocated() / (1024 ** 2)  # MB

    def on_train_end(self, args, state, control, **kwargs):
        wandb.log({
            "total_training_hours": (time() - self.start_time) / 3600,
            "final_step": state.global_step
        })
        
