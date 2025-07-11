import wandb
import time
import torch
from transformers import TrainerCallback

# Custom callback to log additional metrics
class CustomWandbCallback(TrainerCallback):
    def __init__(self):
        self.start_time = time.time()
        self.total_samples = 0

    def on_step_end(self, args, state, control, **kwargs):
        # Log samples processed
        self.total_samples += args.per_device_train_batch_size * args.gradient_accumulation_steps
        wandb.log({"samples_processed": self.total_samples}, step=state.global_step)

        # Log peak GPU memory (if available)
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
            wandb.log({"peak_gpu_memory_mb": peak_mem}, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        # Log total training time
        training_time = time.time() - self.start_time
        wandb.log({"total_training_time_sec": training_time})
        
        wandb.log({"average_training_time_per_sample": training_time/self.total_samples})


