import torch
from trl import SFTTrainer, SFTConfig
import os


class MedTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_idx_ladder = kwargs.pop("train_idx_ladder", [])
        self.eval_idx_ladder = kwargs.pop("val_idx_ladder", [])
        self.dataset_names = kwargs.pop("dataset_names", {})
        self.tracker = kwargs.pop("tracker", None)
        self.output_dir = kwargs.pop("output_dir", "temp")
        self.error_log_path = os.path.join(self.output_dir, "training_errors.jsonl")
        self.skipped_samples = 0
        self.error_log = []
        self.samples_seen = 0
        self.benchmark_evaluator = kwargs.pop("benchmark_evaluator", None)
        self._train_dataloader = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def _get_dataset_from_index(self, batch_idx: int, ladder: list):
        cumulative = 0
        for (start, end), tag in ladder:
            if start <= batch_idx < end:
                return tag
            cumulative += end - start
        return "unknown"

    def training_step(self, model, inputs, num_items_in_batch=None):

        
        if self.benchmark_evaluator:
            new_samples = (
                self.args.per_device_train_batch_size  # Batch per GPU
                * self.args.world_size  # Number of GPUs
                * self.args.gradient_accumulation_steps  # Gradient accumulation
            )
            self.benchmark_evaluator.update_samples(new_samples)
            if self.benchmark_evaluator.should_evaluate():
                self.benchmark_evaluator.run_evaluation()
        loss = super().training_step(model, inputs, num_items_in_batch)

        return loss

       
    def _log_error(self, error_type, error_msg, example=None, dataset_tag=None):
        import json
        from datetime import datetime

        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": self.state.global_step,
            "type": error_type,
            "message": str(error_msg),
            "dataset": dataset_tag,
            "example": example if example else None,
        }

        with open(self.error_log_path, "a") as f:
            f.write(json.dumps(error_entry) + "\n")

        if self.tracker and (self.state.global_step % 10 == 0):
            self.tracker.log(
                {
                    "error/type": error_type,
                    "error/message": error_msg[:200],
                    "error/dataset": dataset_tag,
                    "error/count": self.skipped_samples,
                }
            )


    def log(self, logs, start_time=None, **kwargs):
        """Enhanced logging with error stats"""
        if torch.cuda.is_available():
            logs.update(
                {
                    "gpu_mem_alloc_gb": torch.cuda.memory_allocated() / 1e9,
                    "gpu_mem_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                }
            )

        logs["skipped_samples"] = self.skipped_samples
        if self.error_log:
            logs["last_error"] = self.error_log[-1][:100] 

        super().log(logs, start_time, **kwargs)


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
