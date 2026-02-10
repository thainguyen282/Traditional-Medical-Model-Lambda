import torch
from trl import SFTTrainer
import os


class MedTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        # Pop custom kwargs TRƯỚC khi gọi super()
        self.train_idx_ladder = kwargs.pop("train_idx_ladder", [])
        self.eval_idx_ladder = kwargs.pop("val_idx_ladder", [])
        self.dataset_names = kwargs.pop("dataset_names", {})
        self.tracker = kwargs.pop("tracker", None)
        self.benchmark_evaluator = kwargs.pop("benchmark_evaluator", None)
        
        # Gọi super().__init__
        super().__init__(*args, **kwargs)
        
        # Setup error logging
        self.error_log_path = os.path.join(self.args.output_dir, "training_errors.jsonl")
        self.skipped_samples = 0
        self.error_log = []
        self.samples_seen = 0

    def _get_dataset_from_index(self, batch_idx: int, ladder: list):
        """Get dataset tag from batch index using ladder"""
        for (start, end), tag in ladder:
            if start <= batch_idx < end:
                return tag
        return "unknown"

    def _log_error(self, error_type, error_msg, example=None, dataset_tag=None):
        """Log training errors to file and tracker"""
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

        # Đảm bảo thư mục tồn tại
        os.makedirs(os.path.dirname(self.error_log_path), exist_ok=True)
        
        with open(self.error_log_path, "a") as f:
            f.write(json.dumps(error_entry) + "\n")

        # Log to tracker if available
        if self.tracker and (self.state.global_step % 10 == 0):
            self.tracker.log(
                {
                    "error/type": error_type,
                    "error/message": str(error_msg)[:200],
                    "error/dataset": dataset_tag,
                    "error/count": self.skipped_samples,
                }
            )

    def log(self, logs, start_time=None, **kwargs):
        """Enhanced logging with GPU memory and error stats"""
        # Add GPU memory stats
        if torch.cuda.is_available():
            logs.update(
                {
                    "gpu_mem_alloc_gb": torch.cuda.memory_allocated() / 1e9,
                    "gpu_mem_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                }
            )

        # Add error tracking stats
        logs["skipped_samples"] = self.skipped_samples
        if self.error_log:
            logs["last_error"] = str(self.error_log[-1])[:100]

        super().log(logs, start_time, **kwargs)


"""
Batch Index Calculation Explanation:

Given:
- global_step = 10
- gradient_accumulation_steps = 2
- per_device_train_batch_size = 8
- world_size = 2 (2 GPUs)
- len(dataset) = 1000

Calculations:
absolute_batch_idx = global_step * gradient_accumulation_steps
                   = 10 * 2 = 20

total_batch_size = per_device_train_batch_size * world_size
                 = 8 * 2 = 16

start_idx = absolute_batch_idx * total_batch_size
          = 20 * 16 = 320

end_idx = start_idx + total_batch_size
        = 320 + 16 = 336

So at global_step=10, we're processing samples 320-336 from the dataset.
"""