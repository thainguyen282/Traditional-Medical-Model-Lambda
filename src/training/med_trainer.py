import torch
from unsloth import UnslothTrainer
import os


class MedTrainer(UnslothTrainer):
    def __init__(self, *args, **kwargs):
        # Pop custom kwargs TRƯỚC khi gọi super()
        self.train_idx_ladder = kwargs.pop("train_idx_ladder", [])
        self.eval_idx_ladder = kwargs.pop("val_idx_ladder", [])
        self.dataset_names = kwargs.pop("dataset_names", {})
        self.tracker = kwargs.pop("tracker", None)
        self.benchmark_evaluator = kwargs.pop("benchmark_evaluator", None)
        self.run_id = kwargs.pop("run_id", None)

        super().__init__(*args, **kwargs)

        self.error_log_path = os.path.join(self.args.output_dir, "training_errors.jsonl")
        self.skipped_samples = 0
        self.error_log = []
        self.samples_seen = 0

    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)

        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(loss, device=self.args.device)

        return loss
