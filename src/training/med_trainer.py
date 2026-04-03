import torch
from trl import SFTTrainer
from datasets import load_dataset
import os
import copy


class MedTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        # Pop custom kwargs TRƯỚC khi gọi super()
        self.train_idx_ladder = kwargs.pop("train_idx_ladder", [])
        self.eval_idx_ladder = kwargs.pop("val_idx_ladder", [])
        self.dataset_names = kwargs.pop("dataset_names", {})
        self.tracker = kwargs.pop("tracker", None)
        self.benchmark_evaluator = kwargs.pop("benchmark_evaluator", None)
        self.run_id = kwargs.pop("run_id", None)
        self.ewc_config = kwargs.pop("ewc_config", None)

        super().__init__(*args, **kwargs)

        self.error_log_path = os.path.join(self.args.output_dir, "training_errors.jsonl")
        self.skipped_samples = 0
        self.error_log = []
        self.samples_seen = 0

        # EWC: store pretrained params and Fisher information
        self.ewc_lambda = 0.0
        self.fisher_dict = {}
        self.pretrained_params = {}

        if self.ewc_config and self.ewc_config.get("enabled", False):
            self.ewc_lambda = self.ewc_config.get("lambda", 500)
            self._compute_ewc_terms()

    def _compute_ewc_terms(self):
        """Store pretrained params and estimate Fisher information on general data.

        Fisher must be computed on data representative of the pretrained model's
        general knowledge (not the fine-tuning data), otherwise EWC protects
        parameters important for the SFT task instead of preserving general
        capabilities.
        """
        print("[EWC] Storing pretrained parameters...")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.pretrained_params[name] = param.data.clone()

        print("[EWC] Loading data for Fisher estimation...")
        fisher_dataset_path = self.ewc_config.get("fisher_dataset", None)
        num_samples = self.ewc_config.get("num_samples", 200)
        max_length = self.ewc_config.get("fisher_max_length", 512)

        if fisher_dataset_path:
            raw_dataset = load_dataset(
                fisher_dataset_path,
                split="train",
                data_files=self.ewc_config.get("fisher_data_files", None),
            )
        else:
            raise ValueError(
                "[EWC] fisher_dataset is required. Point it to "
                "the CPT data (e.g. 'CPT/') so Fisher protects "
                "the domain knowledge learned during CPT."
            )

        # Tokenize and filter out empty samples
        text_column = "text" if "text" in raw_dataset.column_names else raw_dataset.column_names[0]
        samples = []
        for example in raw_dataset:
            text = example[text_column].strip()
            if len(text) < 20:
                continue
            _tokenizer = getattr(self, "processing_class", None) or self.tokenizer
            tokens = _tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            )
            if tokens["input_ids"].shape[1] > 1:
                samples.append(tokens["input_ids"])
            if len(samples) >= num_samples:
                break

        print(f"[EWC] Estimating Fisher information matrix on {len(samples)} general samples...")

        fisher_accum = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        self.model.train()

        for input_ids in samples:
            input_ids = input_ids.to(self.model.device)

            self.model.zero_grad()
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_accum[name] += param.grad.data.pow(2)

        # Average Fisher
        actual_samples = len(samples)
        for name in fisher_accum:
            self.fisher_dict[name] = fisher_accum[name] / actual_samples

        self.model.zero_grad()
        print(f"[EWC] Fisher computed over {actual_samples} general samples, lambda={self.ewc_lambda}")

    def _ewc_penalty(self):
        """Compute EWC penalty: sum of F_i * (theta_i - theta_pretrained_i)^2"""
        penalty = 0.0
        for name, param in self.model.named_parameters():
            if name in self.fisher_dict:
                diff = param - self.pretrained_params[name]
                penalty += (self.fisher_dict[name] * diff.pow(2)).sum()
        return penalty

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs,
    ):
        outputs = model(**inputs)
        loss = outputs.loss

        # Ép cứng loss luôn là torch.Tensor
        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(
                loss,
                device=self.args.device,
                dtype=torch.float32,
            )

        # Add EWC penalty
        if self.ewc_lambda > 0 and self.fisher_dict:
            ewc_loss = self._ewc_penalty()
            loss = loss + (self.ewc_lambda / 2.0) * ewc_loss

        if return_outputs:
            return loss, outputs

        return loss


    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)

        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(loss, device=self.args.device)

        return loss
