from transformers import Trainer
from tqdm.auto import tqdm
import torch
from trl import SFTTrainer, SFTConfig
from collections import defaultdict
import wandb  
class MultiDatasetTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_idx_ladder = kwargs.pop("train_idx_ladder", [])
        self.eval_idx_ladder = kwargs.pop("val_idx_ladder", [])
        self.dataset_names = kwargs.pop("dataset_names", {})
        print(f"Train index ladder: {self.train_idx_ladder}")
        print(f"Val index ladder: {self.eval_idx_ladder}")

        self.current_train_dataset = None
        self.current_val_dataset = None
        self.current_global_step = -1

        self.eval_step_counter = 0
        self.eval_loss_accumulator = defaultdict(list)

    def _get_dataset_from_index(self, index, ladder):
        """Helper to find which dataset an index belongs to"""
        for end_idx, tag in ladder:
            if index <= end_idx:
                return tag
        
        return ladder[-1][1] 

    def training_step(self, model, inputs, return_outputs=False):

        # print("global step ", self.state.global_step)
        start_idx, end_idx = get_batch_indices(
            global_step=self.state.global_step,
            batch_size=self.args.per_device_train_batch_size,
            grad_accum_steps=self.args.gradient_accumulation_steps,
            world_size=self.args.world_size,
            dataset_length=len(self.train_dataset),
        )
        # print("start idx",start_idx, "end idx",end_idx)
        dataset_tag = self._get_dataset_from_index(end_idx, self.train_idx_ladder)
        loss = super().training_step(model, inputs, return_outputs)

        if self.state.global_step != self.current_global_step:
            self.current_global_step = self.state.global_step
            self.log(
                {
                    f"{dataset_tag}_loss": loss.item(),
                }
            )

        return loss

    def evaluate(self, eval_dataset=None, **kwargs):
        self.eval_step_counter = 0
        self.eval_loss_accumulator.clear()
        
        output = super().evaluate(eval_dataset, **kwargs)
        
        for tag, losses in self.eval_loss_accumulator.items():
            metric_name = f"eval_{tag}_loss"  # ← Must start with 'eval_'
            output[metric_name] = sum(losses) / len(losses)
        
        if wandb.run is not None:
            wandb.log(output, commit=True)
        
        self.log(output)
        
        print("Logged metrics:", output)
        return output

    def prediction_step(
        self, model, inputs, prediction_loss_only=False, ignore_keys=None
    ):
        batch_size = self.args.per_device_eval_batch_size
        print("eval batch size", batch_size)
        start_idx = self.eval_step_counter * batch_size
        end_idx = start_idx + batch_size

        dataset_tag = self._get_dataset_from_index(end_idx, self.eval_idx_ladder)
        print(f"Processing eval batch {self.eval_step_counter} for {dataset_tag}")

        loss, logits, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys
        )

        if loss is not None:
            self.eval_loss_accumulator[dataset_tag].append(loss.item())

        self.eval_step_counter += 1
        return (loss, logits, labels)

    def _prepare_inputs(self, inputs):
        prepared = super()._prepare_inputs(inputs)
        if "dataset_tag" in inputs:
            prepared["dataset_tag"] = inputs["dataset_tag"]
        return prepared


def get_batch_indices(
    global_step, batch_size, grad_accum_steps, world_size, dataset_length
):
    absolute_batch_idx = global_step * grad_accum_steps

    total_batch_size = batch_size * world_size
    start_idx = absolute_batch_idx * total_batch_size
    end_idx = start_idx + total_batch_size

    start_idx %= dataset_length
    end_idx %= dataset_length
    return start_idx, end_idx


""""

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
