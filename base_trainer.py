from transformers import Trainer
from tqdm.auto import tqdm
import torch
from trl import SFTTrainer, SFTConfig
class MultiDatasetTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_idx_ladder = kwargs.pop("train_idx_ladder", [])
        self.val_idx_ladder = kwargs.pop("val_idx_ladder", [])
        self.dataset_names = kwargs.pop("dataset_names", {})
        print(f"Train index ladder: {self.train_idx_ladder}")
        print(f"Val index ladder: {self.val_idx_ladder}")
        
        self.current_train_dataset = None
        self.current_val_dataset = None
        self.current_global_step=-1
    def _get_dataset_from_index(self, index, ladder):
        """Helper to find which dataset an index belongs to"""
        for end_idx, tag in ladder:
            if index <= end_idx:
                return tag
        return None

    def training_step(self, model, inputs, return_outputs=False):

        print("global step ",self.state.global_step)
        start_idx,end_idx = get_batch_indices(
            global_step=self.state.global_step,
            batch_size=self.args.per_device_train_batch_size,
            grad_accum_steps=self.args.gradient_accumulation_steps,
            world_size=self.args.world_size,
            dataset_length=len(self.train_dataset)
        )
        # print("start idx",start_idx, "end idx",end_idx)
        dataset_tag = self._get_dataset_from_index(end_idx, self.train_idx_ladder)
        # print("dataset_tag",dataset_tag)
        loss = super().training_step(model, inputs, return_outputs)
        
        if self.state.global_step !=self.current_global_step:
            self.current_global_step=self.state.global_step
            self.log({
                f'{dataset_tag}_loss': loss.item(),
            })
        
        return loss
    
    def eval_step(self, model, inputs, return_outputs=False):
        print("val global step",self.state.global_step)
        start_idx, end_idx = get_batch_indices(
            global_step=self.state.global_step,  # or use a validation step counter
            batch_size=self.args.per_device_eval_batch_size,
            grad_accum_steps=1,  # No gradient accumulation in eval
            world_size=self.args.world_size,
            dataset_length=len(self.eval_dataset)  # Ensure you have `eval_dataset` defined
        )

        dataset_tag = self._get_dataset_from_index(end_idx, self.eval_idx_ladder)  # Ensure `eval_idx_ladder` exists

        print("val tag",dataset_tag)
        loss = super().eval_step(model, inputs, return_outputs)

        if self.state.global_step != self.current_global_step:  # Or track eval steps separately
            self.current_global_step = self.state.global_step
            self.log({
                f'{dataset_tag}_eval_loss': loss.item(),
            }, on_step=False, on_epoch=True)  # Typically, validation logs are epoch-based

    return loss

    def evaluate(self, eval_datasets=None, **kwargs):
        metrics = {}
        
        # Single dataset evaluation
        if eval_datasets is None:
            return super().evaluate(**kwargs)
        
        # Multi-dataset evaluation
        for name, dataset in eval_datasets.items():
            print(f"\nEvaluating on {name}...")
            dataset_metrics = super().evaluate(eval_dataset=dataset, **kwargs)
            
            # Prefix metrics with dataset name
            metrics.update({
                f"{name}/{k}": v 
                for k, v in dataset_metrics.items()
                if not k.startswith("eval_")  # Remove duplicate prefixes
            })
        
        return metrics

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        # Add dataset tracking if available
        if hasattr(self, "current_eval_dataset"):
            inputs["dataset_tag"] = self.current_eval_dataset
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    def _prepare_inputs(self, inputs):
        prepared = super()._prepare_inputs(inputs)
        if "dataset_tag" in inputs:
            prepared["dataset_tag"] = inputs["dataset_tag"]
        return prepared
    
def get_batch_indices(global_step, batch_size, grad_accum_steps, world_size, dataset_length):
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
