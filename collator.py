from transformers import DataCollatorForLanguageModeling
import torch
import re
class MultiDatasetCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=True, mlm_probability=0.15):
        super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)
        self.tokenizer = tokenizer
    def __call__(self, features):
        # batch = super().__call__(features)
        # return batch
        
        
        batch = super().__call__(features)
        # batch["dataset_ids"] = torch.tensor(dataset_ids)
        
        return batch