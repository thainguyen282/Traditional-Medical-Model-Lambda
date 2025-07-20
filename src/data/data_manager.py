import sys 
sys.path.append("/root/VeneraMedicalM1")
from src.data.dataset_factory import DatasetFactory
from datasets import concatenate_datasets
from standardizer import get_standardizer, ToolCallStandardizer, BasicStandardizer
from schema import TOOL_CALL_SCHEMA, BASIC_SCHEMA
class DatasetManager:
    def __init__(self, ):
        self.library={}
        self.standardizers = {
            "tool": ToolCallStandardizer,
            "basic": BasicStandardizer
        }
        self.default_standardizer = BasicStandardizer
        self.default_schema = BASIC_SCHEMA
    
    def process_dataset(self, dataset):
       
        return dataset.map(
            self.standardize,
            features=self.schema,
            batched=True,
            batch_size=1000,
            num_proc=4
        )
    def store_datasets(self,config):
       for dataset_config in config:
            tag=dataset_config["tag"]
            if tag in self.library:
                raise ValueError(f"Dataset with tag {tag} already exists in the library")
            train_set,val_set=DatasetFactory.create_datasets(tag,dataset_config)
            train_set=self.process_dataset(train_set)
            val_set=self.process_dataset(val_set)
            self.library[tag]={"train":train_set,"val":val_set,"tag":tag}
    
    def validate_datasets(self,n_samples,formatting_func=None):
        
        for tag in self.library:
            
                train_dataset=self.library[tag]["train"]
                val_dataset=self.library[tag]["val"]
               
                for i in range( min(len(train_dataset),n_samples)):
                    try:
                        formatting_func(train_dataset[i])
                    except Exception as e:
                        print(f"Error formatting train dataset {tag} at index {i} sample {train_dataset[i]} : {e}")
                for i in range( min(len(val_dataset),n_samples)):
                    try:
                        formatting_func(val_dataset[i])
                    except Exception as e:
                        print(f"Error formatting val dataset {tag} at index {i} sample {val_dataset[i]} : {e}")
                        return False
        return True
         
    def load_combined_datasets(self):
        combine_train_datasets = []
        combine_val_datasets = []
        train_idx_ladder = []
        val_idx_ladder = []
        cur_train_idx = 0
        cur_val_idx = 0
        for tag in self.library:
            train_dataset=self.library[tag]["train"]
            val_dataset=self.library[tag]["val"]
            if train_dataset is not None:
                print(f"Train size ({tag}): {len(train_dataset)}")
                combine_train_datasets.append(train_dataset)
                cur_train_idx += len(train_dataset)
                train_idx_ladder.append((cur_train_idx, tag))
        
            if val_dataset is not None:
                print(f"Validation size ({tag}): {len(val_dataset)}")
                combine_val_datasets.append(val_dataset)
                cur_val_idx += len(val_dataset)
                val_idx_ladder.append((cur_val_idx, tag))
        if len(combine_train_datasets) > 0:
            combine_train_datasets = concatenate_datasets(combine_train_datasets)
        if len(combine_val_datasets) > 0:
            combine_val_datasets = concatenate_datasets(combine_val_datasets)
        return combine_train_datasets,combine_val_datasets,train_idx_ladder,val_idx_ladder
    
if __name__=="__main__":
    from unsloth import FastLanguageModel
    from format import formatting_func
    from functools import partial
    config = [
    # {
    #     "tag": "fine_tome",
    #     "train_samples": 100,
    #     "train_val_split": 0.7
    # },
    {
        "tag": "med_sample",
        "train_path": "/root/VeneraMedicalM1/training_data_v3.jsonl",
        "val_path": "/root/VeneraMedicalM1/training_data_v5.jsonl",
        "train_samples": 100,  
        "val_samples": 50
    }
]
    dataset_manager=DatasetManager()
    dataset_manager.store_datasets(config)
    n_samples=4
    model_names=["unsloth/Qwen3-0.6B","unsloth/Qwen3-1.7B"]
    for model_name in model_names:
        model,tokenizer=FastLanguageModel.from_pretrained(
                    model_name=model_name,
                    max_seq_length=2048,
                    load_in_4bit=True,
                    attn_implementation="sdpa",
                )
        output=dataset_manager.validate_datasets(n_samples,partial(formatting_func,tokenizer=tokenizer))
        print(f"Validation result for {model_name}: {output}")
        
    
    