import json
from datasets import Dataset

def load_med_sample(config):
    file_path = config["path"]
    with open(file_path, 'r') as f:
        dataset = [json.loads(line) for line in f]
    dataset = Dataset.from_list(dataset)
    if config["train_samples"] == -1:
        total_samples = len(dataset)
    else:
        total_samples = min(len(dataset), config["train_samples"])

    # Shuffle and select subset
    train_dataset = dataset.shuffle(seed=42).select(range(total_samples))

    # Split with correct ratio (weight=0.6 means 60% train, 40% val)
    split_ratio = 1 - config["weight"]  # Convert to test_size
    dataset = train_dataset.train_test_split(
        test_size=split_ratio,  # Now 0.4 when weight=0.6
        seed=42
    )
    train_set, val_set = dataset["train"], dataset["test"]
    return train_set, val_set

if __name__ == "__main__":
    import os 
    import sys
    sys.path.append("/root/VeneraMedicalM1")
    from unsloth import FastLanguageModel

    from tools.format import formatting_func
    model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen3-1.7B",
            max_seq_length=2048,
            load_in_4bit=True,
            attn_implementation="sdpa",
        )
    config = {
        "path": "sample_training_data_v1.jsonl",
        "train_samples": 200,
        "weight":0.7
    }
    train_set,val_set=load_med_sample(config)
    for item in val_set:
        new_item=formatting_func(item,tokenizer)
        if not len(new_item):
            raise("find non valid",item)
            # print("find non valid",item)
    for item in train_set:
        print("item",item)
        new_item=formatting_func(item,tokenizer)
        if not len(new_item):
            raise("find non valid",item)
    
    
        
        
    
    # The function will print the loaded data