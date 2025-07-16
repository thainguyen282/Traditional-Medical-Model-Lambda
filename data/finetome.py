from datasets import load_dataset
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from datasets import Dataset, Features, Value, Sequence
def load_fine_tome_ds(config, dataset_id):
    print("Config:", config)
    
    dataset_dict = load_dataset(config["name"])
    train_dataset = dataset_dict[list(dataset_dict.keys())[0]]  # Get first split
    
    train_dataset = train_dataset.add_column("dataset_id", [dataset_id] * len(train_dataset))
    
    n_total_samples = min(config["train"]["n_samples"], len(train_dataset))
    train_dataset = train_dataset.shuffle(seed=42).select(range(n_total_samples))
    
    if config.get("val"):
        split_ratio = config["val"]["percent"]
        dataset = train_dataset.train_test_split(
            test_size=split_ratio, 
            seed=42
        )
        train_set, val_set = dataset["train"], dataset["test"]
    else:
        train_set, val_set = train_dataset, None
    
    return train_set, val_set

def load_dummy_ds(**kwargs):
    """Returns hardcoded train (4 samples) and val (3 samples) datasets"""
    # Hardcoded training samples
    train_data = {
        "conversations": [
            # Sample 1
            [
                {"from": "human", "value": "What is the capital of France?"},
                {"from": "assistant", "value": "The capital of France is Paris."}
            ],
            # Sample 2
            [
                {"from": "human", "value": "Explain quantum computing"},
                {"from": "assistant", "value": "Quantum computing uses qubits..."}
            ],
            # Sample 3
            [
                {"from": "human", "value": "How to make pasta?"},
                {"from": "assistant", "value": "Boil water, add pasta..."}
            ],
            # Sample 4
            [
                {"from": "human", "value": "Python vs JavaScript"},
                {"from": "assistant", "value": "Python is better for data..."}
            ],
            [
                {"from": "human", "value": "Python vs JavaScript"},
                {"from": "assistant", "value": "Python is better for data..."}
            ],
            [
                {"from": "human", "value": "Python vs JavaScript"},
                {"from": "assistant", "value": "Python is better for data..."}
            ],
            [
                {"from": "human", "value": "Python vs JavaScript"},
                {"from": "assistant", "value": "Python is better for data..."}
            ]
        ]
    }
    
    # Hardcoded validation samples
    val_data = {
        "conversations": [
            # Sample 1
            [
                {"from": "human", "value": "What is AI?"},
                {"from": "assistant", "value": "AI is artificial intelligence..."}
            ],
            # Sample 2
            [
                {"from": "human", "value": "Best programming language?"},
                {"from": "assistant", "value": "Depends on your use case..."}
            ],
            # Sample 3
            [
                {"from": "human", "value": "How does photosynthesis work?"},
                {"from": "assistant", "value": "Plants convert sunlight..."}
            ],
            [
                {"from": "human", "value": "Python vs JavaScript"},
                {"from": "assistant", "value": "Python is better for data..."}
            ],
            [
                {"from": "human", "value": "Python vs JavaScript"},
                {"from": "assistant", "value": "Python is better for data..."}
            ]
        ]
    }
    
    # Convert to datasets
    train_dataset = Dataset.from_dict(train_data)
    val_dataset = Dataset.from_dict(val_data)
    
    return train_dataset, val_dataset