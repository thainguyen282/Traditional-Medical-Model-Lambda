import json
def load_med_sample(config):
    file_path="sample_training_data_v1.jsonl"
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    print("Loaded data :", data)
    return data

if __name__ == "__main__":
    config = {
        "name": "mlabonne/MedSample",
        "train": {"n_samples": 30},
        "val": {"percent": 0.2}
    }
    load_med_sample(config)
    # The function will print the loaded data