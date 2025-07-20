from typing import List
from format import formatting_func
from tqdm import tqdm
import sys

sys.path.append("/root/VeneraMedicalM1")

# This function validates the datasets by formatting samples and checking their structure.
# def validate(datasets: List = ["dummy", "finetome", "med_sample"]):
#     for dataset in datasets:
#         if dataset == "dummy":
#             print("Validateing Dummy dataset...")
#             from data.dataset.dummy import load_dummy_ds
#             config={
#                 "train_samples": 10,
#                 "weight": 0.5
#             }
#             train_dataset, val_dataset = load_dummy_ds(config)
#         elif dataset == "finetome":
#             print("Validating FineTome dataset...")
#             from data.dataset.finetome import load_fine_tome_ds

#             train_dataset, val_dataset = load_fine_tome_ds(
#                 config={
#                     "path": "mlabonne/FineTome-100k",
#                     "train_samples": -1,
#                     "weight": 0.2,
#                 }
#             )
#         elif dataset == "med_sample":
#             print("Validating Med Sample dataset...")
#             train_dataset, val_dataset = load_fine_tome_ds(
#                 config={
#                     "path": "sample_training_data_v1.jsonl",
#                     "train_samples": -1,
#                     "weight": 0.7,
#                 }
#             )
#         else:
#             print(f"Unknown dataset: {dataset}. Skipping validation.")
#         print("len(train_dataset):", len(train_dataset))
#         print("len(val_dataset):", len(val_dataset))
#         print("Checking train dataset...")
#         for i in tqdm(range(len(train_dataset))):
#             formatting_func(train_dataset[i])
#         print("Checking validation dataset...")
#         for i in tqdm(range(len(val_dataset))):
#             formatting_func(val_dataset[i])
#     print("All dataset validations completed.")
#     return True

if __name__ == "__main__":
    validate()