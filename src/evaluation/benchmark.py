import yaml
import sys 
import os
import shutil
sys.path.append("/root/VeneraMedicalM1")
from src.evaluation.unsloth_evaluator import evaluate_model

def benchmark(config: dict, model_name: str, tags: list, output_dir: str,max_length=2048,batch_size=8, save_storage: str=None):
    
    for dataset in config:
        dataset_name = dataset["name"]
        evaluator=None
        if "unsloth" in tags:
            evaluator=evaluate_model
        else:
            raise "not implemented"
        tasks=[]
        if dataset.get("subtasks"):
            tasks=[item["name"] for item in dataset["subtasks"]]
        output_dir=os.path.join(output_dir,dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        evaluator(
            model_path=model_name,
            tasks=tasks,
            benchmark_config=dataset.get("config",{}),
            output_dir=output_dir,
            max_length=max_length,
            batch_size=batch_size
        )
if __name__=="__main__":
    import os
    import shutil
    path = "configs/evals/green.yaml"
    with open(path, "r") as f:
        config = yaml.safe_load(f)["datasets"]
    model_name="unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit"
    tags=["unsloth"]
    output_dir="group_outputs/benchmark/unsloth/test_m1"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)    
    os.makedirs(output_dir, exist_ok=True)
    benchmark(config, model_name, tags, output_dir)
    
        
