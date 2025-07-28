import yaml
import sys 
import os
import shutil
from dotenv import load_dotenv
load_dotenv()
# ROOT=
from src.evaluation.unsloth_evaluator import evaluate_model

def benchmark(config: dict,model_path=None ,model=None,tokenizer=None, tags: list=["unsloth"], output_dir: str="./outputs",max_length=2048,batch_size=8, ):
    
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
            model_path=model_path,
            model=model,
            tokenizer=tokenizer,
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
    # model_path="unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit"
    model_path="group_outputs/outputs/04813fc9/checkpoint-2"
    tags=["unsloth"]
    output_dir="group_outputs/benchmark/unsloth/test_m1"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)    
    os.makedirs(output_dir, exist_ok=True)
    benchmark(config, model_path, tags, output_dir)
    
        
