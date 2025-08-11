from dotenv import load_dotenv
load_dotenv()
import sys 
import os
sys.path.append(os.getenv("ROOT"))

import yaml
import shutil
from src.evaluation.evaluators.unsloth_adapter import UnslothAdapter
from src.core.loggers.term_logger import  TerminalLogger
from src.core.loggers.interfaces.Ilogger import ILogger
from src.evaluation.evaluators.intefaces.Ilm_evaluator import IlmEvaluator
class Benchmarker:
    def __init__(self, baseline_dir="group_outputs/baseline/",tracker:None=None,logger:ILogger=None):
        self.baseline_dir=baseline_dir
        self.logger=logger or TerminalLogger()
            
    def benchmark(self,config: dict,model_path=None ,model=None,tokenizer=None, tags: list=["unsloth"], output_dir: str="./outputs",max_length=2048,batch_size=8, ):
        
        for dataset in config:
            dataset_name = dataset["name"]
            evaluator:IlmEvaluator=None
            if "unsloth" in tags:
                evaluator=UnslothAdapter(self.logger)
            else:
                raise f"Not implemented tags {tags} for benchmark"
            tasks=[]
            if dataset.get("subtasks"):
                tasks=[item["name"] for item in dataset["subtasks"]]
            
            os.makedirs(output_dir, exist_ok=True)
            output_path=os.path.join(output_dir,f"{dataset_name}.json")
            
            config=dataset.get("config",{})
            evaluator.load(model_path=model_path,model=model,tokenizer=tokenizer,max_length=max_length,batch_size=batch_size)
            self.logger.info(f"Running benchmark on dataset {dataset_name} OUTPUT PATH {output_path }")
            
            num_fewshot=config.get("num_fewshot",5)
            limit=config.get("limit",None)
            evaluator.run_evaluate(
                tasks=tasks,
                num_fewshot=num_fewshot,
                limit=limit,
                random_seed=42,
                save_ouput_path=output_path,
                print_output=True
            )
           
if __name__=="__main__":
    import os
    import shutil
    path = "configs/evals/minimal_mmlu.yaml"
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    model_path="unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit"
    # model_path="group_outputs/outputs/04813fc9/checkpoint-2"
    tags=["unsloth"]
    output_dir="group_outputs/benchmark/unsloth/test_m1"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)    
    benchmarker=Benchmarker()
    benchmarker.benchmark(config=config,model_path= model_path, tags=tags, tokenizer=None, output_dir=output_dir)
    
        
