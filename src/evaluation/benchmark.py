from dotenv import load_dotenv
load_dotenv()
import sys 
import os
sys.path.append(os.getenv("ROOT"))

import yaml
import shutil
from uuid import uuid4
from datetime import datetime

from src.evaluation.evaluators.unsloth_adapter import UnslothAdapter
from src.core.loggers.term_logger import  TerminalLogger
from src.core.loggers.interfaces.Ilogger import ILogger
from src.evaluation.evaluators.intefaces.Ilm_evaluator import IlmEvaluator
from src.core.trackers.interfaces.Itracker import ITracker
from src.core.trackers.dummy_tracker import DummyTracker
from src.utils.timer import Timer

class Benchmarker:
    def __init__(self, baseline_dir="group_outputs/baseline/",history_path="group_outputs/benchmark_history.log",tracker:ITracker=None,logger:ILogger=None):
        if not os.path.exists(history_path):
            with open(history_path, 'w', encoding='utf-8') as f:
                f.write("")
        self.history_path=history_path
        self.baseline_dir=baseline_dir
        self.logger=logger or TerminalLogger()
        self.tracker=tracker or DummyTracker()
    
    def find_exist_baseline(self, compare_eval_config: dict, compare_train_config: dict):
        #  check if existing baseline experiement with given train_config and val_config has been performed or not
       
        baseline_subdirs = os.listdir(self.baseline_dir)
        
        for baseline_subdir in baseline_subdirs:
            baseline_dir_path = os.path.join(self.baseline_dir, baseline_subdir)
            eval_config_path = os.path.join(baseline_dir_path, "eval_config.yaml")
            train_config_path = os.path.join(baseline_dir_path, "train_config.yaml")
            
            if os.path.exists(eval_config_path) and os.path.exists(train_config_path):
                if eval_config_path.endswith(".yaml"):
                    with open(eval_config_path) as f:
                        eval_config = yaml.safe_load(f)
                else:
                    message = f"Unsupported eval config path {eval_config_path}"
                    self.logger.warning(message)
                    continue
                
                with open(train_config_path) as f:
                    train_config = yaml.safe_load(f)
                if (
                    compare_train_config == train_config
                    and compare_eval_config == eval_config
                ):
                    self.logger.info(f"Baseline experiment with train_config path {train_config_path} and eval_config {eval_config_path} already exists")
                    
                    return baseline_dir_path
        return None
    
    def benchmark(self,config, model_path=None ,model=None,tokenizer=None,  tags: list=["unsloth"], output_dir: str="./outputs",max_length=2048,batch_size=8):
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
    def run_multiple(self,configs):
        multiple_id=uuid4().hex
        self.logger.info("Running multiple benchmarks")
        for config in configs:
            # (benchmark_tags,eval_config,train_config, model_path,model,tokenizer,tags,output_dir,max_length,batch_size)=config
            benchmark_tags=config.get("benchmark_tags")
            eval_config=config.get("eval_config")
            train_config=config.get("train_config")
            model_path=config.get("model_path")
            model=config.get("model")
            tokenizer=config.get("tokenizer")
            tags=config.get("tags")
            output_dir=config.get("output_dir")
            max_length=config.get("max_length",2048)
            batch_size=config.get("batch_size",8)
            trained=config.get("trained",True)
            rerun=config.get("rerun",True)
            self.logger.info(f"Running benchmark with tags {benchmark_tags}")
            Timer.reset()
            start_time=Timer.start()
            run_output_path=self.run(eval_config=eval_config,train_config=train_config, model_path=model_path,model=model,tokenizer=tokenizer,  tags=tags, output_dir=output_dir,max_length=max_length,batch_size=batch_size,trained=trained, rerun=rerun)
            end_time=Timer.end()
            duration=Timer.duration()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = f"Ran benchmark group id {multiple_id} with tags {benchmark_tags}, on {run_output_path}. Start Time: {start_time}, End Time: {end_time}, Duration: {duration} "
            log_entry = f"{timestamp} -  {message}\n"
            self.logger.info(message)
            with open(self.history_path, "a", encoding="utf-8") as f:
                f.write(log_entry)
    def run(self,eval_config: str|dict,train_config: str|dict,  model_path=None ,model=None,tokenizer=None,  tags: list=["unsloth"], output_dir: str="./outputs",max_length=2048,batch_size=8,trained=True, rerun=False ):
    
        if isinstance(eval_config,str):
            if eval_config.endswith(".yaml"):
                with open(eval_config, "r") as f:
                    eval_config = yaml.safe_load(f)
            else:
                message = f"Unsupported eval config path {eval_config}"
                self.logger.error(message)
                raise ValueError(message)
        if isinstance(train_config,str):
            if train_config.endswith(".yaml"):
                with open(train_config, "r") as f:
                    train_config = yaml.safe_load(f)
            else:
                message = f"Unsupported train config path {train_config}"
                self.logger.error(message)
                raise ValueError(message)
      
      
        if not trained :
            if not rerun:
                exist_baseline_path = self.find_exist_baseline(eval_config, train_config)
                if exist_baseline_path:
                    return exist_baseline_path

            output_dir = os.path.join(self.baseline_dir, str(uuid4()))
            self.logger.info(f"Run baseline benchmark on output dir {output_dir}")
            eval_config_path = os.path.join(output_dir, "eval_config.yaml")
            train_config_path = os.path.join(output_dir, "train_config.yaml")
        else:
            self.logger.info(f"Run benchmark ontrained model  on output dir {output_dir}")
            eval_config_path = os.path.join(output_dir, "eval_config.yaml")
            train_config_path = os.path.join(output_dir, "train_config.yaml")
        
        os.makedirs(output_dir, exist_ok=True)
        with open(eval_config_path, "w") as f:
            yaml.dump(eval_config, f)
        with open(train_config_path, "w") as f:
            yaml.dump(train_config, f)
        
        self.benchmark(eval_config, model_path=model_path, model=model, tokenizer=tokenizer, tags=tags, output_dir=output_dir,max_length=max_length,batch_size=batch_size)
        return output_dir

       
           
if __name__=="__main__":
    from unsloth import FastLanguageModel
    import os
    import shutil
    eval_config_path = "configs/evals/minimal_mmlu.yaml"
    train_config_path = "configs/experiments/dry_run.yaml"
    model_path="unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit"
    # model_path="group_outputs/outputs/04813fc9/checkpoint-2"
    load_in_4bit=True
    max_length=2048
    batch_size=8
    dtype="float16"
    model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=max_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
            )
    tags=["unsloth"]
    baseline_output_dir="group_outputs/benchmark/unsloth/test_m1"
    trained_output_dir="group_outputs/benchmark/unsloth/test_m2"
    if os.path.exists(baseline_output_dir):
        shutil.rmtree(baseline_output_dir)    
    if os.path.exists(trained_output_dir):
        shutil.rmtree(trained_output_dir)
    benchmarker=Benchmarker()
    # benchmarker.run(eval_config=eval_config_path,train_config=train_config_path,model_path= model_path, tags=tags, tokenizer=None, output_dir=output_dir)
    # benchmarker.run(eval_config=eval_config_path,train_config=train_config_path,model_path= model_path, tags=tags, tokenizer=None, output_dir=output_dir,trained=False)
    # benchmarker.run(eval_config=eval_config_path,train_config=train_config_path,model_path= model_path, tags=tags, tokenizer=None, output_dir=output_dir,trained=False,rerun=True)
    
    multi_exps=[
        {
            "benchmark_tags":"Base line unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit",
            "eval_config":eval_config_path,
            "train_config":train_config_path,
            "model_path":model_path,
            
            "tags":tags,
            "output_dir":baseline_output_dir,
            "trained":False
        },
        {
            "benchmark_tags":"trained unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit",
            "eval_config":eval_config_path,
            "train_config":train_config_path,
            # "model_path":"",
            "model":model,
            "tokenizer":tokenizer,
            "tags":tags,
            "output_dir":trained_output_dir,
            "trained":True
        },
       
        
    ]
    benchmarker.run_multiple(multi_exps)
        
