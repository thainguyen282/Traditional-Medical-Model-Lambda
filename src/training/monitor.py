import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
import yaml
from src.evaluation.benchmark import benchmark
@dataclass
class EvalMilestone:
    samples_interval: int     
    eval_config:str

class TrainingEvaluator:
    def __init__(
        self,
        model_path: str=None,
        output_dir: str="outputs",
        model=None,
        tokenizer=None,
        max_length=2048,
        batch_size=8,
        tags: List[str] = ["unsloth"],
        milestones: Optional[List[EvalMilestone]] = None,
        tracker = None ,
        
        
    ):
        self.model_path = model_path
        self.model = model
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.state_path = self.output_dir / "eval_state.json"
        self.tracker = tracker
        
        # Update the eval mileston when adding new tasks
        self.milestones = milestones or [
             EvalMilestone(5, "configs/evals/green.yaml"),           
            EvalMilestone(500, "configs/evals/white.yaml"),           
            EvalMilestone(2000, "configs/evals/white.yaml"),
             EvalMilestone(5000, "configs/evals/white.yaml"),   
            EvalMilestone(10000, "configs/evals/white.yaml"),           
            EvalMilestone(float('inf'),"configs/evals/white.yaml")   
        ]
        
        self.state = {
            "samples_seen": 0,
            "last_eval_at": 0,
            "eval_history": []
        }
        self._load_state()

    def _load_state(self):
        if self.state_path.exists():
            with open(self.state_path) as f:
                self.state = json.load(f)

    def save_state(self):
        self.output_dir.mkdir(exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=2)

    def update_samples(self, new_samples: int):
        """Call after each batch (accounts for multi-GPU)"""
        self.state["samples_seen"] += new_samples
        self.save_state()

    def should_evaluate(self) -> bool:
        current = self.state["samples_seen"]
        last = self.state["last_eval_at"]
        
        for milestone in self.milestones:
            if last < milestone.samples_interval and current >= milestone.samples_interval:
                return True
        return False

    def get_eval_params(self) -> Dict:
        """Get current evaluation parameters"""
        current = self.state["samples_seen"]
        for milestone in self.milestones:
            if current <= milestone.samples_interval:
                eval_config = milestone.eval_config
                with open(eval_config) as f:
                    config=yaml.safe_load(f)["datasets"]
                return config
        return {}

    def run_evaluation(self) -> Dict:
        """Execute MMLU benchmark"""
        print("--------Eval at samples:", self.state["samples_seen"])
        eval_config = self.get_eval_params()
        output_dir = self.output_dir / f"eval_{self.state['samples_seen']}"
        output_dir.mkdir(exist_ok=True)
        benchmark(config=eval_config,model_path= self.model_path,model= self.model,tokenizer= self.tokenizer,tags= self.tags,output_dir= output_dir,max_length= self.max_length, batch_size=self.batch_size)
        self.state["last_eval_at"] = self.state["samples_seen"]
        self.state["eval_history"].append({
            "samples": self.state["samples_seen"],
        })
        self.save_state()
        
       

    def _log_results(self, results: Dict, params: Dict):
        metrics = {
            "eval/samples_seen": self.state["samples_seen"],
            "eval/samples_used": params["samples"],
            "eval/accuracy": results["results"].get("acc,none", 0),
        }
        
        if params["subjects"] != "all":
            for subject in params["subjects"]:
                task_name = f"mmlu_{subject}"
                if task_name in results["results"]:
                    metrics[f"eval/{subject}_acc"] = results["results"][task_name]["acc,none"]
        
        self.tracker.log(metrics)

