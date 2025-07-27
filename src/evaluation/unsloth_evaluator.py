
import os
import logging
from unsloth import FastLanguageModel
import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
import json 
import numpy as np
import torch

class UnslothLM(HFLM):
    """Wrapper for Unsloth models to work with lm_eval"""
    def __init__(self, model, tokenizer, max_length=2048, batch_size=8):
        super().__init__(
            pretrained=model,
            tokenizer=tokenizer,
            max_length=max_length,
            batch_size=batch_size,
        )
        self._model = model
        self._tokenizer = tokenizer

def setup_logging(log_level="INFO", log_dir="logs"):
    """Proper logging setup that ensures file writing works"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "evaluation.log")
    
    with open(log_file, 'w'):
        pass
    
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info("Logging initialized successfully")
    return logger

def load_unsloth_model(model_path,max_length=2048,dtype="float16",load_in_4bit=True):
    """Load Unsloth model with proper configuration"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    return model, tokenizer
def convert_to_serializable(obj):
    """Recursively convert non-serializable objects to serializable formats"""
    if isinstance(obj, (np.generic, np.ndarray)):
        return obj.item() if obj.ndim == 0 else obj.tolist()
    elif isinstance(obj, torch.dtype):
        return str(obj)
    elif isinstance(obj, torch.device):
        return str(obj)
    elif isinstance(obj, (set, tuple)):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list,)):
        return [convert_to_serializable(x) for x in obj]
    elif hasattr(obj, '__dict__'):
        return convert_to_serializable(obj.__dict__)
    return obj

def save_results(results, filename="eval_results.json"):
    try:
        if 'config' in results:
            results['config'] = convert_to_serializable(results['config'])
        
        serializable_results = convert_to_serializable(results)
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        print(f"✅ Successfully saved results to {filename}")
    except Exception as e:
        print(f"❌ Failed to save results: {str(e)}")


def evaluate_model(model_path, tasks=["mmlu"],benchmark_config:dict=None,max_length=2048,batch_size=8, output_dir="."):
    logger = setup_logging("INFO",log_dir=output_dir)
    
    try:
        logger.info(f"______Loading model from {model_path}")
        model, tokenizer = load_unsloth_model(model_path)
        
        logger.info("Creating evaluator...")
        lm = UnslothLM(
            model=model,
            tokenizer=tokenizer,
            max_length=max_length,
            batch_size=batch_size
        )
      
        logger.info(f"Starting evaluation on tasks: {tasks}")
        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=benchmark_config.get("num_fewshot", 5),
            task_manager=TaskManager(include_path=None),
            limit=benchmark_config.get("limit", None),
             random_seed=42, 
        )
        output_path=os.path.join(output_dir,"unsloth_eval_results.json")
        save_results(results, filename=output_path)
        print(lm_eval.utils.make_table(results))
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    model_path="unsloth/Qwen3-4B-unsloth-bnb-4bit"
    
    evaluate_model(
        model_path=model_path,
        tasks=["mmlu_humanities"],  
        num_fewshot=1
    )