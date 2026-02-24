import sys
import os
import gc
import json
import numpy as np
import torch

from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv("ROOT"))

from unsloth import FastLanguageModel
import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

from src.core.loggers.interfaces.Ilogger import ILogger
from src.core.loggers.term_logger import TerminalLogger
from src.evaluation.evaluators.intefaces.Ilm_evaluator import IlmEvaluator


# =========================
# Custom LM Wrapper
# =========================
class UnslothLM(HFLM):
    """Wrapper for Unsloth models to work with lm_eval"""

    def __init__(self, model, tokenizer, max_length=2048, batch_size=1):
        super().__init__(
            pretrained=model,
            tokenizer=tokenizer,
            max_length=max_length,
            batch_size=1,          # 🔥 force 1
            max_batch_size=1,      # 🔥 extra safety
        )
        self._model = model
        self._tokenizer = tokenizer
        self.recent_result = None


# =========================
# Adapter
# =========================
class UnslothAdapter(IlmEvaluator):
    def __init__(self, logger: ILogger = None):
        self.logger = logger or TerminalLogger()

    def load(
        self,
        model_path: str,
        model=None,
        tokenizer=None,
        max_length=2048,
        dtype="float16",
        batch_size=1,         # 🔥 force 1
        load_in_4bit=True,
    ):
        if model_path is not None:
            self.logger.info(f"Loading model from {model_path}")

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=max_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
            )

        # 🔥 inference mode
        FastLanguageModel.for_inference(model)
        model.eval()

        self.logger.info("Creating evaluator...")

        self.unsloth_lm = UnslothLM(
            model=model,
            tokenizer=tokenizer,
            max_length=max_length,
            batch_size=1,
        )

        self.model = model
        self.tokenizer = tokenizer

        return model, tokenizer

    def print_output(self, results):
        print(lm_eval.utils.make_table(results))

    def save(self, results, filename="eval_results.json"):
        try:
            if "config" in results:
                results["config"] = convert_to_serializable(results["config"])

            serializable_results = convert_to_serializable(results)

            with open(filename, "w") as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Successfully saved results to {filename}")

        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")

    # =========================
    # RUN EVALUATION
    # =========================
    def run_evaluate(
        self,
        tasks: list,
        num_fewshot=5,
        limit=None,
        random_seed=42,
        save_ouput_path=None,
        print_output=True,
    ):
        self.logger.info(f"Running evaluation on tasks: {tasks}")

        # 🔥 HARD VRAM CLEAN
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

        with torch.no_grad():
            results = lm_eval.simple_evaluate(
                model=self.unsloth_lm,
                tasks=tasks,
                num_fewshot=num_fewshot,
                task_manager=TaskManager(include_path=None),
                limit=limit,
                random_seed=random_seed,
                batch_size=1,          # 🔥 IMPORTANT
                max_batch_size=1,      # 🔥 IMPORTANT
            )

        self.recent_result = results

        if print_output:
            self.print_output(results)

        if save_ouput_path is not None:
            self.save(results, save_ouput_path)

        return results


# =========================
# JSON SAFE CONVERTER
# =========================
def convert_to_serializable(obj):
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

    elif isinstance(obj, list):
        return [convert_to_serializable(x) for x in obj]

    elif hasattr(obj, "__dict__"):
        return convert_to_serializable(obj.__dict__)

    return obj


# =========================
# DEBUG RUN
# =========================
if __name__ == "__main__":
    model_path = "unsloth/Qwen3-4B-unsloth-bnb-4bit"

    adapter = UnslothAdapter()
    adapter.load(model_path=model_path)

    adapter.run_evaluate(
        tasks=["hellaswag"],
        num_fewshot=0,
        limit=10,
    )