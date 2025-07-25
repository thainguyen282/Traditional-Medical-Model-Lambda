import torch
import pandas as pd
from lm_eval import tasks
from lm_eval.evaluator import evaluate
from lm_eval.models import huggingface

def evaluate_against_benchmark(model, tokenizer):
    MMLU_SUBSETS = ["stem", "humanities", "social_sciences", "other"]
    NUM_FEWSHOT = 5
    MAX_SAMPLES_PER_SUBSET = 100  # Limit to 100 questions per subject
    
    # Create task list using string names instead of task objects
    mmlu_task_names = [f"mmlu_{subset}" for subset in MMLU_SUBSETS]
    
    results = evaluate(
        lm=model,
        tasks=mmlu_task_names,
        num_fewshot=NUM_FEWSHOT,
        batch_size=4,  # Reduce if OOM errors occur
        device="cuda" if torch.cuda.is_available() else "cpu",
        limit=MAX_SAMPLES_PER_SUBSET  # Use limit parameter instead of manual trimming
    )
    
    print("\n=== MMLU Results ===")
    for task_name, task_results in results["results"].items():
        accuracy = task_results.get('acc', task_results.get('acc_norm', 0))
        print(f"{task_name}: {accuracy:.4f}")
    
    # Create DataFrame with results
    df = pd.DataFrame([
        {"Task": task_name, "Accuracy": res.get('acc', res.get('acc_norm', 0))}
        for task_name, res in results["results"].items()
    ])
    
    df.to_csv("qwen_0.6b_mmlu_results.csv", index=False)
    print(f"\nResults saved to qwen_0.6b_mmlu_results.csv")
    
    # Calculate and print average accuracy
    avg_accuracy = df["Accuracy"].mean()
    print(f"Average MMLU Accuracy: {avg_accuracy:.4f}")
    
    return results

if __name__ == "__main__":
    MODEL_NAME = "Qwen/Qwen2.5-0.5B" 
    
    try:
        model = huggingface.HFLM(
            pretrained=MODEL_NAME,
            device_map="auto",
        )
        
        results = evaluate_against_benchmark(model, None)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install lm-eval transformers torch accelerate")