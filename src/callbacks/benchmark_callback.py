from transformers import TrainerCallback
import wandb

class FinalEvalCallback(TrainerCallback):
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def on_train_end(self, args, state, control, **kwargs):
        print("\nRunning final benchmark evaluation...")
        results = self.evaluator.run_evaluation()  # Your evaluation method
        self.evaluator.run_final_evaluation()
       
        print("Final Evaluation complete")
