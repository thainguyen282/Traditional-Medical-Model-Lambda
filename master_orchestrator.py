import wandb
import yaml
from pathlib import Path

class TrainingOrchestrator:
    def __init__(self, config_path):
        self.config = yaml.safe_load(Path(config_path).read_text())
        self.wandb_project = self.config['wandb_project']
        
    def run_complete_pipeline(self):
        """Execute full training pipeline"""
        # Stage 1: Data validation
        self.validate_data()
        
        # Stage 2: Coarse hyperparameter search
        coarse_results = self.run_coarse_search()
        
        # Stage 3: Fine-tuning search
        fine_results = self.run_fine_search(coarse_results)
        
        # Stage 4: Final training
        final_model = self.run_final_training(fine_results)
        
        # Stage 5: Model validation
        self.validate_model(final_model)
        
        return final_model
    
    def run_coarse_search(self):
        """Run coarse grid search"""
        sweep_config = {
            'method': 'grid',
            'metric': {'name': 'validation_loss', 'goal': 'minimize'},
            'parameters': {
                'learning_rate': {'values': [1e-5, 5e-5, 1e-4]},
                'batch_size': {'values': [16, 32, 64]},
                'context_length': {'values': [512, 1024, 2048]}
            },
            'early_terminate': {
                'type': 'hyperband',
                'min_iter': 10,
                'max_iter': 50
            }
        }
        
        sweep_id = wandb.sweep(sweep_config, project=self.wandb_project)
        wandb.agent(sweep_id, function=self.train_with_config, count=27)
        
        return self.get_best_runs(sweep_id, top_k=5)