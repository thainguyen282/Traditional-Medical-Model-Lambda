import wandb
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/train_{wandb.run.id}.log"),
        logging.StreamHandler()
    ]
)

def main():
    wandb.init()
    logging.info(f"Starting training with config: {wandb.config}")
    
if __name__ == "__main__":
    main()