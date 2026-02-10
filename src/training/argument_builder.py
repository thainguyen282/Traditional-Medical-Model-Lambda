from typing import Dict, Any
from transformers import TrainingArguments

class ArgumentBuilder:
    @staticmethod
    def build(train_config: Dict[str, Any]) -> TrainingArguments:
        output_dir = train_config.get("output_dir", "./outputs")

        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=train_config.get("per_device_train_batch_size", 2),
            gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 4),
            num_train_epochs=train_config.get("num_train_epochs", 3),
            learning_rate=float(train_config.get("learning_rate", 2e-4)),
            weight_decay=float(train_config.get("weight_decay", 0.01)),
            warmup_steps=train_config.get("warmup_steps", 100),
            lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),
            optim=train_config.get("optim", "adamw_8bit"),

            logging_steps=train_config.get("logging_steps", 10),

            save_strategy=train_config.get("save_strategy", "steps"),
            save_steps=train_config.get("save_steps", 50),

            eval_strategy=train_config.get("eval_strategy", "steps"),
            eval_steps=train_config.get("eval_steps", 50),

            seed=train_config.get("seed", 3407),
            report_to=train_config.get("report_to", "wandb"),

            remove_unused_columns=False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,

            fp16=train_config.get("fp16", False),
            bf16=train_config.get("bf16", False),
        )